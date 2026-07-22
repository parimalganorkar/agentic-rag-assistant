"""Phase 1b — Normalize MDX/MD source files to plain Markdown.

The langchain-ai/docs corpus is Mintlify MDX. Each file mixes:
  - YAML frontmatter (title, description, sidebarTitle)
  - Snippet imports (`import X from '/snippets/...'`) rendered as `<X />`
  - Language-conditional prose blocks (`:::python ... :::` and `:::js ... :::`)
  - Mintlify JSX components (<Note>, <Tip>, <Warning>, <CodeGroup>, ...)
  - Inline reference syntax (@[`create_agent`])
  - JSX comments ({/* ... */}) and HTML comments (<!-- ... -->)

This module inlines snippet imports (so their code survives) and strips or
unwraps everything else into plain Markdown, so the downstream loader/chunker
never has to know MDX exists.

Snippet policy: many pages keep their code examples in `/snippets/...mdx` files
pulled in via `<ComponentName />`. We resolve and inline those FIRST (given a
resolver) — otherwise the import line and the `<X />` tag are both stripped and
the code is silently lost, which is exactly what a code-focused retriever needs.

Language policy: we keep :::python blocks and drop :::js blocks, because this
is a Python-native RAG project and keeping both variants would create
near-duplicate chunks that hurt retrieval quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import frontmatter


# A resolver maps a snippet import path (e.g. "/snippets/code-samples/foo.mdx")
# to that snippet's raw text, or None if it can't be resolved. Injected so the
# normalizer stays pure/testable and works with or without local disk access
# (local cleaning passes a disk resolver; the live-doc fetch passes none).
SnippetResolver = Callable[[str], "str | None"]


@dataclass
class NormalizedDoc:
    """Output of the normalizer for a single source file."""
    text: str
    title: str
    description: str | None = None
    sidebar_title: str | None = None
    warnings: list[str] = field(default_factory=list)


# --- Regex patterns ---------------------------------------------------------

# Top-of-file ES-module imports Mintlify uses to pull snippets.
# Only strip these when they appear at the top of the doc (before any prose),
# to avoid accidentally stripping example code inside fenced code blocks.
_IMPORT_LINE = re.compile(r"^\s*import\s+.+?from\s+['\"][^'\"]+['\"];?\s*$", re.MULTILINE)

# Default-import statement, capturing (ComponentName, snippet_path). Used to
# build the map that snippet inlining consults before those imports are stripped.
_IMPORT_STMT = re.compile(
    r"^\s*import\s+([A-Za-z_]\w*)\s+from\s+['\"]([^'\"]+)['\"];?\s*$", re.MULTILINE
)

# Self-closing component reference, e.g. <AgentsIntroPy /> — the placeholder a
# snippet import fills in. Capitalized initial distinguishes JSX from HTML.
_COMPONENT_REF = re.compile(r"<([A-Z][A-Za-z0-9_]*)\s*/>")

# JSX comments: {/* ... */}   (may span multiple lines)
_JSX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)

# HTML comments: <!-- ... -->  (may span multiple lines)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# :::python ... :::   — keep the inner content
_PYTHON_BLOCK = re.compile(r"^:::python\s*\n(.*?)^:::\s*$", re.DOTALL | re.MULTILINE)

# :::js ... :::   — drop entirely (including the markers)
_JS_BLOCK = re.compile(r"^:::js\s*\n.*?^:::\s*$", re.DOTALL | re.MULTILINE)

# :::typescript, :::ts etc — same treatment as :::js
_TS_BLOCK = re.compile(r"^:::(?:typescript|ts)\s*\n.*?^:::\s*$", re.DOTALL | re.MULTILINE)

# Self-closing JSX tags: <ComponentName ... />
_SELF_CLOSING_JSX = re.compile(r"<[A-Z][A-Za-z0-9_]*(?:\s+[^>]*?)?\s*/>")

# <img ...> and <img ... /> (lowercase — Mintlify uses <img> with style props)
_IMG_TAG = re.compile(r"<img\b[^>]*/?>", re.IGNORECASE | re.DOTALL)

# Opening JSX wrapper tag for whitelisted Mintlify components.
# We match <Component ...> and </Component> and remove them, keeping inner text.
# NOTE: we deliberately only whitelist components used as prose-level wrappers.
# JSX-looking tokens inside fenced code blocks (like <Markdown> or <AgentState>)
# are intentional code content and must NOT be stripped.
_MINTLIFY_COMPONENTS = (
    "Note", "Tip", "Warning", "Info", "Callout",
    "CodeGroup", "Tabs", "Tab",
    "Accordion", "AccordionGroup",
    "Card", "CardGroup", "Columns",
    "Steps", "Step",
    "Frame", "ParamField", "ResponseField",
    "Icon", "Tooltip", "Badge",
    "Check", "Expandable",
    "Update",                                # Mintlify changelog entry
    r"Tree\.Folder", r"Tree\.File", "Tree",  # Mintlify file-tree components
)
_JSX_OPEN_TAG = re.compile(
    r"<(" + "|".join(_MINTLIFY_COMPONENTS) + r")(\s[^>]*)?>",
)
_JSX_CLOSE_TAG = re.compile(
    r"</(" + "|".join(_MINTLIFY_COMPONENTS) + r")>",
)

# Inline reference syntax: @[`create_agent`]  →  `create_agent`
_INLINE_REF = re.compile(r"@\[(`[^`]+`)\]")

# Runs of 3+ blank lines → 2 blank lines (i.e. one blank line between paragraphs)
_EXCESS_BLANKS = re.compile(r"\n{3,}")


# --- Cleaning steps ---------------------------------------------------------

def _inline_snippets(text: str, resolver: SnippetResolver | None, _depth: int = 0) -> str:
    """Replace `<ComponentName />` refs with the imported snippet's content.

    Mintlify pages keep their code examples in `/snippets/...mdx` files pulled in
    via `import Name from '...'` and rendered as `<Name />` (often inside a
    :::python block). If we don't inline these first, `_strip_top_imports` drops
    the import and `_strip_media_and_self_closing` deletes the `<Name />` tag —
    silently losing the code, exactly what a code-focused retriever needs.

    Snippets can themselves import snippets, so we recurse (shallow, guarded).
    Without a resolver (e.g. the live-doc fetch has no local disk map), this is a
    no-op and the old behavior stands."""
    if resolver is None or _depth > 3:
        return text
    imports = {name: path for name, path in _IMPORT_STMT.findall(text)}
    if not imports:
        return text

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        path = imports.get(name)
        if path is None:
            return match.group(0)  # not a snippet import — leave for later steps
        content = resolver(path)
        if content is None:
            return match.group(0)  # unresolved — don't destroy the reference
        return _inline_snippets(content, resolver, _depth + 1)

    return _COMPONENT_REF.sub(_replace, text)


def _strip_top_imports(text: str) -> str:
    """Strip all ES-module import lines. Safe because MDX docs never use
    `import` in prose or in fenced code blocks (which use ```python fences)."""
    return _IMPORT_LINE.sub("", text)


def _drop_language_blocks(text: str) -> str:
    """Drop :::js and :::typescript blocks; unwrap :::python blocks."""
    text = _JS_BLOCK.sub("", text)
    text = _TS_BLOCK.sub("", text)
    text = _PYTHON_BLOCK.sub(lambda m: m.group(1), text)
    return text


def _strip_comments(text: str) -> str:
    text = _JSX_COMMENT.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    return text


def _strip_media_and_self_closing(text: str) -> str:
    text = _IMG_TAG.sub("", text)
    text = _SELF_CLOSING_JSX.sub("", text)
    return text


def _unwrap_jsx_wrappers(text: str) -> str:
    """Remove opening/closing tags for known Mintlify wrapper components,
    keeping the inner content. Run several passes because components can nest."""
    prev = None
    passes = 0
    while prev != text and passes < 5:
        prev = text
        text = _JSX_OPEN_TAG.sub("", text)
        text = _JSX_CLOSE_TAG.sub("", text)
        passes += 1
    return text


_FENCE_LINE = re.compile(r"^(\s*)(```|~~~)")


def _dedent_code_fences(text: str) -> str:
    """Strip an opening fence's own indentation from its whole block.

    Inlined snippet code arrives indented (it was nested inside a <CodeGroup>/
    <Tabs> wrapper), which leaves ```fences indented after the wrapper tags are
    removed. Dedent each block by exactly its opening-fence indent so fences sit
    at column 0 while relative indentation inside the code is preserved."""
    out: list[str] = []
    in_fence = False
    indent = ""
    for line in text.split("\n"):
        if not in_fence:
            m = _FENCE_LINE.match(line)
            if m:
                in_fence = True
                indent = m.group(1)
            out.append(line[len(indent):] if indent and line.startswith(indent) else line)
        else:
            out.append(line[len(indent):] if indent and line.startswith(indent) else line)
            if _FENCE_LINE.match(line):
                in_fence = False
                indent = ""
    return "\n".join(out)


def _rewrite_inline_refs(text: str) -> str:
    return _INLINE_REF.sub(r"\1", text)


def _normalize_whitespace(text: str) -> str:
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse excess blank lines
    text = _EXCESS_BLANKS.sub("\n\n", text)
    # Guarantee single trailing newline
    return text.strip() + "\n"


def _ensure_h1(text: str, title: str) -> str:
    """Prepend '# {title}' if the doc doesn't already start with an H1."""
    stripped = text.lstrip()
    if stripped.startswith("# "):
        return text
    return f"# {title}\n\n{text.lstrip()}"


# --- Public API -------------------------------------------------------------

def clean_mdx(raw: str, snippet_resolver: SnippetResolver | None = None) -> NormalizedDoc:
    """Convert a raw MDX string to a NormalizedDoc containing plain Markdown.

    If `snippet_resolver` is given, `<Component />` snippet references are
    inlined from their imported source first, so code kept in `/snippets/...`
    files survives cleaning. Without it, snippet refs are stripped as before."""
    post = frontmatter.loads(raw)
    fm = post.metadata or {}
    body = post.content

    title = str(fm.get("title") or "Untitled").strip()
    description = fm.get("description")
    if description is not None:
        description = str(description).strip()
    sidebar_title = fm.get("sidebarTitle")
    if sidebar_title is not None:
        sidebar_title = str(sidebar_title).strip()

    warnings: list[str] = []
    if "title" not in fm:
        warnings.append("no title in frontmatter")

    # Order matters here — inline snippets FIRST (while the import map is still
    # present), then comments, then structural, then wrappers.
    body = _inline_snippets(body, snippet_resolver)
    body = _strip_comments(body)
    body = _strip_top_imports(body)
    body = _drop_language_blocks(body)
    body = _strip_media_and_self_closing(body)
    body = _unwrap_jsx_wrappers(body)
    body = _dedent_code_fences(body)
    body = _rewrite_inline_refs(body)
    body = _normalize_whitespace(body)
    body = _ensure_h1(body, title)

    return NormalizedDoc(
        text=body,
        title=title,
        description=description,
        sidebar_title=sidebar_title,
        warnings=warnings,
    )


def clean_markdown(raw: str, snippet_resolver: SnippetResolver | None = None) -> NormalizedDoc:
    """Plain-Markdown cleaner. The one .md file in the corpus still uses
    Mintlify JSX components, so we route it through the same pipeline."""
    return clean_mdx(raw, snippet_resolver=snippet_resolver)


# Format registry — extension → cleaner. Adding a new format (e.g. .rst) means
# registering one function here; nothing else in the pipeline changes.
FORMAT_REGISTRY: dict[str, Callable[..., NormalizedDoc]] = {
    ".mdx": clean_mdx,
    ".md": clean_markdown,
}


def _make_local_snippet_resolver(src_root: Path, source_dir: Path) -> SnippetResolver:
    """Resolve snippet import paths against the local corpus.

    Absolute paths ('/snippets/...') are relative to the Mintlify content root
    (the `src/` dir); anything else is relative to the importing file's folder."""
    def resolve(import_path: str) -> str | None:
        target = src_root / import_path.lstrip("/") if import_path.startswith("/") \
            else source_dir / import_path
        try:
            return target.read_text(encoding="utf-8") if target.is_file() else None
        except OSError:
            return None
    return resolve


def normalize_file(path: Path) -> NormalizedDoc:
    """Read a file from disk and route it to the correct cleaner by extension.

    Builds a local snippet resolver so `/snippets/...` code imports are inlined
    (the Mintlify content root is the nearest ancestor dir containing snippets/)."""
    cleaner = FORMAT_REGISTRY.get(path.suffix)
    if cleaner is None:
        raise ValueError(f"No cleaner registered for extension {path.suffix!r}")
    raw = path.read_text(encoding="utf-8")
    src_root = next((p for p in path.parents if (p / "snippets").is_dir()), path.parent)
    resolver = _make_local_snippet_resolver(src_root, path.parent)
    return cleaner(raw, snippet_resolver=resolver)


if __name__ == "__main__":
    # Smoke test: normalize a known file and print the result.
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sample = repo_root / "data" / "raw" / "src" / "oss" / "langchain" / "agents.mdx"
    if not sample.exists():
        print(f"Sample not found: {sample}")
        sys.exit(1)

    doc = normalize_file(sample)
    print(f"--- title:       {doc.title}")
    print(f"--- description: {doc.description}")
    print(f"--- warnings:    {doc.warnings}")
    print(f"--- text preview (first 1500 chars):\n")
    print(doc.text[:1500])
    print("\n---")
    print(f"total chars: {len(doc.text)}")
