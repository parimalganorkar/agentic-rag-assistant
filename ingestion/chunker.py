"""Phase 2 — Split cleaned docs into embedding-sized chunks.

Two-stage strategy:

  1. MarkdownHeaderTextSplitter walks the doc by H1/H2/H3, producing one
     candidate chunk per leaf section. Header text is preserved in metadata
     so the retriever can show the reader which section a citation came from.

  2. Any candidate that's still too big for a single embedding is further
     split with RecursiveCharacterTextSplitter, measuring size with the
     embedding model's own tokenizer (bge-small-en-v1.5's WordPiece).
     Overlap is 50 tokens so context near a split boundary isn't lost.

Tokenizer note: we count with bge-small's real tokenizer, NOT tiktoken. The
two tokenizers segment text very differently (WordPiece vs BPE), and a
"400-token chunk" measured with one is not the same as measured with the
other. Since bge-small has a hard 512-token embedding window, sizing chunks
in its own tokenizer is the only way to guarantee we never silently truncate
at embed time.

Chunk IDs are deterministic: `{content_hash[:12]}_{index:03d}`. That means
if a source file hasn't changed since the last run, its chunk IDs are stable
across runs, so Phase 3's Chroma upsert can be truly incremental instead of
"delete everything, re-insert".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from ingestion.loader import LoadedDoc


# --- Config -----------------------------------------------------------------

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# Embedding model whose tokenizer defines the "token" unit throughout this
# pipeline. Must match the model used in Phase 3's embed_and_store.py — if
# they diverge, chunk sizes will lie about what actually fits in a single
# embedding.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Load once at module load. AutoTokenizer downloads from HuggingFace on first
# use and caches under ~/.cache/huggingface/ thereafter.
_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

# Header levels we split on. H1 is the doc title (added by the cleaner), so
# splitting on H2/H3 gives us section-level chunks.
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


# --- Data class -------------------------------------------------------------

@dataclass
class Chunk:
    """One embedding-sized chunk. Per project convention every chunk carries
    at minimum `source_file`, `section`, and `chunk_id`; everything else is
    extra metadata to help retrieval and downstream eval."""
    chunk_id: str
    source_file: str          # original raw path, e.g. "src/oss/langchain/agents.mdx"
    section: str              # coarse label from the loader
    text: str
    title: str
    product: str
    doc_type: str
    headers: dict[str, str] = field(default_factory=dict)  # {"h1": "...", "h2": "..."}
    token_count: int = 0
    content_hash: str = ""    # SHA256 of the raw source; used for chunk_id stability

    def to_dict(self) -> dict:
        return asdict(self)


# --- Helpers ----------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Count content tokens as the embedding model will see them (no [CLS]/[SEP])."""
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def _make_chunk_id(content_hash: str, index: int) -> str:
    return f"{content_hash[:12]}_{index:03d}"


def _build_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """Recursive splitter that measures size with bge-small's tokenizer."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_count_tokens,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )


# --- Public API -------------------------------------------------------------

def chunk_doc(doc: LoadedDoc) -> list[Chunk]:
    """Split one LoadedDoc into a list of Chunks."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep headers inline so the chunk reads as prose
    )
    header_docs = header_splitter.split_text(doc.text)

    recursive = _build_recursive_splitter()

    chunks: list[Chunk] = []
    for hd in header_docs:
        section_text = hd.page_content
        section_headers = {
            k: v for k, v in hd.metadata.items() if k in {"h1", "h2", "h3"}
        }
        section_label = " > ".join(
            section_headers.get(k, "") for k in ("h1", "h2", "h3") if section_headers.get(k)
        ) or doc.section

        # RecursiveCharacterTextSplitter only splits when a section exceeds
        # CHUNK_SIZE_TOKENS; sub-threshold text is returned unchanged as a
        # single piece. So we call it unconditionally instead of pre-checking
        # the size ourselves — that pre-check was a redundant full tokenization
        # pass on every section (the splitter re-tokenizes internally anyway).
        pieces = recursive.split_text(section_text)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            token_count = _count_tokens(piece)
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc.content_hash, index),
                    source_file=doc.source_file,
                    section=section_label,
                    text=piece,
                    title=doc.title,
                    product=doc.product,
                    doc_type=doc.doc_type,
                    headers=section_headers,
                    token_count=token_count,
                    content_hash=doc.content_hash,
                )
            )
    return chunks


def chunk_all(docs: Iterable[LoadedDoc]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_doc(doc))
    return out


if __name__ == "__main__":
    # Smoke test on one doc
    from ingestion.loader import load_cleaned_docs

    docs = list(load_cleaned_docs())
    if not docs:
        raise SystemExit("no cleaned docs found — run the cleaner first")
    sample_doc = next(d for d in docs if d.source_file.endswith("langchain/agents.mdx"))
    chunks = chunk_doc(sample_doc)
    print(f"doc: {sample_doc.source_file}")
    print(f"chars: {len(sample_doc.text)}   ->   chunks: {len(chunks)}")
    print()
    for c in chunks[:3]:
        print(f"--- {c.chunk_id}  ({c.token_count} tokens)  section={c.section!r}")
        print(c.text[:400])
        print()
