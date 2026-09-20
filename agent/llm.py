"""LLM provider factory — the ONE place that decides which chat model generates.

Two providers, selected by `LLM_PROVIDER` in .env:

    bedrock             Amazon Nova Pro on AWS Bedrock — the chosen generator.
                        Measured against local llama on the same code
                        (2026-09-20): routing 27/28 vs 24/28, all four
                        off-topic probes caught vs two, ~5 s answers vs ~20 s,
                        and no GPU needed. Auth comes from the IAM role (EC2)
                        or `aws configure` (local dev). No AWS access keys go
                        in .env, ever.
    ollama              llama3.1:8b served locally. The baseline the system was
                        built and first evaluated on; kept as a free, offline,
                        no-account mode. It is also the code's fallback when
                        LLM_PROVIDER is unset, so a bare clone still runs.

The router, generator and clarify nodes all call `chat()` and never build a
model themselves, so switching provider is one env var. The guardrail
verifiers deliberately do NOT come through `get_llm()`: a guard must run on a
DIFFERENT model family from the generator so it checks someone else's work,
not its own (see agent/guardrails.py). They use `bedrock_chat_model()` with
their own model id instead.

    python -m agent.llm            # smoke-test the provider named in .env
    python -m agent.llm bedrock    # force a provider for the check
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rag.naive import LLM_MODEL as OLLAMA_MODEL, OLLAMA_HOST

REPO_ROOT = Path(__file__).resolve().parents[1]
# Load (never read) .env so LLM_PROVIDER / AWS_REGION are visible to the factory.
load_dotenv(REPO_ROOT / ".env")

PROVIDERS = ("ollama", "bedrock")

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
# Generator/router model on Bedrock: Amazon Nova Pro. In ap-south-1 the bare
# `amazon.nova-pro-v1:0` id refuses on-demand use — only the `apac.` inference
# profile invokes. Override per region with BEDROCK_MODEL_ID.
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "apac.amazon.nova-pro-v1:0")
# Guardrail FALLBACK model — Amazon Nova Lite. The PRIMARY judge is Gemini
# (agent/guardrails.py), a different family from the generator; this one only
# runs when Gemini is down, and shares the generator's family — accepted
# knowingly (2026-09-20): the independence property lives in the primary.
# Same region caveat: the bare `amazon.nova-lite-v1:0` id refuses on-demand
# use in ap-south-1; the `apac.` inference profile invokes.
BEDROCK_GUARD_MODEL_ID = os.getenv("BEDROCK_GUARD_MODEL_ID", "apac.amazon.nova-lite-v1:0")


def provider() -> str:
    """The configured provider, validated. Read at call time so a test or eval
    run can flip LLM_PROVIDER without re-importing."""
    p = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if p not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER: {p!r} (expected one of {PROVIDERS})")
    return p


def provider_label() -> str:
    """Human-readable 'what is generating' string for the UI status panel."""
    if provider() == "bedrock":
        return f"{BEDROCK_MODEL_ID} · AWS Bedrock ({AWS_REGION})"
    return f"{OLLAMA_MODEL} · local (Ollama)"


def bedrock_chat_model(model_id: str, temperature: float = 0.0, max_tokens: int = 1024):
    """A Bedrock chat model via the Converse API.

    `ChatBedrockConverse` rather than `ChatBedrock`: Converse is the one Bedrock
    API that takes every model family uniformly, with temperature
    and max_tokens as first-class arguments. The legacy invoke path needs
    per-family `model_kwargs` and fails outright on Nova. Credentials resolve
    through boto3's normal chain — IAM role, env, `~/.aws/credentials` — never
    from our .env.
    """
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model_id=model_id,
        region_name=AWS_REGION,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=16)
def _cached_llm(provider_name: str, temperature: float, max_tokens: int, json_mode: bool):
    if provider_name == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = dict(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=temperature,
            num_predict=max_tokens,
        )
        if json_mode:
            kwargs["format"] = "json"  # Ollama-native constrained decoding
        return ChatOllama(**kwargs)
    if provider_name == "bedrock":
        # No constrained-JSON mode on Bedrock: the prompt asks for a bare JSON
        # object and the caller parses tolerantly (agent/router.py).
        return bedrock_chat_model(BEDROCK_MODEL_ID, temperature, max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider_name!r}")


def get_llm(
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
    provider_name: str | None = None,
):
    """Return the generator chat model for the configured (or given) provider.

    Instances are cached per (provider, temperature, max_tokens, json_mode):
    a Bedrock model owns a boto3 client, which is not free to rebuild per call.
    """
    return _cached_llm(provider_name or provider(), float(temperature), int(max_tokens), bool(json_mode))


def to_messages(system: str, user: str, history: list[dict] | None = None) -> list[BaseMessage]:
    """Adapt the {role, content} dicts the nodes already build into LangChain
    messages. `history` rows are prior turns (user/assistant) from
    `agent.nodes.history_messages`, already PII-scrubbed."""
    out: list[BaseMessage] = [SystemMessage(content=system)]
    for m in history or []:
        role = m.get("role")
        content = m.get("content") or ""
        if not content:
            continue
        out.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    out.append(HumanMessage(content=user))
    return out


def message_text(resp: Any) -> str:
    """Plain text from a chat response. `.content` is a str from Ollama but a
    LIST of content blocks from Bedrock Converse; handle both."""
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()
    return str(content or "").strip()


def chat(
    system: str,
    user: str,
    history: list[dict] | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """One chat completion on the configured provider. This is what the
    router, generate and clarify nodes call."""
    llm = get_llm(temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
    return message_text(llm.invoke(to_messages(system, user, history)))


# ============================================================================
# Smoke test — python -m agent.llm [ollama|bedrock]
# ============================================================================

if __name__ == "__main__":
    import json
    import sys
    import time

    if len(sys.argv) > 1:
        os.environ["LLM_PROVIDER"] = sys.argv[1]
    _cached_llm.cache_clear()

    from agent.router import _parse_router_json, _run_router_llm

    p = provider()
    print(f"provider = {p}   ({provider_label()})")
    print("=" * 68)

    # 1. Router: the JSON-mode / tool-arg extraction contract must survive the
    #    provider swap. Each case names the route AND the argument it must fill.
    router_cases = [
        ("How do I add a tool to an agent?",             "retrieve",            None),
        ("What's the latest version of langgraph?",      "get_package_version", ("package", "langgraph")),
        ("Do you have anything on middleware?",          "get_corpus_status",   ("topic", "middleware")),
        ("Show me the current page on persistence",      "fetch_live_doc",      ("live_query", None)),
        ("help",                                         "clarify",             None),
    ]
    ok = 0
    for q, want_route, want_arg in router_cases:
        t0 = time.perf_counter()
        raw = _run_router_llm(q)
        parsed = _parse_router_json(raw)
        dt = time.perf_counter() - t0
        route_ok = parsed is not None and parsed["routes"][0] == want_route
        arg_ok = True
        if want_arg and parsed is not None:
            key, expect = want_arg
            val = parsed.get(key)
            arg_ok = bool(val) and (expect is None or expect.lower() in str(val).lower())
        good = route_ok and arg_ok
        ok += int(good)
        print(f"  [{'ok ' if good else 'BAD'}] {dt:5.1f}s  {q!r}")
        print(f"        parsed={json.dumps(parsed)}")
        if not good:
            print(f"        raw={raw[:200]!r}")
    print(f"\n  router: {ok}/{len(router_cases)} parsed to the expected route + args\n")

    # 2. Generation on the same provider.
    t0 = time.perf_counter()
    text = chat("You are terse. Answer in one sentence.", "What is a LangGraph checkpointer?")
    print(f"  generate ({time.perf_counter() - t0:.1f}s): {text[:160]!r}\n")

    # 3. On Bedrock, also prove the guard FALLBACK model answers — this is the
    #    path that only fires when Gemini is down, so it never gets exercised by
    #    normal use and is exactly the kind of thing that silently rots.
    if p == "bedrock":
        from agent.guardrails import _build_verify_prompt, _parse_verdict, VERIFY_SYSTEM
        guard = bedrock_chat_model(BEDROCK_GUARD_MODEL_ID, 0.0, 400)
        ctx = "Pass a checkpointer to compile(), e.g. compile(checkpointer=MemorySaver())."
        prompt = _build_verify_prompt("How do I persist state?", ctx,
                                      "Pass a checkpointer such as MemorySaver to compile().")
        t0 = time.perf_counter()
        raw = message_text(guard.invoke([SystemMessage(content=VERIFY_SYSTEM), HumanMessage(content=prompt)]))
        verdict = _parse_verdict(raw)
        print(f"  guard fallback {BEDROCK_GUARD_MODEL_ID} ({time.perf_counter() - t0:.1f}s): "
              f"{'parsed ' + json.dumps(verdict) if verdict else 'UNPARSEABLE: ' + raw[:160]!r}")

    sys.exit(0 if ok == len(router_cases) else 1)