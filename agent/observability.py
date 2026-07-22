"""Phase 10 — LangSmith tracing setup.

LangGraph/LangChain emit traces automatically when the LANGSMITH_* environment
is configured, so this module's whole job is: load .env, and if a LangSmith key
is present, switch tracing on. If no key is present it NO-OPS silently — the
agent must run fully offline with no account, per the project ethos, so a missing
key is a normal state, not an error.

We never read the .env file directly (project rule); load_dotenv pulls the key
into the process, and `tracing_status()` reports whether it worked by checking
os.environ, not by opening the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DEFAULT_PROJECT = "rag-build-agent"


_validated: bool | None = None  # per-process: None=unchecked, True/False=result


def _disable() -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# The three LangSmith regions. A key only authenticates against its OWN region,
# so a US-default validation 403s for EU/APAC accounts. We probe these in order
# when no endpoint is pinned, so tracing "just works" without the user having to
# know their region.
_REGION_ENDPOINTS = [
    "https://api.smith.langchain.com",        # US (default)
    "https://eu.api.smith.langchain.com",     # EU
    "https://apac.api.smith.langchain.com",   # APAC
]


def _key_works(endpoint: str | None = None) -> bool:
    """One lightweight authenticated call against `endpoint` (or the SDK default),
    so a bad key/region is caught up front instead of spamming 403s from the
    background trace uploader on every run."""
    try:
        from langsmith import Client
        client = Client(api_url=endpoint) if endpoint else Client()
        next(iter(client.list_projects(limit=1)), None)
        return True
    except Exception:
        return False


def configure_tracing(validate: bool = True) -> bool:
    """Enable LangSmith tracing if a WORKING API key is available. Returns True if on.

    Idempotent — safe to call from every entry point. Sets both the modern
    LANGSMITH_* and legacy LANGCHAIN_* names because different library versions
    read different ones. When `validate` is set (default), the key is checked with
    one cheap authenticated request the first time this runs in a process; a
    rejected key (403/401 — wrong key, wrong region, revoked) disables tracing
    with a single clear message instead of letting the uploader retry-spam.
    """
    global _validated
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not key:
        _disable()  # a stray env var must not half-enable tracing with no key
        return False

    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or DEFAULT_PROJECT
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project

    if not validate:
        return True
    if _validated is True:
        return True
    if _validated is False:
        _disable()
        return False

    # If the user pinned an endpoint, respect it and validate only that. Otherwise
    # probe the regions in order and adopt whichever the key authenticates against.
    explicit = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    candidates = [explicit] if explicit else _REGION_ENDPOINTS

    working = next((ep for ep in candidates if _key_works(ep)), None)
    if working is None:
        _validated = False
        _disable()
        print("[observability] LangSmith key REJECTED on "
              f"{'the pinned endpoint' if explicit else 'all regions (US/EU/APAC)'} "
              "— tracing DISABLED.\n"
              "                Likely: key revoked, or copied with stray quotes/spaces"
              + (f"; or the pinned LANGSMITH_ENDPOINT ({explicit}) is wrong." if explicit else "."))
        return False

    # Pin the working endpoint so the background trace uploader targets it too.
    os.environ["LANGSMITH_ENDPOINT"] = working
    os.environ["LANGCHAIN_ENDPOINT"] = working
    _validated = True
    if not explicit and working != _REGION_ENDPOINTS[0]:
        region = "EU" if "eu." in working else ("APAC" if "apac." in working else "regional")
        print(f"[observability] LangSmith: auto-detected {region} region ({working}). "
              f"Add LANGSMITH_ENDPOINT={working} to .env to skip this probe next time.")
    return True


def tracing_status() -> dict:
    """Report tracing state without touching the .env file."""
    on = os.getenv("LANGSMITH_TRACING") == "true"
    return {
        "tracing": on,
        "project": os.getenv("LANGSMITH_PROJECT") if on else None,
        "has_key": bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")),
    }


if __name__ == "__main__":
    active = configure_tracing(validate=True)
    if active:
        print("LangSmith tracing: ENABLED (key validated against the server)")
    else:
        has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
        print("LangSmith tracing: DISABLED "
              + ("(key present but REJECTED — see message above)" if has_key
                 else "(no key in .env)"))
    print(tracing_status())
