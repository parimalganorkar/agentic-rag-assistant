"""Phase 11 — Streamlit chat UI for the agentic RAG assistant (styled).

Run:  ./env/Scripts/python.exe -m streamlit run app/streamlit_app.py

Presentation only. The warm MCP session, agent and multi-turn memory live in
app/runtime.py; the AgentRuntime is created once via @st.cache_resource. The
premium look comes from .streamlit/config.toml (base theme) + app/style.css
(injected below). Widgets carry explicit key= values so the CSS can target them
via Streamlit's .st-key-<key> container classes.

Conversations behave like a chat app's left rail — switchable, deletable, each
keeping its own agent memory (its id IS the LangGraph thread_id). The list lives
in st.session_state: per browser session, RAM-only, so it does not survive a
server restart (a SqliteSaver + disk store would persist it).
"""

from __future__ import annotations

import hmac
import os
import sys
import time
import uuid
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts THIS file's folder (app/) on sys.path,
# not the project root — so `import app.runtime` and the agent/retrieval imports
# inside it fail with ModuleNotFoundError. Put the project root first so the app
# runs with a plain `streamlit run`, no PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from dotenv import load_dotenv

# Load (never read) .env before anything else so APP_PASSWORD and the limits
# below are in place before the gate renders.
load_dotenv(_REPO_ROOT / ".env")

from app.runtime import AgentRuntime, TurnResult

st.set_page_config(page_title="Docs Assistant", page_icon="📚", layout="centered")

# --- Deployment safety knobs (all env-configurable) --------------------------
# Shared passcode for the login gate. EMPTY = the app refuses to start rather
# than silently running open: on a public box "forgot to set it" must not mean
# "anyone can spend our LLM budget".
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
# Longest message we'll send to the LLM. A hard server-side cap, not just the
# chat box's client-side max_chars (which a request can bypass).
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2000"))
# Messages per browser session. Session-state only — no Redis, no accounts —
# which is the level of protection a passcode-gated demo needs.
MAX_MESSAGES_PER_SESSION = int(os.getenv("MAX_MESSAGES_PER_SESSION", "20"))

USER_AVATAR = "🧑‍💻"
AI_AVATAR = "📚"

# Rerank scores are cross-encoder logits (roughly -11..+8), not 0-1 similarities,
# so the UI never shows them as a percentage — see the sources expander.
# Colors are the lighter GitHub-dark variants so the pills stay legible on the
# dark surface (the earlier dark hexes vanished against the background).
GUARD_BADGE = {
    "pass":   ("✓ grounded", "#3FB950"),
    "repair": ("✎ repaired", "#D29922"),
    "refuse": ("⚠ refused",  "#F85149"),
}


# --------------------------------------------------------------------------- #
# One-time resources                                                            #
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Warming up the agent (loading models + MCP tools)…")
def get_runtime() -> AgentRuntime:
    return AgentRuntime()


@st.cache_resource(show_spinner=False)
def _init_tracing() -> bool:
    """Turn on LangSmith tracing ONCE, up front — before the sidebar reads its
    status. Previously tracing was only configured deep inside get_runtime(),
    which runs AFTER the sidebar renders, so the status panel always showed 'off'
    on first paint even with a valid key. Cached so the region probe runs once."""
    from agent.observability import configure_tracing
    return configure_tracing(validate=True)


def _load_css() -> str:
    # Not cached: a tiny file, and reading it each run means style.css edits show
    # up on rerun without a server restart.
    path = Path(__file__).with_name("style.css")
    return path.read_text(encoding="utf-8") if path.exists() else ""


@st.cache_data
def _corpus_docs() -> int | None:
    """Count of in-scope docs, read straight from the manifest (cheap, local)."""
    import json
    mp = Path(__file__).resolve().parents[1] / "data" / "cleaned" / "manifest.json"
    if not mp.exists():
        return None
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
        entries = data.get("entries", data) if isinstance(data, dict) else data
        seq = entries.values() if isinstance(entries, dict) else entries
        return sum(1 for e in seq if isinstance(e, dict) and e.get("status") == "ok")
    except Exception:
        return None


def _inject_css() -> None:
    css = _load_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Login gate + session limits                                                   #
# --------------------------------------------------------------------------- #

def _login_gate() -> None:
    """Block the whole page behind a shared passcode. Nothing below main()'s
    call to this renders until `st.session_state.authed` is True — including
    the sidebar and the (expensive) runtime warm-up."""
    if st.session_state.get("authed"):
        return

    st.title("LangChain Docs Assistant")
    if not APP_PASSWORD:
        st.error("APP_PASSWORD is not set. Add it to .env (see .env.example) and restart.")
        st.stop()

    with st.form("login", clear_on_submit=True):
        pw = st.text_input("Passcode", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        # Constant-time compare — cheap insurance against timing leaks on a
        # single shared secret.
        if hmac.compare_digest(pw.encode("utf-8"), APP_PASSWORD.encode("utf-8")):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong passcode.")
    st.stop()


def _messages_left() -> int:
    return MAX_MESSAGES_PER_SESSION - st.session_state.get("messages_sent", 0)


# --------------------------------------------------------------------------- #
# Conversation store (session_state): {id -> {title, history, updated}}         #
# --------------------------------------------------------------------------- #

def _new_conversation() -> str:
    cid = f"web-{uuid.uuid4().hex[:12]}"
    st.session_state.conversations[cid] = {"title": "New chat", "history": [], "updated": time.time()}
    st.session_state.active_id = cid
    return cid


def _ensure_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if not st.session_state.conversations:
        _new_conversation()
    if "active_id" not in st.session_state or st.session_state.active_id not in st.session_state.conversations:
        st.session_state.active_id = next(iter(st.session_state.conversations))


def _active() -> dict:
    return st.session_state.conversations[st.session_state.active_id]


def _title_from(text: str) -> str:
    t = " ".join(text.split())
    return (t[:34] + "…") if len(t) > 34 else t


# --------------------------------------------------------------------------- #
# Rendering                                                                     #
# --------------------------------------------------------------------------- #

def _pill(text: str, color: str = "#444") -> str:
    return (f"<span style='background:{color}22;color:{color};border:1px solid {color}55;"
            f"border-radius:10px;padding:1px 8px;font-size:0.72rem;"
            f"white-space:nowrap'>{text}</span>")


def _render_meta(m: TurnResult) -> None:
    route = " → ".join(m.route) if m.route else "—"
    if m.escalated:
        route += " (escalated→live)"
    guard_text, guard_color = GUARD_BADGE.get(m.guard_action, (m.guard_action, "#444"))

    pills = [_pill(f"route: {route}", "#58A6FF")]
    if m.tool_name:
        pills.append(_pill(f"tool: {m.tool_name}", "#A371F7"))
    pills.append(_pill(guard_text, guard_color))
    if m.cache_hit:
        pills.append(_pill("⚡ cached", "#3FB950"))
    pills.append(_pill(f"{m.latency_s}s", "#8B949E"))
    if m.pii_redacted:
        pills.append(_pill(f"🔒 redacted: {', '.join(m.pii_redacted)}", "#F85149"))
    st.markdown(" ".join(pills), unsafe_allow_html=True)

    if m.guard_repairs:
        st.caption("Repaired before showing: " + "; ".join(m.guard_repairs))

    if m.sources:
        with st.expander(f"📄 Sources ({len(m.sources)})"):
            st.caption("Ranked by the cross-encoder reranker (score = rerank logit, "
                       "higher = more relevant — not a percentage).")
            for i, s in enumerate(m.sources, 1):
                title = s.get("title") or s["source_file"].split("/")[-1]
                st.markdown(
                    f"**[S{i}]** {title}  \n"
                    f"<span style='color:#8B949E;font-size:0.8rem'>"
                    f"{s['source_file']} · {s.get('section','')} · score {s['score']}</span>",
                    unsafe_allow_html=True,
                )


def _bubble(role: str, avatar: str, content: str, meta: TurnResult | None = None) -> None:
    with st.chat_message(role, avatar=avatar):
        # Invisible role marker so style.css can target user vs assistant bubbles
        # reliably (avatar testids change when a custom avatar is set).
        st.markdown(f"<span class='role-{role}'></span>", unsafe_allow_html=True)
        st.markdown(content)
        if meta is not None:
            _render_meta(meta)


def _toast_for(result: TurnResult) -> None:
    if result.guard_action == "refuse":
        st.toast("Answer withheld by the safety guard", icon="⚠️")
    elif result.cache_hit:
        st.toast("Served from cache", icon="⚡")
    else:
        st.toast(f"Answered in {result.latency_s}s", icon="✅")


def _status_row(label: str, on: bool, detail: str) -> str:
    dot = "dot-on" if on else "dot-off"
    return (f"<div style='font-size:0.82rem;margin:2px 0'>"
            f"<span class='status-dot {dot}'></span><b>{label}</b> "
            f"<span style='color:#6b6b7b'>{detail}</span></div>")


def _sidebar() -> None:
    from agent.observability import tracing_status

    with st.sidebar:
        st.header("📚 Docs Assistant")
        if st.button("➕  New chat", key="new_chat", use_container_width=True, type="primary"):
            _new_conversation()
            st.toast("New chat started", icon="➕")
            st.rerun()

        st.caption("Conversations")
        for cid, conv in sorted(st.session_state.conversations.items(),
                                key=lambda kv: kv[1]["updated"], reverse=True):
            is_active = cid == st.session_state.active_id
            col_open, col_del = st.columns([0.83, 0.17])
            if col_open.button(conv["title"], key=f"open_{cid}", use_container_width=True,
                               disabled=is_active):
                st.session_state.active_id = cid
                st.rerun()
            if col_del.button("🗑", key=f"del_{cid}", use_container_width=True,
                              help="Delete this conversation"):
                del st.session_state.conversations[cid]
                if not st.session_state.conversations:
                    _new_conversation()
                elif st.session_state.active_id == cid:
                    st.session_state.active_id = next(iter(st.session_state.conversations))
                st.toast("Conversation deleted", icon="🗑️")
                st.rerun()

        st.divider()

        # Backend status — a real indicator, not decoration.
        with st.expander("⚙️  Backend status", expanded=False):
            ts = tracing_status()
            docs = _corpus_docs()
            if ts["tracing"]:
                trace_detail = f"on · {ts['project']}"
            elif ts["has_key"]:
                trace_detail = "off · key present but not connected"
            else:
                trace_detail = "off · no key in .env"
            from agent.llm import provider_label
            rows = [
                _status_row("LLM", True, provider_label()),
                _status_row("Retrieval", True, "hybrid + rerank · bge-small"),
                _status_row("Corpus", docs is not None,
                            f"{docs} docs · pinned commit" if docs else "manifest not found"),
                _status_row("LangSmith tracing", ts["tracing"], trace_detail),
            ]
            st.markdown("".join(rows), unsafe_allow_html=True)

        st.divider()
        with st.expander("💡  Try asking"):
            st.caption("• How do I add a tool to an agent?\n\n"
                       "• What's the latest version of langgraph?\n\n"
                       "• How fresh are your docs / what do you cover?\n\n"
                       "• Fetch the current LangGraph Studio guide")
        st.caption("Memory is per browser session (RAM-only); a server restart clears it.")
        st.caption(f"Messages this session: {st.session_state.get('messages_sent', 0)} / "
                   f"{MAX_MESSAGES_PER_SESSION}")


def main() -> None:
    _inject_css()
    _login_gate()            # st.stop()s here until the passcode is accepted
    _ensure_state()
    _init_tracing()          # set LangSmith state BEFORE the sidebar reads it
    _sidebar()

    st.title("LangChain Docs Assistant")
    st.caption("Agentic RAG over LangChain / LangGraph docs — routing, hybrid retrieval, "
               "MCP tools, and a five-guard safety layer.")

    runtime = get_runtime()
    conv = _active()

    for role, content, meta in conv["history"]:
        avatar = USER_AVATAR if role == "user" else AI_AVATAR
        _bubble(role, avatar, content, meta)

    left = _messages_left()
    if left <= 0:
        st.warning(f"This session has reached its limit of {MAX_MESSAGES_PER_SESSION} messages. "
                   "Start a new browser session to continue.")
        st.chat_input("Session limit reached", disabled=True)
        return

    prompt = st.chat_input(f"Ask about LangChain or LangGraph… ({left} left this session)",
                           max_chars=MAX_INPUT_CHARS)
    if not prompt:
        return
    prompt = prompt.strip()
    if not prompt:
        return
    # max_chars is enforced by the browser widget; re-check here because the
    # server must not trust the client on anything that costs money.
    if len(prompt) > MAX_INPUT_CHARS:
        st.error(f"Message too long ({len(prompt)} chars). The limit is {MAX_INPUT_CHARS}.")
        return
    st.session_state.messages_sent = st.session_state.get("messages_sent", 0) + 1

    conv["history"].append(("user", prompt, None))
    if conv["title"] == "New chat":
        conv["title"] = _title_from(prompt)
    _bubble("user", USER_AVATAR, prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        st.markdown("<span class='role-assistant'></span>", unsafe_allow_html=True)
        with st.spinner("Thinking… (routing → retrieval → guards)"):
            result = runtime.ask(prompt, st.session_state.active_id)
        st.markdown(result.answer)
        _render_meta(result)

    conv["history"].append(("assistant", result.answer, result))
    conv["updated"] = time.time()
    _toast_for(result)


if __name__ == "__main__":
    main()
