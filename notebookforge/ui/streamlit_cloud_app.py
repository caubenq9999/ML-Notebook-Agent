"""Streamlit Community Cloud entrypoint for NotebookForge.

This file loads deployment secrets before importing the main UI, then switches
the UI to the in-process pipeline runner so no separate FastAPI service is
required.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
import runpy

import streamlit as st


st.set_page_config(page_title="NotebookForge", page_icon="📓", layout="wide")

_ENV_SECRET_KEYS = (
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "NOTEBOOKFORGE_PROVIDER",
    "NOTEBOOKFORGE_MODEL",
    "NOTEBOOKFORGE_MODEL_JUDGE",
    "NOTEBOOKFORGE_DEEPSEEK_THINKING",
    "NOTEBOOKFORGE_NOTEBOOK_MAX_TOKENS",
    "NOTEBOOKFORGE_MAX_TOKENS",
    "NOTEBOOKFORGE_TEMPERATURE",
    "NOTEBOOKFORGE_LLM_RETRIES",
    "NOTEBOOKFORGE_CELL_TIMEOUT",
    "NOTEBOOKFORGE_UI_ALLOW_MOCK_FALLBACK",
)


def _cloud_secrets() -> dict[str, object]:
    try:
        return dict(st.secrets)
    except Exception:  # No local secrets.toml while developing.
        return {}


def _copy_secrets_to_environment(secrets: dict[str, object]) -> None:
    for key in _ENV_SECRET_KEYS:
        value = secrets.get(key)
        if value is not None and str(value).strip():
            os.environ[key] = str(value)


def _require_demo_access(secrets: dict[str, object]) -> None:
    password = str(secrets.get("APP_PASSWORD") or os.getenv("APP_PASSWORD", ""))
    allow_public = str(
        secrets.get("NOTEBOOKFORGE_ALLOW_PUBLIC")
        or os.getenv("NOTEBOOKFORGE_ALLOW_PUBLIC", "false")
    ).strip().lower() in {"1", "true", "yes"}

    if allow_public:
        st.warning(
            "Public demo mode is enabled. Every visitor can consume the configured"
            " LLM API credit."
        )
        return

    if not password:
        st.error(
            "Cloud demo is locked because APP_PASSWORD is not configured. Add it"
            " in Streamlit App settings > Secrets."
        )
        st.stop()

    if st.session_state.get("_cloud_authenticated"):
        return

    st.title("📓 NotebookForge Cloud Demo")
    entered = st.text_input("Mật khẩu demo", type="password")
    if st.button("Đăng nhập", type="primary"):
        if hmac.compare_digest(entered, password):
            st.session_state._cloud_authenticated = True
            st.rerun()
        st.error("Mật khẩu không đúng.")
    st.stop()


cloud_secrets = _cloud_secrets()
_copy_secrets_to_environment(cloud_secrets)
os.environ["NOTEBOOKFORGE_DEPLOY_MODE"] = "direct"
os.environ.setdefault("NOTEBOOKFORGE_UI_ALLOW_MOCK_FALLBACK", "0")
_require_demo_access(cloud_secrets)

runpy.run_path(
    str(Path(__file__).with_name("streamlit_app.py")),
    run_name="__main__",
)
