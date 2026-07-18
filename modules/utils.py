"""Common utility helpers for ResumeScreening-AI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


__all__ = [
    "load_css",
    "safe_filename",
    "error_message",
]


def load_css(css_path: str | Path) -> None:
    """Load a CSS stylesheet into the Streamlit application."""
    path = Path(css_path)

    if not path.exists():
        return

    st.markdown(
        f"<style>{path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def safe_filename(filename: str) -> str:
    """Return a readable filename without extension."""
    return Path(filename).stem.replace("_", " ").replace("-", " ")


def error_message(error: Exception) -> str:
    """Return a clean user-facing error message."""
    message = str(error).strip()

    if not message:
        return "An unexpected error occurred."

    return message