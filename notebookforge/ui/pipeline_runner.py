"""Adapters for running NotebookForge without a separate HTTP backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class DirectPipelineError(RuntimeError):
    """The pipeline finished without a notebook that the UI can render."""


def run_pipeline_direct(
    profile_data: dict[str, Any],
    *,
    generate_fn: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``main.generate`` in-process and return the same shape as the API UI.

    Imports are intentionally lazy. The Streamlit Cloud entrypoint copies secrets
    to environment variables before this function imports ``main``/``llm_client``.
    """
    from schemas import GenerationResult, LearnerProfile
    from ui.report_adapter import normalize_report_payload

    if generate_fn is None:
        from main import generate

        generate_fn = generate

    profile = LearnerProfile.model_validate(profile_data)
    result = generate_fn(profile)
    if not isinstance(result, GenerationResult):
        raise DirectPipelineError(
            "main.generate phải trả GenerationResult, nhận "
            f"{type(result).__name__}"
        )

    notebook_path = Path(result.notebook_path) if result.notebook_path else None
    if notebook_path is None or not notebook_path.is_file():
        detail = result.error or f"decision={result.decision}"
        raise DirectPipelineError(
            f"Pipeline không tạo được notebook để hiển thị ({detail})."
        )

    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DirectPipelineError(
            f"Không đọc được notebook '{notebook_path}': {error}"
        ) from error

    payload = result.model_dump(mode="json")
    report = normalize_report_payload(payload)
    if report is None:
        raise DirectPipelineError("Không chuẩn hóa được Quality Report.")
    return notebook, report


__all__ = ["DirectPipelineError", "run_pipeline_direct"]
