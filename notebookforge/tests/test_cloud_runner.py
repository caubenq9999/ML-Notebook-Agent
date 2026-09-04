"""No-cost tests for the Streamlit Cloud direct pipeline adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tests import mocks
from ui.pipeline_runner import DirectPipelineError, run_pipeline_direct


def test_direct_runner_returns_fail_max_retry_notebook() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        notebook_path = Path(tmp) / "notebook.ipynb"
        notebook_path.write_text(
            json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
            encoding="utf-8",
        )
        result = mocks.MOCK_RESULT.model_copy(
            update={
                "notebook_path": str(notebook_path),
                "decision": "FAIL_MAX_RETRY",
            }
        )
        notebook, report = run_pipeline_direct(
            mocks.MOCK_PROFILE.model_dump(mode="json"),
            generate_fn=lambda profile: result,
        )

    assert notebook["nbformat"] == 4
    assert report["status"] == "FAIL_MAX_RETRY"


def test_direct_runner_rejects_missing_notebook() -> None:
    result = mocks.MOCK_RESULT.model_copy(
        update={"notebook_path": None, "decision": "FAIL_EXECUTION"}
    )
    try:
        run_pipeline_direct(
            mocks.MOCK_PROFILE.model_dump(mode="json"),
            generate_fn=lambda profile: result,
        )
    except DirectPipelineError as error:
        assert "không tạo được notebook" in str(error)
        return
    raise AssertionError("Missing notebook must raise DirectPipelineError")


if __name__ == "__main__":
    from tests._runner import run_module

    raise SystemExit(run_module(__name__))
