"""UI report adapter - giữ contract giữa FastAPI và Streamlit."""

from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ui.report_adapter import normalize_report_payload


def test_current_generation_result_is_flattened_for_ui() -> None:
    """GenerationResult.report.llm_scores phải hiện được trong bảng UI."""
    payload = {
        "status": "completed",
        "result": {
            "decision": "PASS",
            "attempts_used": 1,
            "total_cost_usd": 0.0075,
            "retry_history": [{"attempt": 1, "average_score": 4.75}],
            "report": {
                "llm_scores": {
                    "groundedness": 5.0,
                    "difficulty_fit": 5.0,
                    "pedagogical_order": 4.0,
                    "content_completeness": 4.5,
                    "learning_coverage": 4.5,
                    "average": 4.6,
                },
                "rule_checks": {"all_passed": True},
                "feedback": "[CELL 1] lỗi. FIX: sửa.",
            },
        },
    }

    report = normalize_report_payload(payload)
    assert report is not None
    assert report["scores"]["average"] == 4.6
    assert report["status"] == "PASS"
    assert report["attempts_used"] == 1
    assert report["feedback"].startswith("[CELL 1]")


def test_api_report_alias_is_supported() -> None:
    """Field report ngoài cùng của API vẫn được đọc như GenerationResult."""
    payload = {
        "report": {
            "decision": "FAIL_MAX_RETRY",
            "attempts_used": 2,
            "report": {"llm_scores": {"content_completeness": 3.0}},
        }
    }
    report = normalize_report_payload(payload)
    assert report is not None
    assert report["scores"]["content_completeness"] == 3.0
    assert report["attempts_used"] == 2


def test_generation_result_stored_in_session_state_is_supported() -> None:
    """UI rerun phải đọc được GenerationResult đã lưu từ lần chạy trước."""
    generation = {
        "decision": "PASS",
        "attempts_used": 1,
        "retry_history": [{"attempt": 1}],
        "report": {
            "llm_scores": {"average": 4.75},
            "feedback": None,
        },
    }
    report = normalize_report_payload(generation)
    assert report is not None
    assert report["scores"]["average"] == 4.75
    assert report["attempts_used"] == 1


def test_legacy_flat_report_remains_compatible() -> None:
    """Mock/report cũ có scores trực tiếp không bị phá."""
    legacy = {
        "scores": {"executability": 1.0},
        "status": "completed",
        "feedback": "mock",
    }
    assert normalize_report_payload(legacy) == legacy


if __name__ == "__main__":
    from tests._runner import run_module

    raise SystemExit(run_module(__name__))
