"""API 2 endpoint + schema contract (đề cương mục 1.1, 2.8)

    POST /generate -> 202 · GET /report/{id} -> kết quả · quy tắc hạ trình độ

Chạy: python tests/test_api_schemas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from schemas import (  # noqa: E402
    COST_CAP_USD,
    MAX_ATTEMPT,
    Citation,
    Constraints,
    GenerationResult,
    LearnerProfile,
    ResearchBundle,
    Source,
    decide_level_final,
)
from tests import mocks  # noqa: E402


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_seed_random_moi_session():
    """Sprint 1 sửa lỗi seed cố định -> mỗi session một seed khác nhau"""
    a = LearnerProfile(topic="t", level_declared=1, level_final=1, quiz_score=3,
                       session_id="a").dataset_seed
    b = LearnerProfile(topic="t", level_declared=1, level_final=1, quiz_score=3,
                       session_id="b").dataset_seed
    assert a != b, "seed phải random, không được fix cứng"


def test_extra_forbid_bat_typo():
    """Gõ sai tên field -> báo lỗi ngay, không âm thầm nuốt"""
    try:
        LearnerProfile(topic="t", level_declared=1, level_final=1, quiz_score=3,
                       session_id="s", level_finaL=2)
    except ValidationError as exc:
        assert "extra_forbidden" in str(exc)
        return
    raise AssertionError("phải chặn field lạ")


def test_citation_phai_tro_toi_source_co_that():
    """Citation trỏ tới source_id không tồn tại -> chặn"""
    try:
        ResearchBundle(topic="t", sources=[], key_concepts=["a"],
                       citations=[Citation(concept="a", source_id="ma")])
    except ValidationError:
        return
    raise AssertionError("phải chặn citation mồ côi")


def test_unresolved_concepts_tu_suy_ra():
    """Khái niệm không có citation -> tự vào unresolved_concepts"""
    assert mocks.MOCK_BUNDLE.unresolved_concepts == ["regularization"]


def test_round_trip_json():
    """model_validate(model_dump()) - api.py và Streamlit truyền JSON qua lại"""
    back = GenerationResult.model_validate_json(mocks.MOCK_RESULT.model_dump_json())
    assert back.report.average_score == mocks.MOCK_RESULT.report.average_score


def test_average_score_chi_mot_cong_thuc():
    """average_score là computed field, không ai set tay được"""
    assert mocks.MOCK_REPORT_PASS.average_score == 4.0
    assert mocks.MOCK_REPORT_RETRY.average_score == 2.9


def test_quy_tac_ha_trinh_do():
    """Đề cương mục 1.1: intermediate mà quiz < 3/5 thì hạ xuống beginner"""
    assert decide_level_final(2, 0) == 1
    assert decide_level_final(2, 2) == 1
    assert decide_level_final(2, 3) == 2
    assert decide_level_final(2, 5) == 2
    assert decide_level_final(1, 0) == 1, "beginner không hạ thêm được"
    assert decide_level_final(1, 5) == 1, "không có chiều nâng"


def test_source_web_search_khop_research_sprint22():
    """Research của Trí dùng type=web_search cho nguồn fallback."""
    source = Source(
        source_id="web_demo",
        type="web_search",
        path_or_url="https://example.com",
    )
    assert source.type == "web_search"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def _client():
    from fastapi.testclient import TestClient

    import api

    return TestClient(api.app)


def test_health_phan_anh_dung_cau_hinh():
    """/health trả đúng hằng số đang chạy"""
    h = _client().get("/health").json()
    assert h["max_attempt"] == MAX_ATTEMPT
    assert h["cost_cap_usd"] == COST_CAP_USD


def test_generate_roi_report():
    """POST /generate -> 202, GET /report/{id} -> completed + payload cho UI"""
    c = _client()
    r = c.post("/generate", json={
        "topic": "logistic_regression", "level": 2, "quiz_score": 2,
        "duration_minutes": 60, "num_exercises": 3, "use_mock": True,
    })
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]
    assert r.json()["task_id"] == sid, "UI của Nam đọc task_id"
    assert r.json()["poll_url"] == f"/report/{sid}"

    res = c.get(f"/report/{sid}").json()
    assert res["status"] == "completed", res
    gr = res["result"]
    assert gr["decision"] == "PASS"
    assert gr["profile"]["level_final"] == 1, "khai intermediate + quiz 2/5 -> hạ beginner"
    assert set(res["artifact_urls"]) == {"notebook", "path", "quality_report"}
    assert isinstance(res["notebook"], dict), "UI cần notebook JSON để tải xuống"
    assert isinstance(res["report"], dict), "UI cần report dạng dict"
    assert isinstance(res["quality_report"], str)

    expected = {
        "notebook": "notebook.ipynb",
        "path": "path.json",
        "quality_report": "quality_report.md",
    }
    for name, filename in expected.items():
        artifact = c.get(res["artifact_urls"][name])
        assert artifact.status_code == 200, (name, artifact.text)
        assert filename in artifact.headers.get("content-disposition", "")
        assert artifact.content, f"artifact {name} không được rỗng"


def test_fail_max_retry_van_la_pipeline_completed_va_giu_result():
    """Không đạt quality gate vẫn phải trả result/notebook tốt nhất cho UI."""
    import api

    sid = "test-fail-max-retry"
    failed_quality = mocks.MOCK_RESULT.model_copy(
        update={"decision": "FAIL_MAX_RETRY"}
    )
    api._store_finish(sid, failed_quality)

    try:
        entry = api._STORE[sid]
        assert entry["status"] == "completed"
        assert entry["result"].decision == "FAIL_MAX_RETRY"
    finally:
        with api._LOCK:
            api._STORE.pop(sid, None)


def test_artifact_la_allowlist_khong_nhan_duong_dan_tuy_y():
    """Endpoint download chỉ nhận ba tên logic, không nhận filename/path tuỳ ý."""
    c = _client()
    r = c.post("/generate", json={
        "topic": "logistic_regression", "level": 1, "quiz_score": 3,
        "duration_minutes": 60, "num_exercises": 3, "use_mock": True,
    })
    sid = r.json()["session_id"]
    assert c.get(f"/artifacts/{sid}/result.json").status_code == 404
    assert c.get(f"/artifacts/{sid}/unknown").status_code == 404


def test_report_khong_co_thi_404():
    """Session lạ -> 404, không trả 200 rỗng"""
    assert _client().get("/report/khong-ton-tai").status_code == 404


def test_generate_chan_input_sai_khoang():
    """Đề cương: 60-120 phút, 3-5 bài. Ngoài khoảng -> 422"""
    c = _client()
    assert c.post("/generate", json={
        "topic": "t", "level": 1, "quiz_score": 3,
        "duration_minutes": 15, "num_exercises": 3,
    }).status_code == 422, "15 phút phải bị chặn"
    assert c.post("/generate", json={
        "topic": "t", "level": 1, "quiz_score": 3,
        "duration_minutes": 60, "num_exercises": 1,
    }).status_code == 422, "1 bài tập phải bị chặn"


def test_payload_learner_profile_cua_ui_chay_dung_contract():
    """Payload thật của Streamlit không được bị API bỏ qua rồi dùng default."""
    c = _client()
    payload = {
        "session_id": "ui-contract-test",
        "created_at": "2026-08-18T00:00:00+07:00",
        "topic": "decision_tree",
        "level_declared": 2,
        "level_final": 1,
        "quiz_score": 2,
        "constraints": {"duration_minutes": 90, "num_exercises": 4},
        "use_mock": True,
    }
    response = c.post("/generate", json=payload)
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]
    assert task_id == payload["session_id"]

    report = c.get(f"/report/{task_id}").json()
    assert report["status"] == "completed", report
    profile = report["result"]["profile"]
    assert profile["topic"] == "decision_tree"
    assert profile["level_declared"] == 2
    assert profile["level_final"] == 1
    assert profile["constraints"]["duration_minutes"] == 90
    assert profile["constraints"]["num_exercises"] == 4


def test_api_chan_field_la_va_level_final_sai():
    """Sai contract phải 422, không âm thầm chạy bằng giá trị mặc định."""
    c = _client()
    base = {
        "topic": "logistic_regression",
        "level_declared": 2,
        "level_final": 1,
        "quiz_score": 2,
        "constraints": {"duration_minutes": 60, "num_exercises": 3},
    }
    assert c.post("/generate", json={**base, "level_decalred": 2}).status_code == 422
    assert c.post(
        "/generate",
        json={**base, "quiz_score": 5, "level_final": 1},
    ).status_code == 422


if __name__ == "__main__":
    from tests._runner import run_module

    raise SystemExit(run_module(__name__))
