"""Vòng lặp main.py - 3 điều kiện dừng + 2 nhánh lỗi (đề cương mục 2.1)

    PASS (>= 3.5) · FAIL_COST_CAP ($0.30) · FAIL_MAX_RETRY (2 vòng) · FAIL_EXECUTION

Chạy toàn bằng mock, KHÔNG gọi LLM, không tốn quota.
Chạy: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as m  # noqa: E402
from llm_client import Usage, get_tracker, reset_tracker  # noqa: E402
from schemas import (  # noqa: E402
    COST_CAP_USD,
    MAX_ATTEMPT,
    Constraints,
    LearnerProfile,
    LlmScores,
    RuleChecks,
)
from tests import mocks  # noqa: E402

_BASE = dict(
    topic="logistic_regression",
    level_declared=1,
    level_final=1,
    quiz_score=4,
    constraints=Constraints(duration_minutes=60, num_exercises=3),
)


def _profile(session_id: str) -> LearnerProfile:
    reset_tracker(session_id)
    return LearnerProfile(session_id=session_id, dataset_seed=1, **_BASE)


def _wire(**overrides):
    """Thay _resolve_impls bằng bộ hàm giả, cho phép ghi đè từng bước."""
    impls = {
        "run_research": mocks.fake_run_research,
        "run_curriculum": mocks.fake_run_curriculum,
        "run_notebook_gen": mocks.fake_run_notebook_gen,
        "run_verifier": mocks.fake_run_verifier,
    }
    impls.update(overrides)
    m._resolve_impls = lambda *a, **kw: impls


def _always_retry(nb_path, exc, bundle):
    """Verifier luôn chấm dưới ngưỡng -> loop không bao giờ PASS."""
    return mocks.MOCK_REPORT_RETRY.model_copy(
        update={"nb_path": nb_path, "attempt": exc.attempt}
    )


def test_chay_that_khong_am_tham_thay_module_thieu_bang_mock():
    """Thiếu implementation thật phải FAIL rõ, chỉ mock khi người chạy yêu cầu."""
    import builtins

    original_import = builtins.__import__

    def fail_verifier_import(name, *args, **kwargs):
        if name == "agents.verifier":
            raise ImportError("giả lập verifier chưa được bàn giao")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = fail_verifier_import
    try:
        try:
            m._resolve_impls(
                force_mock=False,
                mock_steps=("research", "curriculum", "notebook_gen"),
            )
        except RuntimeError as exc:
            assert "agents.verifier.run_verifier" in str(exc)
        else:
            raise AssertionError("không được âm thầm dùng verifier mock")
    finally:
        builtins.__import__ = original_import


def test_ten_buoc_mock_sai_bao_loi_ngay():
    """Gõ sai --mock phải báo lỗi, không được âm thầm chạy chế độ khác."""
    try:
        m._resolve_impls(force_mock=False, mock_steps=("verifer",))
    except ValueError as exc:
        assert "verifer" in str(exc)
        return
    raise AssertionError("tên bước mock sai phải bị chặn")


def test_pass_khi_diem_du_nguong():
    """average_score >= 3.5 -> PASS ngay vòng 1"""
    _wire()
    r = m.generate(_profile("t-pass"), force_mock=True)
    assert r.decision == "PASS", r.decision
    assert r.attempts_used == 1
    assert r.notebook_path and Path(r.notebook_path).is_file()


def test_fail_max_retry():
    """Điểm luôn thấp -> FAIL_MAX_RETRY sau đúng MAX_ATTEMPT vòng"""
    _wire(run_verifier=_always_retry)
    r = m.generate(_profile("t-maxretry"), force_mock=True)
    assert r.decision == "FAIL_MAX_RETRY", r.decision
    assert r.attempts_used == MAX_ATTEMPT == 2
    assert len(r.retry_history) == 2, "audit trail phải ghi đủ 2 vòng"


def test_fail_cost_cap():
    """Chạm trần $0.30 -> FAIL_COST_CAP, dừng trước khi hết lượt"""
    sid = "t-costcap"

    def burn_then_gen(path, profile, attempt, feedback):
        get_tracker(sid).add(Usage(model="test", cost_usd=0.31))
        return mocks.fake_run_notebook_gen(path, profile, attempt, feedback)

    _wire(run_notebook_gen=burn_then_gen, run_verifier=_always_retry)
    r = m.generate(_profile(sid), force_mock=True)
    assert r.decision == "FAIL_COST_CAP", r.decision
    assert r.attempts_used == 1, "phải dừng ngay vòng 1, chưa dùng hết lượt"
    assert r.total_cost_usd >= COST_CAP_USD


def test_giu_duoc_best_attempt():
    """Vòng cuối tệ hơn -> trả notebook/report của vòng tốt nhất, không phải vòng cuối."""

    def decreasing_score(nb_path, exc, bundle):
        score = 3.4 if exc.attempt == 1 else 2.0
        return mocks.MOCK_REPORT_RETRY.model_copy(
            update={
                "nb_path": nb_path,
                "attempt": exc.attempt,
                "llm_scores": LlmScores(
                    groundedness=score,
                    difficulty_fit=score,
                    pedagogical_order=score,
                    content_completeness=score,
                    learning_coverage=score,
                ),
            }
        )

    _wire(run_verifier=decreasing_score)
    r = m.generate(_profile("t-best"), force_mock=True)
    assert r.best_attempt_so_far is not None
    best = r.best_attempt_so_far
    assert best.average_score == max(x.average_score for x in r.retry_history)
    assert best.attempt == 1
    assert r.notebook_path == best.nb_path
    assert r.report is not None and r.report.average_score == best.average_score


def test_tie_break_uu_tien_hard_rules_pass():
    """Cùng điểm LLM -> giữ vòng qua hard rules, không giữ vòng cũ theo quán tính."""

    def tied_score_better_rules(nb_path, exc, bundle):
        checks = (
            RuleChecks(
                has_instructions=True,
                has_todo=True,
                has_assert=False,
                no_hardcoded_answers=True,
                has_train_test_split=True,
                has_visualization=True,
                has_demo_per_module=True,
                min_cells_by_level=True,
            )
            if exc.attempt == 1
            else RuleChecks(
                has_instructions=True,
                has_todo=True,
                has_assert=True,
                no_hardcoded_answers=True,
                has_train_test_split=True,
                has_visualization=True,
                has_demo_per_module=True,
                min_cells_by_level=True,
            )
        )
        return mocks.MOCK_REPORT_PASS.model_copy(
            update={
                "nb_path": nb_path,
                "attempt": exc.attempt,
                "rule_checks": checks,
                "decision": "RETRY" if exc.attempt == 1 else "PASS",
            }
        )

    _wire(run_verifier=tied_score_better_rules)
    r = m.generate(_profile("t-best-tie-rules"), force_mock=True)
    assert r.decision == "PASS"
    assert r.attempts_used == 2
    assert r.best_attempt_so_far is not None
    assert r.best_attempt_so_far.attempt == 2
    assert r.best_attempt_so_far.rules_all_passed is True
    assert r.best_attempt_so_far.execution_ok is True
    assert r.notebook_path == r.best_attempt_so_far.nb_path


def test_diem_cao_nhung_rule_fail_thi_khong_pass():
    """Điểm LLM >= 3.5 nhưng rule fail -> retry rồi FAIL_MAX_RETRY."""

    def high_score_bad_rules(nb_path, exc, bundle):
        return mocks.MOCK_REPORT_PASS.model_copy(
            update={
                "nb_path": nb_path,
                "attempt": exc.attempt,
                "rule_checks": RuleChecks(
                    has_instructions=True,
                    has_todo=True,
                    has_assert=False,
                    no_hardcoded_answers=True,
                    has_train_test_split=True,
                ),
            }
        )

    _wire(run_verifier=high_score_bad_rules)
    r = m.generate(_profile("t-rule-gate"), force_mock=True)
    assert r.decision == "FAIL_MAX_RETRY"
    assert r.attempts_used == MAX_ATTEMPT


def test_rule_moi_cua_verifier_fail_thi_main_khong_tu_pass():
    """Ba rule Sprint 2.2 là cổng cứng, không chỉ là ghi chú trong report."""

    def extended_rule_fail(nb_path, exc, bundle, **kwargs):
        return mocks.MOCK_REPORT_PASS.model_copy(
            update={
                "nb_path": nb_path,
                "attempt": exc.attempt,
                "decision": "RETRY",
                "rule_checks": RuleChecks(
                    has_instructions=True,
                    has_todo=True,
                    has_assert=True,
                    no_hardcoded_answers=True,
                    has_train_test_split=True,
                    has_visualization=False,
                    has_demo_per_module=True,
                    min_cells_by_level=True,
                ),
            }
        )

    _wire(run_verifier=extended_rule_fail)
    r = m.generate(_profile("t-extended-rule-gate"), force_mock=True)
    assert r.decision == "FAIL_MAX_RETRY"
    assert r.attempts_used == MAX_ATTEMPT
    assert r.retry_history[0].rules_total == 8
    assert r.retry_history[0].rules_passed == 7


def test_main_truyen_profile_cho_research_va_level_cho_verifier():
    """Adapter Sprint 2.2 phải nhận ngữ cảnh cá nhân hoá từ main."""
    seen = {}

    def research(topic, learner_profile=None):
        seen["profile"] = learner_profile
        return mocks.fake_run_research(topic)

    def verifier(nb_path, exc, bundle, *, level=None, retry_history=None):
        seen["level"] = level
        seen["history"] = retry_history
        return mocks.fake_run_verifier(nb_path, exc, bundle)

    _wire(run_research=research, run_verifier=verifier)
    profile = _profile("t-context-contract")
    result = m.generate(profile, force_mock=True)
    assert result.decision == "PASS"
    assert seen["profile"] == profile
    assert seen["level"] == profile.level_final
    assert seen["history"] == []


def test_diem_cao_nhung_executor_fail_thi_khong_pass():
    """Điểm và rules đều cao nhưng notebook chạy lỗi -> không được PASS."""
    import executor

    original = executor.run_notebook

    def fail_executor(nb_path, attempt=1, timeout=None):
        return mocks.MOCK_EXC_FAIL.model_copy(
            update={"nb_path": nb_path, "attempt": attempt}
        )

    _wire(run_verifier=lambda nb_path, exc, bundle: mocks.MOCK_REPORT_PASS.model_copy(
        update={"nb_path": nb_path, "attempt": exc.attempt}
    ))
    executor.run_notebook = fail_executor
    try:
        r = m.generate(_profile("t-exec-gate"), force_mock=True)
    finally:
        executor.run_notebook = original

    assert r.decision == "FAIL_MAX_RETRY"
    assert r.attempts_used == MAX_ATTEMPT


def test_fail_execution_khong_lay_report_cu():
    """Vòng 2 chết -> FAIL_EXECUTION, KHÔNG lấy nhầm report của vòng 1"""

    def crash_on_second(path, profile, attempt, feedback):
        if attempt == 2:
            raise RuntimeError("notebook_gen hỏng ở vòng 2")
        return mocks.fake_run_notebook_gen(path, profile, attempt, feedback)

    _wire(run_notebook_gen=crash_on_second, run_verifier=_always_retry)
    r = m.generate(_profile("t-crash"), force_mock=True)
    assert r.decision == "FAIL_EXECUTION", f"lấy nhầm decision vòng trước: {r.decision}"
    assert "vòng 2" in (r.error or "")
    assert r.best_attempt_so_far is not None, "vẫn phải giữ được vòng 1"


def test_research_chet_thi_dung_ngay():
    """Lỗi trước vòng lặp -> attempts_used=0, không vào loop"""

    def boom(topic):
        raise ValueError("KB rỗng")

    _wire(run_research=boom)
    r = m.generate(_profile("t-research-fail"), force_mock=True)
    assert r.decision == "FAIL_EXECUTION"
    assert r.attempts_used == 0


def test_sinh_du_3_file_san_pham():
    """Đề cương mục 1.1: notebook.ipynb + path.json + quality_report.md"""
    _wire()
    r = m.generate(_profile("t-outputs"), force_mock=True)
    out = m.OUTPUT_ROOT / r.session_id
    for name in ("notebook.ipynb", "path.json", "quality_report.md"):
        assert (out / name).is_file(), f"thiếu {name}"


if __name__ == "__main__":
    from tests._runner import run_module

    raise SystemExit(run_module(__name__))
