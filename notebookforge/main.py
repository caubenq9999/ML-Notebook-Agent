"""
main.py - NotebookForge (Nhóm 19)
=================================
Chủ sở hữu: HOÀNG. Nơi DUY NHẤT nối 5 bước lại với nhau (Phần 2 - Hướng dẫn Sprint 2).
Không ai import code của ai, tất cả đi qua schema.

    Trí   run_research(topic)                  -> ResearchBundle
    Hợp   run_curriculum(bundle, profile)      -> LearningPath
    Hợp   run_notebook_gen(path, profile, ...) -> nb_path
    Hoàng run_notebook(nb_path)                -> ExcRes
    Huy   run_verifier(nb_path, exc, bundle)   -> VerifierReport

Vòng lặp dừng ở đúng 3 điều kiện (Schema_Brainstorm mục "giải thích cụ thể retry loop"):
    1. average_score >= 3.5              -> PASS
    2. tổng tiền >= $0.30                -> FAIL_COST_CAP
    3. attempt >= MAX_ATTEMPT (=2)       -> FAIL_MAX_RETRY

Mock phải được bật TƯỜNG MINH. Chế độ chạy thật không được âm thầm thay module
đang thiếu bằng mock, vì như vậy một pipeline chưa hoàn chỉnh vẫn có thể báo PASS.

    python main.py --topic logistic_regression --mock   # toàn bộ bằng mock
    python main.py --topic logistic_regression          # dùng code thật nếu đã có
"""

from __future__ import annotations

import argparse
import inspect
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import (  # noqa: E402
    COST_CAP_USD,
    MAX_ATTEMPT,
    PASS_THRESHOLD,
    AttemptRecord,
    BestAttempt,
    Constraints,
    ExcRes,
    GenerationResult,
    LearnerProfile,
    LearningPath,
    ResearchBundle,
    VerifierReport,
    check_path_against_constraints,
    decide_level_final,
)

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def _enable_utf8() -> None:
    """Console Windows mặc định cp1252 -> log tiếng Việt ra thành ký tự rác.

    Ép cả stdout lẫn stderr về UTF-8 ngay khi import, vì _log() ghi ra stderr
    kể cả khi chạy dưới uvicorn (api.py) chứ không riêng lúc chạy CLI.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


_enable_utf8()


def _log(msg: str) -> None:
    print(f"[main] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Ba file sản phẩm - đề cương mục 1.1 phần OUTPUT
# ---------------------------------------------------------------------------


def _write_quality_report(result: GenerationResult) -> str:
    """Sinh quality_report.md: điểm số chi tiết + nhật ký kiểm định qua các vòng."""
    p = result.profile
    lines = [
        f"# Quality Report - {p.topic}",
        "",
        f"- Session: `{result.session_id}`",
        f"- Trình độ: khai `{p.level_declared}` -> chốt `{p.level_final}` "
        f"(quiz {p.quiz_score}/5)",
        f"- Ràng buộc: {p.constraints.duration_minutes} phút, "
        f"{p.constraints.num_exercises} bài tập",
        f"- Dataset seed: `{p.dataset_seed}`",
        "",
        "## Kết luận",
        "",
        f"- **Decision: {result.decision}**",
        f"- Số vòng đã dùng: {result.attempts_used}/{MAX_ATTEMPT}",
        f"- Tổng chi phí: ${result.total_cost_usd:.4f} / trần ${COST_CAP_USD}",
        f"- Notebook trả về: `{result.notebook_path}`",
    ]
    if result.error:
        lines.append(f"- Lỗi: `{result.error}`")

    if result.report is not None:
        s = result.report.llm_scores
        r = result.report.rule_checks
        rule_rows = [
            f"| has_instructions | {'PASS' if r.has_instructions else 'FAIL'} |",
            f"| has_todo | {'PASS' if r.has_todo else 'FAIL'} |",
            f"| has_assert | {'PASS' if r.has_assert else 'FAIL'} |",
            f"| no_hardcoded_answers | {'PASS' if r.no_hardcoded_answers else 'FAIL'} |",
            f"| has_train_test_split | {'PASS' if r.has_train_test_split else 'FAIL'} |",
        ]
        if r.num_rules == 8:
            rule_rows += [
                f"| has_visualization | {'PASS' if r.has_visualization else 'FAIL'} |",
                f"| has_demo_per_module | {'PASS' if r.has_demo_per_module else 'FAIL'} |",
                f"| min_cells_by_level | {'PASS' if r.min_cells_by_level else 'FAIL'} |",
            ]
        lines += [
            "",
            "## Điểm vòng cuối",
            "",
            "| Tiêu chí LLM (1-5) | Điểm |",
            "|---|---|",
            f"| Executability | {s.executability} |",
            f"| Groundedness | {s.groundedness} |",
            f"| Difficulty-fit | {s.difficulty_fit} |",
            f"| Pedagogical-order | {s.pedagogical_order} |",
            f"| **Trung bình** | **{result.report.average_score}** "
            f"(ngưỡng {PASS_THRESHOLD}) |",
            "",
            "| Rule check (không dùng LLM) | Kết quả |",
            "|---|---|",
            *rule_rows,
            f"| **Tổng** | **{r.num_passed}/{r.num_rules}** |",
        ]
        if result.report.ungrounded_claims:
            lines += ["", "### Câu khẳng định không có nguồn", ""]
            lines += [f"- {c}" for c in result.report.ungrounded_claims]

    if result.retry_history:
        lines += [
            "",
            "## Nhật ký các vòng",
            "",
            "| Vòng | Điểm TB | Rule | Chạy được | Chi phí | Phản hồi cho vòng sau |",
            "|---|---|---|---|---|---|",
        ]
        for rec in result.retry_history:
            fb = (rec.feedback or "-").replace("\n", " ")[:80]
            lines.append(
                f"| {rec.attempt} | {rec.average_score} | "
                f"{rec.rules_passed}/{rec.rules_total} | "
                f"{'có' if rec.execution_ok else 'KHÔNG'} | "
                f"${rec.cost_this_attempt:.4f} | {fb} |"
            )

    if result.best_attempt_so_far:
        b = result.best_attempt_so_far
        lines += [
            "",
            f"Vòng tốt nhất: **#{b.attempt}** (điểm {b.average_score}) -> `{b.nb_path}`",
        ]
    return "\n".join(lines) + "\n"


def write_outputs(result: GenerationResult) -> dict[str, str]:
    """Ghi 3 file sản phẩm vào outputs/<session_id>/ theo đúng đề cương:

        notebook.ipynb      - file thực hành đã chạy và đánh giá
        path.json           - lộ trình học gồm các module có thứ tự
        quality_report.md   - điểm số chi tiết và nhật ký kiểm định qua các vòng
    """
    out_dir = OUTPUT_ROOT / result.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    if result.notebook_path and Path(result.notebook_path).is_file():
        target = out_dir / "notebook.ipynb"
        target.write_bytes(Path(result.notebook_path).read_bytes())
        written["notebook"] = str(target)

    if result.learning_path is not None:
        target = out_dir / "path.json"
        target.write_text(result.learning_path.model_dump_json(indent=2), encoding="utf-8")
        written["path"] = str(target)

    target = out_dir / "quality_report.md"
    target.write_text(_write_quality_report(result), encoding="utf-8")
    written["report"] = str(target)

    # Bản đầy đủ cho api.py / debug - không nằm trong 3 file đề cương yêu cầu.
    target = out_dir / "result.json"
    target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    written["result"] = str(target)

    return written


# ---------------------------------------------------------------------------
# Nối 5 hàm của 5 người - thiếu ai thì thay bằng mock của người đó
# ---------------------------------------------------------------------------


STEP_ALIASES = {
    "research": "run_research",
    "curriculum": "run_curriculum",
    "notebook_gen": "run_notebook_gen",
    "verifier": "run_verifier",
}


def _resolve_impls(force_mock: bool = False, mock_steps: tuple[str, ...] = ()) -> dict:
    """Trả về dict 4 hàm cần cho pipeline (executor luôn là hàng thật của Hoàng).

    - force_mock=True        -> mock toàn bộ, không gọi LLM, không tốn quota
    - mock_steps=("verifier",) -> ÉP mock riêng bước đó, các bước khác chạy thật.
      Dùng khi một người chưa xong mà vẫn muốn đo end-to-end phần còn lại.
    - Module thật nào chưa tồn tại -> FAIL rõ ràng; muốn tạm thay phải ghi tên
      trong mock_steps hoặc dùng cờ CLI `--mock <bước>`.
    """
    from tests import mocks

    impls = {
        "run_research": mocks.fake_run_research,
        "run_curriculum": mocks.fake_run_curriculum,
        "run_notebook_gen": mocks.fake_run_notebook_gen,
        "run_verifier": mocks.fake_run_verifier,
    }
    if force_mock:
        _log("chế độ --mock: dùng toàn bộ hàm giả")
        return impls

    valid_mock_names = set(STEP_ALIASES) | set(STEP_ALIASES.values())
    unknown = sorted(set(mock_steps) - valid_mock_names)
    if unknown:
        raise ValueError(
            f"Tên bước mock không hợp lệ: {unknown}. Chọn: {list(STEP_ALIASES)}"
        )

    forced = {STEP_ALIASES.get(s, s) for s in mock_steps}
    wiring = [
        ("run_research", "agents.research", "TRÍ"),
        ("run_curriculum", "agents.curriculum", "HỢP"),
        ("run_notebook_gen", "agents.notebook_gen", "HỢP"),
        ("run_verifier", "agents.verifier", "HUY"),
    ]
    missing: list[str] = []
    for func_name, module_path, owner in wiring:
        if func_name in forced:
            _log(f"ÉP MOCK: {func_name} ({owner}) - bỏ qua bản thật theo yêu cầu")
            continue
        try:
            module = __import__(module_path, fromlist=[func_name])
            impls[func_name] = getattr(module, func_name)
            _log(f"dùng bản thật: {module_path}.{func_name}")
        except (ImportError, AttributeError) as exc:
            missing.append(f"{module_path}.{func_name} ({owner}): {type(exc).__name__}")

    if missing:
        steps = ", ".join(
            name for name, func in STEP_ALIASES.items()
            if any(func in item for item in missing)
        )
        raise RuntimeError(
            "Thiếu implementation thật: " + "; ".join(missing) + ". "
            f"Muốn chạy tích hợp tạm thời, dùng --mock {steps}"
        )
    return impls


def _accepts_keyword(func, name: str) -> bool:
    """Kiểm tra adapter có nhận keyword mới mà không gọi thử hai lần.

    Không bắt ``TypeError`` để fallback vì TypeError có thể phát sinh bên
    trong implementation; nuốt lỗi đó sẽ làm debug sai nguyên nhân.
    """
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD or p.name == name
        for p in params
    )


def _run_research_compatible(func, profile: LearnerProfile) -> ResearchBundle:
    """Gọi Research Sprint 2.2 có profile, vẫn chạy được mock Sprint 2.1."""
    kwargs = {}
    if _accepts_keyword(func, "learner_profile"):
        kwargs["learner_profile"] = profile
    bundle = func(profile.topic, **kwargs)
    if not isinstance(bundle, ResearchBundle):
        raise TypeError(
            f"run_research phải trả ResearchBundle, nhận {type(bundle).__name__}"
        )
    return bundle


def _run_verifier_compatible(
    func,
    nb_path: str,
    exc_res: ExcRes,
    bundle: ResearchBundle,
    profile: LearnerProfile,
    history: list[AttemptRecord],
) -> VerifierReport:
    """Truyền level/history cho Verifier mới mà không phá mock cũ."""
    kwargs = {}
    if _accepts_keyword(func, "level"):
        kwargs["level"] = profile.level_final
    if _accepts_keyword(func, "retry_history"):
        kwargs["retry_history"] = [item.model_dump(mode="json") for item in history]
    if _accepts_keyword(func, "session_id"):
        kwargs["session_id"] = profile.session_id
    report = func(nb_path, exc_res, bundle, **kwargs)
    if not isinstance(report, VerifierReport):
        raise TypeError(
            f"run_verifier phải trả VerifierReport, nhận {type(report).__name__}"
        )
    return report


# ---------------------------------------------------------------------------
# Vòng lặp chính
# ---------------------------------------------------------------------------


def generate(
    profile: LearnerProfile,
    *,
    force_mock: bool = False,
    mock_steps: tuple[str, ...] = (),
) -> GenerationResult:
    """Sinh notebook cho một học viên. Không raise: lỗi được gói vào GenerationResult."""
    from executor import run_notebook
    from llm_client import get_tracker, reset_tracker

    reset_tracker(profile.session_id)
    tracker = get_tracker(profile.session_id)

    result = GenerationResult(
        session_id=profile.session_id,
        profile=profile,
        decision="RETRY",
    )

    try:
        impls = _resolve_impls(force_mock, mock_steps)
    except Exception as exc:  # noqa: BLE001 - lỗi wiring phải thành kết quả audit được
        _log(f"không thể dựng pipeline: {exc}")
        result.decision = "FAIL_EXECUTION"
        result.error = f"{type(exc).__name__}: {exc}"
        write_outputs(result)
        return result

    # --- Bước 1 + 2: research -> curriculum (chạy 1 lần, ngoài vòng lặp) ---
    try:
        _log(f"[1/5] research: {profile.topic}")
        bundle = _run_research_compatible(impls["run_research"], profile)
        if bundle.unresolved_concepts:
            _log(f"      khái niệm chưa có nguồn: {bundle.unresolved_concepts}")

        _log("[2/5] curriculum")
        path = impls["run_curriculum"](bundle, profile)
        if not isinstance(path, LearningPath):
            raise TypeError(
                f"run_curriculum phải trả LearningPath, nhận {type(path).__name__}"
            )
        result.learning_path = path

        for warning in check_path_against_constraints(path, profile):
            _log(f"      cảnh báo constraints: {warning}")
    except Exception as exc:  # noqa: BLE001
        _log(f"pipeline chết ở bước research/curriculum: {exc}")
        traceback.print_exc()
        result.decision = "FAIL_EXECUTION"
        result.error = f"{type(exc).__name__}: {exc}"
        result.total_cost_usd = round(tracker.total_usd, 6)
        return result

    # --- Bước 3-5: sinh -> chạy -> chấm -> sửa ---
    attempt = 1
    feedback: str | None = None
    nb_path: str | None = None
    report = None
    best_report = None

    while True:
        mark = tracker.mark()  # mốc tiền đầu attempt
        _log(f"--- attempt {attempt}/{MAX_ATTEMPT} ---")

        try:
            _log("[3/5] notebook_gen")
            nb_path = impls["run_notebook_gen"](path, profile, attempt, feedback)
            if not isinstance(nb_path, str) or not nb_path.strip():
                raise TypeError(
                    f"run_notebook_gen phải trả path chuỗi không rỗng, "
                    f"nhận {type(nb_path).__name__}"
                )

            _log("[4/5] executor")
            exc_res: ExcRes = run_notebook(nb_path, attempt=attempt)

            # executor không gọi LLM: chi phí thật của attempt nằm ở tracker.
            exc_res = exc_res.model_copy(
                update={"cost_this_attempt": tracker.cost_since(mark)}
            )
            _log(
                f"      chạy {exc_res.executed_cells}/{exc_res.total_cells} cell, "
                f"{exc_res.failed_cell_count} lỗi, ${exc_res.cost_this_attempt:.4f}"
            )

            _log("[5/5] verifier")
            report = _run_verifier_compatible(
                impls["run_verifier"],
                nb_path,
                exc_res,
                bundle,
                profile,
                result.retry_history,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"attempt {attempt} chết: {exc}")
            traceback.print_exc()
            result.decision = "FAIL_EXECUTION"
            result.error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            break

        cost_this_attempt = tracker.cost_since(mark)
        _log(
            f"      average_score={report.average_score} "
            f"(rules {report.rule_checks.num_passed}/{report.rule_checks.num_rules}), "
            f"tổng tiền ${tracker.total_usd:.4f}"
        )

        # Audit trail - Huy dùng cho quality_report.md
        result.retry_history.append(
            AttemptRecord(
                attempt=attempt,
                nb_path=nb_path,
                average_score=report.average_score,
                rules_passed=report.rule_checks.num_passed,
                rules_total=report.rule_checks.num_rules,
                execution_ok=exc_res.success,
                cost_this_attempt=cost_this_attempt,
                feedback=report.feedback,
            )
        )

        # Giữ bản tốt nhất - vòng cuối chưa chắc đã là vòng ngon nhất.
        # Khi điểm LLM hòa, ưu tiên bản qua hard rules, rồi execution sạch;
        # nếu vẫn hòa thì lấy vòng mới hơn vì đó là bản đã nhận feedback sửa lỗi.
        candidate_rank = (
            report.average_score,
            report.rule_checks.all_passed,
            exc_res.success,
            attempt,
        )
        best = result.best_attempt_so_far
        best_rank = (
            best.average_score,
            best.rules_all_passed,
            best.execution_ok,
            best.attempt,
        ) if best is not None else None
        if best_rank is None or candidate_rank > best_rank:
            result.best_attempt_so_far = BestAttempt(
                attempt=attempt,
                nb_path=nb_path,
                average_score=report.average_score,
                rules_all_passed=report.rule_checks.all_passed,
                execution_ok=exc_res.success,
            )
            best_report = report

        # --- 3 điều kiện dừng, đúng thứ tự trong tài liệu ---
        # Điểm LLM chỉ là một phần của cổng chất lượng. Notebook chỉ được PASS
        # khi code chạy sạch VÀ cả 5 rule check đều đạt; nếu không, một Judge
        # chấm rộng tay có thể cho qua notebook đang crash hoặc thiếu TODO/assert.
        if (
            exc_res.success
            and report.rule_checks.all_passed
            and report.average_score >= PASS_THRESHOLD
            and report.decision == "PASS"
        ):
            report.decision = "PASS"
            break
        if tracker.total_usd >= COST_CAP_USD:
            report.decision = "FAIL_COST_CAP"
            _log(f"chạm trần chi phí ${COST_CAP_USD}")
            break
        if attempt >= MAX_ATTEMPT:
            report.decision = "FAIL_MAX_RETRY"
            break

        feedback = report.feedback
        attempt += 1

    # --- Gói kết quả ---
    result.attempts_used = attempt
    result.total_cost_usd = round(tracker.total_usd, 6)
    # Chỉ lấy decision từ report khi attempt cuối chạy trót lọt. Nếu vòng này chết
    # giữa chừng thì report còn là của vòng TRƯỚC - lấy vào là báo cáo sai.
    if result.error is None and report is not None:
        result.decision = report.decision

    # Nếu pipeline không PASS, trả notebook tốt nhất đã thấy thay vì mặc định
    # lấy attempt cuối (attempt cuối có thể tệ hơn hoặc chết giữa chừng).
    selected_path = nb_path
    selected_report = report
    if result.decision != "PASS" and result.best_attempt_so_far is not None:
        selected_path = result.best_attempt_so_far.nb_path
        selected_report = best_report

    result.notebook_path = selected_path
    if selected_report is not None:
        # Decision là kết luận của cả session, kể cả report được chọn đến từ
        # một attempt trước đó.
        result.report = selected_report.model_copy(update={"decision": result.decision})

    written = write_outputs(result)
    _log(
        f"XONG: {result.decision} sau {result.attempts_used} attempt, "
        f"${result.total_cost_usd:.4f} -> {Path(written['report']).parent}"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sinh notebook học ML cho 1 học viên")
    parser.add_argument("--topic", default="logistic_regression")
    parser.add_argument(
        "--level", type=int, default=1, choices=[1, 2],
        help="Trình độ tự khai: 1=beginner, 2=intermediate",
    )
    parser.add_argument(
        "--quiz-score", type=int, default=3, choices=[0, 1, 2, 3, 4, 5],
        help="Số câu đúng trên 5 câu quiz",
    )
    parser.add_argument("--duration", type=int, default=60, help="Số phút (60-120)")
    parser.add_argument("--exercises", type=int, default=3, help="Số bài tập (3-5)")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Cố định seed để tái lập")
    parser.add_argument(
        "--mock", nargs="*", default=None, metavar="BƯỚC",
        help="Không kèm tên = mock toàn bộ. Kèm tên = chỉ mock bước đó, "
             "vd: --mock verifier (chọn: research, curriculum, notebook_gen, verifier)",
    )
    args = parser.parse_args(argv)

    # --mock         -> [] -> mock hết
    # --mock verifier-> ['verifier'] -> chỉ mock bước đó, còn lại chạy thật
    # không có cờ    -> None -> chạy thật hết
    force_mock = args.mock is not None and len(args.mock) == 0
    mock_steps = tuple(args.mock or ())

    import uuid

    session_id = args.session_id or f"cli-{uuid.uuid4().hex[:8]}"
    profile_kwargs = dict(
        topic=args.topic,
        level_declared=args.level,
        level_final=decide_level_final(args.level, args.quiz_score),
        quiz_score=args.quiz_score,
        constraints=Constraints(duration_minutes=args.duration, num_exercises=args.exercises),
        session_id=session_id,
    )
    if args.seed is not None:
        profile_kwargs["dataset_seed"] = args.seed

    profile = LearnerProfile(**profile_kwargs)
    result = generate(profile, force_mock=force_mock, mock_steps=mock_steps)

    print()
    print(f"decision      : {result.decision}")
    print(f"attempts_used : {result.attempts_used}/{MAX_ATTEMPT}")
    print(f"total_cost    : ${result.total_cost_usd:.4f} / ${COST_CAP_USD}")
    print(f"notebook      : {result.notebook_path}")
    if result.best_attempt_so_far:
        best = result.best_attempt_so_far
        print(f"best_attempt  : #{best.attempt} - score {best.average_score}")
    if result.retry_history:
        print("lịch sử       :")
        for rec in result.retry_history:
            print(
                f"  #{rec.attempt} score={rec.average_score} "
                f"rules={rec.rules_passed}/{rec.rules_total} exec_ok={rec.execution_ok} "
                f"${rec.cost_this_attempt:.4f}"
            )
    if result.error:
        print(f"lỗi           : {result.error}")

    print(f"sản phẩm      : {OUTPUT_ROOT / session_id}")
    for name in ("notebook.ipynb", "path.json", "quality_report.md"):
        mark = "có" if (OUTPUT_ROOT / session_id / name).is_file() else "THIẾU"
        print(f"  {name:<20} {mark}")

    return 0 if result.decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
