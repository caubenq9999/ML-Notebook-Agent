"""
tests/mocks.py - NotebookForge (Nhóm 19)
========================================
HOÀNG tạo, CẢ NHÓM DÙNG CHUNG. Mục đích: 4 người code song song ngay hôm nay,
không phải chờ nhau.

Cách dùng - lấy đúng thứ mình cần làm input, phần còn lại xài mock:

    # HỢP làm curriculum, chưa có Research Agent của Trí:
    from tests.mocks import MOCK_BUNDLE, MOCK_PROFILE
    path = run_curriculum(MOCK_BUNDLE, MOCK_PROFILE)

    # HUY làm verifier, chưa có notebook thật của Hợp:
    from tests.mocks import MOCK_BUNDLE, MOCK_EXC_OK, write_mock_notebook
    nb = write_mock_notebook()          # file .ipynb thật, mở được
    report = run_verifier(nb, MOCK_EXC_OK, MOCK_BUNDLE)

    # NAM làm UI, chưa có pipeline:
    from tests.mocks import MOCK_RESULT   # GenerationResult đầy đủ để render

Khi người kia code xong, chỉ cần đổi MOCK_X -> hàm thật. Không sửa dòng nào khác,
vì cùng schema.

Các hàm fake_* bên dưới có ĐÚNG chữ ký của 6 hàm chính -> ai muốn chạy thử
main.py end-to-end mà chưa đủ người thì monkeypatch bằng chúng.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Cho phép chạy `python tests/mocks.py` hoặc import từ thư mục gốc repo đều được.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import (  # noqa: E402
    BestAttempt,
    CellError,
    Citation,
    Constraints,
    Exercise,
    ExcRes,
    GenerationResult,
    LearnerProfile,
    LearningPath,
    LlmScores,
    Module,
    ResearchBundle,
    RuleChecks,
    Source,
    VerifierReport,
)

# ---------------------------------------------------------------------------
# 1. LearnerProfile
# ---------------------------------------------------------------------------

# dataset_seed cố định Ở ĐÂY là CỐ Ý: test phải tái lập được.
# Trong chạy thật thì bỏ trống -> schemas.new_dataset_seed() tự random.
MOCK_PROFILE = LearnerProfile(
    topic="logistic_regression",
    level_declared=1,
    level_final=1,
    quiz_score=4,
    constraints=Constraints(duration_minutes=60, num_exercises=3),
    session_id="test-001",
    dataset_seed=20250805,
)

# Học viên khá hơn - dùng để test difficulty_fit và nhánh level cao.
MOCK_PROFILE_ADVANCED = LearnerProfile(
    topic="decision_tree",
    level_declared=2,
    level_final=2,
    quiz_score=5,
    constraints={"duration_minutes": 90, "num_exercises": 5},  # dict cũng được
    session_id="test-002",
    dataset_seed=20250806,
)

# ---------------------------------------------------------------------------
# 2. ResearchBundle (output của TRÍ)
# ---------------------------------------------------------------------------

MOCK_BUNDLE = ResearchBundle(
    topic="logistic_regression",
    sources=[
        Source(
            source_id="kb_01",
            type="kb_file",
            path_or_url="kb/logistic_regression/01_core_concepts.md",
            title="Logistic Regression - khái niệm cốt lõi",
        ),
        Source(
            source_id="kb_02",
            type="kb_file",
            path_or_url="kb/logistic_regression/02_evaluation.md",
            title="Đánh giá mô hình phân loại",
        ),
    ],
    key_concepts=["sigmoid", "log loss", "decision boundary", "regularization"],
    citations=[
        Citation(
            concept="sigmoid",
            source_id="kb_01",
            quote="Hàm sigmoid ánh xạ giá trị thực về khoảng (0, 1).",
            locator="## Sigmoid",
        ),
        Citation(concept="log loss", source_id="kb_01", locator="## Hàm mất mát"),
        Citation(concept="decision boundary", source_id="kb_02"),
    ],
    prerequisites=["linear regression", "gradient descent"],
    common_pitfalls=[
        "Quên scale feature khiến gradient descent hội tụ chậm",
        "Dùng accuracy trên dữ liệu mất cân bằng",
    ],
    # KHÔNG truyền unresolved_concepts -> schema tự suy ra: ["regularization"]
    # (khái niệm này không có citation nào). Huy test groundedness với case đó.
)

# Case xấu: research không tìm được nguồn nào -> mọi khái niệm đều unresolved.
MOCK_BUNDLE_EMPTY = ResearchBundle(
    topic="kmeans",
    sources=[],
    key_concepts=["centroid", "inertia"],
    citations=[],
)

# ---------------------------------------------------------------------------
# 3. LearningPath (output của HỢP)
# ---------------------------------------------------------------------------

MOCK_PATH = LearningPath(
    topic="logistic_regression",
    level=1,
    session_id="test-001",
    modules=[
        Module(
            module_id="m1",
            title="Trực giác về sigmoid",
            objective="Giải thích được vì sao dùng sigmoid thay vì đường thẳng",
            concepts=["sigmoid"],
            estimated_minutes=20,
            planned_exercises=[
                Exercise(
                    exercise_id="m1_ex1",
                    title="Cài đặt hàm sigmoid",
                    prompt="Viết hàm sigmoid(z) bằng numpy, không dùng thư viện có sẵn.",
                    type="code",
                    difficulty=1,
                    concepts=["sigmoid"],
                    expected_check="assert abs(sigmoid(0) - 0.5) < 1e-9",
                )
            ],
            source_ids=["kb_01"],
        ),
        Module(
            module_id="m2",
            title="Huấn luyện và log loss",
            objective="Train được model trên dữ liệu đã tách train/test",
            concepts=["log loss"],
            estimated_minutes=25,
            planned_exercises=[
                Exercise(
                    exercise_id="m2_ex1",
                    title="Tách train/test rồi fit model",
                    prompt="Dùng train_test_split với test_size=0.2 rồi fit LogisticRegression.",
                    type="code",
                    difficulty=1,
                    concepts=["log loss"],
                    expected_check="assert X_train.shape[0] > X_test.shape[0]",
                )
            ],
            source_ids=["kb_01"],
        ),
        Module(
            module_id="m3",
            title="Đọc decision boundary",
            objective="Vẽ và giải thích được ranh giới quyết định",
            concepts=["decision boundary"],
            estimated_minutes=15,
            planned_exercises=[
                Exercise(
                    exercise_id="m3_ex1",
                    title="Vẽ decision boundary",
                    prompt="Vẽ ranh giới quyết định của model vừa train trên 2 feature đầu.",
                    type="analysis",
                    difficulty=2,
                    concepts=["decision boundary"],
                    expected_check="assert plt.gcf().axes",
                )
            ],
            source_ids=["kb_02"],
        ),
        Module(
            module_id="m4",
            title="Tổng kết và đánh giá mô hình",
            objective="Đọc được accuracy / confusion matrix và biết mô hình sai ở đâu",
            concepts=["log loss"],
            estimated_minutes=5,
            planned_exercises=[],  # module lý thuyết, không có bài tập
            source_ids=["kb_02"],
        ),
    ],
    notes="4 module / 65 phút / 3 bài tập - khớp constraints của MOCK_PROFILE "
    "và yêu cầu 4-6 module của đề cương.",
)

# ---------------------------------------------------------------------------
# 4. Notebook giả (file .ipynb THẬT, để executor/verifier có cái mà đọc)
# ---------------------------------------------------------------------------

_MOCK_NB = {
    "cells": [
        {
            # nbformat >= 4.5 bắt buộc mỗi cell có "id" - thiếu là nbclient cảnh báo
            "id": "md-intro",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Logistic Regression - Bài 1\n",
                "\n",
                "**Hướng dẫn:** đọc phần lý thuyết rồi hoàn thành phần TODO bên dưới.\n",
            ],
        },
        {
            "id": "code-setup",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "from sklearn.model_selection import train_test_split\n",
                "\n",
                "rng = np.random.default_rng(20250805)\n",
                "X = rng.normal(size=(100, 2))\n",
                "y = (X[:, 0] + X[:, 1] > 0).astype(int)\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.2, random_state=20250805\n",
                ")\n",
            ],
        },
        {
            "id": "code-todo",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def sigmoid(z):\n",
                "    # TODO: cài đặt hàm sigmoid\n",
                "    raise NotImplementedError\n",
                "\n",
                "\n",
                "# assert abs(sigmoid(0) - 0.5) < 1e-9\n",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def write_mock_notebook(dest: str | Path | None = None) -> str:
    """Ghi ra một file .ipynb hợp lệ, trả về đường dẫn.

    Dùng khi cần một nb_path THẬT (Huy chấm verifier, Hoàng test executor)
    mà notebook_gen của Hợp chưa xong. Mặc định ghi vào thư mục tạm.
    """
    if dest is None:
        dest = Path(tempfile.gettempdir()) / "notebookforge_mock.ipynb"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(_MOCK_NB, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(dest)


MOCK_NB_PATH = "outputs/test-001/notebook_attempt1.ipynb"

# ---------------------------------------------------------------------------
# 5. ExcRes (output của HOÀNG - executor)
# ---------------------------------------------------------------------------

MOCK_EXC_OK = ExcRes(
    nb_path=MOCK_NB_PATH,
    attempt=1,
    success=True,
    total_cells=12,
    executed_cells=12,
    errors=[],
    duration_seconds=8.4,
    timeout_hit=False,
    executed_nb_path="outputs/test-001/notebook_attempt1.executed.ipynb",
    cost_this_attempt=0.042,
)

MOCK_EXC_FAIL = ExcRes(
    nb_path=MOCK_NB_PATH,
    attempt=2,
    success=False,
    total_cells=12,
    executed_cells=7,
    errors=[
        CellError(
            cell_index=7,
            ename="NameError",
            evalue="name 'X_train' is not defined",
            traceback_tail=[
                "----> 3 model.fit(X_train, y_train)",
                "NameError: name 'X_train' is not defined",
            ],
        )
    ],
    duration_seconds=5.1,
    timeout_hit=False,
    cost_this_attempt=0.038,
)

MOCK_EXC_TIMEOUT = ExcRes(
    nb_path=MOCK_NB_PATH,
    attempt=3,
    success=False,
    total_cells=12,
    executed_cells=4,
    errors=[
        CellError(
            cell_index=4,
            ename="CellTimeoutError",
            evalue="Cell vượt quá 120s",
            traceback_tail=["TimeoutError: cell execution exceeded 120 seconds"],
        )
    ],
    duration_seconds=120.0,
    timeout_hit=True,
    cost_this_attempt=0.051,
)

# ---------------------------------------------------------------------------
# 6. VerifierReport (output của HUY)
# ---------------------------------------------------------------------------

# average_score là computed field (trung bình 5 điểm LLM) -> KHÔNG truyền tay.
# Ở đây: (4.0 + 4.0 + 3.5 + 4.0 + 4.5) / 5 = 4.0 -> PASS.
MOCK_REPORT_PASS = VerifierReport(
    nb_path=MOCK_NB_PATH,
    attempt=1,
    rule_checks=RuleChecks(
        has_instructions=True,
        has_todo=True,
        has_assert=True,
        no_hardcoded_answers=True,
        has_train_test_split=True,
    ),
    llm_scores=LlmScores(
        groundedness=4.0,
        difficulty_fit=4.0,
        pedagogical_order=3.5,
        content_completeness=4.0,
        learning_coverage=4.5,
    ),
    decision="PASS",
    feedback=None,
    notes="Notebook chạy sạch, bám nguồn kb_01/kb_02.",
)

# (3.0 + 3.5 + 3.0 + 2.5 + 2.5) / 5 = 2.9 < 3.5 -> còn RETRY.
MOCK_REPORT_RETRY = VerifierReport(
    nb_path=MOCK_NB_PATH,
    attempt=2,
    rule_checks=RuleChecks(
        has_instructions=True,
        has_todo=True,
        has_assert=False,
        no_hardcoded_answers=True,
        has_train_test_split=False,
    ),
    llm_scores=LlmScores(
        groundedness=3.0,
        difficulty_fit=3.5,
        pedagogical_order=3.0,
        content_completeness=2.5,
        learning_coverage=2.5,
    ),
    decision="RETRY",
    feedback=(
        "Cell 7 lỗi NameError vì X_train chưa được tạo - hãy thêm train_test_split "
        "trước khi fit. Mỗi bài tập cần ít nhất 1 assert để học viên tự kiểm tra."
    ),
    ungrounded_claims=[
        "Logistic regression luôn tốt hơn decision tree",  # không có nguồn nào nói vậy
    ],
)

# ---------------------------------------------------------------------------
# 7. GenerationResult (output cuối của main.generate - NAM render cái này)
# ---------------------------------------------------------------------------

MOCK_RESULT = GenerationResult(
    session_id="test-001",
    profile=MOCK_PROFILE,
    learning_path=MOCK_PATH,
    notebook_path=MOCK_NB_PATH,
    report=MOCK_REPORT_PASS,
    decision="PASS",
    attempts_used=1,
    total_cost_usd=0.042,
    best_attempt_so_far=BestAttempt(
        attempt=1,
        nb_path=MOCK_NB_PATH,
        average_score=MOCK_REPORT_PASS.average_score,
        rules_all_passed=True,
    ),
)

# Case dừng vì hết tiền: notebook cuối tệ, nhưng best_attempt_so_far giữ bản tốt hơn.
MOCK_RESULT_COST_CAP = GenerationResult(
    session_id="test-003",
    profile=MOCK_PROFILE,
    learning_path=MOCK_PATH,
    notebook_path="outputs/test-003/notebook_attempt3.ipynb",
    report=MOCK_REPORT_RETRY,
    decision="FAIL_COST_CAP",
    attempts_used=3,
    total_cost_usd=0.31,
    best_attempt_so_far=BestAttempt(
        attempt=2,
        nb_path="outputs/test-003/notebook_attempt2.ipynb",
        average_score=3.25,
        rules_all_passed=False,
    ),
)

# ---------------------------------------------------------------------------
# 8. Bản giả của 6 hàm chính - ĐÚNG chữ ký đã chốt ở Phần 3
#    Ai muốn chạy thử main.py end-to-end khi nhóm chưa code xong thì thay tạm.
# ---------------------------------------------------------------------------


def fake_run_research(topic: str) -> ResearchBundle:  # TRÍ
    return MOCK_BUNDLE.model_copy(update={"topic": topic})


def fake_run_curriculum(bundle: ResearchBundle, profile: LearnerProfile) -> LearningPath:  # HỢP
    return MOCK_PATH.model_copy(
        update={
            "topic": bundle.topic,
            "level": profile.level_final,
            "session_id": profile.session_id,
        }
    )


def fake_run_notebook_gen(  # HỢP
    path: LearningPath,
    profile: LearnerProfile,
    attempt: int,
    prior_feedback: str | None,
) -> str:
    dest = Path(tempfile.gettempdir()) / profile.session_id / f"notebook_attempt{attempt}.ipynb"
    return write_mock_notebook(dest)


def fake_run_notebook(nb_path: str) -> ExcRes:  # HOÀNG
    return MOCK_EXC_OK.model_copy(update={"nb_path": nb_path})


def fake_run_verifier(  # HUY
    nb_path: str, exc: ExcRes, bundle: ResearchBundle
) -> VerifierReport:
    base = MOCK_REPORT_PASS if exc.success else MOCK_REPORT_RETRY
    return base.model_copy(update={"nb_path": nb_path, "attempt": exc.attempt})


def fake_get_dataset_code(topic: str, seed: int) -> str:  # NAM
    return (
        "import numpy as np\n"
        "from sklearn.model_selection import train_test_split\n"
        f"rng = np.random.default_rng({seed})\n"
        "X = rng.normal(size=(200, 4))\n"
        "y = (X[:, 0] + X[:, 1] > 0).astype(int)\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        f"    X, y, test_size=0.2, random_state={seed}\n"
        ")\n"
    )


__all__ = [
    "MOCK_PROFILE",
    "MOCK_PROFILE_ADVANCED",
    "MOCK_BUNDLE",
    "MOCK_BUNDLE_EMPTY",
    "MOCK_PATH",
    "MOCK_NB_PATH",
    "MOCK_EXC_OK",
    "MOCK_EXC_FAIL",
    "MOCK_EXC_TIMEOUT",
    "MOCK_REPORT_PASS",
    "MOCK_REPORT_RETRY",
    "MOCK_RESULT",
    "MOCK_RESULT_COST_CAP",
    "write_mock_notebook",
    "fake_run_research",
    "fake_run_curriculum",
    "fake_run_notebook_gen",
    "fake_run_notebook",
    "fake_run_verifier",
    "fake_get_dataset_code",
]


if __name__ == "__main__":
    # `python tests/mocks.py` -> tự kiểm tra mọi mock dựng được và in tóm tắt.
    # Console Windows mặc định cp1252, in tiếng Việt sẽ lỗi -> ép UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print("schemas OK - tất cả mock hợp lệ\n")
    print(f"profile        : {MOCK_PROFILE.topic} | seed={MOCK_PROFILE.dataset_seed}")
    print(f"bundle         : {len(MOCK_BUNDLE.key_concepts)} khái niệm, "
          f"unresolved={MOCK_BUNDLE.unresolved_concepts}")
    print(f"path           : {len(MOCK_PATH.modules)} module, "
          f"{MOCK_PATH.total_planned_exercises} bài tập, "
          f"{MOCK_PATH.total_estimated_minutes} phút")
    print(f"exc            : success={MOCK_EXC_OK.success}, "
          f"failed_cells={MOCK_EXC_FAIL.failed_cell_count}")
    print(f"report PASS    : average_score={MOCK_REPORT_PASS.average_score} "
          f"(rules {MOCK_REPORT_PASS.rule_checks.num_passed}/5)")
    print(f"report RETRY   : average_score={MOCK_REPORT_RETRY.average_score} "
          f"(rules {MOCK_REPORT_RETRY.rule_checks.num_passed}/5)")
    print(f"result         : {MOCK_RESULT.decision}, cost=${MOCK_RESULT.total_cost_usd}")
    print(f"notebook giả   : {write_mock_notebook()}")
