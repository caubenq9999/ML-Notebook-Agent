"""
schemas.py - NotebookForge (Nhóm 19) - BẢN CHÍNH THỨC SPRINT 2
==============================================================
Chủ sở hữu: HOÀNG. Đây là single source of truth cho toàn bộ contract giữa 5 người.
Cần thêm/đổi field -> NHẮN HOÀNG, không tự sửa file này.

Quy tắc nền tảng: mỗi người viết 1 hàm, nhận 1 schema, trả 1 schema.
6 chữ ký hàm đã chốt (Phần 3 - Hướng dẫn kỹ thuật Sprint 2):

    run_research(topic: str) -> ResearchBundle                                  # TRÍ
    run_curriculum(bundle: ResearchBundle, profile: LearnerProfile)
        -> LearningPath                                                         # HỢP
    run_notebook_gen(path: LearningPath, profile: LearnerProfile,
                     attempt: int, prior_feedback: str | None) -> str           # HỢP
    run_notebook(nb_path: str) -> ExcRes                                        # HOÀNG
    run_verifier(nb_path: str, exc: ExcRes, bundle: ResearchBundle)
        -> VerifierReport                                                       # HUY
    get_dataset_code(topic: str, seed: int) -> str                              # NAM

5 lỗi phát hiện ở Sprint 1 đã được sửa trong bản này:
  (1) seed dataset KHÔNG còn cố định -> LearnerProfile.dataset_seed random mỗi session.
  (2) enum decision đã có FAIL_COST_CAP.
  (3) ResearchBundle có unresolved_concepts (khái niệm không tìm được nguồn).
  (4) Module có planned_exercises (bài tập dự kiến của TỪNG module).
  (5) GenerationResult có best_attempt_so_far (giữ lần sinh tốt nhất khi loop fail).

Ghi chú config: mọi model đều extra="forbid" -> truyền sai tên field là báo lỗi ngay,
không âm thầm nuốt. validate_assignment=True -> `report.decision = "PASS"` cũng được
validate đúng enum.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

# ---------------------------------------------------------------------------
# Hằng số toàn hệ thống (main.py đọc từ đây, không hardcode chỗ khác)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "2.0.0"

# [CHỐT TRONG BUỔI HỌP] brainstorm ghi 2, đề cương ghi 3 -> đang lấy 3 theo đề cương.
# Đổi ý thì sửa đúng 1 dòng này, không ai phải sửa code.
MAX_ATTEMPT = 3

COST_CAP_USD = 0.30      # tổng chi phí 1 session, chạm ngưỡng -> FAIL_COST_CAP
PASS_THRESHOLD = 3.5     # average_score >= 3.5 -> PASS

# ---------------------------------------------------------------------------
# Kiểu dùng chung
# ---------------------------------------------------------------------------

Level = Literal[1, 2, 3]                       # 1=beginner, 2=intermediate, 3=advanced
SourceType = Literal["kb_file", "web", "paper", "textbook"]
ExerciseType = Literal["code", "concept", "analysis"]

# RETRY = còn trong vòng lặp, chưa kết luận. 4 giá trị còn lại là trạng thái dừng.
Decision = Literal[
    "RETRY",
    "PASS",
    "FAIL_COST_CAP",
    "FAIL_MAX_RETRY",
    "FAIL_EXECUTION",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_dataset_seed() -> int:
    """Seed random cho dataset (LỖI SỐ 1 ở Sprint 1: trước đây seed bị fix cứng).

    Mỗi session một seed khác nhau -> hai học viên cùng topic không nhận
    y hệt một bộ số. Seed vẫn được LƯU trong LearnerProfile nên vẫn tái tạo
    lại được notebook cũ khi cần debug.
    """
    return random.randint(1, 2**31 - 1)


class ForgeModel(BaseModel):
    """Base chung. Mọi schema trong file này kế thừa để có cùng hành vi."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data):
        """Cho phép round-trip: model_validate(obj.model_dump()).

        model_dump() có kèm các computed field (average_score, ...); nếu không
        bỏ đi thì extra="forbid" sẽ báo lỗi khi nạp lại. Cần thiết cho api.py
        (FastAPI) và ui/streamlit_app.py khi truyền JSON qua lại.
        """
        computed = getattr(cls, "model_computed_fields", None)
        if isinstance(data, dict) and computed:
            if any(key in data for key in computed):
                data = {k: v for k, v in data.items() if k not in computed}
        return data


# ===========================================================================
# 1. LearnerProfile - input của cả pipeline (NAM sinh từ form Streamlit)
# ===========================================================================


class Constraints(ForgeModel):
    """Ràng buộc học viên đặt ra. Nhận được cả dict, Pydantic tự ép kiểu."""

    duration_minutes: int = Field(60, ge=15, le=240)
    num_exercises: int = Field(3, ge=1, le=10)
    language: Literal["vi", "en"] = "vi"


class LearnerProfile(ForgeModel):
    """Hồ sơ người học sau khi làm quiz phân loại."""

    topic: str = Field(..., min_length=1, description="vd: logistic_regression")
    level_declared: Level = Field(..., description="Trình độ học viên TỰ khai")
    level_final: Level = Field(..., description="Trình độ chốt sau quiz")
    quiz_score: int = Field(..., ge=0, le=5)
    constraints: Constraints = Field(default_factory=Constraints)
    session_id: str = Field(..., min_length=1)

    # LỖI 1: seed phải random, nhưng vẫn lưu lại để tái tạo được.
    dataset_seed: int = Field(default_factory=new_dataset_seed, ge=0)

    created_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = SCHEMA_VERSION


# Bản hướng dẫn Sprint 2 gõ nhầm thành "LeanerProfile" (thiếu chữ r).
# Tên đúng từ nay là LearnerProfile; giữ alias để code ai lỡ gõ theo doc vẫn chạy.
LeanerProfile = LearnerProfile


# ===========================================================================
# 2. ResearchBundle - output của TRÍ
# ===========================================================================


class Source(ForgeModel):
    """Một nguồn tài liệu. source_id là khoá để Citation trỏ tới."""

    source_id: str = Field(..., min_length=1, description="vd: kb_01")
    type: SourceType
    path_or_url: str = Field(..., min_length=1)
    title: str | None = None
    retrieved_at: datetime = Field(default_factory=_utcnow)


class Citation(ForgeModel):
    """Gắn 1 khái niệm với 1 nguồn. Huy dùng field này để chấm groundedness."""

    concept: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    quote: str | None = Field(None, description="Trích dẫn nguyên văn, nếu có")
    locator: str | None = Field(None, description="Heading / dòng trong file nguồn")


class ResearchBundle(ForgeModel):
    """Kết quả research: khái niệm + nguồn dẫn chứng cho 1 topic."""

    topic: str = Field(..., min_length=1)
    sources: list[Source] = Field(default_factory=list)
    key_concepts: list[str] = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)

    # LỖI 3: khái niệm nêu ra nhưng KHÔNG có nguồn nào chống lưng.
    # Tự động suy ra nếu Trí không truyền vào. Huy dùng để trừ điểm groundedness,
    # Hợp dùng để tránh đưa khái niệm "trôi nổi" vào notebook.
    unresolved_concepts: list[str] = Field(default_factory=list)

    prerequisites: list[str] = Field(default_factory=list)
    common_pitfalls: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="before")
    @classmethod
    def _derive_unresolved(cls, data):
        if isinstance(data, dict) and not data.get("unresolved_concepts"):
            cited = set()
            for c in data.get("citations") or []:
                cited.add(c["concept"] if isinstance(c, dict) else c.concept)
            data = {
                **data,
                "unresolved_concepts": [
                    k for k in (data.get("key_concepts") or []) if k not in cited
                ],
            }
        return data

    @model_validator(mode="after")
    def _citations_point_to_real_sources(self) -> "ResearchBundle":
        known = {s.source_id for s in self.sources}
        dangling = sorted({c.source_id for c in self.citations} - known)
        if dangling:
            raise ValueError(
                f"citations trỏ tới source_id không tồn tại trong sources: {dangling}"
            )
        return self

    @computed_field
    @property
    def grounded_concepts(self) -> list[str]:
        """key_concepts có ít nhất 1 citation."""
        cited = {c.concept for c in self.citations}
        return [k for k in self.key_concepts if k in cited]


# ===========================================================================
# 3. LearningPath - output của HỢP (Curriculum Agent)
# ===========================================================================


class Exercise(ForgeModel):
    """Một bài tập dự kiến. Đây là BẢN KẾ HOẠCH, code thật do notebook_gen sinh."""

    exercise_id: str = Field(..., min_length=1, description="vd: m1_ex1")
    title: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, description="Đề bài cho học viên")
    type: ExerciseType = "code"
    difficulty: Level = 1
    concepts: list[str] = Field(default_factory=list)
    has_starter_code: bool = True
    expected_check: str | None = Field(
        None, description="Mô tả assert cần có - Huy check has_assert dựa vào đây"
    )


class Module(ForgeModel):
    """Một module (một chặng) trong lộ trình."""

    module_id: str = Field(..., min_length=1, description="vd: m1")
    title: str = Field(..., min_length=1)
    objective: str = Field(..., min_length=1, description="Học xong module này làm được gì")
    concepts: list[str] = Field(default_factory=list)
    estimated_minutes: int = Field(..., ge=1, le=240)

    # LỖI 4: trước đây bài tập chỉ đếm tổng ở cấp path -> không biết bài nào
    # thuộc module nào. Giờ mỗi module tự khai bài tập của mình.
    planned_exercises: list[Exercise] = Field(default_factory=list)

    source_ids: list[str] = Field(
        default_factory=list, description="Nguồn (ResearchBundle.sources) module này dựa vào"
    )


class LearningPath(ForgeModel):
    """Lộ trình học đã cá nhân hoá theo level_final của học viên."""

    topic: str = Field(..., min_length=1)
    level: Level
    session_id: str = Field(..., min_length=1)
    modules: list[Module] = Field(..., min_length=1)
    notes: str | None = None
    generated_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = SCHEMA_VERSION

    @computed_field
    @property
    def total_estimated_minutes(self) -> int:
        return sum(m.estimated_minutes for m in self.modules)

    @computed_field
    @property
    def total_planned_exercises(self) -> int:
        return sum(len(m.planned_exercises) for m in self.modules)


# ===========================================================================
# 4. ExcRes - output của HOÀNG (executor.py, nbclient sandbox)
# ===========================================================================


class CellError(ForgeModel):
    """Một cell chạy lỗi. Huy đọc field này để viết feedback cho lần retry."""

    cell_index: int = Field(..., ge=0)
    ename: str = Field(..., min_length=1, description="vd: NameError")
    evalue: str = ""
    traceback_tail: list[str] = Field(
        default_factory=list, description="Vài dòng cuối traceback, đã cắt bớt"
    )


class ExcRes(ForgeModel):
    """Kết quả chạy notebook trong sandbox."""

    nb_path: str = Field(..., min_length=1)
    attempt: int = Field(1, ge=1)
    success: bool
    total_cells: int = Field(..., ge=0)
    executed_cells: int = Field(..., ge=0)
    errors: list[CellError] = Field(default_factory=list)
    duration_seconds: float = Field(..., ge=0)
    timeout_hit: bool = False
    executed_nb_path: str | None = Field(
        None, description="Notebook đã có output, để Huy chấm executability"
    )

    # Chi phí LLM của LƯỢT này. executor không gọi LLM nên mặc định 0.0;
    # main.py điền lại từ llm_client trước khi cộng dồn vào cost cap.
    cost_this_attempt: float = Field(0.0, ge=0)

    executed_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _consistent(self) -> "ExcRes":
        if self.executed_cells > self.total_cells:
            raise ValueError("executed_cells không được lớn hơn total_cells")
        if self.errors and self.success:
            raise ValueError("có errors thì success phải là False")
        return self

    @computed_field
    @property
    def failed_cell_count(self) -> int:
        return len(self.errors)


ExecutionResult = ExcRes  # alias dễ đọc, dùng tên nào cũng được


# ===========================================================================
# 5. VerifierReport - output của HUY
# ===========================================================================


class RuleChecks(ForgeModel):
    """5 tiêu chí kiểm tra bằng luật (không tốn tiền LLM).

    [CHỐT TRONG BUỔI HỌP] đang lấy đúng 5 đề xuất trong tài liệu Sprint 2.
    Thêm/bớt tiêu chí -> Huy nhắn Hoàng, vì đổi ở đây là đổi contract.
    """

    has_instructions: bool = Field(..., description="Có markdown hướng dẫn trước mỗi bài")
    has_todo: bool = Field(..., description="Có chỗ TODO cho học viên tự làm")
    has_assert: bool = Field(..., description="Có assert để học viên tự kiểm tra")
    no_hardcoded_answers: bool = Field(..., description="Không lộ đáp án sẵn trong code")
    has_train_test_split: bool = Field(..., description="Có tách train/test")

    @computed_field
    @property
    def num_passed(self) -> int:
        return sum(
            [
                self.has_instructions,
                self.has_todo,
                self.has_assert,
                self.no_hardcoded_answers,
                self.has_train_test_split,
            ]
        )

    @computed_field
    @property
    def all_passed(self) -> bool:
        return self.num_passed == 5


class LlmScores(ForgeModel):
    """4 tiêu chí chấm bằng LLM, thang 1-5.

    [CHỐT TRONG BUỔI HỌP] đang lấy đúng 4 đề xuất trong tài liệu Sprint 2.
    """

    executability: float = Field(..., ge=1, le=5, description="Chạy trơn tru không")
    groundedness: float = Field(..., ge=1, le=5, description="Bám nguồn trong ResearchBundle")
    difficulty_fit: float = Field(..., ge=1, le=5, description="Vừa với level_final")
    pedagogical_order: float = Field(..., ge=1, le=5, description="Thứ tự dạy hợp lý")

    @computed_field
    @property
    def average(self) -> float:
        return round(
            (
                self.executability
                + self.groundedness
                + self.difficulty_fit
                + self.pedagogical_order
            )
            / 4,
            3,
        )


class VerifierReport(ForgeModel):
    """Phiếu chấm 1 lần sinh notebook."""

    nb_path: str = Field(..., min_length=1)
    attempt: int = Field(1, ge=1)
    rule_checks: RuleChecks
    llm_scores: LlmScores
    decision: Decision = Field(
        "RETRY", description="Huy để mặc định RETRY, main.py mới là nơi chốt kết luận"
    )
    feedback: str | None = Field(
        None, description="Cần sửa gì cho lần sau - main.py truyền lại cho notebook_gen"
    )
    ungrounded_claims: list[str] = Field(
        default_factory=list, description="Câu khẳng định trong notebook không có nguồn"
    )
    notes: str | None = None
    verified_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = SCHEMA_VERSION

    @computed_field
    @property
    def average_score(self) -> float:
        """Trung bình 4 điểm LLM. CHỈ MỘT công thức duy nhất, không ai tính lại tay.

        rule_checks không tính vào đây - chúng là cổng chặn cứng, thể hiện qua
        feedback. Muốn đổi cách tính -> nhắn Hoàng.
        """
        return self.llm_scores.average

    @computed_field
    @property
    def is_passing(self) -> bool:
        return self.average_score >= PASS_THRESHOLD


# ===========================================================================
# 6. GenerationResult - output cuối của main.generate() (HOÀNG)
# ===========================================================================


class BestAttempt(ForgeModel):
    """LỖI 5: giữ lại lần sinh tốt nhất.

    Loop có thể dừng vì hết tiền hoặc hết lượt trong khi attempt số 2 lại tốt hơn
    attempt cuối. Không có field này thì trả về đúng bản cuối cùng - phí.
    """

    attempt: int = Field(..., ge=1)
    nb_path: str = Field(..., min_length=1)
    average_score: float = Field(..., ge=1, le=5)
    rules_all_passed: bool = False


class GenerationResult(ForgeModel):
    """Trả cho api.py / streamlit_app.py. Gói trọn 1 session."""

    session_id: str = Field(..., min_length=1)
    profile: LearnerProfile
    learning_path: LearningPath | None = None
    notebook_path: str | None = None
    report: VerifierReport | None = None
    decision: Decision = "RETRY"
    attempts_used: int = Field(0, ge=0)
    total_cost_usd: float = Field(0.0, ge=0)
    best_attempt_so_far: BestAttempt | None = None
    error: str | None = Field(None, description="Thông báo lỗi nếu pipeline chết giữa chừng")
    finished_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = SCHEMA_VERSION


__all__ = [
    # hằng số
    "SCHEMA_VERSION",
    "MAX_ATTEMPT",
    "COST_CAP_USD",
    "PASS_THRESHOLD",
    # kiểu
    "Level",
    "SourceType",
    "ExerciseType",
    "Decision",
    # helper
    "new_dataset_seed",
    "ForgeModel",
    # schema
    "Constraints",
    "LearnerProfile",
    "LeanerProfile",
    "Source",
    "Citation",
    "ResearchBundle",
    "Exercise",
    "Module",
    "LearningPath",
    "CellError",
    "ExcRes",
    "ExecutionResult",
    "RuleChecks",
    "LlmScores",
    "VerifierReport",
    "BestAttempt",
    "GenerationResult",
]
