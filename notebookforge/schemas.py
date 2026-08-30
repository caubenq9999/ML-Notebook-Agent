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

# Số lần SINH notebook tối đa trong 1 session.
# Đề cương ghi "tối đa 2" ở 4 chỗ (mục 2.1, 2.3, 2.5 cost guard, kế hoạch tuần 3),
# không chỗ nào ghi 3. Ví dụ tính chi phí của đề cương cũng là "2 vòng x $0.12 = $0.24".
# (Bản Hướng dẫn Sprint 2 ghi "đề cương ghi 3" -> ghi nhầm, mình đã đối chiếu lại.)
MAX_ATTEMPT = 2

COST_CAP_USD = 0.30      # tổng chi phí 1 session, chạm ngưỡng -> FAIL_COST_CAP
PASS_THRESHOLD = 3.5     # average_score >= 3.5 -> PASS

# ---------------------------------------------------------------------------
# Kiểu dùng chung
# ---------------------------------------------------------------------------

# CHỐT NGÀY 10/8: đúng 2 bậc, viết bằng SỐ.
#
# Vì sao 2 bậc: đề cương mục 1.1 chỉ có radio button "Beginner / Intermediate".
# Bậc 3 (advanced) ở bản schema đầu là mình tự thêm, không ai dùng tới - Huy ghi
# thẳng trong golden_set.py là "không dùng tới 3 (advanced)", UI của Nam cũng chỉ
# có 2 lựa chọn.
#
# Vì sao dùng số chứ không phải chuỗi: Huy (golden_set.py) và Nam (streamlit_app.py)
# đều đã viết theo số 1/2. Đề cương mục 1.1 ghi JSON mẫu là chuỗi ("intermediate"),
# nên chỗ này LỆCH ĐỀ CƯƠNG có chủ đích - phải giải trình trong báo cáo cuối kỳ.
Level = Literal[1, 2]  # 1 = beginner, 2 = intermediate
SourceType = Literal["kb_file", "web", "web_search", "paper", "textbook"]
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

    # Khoảng giá trị lấy đúng theo thanh trượt Streamlit trong đề cương mục 1.1:
    # thời lượng 60-120 phút, số bài tập 3-5 câu.
    duration_minutes: int = Field(60, ge=60, le=120)
    num_exercises: int = Field(3, ge=3, le=5)
    language: Literal["vi", "en"] = "vi"


class LearnerProfile(ForgeModel):
    """Hồ sơ người học sau khi làm quiz phân loại."""

    topic: str = Field(..., min_length=1, description="vd: logistic_regression")
    level_declared: Level = Field(..., description="Trình độ học viên TỰ khai")
    level_final: Level = Field(..., description="Trình độ chốt sau quiz")
    # Quiz 5 câu trắc nghiệm -> số câu đúng là 0-5 (có thể sai hết).
    # Quy tắc hạ level của đề cương: intermediate mà đúng dưới 3/5 -> hạ beginner.
    quiz_score: int = Field(..., ge=0, le=5, description="Số câu đúng trên 5 câu quiz")
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


class TheoryChunk(ForgeModel):
    """1 đoạn kiến thức đã được SEMANTIC CHUNKING gom lại (các đoạn văn liên quan về mặt
    ngữ nghĩa được gộp chung), rồi gắn với các key_concepts liên quan nhất bằng embedding
    similarity. Đây là input RAG cho Curriculum/Notebook Gen để 2 agent đó không phải tự
    "nhớ lại" lý thuyết bằng kiến thức nền của LLM mà bám đúng văn bản KB thật.
    """

    chunk_id: str = Field(..., min_length=1, description="vd: logreg_01_c0")
    source_id: str = Field(..., min_length=1, description="KBEntry.source_id chứa đoạn này")
    concepts: list[str] = Field(
        default_factory=list,
        description="Các key_concepts được coi là thuộc đoạn này (1 chunk có thể phục vụ nhiều concept)",
    )
    text: str = Field(..., min_length=1, description="Nội dung đoạn kiến thức")
    similarity: float | None = Field(
        None, description="Điểm cosine similarity cao nhất đạt được với 1 trong các concepts trên (debug/tune threshold)"
    )


class ResearchBundle(ForgeModel):
    """Kết quả research: khái niệm + nguồn dẫn chứng cho 1 topic."""

    topic: str = Field(..., min_length=1)
    sources: list[Source] = Field(default_factory=list)
    key_concepts: list[str] = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)

    theory_chunks: list[TheoryChunk] = Field(
        default_factory=list,
        description=(
            "Đoạn lý thuyết đã gom theo semantic chunking (embedding), phục vụ RAG cho "
            "Curriculum Agent (bản tóm tắt) và Notebook Gen Agent (bản đầy đủ, xem "
            "Module.theory_context). Rỗng nếu topic không có KB hoặc thư viện embedding "
            "không cài được (kb_reader.py tự fallback về BM25/regex khi đó)."
        ),
    )

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

    theory_context: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "concept -> đoạn text KB liên quan (lấy từ ResearchBundle.theory_chunks). "
            "Notebook Gen dùng để bám sát KB gốc khi viết cell lý thuyết. Field này do "
            "curriculum.py gán bằng code Python (tra cứu, KHÔNG qua LLM) SAU KHI LLM trả "
            "JSON xong -- xem prompts/RAG design. Rỗng nếu chưa wiring (chưa sửa curriculum.py)."
        ),
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

    @computed_field
    @property
    def failed_cell_index(self) -> int | None:
        """Index cell hỏng ĐẦU TIÊN, None nếu chạy sạch.

        Schema_Brainstorm mục D yêu cầu field này. Giữ luôn `errors` đầy đủ vì
        executor chạy allow_errors=True nên gom được MỌI cell hỏng, không chỉ cell
        đầu - Huy cần cả danh sách để viết feedback một lần cho đủ.
        """
        return self.errors[0].cell_index if self.errors else None


ExecutionResult = ExcRes  # alias dễ đọc, dùng tên nào cũng được


# ===========================================================================
# 5. VerifierReport - output của HUY
# ===========================================================================


class RuleChecks(ForgeModel):
    """Tám tiêu chí kiểm tra bằng luật (không tốn tiền LLM).

    Sprint 2.2 bổ sung ba rule về trực quan hoá, demo theo module và số
    cell tối thiểu theo level. Ba field mới có default False để đọc được
    report cũ; report mới do Verifier sinh phải truyền tường minh cả tám field.
    """

    has_instructions: bool = Field(
        ...,
        description="Đủ markdown theo level: beginner >= 8, intermediate >= 10",
    )
    has_todo: bool = Field(..., description="Có chỗ TODO cho học viên tự làm")
    has_assert: bool = Field(..., description="Có assert để học viên tự kiểm tra")
    no_hardcoded_answers: bool = Field(..., description="Không lộ đáp án sẵn trong code")
    has_train_test_split: bool = Field(..., description="Có tách train/test")
    has_visualization: bool = Field(
        False, description="Có ít nhất 2 lời gọi trực quan hoá"
    )
    has_demo_per_module: bool = Field(False, description="Mỗi module có code demo")
    min_cells_by_level: bool = Field(
        False, description="Đủ tổng cell theo level: beginner >= 12, intermediate >= 18"
    )

    def _values_for_gate(self) -> list[bool]:
        base = [
            self.has_instructions,
            self.has_todo,
            self.has_assert,
            self.no_hardcoded_answers,
            self.has_train_test_split,
        ]
        extended_names = {
            "has_visualization",
            "has_demo_per_module",
            "min_cells_by_level",
        }
        # Tương thích report/mock cũ chỉ có 5 rule. Khi Verifier Sprint 2.2
        # truyền bất kỳ rule mới nào thì cổng chất lượng dùng đủ cả 8.
        if self.model_fields_set & extended_names:
            base.extend(
                [self.has_visualization, self.has_demo_per_module, self.min_cells_by_level]
            )
        return base

    @computed_field
    @property
    def num_passed(self) -> int:
        return sum(self._values_for_gate())

    @computed_field
    @property
    def num_rules(self) -> int:
        return len(self._values_for_gate())

    @computed_field
    @property
    def all_passed(self) -> bool:
        return self.num_passed == self.num_rules


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
    execution_ok: bool = False


class AttemptRecord(ForgeModel):
    """Một dòng trong retry_history (Schema_Brainstorm mục E: audit trail).

    main.py ghi lại sau mỗi vòng lặp. Huy dùng để viết quality_report.md cuối kỳ:
    thấy rõ điểm đi lên hay đứng yên qua từng attempt.
    """

    attempt: int = Field(..., ge=1)
    nb_path: str = Field(..., min_length=1)
    average_score: float = Field(..., ge=1, le=5)
    rules_passed: int = Field(..., ge=0, le=8)
    rules_total: int = Field(5, ge=1, le=8)
    execution_ok: bool
    cost_this_attempt: float = Field(0.0, ge=0)
    feedback: str | None = None


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
    retry_history: list[AttemptRecord] = Field(
        default_factory=list, description="Audit trail từng attempt - phục vụ quality_report.md"
    )
    error: str | None = Field(None, description="Thông báo lỗi nếu pipeline chết giữa chừng")
    finished_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = SCHEMA_VERSION


def decide_level_final(level_declared: int, quiz_score: int) -> int:
    """Chốt trình độ thật từ trình độ tự khai + điểm quiz.

    Quy tắc lấy nguyên từ đề cương mục 1.1: "Nếu chọn Intermediate mà làm đúng
    dưới 3/5, tự động hạ xuống Beginner". Không có chiều nâng lên.

    Tách riêng level_declared và level_final là CỐ Ý (Schema_Brainstorm mục A):
    khi logic hạ level chạy sai thì còn cả hai số để lần ra chỗ hỏng.

    Nam gọi hàm này thay vì tự cài lại trong streamlit_app.py - hai chỗ tính cùng
    một thứ thì sớm muộn cũng lệch nhau.
    """
    if level_declared == 2 and quiz_score < 3:  # intermediate + quiz thấp -> hạ
        return 1
    return level_declared


def check_path_against_constraints(
    path: "LearningPath", profile: "LearnerProfile"
) -> list[str]:
    """Đối chiếu LearningPath với constraints của học viên (Schema_Brainstorm mục C).

    Không raise: chỉ trả về danh sách cảnh báo. Lý do là LLM hiếm khi khớp chính xác
    tuyệt đối, chặn cứng thì hỏng cả pipeline - nên main.py log cảnh báo, còn Huy
    trừ điểm difficulty_fit dựa trên chúng.

    Sai số cho phép: ±20% thời lượng, ±1 bài tập.
    """
    warnings: list[str] = []
    want_minutes = profile.constraints.duration_minutes
    got_minutes = path.total_estimated_minutes
    if abs(got_minutes - want_minutes) > 0.2 * want_minutes:
        warnings.append(
            f"Thời lượng lệch: lộ trình {got_minutes} phút, học viên yêu cầu {want_minutes} phút"
        )

    want_ex = profile.constraints.num_exercises
    got_ex = path.total_planned_exercises
    if abs(got_ex - want_ex) > 1:
        warnings.append(
            f"Số bài tập lệch: lộ trình {got_ex} bài, học viên yêu cầu {want_ex} bài"
        )

    # Đề cương mục 1.2 + 2.3: LearningPath phải có 4-6 module.
    n_modules = len(path.modules)
    if not 4 <= n_modules <= 6:
        warnings.append(f"Số module = {n_modules}, đề cương yêu cầu 4-6 module")

    if path.level != profile.level_final:
        warnings.append(
            f"Level lệch: lộ trình level {path.level}, học viên level_final {profile.level_final}"
        )

    if path.topic != profile.topic:
        warnings.append(f"Topic lệch: lộ trình '{path.topic}', học viên '{profile.topic}'")

    return warnings


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
    "decide_level_final",
    "check_path_against_constraints",
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
    "AttemptRecord",
    "GenerationResult",
]
