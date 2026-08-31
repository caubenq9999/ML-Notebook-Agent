"""
api.py - NotebookForge (Nhóm 19)
================================
Chủ sở hữu: HOÀNG. FastAPI, 2 endpoint như đề cương:

    POST /generate              -> nhận form của Nam, trả session_id (chạy nền)
    GET  /report/{session_id}   -> hỏi trạng thái + lấy GenerationResult

Vì sao tách 2 endpoint thay vì 1 endpoint chạy thẳng: một session có thể mất
vài phút (2 attempt x sinh + chạy + chấm). Giữ HTTP request mở suốt thời gian đó
sẽ timeout. Nam gọi POST rồi poll GET vài giây/lần để vẽ thanh tiến trình.

Chạy server:
    uvicorn api:app --reload --port 8000
    -> mở http://127.0.0.1:8000/docs để bấm thử

Test nhanh không tốn tiền LLM (dùng mock):
    POST /generate với {"topic": "Logistic Regression", "level": 1,
    "quiz_score": 3, "duration_minutes": 60, "num_exercises": 3,
    "use_mock": true}

Body phẳng của POST /generate:
    {"topic": "Logistic Regression", "level": 2,
     "quiz_score": 4, "duration_minutes": 90, "num_exercises": 4}

API cũng nhận body LearnerProfile của Streamlit: level_declared, level_final,
constraints, session_id và created_at. Nếu hai format cùng xuất hiện thì giá trị
phải khớp nhau; field lạ bị trả 422.

(Đề cương mục 1.1 ghi JSON mẫu là "level": "intermediate" dạng chuỗi. Nhóm chốt
ngày 10/8 dùng số 1/2 cho khớp code Huy + Nam - lệch có chủ đích, ghi vào báo cáo.)

GHI CHÚ SPRINT 2: kết quả lưu trong RAM (dict). Restart server là mất.
Đủ cho demo; nếu Sprint 3 cần lưu lâu thì đổi _STORE sang SQLite - chỉ phải
sửa 3 hàm _store_* bên dưới, không ảnh hưởng ai.
"""

from __future__ import annotations

import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import BackgroundTasks, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, model_validator  # noqa: E402

from schemas import (  # noqa: E402
    COST_CAP_USD,
    MAX_ATTEMPT,
    PASS_THRESHOLD,
    SCHEMA_VERSION,
    Constraints,
    GenerationResult,
    LearnerProfile,
    Level,
    decide_level_final,
)

app = FastAPI(
    title="NotebookForge API",
    version=SCHEMA_VERSION,
    description="Sinh notebook học ML cá nhân hoá theo trình độ người học (Nhóm 19)",
)

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
ARTIFACTS: dict[str, tuple[str, str]] = {
    "notebook": ("notebook.ipynb", "application/x-ipynb+json"),
    "path": ("path.json", "application/json"),
    "quality_report": ("quality_report.md", "text/markdown; charset=utf-8"),
}

# ---------------------------------------------------------------------------
# Kho kết quả trong RAM
# ---------------------------------------------------------------------------

SessionStatus = Literal["running", "completed", "error"]

_STORE: dict[str, dict] = {}
_LOCK = threading.Lock()


def _store_init(session_id: str) -> None:
    with _LOCK:
        _STORE[session_id] = {"status": "running", "result": None, "error": None}


def _store_finish(session_id: str, result: GenerationResult) -> None:
    """Lưu một pipeline đã chạy xong, bất kể có vượt quality gate hay không.

    FAIL_MAX_RETRY/FAIL_COST_CAP là kết quả nghiệp vụ hợp lệ và vẫn có thể có
    notebook tốt nhất để người dùng xem hoặc tải xuống. Chỉ exception thật sự
    trong _run_job mới dùng trạng thái HTTP-level ``error`` qua _store_fail().
    """
    with _LOCK:
        _STORE[session_id] = {
            "status": "completed",
            "result": result,
            "error": result.error,
        }


def _store_fail(session_id: str, message: str) -> None:
    with _LOCK:
        _STORE[session_id] = {"status": "error", "result": None, "error": message}


# ---------------------------------------------------------------------------
# Schema riêng của tầng HTTP (không đụng schemas.py - đó là contract nội bộ)
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Nhận cả body phẳng của API và body LearnerProfile của Streamlit.

    Hai format phải mang cùng giá trị nếu cùng xuất hiện. Field lạ bị
    chặn thay vì âm thầm bỏ qua rồi chạy bằng default.
    """

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1, examples=["Logistic Regression"])
    level: Level | None = Field(None, description="Format phẳng: level tự khai")
    level_declared: Level | None = Field(None, description="Format LearnerProfile của UI")
    level_final: Level | None = Field(None, description="UI tính trước; API sẽ kiểm tra lại")
    quiz_score: int = Field(..., ge=0, le=5, description="Số câu đúng trên 5 câu quiz")
    duration_minutes: int | None = Field(None, ge=60, le=120)
    num_exercises: int | None = Field(None, ge=3, le=5)
    constraints: Constraints | None = None
    session_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    created_at: datetime | None = None
    dataset_seed: int | None = Field(None, description="Bỏ trống = random (khuyến nghị)")
    use_mock: bool = Field(False, description="True = chạy bằng hàm giả, không tốn tiền LLM")

    @model_validator(mode="after")
    def validate_contract(self) -> "GenerateRequest":
        if self.level is None and self.level_declared is None:
            raise ValueError("Thiếu level hoặc level_declared")
        if (
            self.level is not None
            and self.level_declared is not None
            and self.level != self.level_declared
        ):
            raise ValueError("level và level_declared không khớp")

        if self.constraints is None:
            if self.duration_minutes is None or self.num_exercises is None:
                raise ValueError(
                    "Thiếu constraints hoặc cặp duration_minutes/num_exercises"
                )
        else:
            if (
                self.duration_minutes is not None
                and self.duration_minutes != self.constraints.duration_minutes
            ):
                raise ValueError("duration_minutes không khớp constraints")
            if (
                self.num_exercises is not None
                and self.num_exercises != self.constraints.num_exercises
            ):
                raise ValueError("num_exercises không khớp constraints")

        expected = decide_level_final(self.resolved_level, self.quiz_score)
        if self.level_final is not None and self.level_final != expected:
            raise ValueError(
                f"level_final={self.level_final} sai quy tắc; giá trị đúng là {expected}"
            )
        return self

    @property
    def resolved_level(self) -> Level:
        value = self.level if self.level is not None else self.level_declared
        assert value is not None  # model_validator đã chặn
        return value

    @property
    def resolved_constraints(self) -> Constraints:
        if self.constraints is not None:
            return self.constraints
        assert self.duration_minutes is not None and self.num_exercises is not None
        return Constraints(
            duration_minutes=self.duration_minutes,
            num_exercises=self.num_exercises,
        )


class GenerateResponse(BaseModel):
    session_id: str
    task_id: str
    status: SessionStatus
    poll_url: str
    message: str


class ResultResponse(BaseModel):
    session_id: str
    status: SessionStatus
    result: GenerationResult | None = None
    error: str | None = None
    error_message: str | None = None
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    notebook: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    quality_report: str | None = None


# ---------------------------------------------------------------------------
# Logic phụ
# ---------------------------------------------------------------------------


def _run_job(profile: LearnerProfile, force_mock: bool) -> None:
    """Chạy pipeline dưới nền rồi cất kết quả vào _STORE."""
    from main import generate

    try:
        result = generate(profile, force_mock=force_mock)
        _store_finish(profile.session_id, result)
    except Exception as exc:  # noqa: BLE001 - job nền không được phép làm chết server
        _store_fail(profile.session_id, f"{type(exc).__name__}: {exc}")


def _read_result_artifacts(
    session_id: str, result: GenerationResult | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Nạp payload UI cần từ ba artifact; thiếu file thì trả None an toàn."""
    notebook_data = None
    quality_report = None
    notebook_path = OUTPUT_ROOT / session_id / ARTIFACTS["notebook"][0]
    report_path = OUTPUT_ROOT / session_id / ARTIFACTS["quality_report"][0]
    try:
        if notebook_path.is_file():
            import json

            notebook_data = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        notebook_data = None
    try:
        if report_path.is_file():
            quality_report = report_path.read_text(encoding="utf-8")
    except OSError:
        quality_report = None
    report_data = result.model_dump(mode="json") if result is not None else None
    return notebook_data, report_data, quality_report


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/generate", response_model=GenerateResponse, status_code=202)
def generate_notebook(req: GenerateRequest, background: BackgroundTasks) -> GenerateResponse:
    """Nhận yêu cầu, trả session_id ngay, pipeline chạy nền."""
    session_id = req.session_id or f"web-{uuid.uuid4().hex[:8]}"
    with _LOCK:
        if session_id in _STORE:
            raise HTTPException(status_code=409, detail=f"Session '{session_id}' đã tồn tại")

    constraints = req.resolved_constraints
    level_declared = req.resolved_level

    profile_kwargs = dict(
        topic=req.topic,
        level_declared=level_declared,
        # Quy tắc hạ trình độ nằm ở schemas.py để Nam và mình dùng chung một hàm,
        # không ai tự cài lại rồi lệch nhau.
        level_final=decide_level_final(level_declared, req.quiz_score),
        quiz_score=req.quiz_score,
        constraints=constraints,
        session_id=session_id,
    )
    if req.created_at is not None:
        profile_kwargs["created_at"] = req.created_at
    if req.dataset_seed is not None:
        profile_kwargs["dataset_seed"] = req.dataset_seed

    profile = LearnerProfile(**profile_kwargs)

    _store_init(session_id)
    background.add_task(_run_job, profile, req.use_mock)

    return GenerateResponse(
        session_id=session_id,
        task_id=session_id,
        status="running",
        poll_url=f"/report/{session_id}",
        message=(
            f"Đang sinh notebook (tối đa {MAX_ATTEMPT} attempt, trần ${COST_CAP_USD}). "
            f"Poll {session_id} vài giây một lần."
        ),
    )


@app.get("/report/{session_id}", response_model=ResultResponse)
def get_report(session_id: str) -> ResultResponse:
    """Hỏi trạng thái. status='running' -> poll tiếp; terminal -> đọc result.

    Tên endpoint lấy đúng theo đề cương mục 4 (phần vai của Nam):
    "polling GET /report/{id}, không để user tưởng app treo".
    """
    with _LOCK:
        entry = _STORE.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Không có session '{session_id}'")
    artifact_urls = {}
    if entry["status"] in {"completed", "error"}:
        artifact_urls = {
            name: f"/artifacts/{session_id}/{name}"
            for name, (filename, _) in ARTIFACTS.items()
            if (OUTPUT_ROOT / session_id / filename).is_file()
        }
    notebook_data, report_data, quality_report = _read_result_artifacts(
        session_id, entry["result"]
    )
    return ResultResponse(
        session_id=session_id,
        status=entry["status"],
        result=entry["result"],
        error=entry["error"],
        error_message=entry["error"],
        artifact_urls=artifact_urls,
        notebook=notebook_data,
        report=report_data,
        quality_report=quality_report,
    )


@app.get("/artifacts/{session_id}/{artifact_name}", response_class=FileResponse)
def download_artifact(session_id: str, artifact_name: str) -> FileResponse:
    """Tải một trong ba sản phẩm cuối của session.

    Dùng tên logic cố định thay vì nhận đường dẫn tuỳ ý để không mở lỗ hổng
    path traversal. Nam lấy các URL này trực tiếp từ `artifact_urls` của report.
    """
    with _LOCK:
        entry = _STORE.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Không có session '{session_id}'")
    if entry["status"] == "running":
        raise HTTPException(status_code=409, detail="Session chưa sinh xong artifact")

    artifact = ARTIFACTS.get(artifact_name)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact không hợp lệ. Chọn: {list(ARTIFACTS)}",
        )
    filename, media_type = artifact
    path = OUTPUT_ROOT / session_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Session thiếu file '{filename}'")
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/health")
def health() -> dict:
    """Kiểm tra server sống + xem cấu hình đang chạy (tiện lúc demo)."""
    with _LOCK:
        running = sum(1 for e in _STORE.values() if e["status"] == "running")
        total = len(_STORE)
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "max_attempt": MAX_ATTEMPT,
        "cost_cap_usd": COST_CAP_USD,
        "pass_threshold": PASS_THRESHOLD,
        "sessions_total": total,
        "sessions_running": running,
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
