"""Verifier rule-based + Groq LLM-as-a-Judge của Huy."""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Any, Callable, MutableSequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas import (
    PASS_THRESHOLD,
    Decision,
    ExcRes,
    LlmScores,
    ResearchBundle,
    RuleChecks,
    VerifierReport,
)


DEFAULT_JUDGE_MODEL = "qwen/qwen3.6-27b"
MAX_NOTEBOOK_CHARS = 40_000
MIN_CELLS_BY_LEVEL = {1: 8, 2: 12, 3: 16}

_CLUSTERING_TOPICS = frozenset(
    {"kmeans", "k_means", "dbscan", "hierarchical_clustering"}
)
_TRAIN_CSV = re.compile(r"\btrain[\w-]*\.csv\b", re.I)
_HARDCODE_PATTERN = (
    r"(accuracy|acc|score|f1|precision|recall|silhouette|inertia)"
    r"\w*\s*=\s*[\d.]+"
)
_MODULE_HEADING = re.compile(
    r"(?im)^\s{0,3}#{1,6}\s*(?:module|m[oô]-?đun|phần)"
    r"\s*([\w.-]+)?\s*[:.-]?\s*(.*)$"
)
_FEEDBACK_ITEM = re.compile(
    r"\[CELL\s+\d+\]\s+.+?\bFIX:\s*.+?"
    r"(?=(?:\s*\[CELL\s+\d+\])|\Z)",
    re.I | re.S,
)
_SCORE_FIELDS = (
    "executability",
    "groundedness",
    "difficulty_fit",
    "pedagogical_order",
)
_EXTENDED_RULES = (
    "has_visualization",
    "has_demo_per_module",
    "min_cells_by_level",
)

# Sprint 2.2 đã sửa has_assert nên không còn miễn rule nào.
_WAIVED_RULES: frozenset[str] = frozenset()


class VerifierConfigurationError(RuntimeError):
    """Thiếu cấu hình để gọi judge."""


class JudgeResponseError(ValueError):
    """Judge trả dữ liệu sai contract."""


class JudgeOutput(BaseModel):
    """Schema truyền cho ``llm_client.call_json`` để retry và tính cost chung."""

    model_config = ConfigDict(extra="forbid")

    executability: float = Field(..., ge=1, le=5)
    groundedness: float = Field(..., ge=1, le=5)
    difficulty_fit: float = Field(..., ge=1, le=5)
    pedagogical_order: float = Field(..., ge=1, le=5)
    feedback: str | None = None
    ungrounded_claims: list[str] = Field(default_factory=list)

    @field_validator("feedback")
    @classmethod
    def _feedback_has_cell_and_fix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("feedback phải là string không rỗng hoặc null")
        matches = _FEEDBACK_ITEM.findall(value)
        def compact(text: str) -> str:
            return re.sub(r"\s+", "", text)

        if not matches or compact("".join(matches)) != compact(value):
            raise ValueError("feedback sai format [CELL n] ... FIX: ...")
        return " ".join(" ".join(item.split()) for item in matches)

    @field_validator("ungrounded_claims")
    @classmethod
    def _clean_claims(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


JudgeCall = Callable[..., tuple[Any, Any]]


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _strip_comments(text: str) -> str:
    """Bỏ comment Python, giữ string và cấu trúc dòng."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return tokenize.untokenize(
            token for token in tokens if token.type != tokenize.COMMENT
        )
    except (IndentationError, tokenize.TokenError):
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _mask_comments_strings(text: str) -> str:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return tokenize.untokenize(
            token
            for token in tokens
            if token.type not in {tokenize.COMMENT, tokenize.STRING}
        )
    except (IndentationError, tokenize.TokenError):
        return _strip_comments(text)


def _tree(text: str) -> ast.AST | None:
    cleaned = "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in text.splitlines()
    )
    try:
        return ast.parse(cleaned)
    except SyntaxError:
        return None


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.AST = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _topic_key(topic: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (topic or "").lower()).strip("_")


def _infer_level(nb: dict, explicit: int | None = None) -> int:
    if explicit in MIN_CELLS_BY_LEVEL:
        return int(explicit)
    metadata = nb.get("metadata") or {}
    forge = metadata.get("notebookforge") or {}
    values = (
        forge.get("level"),
        forge.get("level_final"),
        metadata.get("level"),
        metadata.get("level_final"),
    )
    labels = {"beginner": 1, "intermediate": 2, "advanced": 3}
    for value in values:
        if isinstance(value, int) and value in MIN_CELLS_BY_LEVEL:
            return value
        if str(value).lower() in labels:
            return labels[str(value).lower()]
    # Hàm run_verifier chưa nhận LearnerProfile. Harness truyền level thật;
    # khi gọi riêng dùng ngưỡng beginner để tránh fail oan.
    return 1


def _real_assert_count(text: str) -> int:
    try:
        return sum(
            token.type == tokenize.NAME and token.string == "assert"
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
        )
    except (IndentationError, tokenize.TokenError):
        return len(re.findall(r"^[ \t]*assert\b", _strip_comments(text), re.M))


def count_asserts(nb: dict) -> int:
    """Đếm assert thực, không tính comment hoặc string."""
    return sum(
        _real_assert_count(_source(cell))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _calls_split(code: list[str]) -> bool:
    for text in code:
        parsed = _tree(text)
        if parsed and any(
            isinstance(node, ast.Call)
            and _call_name(node).endswith("train_test_split")
            for node in ast.walk(parsed)
        ):
            return True
        if re.search(r"\btrain_test_split\s*\(", _mask_comments_strings(text)):
            return True
    return False


def _reads_train_csv(code: list[str]) -> bool:
    """Chỉ nhận read_csv(train*.csv), không nhận to_csv hoặc prose."""
    for text in code:
        parsed = _tree(text)
        if parsed:
            for node in ast.walk(parsed):
                if (
                    not isinstance(node, ast.Call)
                    or not _call_name(node).endswith("read_csv")
                ):
                    continue
                values: list[ast.AST] = list(node.args[:1])
                values.extend(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"filepath_or_buffer", "path"}
                )
                if any(
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and _TRAIN_CSV.search(value.value)
                    for value in values
                ):
                    return True
        if re.search(
            r"\bread_csv\s*\([^)]*['\"][^'\"]*train[\w-]*\.csv['\"]",
            _strip_comments(text),
            re.I | re.S,
        ):
            return True
    return False


def _has_visualization(code: list[str]) -> bool:
    plot_calls = {
        "plot", "scatter", "bar", "barh", "hist", "boxplot", "violinplot",
        "imshow", "matshow", "pie", "heatmap", "pairplot", "lineplot",
        "countplot", "clustermap", "plot_tree",
    }
    display_classes = {
        "ConfusionMatrixDisplay", "DecisionBoundaryDisplay",
        "PrecisionRecallDisplay", "RocCurveDisplay",
    }
    for text in code:
        parsed = _tree(text)
        if parsed:
            for node in ast.walk(parsed):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node)
                tail = name.rsplit(".", 1)[-1]
                if tail in plot_calls or tail in display_classes:
                    return True
                if tail in {"from_estimator", "from_predictions"} and any(
                    display in name for display in display_classes
                ):
                    return True
        if re.search(
            r"\.(?:plot|scatter|bar|hist|boxplot|imshow|heatmap|pairplot|plot_tree)\s*\(",
            _mask_comments_strings(text),
        ):
            return True
    return False


def _module_label(cell: dict) -> str | None:
    metadata = cell.get("metadata") or {}
    nested = metadata.get("notebookforge") or {}
    module_id = metadata.get("module_id") or nested.get("module_id")
    if module_id:
        return str(module_id)
    if cell.get("cell_type") == "markdown":
        match = _MODULE_HEADING.search(_source(cell))
        if match:
            return (match.group(1) or match.group(2) or "module").strip()
    return None


def _complete_demo(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    text = _source(cell)
    if not text.strip() or re.search(r"\bTODO\b|NotImplementedError", text, re.I):
        return False
    parsed = _tree(text)
    if parsed is None:
        meaningful = _mask_comments_strings(text).strip()
        return bool(meaningful and not re.fullmatch(r"(?:pass|\.\.\.)\s*", meaningful))
    useful = (
        ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Call, ast.FunctionDef,
        ast.AsyncFunctionDef, ast.ClassDef, ast.For, ast.While, ast.If,
    )
    return any(isinstance(node, useful) for node in ast.walk(parsed))


def _module_sections(nb: dict) -> list[tuple[str, int, list[dict]]]:
    sections: list[tuple[str, int, list[dict]]] = []
    label: str | None = None
    start = 0
    section_cells: list[dict] = []
    for index, cell in enumerate(nb.get("cells", [])):
        new_label = _module_label(cell)
        if new_label and new_label != label:
            if label is not None:
                sections.append((label, start, section_cells))
            label, start, section_cells = new_label, index, []
        if label is not None:
            section_cells.append(cell)
    if label is not None:
        sections.append((label, start, section_cells))
    return sections


def _modules_have_demo(nb: dict) -> bool:
    sections = _module_sections(nb)
    return bool(sections) and all(
        any(_complete_demo(cell) for cell in cells)
        for _, _, cells in sections
    )


def rule_checks(
    nb: dict,
    topic: str | None = None,
    level: int | None = None,
) -> dict[str, bool]:
    """Chấm đủ 8 rule Sprint 2.2."""
    cells = nb.get("cells", [])
    markdown = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    code = [_source(cell) for cell in cells if cell.get("cell_type") == "code"]
    resolved_level = _infer_level(nb, level)
    hardcoded = any(
        re.search(_HARDCODE_PATTERN, _strip_comments(text), re.I)
        for text in code
    )
    return {
        "has_instructions": len(markdown) >= 3,
        "has_todo": any(re.search(r"\bTODO\b", text, re.I) for text in code),
        "has_assert": any(_real_assert_count(text) for text in code),
        "no_hardcoded_answers": not hardcoded,
        "has_train_test_split": (
            _topic_key(topic) in _CLUSTERING_TOPICS
            or _calls_split(code)
            or _reads_train_csv(code)
        ),
        "has_visualization": _has_visualization(code),
        "has_demo_per_module": _modules_have_demo(nb),
        "min_cells_by_level": len(cells) >= MIN_CELLS_BY_LEVEL[resolved_level],
    }


def blocking_failures(checks: dict[str, bool]) -> list[str]:
    return [
        name
        for name, passed in checks.items()
        if not passed and name not in _WAIVED_RULES
    ]


def decide(
    checks: dict[str, bool],
    average_score: float,
    execution_succeeded: bool = True,
) -> tuple[Decision, str | None]:
    """Verifier quyết PASS/RETRY; main chỉ quyết hết lượt/hết cost."""
    if not execution_succeeded:
        return "RETRY", "Notebook thực thi lỗi nên chưa thể PASS."
    failures = blocking_failures(checks)
    if failures:
        return "RETRY", "Các rule chưa đạt: " + ", ".join(failures)
    if average_score < PASS_THRESHOLD:
        return "RETRY", None
    return "PASS", None


def _first_code_cell(nb: dict) -> int:
    return next(
        (
            index
            for index, cell in enumerate(nb.get("cells", []))
            if cell.get("cell_type") == "code"
        ),
        0,
    )


def _hardcode_cell(nb: dict) -> int:
    for index, cell in enumerate(nb.get("cells", [])):
        if (
            cell.get("cell_type") == "code"
            and re.search(_HARDCODE_PATTERN, _strip_comments(_source(cell)), re.I)
        ):
            return index
    return _first_code_cell(nb)


def build_rule_feedback(
    nb: dict,
    failures: list[str],
    level: int | None = None,
) -> str | None:
    """Feedback rule theo format [CELL n] lỗi. FIX: cách sửa."""
    first_code = _first_code_cell(nb)
    resolved_level = _infer_level(nb, level)
    missing_demos = [
        (label, start)
        for label, start, cells in _module_sections(nb)
        if not any(_complete_demo(cell) for cell in cells)
    ]
    messages: list[str] = []
    for failure in failures:
        if failure == "has_instructions":
            messages.append(
                f"[CELL {first_code}] Thiếu tối thiểu 3 cell markdown hướng dẫn. "
                "FIX: thêm phần giải thích và yêu cầu bài tập trước code."
            )
        elif failure == "has_todo":
            messages.append(
                f"[CELL {first_code}] Không có chỗ để học viên tự làm. "
                "FIX: để trống phần cần hoàn thành và đánh dấu '# TODO:'."
            )
        elif failure == "has_assert":
            last_code = max(
                (
                    index
                    for index, cell in enumerate(nb.get("cells", []))
                    if cell.get("cell_type") == "code"
                ),
                default=first_code,
            )
            messages.append(
                f"[CELL {last_code}] Không có assert thực thi để tự kiểm tra. "
                "FIX: thêm ít nhất một assert kiểm tra kết quả bài tập."
            )
        elif failure == "no_hardcoded_answers":
            index = _hardcode_cell(nb)
            messages.append(
                f"[CELL {index}] Có số gán sẵn vào biến kết quả, làm lộ đáp án. "
                "FIX: tính kết quả từ dữ liệu hoặc output mô hình."
            )
        elif failure == "has_train_test_split":
            messages.append(
                f"[CELL {first_code}] Dữ liệu có nhãn chưa được tách để đánh giá. "
                "FIX: gọi train_test_split(...) hoặc đọc tập train đã tách sẵn."
            )
        elif failure == "has_visualization":
            messages.append(
                f"[CELL {first_code}] Notebook chưa tạo biểu đồ minh hoạ. "
                "FIX: thêm ít nhất một lệnh vẽ phù hợp và giải thích biểu đồ."
            )
        elif failure == "has_demo_per_module":
            if missing_demos:
                for label, index in missing_demos:
                    messages.append(
                        f"[CELL {index}] Module {label} chưa có code demo hoàn chỉnh. "
                        "FIX: thêm code cell chạy được, không chứa TODO, để minh hoạ module."
                    )
            else:
                messages.append(
                    "[CELL 0] Không nhận diện được heading hoặc metadata module. "
                    "FIX: đặt heading dạng '## Module 1: ...' và thêm demo từng module."
                )
        elif failure == "min_cells_by_level":
            required = MIN_CELLS_BY_LEVEL[resolved_level]
            messages.append(
                f"[CELL 0] Notebook level {resolved_level} chỉ có "
                f"{len(nb.get('cells', []))}/{required} cell tối thiểu. "
                "FIX: bổ sung cell lý thuyết, demo và bài tập còn thiếu."
            )
    return " ".join(messages) or None


def build_execution_feedback(exc: ExcRes) -> str | None:
    """Chẩn đoán ExcRes, không để LLM đoán sai nguyên nhân NameError."""
    if exc.success:
        return None
    if exc.timeout_hit:
        cell = exc.errors[0].cell_index if exc.errors else max(exc.executed_cells - 1, 0)
        return (
            f"[CELL {cell}] Notebook vượt timeout thực thi. "
            "FIX: giảm vòng lặp/tìm kiếm tham số và tránh tải dữ liệu khi chạy."
        )

    messages: list[str] = []
    for error in exc.errors:
        value = " ".join(error.evalue.split())
        if error.ename == "NameError":
            match = re.search(r"name ['\"]([^'\"]+)['\"] is not defined", value)
            if match:
                name = match.group(1)
                fix = (
                    f"define '{name}' in an earlier cell, or correct the variable "
                    "name, then rerun cells in order"
                )
            else:
                fix = "define the missing name before use, then rerun cells in order"
        elif error.ename == "KeyError":
            fix = "check df.columns and use the exact existing column name"
        elif error.ename == "FileNotFoundError":
            fix = "use the dataset path injected by the pipeline and verify it exists"
        elif error.ename in {"ModuleNotFoundError", "ImportError"}:
            fix = "replace the dependency with one available in the executor"
        else:
            fix = "correct the expression using the traceback, then rerun the notebook"
        detail = f": {value}" if value else ""
        messages.append(
            f"[CELL {error.cell_index}] {error.ename}{detail}. FIX: {fix}."
        )
    if not messages:
        messages.append(
            "[CELL 0] Notebook chạy lỗi nhưng executor không trả chi tiết. "
            "FIX: bảo đảm ExcRes.errors có ename, evalue và cell_index."
        )
    return " ".join(messages)


def _notebook_content(nb: dict) -> str:
    chunks: list[str] = []
    used = 0
    for index, cell in enumerate(nb.get("cells", [])):
        header = f"\n[CELL {index} | {cell.get('cell_type', 'unknown')}]\n"
        remaining = MAX_NOTEBOOK_CHARS - used - len(header)
        if remaining <= 0:
            chunks.append("\n[NOTEBOOK TRUNCATED]\n")
            break
        chunk = header + _source(cell)[:remaining]
        chunks.append(chunk)
        used += len(chunk)
    return "".join(chunks)


def _execution_summary(exc: ExcRes | None) -> str:
    if exc is None:
        return "Executor result unavailable."
    value = {
        "success": exc.success,
        "total_cells": exc.total_cells,
        "executed_cells": exc.executed_cells,
        "timeout_hit": exc.timeout_hit,
        "duration_seconds": exc.duration_seconds,
        "errors": [
            {
                "cell_index": error.cell_index,
                "ename": error.ename,
                "evalue": error.evalue,
                "traceback_tail": error.traceback_tail,
            }
            for error in exc.errors
        ],
    }
    return json.dumps(value, ensure_ascii=False)


def _research_summary(bundle: ResearchBundle) -> str:
    value = {
        "topic": bundle.topic,
        "key_concepts": bundle.key_concepts,
        "grounded_concepts": bundle.grounded_concepts,
        "unresolved_concepts": bundle.unresolved_concepts,
        "citations": [
            {"concept": item.concept, "source_id": item.source_id}
            for item in bundle.citations
        ],
    }
    return json.dumps(value, ensure_ascii=False)


def _validate_judge(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {*_SCORE_FIELDS, "feedback", "ungrounded_claims"}
    extra = sorted(set(payload) - allowed)
    missing = [field for field in _SCORE_FIELDS if field not in payload]
    if extra or missing:
        raise JudgeResponseError(f"field thừa={extra}, field thiếu={missing}")

    result: dict[str, Any] = {}
    for field in _SCORE_FIELDS:
        score = payload[field]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise JudgeResponseError(f"{field} phải là số")
        if not 1 <= float(score) <= 5:
            raise JudgeResponseError(f"{field} phải nằm trong [1, 5]")
        result[field] = float(score)

    feedback = payload.get("feedback")
    if feedback is not None:
        if not isinstance(feedback, str) or not feedback.strip():
            raise JudgeResponseError("feedback phải là string không rỗng hoặc null")
        matches = _FEEDBACK_ITEM.findall(feedback.strip())
        def compact(value: str) -> str:
            return re.sub(r"\s+", "", value)

        if not matches or compact("".join(matches)) != compact(feedback.strip()):
            raise JudgeResponseError("feedback sai format [CELL n] ... FIX: ...")
        feedback = " ".join(" ".join(item.split()) for item in matches)
    result["feedback"] = feedback

    claims = payload.get("ungrounded_claims", [])
    if not isinstance(claims, list) or not all(isinstance(item, str) for item in claims):
        raise JudgeResponseError("ungrounded_claims phải là list string")
    result["ungrounded_claims"] = [item.strip() for item in claims if item.strip()]
    return result


def llm_judge(
    nb: dict,
    bundle: ResearchBundle,
    *,
    exc: ExcRes | None = None,
    session_id: str,
    judge_call: JudgeCall | None = None,
    model: str | None = None,
    level: int | None = None,
    prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Chấm bằng Judge qua ``llm_client`` để mọi token/cost vào tracker chung."""
    if not session_id.strip():
        raise VerifierConfigurationError("session_id rỗng; không thể tính cost Judge")

    tracker = None
    tracker_mark = None
    if judge_call is None:
        try:
            from llm_client import MODEL_JUDGE, call_json, get_tracker
        except ImportError as error:
            raise VerifierConfigurationError(
                "Thiếu llm_client; merge module của Hoàng trước khi chạy Verifier."
            ) from error
        judge_call = call_json
        selected_model = model or MODEL_JUDGE
        tracker = get_tracker(session_id)
        tracker_mark = tracker.mark()
    else:
        selected_model = model or DEFAULT_JUDGE_MODEL

    path = (
        Path(prompt_path)
        if prompt_path
        else Path(__file__).parents[1] / "prompts" / "verifier.txt"
    )
    template = path.read_text(encoding="utf-8")
    prompt = (
        template.replace("{level}", str(_infer_level(nb, level)))
        .replace("{exc_summary}", _execution_summary(exc))
        .replace("{research_summary}", _research_summary(bundle))
        .replace("{content}", _notebook_content(nb))
    )
    try:
        payload, usage = judge_call(
            prompt,
            JudgeOutput,
            session_id=session_id,
            max_tokens=1_200,
            temperature=0,
            model=selected_model,
            reasoning_effort="none",
            include_reasoning=False,
        )
    except Exception as error:
        raise RuntimeError(f"Gọi Judge qua llm_client thất bại: {error}") from error

    if isinstance(payload, BaseModel):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        raise JudgeResponseError("llm_client.call_json không trả Pydantic model/dict")
    result = _validate_judge(payload)
    if exc is not None and not exc.success:
        result["executability"] = min(result["executability"], 2.0)
        if exc.timeout_hit:
            result["executability"] = 1.0

    if tracker is not None and tracker_mark is not None:
        judge_cost = tracker.cost_since(tracker_mark)
    else:
        judge_cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
    result["judge_cost_usd"] = round(judge_cost, 6)
    return result


def _merge_feedback(*parts: str | None) -> str | None:
    unique: list[str] = []
    for part in parts:
        if part and part.strip() and part.strip() not in unique:
            unique.append(part.strip())
    return " ".join(unique) or None


def update_retry_history(
    history: list[dict[str, Any]],
    report: VerifierReport,
    exc: ExcRes,
    *,
    extended_checks: dict[str, bool] | None = None,
    judge_cost_usd: float = 0.0,
) -> list[dict[str, Any]]:
    """Upsert attempt theo số; gọi lại không tạo dòng trùng."""
    checks = report.rule_checks.model_dump(exclude_computed_fields=True)
    if extended_checks:
        checks.update(extended_checks)
    item = {
        "attempt": report.attempt,
        "nb_path": report.nb_path,
        "decision": report.decision,
        "average_score": report.average_score,
        "rule_checks": checks,
        "llm_scores": report.llm_scores.model_dump(exclude_computed_fields=True),
        "execution_success": exc.success,
        "duration_seconds": exc.duration_seconds,
        "cost_this_attempt": exc.cost_this_attempt,
        "judge_cost_usd": judge_cost_usd,
        "feedback": report.feedback,
        "ungrounded_claims": report.ungrounded_claims,
    }
    by_attempt = {int(old["attempt"]): dict(old) for old in history}
    by_attempt[report.attempt] = item
    return [by_attempt[key] for key in sorted(by_attempt)]


def render_quality_report(history: list[dict[str, Any]]) -> str:
    """Render retry_history thành Markdown cho Streamlit."""
    lines = [
        "# NotebookForge Quality Report",
        "",
        "| Attempt | Execution | Rules | LLM average | Decision | Attempt cost | Judge cost | Runtime (s) |",
        "| ---: | :---: | :---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for item in history:
        checks = item.get("rule_checks") or {}
        lines.append(
            "| {attempt} | {execution} | {passed}/{total} | {average:.3f} | "
            "{decision} | {cost:.4f} | {judge_cost:.4f} | {runtime:.2f} |".format(
                attempt=item.get("attempt", "?"),
                execution="PASS" if item.get("execution_success") else "FAIL",
                passed=sum(bool(value) for value in checks.values()),
                total=len(checks),
                average=float(item.get("average_score") or 0),
                decision=item.get("decision", "?"),
                cost=float(item.get("cost_this_attempt") or 0),
                judge_cost=float(item.get("judge_cost_usd") or 0),
                runtime=float(item.get("duration_seconds") or 0),
            )
        )
    for item in history:
        lines.extend(["", f"## Attempt {item.get('attempt', '?')}", ""])
        feedback = str(
            item.get("feedback") or "Không có lỗi cần sửa."
        ).replace("\n", " ")
        lines.append(f"- Feedback: {feedback}")
        claims = item.get("ungrounded_claims") or []
        if claims:
            lines.append("- Ungrounded claims: " + "; ".join(map(str, claims)))
    lines.append("")
    return "\n".join(lines)


def write_quality_report(
    history: list[dict[str, Any]],
    markdown_path: str | Path,
) -> Path:
    """Ghi Markdown và JSON sidecar để attempt sau upsert an toàn."""
    path = Path(markdown_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_quality_report(history), encoding="utf-8")
    sidecar = path.with_name(f"{path.stem}.history.json")
    sidecar.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _load_history(markdown_path: Path) -> list[dict[str, Any]]:
    sidecar = markdown_path.with_name(f"{markdown_path.stem}.history.json")
    if not sidecar.exists():
        return []
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def run_verifier(
    nb_path: str,
    exc: ExcRes,
    bundle: ResearchBundle,
    *,
    session_id: str | None = None,
    judge_call: JudgeCall | None = None,
    judge_model: str | None = None,
    judge_prompt_path: str | Path | None = None,
    level: int | None = None,
    retry_history: MutableSequence[dict[str, Any]] | None = None,
    quality_report_path: str | Path | None = None,
) -> VerifierReport:
    """Chấm notebook và cập nhật quality report.

    ``session_id`` phải là ``LearnerProfile.session_id`` mà main.py dùng cho
    CostTracker. Fallback theo tên thư mục notebook chỉ để tương thích code cũ.
    """
    with open(nb_path, encoding="utf-8") as file:
        nb = json.load(file)

    checks = rule_checks(nb, topic=bundle.topic, level=level)
    failures = blocking_failures(checks)
    resolved_session_id = session_id or Path(nb_path).parent.name
    if not resolved_session_id:
        resolved_session_id = f"verifier-attempt-{exc.attempt}"
    judge = llm_judge(
        nb,
        bundle,
        exc=exc,
        session_id=resolved_session_id,
        judge_call=judge_call,
        model=judge_model,
        level=level,
        prompt_path=judge_prompt_path,
    )
    judge_cost = float(judge.get("judge_cost_usd") or 0.0)
    exc.cost_this_attempt = round(exc.cost_this_attempt + judge_cost, 6)
    scores = LlmScores(**{field: judge[field] for field in _SCORE_FIELDS})
    decision, _ = decide(
        checks,
        scores.average,
        execution_succeeded=exc.success,
    )
    feedback = _merge_feedback(
        build_execution_feedback(exc),
        build_rule_feedback(nb, failures, level),
        judge.get("feedback"),
    )

    # Schema của Hoàng hiện có 5 rule. Ba rule mới vẫn chặn PASS và được ghi
    # trong notes/history; khi schema thêm field, chúng tự đi vào RuleChecks.
    supported = set(RuleChecks.model_fields)
    schema_checks = {
        name: passed for name, passed in checks.items() if name in supported
    }
    missing_contract = [name for name in _EXTENDED_RULES if name not in supported]
    notes = None
    if missing_contract:
        notes = "Extended rule checks: " + "; ".join(
            f"{name}={'PASS' if checks[name] else 'FAIL'}"
            for name in missing_contract
        )

    report = VerifierReport(
        nb_path=nb_path,
        attempt=exc.attempt,
        rule_checks=RuleChecks(**schema_checks),
        llm_scores=scores,
        decision=decision,
        feedback=feedback,
        ungrounded_claims=judge.get("ungrounded_claims", []),
        notes=notes,
    )

    markdown = (
        Path(quality_report_path)
        if quality_report_path
        else Path(nb_path).with_name("quality_report.md")
    )
    base = list(retry_history) if retry_history is not None else _load_history(markdown)
    history = update_retry_history(
        base,
        report,
        exc,
        extended_checks={name: checks[name] for name in _EXTENDED_RULES},
        judge_cost_usd=judge_cost,
    )
    if retry_history is not None:
        retry_history[:] = history
    try:
        write_quality_report(history, markdown)
    except OSError as error:
        report.notes = _merge_feedback(
            report.notes,
            f"Không ghi được quality report: {error}",
        )
    return report


_SAMPLE_NB = {
    "cells": [
        {"cell_type": "markdown", "source": ["# Logistic Regression\n"]},
        {"cell_type": "markdown", "source": ["## Module 1: Chuẩn bị\n"]},
        {
            "cell_type": "code",
            "source": [
                "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
                "print(X_train.shape)\n",
            ],
        },
        {"cell_type": "markdown", "source": ["## Module 2: Huấn luyện\n"]},
        {"cell_type": "code", "source": ["model.fit(X_train, y_train)\n"]},
        {"cell_type": "code", "source": ["# TODO: thử tham số khác\n"]},
        {"cell_type": "code", "source": ["assert len(X_train) > 0\n"]},
        {"cell_type": "code", "source": ["plt.plot([0, 1], [0, 1])\n"]},
    ],
    "metadata": {"notebookforge": {"level": 1}},
}


if __name__ == "__main__":
    for rule, passed in rule_checks(
        _SAMPLE_NB,
        topic="logistic_regression",
    ).items():
        print(f"{'PASS' if passed else 'FAIL'}  {rule}")
