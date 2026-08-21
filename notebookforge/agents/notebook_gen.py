"""
agents/notebook_gen.py — HỢP
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0 , str(PROJECT_ROOT))

import nbformat
# new_notebook(): Khởi tạo một đối tượng có cấu trúc file Jupyter Notebook trống.
# new_markdown_cell(source): tạo cell dạng Markdown với nội dung lấy tư source.
# new_code_cell(source): tạo cell dạng code với nội dung lấy tư source.
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from tools.dataset_injector import get_dataset_code

from schemas import LearnerProfile, LearningPath, Module

from llm_client import PROVIDER, call_text

from agents.curriculum import (
    LEVEL_NAMES,
    SUPERVISED_CLASSIFICATION,
    UNSUPERVISED_CLUSTERING,
    clamp_num_exercises,
    get_exercise_range,
    get_problem_type
)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "notebook_gen.txt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_notebooks"
DATASET_PLACEHOLDER = "# DATASET_INJECTION_PLACEHOLDER"
NOTEBOOK_MAX_TOKENS = int(os.getenv("NOTEBOOKFORGE_NOTEBOOK_MAX_TOKENS", "5000"))
NOTEBOOK_TPM_BUDGET = int(os.getenv("NOTEBOOKFORGE_NOTEBOOK_TPM_BUDGET", "7600"))
NOTEBOOK_CHARS_PER_TOKEN = float(
    os.getenv("NOTEBOOKFORGE_NOTEBOOK_CHARS_PER_TOKEN", "3.0")
)
NOTEBOOK_MIN_OUTPUT_TOKENS = int(
    os.getenv("NOTEBOOKFORGE_NOTEBOOK_MIN_OUTPUT_TOKENS", "3000")
)
NOTEBOOK_FEEDBACK_MAX_CHARS = int(
    os.getenv("NOTEBOOKFORGE_NOTEBOOK_FEEDBACK_MAX_CHARS", "600")
)
NOTEBOOK_REASONING_EFFORT = os.getenv(
    "NOTEBOOKFORGE_NOTEBOOK_REASONING_EFFORT", "low"
).strip().lower()
NOTEBOOK_JSON_ATTEMPTS = 2

# =======================
# CHỐT  3 dataset Kaggle
# =======================
DATASET_INFO: dict[str, dict[str, str | None]] = {
    "logistic_regression": {
        "name": "Heart Failure Prediction Dataset",
        "kaggle": "kaggle.com/datasets/fedesoriano/heart-failure-prediction",
        "size": "918 dòng, 12 cột (11 đặc trưng lâm sàng + 1 nhãn)",
        "target": "HeartDisease (0 = không bệnh, 1 = có bệnh tim mạch)",
        "description": (
            "11 đặc trưng lâm sàng (Age, Sex, ChestPainType, RestingBP, Cholesterol, "
            "FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope) dùng để dự "
            "đoán khả năng mắc bệnh tim mạch — bài toán phân loại nhị phân, phù hợp minh "
            "hoạ Logistic Regression."
        ),
        "preprocessing_note": (
            "Một số cột có giá trị 0 bất hợp lý về mặt sinh lý (ví dụ Cholesterol = 0, "
            "RestingBP = 0) đã được coi là giá trị khuyết và impute lại bằng median; dữ "
            "liệu đã được chuẩn hoá (StandardScaler)."
        ),
    },
    "decision_tree": {
        "name": "Red Wine Quality Dataset",
        "kaggle": "kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009",
        "size": "1.599 dòng, 12 cột (11 đặc trưng hoá lý + 1 điểm chất lượng)",
        "target": (
            "quality — điểm chất lượng rượu gốc (thang điểm 3-8, ~6 lớp phân loại), dùng "
            "trực tiếp làm nhãn đa lớp (multi-class), KHÔNG được gộp nhóm lại"
        ),
        "description": (
            "11 đặc trưng hoá lý của rượu vang đỏ (fixed acidity, volatile acidity, citric "
            "acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, "
            "sulphates, alcohol) dùng để phân loại chất lượng rượu — bài toán phân loại đa "
            "lớp (multi-class), phù hợp minh hoạ Decision Tree vì các đặc trưng có ngưỡng "
            "cắt rõ ràng theo từng khoảng giá trị."
        ),
        "preprocessing_note": (
            "Dữ liệu đã được hệ thống loại bỏ các dòng trùng lặp (~240 dòng) trước khi chia "
            "train/test (stratify theo 'quality'). Nhãn 'quality' GIỮ NGUYÊN dạng gốc (đa lớp, "
            "phân bố có thể lệch giữa các lớp) — chưa được gộp nhóm; ngưỡng accuracy 'tốt' nên "
            "được xác định qua thực nghiệm trên dataset này, không kỳ vọng cao như bài toán "
            "nhị phân."
        ),
    },
    "kmeans": {
        "name": "Mall Customer Segmentation Dataset",
        "kaggle": "kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python",
        "size": "200 dòng, 5 cột (sau khi bỏ cột định danh)",
        "target": None,  # không có nhãn thật — bài toán không giám sát
        "description": (
            "Thông tin khách hàng của một trung tâm thương mại: Gender, Age, "
            "Annual Income (k$), Spending Score (1-100) — sau khi đã loại bỏ cột định danh "
            "CustomerID. KHÔNG có nhãn phân khúc sẵn, cần phân cụm khách hàng theo hành "
            "vi chi tiêu."
        ),
        "preprocessing_note": (
            "Cột CustomerID đã bị loại bỏ vì không mang thông tin phân cụm; biến Gender đã "
            "được mã hoá nếu sử dụng; toàn bộ đặc trưng số đã được chuẩn hoá (StandardScaler) "
            "trước khi phân cụm."
        ),
    },
}

# Dataset Injector chạy trong Executor không có mạng, nên metadata dùng cùng các
# dataset được đóng gói sẵn trong scikit-learn thay vì mô tả các file Kaggle cũ.
DATASET_INFO.update({
    "logistic_regression": {
        "name": "Breast Cancer Wisconsin Dataset",
        "kaggle": "sklearn.datasets.load_breast_cancer (local)",
        "size": "569 dòng, 30 đặc trưng số và 1 nhãn nhị phân",
        "target": "target (0 = malignant, 1 = benign)",
        "description": "Bài toán phân loại nhị phân phù hợp Logistic Regression.",
        "preprocessing_note": "Dữ liệu local được chia stratified và chuẩn hoá bằng StandardScaler.",
    },
    "decision_tree": {
        "name": "Wine Recognition Dataset",
        "kaggle": "sklearn.datasets.load_wine (local)",
        "size": "178 dòng, 13 đặc trưng số và 1 nhãn ba lớp",
        "target": "target (ba giống nho)",
        "description": "Bài toán phân loại đa lớp phù hợp Decision Tree.",
        "preprocessing_note": "Dữ liệu local được bỏ trùng và chia stratified.",
    },
    "kmeans": {
        "name": "Iris Dataset",
        "kaggle": "sklearn.datasets.load_iris (local)",
        "size": "150 dòng và 4 đặc trưng số",
        "target": None,
        "description": "Bỏ nhãn gốc và dùng bốn đặc trưng để minh hoạ K-Means.",
        "preprocessing_note": "Bốn đặc trưng được chuẩn hoá bằng StandardScaler trước khi phân cụm.",
    },
})

_DEFAULT_DATASET_INFO = {
    "name": "(dataset sẽ được hệ thống chèn tự động)",
    "kaggle": "",
    "size": "",
    "target": "",
    "description": (
        "Dataset thật phù hợp với topic, do hệ thống tự động chèn vào — mô tả khái quát "
        "dựa trên các concepts của topic này."
    ),
    "preprocessing_note": "",
}

# Đồng bộ hoá dạng bài toán
def _normalize_topic(topic: str) -> str:
    normalized = str(topic).strip().lower().replace(" ", "_").replace("-", "_")
    return "kmeans" if normalized in {"k_means", "k_mean"} else normalized

# Lấy dataset_info (raw)
def get_dataset_info(topic: str) -> dict[str, str | None]:
    return DATASET_INFO.get(_normalize_topic(topic), _DEFAULT_DATASET_INFO)

# Chuẩn hoá phần dataset_info
def build_dataset_info_block(topic: str) -> str:
    info = get_dataset_info(topic)
    lines = [
        f"Tên dataset: {info['name']}",
        f"Nguồn: {info['kaggle']}" if info.get("kaggle") else "Nguồn: (đang cập nhật)",
        f"Kích thước: {info['size']}" if info.get("size") else None,
        (
            f"Biến mục tiêu: {info['target']}"
            if info.get("target")
            else "Biến mục tiêu: KHÔNG có (bài toán học không giám sát / phân cụm)"
        ),
        f"Mô tả: {info['description']}",
        f"Một số chú ý tiền xử lý : {info['preprocessing_note']}"
        if info.get("preprocessing_note")
        else None,
    ]
    return "\n".join(line for line in lines if line)


# ================================================================================
# Loại bài toán -> biến số sẽ có sẵn sau khi hệ thống chèn dataset thật ở KHỐI 2.
# K-Means (unsupervised) KHÔNG train_test_split, KHÔNG accuracy.
# ================================================================================
def build_problem_type_block(topic: str) -> str:
    ptype = get_problem_type(topic)
    if ptype == UNSUPERVISED_CLUSTERING:
        return (
            "Loại bài toán: KHÔNG GIÁM SÁT / PHÂN CỤM (unsupervised clustering).\n"
            "Biến có sẵn sau KHỐI 2 (do hệ thống chèn): X KHÔNG có train_test_split, KHÔNG có y thật để so khớp).\n"
            "Metric đánh giá hợp lệ: inertia (WCSS), silhouette_score (khoảng [-1, 1], "
            "càng gần 1 càng tốt), davies_bouldin_score nếu có trong concepts. TUYỆT ĐỐI "
            "KHÔNG dùng accuracy/precision/recall/f1 vì không có nhãn thật để so sánh.\n"
            "KHÔNG được viết hoặc yêu cầu học viên viết train_test_split ở bất kỳ đâu."
        )
    note_multiclass = ""
    if _normalize_topic(topic) == "decision_tree":
        note_multiclass = (
            "\nLƯU Ý: target 'quality' của dataset này là ĐA LỚP (~6 lớp: 3-8), không phải nhị "
            "phân — nếu dùng precision/recall/f1/roc_auc, PHẢI truyền tham số average="
            "'weighted' (hoặc 'macro') cho precision/recall/f1, và multi_class='ovr' cho "
            "roc_auc_score; accuracy_score dùng bình thường không cần tham số thêm."
        )
    return (
        "Loại bài toán: CÓ GIÁM SÁT / PHÂN LOẠI (supervised classification).\n"
        "Biến có sẵn sau KHỐI 2 (do hệ thống chèn): X_train, X_test, y_train, y_test "
        "(đã tách sẵn bằng train_test_split).\n"
        "Metric đánh giá hợp lệ: accuracy, precision, recall, f1, roc_auc, confusion matrix, "
        "hoặc feature_importances_/plot_tree nếu concepts của module thuộc nhóm diễn giải cây."
        f"{note_multiclass}"
    )


# =====================
# 1. Xây dựng prompt
# =====================
def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding = "utf-8")

def compact_prior_feedback(prior_feedback: str) -> str:
    """Deduplicate repeated `[CELL n]` issues and cap retry-only input growth."""
    entries = [
        item.strip()
        for item in re.split(r"(?=\[CELL\s+\d+\])", prior_feedback.strip())
        if item.strip()
    ]
    if not entries:
        return prior_feedback.strip()[:NOTEBOOK_FEEDBACK_MAX_CHARS]

    unique: dict[str, str] = {}
    for entry in entries:
        issue = entry.split("FIX:", 1)[0].strip().rstrip(".")
        key = re.sub(r"\s+", " ", issue).casefold()
        # Feedback Judge thường đứng sau feedback Executor và có cách sửa cụ thể hơn.
        unique[key] = entry

    selected: list[str] = []
    used = 0
    for entry in unique.values():
        separator = 1 if selected else 0
        if used + separator + len(entry) > NOTEBOOK_FEEDBACK_MAX_CHARS:
            continue
        selected.append(entry)
        used += separator + len(entry)
    if selected:
        return " ".join(selected)
    return next(iter(unique.values()))[:NOTEBOOK_FEEDBACK_MAX_CHARS]


def check_prior_feedback(prior_feedback : Optional[str]) -> str:
    if not prior_feedback:
        return ""

    compact_feedback = compact_prior_feedback(prior_feedback)
    return (
        "<prior_feedback>\n"
        "Notebook ở lần sinh trước đã bị Verifier từ chối vì lý do bên dưới. "
        "Đây là yêu cầu BẮT BUỘC phải sửa đúng chỗ lỗi, không lặp lại lỗi cũ, và "
        "không được sinh lại nội dung gần như giống hệt bản trước:\n"
        f"{compact_feedback}\n"
        "</prior_feedback>"
    )


def notebook_request_max_tokens(prompt: str) -> int:
    """Keep Groq input + reserved output below a safe per-request TPM budget."""
    if PROVIDER != "groq" or NOTEBOOK_TPM_BUDGET <= 0:
        return NOTEBOOK_MAX_TOKENS
    if NOTEBOOK_CHARS_PER_TOKEN <= 0:
        raise ValueError("NOTEBOOKFORGE_NOTEBOOK_CHARS_PER_TOKEN phải > 0")

    estimated_input = math.ceil(len(prompt) / NOTEBOOK_CHARS_PER_TOKEN)
    available_output = NOTEBOOK_TPM_BUDGET - estimated_input
    if available_output < NOTEBOOK_MIN_OUTPUT_TOKENS:
        raise ValueError(
            "Prompt NotebookGen quá lớn cho TPM budget: "
            f"ước tính input={estimated_input}, output còn={available_output}, "
            f"tối thiểu cần={NOTEBOOK_MIN_OUTPUT_TOKENS}. Hãy rút input prompt."
        )
    return min(NOTEBOOK_MAX_TOKENS, available_output)

# Lấy các concepts không có trong "planned_exercises", giải quyết phần 3.2
def _theory_only_concepts(module: Module) -> list[str]:
    used_in_exercises: set[str] = set()
    for ex in module.planned_exercises:
        used_in_exercises.update(ex.concepts)
    return [c for c in module.concepts if c not in used_in_exercises]


# Chuyển modules[] của LearningPath thành text chi tiết đưa vào prompt — tách rõ
# "concepts lý thuyết riêng" khỏi "concepts đã có exercise", để notebook_gen.txt (KHỐI 3,
# bước 3.2) biết chính xác phần nào cần giải thích lý thuyết riêng.
def build_modules_block(path: LearningPath) -> str:
    blocks = []
    for m in path.modules:
        theory_only = _theory_only_concepts(m)
        lines = [
            f'module_id = "{m.module_id}" | title = "{m.title}"',
            f'  objective = "{m.objective}"',
            f"  concepts (đầy đủ) = {m.concepts}",
            f"  concepts lý thuyết riêng (KHÔNG có exercise nào dùng, PHẢI giải thích ở bước 3.2) = {theory_only}",
            f"  estimated_minutes = {m.estimated_minutes}",
        ]
        if m.planned_exercises:
            lines.append("  planned_exercises:")
            for ex in m.planned_exercises:
                lines.append(
                    f'    - exercise_id = "{ex.exercise_id}" | title = "{ex.title}" | '
                    f'type = "{ex.type}" | difficulty = {ex.difficulty} | '
                    f"has_starter_code = {ex.has_starter_code} | concepts = {ex.concepts}"
                )
                lines.append(f'      prompt = "{ex.prompt}"')
                lines.append(f'      expected_check = "{ex.expected_check or ""}"')
        else:
            lines.append(
                "  planned_exercises: [] (module KHÔNG có bài tập — chỉ cần markdown lý "
                "thuyết ở bước 3.2, và cell code demo ở bước 3.3 NẾU cần thiết cho pipeline "
                "chạy được, xem quy tắc trong notebook_gen.txt)"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# Xây dựng prompt để gửi cho model
def build_prompt_notebook_gen(
    path : LearningPath,
    profile : LearnerProfile,
    prior_feedback : Optional[str] = None,
) -> str:
    template_notebook_gen = load_prompt()

    # Số bài tập mục tiêu: ưu tiên total_planned_exercises của LearningPath (đã được
    # Curriculum Agent chốt và ép theo khoảng level ở validate_and_adjust);
    # nếu path chưa có planned_exercises, fallback về constraints.num_exercises
    # đã ép theo level (clamp_num_exercises) để phù hợp với Difficulty-Fit.
    target_exercises = path.total_planned_exercises or clamp_num_exercises(
        path.level, profile.constraints.num_exercises
    )
    level_name = LEVEL_NAMES.get(path.level , str(path.level))
    lowest, highest = get_exercise_range(path.level)

    # Đếm số cell TODO
    todo_count_hint = (
        f"Level {path.level} ({level_name}) -> tổng số cell bài tập có TODO ở KHỐI 3 "
        f"PHẢI nằm trong khoảng [{lowest}, {highest}] bài (beginner: 2-3 bài, intermediate: "
        f"4-5 bài). Số bài tập cụ thể nên khớp với target_exercises = {target_exercises} "
        "nếu số này đã nằm trong khoảng cho phép; nếu không, ưu tiên khoảng cho phép theo level."
    )

    prompt = template_notebook_gen
    prompt = prompt.replace("{topic}", str(path.topic))
    prompt = prompt.replace("{final_level}", level_name)
    prompt = prompt.replace("{num_exercises_planned}", str(target_exercises))
    prompt = prompt.replace("{modules_block}", build_modules_block(path))
    prompt = prompt.replace("{dataset_info_block}", build_dataset_info_block(path.topic))
    prompt = prompt.replace("{problem_type_block}", build_problem_type_block(path.topic))
    prompt = prompt.replace("{todo_count_hint}", todo_count_hint)
    prompt = prompt.replace("{prior_feedback_block}", check_prior_feedback(prior_feedback))

    return prompt
    
# ===============
# 2a. Xử lý JSON
# ===============
def processing_json(raw_text: str) -> dict:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return json.loads(text)


_PANDAS_POSITIONAL_SLICE = re.compile(
    r"\b(?P<name>X_(?:train|test))\s*\[\s*:\s*,\s*(?P<column>[^\]]+)\]"
)
_SEABORN_IMPORT = re.compile(
    r"(?m)^[ \t]*import[ \t]+seaborn(?:[ \t]+as[ \t]+(?P<alias>[A-Za-z_]\w*))?[ \t]*\n?"
)
_MARKDOWN_ESCAPED_NEWLINE = re.compile(
    r"\\n(?=(?:\\n|#{1,6}\s|[-*+]\s|\d+[.)]\s|\*{1,2}|`|[A-ZÀ-Ỹ]))"
)
_MARKDOWN_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_MARKDOWN_BLOCK_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)


def normalise_markdown_newlines(source: str) -> str:
    """Turn LLM-escaped line breaks into Markdown breaks without touching LaTeX."""
    return _MARKDOWN_ESCAPED_NEWLINE.sub("\n", source)


def normalise_markdown_source(source: str) -> str:
    """Normalise line breaks and use math delimiters supported by VS Code/Jupyter."""
    source = normalise_markdown_newlines(source)
    source = _MARKDOWN_BLOCK_MATH.sub(lambda match: f"$${match.group(1)}$$", source)
    return _MARKDOWN_INLINE_MATH.sub(lambda match: f"${match.group(1)}$", source)


def _normalise_source_if_needed(source: str, index: int) -> str:
    """Compile source and repair only the known double-escaped newline failure."""
    try:
        compile(source, f"<notebook-cell-{index}>", "exec")
        return source
    except SyntaxError as original_error:
        if "\\n" in source:
            repaired = source.replace("\\n", "\n")
            try:
                compile(repaired, f"<notebook-cell-{index}>", "exec")
                return repaired
            except SyntaxError:
                pass
        raise ValueError(
            f"cells[{index}] có Python syntax lỗi: {original_error.msg} "
            f"(dòng {original_error.lineno})"
        ) from original_error


def _replace_unsupported_seaborn(source: str, index: int) -> str:
    """Replace the common heatmap use; reject other seaborn-dependent code."""
    aliases = [match.group("alias") or "seaborn" for match in _SEABORN_IMPORT.finditer(source)]
    if not aliases and not re.search(r"(?m)^\s*from\s+seaborn\b", source):
        return source

    converted = source
    replaced_heatmap = False
    for alias in aliases:
        heatmap = re.compile(
            rf"(?m)^(?P<indent>[ \t]*){re.escape(alias)}\.heatmap\(\s*"
            r"(?P<matrix>[A-Za-z_]\w*)[^\n]*\)[ \t]*$"
        )

        def replace_call(match: re.Match[str]) -> str:
            nonlocal replaced_heatmap
            replaced_heatmap = True
            indent = match.group("indent")
            matrix = match.group("matrix")
            return (
                f"{indent}plt.imshow({matrix}, cmap='Blues', interpolation='nearest')\n"
                f"{indent}plt.colorbar()"
            )

        converted = heatmap.sub(replace_call, converted)

    converted = _SEABORN_IMPORT.sub("", converted)
    if re.search(r"(?m)^\s*from\s+seaborn\b", converted) or any(
        re.search(rf"\b{re.escape(alias)}\.", converted) for alias in aliases
    ):
        raise ValueError(
            f"cells[{index}] dùng seaborn ngoài heatmap đơn giản; sandbox không có seaborn"
        )
    if replaced_heatmap and not re.search(r"(?m)^\s*(?:import\s+matplotlib\.pyplot\s+as\s+plt|from\s+matplotlib)", converted):
        converted = "import matplotlib.pyplot as plt\n" + converted
    return converted


def _guard_exercise_asserts(source: str) -> str:
    """Keep starter notebooks executable while TODO answers are still empty."""
    tree = ast.parse(source)
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        return source
    already_catches_assert = re.search(
        r"except\s+(?:AssertionError|Exception|BaseException|\([^\n)]*AssertionError)",
        source,
    )
    if already_catches_assert:
        return source
    return (
        "try:\n"
        f"{textwrap.indent(source, '    ')}\n"
        "except (AssertionError, TypeError, ValueError, NameError, NotImplementedError) "
        "as exercise_error:\n"
        "    print(f'Bài tập chưa hoàn thành: {exercise_error}')"
    )


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for item in target.elts for name in _target_names(item)}
    return set()


def _module_definitions(source: str) -> set[str]:
    """Return names created at cell top level, excluding repeated imports."""
    names: set[str] = set()
    for statement in ast.parse(source).body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(statement.name)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                names.update(_target_names(target))
        elif isinstance(statement, ast.AnnAssign):
            names.update(_target_names(statement.target))
        elif isinstance(statement, ast.AugAssign):
            names.update(_target_names(statement.target))
    return names


def _rename_todo_collisions(cells: list[dict]) -> list[dict]:
    """Keep TODO starter variables from overwriting completed demo variables."""
    established: set[str] = set()
    todo_renames: list[dict[str, str]] = []
    assert_indices: list[int] = []

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        tree = ast.parse(source)
        if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
            assert_indices.append(index)
        definitions = _module_definitions(source)
        if "TODO" not in source:
            established.update(definitions)
            continue

        ordinal = len(todo_renames) + 1
        collisions = sorted(definitions & established)
        rename_map = {
            name: f"{name}_exercise_{ordinal}" for name in collisions
        }
        for old_name, new_name in rename_map.items():
            source = re.sub(rf"\b{re.escape(old_name)}\b", new_name, source)
        cell["source"] = source
        todo_renames.append(rename_map)

    for rename_map, assert_index in zip(todo_renames, assert_indices):
        source = cells[assert_index]["source"]
        for old_name, new_name in rename_map.items():
            source = re.sub(rf"\b{re.escape(old_name)}\b", new_name, source)
        cells[assert_index]["source"] = source
    return cells


def prepare_generated_cells(cells: list[dict]) -> list[dict]:
    """Apply bounded runtime-safety fixes and compile every generated code cell."""
    prepared: list[dict] = []
    for index, cell in enumerate(cells):
        updated = dict(cell)
        if updated.get("cell_type") == "markdown":
            updated["source"] = normalise_markdown_source(updated["source"])
            prepared.append(updated)
            continue
        if updated.get("cell_type") != "code":
            prepared.append(updated)
            continue

        source = _normalise_source_if_needed(updated["source"], index)
        source = _PANDAS_POSITIONAL_SLICE.sub(
            r"\g<name>.iloc[:, \g<column>]", source
        )
        source = _replace_unsupported_seaborn(source, index)
        source = _guard_exercise_asserts(source)
        try:
            compile(source, f"<notebook-cell-{index}>", "exec")
        except SyntaxError as error:
            raise ValueError(
                f"cells[{index}] có Python syntax lỗi sau preflight: {error.msg} "
                f"(dòng {error.lineno})"
            ) from error
        updated["source"] = source
        prepared.append(updated)

    prepared = _rename_todo_collisions(prepared)
    for index, cell in enumerate(prepared):
        if cell.get("cell_type") == "code":
            try:
                compile(cell["source"], f"<notebook-cell-{index}>", "exec")
            except SyntaxError as error:
                raise ValueError(
                    f"cells[{index}] lỗi sau khi bảo vệ biến TODO: {error.msg}"
                ) from error
    return prepared


def _request_notebook_cells(prompt: str, session_id: str) -> list[dict]:
    """Generate notebook JSON without provider-side JSON enforcement.

    Groq can reject a long otherwise-useful generation with
    ``json_validate_failed`` before returning any text.  NotebookGen therefore
    receives plain text, parses it locally, and regenerates once when the JSON
    is incomplete or has the wrong cell contract.
    """
    current_prompt = prompt
    last_error: Exception | None = None
    last_raw = ""

    for generation_attempt in range(1, NOTEBOOK_JSON_ATTEMPTS + 1):
        request_max_tokens = notebook_request_max_tokens(current_prompt)
        raw, _usage = call_text(
            current_prompt,
            session_id=session_id,
            max_tokens=request_max_tokens,
            json_mode=False,
            reasoning_effort=NOTEBOOK_REASONING_EFFORT,
            include_reasoning=False,
        )
        if _usage is not None:
            print(
                "[Notebook Gen Agent] "
                f"LLM lần {generation_attempt}: input={_usage.input_tokens}, "
                f"output={_usage.output_tokens}, finish={_usage.finish_reason}"
            )
        last_raw = raw
        try:
            data = processing_json(raw)
            if not isinstance(data, dict):
                raise TypeError("output phải là JSON object")
            cells = data.get("cells")
            if not isinstance(cells, list):
                raise TypeError("field 'cells' phải là list")
            for index, cell in enumerate(cells):
                if not isinstance(cell, dict):
                    raise TypeError(f"cells[{index}] phải là object")
                if cell.get("cell_type") not in {"markdown", "code"}:
                    raise ValueError(
                        f"cells[{index}].cell_type phải là 'markdown' hoặc 'code'"
                    )
                if not isinstance(cell.get("source"), str):
                    raise TypeError(f"cells[{index}].source phải là string")
            return prepare_generated_cells(cells)
        except (json.JSONDecodeError, TypeError, ValueError) as error_json:
            last_error = error_json
            if generation_attempt == NOTEBOOK_JSON_ATTEMPTS:
                break
            current_prompt = (
                f"{prompt}\n\n"
                "LẦN TRƯỚC OUTPUT SAI JSON, SAI CONTRACT HOẶC CODE KHÔNG QUA PREFLIGHT. "
                f"Lỗi: {error_json}. Hãy sinh lại từ đầu, chỉ trả một JSON object "
                "hợp lệ có field cells; code phải compile, không dùng seaborn, dùng .iloc khi "
                "lấy cột của X_train/X_test và không để assert bài TODO làm notebook dừng."
            )

    raise ValueError(
        "Notebook Generator trả JSON không hợp lệ sau "
        f"{NOTEBOOK_JSON_ATTEMPTS} lần: {last_error}. Raw cuối: {last_raw[:500]}"
    ) from last_error


# ============================================================================
# 2b. Validate cells sinh ra (chỉ cảnh báo qua print, KHÔNG tự sửa nội dung)
# ============================================================================
def validate_cells(cells: list[dict], path: LearningPath) -> list[str]:
    warnings: list[str] = []

    code_sources = [c.get("source", "") for c in cells if c.get("cell_type") == "code"]
    joined_code_sources = "\n".join(code_sources)

    # Tổng số cell TODO phải nằm trong khoảng cho phép theo level.
    todo_cells = [t for t in code_sources if "TODO" in t]
    lowest, highest = get_exercise_range(path.level)
    if not (lowest <= len(todo_cells) <= highest):
        warnings.append(
            f"Số cell có TODO ({len(todo_cells)}) NẰM NGOÀI khoảng cho phép theo level "
            f"{path.level} ({lowest}-{highest}) — kiểm tra lại KHỐI 3, có thể model sinh thừa/thiếu bài."
        )

    # Số cell assert (KHỐI 4) nên >= số cell TODO (mỗi bài tập có ít nhất 1 assert).
    assert_cells = [t for t in code_sources if "assert" in t]
    if len(assert_cells) < len(todo_cells):
        warnings.append(
            f"Số cell assert ({len(assert_cells)}) ít hơn số cell TODO ({len(todo_cells)}) "
            "— có bài tập ở KHỐI 3 đang thiếu cell kiểm tra tương ứng ở KHỐI 4."
        )

    # Nếu topic là unsupervised (K-Means), không được có train_test_split.
    if get_problem_type(path.topic) == UNSUPERVISED_CLUSTERING:
        if "train_test_split" in joined_code_sources:
            warnings.append(
                "Topic là unsupervised/phân cụm nhưng code vẫn chứa 'train_test_split' "
                "cần sửa lại prompt hoặc regen."
            )

    return warnings


# ===========================================================================================
# 3. Chèn dataset thật thay thế cho "# DATASET_INJECTION_PLACEHOLDER" và build file .ipynb
# ===========================================================================================
def inject_dataset(cells : list[dict], topic : str, seed : int) -> list[dict]:
    # Tìm đúng vị trí cell chứa placeholder
    placeholder_idx: Optional[int] = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and DATASET_PLACEHOLDER in c.get("source", ""):
            placeholder_idx = i
            break

    if placeholder_idx is None:
        print(
            f"Thiếu placeholder '{DATASET_PLACEHOLDER}' ở cell chuẩn bị dữ liệu (bắt buộc để "
            "hệ thống chèn dataset thật). KHÔNG chèn dataset — trả nguyên cells để không làm "
            "vỡ notebook, nhưng notebook này gần như chắc chắn sẽ lỗi khi chạy vì thiếu dữ liệu."
        )
        return cells

    eda_cells_raw = get_dataset_code(topic, seed)

    # Phòng hờ trường hợp interface bị đổi lại thành str (bản cũ) — vẫn xử lý được, không crash.
    if isinstance(eda_cells_raw, str):
        injected_cells = [{"cell_type": "code", "source": eda_cells_raw}]
    else:
        injected_cells = []
        for item in eda_cells_raw:
            title = item.get("title")
            if title:
                injected_cells.append({"cell_type": "markdown", "source": f"### {title}"})
            injected_cells.append({"cell_type": "code", "source": item.get("code", "")})

    # Thay thế đúng 1 cell placeholder bằng toàn bộ các cell EDA/load/split thật
    return cells[:placeholder_idx] + injected_cells + cells[placeholder_idx + 1:]


def build_notebook_file(cells : list[dict], notebook_path : Path) -> None:
    # Tạo notebook object mới
    notebook = new_notebook()

    for c in cells:
        if c["cell_type"] == "markdown":
            notebook.cells.append(new_markdown_cell(c["source"]))
        else:
            notebook.cells.append(new_code_cell(c["source"]))

    # Kiểm tra lại đường dẫn
    notebook_path.parent.mkdir(parents = True, exist_ok = True)

    # Tạo file jupyter notebook đúng cấu trúc và lưu ở đường dẫn
    nbformat.write(notebook , str(notebook_path))


# ===============
# 4. Hàm chính
# ===============

def run_notebook_gen(
    path: LearningPath,
    profile: LearnerProfile,
    attempt: int = 1,
    prior_feedback: Optional[str] = None,
) -> str:

    # Lệnh prompt (Kèm prior_feedback để sửa nếu có)
    prompt = build_prompt_notebook_gen(path, profile, prior_feedback)
    cells = _request_notebook_cells(prompt, profile.session_id)

    for w in validate_cells(cells, path):
        print(f"[Notebook Gen Agent] cảnh báo : {w}")

    # dataset_seed nằm ở profile
    cells = inject_dataset(cells, path.topic, profile.dataset_seed)

    # Dataset Injector là code local nhưng vẫn compile chung trước khi ghi notebook.
    cells = prepare_generated_cells(cells)

    notebook_path = OUTPUT_DIR / f"{profile.session_id}_attempt{attempt}.ipynb"

    # Sinh notebook và lưu tại "notebook_path"
    build_notebook_file(cells , notebook_path)

    return str(notebook_path)

# ------------------------
# 5. Test nhanh bằng mock
# ------------------------
if __name__ == "__main__":
    from tests.mocks import MOCK_PATH, MOCK_PROFILE
    nb_path = run_notebook_gen(MOCK_PATH, MOCK_PROFILE, attempt = 1, prior_feedback = None)
    print("Notebook đã tạo tại:", nb_path)
