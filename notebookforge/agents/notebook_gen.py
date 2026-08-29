"""
agents/notebook_gen.py — HỢP
"""
from __future__ import annotations

import json
import re
import sys
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

from llm_client import call_text 

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
            "Một số cột có giá trị 0 bất hợp lý về mặt sinh lý(Cholesterol=0 được "
            "impute bằng median theo HeartDisease, 1 dòng có RestingBP=0 đã bị loại bỏ); "
            "Các biến có kiểu dữ liệu là string được mã hoá bằng One-Hot Encoding; "
            "Sau cùng, chia tập train/test và chuẩn hoá lại bằng StandardScaler chỉ trên tập train."
        ),
    },
    "decision_tree": {
        "name": "Red Wine Quality Dataset",
        "kaggle": "kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009",
        "size": "1.599 dòng, 12 cột (11 đặc trưng hoá lý + 1 điểm chất lượng)",
        "target": (
            "quality — điểm chất lượng rượu, dùng làm nhãn đa lớp (multi-class)"
        ),
        "description": (
            "11 đặc trưng hoá lý của rượu vang đỏ (fixed acidity, volatile acidity, citric "
            "acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, "
            "sulphates, alcohol) dùng để phân loại chất lượng rượu ('quality');"
            "các đặc trưng có ngưỡng cắt rõ ràng theo từng khoảng giá trị, phù hợp minh hoạ Decision Tree"
        ),
        "preprocessing_note": (
            "Các dòng trùng lặp(~240 dòng) bị loại bỏ bằng drop_duplicates() trước khi chia "
            "train/test (stratify theo 'quality'). Nhãn 'quality' giữ nguyên dạng gốc (đa lớp, "
            "phân bố bị lệch giữa các lớp) - nên gộp thành các nhóm low (<=5), medium (=6), high(>=7); "
            "ngưỡng accuracy 'tốt' nên được xác định qua thực nghiệm trên dataset này."
        ),
    },
    "kmeans": {
        "name": "Mall Customer Segmentation Dataset",
        "kaggle": "kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python",
        "size": "200 dòng, 5 cột (sau khi bỏ cột định danh)",
        "target": None,  # không có nhãn thật — bài toán không giám sát
        "description": (
            "Thông tin khách hàng của một trung tâm thương mại: Gender, Age, "
            "Annual Income (k$), Spending Score (1-100) — sau khi đã loại bỏ cột định danh CustomerID. "
            "KHÔNG có nhãn phân khúc sẵn, cần phân cụm khách hàng theo hành vi chi tiêu."
        ),
        "preprocessing_note": (
            "Cột CustomerID đã bị loại bỏ vì không mang thông tin phân cụm; biến Gender đã"
            "được mã hoá bằng One-Hot Encoding; dữ liệu được chuẩn hoá lại bằng StandardScaler và xáo trộn (shuffle) "
            "trước khi phân cụm."
        ),
    },
}

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
    return str(topic).strip().lower().replace(" ", "_").replace("-", "_")

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
            else "Bài toán học không giám sát / phân cụm"
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
            "Loại bài toán: KHÔNG GIÁM SÁT / PHÂN CỤM.\n"
            "Metric đánh giá hợp lệ: inertia (WCSS), silhouette_score, davies_bouldin_score nếu có trong concepts. "
            "TUYỆT ĐỐI KHÔNG dùng accuracy/precision/recall/f1 và train_test_split ở bất kỳ đâu"
        )
    note_multiclass = ""
    if _normalize_topic(topic) == "decision_tree":
        note_multiclass = (
            "Target 'quality' của dataset này là ĐA LỚP (~6 lớp: 3-8), "
            "nếu dùng precision/recall/f1/roc_auc, PHẢI truyền tham số average='weighted' (hoặc 'macro') cho precision/recall/f1, "
            "và multi_class='ovr' cho roc_auc_score; accuracy_score dùng bình thường không cần tham số thêm."
            "Nếu không, dựa trên số điểm có thể gộp thành các nhóm low (<=5), medium (=6), high(>=7)"
        )
    return (
        "Loại bài toán: CÓ GIÁM SÁT / PHÂN LOẠI.\n"
        "Biến có sẵn sau KHỐI 2(do hệ thống chèn): X_train, X_test, y_train, y_test "
        "Metric đánh giá hợp lệ: accuracy, precision, recall, f1, roc_auc, confusion matrix, feature_importances_/plot_tree "
        "nếu có concepts tương ứng."
        f"{note_multiclass}"
    )


# =====================
# 1. Xây dựng prompt
# =====================
def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding = "utf-8")

def check_prior_feedback(prior_feedback : Optional[str]) -> str:
    if not prior_feedback:
        return ""
    
    return (
        "<prior_feedback>\n"
        "Notebook ở lần sinh trước đã bị Verifier từ chối vì lý do bên dưới. "
        "BẮT BUỘC phải sửa đúng chỗ lỗi, không lặp lại lỗi cũ, và "
        "Phần KHÔNG LIÊN QUAN tới lỗi thì giữ nguyên tinh thần nội dung cũ, không cần viết lại từ đầu:\n"
        f"{prior_feedback}\n"
        "</prior_feedback>"
    )

# Lấy các concepts không có trong "planned_exercises", giải quyết phần 3.2
def _theory_only_concepts(module: Module) -> list[str]:
    used_in_exercises: set[str] = set()
    for ex in module.planned_exercises:
        used_in_exercises.update(ex.concepts)
    return [c for c in module.concepts if c not in used_in_exercises]

# Chuyển modules[] của LearningPath thành text chi tiết đưa vào prompt 
# tách rõ "concepts lý thuyết riêng" khỏi "concepts đã có exercise" giải quyết phần 3.2
def build_modules_block(path: LearningPath) -> str:
    blocks = []
    for m in path.modules:
        theory_only = _theory_only_concepts(m)
        lines = [
            f'module_id="{m.module_id}" | title="{m.title}"',
            f' objective="{m.objective}"',
            f" concepts(đầy đủ)={m.concepts}",
            f" concepts lý thuyết riêng(Do không có exercise nào dùng, nên PHẢI giải thích theo hướng dẫn bước 3.2)={theory_only}",
            f" estimated_minutes={m.estimated_minutes}",
        ]
        if m.theory_context:
            lines.append(
                " theory_context (trích Knowledge Base THẬT qua RAG — PHẢI bám sát khi giải "
                "thích lý thuyết ở bước 3.2/3.4: diễn đạt lại tự nhiên nhưng KHÔNG đổi công "
                "thức/thuật ngữ/số liệu/code mẫu so với bản gốc; concept nào KHÔNG có trong "
                "danh sách này thì giải thích theo kiến thức chuẩn, không tự bịa số liệu cụ thể):"
            )
            for concept, text in m.theory_context.items():
                lines.append(f'  --- theory_context["{concept}"] ---')
                lines.append(text)
        if m.planned_exercises:
            lines.append(" planned_exercises:")
            for ex in m.planned_exercises:
                lines.append(
                    f'  - exercise_id="{ex.exercise_id}"|title="{ex.title}"|'
                    f'type="{ex.type}"|difficulty={ex.difficulty}|'
                    f"has_starter_code={ex.has_starter_code}|concepts={ex.concepts}"
                )
                lines.append(f'  prompt="{ex.prompt}"')
                lines.append(f'  expected_check="{ex.expected_check or ""}"')
        else:
            lines.append(
                "  planned_exercises: [] (module KHÔNG có bài tập — chỉ cần markdown lý "
                "thuyết ở bước 3.2 + BẮT BUỘC 1 cell code demo ở bước 3.3, không có ngoại lệ)"
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

    # Số bài tập mục tiêu: ưu tiên total_planned_exercises của LearningPath
    # và ép theo khoảng level + thông báo thay đổi ở validate_and_adjust; 
    # nếu toàn bộ module vô tình không có planned_exercises nào,
    # fallback về constraints.num_exercises đã ép theo level (clamp_num_exercises).
    target_exercises = path.total_planned_exercises or clamp_num_exercises(
        path.level, profile.constraints.num_exercises
    )
    level_name = LEVEL_NAMES.get(path.level , str(path.level))
    lowest, highest = get_exercise_range(path.level)

    # Đếm số cell TODO
    todo_count_hint = (
        f"Level {path.level} ({level_name}) -> tổng số cell bài tập có TODO ở KHỐI 3 "
        f"phải nằm trong khoảng [{lowest},{highest}] bài "
        f"Số bài tập cụ thể nên khớp với target_exercises = {target_exercises} "
        f"nếu số này đã nằm trong khoảng cho phép; nếu ít hơn {lowest}, tạo {lowest} bài; nếu lớn hơn {highest}, tạo {highest} bài"
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


# ============================================================================
# 2b. Validate cells sinh ra (chỉ cảnh báo qua print, KHÔNG tự sửa nội dung)
# ============================================================================
def validate_cells(cells: list[dict], path: LearningPath) -> list[str]:
    warnings: list[str] = []

    code_sources = [c.get("source", "") for c in cells if c.get("cell_type") == "code"]
    joined_code_sources = "\n".join(code_sources)

    # Tổng số cell bài tập phải nằm trong khoảng cho phép theo level.
    todo_cells = [t for t in code_sources if "TODO" in t]
    lowest, highest = get_exercise_range(path.level)
    if not (lowest <= len(todo_cells) <= highest):
        warnings.append(
            f"Số cell bài tập là: ({len(todo_cells)}), NẰM NGOÀI khoảng cho phép theo level "
            f"{path.level} [{lowest},{highest}] — kiểm tra lại KHỐI 3."
        )

    # Số cell assert (KHỐI 4) nên >= số cell TODO (mỗi bài tập có ít nhất 1 assert).
    assert_cells = [t for t in code_sources if "assert" in t]
    if len(assert_cells) < len(todo_cells):
        warnings.append(
            f"Số cell assert ({len(assert_cells)}) ít hơn số cell bài tập ({len(todo_cells)}) "
            "— có bài tập đang thiếu cell kiểm tra tương ứng."
        )

    # Nếu topic là unsupervised (K-Means), không được có train_test_split.
    if get_problem_type(path.topic) == UNSUPERVISED_CLUSTERING:
        if "train_test_split" in joined_code_sources:
            warnings.append(
                "Topic là unsupervised/phân cụm nhưng code vẫn chứa 'train_test_split', "
                "cần siết chặt lại prompt."
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
            f"Thiếu placeholder '{DATASET_PLACEHOLDER}' ở cell chuẩn bị dữ liệu"
            f"KHÔNG chèn dataset — Yêu cầu sinh lại notebook và PHẢI CÓ {DATASET_PLACEHOLDER}."
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

# Xử lý JSON/ValidationError
class NotebookGenerationError(Exception):
    """Raised khi Notebook Gen Agent không tạo được JSON cells hợp lệ sau tất cả các lần thử."""

def run_notebook_gen(
    path: LearningPath,
    profile: LearnerProfile,
    attempt: int = 1,
    prior_feedback: Optional[str] = None,
) -> str:
    
    # Lệnh prompt (kèm prior_feedback để sửa nếu có, và kèm lỗi JSON của lần thử
    prompt = build_prompt_notebook_gen(path, profile, prior_feedback)

    raw = ""
    try:
        raw, _usage = call_text(
            prompt,
            session_id = profile.session_id,
            json_mode = True,
            reasoning_effort = "low"
        )
        data = processing_json(raw)
        cells = data["cells"]
        if not isinstance(cells, list) or len(cells) == 0:
            raise ValueError("Field 'cells' rỗng hoặc không phải list")
        
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error_json:
        error_message = f"{type(error_json).__name__}: {error_json}. Raw (rút gọn): {raw[:500]}"
        print(
            f"[Notebook Gen Agent] Lỗi JSON: {error_message}"
        )

        raise NotebookGenerationError(
            f"Notebook Gen Agent tạo JSON không hợp lệ. "
            f"Lỗi: {error_message}"
        )

    for w in validate_cells(cells, path):
        print(f"[Notebook Gen Agent] cảnh báo : {w}")

    # dataset_seed nằm ở profile
    cells = inject_dataset(cells, path.topic, profile.dataset_seed)

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
