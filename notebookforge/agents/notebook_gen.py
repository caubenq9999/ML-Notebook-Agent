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

from schemas import LearnerProfile, LearningPath

from llm_client import call_text 

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "notebook_gen.txt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_notebooks"
DATASET_PLACEHOLDER = "# DATASET_INJECTION_PLACEHOLDER"

# level trong schema là số 1/2 — map sang chữ để prompt dễ đọc hơn cho LLM
LEVEL_NAMES = {1: "beginner" , 2: "intermediate"}

# -------------------
# 1. Xây dựng prompt
# -------------------
def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding = "utf-8")

def check_prior_feedback(prior_feedback : Optional[str]) -> str:
    if not prior_feedback:
        return ""
    
    return (
        "<prior_feedback>\n"
        "Notebook ở lần sinh trước đã bị Verifier từ chối vì lý do bên dưới. "
        "Hãy sửa lại cho đúng, không lặp lại lỗi cũ:\n"
        f"{prior_feedback}\n"
        "</prior_feedback>"
    )


# Chuyển modules[] của LearningPath thành các dòng text để đưa vào prompt.
def summarize_modules(path : LearningPath) -> str:
    text_module = []        # list rỗng chứa thông tin các module

    # Lấy thông tin các module và sắp xếp theo order
    for current_module in path.modules:
        concepts = ", ".join(current_module.concepts)
        text = (
            f'module_id = "{current_module.module_id}" | title = "{current_module.title}" | objective = "{current_module.objective}" | '
            f"concepts = [{concepts}] | estimated_minutes = {current_module.estimated_minutes}"
        )
        if current_module.planned_exercises:
            for ex in current_module.planned_exercises:
                text += (
                    f'\n  - Bài tập: exercise_id = "{ex.exercise_id}" | '
                    f'title = "{ex.title}" | prompt = "{ex.prompt}" | '
                    f'expected_check = "{ex.expected_check or ""}"'
                )
        text_module.append(text)

    # In ra thông tin các module khi chuyển qua dạng text, mỗi module một dòng
    return "\n".join(text_module)


# Xây dựng prompt để gửi cho model
def build_prompt_notebook_gen(
    path : LearningPath,
    profile : LearnerProfile,
    prior_feedback : Optional[str] = None,
) -> str:
    template_notebook_gen = load_prompt()

    target_exercises = path.total_planned_exercises or profile.constraints.num_exercises
    level_name = LEVEL_NAMES.get(path.level , str(path.level))

    prompt = template_notebook_gen
    prompt = prompt.replace("{topic}", str(path.topic))
    prompt = prompt.replace("{final_level}", level_name)
    prompt = prompt.replace("{num_exercises_planned}", str(target_exercises))
    prompt = prompt.replace("{modules_summary}", summarize_modules(path))
    prompt = prompt.replace("{prior_feedback_block}", check_prior_feedback(prior_feedback))

    return prompt
    
# ---------------
# 2. Xử lý JSON
# ---------------
def processing_json(raw_text: str) -> dict:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return json.loads(text)

# -----------------------------------------------------------------------------------------
# 3. Chèn dataset thật thay thế cho "# DATASET_INJECTION_PLACEHOLDER" và build file .ipynb
# -----------------------------------------------------------------------------------------
def inject_dataset(cells : list[dict], topic : str, seed : int) -> list[dict]:
    # Lấy phần "source" của các code cell (code cell không có "source" thì source = "")
    code_sources = [c.get("source" , "") for c in cells if c.get("cell_type") == "code"]
    all_source_codecell = "\n".join(code_sources)
    if DATASET_PLACEHOLDER not in all_source_codecell:
        print(f"Thiếu placeholder '{DATASET_PLACEHOLDER}' ở cell chuẩn bị dữ liệu (bắt buộc để hệ thống chèn dataset thật).")
        return ""

    # Trả về đoạn code chứa dataset và được chia ra thành các tập train, test
    real_dataset = get_dataset_code(topic , seed)

    injected = False
    for c in cells:
        # Đã có DATASET_PLACEHOLDER để thay thế bằng dataset thật
        if c.get("cell_type") == "code" and DATASET_PLACEHOLDER in c.get("source", ""):
            c["source"] = c["source"].replace(DATASET_PLACEHOLDER , real_dataset)
            injected = True

    # Nếu chưa có DATASET_PLACEHOLDER -> thêm 1 cell mới cho dataset thật
    if not injected:
        cells.insert(1, {"cell_type": "code", "source": real_dataset})

    return cells


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


# -------------
# 4. Hàm chính
# -------------

def run_notebook_gen(
    path: LearningPath,
    profile: LearnerProfile,
    attempt: int = 1,
    prior_feedback: Optional[str] = None,
) -> str:

    prompt = build_prompt_notebook_gen(path, profile, prior_feedback)
    raw, _usage = call_text(
        prompt,
        session_id = profile.session_id,
        json_mode = True,
    )

    try:
        data = processing_json(raw)
        cells = data["cells"]
    except (json.JSONDecodeError, KeyError, TypeError) as error_json:
        last_error = f"JSON không hợp lệ hoặc thiếu field cells: {error_json}. Raw: {raw[:500]}"

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
