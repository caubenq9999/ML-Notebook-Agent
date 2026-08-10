"""
agents/notebook_gen.py — HỢP
"""
from __future__ import annotations

import json
import os
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

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "notebook_gen.txt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_notebooks"
MODEL_NAME = "gemini-2.0-flash"
MAX_LLM_RETRIES = 3  # số lần thử LLM tối đa
DATASET_PLACEHOLDER = "# DATASET_INJECTION_PLACEHOLDER"

# level trong schema là số 1/2 — map sang chữ để prompt dễ đọc hơn cho LLM
LEVEL_NAMES = {1: "beginner", 2: "intermediate"}

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
    last_error : Optional[str] = None,
) -> str:
    template_notebook_gen = load_prompt()

    target_exercises = path.total_planned_exercises or profile.constraints.num_exercises
    level_name = LEVEL_NAMES.get(path.level, str(path.level))

    prompt = template_notebook_gen
    prompt = prompt.replace("{topic}", str(path.topic))
    prompt = prompt.replace("{final_level}", level_name)
    prompt = prompt.replace("{num_exercises_planned}", str(target_exercises))
    prompt = prompt.replace("{modules_summary}", summarize_modules(path))
    prompt = prompt.replace("{prior_feedback_block}", check_prior_feedback(prior_feedback))
    
    if last_error:
        prompt += (
            "\n# LỖI Ở LẦN TRẢ LỜI TRƯỚC — Bắt buộc sửa\n"
            f"{last_error}\n"
            "Hãy trả lại đúng định dạng JSON theo yêu cầu ở trên."
        )

    return prompt


# ---------------------
# 2. Gọi LLM (Gemini)
# ---------------------
def call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")

    # Dùng api_client
    from google import genai
    from google.genai import types

    client = genai.Client(api_key = api_key)

    response = client.models.generate_content(
        model = MODEL_NAME,
        contents = prompt,
        config = types.GenerateContentConfig(
            temperature = 0.3,
            response_mime_type = "application/json",
        ),
    )

    return response.text

# Nếu chưa có API key
def dry_run_response() -> str:
    return json.dumps(
        {
            "cells" : [
                {
                    "cell_type" : "markdown",
                    "source": (
                        "# Logistic Regression\n\n"
                        "Tổng quan: Logistic Regression là thuật toán học có giám sát, dùng để "
                        "giải quyết các bài toán phân loại nhị phân, xử lý theo pipeline: "
                        "Đầu vào X -> tổng tuyến tính z -> hàm sigmoid(z) -> ngưỡng decision -> "
                        "dự đoán.\n\n"
                        "Mục tiêu: Hiểu ý nghĩa và công thức của hàm sigmoid."
                    ),
                },
                {
                    "cell_type" : "markdown",
                    "source" : "## Chuẩn bị dữ liệu\nDataset phân loại nhị phân (X : đặc trưng, y : nhãn 0/1).",
                },
                {"cell_type" : "code", "source" : "# DATASET_INJECTION_PLACEHOLDER"},
                {
                    "cell_type" : "markdown",
                    "source" : (
                        "## Part_1 - Module 1: Sigmoid\nHàm sigmoid biến đổi z (số thực bất kỳ) thành xác "
                        "suất trong khoảng (0,1) theo công thức sigmoid(z) = 1 / (1 + e^(-z)). "
                        "Đây là hàm đơn điệu tăng. Ngưỡng 0.5 (ứng với z = 0) dùng để quyết định nhãn."
                    ),
                },
                {
                    "cell_type" : "code",
                    "source" : (
                        "import numpy as np\n"
                        "import matplotlib.pyplot as plt\n\n"
                        "def sigmoid_demo(z):\n"
                        "    return 1 / (1 + np.exp(-z))\n\n"
                        "z_values = np.linspace(-10, 10, 200)\n"
                        "plt.plot(z_values, sigmoid_demo(z_values))\n"
                        "plt.axhline(0.5, color = 'gray', linestyle = '--')\n"
                        "plt.xlabel('z'); plt.ylabel('sigmoid(z)')\n"
                        "plt.title('Đường cong hàm Sigmoid')\n"
                        "plt.show()"
                    ),
                },
                {
                    "cell_type" : "markdown",
                    "source" : (
                        "## Bài tập 1\nCài đặt hàm sigmoid(z) bằng NumPy theo đúng công thức đã học ở Module 1. "
                        "Sau đó tính result_at_zero = sigmoid(0)."
                    ),
                },
                {
                    "cell_type" : "code",
                    "source" : (
                        "import numpy as np\n\n"
                        "def sigmoid(z):\n"
                        "    # TODO: cài đặt công thức sigmoid\n\n\n"
                        "    pass\n\n"
                        "result_at_zero = sigmoid(0)"
                    ),
                },
                {
                    "cell_type" : "code",
                    "source" : (
                        "assert result_at_zero is not None, \"Chưa cài đặt hàm sigmoid\"\n"
                        "assert abs(result_at_zero - 0.5) < 1e-6, \"sigmoid(0) phải xấp xỉ 0.5\"\n"
                        "assert 0 <= sigmoid(5) <= 1, \"Kết quả sigmoid phải nằm trong (0,1)\"\n"
                        "assert sigmoid(-100) < sigmoid(100), \"sigmoid phải là hàm đồng biến\""
                    ),
                },
            ]
        },
        ensure_ascii = False
    )


# ---------------
# 3. Xử lý JSON
# ---------------
def processing_json(raw_text: str) -> dict:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return json.loads(text)


# ------------------------------------
# 4. Validate theo đúng 5 rule_checks
# ------------------------------------
def check_cells(cells : list[dict]) -> list[str]:
    issues : list[str] = []     # List chứa các lỗi (nếu có)

    # Phải có ít nhất 3 cell markdown hướng dẫn
    md_count = sum(1 for c in cells if c.get("cell_type") == "markdown")
    if md_count < 3:
        issues.append(f"Chỉ có {md_count} cell markdown hướng dẫn (cần ít nhất 03).")

    # Lấy phần "source" của các code cell (code cell không có "source" thì source = "")
    code_sources = [c.get("source" , "") for c in cells if c.get("cell_type") == "code"]
    all_source_codecell = "\n".join(code_sources)

    # Phải có ít nhất 1 cell code có 'TODO'
    if "TODO" not in all_source_codecell:
        issues.append("Không có cell code nào chứa 'TODO' (rule has_todo).")

    # Phải có test cell
    if "assert" not in all_source_codecell:
        issues.append("Không có test cell.")

    # Kiểm tra có placeholder hay không 
    if DATASET_PLACEHOLDER not in all_source_codecell:
        issues.append(
            f"Thiếu placeholder '{DATASET_PLACEHOLDER}' ở cell chuẩn bị dữ liệu (bắt buộc để hệ thống chèn dataset thật)."
        )

    # Không được hardcore đáp án (tương đối chỉ dừng để chỉ ra phần đáng nghi,
    # do có thể có nhiều tham số như learning_rate vẫn có thể có dạng cố định 0.xxx) 
    magic_numbers = re.findall(r"=\s*0\.\d{3,}" , all_source_codecell)
    if len(magic_numbers) >= 3:
        issues.append(
            f"Nghi ngờ có đáp án hardcore"
        )

    return issues


# -----------------------------------------------------------------------------------------
# 5. Chèn dataset thật thay thế cho "# DATASET_INJECTION_PLACEHOLDER" và build file .ipynb
# -----------------------------------------------------------------------------------------
def inject_dataset(cells : list[dict], topic : str, seed : int) -> list[dict]:
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
# 6. Hàm chính
# -------------

def run_notebook_gen(
    path: LearningPath,
    profile: LearnerProfile,
    attempt: int = 1,
    prior_feedback: Optional[str] = None,
) -> str:
    last_error: Optional[str] = None

    # Tối đa 3 lần thử
    for llm_try in range(MAX_LLM_RETRIES):
        prompt = build_prompt_notebook_gen(path, profile, prior_feedback, last_error)
        raw = call_gemini(prompt)

        try:
            data = processing_json(raw)
            cells = data["cells"]
        except (json.JSONDecodeError, KeyError, TypeError) as error_json:
            last_error = f"JSON không hợp lệ hoặc thiếu field cells: {error_json}. Raw: {raw[:500]}"
            continue

        # Kiểm tra lỗi các cells
        issues = check_cells(cells)
        if issues:
            print(f"[NotebookGen] cần sinh lại (attempt {llm_try + 1}):\n {issues}")
            last_error = "\n".join(f"- {i}" for i in issues)
            continue

        # dataset_seed nằm ở profile
        cells = inject_dataset(cells, path.topic, profile.dataset_seed)

        final_code = "\n".join(
            c.get("source" , "") for c in cells if c.get("cell_type") == "code"
        )
        # Phải có "train_test_split"
        if "train_test_split" not in final_code:
            raise RuntimeError(
                "Notebook thiếu 'train_test_split' sau khi đã chèn dataset thật."
                f" Kiểm tra lại file tools/dataset_injector.py (topic = {path.topic})."
            )

        notebook_path = OUTPUT_DIR / f"{profile.session_id}_attempt{attempt}.ipynb"

        # Sinh notebook và lưu tại "notebook_path"
        build_notebook_file(cells , notebook_path)
        return str(notebook_path)

    raise RuntimeError(
        f"Notebook Generator thất bại sau {MAX_LLM_RETRIES} lần thử. "
        f"Lỗi cuối cùng: {last_error}"
    )


# ------------------------
# 7. Test nhanh bằng mock
# ------------------------
if __name__ == "__main__":
    from tests.mocks import MOCK_PATH, MOCK_PROFILE
    nb_path = run_notebook_gen(MOCK_PATH, MOCK_PROFILE, attempt = 1, prior_feedback = None)
    print("Notebook đã tạo tại:", nb_path)
