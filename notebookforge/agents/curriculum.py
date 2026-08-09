"""
agents/curriculum.py
"""

from __future__ import annotations      # Hoãn việc đánh giá các type hint (class,...) khi chúng chưa được định nghĩa
import json                             # Thư viện thao tác với JSON
import os                               # Thư viện thao tác với hệ điều hành
import re                               # Tìm kiếm, lọc, kiểm tra định dạng và thay thế chuỗi theo pattern        
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

# Chạy được schemas.py ngay trong thư mục agents
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from schemas import LearnerProfile, LearningPath, Module, ResearchBundle


# Gọi model để tạo "LearningPath" dưới dạng JSON
PROMPT_PATH_CURRICULUM = ROOT_DIR / "prompts" / "curriculum.txt"
#PROMPT_PATH = Path(__file__).resolve().parent / "curriculum.txt"
MODEL_NAME = "gemini-2.0-flash"
MAX_LLM_RETRIES = 3  # Số lần thử lại nếu LLM trả JSON sai định dạng


# -------------------
# 1. Xây dựng prompt
# -------------------

# Đọc nội dung prompt từ file curriculum.txt
def load_prompt() -> str:
    return PROMPT_PATH_CURRICULUM.read_text(encoding = "utf-8")


# Hàm tạo prompt để yêu cầu LLM trả về LearningPath
def build_prompt_curriculum(bundle : ResearchBundle, profile : LearnerProfile, last_error : Optional[str] = None) -> str:

    # lưu prompt từ file curriculum.txt 
    template = load_prompt()
    prompt = template.replace("{topic}",str(bundle.topic))
    prompt = prompt.replace("{final_level}" , str(profile.level_final))
    prompt = prompt.replace( "{key_concepts}", json.dumps(bundle.key_concepts, ensure_ascii=False))
    prompt = prompt.replace("{duration_minutes}" , str(profile.constraints.duration_minutes))
    prompt = prompt.replace("{num_exercises}" , str(profile.constraints.num_exercises))

    # Bổ sung thêm lỗi được phát hiện (nếu có) vào prompt
    if last_error:
        # Vòng retry nội bộ của tác nhân Curriculum (do LLM trả file JSON sai định dạng)
        prompt += (
            "\n# LỖI Ở LẦN TRẢ LỜI TRƯỚC — Bắt buộc sửa\n"
            f"{last_error}\n"
            "Hãy trả lại đúng định dạng JSON theo yêu cầu ở trên."
        )

    # Trả về prompt
    return prompt


# --------------------
# 2. Gọi LLM (Gemini)
# --------------------

# vs.code -> Gửi prompt + API key -> Model Gemini-2.0-flash xử lý trên hạ tầng Google
# -> Model gửi respone dạng JSON về vs.code

def call_gemini(prompt : str) -> str:
    # Lấy biến môi trường GEMINI_API_KEY từ hệ điều hành
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return dry_run_response()

    # Nếu có API key:
    import google.generativeai as genai

    genai.configure(api_key = api_key)
    model = genai.GenerativeModel(MODEL_NAME)       # gemini-2.0-flash

    # Gửi request cho 
    response = model.generate_content(
        prompt,
        generation_config = {
            "temperature" : 0.3,  # Thấp, câu trả lời ít ngẫu nhiên nên ổn định hơn
            "response_mime_type" : "application/json",     # trả respone ở dạng JSON
        },
    )
    return response.text


# Nếu chưa có API key (lấy MOCK_PATH từ file mocks.py)
def dry_run_response() -> str:
    return json.dumps(
        {
            "topic": "logistic_regression",
            "level": 1,
            "session_id": "test-001",
            "modules": [
                {
                    "module_id": "m1",
                    "title": "Hàm Sigmoid",
                    "objective": (
                        "Hiểu cách sigmoid biến đổi giá trị thực "
                        "thành xác suất."
                    ),
                    "concepts": ["sigmoid"],
                    "estimated_minutes": 20,
                    "planned_exercises": [],
                    "source_ids": ["kb_01"],
                },
                {
                    "module_id": "m2",
                    "title": "Log Loss",
                    "objective": (
                        "Hiểu cách log loss đo sai số "
                        "của dự đoán."
                    ),
                    "concepts": ["log loss"],
                    "estimated_minutes": 20,
                    "planned_exercises": [],
                    "source_ids": ["kb_01"],
                },
                {
                    "module_id": "m3",
                    "title": "Decision Boundary",
                    "objective": (
                        "Hiểu và giải thích được "
                        "ranh giới quyết định."
                    ),
                    "concepts": ["decision boundary"],
                    "estimated_minutes": 20,
                    "planned_exercises": [],
                    "source_ids": ["kb_02"],
                },
            ],
            "notes": "Dry run curriculum.",
        },
        ensure_ascii = False,
    )


# --------------
# 3. Xử lý JSON
# --------------

# Biến đổi response.text từ Gemini sang dạng dictionary trong Python
def processing_json(raw_text: str) -> dict:
    # Loại bỏ dấu cách, xuống dòng, tab thừa
    text = raw_text.strip()

    """
    Giải quyết vấn đề Markdown fence:
    Gemini có thể trả về dạng
        ``` json
        {
            <content>
        }
        ```
    tìm:
        + Markdown fence: dấu ``` + có 0 hoặc 1 từ 'json'
        + Phần nội dung {...}: group(1)
    """
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```", 
        text, 
        re.DOTALL)
    
    # Nếu có Markdown fence thì chỉ giữa lại phần group(1)
    if fence_match:
        text = fence_match.group(1)

    # Biến json thành dictionary
    return json.loads(text)


# -------------------------------------------------------
# 4. Validate và tự động điều chỉnh cho khớp constraints
# -------------------------------------------------------
def validate_and_adjust(data: dict, profile: LearnerProfile) -> tuple[dict, list[str]]:
    """
    + total_estimated_minutes: nếu lệch thì tự rescale để.
    + num_exercises_planned: chỉ cảnh báo, không tự sửa.
    """
    warnings : list[str] = []               # list rỗng để chứa các cảnh báo
    modules = data.get("modules", [])       # data: dictionary vừa chuyển từ response.text và lấy các module

    #----------------------------Kiểm tra thời gian--------------------------
    target_minutes = profile.constraints.duration_minutes
    # Nếu biến duration_minutes không phải giá trị None và list modules không rỗng
    if target_minutes is not None and modules:
        
        current_total_minutes = sum(_module_["estimated_minutes"] for _module_ in modules)

        # Nếu thời gian dự đoán khác so với thời gian người dùng mong muốn
        if current_total_minutes != target_minutes and current_total_minutes > 0:
            # Thêm cảnh báo vào list warnings
            warnings.append(
                f"total_estimated_minutes bị lệch ({current_total_minutes} != {target_minutes}). "
                "Đã tự rescale."
            )

            ratio = target_minutes / current_total_minutes      # Tỷ lệ điều chỉnh
            track_change_total = 0                              # Theo dõi tổng thời gian các module sau điều chỉnh
            for i, m in enumerate(modules):
                if i < len(modules) - 1:
                    new_time = max(5 , round(m["estimated_minutes"] * ratio))       # 5: giới hạn thời gian nhỏ nhất của một module
                    m["estimated_minutes"] = new_time
                    track_change_total += new_time
                else:
                    # module cuối nhận phần thời gian dư còn lại
                    m["estimated_minutes"] = max(5, target_minutes - track_change_total)

        data["total_estimated_minutes"] = sum(m["estimated_minutes"] for m in modules)

    # Nếu người dùng không chọn thời gian học (target_minutes = None)
    elif modules:
        data["total_estimated_minutes"] = sum(m["estimated_minutes"] for m in modules)

    #--------------------------Kiểm tra số lượng bài tập--------------------------
    target_exercises = profile.constraints.num_exercises
    if target_exercises is not None and data.get("num_exercises_planned") != target_exercises:
        warnings.append(
            f"num_exercises_planned bị lệch ({data.get('num_exercises_planned')} != {target_exercises}). "
            "Xử lý ở bước sau, KHÔNG tự sửa."
        )

    # Trả về LearningPath sau khi điều chỉnh (nếu bị lệch) và danh sách các cảnh báo
    return data, warnings


# -------------
# 5. Hàm chính
# -------------


def run_curriculum(
    bundle : ResearchBundle,
    profile : LearnerProfile,
) -> LearningPath:

    # Lỗi khi tạo file JSON ở lần thử gần nhất
    last_error: Optional[str] = None

    # Tối đa 3 lần gọi LLM
    for llm_try in range(MAX_LLM_RETRIES):

        # Dùng prompt để LLM tạo LearningPath dưới dạng JSON
        prompt_make_LearningPath = build_prompt_curriculum(bundle, profile, last_error)
        LearningPath_raw = call_gemini(prompt_make_LearningPath)

        try:
            data = processing_json(LearningPath_raw)
        except json.JSONDecodeError as error_json:
            last_error = f"JSON không hợp lệ: {error_json}. LearningPath_raw: {LearningPath_raw[:500]}"
            continue

        data, warnings = validate_and_adjust(data, profile)
        for w in warnings:
            print(f"[Curriculum Agent] cảnh báo (attempt {llm_try + 1}) : {w}")

        try:
            data["session_id"] = profile.session_id
            data["level"] = profile.level_final
            # Trả về class LearningPath theo đúng định dạng schema
            return LearningPath(**data)
        except ValidationError as error_validate:
            last_error = f"Dữ liệu không khớp schema LearningPath: {error_validate}"
            continue

    # Hết lượt thử
    raise RuntimeError(
        f"Curriculum Agent thất bại sau {MAX_LLM_RETRIES} lần thử. "
        f"Lỗi cuối cùng: {last_error}"
    )


# ------------------------------------------------------------
# 6. Test nhanh bằng mock — chạy: python -m agents.curriculum
# ------------------------------------------------------------
if __name__ == "__main__":
    from tests.mocks import MOCK_BUNDLE, MOCK_PROFILE  # tests.mocks

    path = run_curriculum(MOCK_BUNDLE, MOCK_PROFILE)
    print(path.model_dump_json(indent = 2))
