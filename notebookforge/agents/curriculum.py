"""
agents/curriculum.py
"""

from __future__ import annotations      # Hoãn việc đánh giá các type hint (class,...) khi chúng chưa được định nghĩa
import json                             # Thư viện thao tác với JSON
import re                               # Tìm kiếm, lọc, kiểm tra định dạng và thay thế chuỗi theo pattern        
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

# Chạy được schemas.py ngay trong thư mục agents
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from schemas import LearnerProfile, LearningPath, Module, ResearchBundle

from llm_client import call_json

# Gọi model để tạo "LearningPath" dưới dạng JSON
PROMPT_PATH_CURRICULUM = ROOT_DIR / "prompts" / "curriculum.txt"


# -------------------
# 1. Xây dựng prompt
# -------------------
# Đọc nội dung prompt từ file curriculum.txt
def load_prompt() -> str:
    return PROMPT_PATH_CURRICULUM.read_text(encoding = "utf-8")

# Hàm tạo prompt để yêu cầu LLM trả về LearningPath
def build_prompt_curriculum(bundle : ResearchBundle, profile : LearnerProfile) -> str:
    # lưu prompt từ file curriculum.txt 
    template = load_prompt()
    prompt = template.replace("{topic}",str(bundle.topic))
    prompt = prompt.replace("{final_level}" , str(profile.level_final))
    prompt = prompt.replace( "{key_concepts}", json.dumps(bundle.key_concepts, ensure_ascii=False))
    prompt = prompt.replace("{duration_minutes}" , str(profile.constraints.duration_minutes))
    prompt = prompt.replace("{num_exercises}" , str(profile.constraints.num_exercises))

    # Trả về prompt
    return prompt

# -------------------------------------------------------
# 2. Validate và tự động điều chỉnh cho khớp constraints
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

    #--------------------------Kiểm tra số lượng bài tập--------------------------
    target_exercises = profile.constraints.num_exercises
    actual_exercises = sum(len(m.get("planned_exercises", [])) for m in modules)
    if actual_exercises != target_exercises:
        warnings.append(
            f"Tổng planned_exercises thực tế ({actual_exercises}) lệch với "
            f"constraints.num_exercises ({target_exercises}). Xử lý ở bước sau, KHÔNG tự sửa."
        )

    # Trả về LearningPath sau khi điều chỉnh (nếu bị lệch) và danh sách các cảnh báo
    return data, warnings


# -------------
# 3. Hàm chính
# -------------
def run_curriculum(
    bundle : ResearchBundle,
    profile : LearnerProfile,
) -> LearningPath:
    
    # Dùng prompt để LLM tạo LearningPath dưới dạng JSON
    prompt_make_LearningPath = build_prompt_curriculum(bundle, profile)
    LearningPath_raw, meta = call_json(
        prompt = prompt_make_LearningPath,
        schema = LearningPath,
        session_id = profile.session_id,
    )

    data = LearningPath_raw.model_dump()

    data, warnings = validate_and_adjust(data, profile)
    for w in warnings:
        print(f"[Curriculum Agent] cảnh báo : {w}")

    try:
        data["session_id"] = profile.session_id
        data["level"] = profile.level_final
        data["topic"] = profile.topic
        # Trả về class LearningPath theo đúng định dạng schema
        return LearningPath(**data)
    except ValidationError as error_validate:
        last_error = f"Dữ liệu không khớp schema LearningPath: {error_validate}"


# ------------------------------------------------------------
# 4. Test nhanh bằng mock — chạy: python -m agents.curriculum
# ------------------------------------------------------------
if __name__ == "__main__":
    from tests.mocks import MOCK_BUNDLE, MOCK_PROFILE

    path = run_curriculum(MOCK_BUNDLE, MOCK_PROFILE)
    print(path.model_dump_json(indent = 2))
