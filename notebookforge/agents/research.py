from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from schemas import Citation, LearnerProfile, ResearchBundle, Source
from tools.kb_reader import find_concept_in_kb, list_available_topics, read_kb_files

logger = logging.getLogger(__name__)


def resolve_topic_alias(raw_topic: str) -> str:
    """Chuẩn hóa tên topic nhập vào về đúng tên folder KB hiện có."""
    if not raw_topic:
        return ""

    clean = raw_topic.strip().lower().replace(" ", "_").replace("-", "_")
    available = list_available_topics()

    # 1. Khớp chính xác với tên folder KB
    if clean in available:
        return clean

    # 2. Bảng mapping gõ tắt / gõ sai chính tả
    alias_map = {
        "logistic": "logistic_regression",
        "logreg": "logistic_regression",
        "logisict": "logistic_regression",
        "logitic": "logistic_regression",
        "dt": "decision_tree",
        "tree": "decision_tree",
        "k_means": "kmeans",
        "k_mean": "kmeans",
    }
    if clean in alias_map:
        return alias_map[clean]

    # 3. Khớp một phần (partial match)
    for topic_folder in available:
        if clean in topic_folder or topic_folder in clean:
            return topic_folder

    # 4. Topic hoàn toàn mới (VD: SVD, Random Forest)
    return clean


def cache_hash(topic: str, learner_profile: Optional[LearnerProfile] = None) -> str:
    """Tạo Hash SHA-256 dựa trên tất cả thông số trong LearnerProfile để phục vụ Cache."""
    actual_topic = resolve_topic_alias(topic)
    level_final = getattr(learner_profile, "level_final", 2) if learner_profile else 2
    level_declared = getattr(learner_profile, "level_declared", level_final) if learner_profile else level_final
    quiz_score = getattr(learner_profile, "quiz_score", 3) if learner_profile else 3
    
    duration = 45
    if learner_profile and hasattr(learner_profile, "constraints") and learner_profile.constraints:
        duration = getattr(learner_profile.constraints, "duration_minutes", 45)

    payload = {
        "topic": actual_topic,
        "level_final": int(level_final),
        "level_declared": int(level_declared),
        "quiz_score": int(quiz_score),
        "duration_minutes": int(duration),
    }
    raw_payload = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


def _extract_quote_from_body(concept: str, body: str) -> Optional[str]:
    """Trích xuất trích dẫn trực tiếp từ bài viết KB."""
    if not body:
        return None
    sentences = re.split(r"(?<=[.!?])\s+|\n+", body)
    for sentence in sentences:
        clean_sentence = sentence.strip().lstrip("#*- ").strip()
        if concept.lower() in clean_sentence.lower() and len(clean_sentence) > 15:
            return clean_sentence[:200]
    return None


def _generate_candidate_concepts_for_web(topic: str) -> List[str]:
    """Tạo danh sách khái niệm chuẩn (N_total = 12) cho các chủ đề ngoài KB (VD: SVD)."""
    title = topic.replace("_", " ").title()
    return [
        f"Overview of {title}",
        f"Mathematical Foundations of {title}",
        f"Core Formulation and Equations of {title}",
        f"Key Assumptions and Properties of {title}",
        f"{title} Algorithm and Execution Steps",
        f"Loss Function and Optimization in {title}",
        f"Python Implementation for {title}",
        f"Hyperparameter Tuning in {title}",
        f"Advantages and Limitations of {title}",
        f"Common Pitfalls and Edge Cases in {title}",
        f"Model Evaluation and Metrics for {title}",
        f"Real-world Applications of {title}"
    ]


def _generate_keywords_via_llm(
    topic: str, 
    learner_profile: Optional[LearnerProfile] = None,
    kb_entries: Optional[List[Any]] = None
) -> List[str]:
    """Tính toán số lượng khái niệm tối ưu (k*) dựa trên công thức toán cho cả KB và Web Fallback."""
    # 1. Thu thập danh sách khái niệm ứng viên
    if kb_entries and len(kb_entries) > 0:
        ordered_concepts: List[str] = []
        for entry in kb_entries:
            concepts = getattr(entry, "key_concepts", []) or []
            ordered_concepts.extend(concepts)
        unique_concepts = list(dict.fromkeys(ordered_concepts))
    else:
        # Nếu không có trong KB, khởi tạo lộ trình 12 bước tiêu chuẩn cho Web Search
        unique_concepts = _generate_candidate_concepts_for_web(topic)

    total_concepts = len(unique_concepts)

    if not learner_profile:
        return unique_concepts[: min(5, total_concepts)]

    # 2. Đọc toàn bộ tham số từ LearnerProfile
    l_final = int(getattr(learner_profile, "level_final", 2))
    l_declared = int(getattr(learner_profile, "level_declared", l_final))
    quiz_score = float(getattr(learner_profile, "quiz_score", 3))
    
    constraints = getattr(learner_profile, "constraints", None)
    duration = getattr(constraints, "duration_minutes", 45) if constraints else 45

    # 3. Tính toán Tỷ lệ Năng lực thực tế (R)
    base_level_factor = l_final / 3.0
    quiz_factor = 0.7 + 0.3 * (quiz_score / 5.0)
    confidence_gap = max(-1.0, min(1.0, float(l_declared - l_final))) * 0.05
    
    raw_r = (base_level_factor * quiz_factor) + confidence_gap
    r = max(0.3, min(1.0, raw_r))

    # 4. Tính toán số lượng khái niệm tối ưu (k*)
    k_depth = int(round(r * total_concepts))
    k_time = int(duration // 9)  # Ước tính 10 phút/khái niệm
    k_min = 3                     # Ngưỡng tối thiểu

    k_opt = max(k_min, min(total_concepts, k_depth, k_time))

    # 5. Cắt danh sách đúng k* khái niệm
    return unique_concepts[:k_opt]


def _fallback_web_search(concept: str) -> Optional[Dict[str, str]]:
    """Tạo nguồn Web Search chuẩn schema với type='web'."""
    clean_concept = concept.strip()
    if not clean_concept:
        return None

    # Mã hóa md5 cố định
    source_hash = hashlib.md5(clean_concept.lower().encode("utf-8")).hexdigest()[:8]
    source_id = f"web_{source_hash}"

    # Kiểm tra tính hợp lệ của source_id qua Regex
    if not re.match(r"^web_[a-f0-9]{8}$", source_id):
        logger.error(f"Source ID không đúng định dạng: {source_id}")
        return None

    formatted_query = clean_concept.replace(" ", "_")
    target_url = f"https://en.wikipedia.org/wiki/{formatted_query}"

    return {
        "source_id": source_id,
        "url": target_url,
        "type": "web",  # Sửa từ 'web_search' thành 'web' để qua validation Pydantic
        "title": f"Wikipedia - {clean_concept}"
    }

def run_research(
    topic: str, 
    learner_profile: Optional[LearnerProfile] = None
) -> ResearchBundle:
    raw_topic = getattr(learner_profile, "topic", topic) if learner_profile else topic

    actual_topic = resolve_topic_alias(raw_topic)
    logger.info(f"Topic thô: '{raw_topic}' -> Đã chuẩn hóa: '{actual_topic}'")

    # 1. Đọc KB theo topic
    kb_entries = read_kb_files(actual_topic)

    # 2. Lọc khái niệm dựa vào công thức toán từ LearnerProfile
    proposed_keywords = _generate_keywords_via_llm(
        actual_topic, 
        learner_profile=learner_profile, 
        kb_entries=kb_entries
    )

    sources_map: Dict[str, Source] = {}
    citations: List[Citation] = []

    # 3. Prerequisites & Common Pitfalls
    topic_clean = actual_topic.lower().strip()
    if "decision_tree" in topic_clean:
        prerequisites = ["Python cơ bản", "Numpy & Pandas", "Xác suất thống kê"]
        common_pitfalls = ["Quá khớp (Overfitting)", "Rò rỉ dữ liệu (Data Leakage)", "Mất cân bằng dữ liệu"]
    elif "kmeans" in topic_clean:
        prerequisites = ["Đại số tuyến tính", "Khoảng cách Euclidean", "Numpy & Pandas"]
        common_pitfalls = ["Nhạy cảm với Điểm khởi tạo", "Ảnh hưởng bởi Outliers", "Chọn sai số cụm K"]
    else:
        prerequisites = ["Đại số tuyến tính", "Python cơ bản", "Numpy & Pandas"]
        common_pitfalls = ["Rò rỉ dữ liệu (Data Leakage)", "Đa cộng tuyến (Multicollinearity)", "Quá khớp (Overfitting)"]

    # 4. Trích xuất Source & Citation
    for kw in proposed_keywords:
        matched_entry: Any = find_concept_in_kb(kw, kb_entries) if kb_entries else None

        if matched_entry:
            source_id = str(getattr(matched_entry, "source_id", "kb_file"))
            path = str(getattr(matched_entry, "path", "kb_path"))
            source_type = "kb_file"
            locator = getattr(matched_entry, "subtopic", None) or "Nội dung cơ bản"
            title = getattr(matched_entry, "subtopic", None) or f"Khái niệm {actual_topic.title()}"
            body_text = getattr(matched_entry, "body", "")
            quote_text = _extract_quote_from_body(kw, body_text)
        else:
            web_result = _fallback_web_search(kw)
            if web_result:
                source_id = web_result["source_id"]
                path = web_result["url"]
                source_type = web_result["type"]
                title = web_result["title"]
                locator = "Web Search Result"
                quote_text = None
            else:
                continue

        if source_id not in sources_map:
            sources_map[source_id] = Source(
                source_id=source_id,
                type=source_type,
                path_or_url=path,
                title=title
            )

        citations.append(
            Citation(
                concept=kw,
                source_id=source_id,
                locator=str(locator) if locator else None,
                quote=quote_text
            )
        )

    return ResearchBundle(
        topic=str(actual_topic),
        sources=list(sources_map.values()),
        key_concepts=proposed_keywords,
        citations=citations,
        prerequisites=prerequisites,
        common_pitfalls=common_pitfalls,
    )


# if __name__ == "__main__":
#     from schemas import Constraints, LearnerProfile

#     print("\n================ [BỘ KIỂM TRẢ RESEARCH AGENT (ĐẦY ĐỦ THÔNG SỐ)] ================")
    
#     raw_topic = input("1. Nhập topic (vd: logisict, dt, kmeans, SVD) [Mặc định: SVD]: ").strip()
#     if not raw_topic:
#         raw_topic = "SVD"

#     raw_declared = input("2. Nhập Level TỰ KHAI (1: Beginner, 2: Intermediate, 3: Advanced) [Mặc định: 2]: ").strip()
#     try:
#         level_declared = int(raw_declared) if int(raw_declared) in (1, 2, 3) else 2
#     except ValueError:
#         level_declared = 2

#     raw_final = input("3. Nhập Level CHỐT SAU QUIZ (1: Beginner, 2: Intermediate, 3: Advanced) [Mặc định: 2]: ").strip()
#     try:
#         level_final = int(raw_final) if int(raw_final) in (1, 2, 3) else 2
#     except ValueError:
#         level_final = 2

#     raw_score = input("4. Nhập Điểm Quiz (0 đến 5) [Mặc định: 3]: ").strip()
#     try:
#         quiz_score = int(raw_score) if 0 <= int(raw_score) <= 5 else 3
#     except ValueError:
#         quiz_score = 3

#     raw_duration = input("5. Nhập Thời lượng học (phút, vd: 15, 30, 45, 60) [Mặc định: 45]: ").strip()
#     try:
#         duration = int(raw_duration) if int(raw_duration) > 0 else 45
#     except ValueError:
#         duration = 45

#     user_profile = LearnerProfile(
#         topic=raw_topic,
#         level_declared=level_declared,  # type: ignore
#         level_final=level_final,        # type: ignore
#         quiz_score=quiz_score,
#         constraints=Constraints(duration_minutes=duration),
#         session_id=f"test_session_{level_final}"
#     )

#     test_hash = cache_hash(raw_topic, user_profile)
#     print(f"\n [CACHE HASH GENERATED FOR HOÀNG]: {test_hash}")

#     print("\n================ [OUTPUT RESEARCH BUNDLE] ================")
#     res = run_research(raw_topic, learner_profile=user_profile)
#     print(res.model_dump_json(indent=2))
#     print(f"\n[KẾT QUẢ]: Thu được {len(res.key_concepts)} khái niệm tối ưu dựa trên công thức toán.")
#     print("==========================================================\n")