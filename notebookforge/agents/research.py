from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from schemas import Citation, LearnerProfile, ResearchBundle, Source
from tools.kb_reader import (
    KB_ROOT,
    KBIndex,
    find_concept_in_kb,
    list_available_topics,
    read_kb_files,
)

try:
    from llm_client import call_text
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Chưa tìm thấy llm_client.py!")
    def call_text(prompt: str, session_id: str) -> Tuple[str, Any]:
        raise NotImplementedError("Dùng llm_client của Hoàng.")

logger = logging.getLogger(__name__)

# =============================================================================
# CACHE TOÀN CỤC
# =============================================================================
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 giờ
_CACHE_LOCK = threading.Lock()
GLOBAL_RESEARCH_CACHE: Dict[str, Tuple[float, ResearchBundle]] = {}

def _cache_get(h_key: str) -> Optional[ResearchBundle]:
    with _CACHE_LOCK:
        cached = GLOBAL_RESEARCH_CACHE.get(h_key)
        if cached is None:
            return None
        ts_luu, bundle = cached
        if time.time() - ts_luu > CACHE_TTL_SECONDS:
            del GLOBAL_RESEARCH_CACHE[h_key]
            return None
        return bundle

def _cache_set(h_key: str, bundle: ResearchBundle) -> None:
    with _CACHE_LOCK:
        GLOBAL_RESEARCH_CACHE[h_key] = (time.time(), bundle)

def resolve_topic_alias(raw_topic: str) -> str:
    if not raw_topic:
        return ""
    clean = raw_topic.strip().lower().replace(" ", "_").replace("-", "_")
    available = list_available_topics()
    if clean in available:
        return clean
    alias_map = {
        "logistic": "logistic_regression", "logreg": "logistic_regression",
        "dt": "decision_tree", "tree": "decision_tree",
        "k_means": "kmeans", "k_mean": "kmeans",
    }
    return alias_map.get(clean, clean)

def _kb_content_fingerprint(topic: str) -> str:
    """Hash nội dung TOÀN BỘ file .md trong kb/<topic>/ để nhận diện thay đổi file."""
    topic_dir = KB_ROOT / topic
    if not topic_dir.exists():
        return "no_kb_dir"
    parts = []
    for path in sorted(topic_dir.glob("*.md")):
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            continue
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

def cache_hash(topic: str, learner_profile: Optional[LearnerProfile] = None) -> str:
    """Tạo mã băm siêu tốc dựa trên profile và dữ liệu KB (dùng cho Cache)."""
    actual_topic = resolve_topic_alias(topic)
    level_final = getattr(learner_profile, "level_final", 2) if learner_profile else 2
    quiz_score = getattr(learner_profile, "quiz_score", 3) if learner_profile else 3
    
    constraints = getattr(learner_profile, "constraints", None)
    duration = getattr(constraints, "duration_minutes", 120) if constraints else 120
    language = getattr(constraints, "language", "vi") if constraints else "vi"

    payload = {
        "topic": actual_topic,
        "level_final": int(level_final),
        "quiz_score": int(quiz_score),
        "duration_minutes": int(duration),
        "language": language,
        "kb_fingerprint": _kb_content_fingerprint(actual_topic),
    }
    raw_payload = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

def _clean_llm_json(raw_response: str) -> str:
    text = raw_response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

# =============================================================================
# KIẾN TRÚC 2 GIAI ĐOẠN (LLM LỌC CONCEPTS)
# =============================================================================
def _stage2_llm_filtering(
    candidate_concepts: List[str],
    topic: str,
    learner_profile: Optional[LearnerProfile] = None,
) -> List[str]:
    level_map = {1: "BEGINNER", 2: "INTERMEDIATE", 3: "ADVANCED"}
    user_level_num = getattr(learner_profile, "level_final", 2) if learner_profile else 2
    user_level_str = level_map.get(user_level_num, "INTERMEDIATE")
    
    constraints = getattr(learner_profile, "constraints", None)
    duration_minutes = getattr(constraints, "duration_minutes", 120) if constraints else 120

    prompt = f"""
    Bạn là chuyên gia giáo dục thiết kế khóa học cho chủ đề: '{topic}'.
    Danh sách khái niệm: {candidate_concepts}
    
    Học viên: {user_level_str}. Quỹ thời gian: {duration_minutes} phút.
    
    Nhiệm vụ: Chọn ra các khái niệm phù hợp trình độ sao cho tổng thời gian học (Beginner ~10p, Inter ~15p, Adv ~25p) KHÔNG VƯỢT QUÁ {duration_minutes} phút.
    
    Chỉ trả về MẢNG TÊN CÁC KHÁI NIỆM (JSON list of strings). Không giải thích.
    Ví dụ: ["Khái niệm A", "Khái niệm B"]
    """

    # Truyền session_id và bóc tách tuple trả về từ hàm của Hoàng 
    session_id = getattr(learner_profile, "session_id", "default_session") if learner_profile else "default_session"

    try:
        raw_response, meta = call_text(prompt=prompt, session_id=session_id)
        clean_response = _clean_llm_json(raw_response)
        assessed_concepts = json.loads(clean_response)

        if not isinstance(assessed_concepts, list):
            raise ValueError("LLM không trả về JSON list")

        # Chống Hallucination: Chỉ lấy những concept thực sự có trong ứng viên
        candidate_set = {c.strip().lower(): c for c in candidate_concepts}
        validated_names = []
        for name in assessed_concepts:
            c_name = str(name).strip().lower()
            if c_name in candidate_set:
                validated_names.append(candidate_set[c_name])

        if validated_names:
            return list(dict.fromkeys(validated_names))
            
    except Exception as e:
        logger.error(f"LLM lọc lỗi, dùng Fallback: {e}")

    # Fallback Rule-based nếu LLM tịt
    max_items = max(1, duration_minutes // 15)
    return candidate_concepts[:max_items]

def _generate_candidate_concepts_for_web(topic: str) -> List[str]:
    title = topic.replace("_", " ").title()
    return [
        f"Overview of {title}", f"Mathematical Foundations of {title}",
        f"Algorithm and Execution Steps of {title}", f"Python Implementation for {title}",
        f"Advantages and Limitations of {title}", f"Real-world Applications of {title}",
    ]

# =============================================================================
# WEB SEARCH FALLBACK (WIKIPEDIA)
# =============================================================================
_WEB_SEARCH_TIMEOUT = 6

def _wikipedia_search_page(query: str) -> Optional[Dict[str, str]]:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
            timeout=_WEB_SEARCH_TIMEOUT,
            headers={"User-Agent": "ResearchAgent/1.0"},
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        if not hits: return None
        return {"title": hits[0]["title"], "snippet": re.sub(r"<[^>]+>", "", hits[0].get("snippet", ""))}
    except Exception as e:
        logger.error(f"Lỗi Wikipedia search '{query}': {e}")
        return None

def _wikipedia_fetch_extract(title: str) -> Optional[str]:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "prop": "extracts", "exintro": True, "explaintext": True, "titles": title, "format": "json"},
            timeout=_WEB_SEARCH_TIMEOUT,
            headers={"User-Agent": "ResearchAgent/1.0"},
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            if extract := page.get("extract", ""): return extract
        return None
    except Exception:
        return None

def _content_matches_concept(concept: str, extract: str, min_token_hit_ratio: float = 0.5) -> bool:
    if not extract: return False
    stopwords = {"of", "in", "and", "the", "a", "an", "for", "to", "with", "on"}
    tokens = [t for t in re.split(r"[\s()/']+", concept.lower()) if len(t) >= 3 and t not in stopwords]
    if not tokens: return False
    extract_norm = re.sub(r"[-_]", "", extract)
    hits = sum(1 for t in tokens if re.search(rf"\b{re.escape(re.sub(r'[-_]', '', t))}\b", extract_norm, re.IGNORECASE))
    return (hits / len(tokens)) >= min_token_hit_ratio

def _fallback_web_search(concept: str) -> Optional[Dict[str, str]]:
    clean_concept = concept.strip()
    if len(clean_concept) < 3: return None

    hit = _wikipedia_search_page(clean_concept)
    if not hit: return None

    extract = _wikipedia_fetch_extract(hit["title"])
    validated_text = extract or hit["snippet"]
    if not _content_matches_concept(clean_concept, validated_text):
        logger.warning(f"Reject kết quả Web cho '{clean_concept}' do không đủ độ khớp.")
        return None

    target_url = "https://en.wikipedia.org/wiki/" + hit["title"].replace(" ", "_")
    source_hash = hashlib.md5(hit["title"].lower().encode("utf-8")).hexdigest()[:8]

    return {
        "source_id": f"web_{source_hash}",
        "url": target_url,
        "type": "web",  # <-- Đã fix lỗi string literal từ schemas.py
        "title": f"Wikipedia - {hit['title']}",
    }

def _extract_quote_from_body(concept: str, body: str) -> Optional[str]:
    if not body: return None
    body_no_code = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body_no_code):
        clean_sentence = sentence.strip().lstrip("*- ")
        if concept.lower() in clean_sentence.lower() and len(clean_sentence) > 15:
            return clean_sentence[:200]
    return None

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_research(
    topic: str,
    learner_profile: Optional[LearnerProfile] = None,
) -> ResearchBundle:
    h_key = cache_hash(topic, learner_profile)
    if cached_bundle := _cache_get(h_key):
        logger.info(f"Cache Hit! Bỏ qua tính toán cho hash: {h_key}")
        return cached_bundle

    actual_topic = resolve_topic_alias(topic)
    
    # 1. SPARSE RETRIEVAL (BM25 từ Local KB)
    kb_entries = read_kb_files(actual_topic)
    kb_index = KBIndex.build(kb_entries)

    raw_concepts: List[str] = []
    if kb_entries:
        for entry in kb_entries:
            raw_concepts.extend(getattr(entry, "key_concepts", []) or [])
    else:
        # Nếu KB trống (chưa có file md), tự sinh ứng viên để Web Search
        raw_concepts = _generate_candidate_concepts_for_web(actual_topic)
        
    unique_raw_concepts = list(dict.fromkeys(raw_concepts))

    # 2. LLM RE-RANKING & FILTERING
    final_concepts = _stage2_llm_filtering(unique_raw_concepts, actual_topic, learner_profile)

    sources_map: Dict[str, Source] = {}
    citations: List[Citation] = []
    unresolved_concepts: List[str] = []  # --- FIX 3: Thêm list chứa các khái niệm rớt đài

    # 3. TRÍCH XUẤT SOURCE VÀ CITATION
    for kw in final_concepts:
        matched_entry = find_concept_in_kb(kw, kb_index) if kb_entries else None
        
        if matched_entry:
            source_id = str(getattr(matched_entry, "source_id", "kb_file"))
            
            # Ưu tiên lấy source_url từ YAML, nếu rỗng thì dùng path file .md
            kb_url = getattr(matched_entry, "source_url", "")
            path = str(kb_url if kb_url else getattr(matched_entry, "path", "kb_path"))
            
            source_type = "kb_file"
            locator = getattr(matched_entry, "subtopic", None) or "Nội dung cơ bản"
            title = getattr(matched_entry, "subtopic", None) or f"Khái niệm {actual_topic.title()}"
            quote_text = _extract_quote_from_body(kw, getattr(matched_entry, "body", ""))
        else:
            # Fallback Web Search khi KB thiếu
            web_result = _fallback_web_search(kw)
            if web_result:
                source_id = web_result["source_id"]
                path = web_result["url"]
                source_type = web_result["type"] # Chắc chắn là "web" vì đã fix ở trên
                title = web_result["title"]
                locator = "Web Search Result"
                quote_text = None
            else:
                # --- FIX 3: Bắt những khái niệm thất bại vào danh sách
                unresolved_concepts.append(kw)
                continue

        if source_id not in sources_map:
            sources_map[source_id] = Source(source_id=source_id, type=source_type, path_or_url=path, title=title)

        citations.append(Citation(concept=kw, source_id=source_id, locator=str(locator), quote=quote_text))

    topic_clean = actual_topic.lower().strip()
    if "decision_tree" in topic_clean:
        prerequisites, pitfalls = ["Python cơ bản", "Numpy & Pandas"], ["Quá khớp", "Mất cân bằng dữ liệu"]
    elif "kmeans" in topic_clean:
        prerequisites, pitfalls = ["Khoảng cách Euclidean", "Numpy"], ["Nhạy cảm điểm khởi tạo", "Chọn sai K"]
    else:
        prerequisites, pitfalls = ["Đại số tuyến tính", "Python cơ bản"], ["Rò rỉ dữ liệu", "Quá khớp"]

    # Truyền unresolved_concepts
    bundle = ResearchBundle(
        topic=str(actual_topic),
        sources=list(sources_map.values()),
        key_concepts=final_concepts,
        citations=citations,
        prerequisites=prerequisites,
        common_pitfalls=pitfalls,
        unresolved_concepts=unresolved_concepts,
    )

    _cache_set(h_key, bundle)
    return bundle


# if __name__ == "__main__":
#     from schemas import Constraints, LearnerProfile

#     print("\n================ [BỘ KIỂM TRẢ RESEARCH AGENT] ================")
#     raw_topic = input("1. Nhập topic (vd: logisict, dt, kmeans, SVD) [Mặc định: SVD]: ").strip() or "SVD"

#     user_profile = LearnerProfile(
#         topic=raw_topic, level_declared=2, level_final=1, quiz_score=1,
#         constraints=Constraints(duration_minutes=120), session_id="test_session",
#     )

#     print(f"\n[CACHE HASH GENERATED]: {cache_hash(raw_topic, user_profile)}")
#     res = run_research(raw_topic, learner_profile=user_profile)
#     print(res.model_dump_json(indent=2))
#     print("==========================================================\n")