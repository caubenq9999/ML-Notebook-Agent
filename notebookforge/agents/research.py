from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# BẮT BUỘC IMPORT TỪ SCHEMAS.PY - Không tự định nghĩa lại!
from schemas import Citation, ResearchBundle, Source
from tools.kb_reader import find_concept_in_kb, read_kb_files

logger = logging.getLogger(__name__)

# Tích hợp BM25 với cơ chế Fallback
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("Thư viện 'rank_bm25' chưa được cài đặt. Hệ thống sẽ dùng Token Matching làm fallback.")

# ==============================================================================
# Trích xuất Keywords từ Knowledge Base (Frontmatter)
# ==============================================================================
def _extract_keywords_from_kb(
    topic: str, 
    kb_entries: Optional[List[Any]] = None
) -> List[str]:
    keywords = []

    # 1. Trích xuất concepts chuẩn từ YAML frontmatter của KB
    if kb_entries:
        for entry in kb_entries:
            kb_concepts = getattr(entry, "key_concepts", [])
            for c in kb_concepts:
                clean_c = str(c).strip()
                if clean_c and clean_c not in keywords:
                    keywords.append(clean_c)

    # 2. Fallback mặc định nếu không tìm thấy gì
    if not keywords:
        clean_topic = str(topic).replace("_", " ").title()
        keywords = [clean_topic, "Cost Function", "Gradient Descent", "Overfitting", "L1 Regularization"]

    # Deduplicate giữ nguyên thứ tự
    seen = set()
    return [x for x in keywords if not (x in seen or seen.add(x))]

# ==============================================================================
# CORE AGENT LOGIC (TRÍ)
# ==============================================================================
def run_research(topic: str) -> ResearchBundle:
    # 1. Đọc danh sách file KB
    kb_entries = read_kb_files(topic)

    # 2. Đề xuất keywords dựa hoàn toàn vào KB
    proposed_keywords = _extract_keywords_from_kb(topic, kb_entries=kb_entries)

    sources_map: Dict[str, Source] = {}
    citations: List[Citation] = []
    
    # Ở giai đoạn sau, các mảng này sẽ được LLM phân tích từ nội dung bài học. 
    # Tạm thời điền mock data cho đúng schema.
    prerequisites: List[str] = ["Linear Algebra", "Python Basics", "Numpy & Pandas"]
    common_pitfalls: List[str] = ["Data Leakage", "Multicollinearity", "Overfitting"]

    # 3. Tra cứu từng khái niệm trên KB (Chỉ lấy Top 1 Citation)
    for kw in proposed_keywords:
        matched_entry: Any = find_concept_in_kb(kw, kb_entries)
        
        if matched_entry:
            source_id = getattr(matched_entry, "source_id", None) or (
                matched_entry.get("source_id") if isinstance(matched_entry, dict) else "kb_file"
            )
            path = getattr(matched_entry, "path", None) or (
                matched_entry.get("path") if isinstance(matched_entry, dict) else "kb_path"
            )

            # Cập nhật vào danh sách nguồn nếu chưa có
            if source_id not in sources_map:
                sources_map[source_id] = Source(
                    source_id=str(source_id),
                    type="kb_file",
                    path_or_url=str(path),
                )
            
            # Gắn trích dẫn (Chỉ top 1 kết quả tốt nhất)
            citations.append(Citation(concept=kw, source_id=str(source_id)))
            
        # QUAN TRỌNG: Nếu không matched, bỏ qua! 
        # schemas.py sẽ tự động tóm lấy các concept bị rớt và đẩy vào unresolved_concepts.

    # 4. Tạo ResearchBundle đầy đủ tất cả các trường Schema
    bundle = ResearchBundle(
        topic=str(topic),
        sources=list(sources_map.values()),
        key_concepts=proposed_keywords,
        citations=citations,
        # unresolved_concepts=..., -> Không cần truyền vào, Pydantic validator ở schemas.py sẽ lo việc này!
        prerequisites=prerequisites,
        common_pitfalls=common_pitfalls,
    )

    return bundle

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Chạy test độc lập CHỈ với topic, tuân thủ tuyệt đối chữ ký hàm và schema
    res = run_research("logistic_regression")
    
    print("\n[OUTPUT RESEARCH BUNDLE]:")
    print(res.model_dump_json(indent=2))