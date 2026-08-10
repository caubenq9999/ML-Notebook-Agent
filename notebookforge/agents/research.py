from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from tools.kb_reader import KBEntry, find_concept_in_kb, read_kb_files

logger = logging.getLogger(__name__)

# Tích hợp BM25 với cơ chế Fallback 
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("Thư viện 'rank_bm25' chưa được cài đặt. Hệ thống sẽ tạm dùng Token Matching làm fallback.")

# ==============================================================================
# SCHEMAS 
# ==============================================================================
try:
    from schemas import Citation, LearnerProfile, ResearchBundle, Source  # type: ignore
except ImportError:
    from pydantic import BaseModel, Field

    class Source(BaseModel):  # type: ignore[no-redef]
        source_id: str
        type: str  # "kb_file" | "web" | "paper" | "textbook"
        path_or_url: str

    class Citation(BaseModel):  # type: ignore[no-redef]
        concept: str
        source_id: str

    class LearnerProfile(BaseModel):  # type: ignore[no-redef]
        level: Optional[str] = "Beginner"
        goals: Optional[List[str]] = Field(default_factory=list)
        weak_concepts: Optional[List[str]] = Field(default_factory=list)
        known_concepts: Optional[List[str]] = Field(default_factory=list)

    class ResearchBundle(BaseModel):  # type: ignore[no-redef]
        topic: str
        sources: List[Source]
        key_concepts: List[str]
        citations: List[Citation]
        unresolved_concepts: List[str] = Field(default_factory=list)


MAX_CONTEXT_CHARS = 6000
MAX_SNIPPET_CHARS_PER_RESULT = 800
WebSearchFn = Callable[[str], Sequence[Dict[str, str]]]


def _default_web_search(query: str) -> Sequence[Dict[str, str]]:
    logger.info("Thực hiện Web Search cho query: '%s'", query)
    return []


def _budget_context(results: Sequence[Dict[str, str]], *, max_total_chars: int = MAX_CONTEXT_CHARS) -> str:
    chunks: List[str] = []
    used = 0
    for i, r in enumerate(results):
        snippet = re.sub(r"\s+", " ", (r.get("snippet") or "").strip())[:MAX_SNIPPET_CHARS_PER_RESULT]
        url = r.get("url", "unknown")
        chunk = f"[NGUỒN {i + 1}] ({url})\n{snippet}"
        if used + len(chunk) > max_total_chars:
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n".join(chunks)


# ==============================================================================
# SPARSE SEARCH BẰNG BM25 
# ==============================================================================
def _ground_concept_in_text_bm25(concept: str, text: str, bm25_threshold: float = 0.1) -> bool:
    """Xác thực sự tồn tại của từ khóa trong văn bản bằng BM25 hoặc Token Matching."""
    if not text or not concept or not text.strip() or not concept.strip():
        return False

    # Tách từ khóa thành tokens
    concept_tokens = [t.lower() for t in re.split(r"\W+", concept) if len(t) >= 2]
    if not concept_tokens:
        return False

    # Kiểm tra nhanh trùng khớp chính xác chuỗi (Exact match fallback)
    text_lower = text.lower()
    if concept.lower().strip() in text_lower:
        return True

    # Tách văn bản thành các đoạn
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    if HAS_BM25:
        tokenized_corpus = [
            [t.lower() for t in re.split(r"\W+", p) if len(t) >= 2]
            for p in paragraphs
        ]
        # Lọc các đoạn rỗng token
        tokenized_corpus = [doc for doc in tokenized_corpus if doc]
        
        if tokenized_corpus:
            try:
                bm25 = BM25Okapi(tokenized_corpus)
                doc_scores = bm25.get_scores(concept_tokens)
                max_score = max(doc_scores, default=0.0)
                if max_score >= bm25_threshold:
                    return True
            except Exception as e:
                logger.debug("BM25 scoring fallback do ngoại lệ: %s", e)

    # Fallback: Nếu tất cả các token quan trọng của concept đều có trong text
    return all(token in text_lower for token in concept_tokens)


# ==============================================================================
# LLM PROPOSE KEYWORDS 
# ==============================================================================
def _llm_propose_keywords(topic: str, profile: Optional[LearnerProfile] = None) -> List[str]:
    clean_topic = str(topic).replace("_", " ").title()
    keywords = [f"Definition of {clean_topic}"]

    if profile:
        level = getattr(profile, "level", "Beginner") or "Beginner"
        
        # Lấy danh sách và làm sạch từ khóa rỗng
        raw_goals = getattr(profile, "goals", []) or []
        raw_weak = getattr(profile, "weak_concepts", []) or []
        
        goals = [str(g).strip() for g in raw_goals if g and str(g).strip()]
        weak_concepts = [str(w).strip() for w in raw_weak if w and str(w).strip()]

        logger.info(
            "LLM phân tích LearnerProfile -> Level: %s | Goals: %s | Weak: %s",
            level, goals, weak_concepts
        )

        if weak_concepts:
            keywords.extend(weak_concepts)
        if goals:
            keywords.extend(goals)
            
        keywords.extend([
            f"Core concepts of {clean_topic}",
            f"Applications of {clean_topic}"
        ])
    else:
        logger.info("Chạy với profile mặc định cho topic '%s'", clean_topic)
        keywords.extend([
            f"Core concepts of {clean_topic}",
            f"Overview of {clean_topic}",
            f"Applications of {clean_topic}"
        ])

    # Trả về danh sách không trùng lặp, giữ nguyên thứ tự
    seen = set()
    return [x for x in keywords if not (x in seen or seen.add(x))]


# ==============================================================================
# CORE AGENT LOGIC
# ==============================================================================
def run_research(topic: str, profile: Optional[LearnerProfile] = None) -> ResearchBundle:
    # 1. LLM đề xuất keywords động hoàn toàn theo topic & profile
    proposed_keywords = _llm_propose_keywords(topic, profile)

    # 2. Đọc file KB
    kb_entries = read_kb_files(topic)

    sources_map: Dict[str, Source] = {}
    citations: List[Citation] = []
    covered: List[str] = []
    unfound_in_kb: List[str] = []

    # 3. Sparse Search trên KB (Hỗ trợ an toàn cả dict lẫn object)
    for kw in proposed_keywords:
        matched_entry: Any = find_concept_in_kb(kw, kb_entries)
        if matched_entry:
            # Phòng thủ trích xuất thuộc tính an toàn
            source_id = getattr(matched_entry, "source_id", None) or (
                matched_entry.get("source_id") if isinstance(matched_entry, dict) else "kb_file"
            )
            path = getattr(matched_entry, "path", None) or (
                matched_entry.get("path") if isinstance(matched_entry, dict) else "kb_path"
            )

            if source_id not in sources_map:
                sources_map[source_id] = Source(
                    source_id=str(source_id),
                    type="kb_file",
                    path_or_url=str(path),
                )
            citations.append(Citation(concept=kw, source_id=str(source_id)))
            if kw not in covered:
                covered.append(kw)
        else:
            unfound_in_kb.append(kw)

    # 4. Web Search bổ sung cho các keyword chưa tìm thấy trong KB
    unresolved: List[str] = []
    web_source_counter = 0
    search_fn = _default_web_search

    if unfound_in_kb:
        logger.warning(
            "run_research('%s'): %d keywords không thấy trong KB, chuyển qua Web Search: %s",
            topic, len(unfound_in_kb), unfound_in_kb,
        )

        for kw in unfound_in_kb:
            query = f"{str(topic).replace('_', ' ')} {kw}"
            raw_results = search_fn(query) or []
            context_text = _budget_context(raw_results)

            # Kiểm tra grounding bằng BM25
            if not context_text or not _ground_concept_in_text_bm25(kw, context_text):
                unresolved.append(kw)
                continue

            web_source_counter += 1
            source_id = f"web_{web_source_counter:02d}"
            matched_url = raw_results[0].get("url", "unknown") if raw_results else "unknown"

            sources_map[source_id] = Source(source_id=source_id, type="web", path_or_url=matched_url)
            citations.append(Citation(concept=kw, source_id=source_id))
            covered.append(kw)

    # Đảm bảo trả về dữ liệu hợp lệ
    if not covered:
        logger.error("Không tìm thấy concept nào cho topic '%s'", topic)

    bundle = ResearchBundle(
        topic=str(topic),
        sources=list(sources_map.values()),
        key_concepts=covered,
        citations=citations,
        unresolved_concepts=unresolved,
    )

    logger.info(
        "run_research('%s') Hoàn tất -> %d sources, %d key_concepts, %d citations, %d unresolved",
        topic, len(bundle.sources), len(covered), len(citations), len(unresolved),
    )
    return bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample_profile = LearnerProfile(
        level="Intermediate",
        goals=["Optimization"],
        weak_concepts=["Overfitting"]
    )
    
    res = run_research("logistic_regression", profile=sample_profile)
    print("\n[MOCK OUTPUT RESEARCH BUNDLE]:")
    print(res.model_dump_json(indent=2) if hasattr(res, "model_dump_json") else res)