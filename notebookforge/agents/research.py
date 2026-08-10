from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Sequence, Optional

from tools.kb_reader import KBEntry, find_concept_in_kb, read_kb_files

logger = logging.getLogger(__name__)

# SCHEMAS
try:
    from schemas import Citation, ResearchBundle, Source  # type: ignore
except ImportError:
    from pydantic import BaseModel, Field

    class Source(BaseModel):  # type: ignore[no-redef]
        source_id: str
        type: str  # "kb_file" | "web" | "paper" | "textbook"
        path_or_url: str

    class Citation(BaseModel):  # type: ignore[no-redef]
        concept: str
        source_id: str

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
    logger.info("Thực hiện Web Search query: '%s'", query)
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


def _ground_concept_in_text(concept: str, text: str) -> bool:
    pattern = re.compile(re.escape(concept.strip()), re.IGNORECASE)
    if pattern.search(text):
        return True
    tokens = [t for t in re.split(r"[\s()/']+", concept) if len(t) >= 3]
    return bool(tokens) and all(
        re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE) for t in tokens
    )


def _llm_propose_keywords(topic: str) -> List[str]:

    logger.info("LLM đang phân tích topic '%s' để đề xuất keywords...", topic)
    clean_topic = topic.replace("_", " ").title()
    
    # Đoạn này sau sẽ thay bằng call LLM thực tế
    return [
        f"Definition of {clean_topic}",
        "Cost Function",
        "Gradient Descent",
        "Overfitting",
        "L1 Regularization",
    ]


def run_research(topic: str) -> ResearchBundle:
    # 1. LLM đề xuất danh sách keywords động cho topic
    proposed_keywords = _llm_propose_keywords(topic)

    # 2. Đọc các file KB của topic
    kb_entries = read_kb_files(topic)

    sources_map: Dict[str, Source] = {}
    citations: List[Citation] = []
    covered: List[str] = []
    unfound_in_kb: List[str] = []

    # 3. Sparse Search: Tìm keywords trong nội dung KB bằng find_concept_in_kb
    for kw in proposed_keywords:
        matched_entry = find_concept_in_kb(kw, kb_entries)
        if matched_entry:
            if matched_entry.source_id not in sources_map:
                sources_map[matched_entry.source_id] = Source(
                    source_id=matched_entry.source_id,
                    type="kb_file",
                    path_or_url=matched_entry.path,
                )
            citations.append(Citation(concept=kw, source_id=matched_entry.source_id))
            if kw not in covered:
                covered.append(kw)
        else:
            unfound_in_kb.append(kw)

    # 4. Web Search cho các keywords chưa có trong KB
    unresolved: List[str] = []
    web_source_counter = 0
    search_fn = _default_web_search

    if unfound_in_kb:
        logger.warning(
            "run_research('%s'): %d keywords không thấy trong KB, chuyển qua Web Search: %s",
            topic, len(unfound_in_kb), unfound_in_kb,
        )

        for kw in unfound_in_kb:
            query = f"{topic.replace('_', ' ')} {kw}"
            raw_results = search_fn(query) or []
            context_text = _budget_context(raw_results)

            if not context_text or not _ground_concept_in_text(kw, context_text):
                unresolved.append(kw)
                continue

            web_source_counter += 1
            source_id = f"web_{web_source_counter:02d}"
            matched_url = raw_results[0].get("url", "unknown") if raw_results else "unknown"

            sources_map[source_id] = Source(source_id=source_id, type="web", path_or_url=matched_url)
            citations.append(Citation(concept=kw, source_id=source_id))
            covered.append(kw)

    if not covered:
        raise RuntimeError(
            f"run_research('{topic}'): Không tìm thấy concept grounded nào từ cả KB lẫn Web. "
            f"unresolved_concepts={unresolved}"
        )

    bundle = ResearchBundle(
        topic=topic,
        sources=list(sources_map.values()),
        key_concepts=covered,
        citations=citations,
        unresolved_concepts=unresolved,
    )

    logger.info(
        "run_research('%s') -> %d sources, %d key_concepts, %d citations, %d unresolved",
        topic, len(bundle.sources), len(covered), len(citations), len(unresolved),
    )
    return bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = run_research("logistic_regression")
    print("\nKết quả chạy thử:")
    print(res.model_dump_json(indent=2) if hasattr(res, "model_dump_json") else res)