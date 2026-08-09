from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Sequence
from tools.kb_reader import KBEntry, concept_is_grounded, read_kb_files

logger = logging.getLogger(__name__)

# Research schema

try:
    from schemas import Citation, ResearchBundle, Source  # type: ignore
except ImportError:
    from pydantic import BaseModel, Field

    class Source(BaseModel):  # type: ignore[no-redef]
        source_id: str
        type: str  # Source type
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
    logger.info(
        query,
    )
    return []


def _budget_context(results: Sequence[Dict[str, str]], *, max_total_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Giới hạn độ dài context từ kết quả tìm kiếm."""
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


def _sources_and_citations_from_kb(entries: List[KBEntry]) -> tuple[List[Source], List[Citation], List[str], List[str]]:
    """Tạo sources và citations từ Knowledge Base."""
    sources: List[Source] = []
    citations: List[Citation] = []
    covered: List[str] = []
    ungrounded: List[str] = []

    for entry in entries:
        sources.append(
            Source(source_id=entry.source_id, type="kb_file", path_or_url=entry.path)
        )
        for concept in entry.key_concepts:
            if concept_is_grounded(concept, entry):
                citations.append(Citation(concept=concept, source_id=entry.source_id))
                if concept not in covered:
                    covered.append(concept)
            else:
                ungrounded.append(f"{concept} (khai trong {entry.source_id}, không thấy trong nội dung)")

    return sources, citations, covered, ungrounded


def run_research(
    topic: str,
    *,
    web_search_fn: Optional[WebSearchFn] = None,
    extra_wanted_concepts: Optional[List[str]] = None,
) -> ResearchBundle:
    """Tạo ResearchBundle từ Knowledge Base và web search."""
    kb_entries = read_kb_files(topic)
    sources, citations, covered, ungrounded = _sources_and_citations_from_kb(kb_entries)

    if ungrounded:
        logger.warning(
            "run_research('%s'): %d concept bị loại vì không grounded: %s",
            topic, len(ungrounded), ungrounded,
        )

    if not kb_entries:
        logger.warning(
            "run_research('%s'): chưa có file KB hợp lệ trong kb/%s/. "
            "Toàn bộ phụ thuộc vào web_search (nếu có).",
            topic, topic,
        )

    search_fn = web_search_fn or _default_web_search
    unresolved: List[str] = []
    web_source_counter = 0

    for concept in (extra_wanted_concepts or []):
        if concept in covered:
            continue
        query = f"{topic.replace('_', ' ')} {concept}"
        raw_results = search_fn(query) or []
        context_text = _budget_context(raw_results)

        if not context_text or not _ground_concept_in_text(concept, context_text):
            unresolved.append(concept)
            continue

        web_source_counter += 1
        source_id = f"web_{web_source_counter:02d}"
        matched_url = raw_results[0].get("url", "unknown") if raw_results else "unknown"
        sources.append(Source(source_id=source_id, type="web_search", path_or_url=matched_url))
        citations.append(Citation(concept=concept, source_id=source_id))
        covered.append(concept)

    bundle_kwargs = dict(
        topic=topic,
        sources=sources,
        key_concepts=covered,
        citations=citations,
    )
    try:
        bundle = ResearchBundle(**bundle_kwargs, unresolved_concepts=unresolved)
    except TypeError:
        # Hỗ trợ schema cũ chưa có unresolved_concepts.
        logger.info(
            "ResearchBundle của Hoàng không có field unresolved_concepts "
            "-- bỏ qua, chỉ log ở đây: %s", unresolved,
        )
        bundle = ResearchBundle(**bundle_kwargs)

    logger.info(
        "run_research('%s') -> %d sources, %d key_concepts, %d citations, %d unresolved",
        topic, len(sources), len(covered), len(citations), len(unresolved),
    )
    return bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_research("logistic_regression")
    dump = result.model_dump_json(indent=2) if hasattr(result, "model_dump_json") else result
    print(dump)