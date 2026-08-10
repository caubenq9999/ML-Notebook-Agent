from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1. MOCK KBENTRY & FIND_CONCEPT_IN_KB (Khớp 100% logic file kb_reader.py của bạn)
# ------------------------------------------------------------------------------
@dataclass
class KBEntry:
    source_id: str
    topic: str
    body: str
    path: str

def concept_is_grounded(concept: str, entry: KBEntry) -> bool:
    needle = concept.strip()
    if not needle:
        return False
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    if pattern.search(entry.body):
        return True
    tokens = [t for t in re.split(r"[\s()/']+", needle) if len(t) >= 3]
    if not tokens:
        return False
    return all(re.search(rf"\b{re.escape(t)}\b", entry.body, re.IGNORECASE) for t in tokens)

def find_concept_in_kb(concept: str, entries: List[KBEntry]) -> Optional[KBEntry]:
    for entry in entries:
        if concept_is_grounded(concept, entry):
            return entry
    return None

def mock_read_kb_files(topic: str) -> List[KBEntry]:
    """Giả lập dữ liệu đọc file từ KB"""
    return [
        KBEntry(
            source_id="logreg_01",
            topic="logistic_regression",
            body="Definition of Logistic Regression is a classification algorithm. It uses Cost Function to calculate loss.",
            path="kb/logistic_regression/01_intro.md"
        ),
        KBEntry(
            source_id="logreg_02",
            topic="logistic_regression",
            body="We use Gradient Descent to minimize the loss in Logistic Regression.",
            path="kb/logistic_regression/02_optimization.md"
        )
    ]

# ------------------------------------------------------------------------------
# 2. SCHEMAS
# ------------------------------------------------------------------------------
class Source(BaseModel):
    source_id: str
    type: str
    path_or_url: str

class Citation(BaseModel):
    concept: str
    source_id: str

class ResearchBundle(BaseModel):
    topic: str
    sources: List[Source]
    key_concepts: List[str]
    citations: List[Citation]
    unresolved_concepts: List[str] = Field(default_factory=list)

# ------------------------------------------------------------------------------
# 3. DEMO RUN_RESEARCH
# ------------------------------------------------------------------------------
def _mock_llm_propose_keywords(topic: str) -> List[str]:
    """LLM đề xuất: 2 từ CÓ trong KB, 2 từ KHÔNG CÓ trong KB"""
    return ["Cost Function", "Gradient Descent", "Overfitting", "L1 Regularization"]

def run_research_demo(topic: str) -> ResearchBundle:
    proposed_concepts = _mock_llm_propose_keywords(topic)
    kb_entries = mock_read_kb_files(topic)

    sources_map: Dict[str, Source] = {}
    citations: List[Citation] = []
    covered: List[str] = []
    unfound_in_kb: List[str] = []

    # Quét qua KB với hàm find_concept_in_kb của kb_reader.py
    for concept in proposed_concepts:
        matched_entry = find_concept_in_kb(concept, kb_entries)
        if matched_entry:
            if matched_entry.source_id not in sources_map:
                sources_map[matched_entry.source_id] = Source(
                    source_id=matched_entry.source_id,
                    type="kb_file",
                    path_or_url=matched_entry.path
                )
            citations.append(Citation(concept=concept, source_id=matched_entry.source_id))
            covered.append(concept)
        else:
            unfound_in_kb.append(concept)

    # Đưa các từ không có trong KB thẳng vào unresolved_concepts
    unresolved: List[str] = []
    if unfound_in_kb:
        logger.warning(f"Các khái niệm thiếu trong KB: {unfound_in_kb} -> Đẩy vào unresolved_concepts")
        unresolved.extend(unfound_in_kb)

    return ResearchBundle(
        topic=topic,
        sources=list(sources_map.values()),
        key_concepts=covered,
        citations=citations,
        unresolved_concepts=unresolved
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 CHẠY DEMO KẾT HỢP KB_READER & DYNAMIC LLM SEARCH (NO WEB SEARCH)")
    print("="*60 + "\n")
    bundle = run_research_demo("logistic_regression")
    print("\n📦 KẾT QUẢ OUTPUT RESEARCH BUNDLE:")
    print(bundle.model_dump_json(indent=2))