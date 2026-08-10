from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, computed_field, model_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

SCHEMA_VERSION = "1.0.0"

# ------------------------------------------------------------------------------
# 1. MOCK KBENTRY & FIND_CONCEPT_IN_KB
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
# 2. SCHEMAS ĐẦY ĐỦ 100% CỦA HUY
# ------------------------------------------------------------------------------
class Source(BaseModel):
    source_id: str = Field(..., min_length=1)
    type: str
    path_or_url: str = Field(..., min_length=1)
    title: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=_utcnow)

class Citation(BaseModel):
    concept: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    quote: Optional[str] = None
    locator: Optional[str] = None

class ResearchBundle(BaseModel):
    topic: str = Field(..., min_length=1)
    sources: List[Source] = Field(default_factory=list)
    key_concepts: List[str] = Field(..., min_length=1)
    citations: List[Citation] = Field(default_factory=list)
    unresolved_concepts: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    common_pitfalls: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="before")
    @classmethod
    def _derive_unresolved(cls, data):
        if isinstance(data, dict) and not data.get("unresolved_concepts"):
            cited = set()
            for c in data.get("citations") or []:
                cited.add(c["concept"] if isinstance(c, dict) else c.concept)
            data = {
                **data,
                "unresolved_concepts": [
                    k for k in (data.get("key_concepts") or []) if k not in cited
                ],
            }
        return data

    @model_validator(mode="after")
    def _citations_point_to_real_sources(self) -> "ResearchBundle":
        known = {s.source_id for s in self.sources}
        dangling = sorted({c.source_id for c in self.citations} - known)
        if dangling:
            raise ValueError(
                f"citations trỏ tới source_id không tồn tại trong sources: {dangling}"
            )
        return self

    @computed_field
    @property
    def grounded_concepts(self) -> list[str]:
        cited = {c.concept for c in self.citations}
        return [k for k in self.key_concepts if k in cited]

# ------------------------------------------------------------------------------
# 3. DEMO RUN_RESEARCH
# ------------------------------------------------------------------------------
def _mock_llm_propose_keywords(topic: str) -> List[str]:
    return ["Cost Function", "Gradient Descent", "Overfitting", "L1 Regularization"]

def run_research_demo(topic: str) -> ResearchBundle:
    proposed_concepts = _mock_llm_propose_keywords(topic)
    kb_entries = mock_read_kb_files(topic)

    sources_map: Dict[str, Source] = {}
    citations: List[Citation] = []
    unfound_in_kb: List[str] = []

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
        else:
            unfound_in_kb.append(concept)

    if unfound_in_kb:
        logger.warning(f"Các khái niệm thiếu trong KB: {unfound_in_kb} -> Đẩy vào unresolved_concepts")

    # FIX LOGIC: key_concepts = proposed_concepts (chứa toàn bộ từ khóa)
    return ResearchBundle(
        topic=topic,
        sources=list(sources_map.values()),
        key_concepts=proposed_concepts,
        citations=citations,
        unresolved_concepts=unfound_in_kb
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 CHẠY DEMO KẾT HỢP KB_READER & DYNAMIC LLM SEARCH (NO WEB SEARCH)")
    print("="*60 + "\n")
    bundle = run_research_demo("logistic_regression")
    print("\n📦 KẾT QUẢ OUTPUT RESEARCH BUNDLE:")
    print(bundle.model_dump_json(indent=2))