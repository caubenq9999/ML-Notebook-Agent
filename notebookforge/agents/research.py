from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tools.kb_reader import KBEntry, find_concept_in_kb, read_kb_files

logger = logging.getLogger(__name__)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

# Tích hợp BM25 với cơ chế Fallback
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("Thư viện 'rank_bm25' chưa được cài đặt. Hệ thống sẽ dùng Token Matching làm fallback.")

# ==============================================================================
# SCHEMAS 
# ==============================================================================
try:
    from schemas import Citation, LearnerProfile, ResearchBundle, Source  # type: ignore
except ImportError:
    from pydantic import BaseModel, Field, computed_field, model_validator

    SCHEMA_VERSION = "1.0.0"

    class Source(BaseModel):
        source_id: str = Field(..., min_length=1, description="vd: kb_01")
        type: str  # "kb_file" | "paper" | "textbook"
        path_or_url: str = Field(..., min_length=1)
        title: Optional[str] = None
        retrieved_at: datetime = Field(default_factory=_utcnow)

    class Citation(BaseModel):
        concept: str = Field(..., min_length=1)
        source_id: str = Field(..., min_length=1)
        quote: Optional[str] = Field(None, description="Trích dẫn nguyên văn, nếu có")
        locator: Optional[str] = Field(None, description="Heading / dòng trong file nguồn")

    class LearnerProfile(BaseModel):
        level: Optional[str] = "Beginner"
        goals: Optional[List[str]] = Field(default_factory=list)
        weak_concepts: Optional[List[str]] = Field(default_factory=list)
        known_concepts: Optional[List[str]] = Field(default_factory=list)

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


# ==============================================================================
# LLM PROPOSE KEYWORDS
# ==============================================================================
def _llm_propose_keywords(topic: str, profile: Optional[LearnerProfile] = None) -> List[str]:
    clean_topic = str(topic).replace("_", " ").title()
    keywords = []

    if profile:
        raw_goals = getattr(profile, "goals", []) or []
        raw_weak = getattr(profile, "weak_concepts", []) or []
        
        goals = [str(g).strip() for g in raw_goals if g and str(g).strip()]
        weak_concepts = [str(w).strip() for w in raw_weak if w and str(w).strip()]

        if weak_concepts:
            keywords.extend(weak_concepts)
        if goals:
            keywords.extend(goals)

    # Dự phòng nếu profile không cung cấp từ khóa
    if not keywords:
        keywords = [clean_topic, "Cost Function", "Gradient Descent", "Overfitting"]

    seen = set()
    return [x for x in keywords if not (x in seen or seen.add(x))]


# ==============================================================================
# CORE AGENT LOGIC
# ==============================================================================
def run_research(topic: str, profile: Optional[LearnerProfile] = None) -> ResearchBundle:
    # 1. LLM đề xuất keywords
    proposed_keywords = _llm_propose_keywords(topic, profile)

    # 2. Đọc file KB
    kb_entries = read_kb_files(topic)

    sources_map: Dict[str, Source] = {}
    citations: List[Citation] = []
    unresolved: List[str] = []

    # 3. Tra cứu trên KB
    for kw in proposed_keywords:
        matched_entry: Any = find_concept_in_kb(kw, kb_entries)
        if matched_entry:
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
        else:
            unresolved.append(kw)

    # 4. Tạo ResearchBundle chuẩn Schema
    bundle = ResearchBundle(
        topic=str(topic),
        sources=list(sources_map.values()),
        key_concepts=proposed_keywords,  # MUST: Truyền TOÀN BỘ khái niệm đề xuất
        citations=citations,
        unresolved_concepts=unresolved,
    )

    return bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample_profile = LearnerProfile(
        level="Intermediate",
        goals=["Optimization", "Gradient Descent"],
        weak_concepts=["Overfitting", "L1 Regularization"]
    )
    
    res = run_research("logistic_regression", profile=sample_profile)
    print("\n[OUTPUT RESEARCH BUNDLE]:")
    print(res.model_dump_json(indent=2))