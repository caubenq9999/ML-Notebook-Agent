from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

# Tích hợp BM25 Sparse Search
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("Thư viện 'rank_bm25' chưa cài đặt. Hệ thống sẽ bỏ qua BM25.")

KB_ROOT = Path(__file__).resolve().parent.parent / "kb"
REQUIRED_FRONTMATTER_FIELDS = ("doc_id", "topic", "key_concepts")

MIN_WORDS = 200
MAX_WORDS = 400

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_URL_RE = re.compile(r"https?://[^\s\"')]+")


class KBFormatError(ValueError):
    """File KB thiếu field bắt buộc hoặc sai định dạng frontmatter."""


@dataclass
class KBEntry:
    source_id: str            # = doc_id, vd "logreg_01"
    topic: str
    subtopic: str = ""
    level: str = ""
    key_concepts: List[str] = field(default_factory=list)
    source_url: str = ""      # rút ra từ source_url hoặc từ references trong body
    source_label: str = ""    # mô tả text của nguồn , luôn có
    body: str = ""
    path: str = ""            # path_or_url tương đối -- dùng cho Source.path_or_url
    word_count: int = 0

    @property
    def in_word_range(self) -> bool:
        return MIN_WORDS <= self.word_count <= MAX_WORDS


def _tokenize(text: str) -> List[str]:
    """Tách từ đơn giản phục vụ BM25."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise KBFormatError(
            "File KB phải bắt đầu bằng YAML frontmatter dạng "
            "'---\\n...\\n---\\n<nội dung>'"
        )
    raw_yaml, body = match.groups()
    try:
        meta = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as exc:
        raise KBFormatError(f"YAML frontmatter không hợp lệ: {exc}") from exc
    if not isinstance(meta, dict):
        raise KBFormatError("YAML frontmatter phải là một mapping (key: value)")
    return meta, body.strip()


def _count_words(body: str) -> int:
    body_no_code = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    return len(body_no_code.split())


def _resolve_source(meta: dict, body: str) -> tuple[str, str]:
    if meta.get("source_url"):
        url = str(meta["source_url"]).strip()
        return url, url

    sources = meta.get("sources")
    label = ""
    if isinstance(sources, list) and sources:
        label = str(sources[0])
    elif isinstance(sources, str):
        label = sources

    body_urls = _URL_RE.findall(body)
    url = body_urls[-1].rstrip(').,') if body_urls else ""
    return url, (label or url)


def parse_kb_file(path: Path) -> KBEntry:
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)

    missing = [f for f in REQUIRED_FRONTMATTER_FIELDS if f not in meta or meta[f] in (None, "", [])]
    if missing:
        raise KBFormatError(f"{path.name} thiếu field bắt buộc: {missing}")

    key_concepts = meta["key_concepts"]
    if isinstance(key_concepts, str):
        key_concepts = [c.strip() for c in key_concepts.split(",") if c.strip()]

    source_url, source_label = _resolve_source(meta, body)
    if not source_url:
        logger.warning(
            "%s: không tìm được URL nguồn (không có source_url, không có "
            "http(s) URL nào trong nội dung) -- source_label='%s'",
            path.name, source_label,
        )

    project_root = KB_ROOT.parent
    try:
        relative_path = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        relative_path = path

    return KBEntry(
        source_id=str(meta["doc_id"]),
        topic=str(meta["topic"]).strip().lower().replace(" ", "_"),
        subtopic=str(meta.get("subtopic", "")),
        level=str(meta.get("level", "")),
        key_concepts=[str(c).strip() for c in key_concepts],
        source_url=source_url,
        source_label=source_label,
        body=body,
        path=str(relative_path).replace("\\", "/"),
        word_count=_count_words(body),
    )


def read_kb_files(topic: str, *, enforce_word_range: bool = False) -> List[KBEntry]:
    folder_name = topic.strip().lower().replace(" ", "_")
    topic_dir = KB_ROOT / folder_name
    if not topic_dir.exists():
        return []

    entries: List[KBEntry] = []
    problems: List[str] = []
    seen_ids: dict[str, str] = {}
    for path in sorted(topic_dir.glob("*.md")):
        try:
            entry = parse_kb_file(path)
        except KBFormatError as exc:
            problems.append(str(exc))
            continue
        if entry.source_id in seen_ids:
            problems.append(
                f"{path.name}: doc_id '{entry.source_id}' TRÙNG với file "
                f"'{seen_ids[entry.source_id]}' -- bỏ qua file này vì source_id "
                f"phải là khoá duy nhất (Source/Citation dựa vào đây)."
            )
            continue
        seen_ids[entry.source_id] = path.name
        if enforce_word_range and not entry.in_word_range:
            problems.append(
                f"{path.name}: {entry.word_count} từ, ngoài khoảng "
                f"{MIN_WORDS}-{MAX_WORDS}"
            )
            continue
        if not entry.in_word_range:
            logger.info(
                "%s: %d từ, ngoài khoảng khuyến nghị %d-%d (đã được nhóm "
                "chấp thuận giữ nguyên, không phải lỗi).",
                path.name, entry.word_count, MIN_WORDS, MAX_WORDS,
            )
        entries.append(entry)

    if problems:
        logger.warning(
            "read_kb_files('%s'): %d file có vấn đề: %s",
            topic, len(problems), problems,
        )

    entries.sort(key=lambda e: e.source_id)
    return entries


def list_available_topics() -> List[str]:
    if not KB_ROOT.exists():
        return []
    return sorted(p.name for p in KB_ROOT.iterdir() if p.is_dir())


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
    return all(
        re.search(rf"\b{re.escape(t)}\b", entry.body, re.IGNORECASE)
        for t in tokens
    )


@dataclass
class KBIndex:
    """BM25 index dựng 1 LẦN cho 1 danh sách KBEntry, tái sử dụng cho nhiều lượt tìm concept.

    [FIX PERF] Trước đây find_concept_in_kb() tự dựng lại BM25Okapi (tokenize lại TOÀN BỘ
    corpus) ở MỖI lần gọi, trong khi research.py gọi hàm này bên trong vòng lặp keyword
    (N keyword -> N lần rebuild index của cùng 1 topic) -> lãng phí O(n_keywords * n_entries)
    không cần thiết. Giờ build 1 lần/topic bằng KBIndex.build(), rồi truyền index đó vào
    find_concept_in_kb() cho tất cả keyword trong cùng 1 lần run_research().
    """

    entries: List[KBEntry]
    bm25: Optional["BM25Okapi"] = None

    @classmethod
    def build(cls, entries: List[KBEntry]) -> "KBIndex":
        bm25 = None
        if HAS_BM25 and entries:
            corpus = [_tokenize(e.body) for e in entries]
            if any(corpus):
                bm25 = BM25Okapi(corpus)
        return cls(entries=entries, bm25=bm25)


def find_concept_in_kb(
    concept: str,
    entries_or_index: "List[KBEntry] | KBIndex",
    bm25_threshold: float = 0.5,
) -> Optional[KBEntry]:
    """
    Tìm kiếm khái niệm theo 3 bước:
    1. Match chính xác trong YAML frontmatter (key_concepts)
    2. Match bằng BM25 Full-text search trên toàn bộ body
    3. Match regex theo token bằng concept_is_grounded

    `entries_or_index` nhận List[KBEntry] (tương thích ngược, sẽ tự build KBIndex tạm --
    dùng cho gọi lẻ / test) hoặc KBIndex đã build sẵn (khuyến nghị khi gọi lặp trong vòng
    lặp, xem KBIndex ở trên).
    """
    index = entries_or_index if isinstance(entries_or_index, KBIndex) else KBIndex.build(entries_or_index)
    entries = index.entries
    if not entries:
        return None

    clean_kw = concept.strip().lower()

    # Tầng 1: Match trong key_concepts (YAML Frontmatter)
    for entry in entries:
        if any(clean_kw == c.strip().lower() for c in entry.key_concepts):
            return entry

    # Tầng 2: BM25 Full-text search (dùng index đã build sẵn, KHÔNG rebuild)
    if index.bm25 is not None:
        query_tokens = _tokenize(concept)
        if query_tokens:
            scores = index.bm25.get_scores(query_tokens)
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            max_score = scores[best_idx]
            if max_score > bm25_threshold:
                logger.info(f"BM25 match thành công '{concept}' trong '{entries[best_idx].source_id}' (Score: {max_score:.2f})")
                return entries[best_idx]

    # Tầng 3: Match Regex token fallback
    for entry in entries:
        if concept_is_grounded(concept, entry):
            return entry

    return None


if __name__ == "__main__":
    for e in read_kb_files("logistic_regression"):
        status = "OK" if e.in_word_range else f"NGOÀI KHOẢNG ({e.word_count} từ)"
        print(f"{e.source_id:12s} | {status:22s} | {e.source_url or '(không có URL)'}")
        print(f"             key_concepts={e.key_concepts}")