from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Tích hợp BM25 Sparse Search
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("Thư viện 'rank_bm25' chưa cài đặt. Hệ thống sẽ bỏ qua BM25.")

# Tích hợp Embedding (Dense/Semantic Search) -- cùng triết lý graceful-fallback với BM25 ở
# trên: nếu chưa cài sentence-transformers, toàn hệ thống VẪN CHẠY được, chỉ tự động rớt
# xuống tầng BM25/regex cũ (xem find_concept_in_kb và semantic_chunk_entries bên dưới).
#
# LƯU Ý: numpy và sentence-transformers tách 2 try/except RIÊNG (không gộp chung) --
# numpy nhẹ và hầu như luôn có sẵn trong môi trường ML, nếu gộp chung mà chỉ thiếu
# sentence-transformers thì numpy cũng bị vô hiệu hoá theo, mất luôn phần cosine
# similarity dù máy hoàn toàn có thể chạy được phần đó.
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("Thư viện 'numpy' chưa cài đặt. Semantic chunking/embedding search sẽ tự fallback về BM25/regex.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("Thư viện 'sentence-transformers' chưa cài đặt. Semantic chunking/embedding search sẽ tự fallback về BM25/regex.")

HAS_EMBEDDINGS = HAS_NUMPY and HAS_SENTENCE_TRANSFORMERS

# Model nhỏ, chạy CPU tốt, đa ngôn ngữ (quan trọng vì KB + key_concepts có cả tiếng Việt
# lẫn thuật ngữ tiếng Anh) -- không gọi API, không tốn phí, chỉ tải 1 lần và cache local.
_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_EMBEDDING_MODEL: Optional["SentenceTransformer"] = None


def _get_embedding_model() -> Optional["SentenceTransformer"]:
    """Load model 1 LẦN DUY NHẤT (singleton), tái sử dụng cho mọi lần embed sau đó --
    tránh load lại model nặng ở mỗi lần gọi run_research()."""
    global _EMBEDDING_MODEL
    if not HAS_EMBEDDINGS:
        return None
    if _EMBEDDING_MODEL is None:
        try:
            _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        except Exception as exc:  # model chưa tải được (không mạng, thiếu ổ đĩa,...)
            logger.warning("Không load được embedding model (%s) -- fallback BM25/regex.", exc)
            return None
    return _EMBEDDING_MODEL


def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

KB_ROOT = Path(__file__).resolve().parent.parent / "kb"

# Mỗi file sau parse_kb_file phải có ít nhất những thành phần sau:
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


# ================================================================================
# topic -> kb/<topic>/ -> đọc tất cả các file -> parse_kb_file() -> list[KBEntry]
# ================================================================================
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


# =====================
# BỔ SUNG: class Chunk
# =====================
@dataclass
class Chunk:
    """1 đoạn nội dung thu được sau semantic chunking - đơn vị nhỏ nhất dùng để so khớp với
    key_concepts bằng embedding similarity (thay vì so khớp cả KBEntry.body rất dài)."""

    chunk_id: str          # source_id_<...>, vd "logreg_01_c0"
    source_id: str         # KBEntry.source_id chứa chunk này
    heading: str           # heading gần nhất bao quanh đoạn này (rỗng nếu văn bản mở đầu không heading)
    text: str              # Nội dung của chunk
    embedding: Optional["np.ndarray"] = None    # Vector ngữ nghĩa ứng với chunk này


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


# =======================================================================================
# Chia body thành các đoạn (heading, section_text) theo heading markdown (#, ##, ###).
# Phần văn bản trước heading đầu tiên (nếu có) được gán heading rỗng.
# =======================================================================================
def _split_by_heading(body: str) -> List[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [("", body.strip())] if body.strip() else []

    sections: List[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = body[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        if section_text:
            sections.append((heading, section_text))
    return sections


# ================================================================================
# Chia 1 section thành các đoạn (paragraph) theo dòng trống, GIỮ NGUYÊN khối code
# ================================================================================
def _split_into_paragraphs(section_text: str) -> List[str]:
    # Bảo vệ code fence bằng cách tạm thay newline bên trong bằng placeholder trước khi split.
    code_blocks: List[str] = []

    def _stash(match: "re.Match") -> str:
        code_blocks.append(match.group(0))
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

    protected = re.sub(r"```.*?```", _stash, section_text, flags=re.DOTALL)
    raw_paragraphs = re.split(r"\n\s*\n", protected)

    paragraphs = []
    for p in raw_paragraphs:
        for i, block in enumerate(code_blocks):
            p = p.replace(f"\x00CODEBLOCK{i}\x00", block)
        p = p.strip()
        if p:
            paragraphs.append(p)
    return paragraphs


# ==================================
# Cách hoạt động của semantic_chunk
# ==================================
def semantic_chunk_entry(
    entry: KBEntry,
    model: Optional["SentenceTransformer"] = None,
    similarity_threshold: float = 0.50,
    max_chunk_chars: int = 1200,
    min_chunk_chars: int = 180,
) -> List[Chunk]:
    """
    Semantic chunking có kiểm soát:
    - Không trộn nội dung các heading.
    - Các paragraph liên quan được gộp.
    - Chunk quá ngắn sẽ cố gắng gộp với paragraph kế tiếp.
    - Tránh tạo chunk chỉ có 1 câu hoặc code fragment.
    """
    sections = _split_by_heading(entry.body)

    chunks: List[Chunk] = []
    chunk_counter = 0

    for heading, section_text in sections:
        paragraphs = _split_into_paragraphs(section_text)
        if not paragraphs:
            continue
        # Fallback khi không có embedding:
        # giữ nguyên section nhưng không tạo chunk rỗng.
        if model is None:
            chunks.append(
                Chunk(
                    chunk_id=f"{entry.source_id}_c{chunk_counter}",
                    source_id=entry.source_id,
                    heading=heading,
                    text=section_text[:max_chunk_chars],
                )
            )
            chunk_counter += 1
            continue
        try:
            embeddings = model.encode(
                paragraphs,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning(
                "Embed paragraph lỗi (%s) -- fallback section nguyên khối.",
                exc,
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{entry.source_id}_c{chunk_counter}",
                    source_id=entry.source_id,
                    heading=heading,
                    text=section_text[:max_chunk_chars],
                )
            )
            chunk_counter += 1
            continue
        current_texts = [paragraphs[0]]
        current_vecs = [embeddings[0]]
        def flush_current():
            nonlocal chunk_counter
            if not current_texts:
                return
            text = "\n\n".join(current_texts)
            chunks.append(
                Chunk(
                    chunk_id=f"{entry.source_id}_c{chunk_counter}",
                    source_id=entry.source_id,
                    heading=heading,
                    text=text,
                    embedding=np.mean(current_vecs, axis=0),
                )
            )
            chunk_counter += 1
        for para, vec in zip(paragraphs[1:], embeddings[1:]):
            current_text = "\n\n".join(current_texts)
            current_len = len(current_text)
            running_mean = np.mean(
                current_vecs,
                axis=0,
            )
            similarity = _cosine_similarity(
                running_mean,
                vec,
            )
            fits_size = (
                current_len + len(para) + 2
                <= max_chunk_chars
            )

            # QUY TẮC: Nếu chunk hiện tại còn quá nhỏ, ưu tiên gộp paragraph kế tiếp để tránh fragment.
            if current_len < min_chunk_chars and fits_size:
                current_texts.append(para)
                current_vecs.append(vec)
            # Chunk đã đủ lớn thì mới dùng semantic similarity
            elif similarity >= similarity_threshold and fits_size:
                current_texts.append(para)
                current_vecs.append(vec)
            else:
                flush_current()
                current_texts = [para]
                current_vecs = [vec]
        flush_current()

    return chunks

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
    bm25: Optional["BM25Okapi"] = None                  # BM25 index của toàn bộ KB
    chunks: List[Chunk] = field(default_factory=list)   # Tất cả các semantic chunk
    chunk_embeddings: Optional["np.ndarray"] = None     # Vector embedding tương ứng với chunks

    @classmethod
    # Chỉ dùng KBIdex.build() một lần và reuse cho tất cả concept để đỡ tốn tài nguyên
    def build(cls, entries: List[KBEntry], *, use_embeddings: bool = True) -> "KBIndex":
        bm25 = None
        if HAS_BM25 and entries:
            corpus = [_tokenize(e.body) for e in entries]
            if any(corpus):
                bm25 = BM25Okapi(corpus)

        chunks: List[Chunk] = []
        chunk_embeddings = None
        model = _get_embedding_model() if (use_embeddings and HAS_EMBEDDINGS) else None
        # Cách embedding các chunks và lưu trong chunk_embeddings
        for entry in entries:
            chunks.extend(semantic_chunk_entry(entry, model=model))
        if model is not None and chunks:
            vecs = [c.embedding for c in chunks if c.embedding is not None]
            if len(vecs) == len(chunks):
                chunk_embeddings = np.vstack(vecs)

        return cls(entries=entries, bm25=bm25, chunks=chunks, chunk_embeddings=chunk_embeddings)


def find_concept_in_kb(
    concept: str,
    entries_or_index: "List[KBEntry] | KBIndex",
    bm25_threshold: float = 0.5,     # Đây là heuristic threshold, không đúng tuyệt đối
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

    # NẾU TẦNG TRÊN THẤT BẠI THÌ TỚI TẦNG DƯỚI
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

# ===================================================
# BỔ SUNG: tìm semantic chunk giống với các concept
# ===================================================
def extract_theory_chunks_for_concepts(
    key_concepts: List[str],
    index: "KBIndex",
    *,
    concept_source_map: Optional[Dict[str, str]] = None,    # match concept với source_id đã được Citation xđịnh
    top_k: int = 2,
    max_chunks_per_concept: int = 2,
    min_similarity: float = 0.5,
    min_chunk_chars: int = 70,
) -> List[dict]:

    if (
        not HAS_EMBEDDINGS
        or index.chunk_embeddings is None
        or not index.chunks
    ):
        return []

    concept_source_map = concept_source_map or {}

    model = _get_embedding_model()
    if model is None:
        return []

    results: Dict[str, dict] = {}

    def is_useful_chunk(chunk: Chunk) -> bool:

        text = chunk.text.strip()
        if len(text) < min_chunk_chars:
            return False

        # Bỏ chunk chỉ toàn code đơn giản
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]
        if not lines:
            return False

        # Chỉ có một dòng và trông giống code
        if len(lines) == 1:
            line = lines[0]
            code_signals = [
                "=",
                ".fit(",
                ".transform(",
                "import ",
            ]
            if any(signal in line for signal in code_signals):
                return False

        return True

    concept_embeddings = model.encode(
        key_concepts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    for concept, concept_embedding in zip(
        key_concepts,
        concept_embeddings,
    ):
        expected_source_id = concept_source_map.get(
            concept.strip().lower()
        )

        # 1. Candidate phải đúng source nếu đã có citation
        candidate_indices = []
        for i, chunk in enumerate(index.chunks):
            if not is_useful_chunk(chunk):
                continue
            if (
                expected_source_id
                and chunk.source_id != expected_source_id
            ):
                continue
            candidate_indices.append(i)
        if not candidate_indices:
            logger.info(
                "Không có candidate chunk cho '%s'",
                concept,
            )
            continue

        # 2. Tính similarity
        scored_candidates = []
        for i in candidate_indices:
            score = _cosine_similarity(
                concept_embedding,
                index.chunk_embeddings[i],
            )
            scored_candidates.append(
                (i, float(score))
            )

        # 3. Sort giảm dần
        scored_candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        # 4. Lấy top-k
        kept = 0

        for chunk_index, score in scored_candidates[:top_k]:
            if score < min_similarity:
                continue

            chunk = index.chunks[chunk_index]
            if chunk.chunk_id not in results:
                results[chunk.chunk_id] = {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "concepts": [],
                    "text": chunk.text,
                    "similarity": round(score, 4),
                }
            results[chunk.chunk_id]["concepts"].append(
                concept
            )
            results[chunk.chunk_id]["similarity"] = max(
                results[chunk.chunk_id]["similarity"],
                round(score, 4),
            )

            kept += 1
            if kept >= max_chunks_per_concept:
                break

        logger.info(
            "RAG '%s': kept=%d, source=%s",
            concept,
            kept,
            expected_source_id or "ALL",
        )
    return list(results.values())


if __name__ == "__main__":
    for e in read_kb_files("logistic_regression"):
        status = "OK" if e.in_word_range else f"NGOÀI KHOẢNG ({e.word_count} từ)"
        print(f"{e.source_id:12s} | {status:22s} | {e.source_url or '(không có URL)'}")
        print(f"             key_concepts={e.key_concepts}")