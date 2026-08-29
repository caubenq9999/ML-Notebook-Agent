"""
agents/curriculum.py
"""

from __future__ import annotations      # Hoãn việc đánh giá các type hint (class,...) khi chúng chưa được định nghĩa
import json                             # Thư viện thao tác với JSON
import re                               # Tìm kiếm, lọc, kiểm tra định dạng và thay thế chuỗi theo pattern        
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

class CurriculumGenerationError(Exception):
    """Raised khi Curriculum Agent không tạo được LearningPath hợp lệ sau tất cả các lần thử."""

# Chạy được schemas.py ngay trong thư mục agents
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from schemas import LearnerProfile, LearningPath, Module, ResearchBundle

from llm_client import call_json

# Gọi model để tạo "LearningPath" dưới dạng JSON
PROMPT_PATH_CURRICULUM = ROOT_DIR / "prompts" / "curriculum.txt"


# ==================
# Chốt là hai level
# ==================
LEVEL_NAMES = {1: "beginner", 2: "intermediate"}

def _normalize_level(level: int) -> int:
    """Ép mọi level lạ (vd 3, hoặc giá trị ngoài 1/2) về đúng 1 hoặc 2."""
    return level if level in (1 , 2) else 2


# ===========================================
# beginner: 2-3 bài | intermediate: 4-5 bài.
# ===========================================
EXERCISE_RANGE_BY_LEVEL: dict[int, tuple[int, int]] = {
    1: (2, 3),
    2: (4, 5),
}

def get_exercise_range(level: int) -> tuple[int, int]:
    return EXERCISE_RANGE_BY_LEVEL[_normalize_level(level)]

def clamp_num_exercises(level: int, requested: int) -> int:
    """
    Ép số bài tập mong muốn của user về đúng khoảng cho phép theo level:
        + Nếu "requested" nhỏ hơn "lowest" thì tạo "lowest" bài
        + Nếu "requested" lớn hơn "highest" thì tạo "highest" bài
        + Nếu "requested" nằm giữa "lowest" và "highest" thì tạo đúng "requested" bài
    Tuy các bài tập theo level là (2,3) và (4,5) không có khoảng giữa nhưng code như thế để 
    dễ nâng cấp hệ thống sau này (nếu có)
    """
    lowest, highest = get_exercise_range(level)
    return max(lowest, min(highest, requested))


# ===========================================================
# prompt cho việc nếu một bài có nhiều cách với cả hai level
# ===========================================================
LEVEL_DEPTH_NOTE = {
    1: (
        "BEGINNER(level 1): ở MỌI giai đoạn, nếu có nhiều cách làm hợp lệ, "
        "chỉ chọn ĐÚNG MỘT cách — là cách tốt nhất/phổ biến nhất "
        "(Bạn có thể xem qua các cách CHỈ CÓ TRONG key_concept và có thể tìm kiếm trên mạng "
        "để so sánh CHỈ TRONG CÁC CÁCH ĐÓ: cách nào phù hợp với bài toán nhất)"
    ),
    2: (
        "INTERMEDIATE(level 2): ở những giai đoạn có nhiều cách làm hợp "
        "lệ, ƯU TIÊN đưa nhiều cách vào cùng 1 bài tập dưới dạng yêu "
        "cầu 'cài đặt cả 2 cách rồi so sánh và chọn ra cách phù hợp hơn cho bài toán này'. "
        "Dùng type='analysis' cho bài tập so sánh này. "
        "Nếu đã hết số lượng bài tập theo level (4-5 bài) thì phần objective của module "
        "này phải nói đến việc so sánh những cách gì và so sánh như thế nào"
    ),
}

def build_level_depth_note(level: int) -> str:
    return LEVEL_DEPTH_NOTE[_normalize_level(level)]


# =============================================================
# Loại bài toán chi phối pipeline tổng quan:
# học có giám sát / phân loại và học không giám sát / phân cụm
# =============================================================
SUPERVISED_CLASSIFICATION = "supervised_classification"
UNSUPERVISED_CLUSTERING = "unsupervised_clustering"

TOPIC_PROBLEM_TYPE: dict[str, str] = {
    "logistic_regression": SUPERVISED_CLASSIFICATION,
    "decision_tree": SUPERVISED_CLASSIFICATION,
    "kmeans": UNSUPERVISED_CLUSTERING,
    "k-means": UNSUPERVISED_CLUSTERING,
    "k_means": UNSUPERVISED_CLUSTERING,
}

PROBLEM_TYPE_LABEL_VI = {
    SUPERVISED_CLASSIFICATION: "học có giám sát/phân loại (supervised classification)",
    UNSUPERVISED_CLUSTERING: "học không giám sát/phân cụm (unsupervised clustering)",
}

# lấy dạng bài toán (học có giám sát / học không có giám sát) 
def get_problem_type(topic: str) -> str:
    key = str(topic).strip().lower().replace(" ", "_").replace("-", "_")
    return TOPIC_PROBLEM_TYPE.get(key, SUPERVISED_CLASSIFICATION)


# ====================================================================
# PIPELINE TỔNG QUAN — đúc kết từ cấu trúc thật của KB
# 2 pipeline khác nhau cho 2 dạng bài toán:
#   - Supervised classification (logistic_regression, decision_tree)
#   - Unsupervised clustering (kmeans)
# ====================================================================

SUPERVISED_PIPELINE_BLOCK = """\
GIAI ĐOẠN 1 — Nền tảng lý thuyết (BẮT BUỘC)
  Giới thiệu đây là bài toán gì, giải quyết vấn đề gì trong học máy và công thức toán học cốt lõi, CHỈ dựa trên các
  "key_concepts" thuộc nhóm lý thuyết được cung cấp (ví dụ: logistic_regression/hàm sigmoid/odds ratio/log loss/...,
  hoặc decision_tree/Gini Impurity/Entropy/Information Gain/...). Nêu input/output của bài toán

  [CÓ NHIỀU CÁCH LÀM] nếu "key_concepts" có nhiều công thức đo cùng một thứ (ví dụ: Gini
  Impurity và Entropy đều dùng để đo độ hỗn loạn tại 1 node) — xem mục ĐỘ SÂU NỘI DUNG
  THEO LEVEL để quyết định trình bày 1 hay nhiều công thức.

GIAI ĐOẠN 2 — Cài đặt với scikit-learn (BẮT BUỘC)
  Giới thiệu class chính trong scikit-learn tương ứng với thuật toán (ví dụ:
  LogisticRegressionScratch cho bài toán logistic_regression, DecisionTreeClassifier cho bài toán decision_tree), 
  các phương thức cốt lõi (.fit/.predict/.predict_proba/...), và các hyperparameter QUAN TRỌNG NHẤT có trong key_concepts (ví dụ:
  penalty/solver/class_weight/... cho logistic_regression, hoặc max_depth/min_samples_split/class_weight/... cho decision_tree).

  [CÓ NHIỀU CÁCH LÀM] nếu "key_concepts" liệt kê nhiều lựa chọn cho cùng 1 hyperparameter
  (ví dụ nhiều solver, hoặc criterion = 'gini' vs 'entropy') — xem mục ĐỘ SÂU NỘI DUNG THEO
  LEVEL để quyết định trình bày 1 hay nhiều công thức.

GIAI ĐOẠN 3 — Tiền xử lý dữ liệu (BẮT BUỘC, nhưng nội dung PHỤ THUỘC THUẬT TOÁN)
  Dựa đúng vào "key_concepts" được cung cấp để biết những phương pháp gì đã được dùng để xử lý dataset — ví dụ:
  * Nếu "key_concepts" có StandardScaler/RobustScaler: BẮT BUỘC dạy scaling và giải thích vì sao thiếu
    bước này sẽ làm gradient/khoảng cách bị lệch.
  * Nếu key_concepts có "Scale Invariance" (thuật toán dạng cây, như decision_tree):
    KHÔNG dạy scaling, giải thích rõ vì sao mô hình cây không cần, giải quyết vấn đề liên quan đến High Cardinality Bias.
  * Nếu key_concepts có Categorical Encoding (Ordinal vs One-Hot Encoding): giải thích vì sao cần Encode
  các biến không phải biến số
  * Nếu key_concepts có xử lý missing/outlier(median, drop_duplicates,...): giải thích cách xử lý, ý nghĩa.

  [CÓ NHIỀU CÁCH LÀM] nếu "key_concepts" có nhiều lựa chọn cùng giải quyết 1 vấn đề 
  (ví dụ One-Hot Encoding vs Ordinal Encoding, hoặc StandardScaler vs RobustScaler)
  — xem mục ĐỘ SÂU NỘI DUNG THEO LEVEL để quyết định trình bày 1 hay nhiều công thức.

GIAI ĐOẠN 4 — Đánh giá mô hình (BẮT BUỘC)
  Dựa vào "key_concepts" để chọn đúng bộ metric: Ma trận nhầm lẫn/Precision/Recall/
  F1-Score/ROC-AUC/... (cho bài toán phân loại theo xác suất như logistic_regression),
  plot_tree/export_text/Decision Path (cho bài toán cần tính diễn giải cao như decision_tree),
  Silhouette/Davies-Bouldin/... (cho k-means).

GIAI ĐOẠN 5 — Common Pitfalls & Best Practices [TÙY CHỌN - CHỈ THÊM NẾU key_concepts CÓ
  các khái niệm dạng này, ví dụ: Overfitting/Data Leakage/Cost-Complexity Pruning/Extrapolation Limitations] 
  Nếu có, tóm tắt các lỗi thường gặp và cách khắc phục (ví dụ overfitting -> pruning /
  giảm max_depth). Nếu "key_concepts" KHÔNG có nhóm khái niệm này, BỎ QUA giai đoạn này
  hoàn toàn, không tự bịa thêm kiến thức ngoài phạm vi.\
"""

UNSUPERVISED_PIPELINE_BLOCK = """\
GIAI ĐOẠN 1 — Nền tảng lý thuyết (BẮT BUỘC)
  Giới thiệu đây là bài toán gì, giải quyết vấn đề gì trong học máy và công thức toán học cốt lõi
  khái niệm Centroid ("Centroids"), khoảng cách Euclidean, hàm mục tiêu WCSS/Inertia ("WCSS (Inertia)"), 
  thuật toán Lloyd's ("Lloyd's Algorithm": Assignment step + Update step), cùng
  chiến lược khởi tạo K-Means++ ("K-Means++ Initialization") nếu có trong "key_concepts".

GIAI ĐOẠN 2 — Cài đặt với scikit-learn (BẮT BUỘC)
  Giới thiệu class KMeans (sklearn.cluster.KMeans), các phương thức cốt lõi (.fit/.fit_predict/.transform/.cluster_centers_/.inertia_), 
  và các hyperparameter quan trọng CÓ TRONG key_concepts (n_clusters/ init/ n_init/ max_iter/ tol/...).
  
  [CÓ NHIỀU CÁCH LÀM] nếu "key_concepts" có nhiều lựa chọn ('algorithm': 'lloyd' vs
  'elkan', hoặc init='k-means++' vs 'random') — xem mục ĐỘ SÂU NỘI DUNG THEO LEVEL.

GIAI ĐOẠN 3 — Tiền xử lý & Scaling (BẮT BUỘC)
  NHẤN MẠNH: StandardScaler LÀ BẮT BUỘC cho K-Means (vì thuật toán chịu ảnh hưởng lớn bởi
  khoảng cách Euclidean, biến có biên độ lớn sẽ chi phối toàn bộ khoảng cách...). 
  Nêu ảnh hưởng của Outlier lên WCSS. Giới thiệu PCA cho dữ liệu nhiều chiều ("PCA Integration").
  Categorical Encoding (Ordinal vs One-Hot Encoding): giải thích vì sao cần Encode các biến không phải biến số.
  
  [CÓ NHIỀU CÁCH LÀM] việc CÓ dùng PCA hay KHÔNG (và chọn bao nhiêu n_components) là một
  lựa chọn có ảnh hưởng lớn — xem mục ĐỘ SÂU NỘI DUNG THEO LEVEL.

GIAI ĐOẠN 4 — Chọn K tối ưu & Đánh giá (BẮT BUỘC)
  QUAN TRỌNG: đây là bài toán KHÔNG GIÁM SÁT nên KHÔNG có accuracy/nhãn thật. Dựa vào
  "key_concepts" để chọn đúng phương pháp: Elbow Method : WCSS theo K ("Elbow Method"), Silhouette
  Coefficient (khoảng [-1,1]) ("Silhouette Coefficient"), "Davies-Bouldin Index" nếu có.

  [CÓ NHIỀU CÁCH LÀM] Elbow Method và Silhouette Coefficient là 2 cách phổ biến để chọn K
  — xem mục ĐỘ SÂU NỘI DUNG THEO LEVEL để quyết định dùng 1 hay kết hợp cả 2.

GIAI ĐOẠN 5 — Common Pitfalls & Best Practices
  [TÙY CHỌN - CHỈ THÊM NẾU key_concepts CÓ các khái niệm dạng này, ví dụ: Spherical Cluster Assumption, Unequal Cluster Sizes]:
  Nếu có, tóm tắt các giới hạn của K-Means (giả định cụm hình cầu ("Spherical Cluster Assumption"), nhạy cảm với khởi tạo,
  không xử lý tốt cụm không đều ("Unequal Cluster Sizes")) và khi nào nên đổi sang thuật toán khác (DBSCAN, Agglomerative). 
  Nếu key_concepts KHÔNG có nhóm khái niệm này thì bỏ qua giai đoạn này.\
"""

# lấy prompt để tạo PIPELINE bài học cho dạng bài toán (học có giám sát / học không có giám sát)
def build_pipeline_block(topic: str) -> str:
    if get_problem_type(topic) == UNSUPERVISED_CLUSTERING:
        return UNSUPERVISED_PIPELINE_BLOCK
    return SUPERVISED_PIPELINE_BLOCK


# =====================================================================
# RAG: đọc ResearchBundle.theory_chunks (do research.py/kb_reader.py sinh
# ra bằng semantic chunking + embedding) để (a) đưa bản RÚT GỌN vào prompt
# giúp LLM chọn concepts/objective/nhóm module chính xác hơn, và (b) sau
# khi LLM trả JSON, TỰ ĐỘNG gán Module.theory_context bằng code Python
# (tra cứu thuần, KHÔNG qua LLM) để Notebook Gen dùng sau này.
# =====================================================================
THEORY_PREVIEW_CHARS = 150


def build_theory_reference_block(bundle: ResearchBundle) -> str:
    """Bản RÚT GỌN cho prompt: mỗi chunk 1 dòng, nêu các concept cùng nhóm + preview ngắn
    (KHÔNG đưa full text vào đây để không phình prompt -- full text chỉ dùng ở bước (b)
    thông qua Module.theory_context, do code gán thẳng, LLM không cần thấy lại)."""
    if not bundle.theory_chunks:
        return (
            "(Không có dữ liệu RAG cho topic này -- tự viết theo hiểu biết chung, "
            "vẫn PHẢI bám đúng key_concepts đã cho, không tự bịa khái niệm khác.)"
        )

    lines = []
    for chunk in bundle.theory_chunks:
        preview = chunk.text.replace("\n", " ").strip()
        if len(preview) > THEORY_PREVIEW_CHARS:
            preview = preview[:THEORY_PREVIEW_CHARS].rstrip() + "..."
        concepts_str = ", ".join(chunk.concepts)
        lines.append(f"- [{concepts_str}] (nguồn {chunk.source_id}): {preview}")
    return "\n".join(lines)


def attach_theory_context(modules: list[dict], bundle: ResearchBundle) -> None:
    """Gán Module.theory_context = {concept: text KB thật} cho từng module, tra cứu
    trực tiếp trong bundle.theory_chunks theo concepts của module -- THUẦN PYTHON,
    không bịa lại qua LLM, để nội dung Notebook Gen viết sau này bám đúng KB gốc.
    Mutate `modules` (list[dict]) tại chỗ."""
    concept_to_text: dict[str, str] = {}
    for chunk in bundle.theory_chunks:
        for concept in chunk.concepts:
            concept_to_text.setdefault(concept, chunk.text)

    for m in modules:
        theory_context = {
            c: concept_to_text[c] for c in m.get("concepts", []) if c in concept_to_text
        }
        if theory_context:
            m["theory_context"] = theory_context


# ====================
# 1. Xây dựng prompt
# ====================
# Đọc nội dung prompt từ file curriculum.txt
def load_prompt() -> str:
    return PROMPT_PATH_CURRICULUM.read_text(encoding = "utf-8")

# Hàm tạo prompt để yêu cầu LLM trả về LearningPath
def build_prompt_curriculum(bundle : ResearchBundle, profile : LearnerProfile) -> str:
    # lưu prompt từ file curriculum.txt 
    template = load_prompt()

    # lấy dạng bài toán (học có giám sát / học không có giám sát)
    level = _normalize_level(profile.level_final)
    problem_type = get_problem_type(bundle.topic)
    problem_type_label = PROBLEM_TYPE_LABEL_VI[problem_type]

    # lấy số lượng bài tập theo level
    lowest, highest = get_exercise_range(level)
    exercise_range_hint = f"{lowest}-{highest} bài tập (level {level}={LEVEL_NAMES[level]})"

    prompt = template.replace("{topic}",str(bundle.topic))
    prompt = prompt.replace("{problem_type_label}", problem_type_label)
    prompt = prompt.replace("{pipeline_block}", build_pipeline_block(bundle.topic))
    prompt = prompt.replace("{level_depth_note}", build_level_depth_note(level))
    prompt = prompt.replace("{final_level}" , str(level))
    prompt = prompt.replace("{key_concepts}", json.dumps(bundle.key_concepts, ensure_ascii=False))
    prompt = prompt.replace("{duration_minutes}" , str(profile.constraints.duration_minutes))
    prompt = prompt.replace("{num_exercises}" , str(profile.constraints.num_exercises))
    prompt = prompt.replace("{exercise_range_hint}", exercise_range_hint)
    prompt = prompt.replace("{theory_reference_block}", build_theory_reference_block(bundle))

    # Trả về prompt
    return prompt


# =======================================================
# 2. Validate và tự động điều chỉnh cho khớp constraints
# =======================================================
def validate_and_adjust(data: dict, profile: LearnerProfile, bundle: Optional[ResearchBundle] = None) -> tuple[dict, list[str]]:
    """
    + total_estimated_minutes: nếu lệch thì tự rescale để.
    + num_exercises_planned: cảnh báo, không tự sửa.
    + số bài tập theo level (2-3 / 4-5): cảnh báo, không tự sửa.
    + source_ids: cảnh báo nếu tham chiếu tới source không tồn tại trong bundle.
    """
    warnings : list[str] = []               # list rỗng để chứa các cảnh báo
    modules = data.get("modules", [])       # data: dictionary vừa chuyển từ response.text và lấy các module
    level = _normalize_level(profile.level_final)

    #----------------------------Kiểm tra thời gian--------------------------
    target_minutes = profile.constraints.duration_minutes
    if target_minutes is not None and modules:
        current_total_minutes = sum(_module_["estimated_minutes"] for _module_ in modules)

        # Nếu thời gian ước tính khác so với thời gian người dùng mong muốn
        if current_total_minutes != target_minutes and current_total_minutes > 0:
            # Thêm cảnh báo vào list warnings
            warnings.append(
                f"total_estimated_minutes bị lệch ({current_total_minutes} != {target_minutes}). "
                "Đã tự rescale."
            )

            # Tự sacle lại thời gian
            ratio = target_minutes / current_total_minutes      # Tỷ lệ điều chỉnh
            track_change_total = 0                              # Theo dõi tổng thời gian các module sau điều chỉnh
            for i, m in enumerate(modules):
                if i < len(modules) - 1:
                    new_time = max(5 , round(m["estimated_minutes"] * ratio))       # 5: giới hạn thời gian nhỏ nhất của một module
                    m["estimated_minutes"] = new_time
                    track_change_total += new_time
                else:
                    # module cuối nhận phần thời gian dư còn lại
                    m["estimated_minutes"] = max(5, target_minutes - track_change_total)

        data["total_estimated_minutes"] = sum(m["estimated_minutes"] for m in modules)

    #--------------------------Kiểm tra số lượng bài tập (so với num_exercises gốc)--------------------------
    target_exercises = profile.constraints.num_exercises
    actual_exercises = sum(len(m.get("planned_exercises", [])) for m in modules)
    if actual_exercises != target_exercises:
        warnings.append(
            f"Tổng planned_exercises thực tế ({actual_exercises}) lệch với "
            f"constraints.num_exercises ({target_exercises}). Xử lý ở bước sau, KHÔNG tự sửa."
        )

    #--------------------------Kiểm tra số lượng bài tập theo LEVEL(2-3 / 4-5)--------------------------
    lo, hi = get_exercise_range(level)
    if not (lo <= actual_exercises <= hi):
        warnings.append(
            f"Tổng planned_exercises ({actual_exercises}) nằm ngoài khoảng cho phép theo "
            f"level {level} ({LEVEL_NAMES[level]}: {lo}-{hi} bài)."
        )

    #--------------------------Kiểm tra source_ids có trỏ tới source thật không--------------------------
    if bundle is not None:
        known_source_ids = {s.source_id for s in bundle.sources}
        dangling: set[str] = set()
        for m in modules:
            for sid in m.get("source_ids", []) or []:
                if sid not in known_source_ids:
                    dangling.add(sid)
        if dangling:
            warnings.append(
                f"source_ids sau đây không tồn tại trong ResearchBundle.sources: "
                f"{sorted(dangling)}. LLM có thể đang bịa source_id"
            )

    #--------------------------Kiểm tra RAG (theory_chunks) có gán được vào module không--------------------------
    if bundle is not None and bundle.theory_chunks:
        all_rag_concepts = {c for chunk in bundle.theory_chunks for c in chunk.concepts}
        all_module_concepts = {c for m in modules for c in m.get("concepts", [])}
        if not (all_rag_concepts & all_module_concepts):
            warnings.append(
                "ResearchBundle có theory_chunks nhưng KHÔNG concept nào của module trùng với "
                "concepts trong theory_chunks -- theory_context sẽ rỗng toàn bộ. Khả năng LLM "
                "đã tự diễn đạt lại concept khác với key_concepts gốc (xem YÊU CẦU #12)."
            )

    # Trả về LearningPath sau khi điều chỉnh (nếu bị lệch) và danh sách các cảnh báo
    return data, warnings


# ==============
# 3. Hàm chính
# ==============
def run_curriculum(
    bundle : ResearchBundle,
    profile : LearnerProfile,
) -> LearningPath:

    prompt_make_LearningPath = build_prompt_curriculum(bundle, profile)

    # nếu LLM trả JSON sai định dạng hoặc thiếu field bắt buộc, 
    # call_json có thể raise ValidationError/json.JSONDecodeError ngay tại đây
    try:
        LearningPath_raw, meta = call_json(
            prompt = prompt_make_LearningPath,
            schema = LearningPath,
            session_id = profile.session_id,
            reasoning_effort = "low"
        )
        data = LearningPath_raw.model_dump()

        data, warnings = validate_and_adjust(data, profile, bundle=bundle)
        for w in warnings:
            print(f"[Curriculum Agent] cảnh báo: {w}")

        # Gán Module.theory_context bằng code Python (KHÔNG qua LLM) -- xem docstring
        # attach_theory_context ở trên.
        attach_theory_context(data.get("modules", []), bundle)

        data["session_id"] = profile.session_id
        data["level"] = _normalize_level(profile.level_final)
        data["topic"] = profile.topic
        # Trả về class LearningPath theo đúng định dạng schema
        return LearningPath(**data)

    except (ValidationError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error_validate:
        error_message = f"{type(error_validate).__name__}: {error_validate}"
        print(
            f"[Curriculum Agent] Lỗi JSON: {error_message}"
        )

        # Không trả về None để biết đường sửa.
        raise CurriculumGenerationError(
            "Curriculum Agent tạo LearningPath không hợp lệ. "
            f"Lỗi: {error_message}"
        )


# =============================================================
# 4. Test nhanh bằng mock — chạy: python -m agents.curriculum
# =============================================================
if __name__ == "__main__":
    from tests.mocks import MOCK_BUNDLE, MOCK_PROFILE

    path = run_curriculum(MOCK_BUNDLE, MOCK_PROFILE)
    print(path.model_dump_json(indent = 2))