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
    """Ép số bài tập mong muốn của user về đúng khoảng cho phép theo level."""
    lowest, highest = get_exercise_range(level)
    return max(lowest, min(highest, requested))


# ===========================================================
# prompt cho việc nếu một bài có nhiều cách với cả hai level
# ===========================================================
LEVEL_DEPTH_NOTE = {
    1: (
        "Học viên hiện tại là BEGINNER (level 1): ở MỌI giai đoạn có nhiều cách làm hợp lệ, "
        "CHỈ chọn và trình bày ĐÚNG MỘT cách — là cách tốt nhất / phổ biến nhất / được khuyến "
        "nghị sử dụng với bài toán đó (Bạn có thể xem qua các cách CHỈ CÓ TRONG key_concept "
        "và có thể tìm kiếm trên mạng để so sánh các cách đó - CHỈ TRONG CÁC CÁCH ĐÓ: cách nào "
        "phù hợp với bài toán nhất)"
    ),
    2: (
        "Học viên hiện tại là INTERMEDIATE (level 2): ở những giai đoạn có nhiều cách làm hợp "
        "lệ, ĐƯỢC PHÉP (và khuyến khích) đưa nhiều cách vào cùng 1 module/bài tập dưới dạng yêu "
        "cầu 'cài đặt/thử cả 2 cách rồi so sánh và chọn ra cách phù hợp hơn cho bài toán này'. "
        "Ưu tiên dùng type='analysis' cho dạng bài tập so sánh này."
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
    SUPERVISED_CLASSIFICATION: "học có giám sát / phân loại (supervised classification)",
    UNSUPERVISED_CLUSTERING: "học không giám sát / phân cụm (unsupervised clustering)",
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
GIAI ĐOẠN 1 — Nền tảng lý thuyết - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/01_theoretical_foundations.md: (BẮT BUỘC)
  Giải thích bản chất thuật toán (giải quyết vấn đề gì trong học máy) và công thức toán học cốt lõi, CHỈ dựa trên các
  key_concepts thuộc nhóm lý thuyết được cung cấp (ví dụ: sigmoid function/odds ratio/log loss/... cho logistic_regression,
  hoặc Gini Impurity/Entropy/Information Gain/... cho decision_tree).

  Nêu rõ input/output của mô hình và ý nghĩa của decision boundary mà thuật toán tạo ra.

  [CÓ NHIỀU CÁCH LÀM] nếu key_concepts có nhiều công thức đo cùng một thứ (ví dụ: Gini
  Impurity và Entropy đều dùng để đo độ hỗn loạn tại 1 node) — xem mục ĐỘ SÂU NỘI DUNG
  THEO LEVEL để quyết định trình bày 1 hay nhiều công thức.

GIAI ĐOẠN 2 — Cài đặt với scikit-learn - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/02_scikit_learn_implementation.md :(BẮT BUỘC)
  Giới thiệu class chính trong scikit-learn tương ứng với thuật toán (ví dụ:
  LogisticRegression cho bài toán logistic_regression, DecisionTreeClassifier cho bài toán decision_tree), 
  các phương thức cốt lõi (.fit/.predict/.predict_proba/...), và các hyperparameter QUAN TRỌNG NHẤT có trong key_concepts (ví dụ:
  penalty/solver/class_weight/... cho logistic_regression, hoặc max_depth/min_samples_split/class_weight/... cho decision_tree).

  [CÓ NHIỀU CÁCH LÀM] nếu key_concepts liệt kê nhiều lựa chọn cho cùng 1 hyperparameter
  (ví dụ nhiều solver, hoặc criterion = 'gini' vs 'entropy') — xem mục ĐỘ SÂU NỘI DUNG THEO
  LEVEL để quyết định trình bày 1 hay nhiều công thức.

GIAI ĐOẠN 3 — Tiền xử lý dữ liệu - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/03_data_preprocessing_eda.md: (BẮT BUỘC, nhưng nội dung PHỤ THUỘC THUẬT TOÁN)
  Dựa đúng vào key_concepts được cung cấp để biết bước này gồm những gì — ví dụ:
  * Nếu key_concepts có StandardScaler/RobustScaler (thuật toán dựa trên khoảng cách
    hoặc gradient, như logistic_regression): BẮT BUỘC dạy scaling, giải thích vì sao thiếu
    bước này sẽ làm gradient/khoảng cách bị lệch.
  * Nếu key_concepts có "Scale Invariance" (thuật toán dạng cây, như decision_tree):
    KHÔNG dạy scaling (giải thích rõ vì sao mô hình cây không cần), tập trung vào
    Categorical Encoding (Ordinal vs One-Hot Encoding) và các vấn đề liên quan đến High Cardinality Bias.
  * Xử lý missing/outlier (SimpleImputer, IQR,...) nếu key_concepts có nhắc tới.

  [CÓ NHIỀU CÁCH LÀM] nếu key_concepts có nhiều lựa chọn encoding/scaling cùng giải quyết
  1 vấn đề (ví dụ One-Hot Encoding vs Ordinal Encoding, hoặc StandardScaler vs
  RobustScaler) — xem mục ĐỘ SÂU NỘI DUNG THEO LEVEL để quyết định trình bày 1 hay nhiều công thức.

  QUAN TRỌNG: bước này đã được HỆ THỐNG xử lý sẵn trong dữ liệu được chèn vào notebook
  (tools/dataset_injector.py lo phần scaling/encoding/missing-value/EDA) — module ứng với
  giai đoạn này KHÔNG ĐƯỢC có planned_exercises (xem YÊU CẦU BẮT BUỘC #13), chỉ dùng để giải
  thích lý thuyết qua "concepts".

GIAI ĐOẠN 4 — Đánh giá mô hình - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/04_model_evaluation.md: (BẮT BUỘC)
  Dựa vào key_concepts để chọn đúng bộ metric: Confusion Matrix/Precision/Recall/
  F1-Score/ROC-AUC/... (cho các bài toán phân loại theo xác suất như logistic_regression),
  hoặc plot_tree/export_text/ Decision Path (cho các mô hình có tính diễn giải cao như decision_tree).

GIAI ĐOẠN 5 — Common Pitfalls & Best Practices [TÙY CHỌN - CHỈ THÊM NẾU key_concepts CÓ
  các khái niệm dạng này, ví dụ: Overfitting/Data Leakage/Cost-Complexity Pruning/Extrapolation Limitations] 
  - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/05_common_pitfalls_best_practices.md:
  Nếu có, tóm tắt các lỗi thường gặp và cách khắc phục (ví dụ overfitting -> pruning /
  giảm max_depth). Nếu key_concepts KHÔNG có nhóm khái niệm này, BỎ QUA giai đoạn này
  hoàn toàn, không tự bịa thêm kiến thức ngoài phạm vi.\
"""

UNSUPERVISED_PIPELINE_BLOCK = """\
GIAI ĐOẠN 1 — Nền tảng lý thuyết - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/01_theoretical_foundations.md: (BẮT BUỘC)
  Giải thích bản chất thuật toán (giải quyết vấn đề gì trong học máy)
  Trình bày các khái niệm Centroid ("Centroids"), khoảng cách Euclidean ("Euclidean Distance"), hàm
  mục tiêu WCSS/Inertia ("WCSS (Inertia)"), thuật toán Lloyd's ("Lloyd's Algorithm": Assignment step + Update step), cùng
  chiến lược khởi tạo K-Means++ ("K-Means++ Initialization") nếu có trong key_concepts.

GIAI ĐOẠN 2 — Cài đặt với scikit-learn - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/02_scikit_learn_implementation.md: (BẮT BUỘC)
  Giới thiệu class KMeans (sklearn.cluster.KMeans), các phương thức cốt lõi (.fit/.fit_predict/.transform/.cluster_centers_/.inertia_), 
  và các hyperparameter quan trọng CÓ TRONG key_concepts (n_clusters/ init/ n_init/ max_iter/ tol/...).
  
  [CÓ NHIỀU CÁCH LÀM] nếu key_concepts có nhiều lựa chọn ('algorithm': 'lloyd' vs
  'elkan', hoặc init='k-means++' vs 'random') — xem mục ĐỘ SÂU NỘI DUNG THEO LEVEL.

GIAI ĐOẠN 3 — Tiền xử lý & Scaling - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/03_data_preprocessing_eda.md: (BẮT BUỘC)
  NHẤN MẠNH: StandardScaler LÀ BẮT BUỘC cho K-Means (vì thuật toán chịu ảnh hưởng lớn bởi
  khoảng cách Euclidean, biến có biên độ lớn sẽ chi phối toàn bộ khoảng cách). 
  Nêu ảnh hưởng của Outlier lên WCSS. Giới thiệu PCA cho dữ liệu nhiều chiều ("PCA Integration").

  [CÓ NHIỀU CÁCH LÀM] việc CÓ dùng PCA hay KHÔNG (và chọn bao nhiêu n_components) là một
  lựa chọn có ảnh hưởng lớn — xem mục ĐỘ SÂU NỘI DUNG THEO LEVEL.

  QUAN TRỌNG: bước này đã được HỆ THỐNG xử lý sẵn trong dữ liệu được chèn vào notebook
  (tools/dataset_injector.py lo phần scaling/PCA/EDA) — module ứng với giai đoạn này KHÔNG
  ĐƯỢC có planned_exercises (xem YÊU CẦU BẮT BUỘC #13), chỉ dùng để giải thích lý thuyết qua
  "concepts".

GIAI ĐOẠN 4 — Chọn K tối ưu & Đánh giá - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/04_model_evaluation.md: (BẮT BUỘC)
  QUAN TRỌNG: đây là bài toán KHÔNG GIÁM SÁT nên KHÔNG có accuracy/nhãn thật. Dựa vào
  key_concepts để chọn đúng phương pháp: Elbow Method : WCSS theo K ("Elbow Method"), Silhouette
  Coefficient (khoảng [-1,1]) ("Silhouette Coefficient"), Davies-Bouldin Index nếu có.

  [CÓ NHIỀU CÁCH LÀM] Elbow Method và Silhouette Coefficient là 2 cách phổ biến để chọn K
  — xem mục ĐỘ SÂU NỘI DUNG THEO LEVEL để quyết định dùng 1 hay kết hợp cả 2.

GIAI ĐOẠN 5 — Common Pitfalls & Best Practices - có thể tìm thêm nội dung ở các file có dạng kb/{problem_type_label}/05_common_pitfalls_best_practices.md (BẮT BUỘC)
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

    # lấy số lượng bài tập  theo level
    lowest, highest = get_exercise_range(level)
    exercise_range_hint = f"{lowest}-{highest} bài tập (level {level} = {LEVEL_NAMES[level]})"

    prompt = template.replace("{topic}",str(bundle.topic))
    prompt = prompt.replace("{problem_type_label}", problem_type_label)
    prompt = prompt.replace("{pipeline_block}", build_pipeline_block(bundle.topic))
    prompt = prompt.replace("{level_depth_note}", build_level_depth_note(level))
    prompt = prompt.replace("{final_level}" , str(level))
    prompt = prompt.replace( "{key_concepts}", json.dumps(bundle.key_concepts, ensure_ascii=False))
    prompt = prompt.replace("{duration_minutes}" , str(profile.constraints.duration_minutes))
    prompt = prompt.replace("{num_exercises}" , str(profile.constraints.num_exercises))
    prompt = prompt.replace("{exercise_range_hint}", exercise_range_hint)

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
            f"level {level} ({LEVEL_NAMES[level]}: {lo}-{hi} bài). Notebook Gen sẽ tự ép về "
            "khoảng này khi sinh cell TODO, nhưng nên sửa lại prompt/hoặc regen LearningPath "
            "nếu lệch quá xa."
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
                f"{sorted(dangling)}. LLM có thể đang bịa source_id — cần xem lại prompt "
                "(xem NOTE trong build_prompt_curriculum) hoặc hỏi anh Trí về format sources."
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
