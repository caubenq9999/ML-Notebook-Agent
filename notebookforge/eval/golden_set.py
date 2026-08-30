"""eval/golden_set.py - HUY

20 case kiểm chuẩn, ĐÓNG BĂNG từ tuần 1.

Nguyên tắc: điểm xấu thì sửa hệ thống, KHÔNG sửa golden set.
Sửa golden set để điểm đẹp là tự lừa mình.

Cấu trúc: 3 topic x 2 level = 6 tổ hợp kiến thức (_SPECS bên dưới), 20 case
sinh ra bằng cách ghép mỗi tổ hợp với các ràng buộc thời lượng / số bài tập
khác nhau. Ràng buộc mới là chỗ hệ thống hay vỡ: Curriculum Agent rất hay
sinh 6 module 90 phút trong khi học viên chỉ xin 60.

level dùng ĐÚNG kiểu schemas.Level (Literal[1, 2]) để harness dựng thẳng
LearnerProfile mà không phải map: 1 = beginner, 2 = intermediate.
Đề cương chỉ có 2 trình độ; level 3 (advanced) không còn trong schema.

Phân bổ:
    logistic_regression  7 case (4 beginner / 3 intermediate)  binary
    decision_tree        7 case (4 beginner / 3 intermediate)  multi-class
    kmeans               6 case (3 beginner / 3 intermediate)  clustering

metric_threshold dò bằng `python -m eval.calibrate` trên chính 3 dataset của
đề cương, 5 seed, ngưỡng = trung bình trừ biên an toàn 0.05:

    topic                dataset                     trung bình  ngưỡng
    logistic_regression  Heart Failure Prediction     acc 0.8522    0.80
    decision_tree        Red Wine Quality (3 nhóm)    acc 0.8344    0.78
    kmeans               Mall Customer Segmentation   sil 0.3147    0.26

Riêng silhouette phụ thuộc mạnh vào số feature notebook chọn: 4 feature ~0.31,
3 feature ~0.42, chỉ Income+Spending ~0.55. Ngưỡng lấy theo trường hợp thấp
nhất hợp lệ, nếu không notebook dùng đủ feature sẽ fail oan.
"""

# ---------------------------------------------------------------------------
# Kỳ vọng kiến thức theo (topic, level). Không phụ thuộc constraints.
# ---------------------------------------------------------------------------

_SPECS = {
    ("logistic_regression", 1): {
        "expected_modules": [
            "Giới thiệu",
            "Cost function",
            "Gradient descent",
            "Thực hành với sklearn",
        ],
        "expected_skills": ["binary classification", "sigmoid", "log loss"],
        "must_not_have": ["neural network", "softmax"],
        "metric": "accuracy",
        "threshold": 0.80,
    },
    ("logistic_regression", 2): {
        "expected_modules": [
            "Ôn sigmoid và log loss",
            "Regularization L1/L2",
            "Đánh giá trên dữ liệu mất cân bằng",
            "Thực hành với sklearn",
        ],
        "expected_skills": [
            "regularization",
            "decision boundary",
            "roc auc",
            "precision recall",
        ],
        "must_not_have": ["neural network", "deep learning"],
        "metric": "accuracy",
        "threshold": 0.80,
    },
    ("decision_tree", 1): {
        "expected_modules": [
            "Entropy & Information Gain",
            "Tree building",
            "Pruning",
            "Thực hành",
        ],
        "expected_skills": ["gini impurity", "entropy", "feature importance"],
        "must_not_have": ["random forest ensemble", "XGBoost"],
        "metric": "accuracy",
        "threshold": 0.78,
    },
    ("decision_tree", 2): {
        "expected_modules": [
            "Entropy so với Gini",
            "Overfitting và pruning",
            "Tinh chỉnh siêu tham số",
            "Thực hành",
        ],
        "expected_skills": [
            "max_depth",
            "min_samples_leaf",
            "cost complexity pruning",
            "feature importance",
        ],
        "must_not_have": ["gradient boosting", "XGBoost", "neural network"],
        "metric": "accuracy",
        "threshold": 0.78,
    },
    ("kmeans", 1): {
        "expected_modules": [
            "Ý tưởng phân cụm",
            "Thuật toán K-Means",
            "Chọn K bằng Elbow",
            "Thực hành",
        ],
        "expected_skills": ["centroid", "euclidean distance", "inertia"],
        "must_not_have": ["DBSCAN", "hierarchical clustering"],
        "metric": "silhouette",
        "threshold": 0.26,
    },
    ("kmeans", 2): {
        "expected_modules": [
            "Distance metrics",
            "K-means algorithm",
            "Elbow method",
            "Thực hành",
        ],
        "expected_skills": [
            "euclidean distance",
            "inertia",
            "silhouette score",
            "feature scaling",
        ],
        "must_not_have": ["hierarchical clustering", "DBSCAN"],
        "metric": "silhouette",
        "threshold": 0.26,
    },
}

# Dataset dùng để dò metric_threshold, theo đề cương phần 3.
DATASET_OF = {
    "logistic_regression": "Heart Failure Prediction",
    "decision_tree": "Red Wine Quality",
    "kmeans": "Mall Customer Segmentation",
}


def _case(case_id, topic, level, duration, num_exercises, note=""):
    """Ghép kỳ vọng kiến thức với một bộ ràng buộc cụ thể."""
    spec = _SPECS[(topic, level)]
    return {
        "id": case_id,
        "topic": topic,
        "level": level,                 # 1 = beginner, 2 = intermediate
        "constraints": {
            "duration_minutes": duration,
            "num_exercises": num_exercises,
        },
        "expected_modules": spec["expected_modules"],
        "expected_skills": spec["expected_skills"],
        "must_not_have": spec["must_not_have"],
        # beginner cần ít assert hơn, intermediate phải tự kiểm nhiều hơn
        "min_asserts": 3 if level == 1 else 4,
        # Dò bằng eval/calibrate.py trên dataset thật, KHÔNG dùng 0.90 mặc
        # định của sklearn (đó là số trên toy dataset).
        "metric_threshold": {spec["metric"]: spec["threshold"]},
        "note": note,
    }


# ---------------------------------------------------------------------------
# 20 case. Đóng băng từ đây.
# ---------------------------------------------------------------------------

GOLDEN_SET = [
    # --- Logistic Regression / beginner (4) ------------------------------
    _case("GS-001", "logistic_regression", 1, 60, 3, "case chuẩn, ngắn nhất"),
    _case("GS-002", "logistic_regression", 1, 90, 4),
    _case("GS-003", "logistic_regression", 1, 120, 5, "dài nhất, dễ lan man"),
    _case("GS-004", "logistic_regression", 1, 60, 5, "CHẬT: 5 bài trong 60 phút"),
    # --- Logistic Regression / intermediate (3) --------------------------
    _case("GS-005", "logistic_regression", 2, 60, 3),
    _case("GS-006", "logistic_regression", 2, 90, 4, "case chuẩn"),
    _case("GS-007", "logistic_regression", 2, 120, 5),
    # --- Decision Tree / beginner (4) ------------------------------------
    _case("GS-008", "decision_tree", 1, 60, 3, "case chuẩn"),
    _case("GS-009", "decision_tree", 1, 90, 3),
    _case("GS-010", "decision_tree", 1, 90, 5),
    _case("GS-011", "decision_tree", 1, 60, 4, "CHẬT"),
    # --- Decision Tree / intermediate (3) --------------------------------
    _case("GS-012", "decision_tree", 2, 90, 4, "case chuẩn"),
    _case("GS-013", "decision_tree", 2, 120, 5),
    _case("GS-014", "decision_tree", 2, 60, 3, "RỘNG: ít bài, dễ sinh thừa module"),
    # --- K-Means / beginner (3) ------------------------------------------
    _case("GS-015", "kmeans", 1, 60, 3, "case chuẩn"),
    _case("GS-016", "kmeans", 1, 90, 4),
    _case("GS-017", "kmeans", 1, 60, 5, "CHẬT"),
    # --- K-Means / intermediate (3) --------------------------------------
    _case("GS-018", "kmeans", 2, 90, 4, "case chuẩn"),
    _case("GS-019", "kmeans", 2, 120, 5),
    _case("GS-020", "kmeans", 2, 60, 3),
]


# ---------------------------------------------------------------------------
# Chốt chặn: import file này là tự kiểm luôn, sai phân bổ báo ngay.
# ---------------------------------------------------------------------------

assert len(GOLDEN_SET) == 20, f"phải đúng 20 case, đang có {len(GOLDEN_SET)}"
assert len({c["id"] for c in GOLDEN_SET}) == 20, "trùng id"
assert {c["topic"] for c in GOLDEN_SET} == set(DATASET_OF), "thiếu/thừa topic"
assert {c["level"] for c in GOLDEN_SET} == {1, 2}, "phải có cả 2 trình độ"
assert all(
    v is not None for c in GOLDEN_SET for v in c["metric_threshold"].values()
), "còn case chưa có ngưỡng - chạy `python -m eval.calibrate` rồi điền"
