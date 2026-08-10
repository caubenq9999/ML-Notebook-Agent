from datetime import datetime
import uuid
from typing import List
from schemas import Constraints, LearnerProfile

# 1. Profile Beginner - Logistic Regression (Hạ level do quiz thấp)
mock_profile_logistic_beginner = LearnerProfile(
    session_id="sess_log_001",
    created_at=datetime.now().isoformat(),
    topic="logistic_regression",
    level_declared=2,  # Khai báo Intermediate
    level_final=1,  # Nhưng điểm Quiz thấp nên bị hạ xuống Beginner (Level 1)
    quiz_score=1,
    constraints=Constraints(
        duration_minutes=30,
        is_custom_duration=False,
        num_exercises=3,
        include_visualization=True,
        preferred_framework="Scikit-learn",
        level_adjustment_reason="Hạ từ Level 2 xuống 1 do điểm Quiz thấp (1/5).",
    ),
)

# 2. Profile Intermediate - Decision Tree (Giữ nguyên level, ưu tiên Pure Python)
mock_profile_tree_intermediate = LearnerProfile(
    session_id="sess_tree_002",
    created_at=datetime.now().isoformat(),
    topic="decision_tree",
    level_declared=2,
    level_final=2,
    quiz_score=4,
    constraints=Constraints(
        duration_minutes=45,
        is_custom_duration=False,
        num_exercises=4,
        include_visualization=True,
        preferred_framework="Pure Python",  # Muốn học tự viết code giải thuật
        level_adjustment_reason="Giữ nguyên Level 2 (Điểm Quiz tốt: 4/5).",
    ),
)

# 3. Profile Fast-Track - K-Means Clustering (Thời lượng ngắn, làm bài tập thực hành)
mock_profile_kmeans_fasttrack = LearnerProfile(
    session_id="sess_kmeans_003",
    created_at=datetime.now().isoformat(),
    topic="k_means",
    level_declared=1,
    level_final=1,
    quiz_score=3,
    constraints=Constraints(
        duration_minutes=15,  # Học nhanh trong 15 phút
        is_custom_duration=False,
        num_exercises=2,
        include_visualization=False,
        preferred_framework="Scikit-learn",
        level_adjustment_reason="Giữ nguyên Level 1 dựa trên kết quả Quiz (3/5).",
    ),
)

# 4. Profile Custom / Fallback Test (Dùng tùy chọn không giới hạn thời gian)
mock_profile_custom = LearnerProfile(
    session_id="sess_custom_004",
    created_at=datetime.now().isoformat(),
    topic="logistic_regression",
    level_declared=1,
    level_final=1,
    quiz_score=5,
    constraints=Constraints(
        duration_minutes="Tuỳ chọn",
        is_custom_duration=True,
        num_exercises=5,
        include_visualization=True,
        preferred_framework="PyTorch",
        level_adjustment_reason="Giữ nguyên Level 1 (Điểm Quiz tốt: 5/5).",
    ),
)

# Danh sách gom lại để chạy loop test cả 3 Agent cùng lúc
mock_profiles: List[LearnerProfile] = [
    mock_profile_logistic_beginner,
    mock_profile_tree_intermediate,
    mock_profile_kmeans_fasttrack,
    mock_profile_custom,
]