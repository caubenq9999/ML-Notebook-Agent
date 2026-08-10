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
        duration_minutes=40,
        num_exercises=3,
        #preferred_framework="Scikit-learn",
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
        duration_minutes=60,
        num_exercises=4,
        #preferred_framework="Pure Python",  # Muốn học tự viết code giải thuật
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
        duration_minutes=80,  # Học nhanh trong 15 phút
        num_exercises=2,
        #preferred_framework="Scikit-learn",
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
        duration_minutes="110",
        num_exercises=5,
        #preferred_framework="PyTorch",
    ),
)

# Danh sách gom lại để chạy loop test cả 3 Agent cùng lúc
mock_profiles: List[LearnerProfile] = [
    mock_profile_logistic_beginner,
    mock_profile_tree_intermediate,
    mock_profile_kmeans_fasttrack,
    mock_profile_custom,
]