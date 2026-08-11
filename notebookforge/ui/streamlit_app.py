import uuid
from datetime import datetime
import streamlit as st

# Giả sử import từ schemas.py của Hoàng
# from schemas import LearnerProfile

# ---------------------------------------------------------
# DATABASE QUIZ MẪU (Tách ra file JSON/Dict riêng được nếu cần)
# ---------------------------------------------------------
QUIZ_BANK = {
    "logistic_regression": [
        {
            "q": "Hàm kích hoạt (activation function) thường dùng trong Logistic Regression là gì?",
            "options": ["ReLU", "Sigmoid", "Softmax", "Tanh"],
            "a": "Sigmoid",
        },
        {
            "q": "Hàm mất mát (Loss Function) chuẩn cho Binary Logistic Regression là gì?",
            "options": [
                "Mean Squared Error (MSE)",
                "Binary Cross-Entropy / Log Loss",
                "Mean Absolute Error (MAE)",
                "Hinge Loss",
            ],
            "a": "Binary Cross-Entropy / Log Loss",
        },
        {
            "q": "Đầu ra của mô hình Logistic Regression đại diện cho điều gì?",
            "options": [
                "Giá trị thực liên tục",
                "Xác suất thuộc về lớp positive (1)",
                "Khoảng cách tới đường phân cách",
                "Số lượng cluster",
            ],
            "a": "Xác suất thuộc về lớp positive (1)",
        },
        {
            "q": "Decision Boundary cơ bản của Binary Logistic Regression ứng với ngưỡng xác suất bao nhiêu?",
            "options": ["0.0", "0.5", "1.0", "0.8"],
            "a": "0.5",
        },
        {
            "q": "Logistic Regression là thuật toán dùng cho bài toán nào?",
            "options": ["Regression (Hồi quy)", "Classification (Phân loại)", "Clustering (Gom nhóm)", "Dimensionality Reduction"],
            "a": "Classification (Phân loại)",
        },
    ],
    "decision_tree": [
        {
            "q": "Độ đo nào sau đây KHÔNG được sử dụng để chọn thuộc tính chia nhánh (split point) trong Cây quyết định (Decision Tree)?",
            "options": [
                "Gini Impurity",
                "Information Gain (Entropy)",
                "Euclidean Distance",
                "Gain Ratio"
            ],
            "a": "Euclidean Distance"
        },
        {
            "q": "Chỉ số Gini Impurity của một nút hoàn toàn tinh khiết (tất cả mẫu thuộc cùng 1 lớp) có giá trị bằng bao nhiêu?",
            "options": ["0.0", "0.5", "1.0", "Không xác định"],
            "a": "0.0"
        },
        {
            "q": "Kỹ thuật \"Pruning\" (Tỉa cành) trong Cây quyết định được sử dụng chủ yếu để làm gì?",
            "options": [
                "Tăng độ sâu tối đa của cây để học kỹ hơn",
                "Giảm bớt các nhánh không quan trọng nhằm kiểm soát Overfitting",
                "Tăng tốc độ xử lý dữ liệu khuyết (Missing data)",
                "Chuyển đổi bài toán phân loại thành bài toán hồi quy"
            ],
            "a": "Giảm bớt các nhánh không quan trọng nhằm kiểm soát Overfitting"
        },
        {
            "q": "So với mô hình tuyến tính hay KNN, ưu điểm nổi bật của Decision Tree là gì?",
            "options": [
                "Không bị ảnh hưởng bởi hiện tượng Overfitting",
                "Khả năng diễn giải (Interpretability) cao và dễ trực quan hóa",
                "Luôn cho độ chính xác cao hơn mọi thuật toán khác",
                "Yêu cầu dữ liệu phải được chuẩn hóa (Normalization) trước"
            ],
            "a": "Khả năng diễn giải (Interpretability) cao và dễ trực quan hóa"
        },
        {
            "q": "Trong Cây quyết định, một nút lá (Leaf Node) đại diện cho điều gì?",
            "options": [
                "Một điều kiện kiểm tra thuộc tính",
                "Nhãn dự đoán cuối cùng (hoặc giá trị đầu ra)",
                "Điểm bắt đầu của cây",
                "Tập thuộc tính bị loại bỏ"
            ],
            "a": "Nhãn dự đoán cuối cùng (hoặc giá trị đầu ra)"
        }
    ],
    "k_means": [
        {
            "q": "K-Means là thuật toán thuộc nhóm nào trong Học máy?",
            "options": [
                "Học có giám sát (Supervised Learning)",
                "Học không giám sát (Unsupervised Learning)",
                "Học tăng cường (Reinforcement Learning)",
                "Học bán giám sát (Semi-supervised Learning)"
            ],
            "a": "Học không giám sát (Unsupervised Learning)"
        },
        {
            "q": "Phương pháp \"Elbow Method\" (Phương pháp góc cùi cỏ tay) thường được dùng trong K-Means để làm gì?",
            "options": [
                "Chọn vị trí khởi tạo tâm cụm ban đầu",
                "Xác định số lượng cụm tối ưu (K)",
                "Tính toán tốc độ hội tụ của thuật toán",
                "Loại bỏ các điểm dữ liệu nhiễu (Outliers)"
            ],
            "a": "Xác định số lượng cụm tối ưu (K)"
        },
        {
            "q": "Thuật toán K-Means++ được cải tiến so với K-Means truyền thống ở bước nào?",
            "options": [
                "Cách tính khoảng cách giữa các điểm dữ liệu",
                "Bước khởi tạo các tâm cụm ban đầu (Centroids Initialization)",
                "Bước cập nhật lại vị trí tâm cụm ở mỗi vòng lặp",
                "Điều kiện dừng thuật toán"
            ],
            "a": "Bước khởi tạo các tâm cụm ban đầu (Centroids Initialization)"
        },
        {
            "q": "Hạn chế chính của thuật toán K-Means là gì?",
            "options": [
                "Nhạy cảm với vị trí khởi tạo tâm cụm và các điểm ngoại lệ (Outliers)",
                "Không làm việc được với dữ liệu có nhiều hơn 2 thuộc tính",
                "Tốc độ tính toán rất chậm trên tập dữ liệu nhỏ",
                "Yêu cầu dữ liệu bắt buộc phải có nhãn sẵn"
            ],
            "a": "Nhạy cảm với vị trí khởi tạo tâm cụm và các điểm ngoại lệ (Outliers)"
        },
        {
            "q": "Trong mỗi vòng lặp của K-Means, vị trí tâm cụm (Centroid) mới được cập nhật bằng cách nào?",
            "options": [
                "Lấy ngẫu nhiên một điểm dữ liệu trong cụm",
                "Tính giá trị trung bình (Mean) tọa độ của tất cả các điểm thuộc cụm đó",
                "Chọn điểm nằm xa tâm cũ nhất",
                "Tính giá trị trung vị (Median) của cụm"
            ],
            "a": "Tính giá trị trung bình (Mean) tọa độ của tất cả các điểm thuộc cụm đó"
        }
    ]
}

TOPIC_LABELS = {
    "logistic_regression": "LOGISTIC REGRESSION",
    "decision_tree": "DECISION TREE",
    "k_means": "K-MEANS CLUSTERING",
}

# TÍNH NĂNG CÓ THỂ PHÁT TRIỂN THÊM NẾU CÓ THỜI GIAN
# FRAMEWORK_LABELS = {
#     "scikit_learn" : "Scikit-learn", 
#     "pure_python": 'Pure Python',
#     "pytorch" : 'PyTorch'
# }

def calculate_final_level(level_declared: int, quiz_score: int) -> tuple[int, str]:
    """Logic tính level_final và ghi nhận lý do traceback."""
    if quiz_score <= 2 and level_declared > 1:
        level_final = level_declared - 1
        reason = f"Hạ từ Level {level_declared} xuống {level_final} do điểm Quiz thấp ({quiz_score}/5)."
    elif quiz_score >= 4 and level_declared < 3:
        level_final = level_declared
        reason = f"Giữ nguyên Level {level_declared} (Điểm Quiz tốt: {quiz_score}/5)."
    else:
        level_final = level_declared
        reason = f"Giữ nguyên Level {level_declared} dựa trên kết quả Quiz ({quiz_score}/5)."

    return level_final, reason


# ---------------------------------------------------------
# UI MAIN APP
# ---------------------------------------------------------
LOGO_URL = "https://lh3.googleusercontent.com/d/1s8zYQqejbKvZs786zWLzMPV8FoclhNHC"

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <img src="{LOGO_URL}" width="150" height="150" style="object-fit: contain;">
        <div>
            <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700; line-height: 1.2;">
                NotebookForge
            </h1>
            <div style="font-size: 1.1rem; font-weight: 600; color: #888888; margin-top: 4px;">
                SET UP LEARNER'S PROFILE
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Khởi tạo session_id & created_at cố định cho lượt làm việc
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.created_at = datetime.now().isoformat()

st.sidebar.caption(f"**Session ID:** `{st.session_state.session_id}`")
st.sidebar.caption(f"**Created At:** {st.session_state.created_at}")

# --- PHASE 1: THÔNG TIN CƠ BẢN ---
st.subheader("1. Cài đặt bài học")

with st.expander("Tùy chỉnh Notebook", expanded=True):
  # Hàng 1: Chọn Topic & Trình độ
  col1, col2 = st.columns(2)

  with col1:
    topic = st.selectbox(
        "Chọn topic:",
        options=list(TOPIC_LABELS.keys()),
        format_func=lambda key: TOPIC_LABELS[key],
    )

  with col2:
    st.write("**Trình độ:**")

    if "is_intermediate" not in st.session_state:
      st.session_state.is_intermediate = False

    if st.session_state.is_intermediate:
      left_style = "color: #FFFFFF; font-weight: normal; opacity: 1;"
      right_style = (
          "color: #00A2FF; font-weight: bold; opacity: 1.0; text-shadow: 0 0"
          " 10px rgba(0, 162, 255, 0.6);"
      )
    else:
      left_style = (
          "color: #00FF88; font-weight: bold; opacity: 1.0; text-shadow: 0 0"
          " 10px rgba(0, 255, 136, 0.6);"
      )
      right_style = "color: #FFFFFF; font-weight: normal; opacity: 1;"

    t_col1, t_col2, t_col3 = st.columns([1, 0.3, 1.2])

    with t_col1:
      st.markdown(
          f"<div style='text-align: right; padding-top: 5px;"
          f" {left_style}'>1 - Beginner</div>",
          unsafe_allow_html=True,
      )

    with t_col2:
      is_intermediate = st.toggle(
          "level_toggle",
          value=st.session_state.is_intermediate,
          label_visibility="collapsed",
          key="is_intermediate",
      )

    with t_col3:
      st.markdown(
          f"<div style='text-align: left; padding-top: 5px;"
          f" {right_style}'>2 - Intermediate</div>",
          unsafe_allow_html=True,
      )

    level_declared = 2 if is_intermediate else 1

  st.write("")  # Khoảng đệm giữa hai hàng

  # Hàng 2: Thời lượng & Số bài tập
  c1, c2 = st.columns(2)
  with c1:
    duration_minutes = st.slider(
        "Thời lượng (phút):", min_value=60, max_value=120, value=60, step=10
    )
  with c2:
    num_exercises = st.slider(
        "Số bài tập thực hành:", min_value=1, max_value=5, value=3
    )

    #selected_framework_key = st.radio(
    #    "Framework ưu tiên:", 
    #    options=list(FRAMEWORK_LABELS.keys()),
    #    format_func=lambda k: FRAMEWORK_LABELS[k],
    #    horizontal=True, 
    #    index=None
    #)
    #preferred_framework = FRAMEWORK_LABELS[selected_framework_key] if selected_framework_key else None

# --- PHASE 2: QUIZ 5 CÂU ---
st.subheader("2. Câu hỏi đánh giá")
st.info("Kết quả Quiz sẽ được dùng để căn chỉnh độ khó thực tế của Notebook.")

questions = QUIZ_BANK.get(topic, QUIZ_BANK["logistic_regression"])
user_answers = {}

for idx, q_data in enumerate(questions):
    st.write(f"**Câu {idx + 1}:** {q_data['q']}")
    user_answers[idx] = st.radio(
        f"Chọn đáp án câu {idx + 1}:",
        q_data["options"],
        index=None,
        # Đưa `topic` vào key để reset trạng thái chọn radio khi đổi topic
        key=f"{topic}_q_{idx}",
        label_visibility="collapsed",
    )
    st.divider()

# Kiểm tra xem người dùng đã chọn đủ tất cả các câu hỏi chưa
all_answered = all(answer is not None for answer in user_answers.values()) and len(user_answers) == len(questions)

# Nếu chưa trả lời đủ, hiển thị thông báo nhắc nhở
if not all_answered:
    st.warning("⚠️ Vui lòng hoàn thành tất cả các câu hỏi quiz bên trên để tiếp tục.")

# Nút Tạo Notebook sẽ bị vô hiệu hóa (disabled) nếu chưa trả lời đủ
submit_quiz = st.button("Tạo Notebook", type="primary", disabled=not all_answered)

# --- PHASE 3: XỬ LÝ & TẠO LEANER PROFILE ---
if submit_quiz:
    # Tính điểm
    quiz_score = sum(1 for idx, q_data in enumerate(questions) if user_answers[idx] == q_data["a"])

    # Tính level_final & Lý do
    level_final, adjustment_reason = calculate_final_level(level_declared, quiz_score)

    constraints = {
        "duration_minutes": duration_minutes,
        "num_exercises": num_exercises,
        #"preferred_framework": preferred_framework,
    }

    # Đóng gói Profile Object
    profile_data = {
        "session_id": st.session_state.session_id,
        "created_at": st.session_state.created_at,
        "topic": topic,
        "level_declared": level_declared,
        "level_final": level_final,
        "quiz_score": quiz_score,
        "constraints": constraints,
    }

    st.success("Tạo Learner's Profile thành công!")

    level_name = "Beginner" if level_final == 1 else "Intermediate"

    st.info(f"""
    📌 **Chủ đề bạn chọn:** {TOPIC_LABELS[topic]}  
    🎯 **Cấp độ của bạn:** {level_name}
    """)

    # Hiển thị kết quả Traceability cho người dùng/tester xem
    st.json(profile_data)

    # if level_final < level_declared:
    #     st.warning(f"⚠️ **Thông báo điều chỉnh:** {adjustment_reason}")
    # else:
    #     st.info(f"ℹ️ **Thông tin level:** {adjustment_reason}")

    # Khi nối với main.py/api.py của Hoàng:
    # learner_profile = LearnerProfile(**profile_data)
    # generate(learner_profile)