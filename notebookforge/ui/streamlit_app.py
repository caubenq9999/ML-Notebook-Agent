import concurrent.futures
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import uuid
import pandas as pd
import requests
import streamlit as st
import textwrap

# Thêm thư mục gốc (notebookforge) vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FASTAPI_URL = "http://localhost:8000"  # Server FastAPI do Hoàng/Hợp quản lý

if str(PROJECT_ROOT) not in sys.path:
  sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------
# DATABASE QUIZ MẪU
# ---------------------------------------------------------
QUIZ_BANK = {
    "logistic_regression": [
        {
            "q": (
                "Hàm kích hoạt (activation function) thường dùng trong Logistic"
                " Regression là gì?"
            ),
            "options": ["ReLU", "Sigmoid", "Softmax", "Tanh"],
            "a": "Sigmoid",
        },
        {
            "q": (
                "Hàm mất mát (Loss Function) chuẩn cho Binary Logistic"
                " Regression là gì?"
            ),
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
            "q": (
                "Decision Boundary cơ bản của Binary Logistic Regression ứng"
                " với ngưỡng xác suất bao nhiêu?"
            ),
            "options": ["0.0", "0.5", "1.0", "0.8"],
            "a": "0.5",
        },
        {
            "q": "Logistic Regression là thuật toán dùng cho bài toán nào?",
            "options": [
                "Regression (Hồi quy)",
                "Classification (Phân loại)",
                "Clustering (Gom nhóm)",
                "Dimensionality Reduction",
            ],
            "a": "Classification (Phân loại)",
        },
    ],
    "decision_tree": [
        {
            "q": (
                "Độ đo nào sau đây KHÔNG được sử dụng để chọn thuộc tính chia"
                " nhánh (split point) trong Cây quyết định (Decision Tree)?"
            ),
            "options": [
                "Gini Impurity",
                "Information Gain (Entropy)",
                "Euclidean Distance",
                "Gain Ratio",
            ],
            "a": "Euclidean Distance",
        },
        {
            "q": (
                "Chỉ số Gini Impurity của một nút hoàn toàn tinh khiết (tất cả"
                " mẫu thuộc cùng 1 lớp) có giá trị bằng bao nhiêu?"
            ),
            "options": ["0.0", "0.5", "1.0", "Không xác định"],
            "a": "0.0",
        },
        {
            "q": (
                'Kỹ thuật "Pruning" (Tỉa cành) trong Cây quyết định được sử'
                " dụng chủ yếu để làm gì?"
            ),
            "options": [
                "Tăng độ sâu tối đa của cây để học kỹ hơn",
                (
                    "Giảm bớt các nhánh không quan trọng nhằm kiểm soát"
                    " Overfitting"
                ),
                "Tăng tốc độ xử lý dữ liệu khuyết (Missing data)",
                "Chuyển đổi bài toán phân loại thành bài toán hồi quy",
            ],
            "a": "Giảm bớt các nhánh không quan trọng nhằm kiểm soát Overfitting",
        },
        {
            "q": (
                "So với mô hình tuyến tính hay KNN, ưu điểm nổi bật của"
                " Decision Tree là gì?"
            ),
            "options": [
                "Không bị ảnh hưởng bởi hiện tượng Overfitting",
                "Khả năng diễn giải (Interpretability) cao và dễ trực quan hóa",
                "Luôn cho độ chính xác cao hơn mọi thuật toán khác",
                "Yêu cầu dữ liệu phải được chuẩn hóa (Normalization) trước",
            ],
            "a": "Khả năng diễn giải (Interpretability) cao và dễ trực quan hóa",
        },
        {
            "q": (
                "Trong Cây quyết định, một nút lá (Leaf Node) đại diện cho"
                " điều gì?"
            ),
            "options": [
                "Một điều kiện kiểm tra thuộc tính",
                "Nhãn dự đoán cuối cùng (hoặc giá trị đầu ra)",
                "Điểm bắt đầu của cây",
                "Tập thuộc tính bị loại bỏ",
            ],
            "a": "Nhãn dự đoán cuối cùng (hoặc giá trị đầu ra)",
        },
    ],
    "k_means": [
        {
            "q": "K-Means là thuật toán thuộc nhóm nào trong Học máy?",
            "options": [
                "Học có giám sát (Supervised Learning)",
                "Học không giám sát (Unsupervised Learning)",
                "Học tăng cường (Reinforcement Learning)",
                "Học bán giám sát (Semi-supervised Learning)",
            ],
            "a": "Học không giám sát (Unsupervised Learning)",
        },
        {
            "q": (
                'Phương pháp "Elbow Method" (Phương pháp góc cùi cỏ tay) thường'
                " được dùng trong K-Means để làm gì?"
            ),
            "options": [
                "Chọn vị trí khởi tạo tâm cụm ban đầu",
                "Xác định số lượng cụm tối ưu (K)",
                "Tính toán tốc độ hội tụ của thuật toán",
                "Loại bỏ các điểm dữ liệu nhiễu (Outliers)",
            ],
            "a": "Xác định số lượng cụm tối ưu (K)",
        },
        {
            "q": (
                "Thuật toán K-Means++ được cải tiến so với K-Means truyền thống"
                " ở bước nào?"
            ),
            "options": [
                "Cách tính khoảng cách giữa các điểm dữ liệu",
                (
                    "Bước khởi tạo các tâm cụm ban đầu (Centroids"
                    " Initialization)"
                ),
                "Bước cập nhật lại vị trí tâm cụm ở mỗi vòng lặp",
                "Điều kiện dừng thuật toán",
            ],
            "a": "Bước khởi tạo các tâm cụm ban đầu (Centroids Initialization)",
        },
        {
            "q": "Hạn chế chính của thuật toán K-Means là gì?",
            "options": [
                (
                    "Nhạy cảm với vị trí khởi tạo tâm cụm và các điểm ngoại lệ"
                    " (Outliers)"
                ),
                "Không làm việc được với dữ liệu có nhiều hơn 2 thuộc tính",
                "Tốc độ tính toán rất chậm trên tập dữ liệu nhỏ",
                "Yêu cầu dữ liệu bắt buộc phải có nhãn sẵn",
            ],
            "a": (
                "Nhạy cảm với vị trí khởi tạo tâm cụm và các điểm ngoại lệ"
                " (Outliers)"
            ),
        },
        {
            "q": (
                "Trong mỗi vòng lặp của K-Means, vị trí tâm cụm (Centroid) mới"
                " được cập nhật bằng cách nào?"
            ),
            "options": [
                "Lấy ngẫu nhiên một điểm dữ liệu trong cụm",
                (
                    "Tính giá trị trung bình (Mean) tọa độ của tất cả các"
                    " điểm thuộc cụm đó"
                ),
                "Chọn điểm nằm xa tâm cũ nhất",
                "Tính giá trị trung vị (Median) của cụm",
            ],
            "a": (
                "Tính giá trị trung bình (Mean) tọa độ của tất cả các điểm"
                " thuộc cụm đó"
            ),
        },
    ],
}

TOPIC_LABELS = {
    "logistic_regression": "LOGISTIC REGRESSION",
    "decision_tree": "DECISION TREE",
    "k_means": "K-MEANS CLUSTERING",
}


def calculate_final_level(
    level_declared: int, quiz_score: int
) -> tuple[int, str]:
  """Logic tính level_final và ghi nhận lý do traceback."""
  if quiz_score <= 2 and level_declared > 1:
    level_final = level_declared - 1
    reason = (
        f"Hạ từ Level {level_declared} xuống {level_final} do điểm Quiz thấp"
        f" ({quiz_score}/5)."
    )
  elif quiz_score >= 4 and level_declared < 3:
    level_final = level_declared
    reason = (
        f"Giữ nguyên Level {level_declared} (Điểm Quiz tốt: {quiz_score}/5)."
    )
  else:
    level_final = level_declared
    reason = (
        f"Giữ nguyên Level {level_declared} dựa trên kết quả Quiz"
        f" ({quiz_score}/5)."
    )

  return level_final, reason


def get_mock_notebook_data(profile_data: dict) -> tuple[dict, dict]:
  """Tạo dữ liệu Notebook & Report giả lập khi Backend chưa sẵn sàng."""
  mock_notebook = {
      "cells": [
          {
              "cell_type": "markdown",
              "source": [
                  f"# Notebook: {profile_data.get('topic', '').upper()}\n"
                  f"**Level:** Level {profile_data.get('level_final')}\n"
                  f"**Duration:** {profile_data.get('constraints', {}).get('duration_minutes')} mins"
              ],
          },
          {
              "cell_type": "code",
              "execution_count": 1,
              "source": [
                  "# Khởi tạo môi trường\nimport pandas as pd\nimport numpy"
                  " as np\nprint('Notebook generated successfully!')"
              ],
              "outputs": [],
          },
      ],
      "metadata": {},
      "nbformat": 4,
      "nbformat_minor": 2,
  }

  mock_report = {
      "status": "completed",
      "scores": {
          "executability": 1.0,
          "groundedness": 0.9,
          "difficulty_fit": 0.85,
          "pedagogical_order": 0.9,
      },
      "feedback": (
          "Fallback Mode: Dữ liệu Notebook được giả lập do Backend chưa trả"
          " về kết quả đúng Schema."
      ),
  }
  return mock_notebook, mock_report


def run_pipeline_via_fastapi(profile_data: dict) -> tuple[dict, dict] | None:
  """Gửi profile lên FastAPI qua POST /generate, sau đó Polling GET /report/{id}

  để lấy kết quả Notebook và Quality Report thực tế.
  """
  try:
    response = requests.post(
        f"{FASTAPI_URL}/generate", json=profile_data, timeout=10
    )

    if response.status_code not in [200, 202]:
      st.warning(
          f"⚠️ Server trả lỗi HTTP {response.status_code}: {response.text}."
          " Đang dùng Mock Data..."
      )
      return get_mock_notebook_data(profile_data)

    data = response.json()
    task_id = data.get("task_id") or data.get("id")

    POLL_INTERVAL = 3
    MAX_RETRIES = 60

    for _ in range(MAX_RETRIES):
      res = requests.get(f"{FASTAPI_URL}/report/{task_id}", timeout=5)
      if res.status_code == 200:
        report_data = res.json()
        status = str(report_data.get("status", "")).lower()

        if status in ["completed", "success"]:
          notebook = report_data.get("notebook") or report_data.get(
              "notebook_data"
          )
          report = report_data.get("report") or report_data.get(
              "quality_report"
          )

          if not notebook:
            st.warning(
                "⚠️ API trả về 'completed' nhưng không tìm thấy key 'notebook'."
                " Đây là Data nhận được:"
            )
            st.json(report_data)
            st.info("🔄 Đang dùng Mock Data để tiếp tục render UI...")
            return get_mock_notebook_data(profile_data)

          return notebook, report

        elif status in ["failed", "error"]:
          st.error(
              "Pipeline thất bại:"
              f" {report_data.get('error_message', 'Lỗi thực thi')}"
          )
          st.json(report_data)
          return None

      time.sleep(POLL_INTERVAL)

    st.warning("⚠️ Timeout 3 phút từ FastAPI. Đang bật Mock Data...")
    return get_mock_notebook_data(profile_data)

  except requests.exceptions.ConnectionError:
    st.warning(
        f"⚠️ Không kết nối được FastAPI tại `{FASTAPI_URL}`. Đang chạy chế độ"
        " Demo (Mock Data)..."
    )
    return get_mock_notebook_data(profile_data)
  except Exception as e:
    st.error(f"❌ Lỗi Exception: {str(e)}")
    return get_mock_notebook_data(profile_data)


def execute_pipeline_with_progress(profile, timeout_seconds=180):
  """Chạy pipeline với giao diện cập nhật 5 bước tiến trình và xử lý Timeout."""
  steps = [
      (
          "🔍 **Bước 1/5:** Đang nghiên cứu chủ đề & thu thập tài liệu (Research"
          " Agent)..."
      ),
      "📚 **Bước 2/5:** Đang thiết kế lộ trình bài học (Curriculum Agent)...",
      (
          "📝 **Bước 3/5:** Đang khởi tạo và viết nội dung Notebook (Notebook"
          " Gen)..."
      ),
      (
          "⚙️ **Bước 4/5:** Đang thực thi kiểm thử Notebook trong Sandbox"
          " (Executor)..."
      ),
      (
          "✅ **Bước 5/5:** Đang đánh giá chất lượng & chấm điểm (Verifier"
          " Agent)..."
      ),
  ]

  with st.status(
      "🚀 **Đang khởi tạo NotebookForge Pipeline...**", expanded=True
  ) as status:
    progress_bar = st.progress(0)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      future = executor.submit(run_pipeline_via_fastapi, profile)

      start_time = time.time()
      current_step = 0

      while future.running():
        elapsed = time.time() - start_time

        if elapsed > timeout_seconds:
          status.update(
              label="❌ **Hệ thống quá tải hoặc phản hồi chậm (Timeout)!**",
              state="error",
              expanded=True,
          )
          st.error(
              "⚠️ Quá trình khởi tạo vượt quá thời gian cho phép"
              f" ({timeout_seconds} giây). Vui lòng thử lại!"
          )
          return None

        calculated_step = min(int((elapsed / 10) * 5), 4)
        if calculated_step != current_step:
          current_step = calculated_step
          status.update(label=steps[current_step])
          progress_bar.progress((current_step + 1) * 20)

        time.sleep(0.5)

      try:
        result_tuple = future.result()
        if result_tuple is None:
          status.update(
              label=(
                  "💥 **Xảy ra lỗi trong quá trình thực thi! (API trả về"
                  " None)**"
              ),
              state="error",
          )
          return None

        progress_bar.progress(100)
        status.update(
            label="🎉 **Đã hoàn thành tạo Notebook thành công!**",
            state="complete",
            expanded=False,
        )
        return result_tuple
      except Exception as e:
        status.update(
            label=f"💥 **Xảy ra lỗi Exception: {str(e)}**", state="error"
        )
        st.error(f"Lỗi chi tiết: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        return None


# ---------------------------------------------------------
# UI MAIN APP
# ---------------------------------------------------------
LOGO_URL = (
    "https://lh3.googleusercontent.com/d/1s8zYQqejbKvZs786zWLzMPV8FoclhNHC"
)

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
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
  st.session_state.session_id = str(uuid.uuid4())[:8]
  st.session_state.created_at = datetime.now().isoformat()

st.sidebar.caption(f"**Session ID:** `{st.session_state.session_id}`")
st.sidebar.caption(f"**Created At:** {st.session_state.created_at}")

# --- PHASE 1: THÔNG TIN CƠ BẢN ---
st.subheader("1. Cài đặt bài học")

with st.expander("Tùy chỉnh Notebook", expanded=True):
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

  st.write("")

  c1, c2 = st.columns(2)
  with c1:
    duration_minutes = st.slider(
        "Thời lượng (phút):", min_value=60, max_value=120, value=60, step=10
    )
  with c2:
    num_exercises = st.slider(
        "Số bài tập thực hành:", min_value=1, max_value=5, value=3
    )

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
      key=f"{topic}_q_{idx}",
      label_visibility="collapsed",
  )
  st.divider()

all_answered = all(
    answer is not None for answer in user_answers.values()
) and len(user_answers) == len(questions)

if not all_answered:
  st.warning(
      "⚠️ Vui lòng hoàn thành tất cả các câu hỏi quiz bên trên để tiếp tục."
  )

submit_quiz = st.button(
    "Tạo Notebook", type="primary", disabled=not all_answered
)

# --- PHASE 3: XỬ LÝ & TẠO LEANER PROFILE & CHẠY PIPELINE ---
if submit_quiz:
  quiz_score = sum(
      1
      for idx, q_data in enumerate(questions)
      if user_answers[idx] == q_data["a"]
  )
  level_final, adjustment_reason = calculate_final_level(
      level_declared, quiz_score
  )

  constraints = {
      "duration_minutes": duration_minutes,
      "num_exercises": num_exercises,
  }

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

  st.info(
    textwrap.dedent(f"""
    📌 **Chủ đề bạn chọn:** {TOPIC_LABELS[topic]}  
    🎯 **Cấp độ xếp hạng:** {level_name}
    """).strip()
    )   

  # KÍCH HOẠT CHẠY PIPELINE THẬT QUA FASTAPI
  notebook_result = execute_pipeline_with_progress(
      profile=profile_data, timeout_seconds=180
  )

  if notebook_result:
    notebook_dict, report_data = notebook_result
    st.session_state.notebook_dict = notebook_dict
    st.session_state.report_data = report_data
    st.session_state.current_topic = topic

# --- PHASE 4: HIỂN THỊ KẾT QUẢ KHI ĐÃ CÓ DATA ---
if "notebook_dict" in st.session_state and st.session_state.notebook_dict:
  st.balloons()
  st.success("🎉 **Notebook của bạn đã được tạo thành công!**")

  # Download Notebook JSON
  notebook_json_bytes = json.dumps(
      st.session_state.notebook_dict, ensure_ascii=False, indent=2
  ).encode("utf-8")

  st.download_button(
      label="📥 Tải xuống Notebook (.ipynb)",
      data=notebook_json_bytes,
      file_name=(
          f"{st.session_state.get('current_topic', 'notebook')}_{st.session_state.session_id}.ipynb"
      ),
      mime="application/x-ipynb+json",
      type="primary",
  )

  # Hiển thị Quality Report dưới dạng Bảng Excel
  report_data = st.session_state.get("report_data")
  if report_data:
    st.divider()
    with st.expander(
        "📊 **Báo cáo Đánh giá Chất lượng (Quality Report)**", expanded=True
    ):
      if isinstance(report_data, dict):
        rows = []
        if "status" in report_data:
          rows.append({
              "Tiêu chí / Metric": "Status",
              "Giá trị": str(report_data["status"]).upper(),
          })

        scores = report_data.get("scores", {})
        if isinstance(scores, dict):
          for metric, score in scores.items():
            metric_name = metric.replace("_", " ").title()
            rows.append({
                "Tiêu chí / Metric": f"Score: {metric_name}",
                "Giá trị": score,
            })

        if "feedback" in report_data:
          rows.append({
              "Tiêu chí / Metric": "Feedback",
              "Giá trị": report_data["feedback"],
          })

        df_report = pd.DataFrame(rows)
        st.table(df_report)

      elif isinstance(report_data, str):
        st.markdown(report_data)
      else:
        st.write(report_data)