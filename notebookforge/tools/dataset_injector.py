import random

# HƯỚNG DẪN TÍCH HỢP DATASET INJECTOR
# 
# Hàm chính: get_dataset_code(topic: str, seed: int = None) -> list[dict]
# Trả về: list[dict], mỗi dict đại diện cho 1 bước (Load, EDA, Split)
#         Cấu trúc dict: {'title': str, 'code': str}
#
# Cách tích hợp vào Pipeline tạo Notebook:

# from tools.dataset_injector import get_dataset_code

# eda_cells = get_dataset_code(topic=profile.topic, seed=profile.seed)

# for cell_info in eda_cells:
#     # 1. Thêm Markdown cell cho tiêu đề section (nếu pipeline chưa tự tạo)
#     notebook.cells.append(create_markdown_cell(f"### {cell_info['title']}"))
    
#     # 2. Thêm Code cell tương ứng
#     notebook.cells.append(create_code_cell(cell_info['code']))


# LƯU Ý KHI CHẠY:
# 1. Môi trường chạy Notebook (Executor) cần đảm bảo Working Directory là thư mục ROOT 
#    của dự án để câu lệnh `pd.read_csv('notebookforge/datasets/...')` tìm thấy file.
# 2. Nếu topic truyền vào không khớp (Logistic/Tree/KMeans), hàm sẽ tự động fallback 
#    về dataset 'logistic_regression' mặc định nên không sợ văng Exception.

TOPIC_DATASETS = {
    "logistic_regression": {
        "name": "Heart Failure Prediction",
        "path": "notebookforge/datasets/heart.csv",  
        "target": "HeartDisease",
        "type": "classification",
        "shape": (918, 12),
        "description": "918 dòng, 12 cột (11 đặc trưng lâm sàng + 1 nhãn)",
    },
    "decision_tree": {
        "name": "Red Wine Quality",
        "path": "notebookforge/datasets/winequality-red.csv",  
        "target": "quality",
        "type": "classification",
        "shape": (1599, 12),
        "description": (
            "1.599 dòng, 12 cột (11 đặc trưng hóa lý + 1 điểm chất lượng)"
        ),
    },
    "k_means": {
        "name": "Mall Customer Segmentation",
        "path": "notebookforge/datasets/Mall_Customers.csv", 
        "type": "clustering",
        "shape": (200, 5),
        "description": "200 dòng, 5 cột",
    },
}

def get_eda_cells(topic: str, seed: int = None) -> list[dict]:
  """Trả về danh sách các code cell dùng Dataset Local từ thư mục data/."""
  if seed is None:
    seed = random.randint(1, 10000)

  topic_key = topic.lower().strip()
  cells = []

  # ==========================================
  # 1. LOGISTIC REGRESSION (Heart Failure)
  # ==========================================
  if "logistic" in topic_key:
    # Cell 1: Load Data từ file Local
    cells.append({
        "title": "1. Load Dataset & Inspection",
        "code": f"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Heart Failure Prediction Dataset từ file local
data_path = "{TOPIC_DATASETS['logistic_regression']['path']}"
df = pd.read_csv(data_path)

print(f"Dataset Shape: {{df.shape}}")
df.head()""",
    })

    # Cell 2: EDA & Data Cleaning
    cells.append({
        "title": "2. EDA & Handling Missing/Outlier Values",
        "code": """# 1. Loại bỏ dòng vô lý RestingBP = 0
df = df[df['RestingBP'] > 0].copy()

# 2. Chuyển Cholesterol = 0 thành NaN để Impute (Missing ngầm y khoa)
df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)

# Impute Cholesterol bằng Median theo từng nhóm HeartDisease
df['Cholesterol'] = df.groupby('HeartDisease')['Cholesterol'].transform(lambda x: x.fillna(x.median()))

print("Data after handling missing/outliers:")
print(df.isnull().sum())""",
    })

    # Cell 3: Encoding & Scaling & Split
    cells.append({
        "title": "3. Feature Encoding, Scaling & Train/Test Split",
        "code": f"""# 1. One-Hot Encoding cho các biến Categorical
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns=['HeartDisease'])
y = df_encoded['HeartDisease']

# 2. Chia Train/Test (Stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

# 3. Standard Scaling cho dữ liệu số
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {{X_train_scaled.shape}}, X_test shape: {{X_test_scaled.shape}}")""",
    })

  # ==========================================
  # 2. DECISION TREE (Red Wine Quality)
  # ==========================================
  elif "tree" in topic_key or "decision" in topic_key:
    # Cell 1: Load Data từ file Local
    cells.append({
        "title": "1. Load Dataset & Inspection",
        "code": f"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load Red Wine Quality Dataset từ file local
data_path = "{TOPIC_DATASETS['decision_tree']['path']}"
df = pd.read_csv(data_path)

print(f"Dataset Shape: {{df.shape}}")
df.head()""",
    })

    # Cell 2: EDA & Drop Duplicates
    cells.append({
        "title": "2. EDA & Deduplication",
        "code": """# Check duplicates (~240 dòng trùng)
duplicate_count = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {duplicate_count}")

# Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape sau khi loại bỏ trùng lặp: {df.shape}")""",
    })

    # Cell 3: Train/Test Split
    cells.append({
        "title": "3. Feature Engineering & Train/Test Split",
        "code": f"""X = df.drop(columns=['quality'])
y = df['quality']

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

print(f"Features: {{list(X.columns)}}")
print(f"Distribution of target 'quality' in Train: \\n{{y_train.value_counts()}}")""",
    })

  # ==========================================
  # 3. K-MEANS CLUSTERING (Mall Customers)
  # ==========================================
  elif "kmeans" in topic_key or "k-means" in topic_key or "k_means" in topic_key:
    # Cell 1: Load Data từ file Local
    cells.append({
        "title": "1. Load Dataset & Inspection",
        "code": f"""import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Mall Customer Dataset từ file local
data_path = "{TOPIC_DATASETS['k_means']['path']}"
df = pd.read_csv(data_path)

print(f"Dataset Shape: {{df.shape}}")
df.head()""",
    })

    # Cell 2: Feature Selection & Encoding & Scaling
    cells.append({
        "title": "2. Preprocessing & Feature Scaling for Clustering",
        "code": f"""# 1. Loại bỏ cột định danh CustomerID
df_features = df.drop(columns=['CustomerID'], errors='ignore')

# 2. Encode biến Gender
df_encoded = pd.get_dummies(df_features, columns=['Gender'], drop_first=True)

# 3. Scaling đặc trưng (Rất quan trọng với KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Xáo trộn dữ liệu theo seed={seed}
np.random.seed({seed})
shuffled_idx = np.random.permutation(len(X_scaled))
X_scaled = X_scaled[shuffled_idx]

print("Features processed & scaled successfully for KMeans!")
print(f"Final Input Shape for Clustering: {{X_scaled.shape}}")""",
    })

  else:
    return get_eda_cells("logistic_regression", seed)

  return cells


def get_dataset_code(topic: str, seed: int = None) -> list[dict]:
  return get_eda_cells(topic=topic, seed=seed)