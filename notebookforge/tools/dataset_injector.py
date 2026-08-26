import os
import random

# HƯỚNG DẪN DÙNG HÀM DATASET INJECTOR
# Hàm chính: get_dataset_code(topic: str, seed: int = None) -> list[dict]
# Trả về: list[dict], mỗi dict đại diện cho 1 bước (Load, EDA, Split)
#         Cấu trúc dict: {'title': str, 'code': str}
#
# LƯU Ý KHI CHẠY:
# 1. Đảm bảo 3 file CSV đã được đặt đúng cấu trúc thư mục dự án:
#    - notebookforge/datasets/heart.csv
#    - notebookforge/datasets/winequality-red.csv
#    - notebookforge/datasets/Mall_Customers.csv
# 2. Nếu không tìm thấy file CSV, code cell sinh ra sẽ chủ động raise FileNotFoundError
#    kèm thông báo chi tiết hướng dẫn cách khắc phục.
# ==============================================================================

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
        "description": "1.599 dòng, 12 cột (11 đặc trưng hóa lý + 1 điểm chất lượng)",
    },
    "k_means": {
        "name": "Mall Customer Segmentation",
        "path": "notebookforge/datasets/Mall_Customers.csv",
        "type": "clustering",
        "shape": (200, 5),
        "description": "200 dòng, 5 cột",
    },
}

def _get_file_check_snippet(file_path: str, dataset_name: str) -> str:
    """Tạo đoạn code snippet kiểm tra file CSV có tồn tại hay không trước khi read_csv."""
    return f"""import os
from pathlib import Path
import pandas as pd
import numpy as np

relative_data_path = Path("{file_path}")
data_candidates = [
    Path.cwd() / relative_data_path,
    Path.cwd() / "datasets" / relative_data_path.name,
    Path.cwd().parent / "datasets" / relative_data_path.name,
]
data_path = next(
    (path.resolve() for path in data_candidates if path.is_file()),
    Path.cwd() / relative_data_path,
)

# Kiểm tra sự tồn tại của Dataset File
if not data_path.is_file():
    raise FileNotFoundError(
        f"❌ KHÔNG TÌM THẤY DATASET: '{dataset_name}' tại đường dẫn '{{os.path.abspath(data_path)}}'.\\n"
        f"👉 Vui lòng đảm bảo bạn đã copy file CSV vào đúng thư mục 'notebookforge/datasets/'!"
    )
"""

def get_eda_cells(topic: str, seed: int = None) -> list[dict]:
    """Trả về danh sách các code cell dùng Dataset Local chuẩn theo Schema dự án."""
    if seed is None:
        seed = random.randint(1, 10000)

    topic_key = topic.lower().strip()
    cells = []

    # 1. LOGISTIC REGRESSION (Heart Failure)
    if "logistic" in topic_key:
        ds_info = TOPIC_DATASETS["logistic_regression"]

        # Cell 1: Load Data & Safe Check
        cells.append({
            "title": "1. Load Dataset & Inspection",
            "code": _get_file_check_snippet(ds_info["path"], ds_info["name"]) + f"""
# Load Heart Failure Prediction Dataset
df = pd.read_csv(data_path)

print(f"Dataset Successfully Loaded! Shape: {{df.shape}}")
df.head()"""
        })

        # Cell 2: EDA & Data Cleaning
        cells.append({
            "title": "2. EDA & Handling Missing/Outlier Values",
            "code": """# 1. Loại bỏ dòng vô lý RestingBP = 0
df = df.copy()
df['RestingBP'] = df['RestingBP'].replace(0, np.nan)

# 2. Chuyển Cholesterol = 0 thành NaN để Impute (Dữ liệu khuyết ngầm y khoa)
df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)

print("Kiểm tra giá trị Null trước khi chia dữ liệu:")
print(df.isnull().sum())"""
        })

        # Cell 3: Encoding & Scaling & Split
        cells.append({
            "title": "3. Feature Encoding, Scaling & Train/Test Split",
            "code": f"""from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. One-Hot Encoding cho các biến Categorical (Sex, ChestPainType, RestECG, ExerciseAngina, ST_Slope)
X = pd.get_dummies(df.drop(columns=['HeartDisease']), drop_first=True)
y = df['HeartDisease']

# 2. Chia Train/Test (Stratify theo nhãn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

# Fit imputer/scaler chỉ trên train để tránh data leakage.
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test = pd.DataFrame(
    imputer.transform(X_test), columns=X_test.columns, index=X_test.index
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ X_train shape: {{X_train_scaled.shape}}, X_test shape: {{X_test_scaled.shape}}")"""
        })

    # 2. DECISION TREE (Red Wine Quality)
    elif "tree" in topic_key or "decision" in topic_key:
        ds_info = TOPIC_DATASETS["decision_tree"]

        # Cell 1: Load Data & Safe Check (Xử lý cả dấu ';' lẫn ',')
        cells.append({
            "title": "1. Load Dataset & Inspection",
            "code": _get_file_check_snippet(ds_info["path"], ds_info["name"]) + f"""
# Load Red Wine Quality Dataset (Tự động nhận diện delimiter ';' hoặc ',')
try:
    df = pd.read_csv(data_path, sep=';')
    if df.shape[1] <= 1:
        df = pd.read_csv(data_path, sep=',')
except Exception:
    df = pd.read_csv(data_path)

print(f"Dataset Successfully Loaded! Shape: {{df.shape}}")
df.head()"""
        })

        # Cell 2: EDA & Drop Duplicates
        cells.append({
            "title": "2. EDA & Deduplication",
            "code": """# Kiểm tra trùng lặp (~240 dòng trùng trong tập Red Wine gốc)
duplicate_count = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp tìm thấy: {duplicate_count}")

# Loaị bỏ trùng lặp
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape sau khi loại bỏ trùng lặp: {df.shape}")"""
        })

        # Cell 3: Train/Test Split
        cells.append({
            "title": "3. Feature Selection & Train/Test Split",
            "code": f"""from sklearn.model_selection import train_test_split

X = df.drop(columns=['quality'])
y = df['quality']

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

print(f"✅ Danh sách Features: {{list(X.columns)}}")
print(f"Phân bố nhãn 'quality' trong tập Train:\\n{{y_train.value_counts().sort_index()}}")"""
        })

    # 3. K-MEANS CLUSTERING (Mall Customers)
    elif "kmeans" in topic_key or "k-means" in topic_key or "k_means" in topic_key:
        ds_info = TOPIC_DATASETS["k_means"]

        # Cell 1: Load Data & Safe Check
        cells.append({
            "title": "1. Load Dataset & Inspection",
            "code": _get_file_check_snippet(ds_info["path"], ds_info["name"]) + f"""
# Load Mall Customer Dataset
df = pd.read_csv(data_path)

print(f"Dataset Successfully Loaded! Shape: {{df.shape}}")
df.head()"""
        })

        # Cell 2: Feature Selection & Encoding & Scaling
        cells.append({
            "title": "2. Preprocessing & Feature Scaling for Clustering",
            "code": f"""from sklearn.preprocessing import StandardScaler

# 1. Loại bỏ cột định danh CustomerID (nếu có)
df_features = df.drop(columns=['CustomerID', 'Customer_ID', 'id'], errors='ignore')

# 2. Encode biến Gender / Genre (nếu có)
gender_col = [c for c in df_features.columns if c.lower() in ['gender', 'genre']]
if gender_col:
    df_encoded = pd.get_dummies(df_features, columns=gender_col, drop_first=True)
else:
    df_encoded = pd.get_dummies(df_features, drop_first=True)

# 3. Scaling đặc trưng (Bắt buộc đối với KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Xáo trộn dữ liệu theo seed={seed} để đảm bảo tính ngẫu nhiên thử nghiệm
np.random.seed({seed})
shuffled_idx = np.random.permutation(len(X_scaled))
X_scaled = X_scaled[shuffled_idx]

print("✅ Dữ liệu đã được tiền xử lý & Standard Scaled cho K-Means!")
print(f"Shape đầu vào cuối cùng: {{X_scaled.shape}}")"""
        })

    else:
        # Fallback về Logistic Regression nếu chủ đề không khớp
        return get_eda_cells("logistic_regression", seed)

    return cells


def get_dataset_code(topic: str, seed: int = None) -> list[dict]:
    """Hàm interface chính gọi bởi Pipeline."""
    return get_eda_cells(topic=topic, seed=seed)
