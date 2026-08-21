import random

# Mục tiêu: get_dataset_code(topic: str, seed: int = None) -> list[dict]
#
# Dữ liệu trả về là một `list[dict]`, mỗi dict đại diện cho một bước (Load, EDA, Split) 
# chứa 'title' (tiêu đề section) và 'code' (chuỗi Python code để inject).
#
# Cách Hợp dùng trong Pipeline tạo Notebook:
#
#   eda_cells = get_dataset_code(topic=profile.topic, seed=profile.seed)
#   
#   for cell_info in eda_cells:
#       # 1. (Tùy chọn) Thêm Markdown cell tiêu đề section
#       notebook.cells.append(create_markdown_cell(f"### {cell_info['title']}"))
#       
#       # 2. Thêm Code cell tương ứng
#       notebook.cells.append(create_code_cell(cell_info['code']))

TOPIC_DATASETS = {
    "logistic_regression": {
        "name": "Breast Cancer Wisconsin (scikit-learn)",
        "source": "sklearn.datasets.load_breast_cancer",
        "target": "target",
        "type": "classification",
        "shape": (569, 31),
        "description": "569 dòng, 30 đặc trưng số + 1 nhãn; bundled local"
    },
    "decision_tree": {
        "name": "Wine Recognition (scikit-learn)",
        "source": "sklearn.datasets.load_wine",
        "target": "target",
        "type": "classification",
        "shape": (178, 14),
        "description": "178 dòng, 13 đặc trưng số + 1 nhãn; bundled local"
    },
    "k_means": {
        "name": "Iris (scikit-learn)",
        "source": "sklearn.datasets.load_iris",
        "type": "clustering",
        "shape": (150, 4),
        "description": "150 dòng, 4 đặc trưng số; bỏ nhãn khi phân cụm"
    }
}

def get_eda_cells(topic: str, seed: int = None) -> list[dict]:
    """
    Trả về danh sách các code cell (bao gồm Load data, EDA, Data Cleaning & Preprocessing)
    đã được cá nhân hóa theo đúng đặc thù của từng Dataset.
    """
    if seed is None:
        seed = random.randint(1, 10000)

    topic_key = topic.lower().strip()
    cells = []

    # ==========================================
    # 1. LOGISTIC REGRESSION (Breast Cancer Wisconsin)
    # ==========================================
    if "logistic" in topic_key:
        # Cell 1: Load Data
        cells.append({
            "title": "1. Load Dataset & Inspection",
            "code": f'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset local bundled with scikit-learn (no network required)
from sklearn.datasets import load_breast_cancer
df = load_breast_cancer(as_frame=True).frame

print(f"Dataset Shape: {{df.shape}}")
df.head()'''
        })
        
        # Cell 2: EDA & Data Quality Checks
        cells.append({
            "title": "2. EDA & Handling Missing/Outlier Values",
            "code": f'''df = df.copy()
missing_counts = df.isnull().sum()
duplicate_count = df.duplicated().sum()

print(f"Total missing values: {{missing_counts.sum()}}")
print(f"Duplicate rows: {{duplicate_count}}")
print("Target distribution:")
print(df["target"].value_counts().sort_index())'''
        })

        # Cell 3: Encoding & Scaling & Split
        cells.append({
            "title": "3. Feature Encoding, Scaling & Train/Test Split",
            "code": f'''# Bộ Breast Cancer Wisconsin gồm các đặc trưng số, không cần one-hot encoding.
X = df.drop(columns=['target'])
y = df['target']

# 2. Chia Train/Test (Stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

# 3. Standard Scaling cho dữ liệu số (Tránh Data Leakage: fit trên Train, transform trên Test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {{X_train_scaled.shape}}, X_test shape: {{X_test_scaled.shape}}")'''
        })

    # ==========================================
    # 2. DECISION TREE (Red Wine Quality)
    # ==========================================
    elif "tree" in topic_key or "decision" in topic_key:
        # Cell 1: Load Data
        cells.append({
            "title": "1. Load Dataset & Inspection",
            "code": f'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_wine
df = load_wine(as_frame=True).frame

print(f"Dataset Shape: {{df.shape}}")
df.head()'''
        })

        # Cell 2: EDA & Drop Duplicates
        cells.append({
            "title": "2. EDA & Deduplication",
            "code": f'''# Kiểm tra số dòng trùng thực tế, không giả định trước.
duplicate_count = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {{duplicate_count}}")

# Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape sau khi loại bỏ trùng lặp: {{df.shape}}")'''
        })

        # Cell 3: Train/Test Split
        cells.append({
            "title": "3. Feature Engineering & Train/Test Split",
            "code": f'''X = df.drop(columns=['target'])
y = df['target']

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

print(f"Features: {{list(X.columns)}}")
print(f"Distribution of target in Train: \\n{{y_train.value_counts()}}")'''
        })

    # ==========================================
    # 3. K-MEANS CLUSTERING (Mall Customers)
    # ==========================================
    elif "kmeans" in topic_key or "k-means" in topic_key or "k_means" in topic_key:
        # Cell 1: Load Data
        cells.append({
            "title": "1. Load Dataset & Inspection",
            "code": f'''import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
df = load_iris(as_frame=True).frame.drop(columns=['target'])

print(f"Dataset Shape: {{df.shape}}")
df.head()'''
        })

        # Cell 2: Feature Selection & Encoding & Scaling
        cells.append({
            "title": "2. Preprocessing & Feature Scaling for Clustering",
            "code": f'''# Iris đã gồm bốn đặc trưng số, không có cột định danh/categorical.
df_encoded = df.copy()

# Scaling đặc trưng (rất quan trọng với KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Xáo trộn dữ liệu theo seed={seed}
np.random.seed({seed})
shuffled_idx = np.random.permutation(len(X_scaled))
X_scaled = X_scaled[shuffled_idx]

print("Features processed & scaled successfully for KMeans!")
print(f"Final Input Shape for Clustering: {{X_scaled.shape}}")'''
        })

    else:
        return get_eda_cells("logistic_regression", seed)

    return cells


def get_dataset_code(topic: str, seed: int = None) -> list[dict]:
    """
    Nhận 2 tham số: topic và seed.
    Trả về danh sách các dict chứa title và code từng cell phục vụ EDA & Preprocessing.
    """
    return get_eda_cells(topic=topic, seed=seed)
