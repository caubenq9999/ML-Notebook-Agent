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
        "name": "Heart Failure Prediction",
        "url": "https://raw.githubusercontent.com/fedesoriano/heart-failure-prediction/main/heart.csv",
        "kaggle_link": "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction",
        "target": "HeartDisease",
        "type": "classification",
        "shape": (918, 12),
        "description": "918 dòng, 12 cột (11 đặc trưng lâm sàng + 1 nhãn)"
    },
    "decision_tree": {
        "name": "Red Wine Quality",
        "url": "https://raw.githubusercontent.com/datasets/red-wine-quality-cortez-et-al-2009/master/data/winequality-red.csv",
        "kaggle_link": "https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009",
        "target": "quality",
        "type": "classification",
        "shape": (1599, 12),
        "description": "1.599 dòng, 12 cột (11 đặc trưng hóa lý + 1 điểm chất lượng)"
    },
    "k_means": {
        "name": "Mall Customer Segmentation",
        "url": "https://raw.githubusercontent.com/Stevenc3/Mall-Customer-Segmentation/main/Mall_Customers.csv",
        "kaggle_link": "https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python",
        "type": "clustering",
        "shape": (200, 5),
        "description": "200 dòng, 5 cột"
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
    # 1. LOGISTIC REGRESSION (Heart Failure)
    # ==========================================
    if "logistic" in topic_key:
        # Cell 1: Load Data
        cells.append({
            "title": "1. Load Dataset & Inspection",
            "code": f'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load Heart Failure Prediction Dataset
url = "https://raw.githubusercontent.com/fedesoriano/heart-failure-prediction/main/heart.csv"
df = pd.read_csv(url)

print(f"Dataset Shape: {{df.shape}}")
df.head()'''
        })
        
        # Cell 2: EDA & Data Cleaning (Sửa Cholesterol=0, RestingBP=0)
        cells.append({
            "title": "2. EDA & Handling Missing/Outlier Values",
            "code": f'''# 1. Loại bỏ dòng vô lý RestingBP = 0
df = df[df['RestingBP'] > 0].copy()

# 2. Chuyển Cholesterol = 0 thành NaN để Impute (Missing ngầm y khoa)
df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)

# Impute Cholesterol bằng Median theo từng nhóm HeartDisease
df['Cholesterol'] = df.groupby('HeartDisease')['Cholesterol'].transform(lambda x: x.fillna(x.median()))

print("Data after handling missing/outliers:")
print(df.isnull().sum())'''
        })

        # Cell 3: Encoding & Scaling & Split
        cells.append({
            "title": "3. Feature Encoding, Scaling & Train/Test Split",
            "code": f'''# 1. One-Hot Encoding cho các biến Categorical (Sex, ChestPainType, ST_Slope,...)
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns=['HeartDisease'])
y = df_encoded['HeartDisease']

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

url = "https://raw.githubusercontent.com/datasets/red-wine-quality-cortez-et-al-2009/master/data/winequality-red.csv"
df = pd.read_csv(url)

print(f"Dataset Shape: {{df.shape}}")
df.head()'''
        })

        # Cell 2: EDA & Drop Duplicates
        cells.append({
            "title": "2. EDA & Deduplication",
            "code": f'''# Check duplicates (~240 dòng trùng)
duplicate_count = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {{duplicate_count}}")

# Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape sau khi loại bỏ trùng lặp: {{df.shape}}")'''
        })

        # Cell 3: Train/Test Split
        cells.append({
            "title": "3. Feature Engineering & Train/Test Split",
            "code": f'''X = df.drop(columns=['quality'])
y = df['quality']

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

print(f"Features: {{list(X.columns)}}")
print(f"Distribution of target 'quality' in Train: \\n{{y_train.value_counts()}}")'''
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

url = "https://raw.githubusercontent.com/Stevenc3/Mall-Customer-Segmentation/main/Mall_Customers.csv"
df = pd.read_csv(url)

print(f"Dataset Shape: {{df.shape}}")
df.head()'''
        })

        # Cell 2: Feature Selection & Encoding & Scaling
        cells.append({
            "title": "2. Preprocessing & Feature Scaling for Clustering",
            "code": f'''# 1. Loại bỏ cột định danh CustomerID
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