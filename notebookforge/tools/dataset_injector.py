TOPIC_DATASETS = {
    "logistic_regression": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "target": "Outcome",
        "type": "classification"
    },
    "decision_tree": {
        "url": "https://raw.githubusercontent.com/selva86/datasets/master/iris.csv",
        "target": "species",
        "type": "classification"
    },
    "k_means": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv",
        "type": "clustering"
    }
}

def get_dataset_code(topic: str, seed: int = None) -> str:
    """
    Sinh Python code dạng string để inject vào cell đầu tiên của Notebook.
    Tự động chọn dataset thực tế từ URL chuẩn theo 3 topic: Logistic Regression, Decision Tree, K-Means.
    """
    if seed is None:
        seed = random.randint(1, 10000)

    topic_key = topic.lower().strip()
    
    # 1. LOGISTIC REGRESSION (Classification - Diabetes Dataset)
    if "logistic" in topic_key:
        url = TOPIC_DATASETS["logistic_regression"]["url"]
        target = TOPIC_DATASETS["logistic_regression"]["target"]
        return f'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load dataset thực tế cho Logistic Regression (Chẩn đoán tiểu đường)
url = "{url}"
df = pd.read_csv(url)

# 2. Định nghĩa Features và Target
X = df.drop(columns=['{target}'])
y = df['{target}']

# 3. Chia Train/Test với seed={seed} động
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

print(f"Dataset Loaded Successfully! Shape: {{df.shape}}")
print(f"Features: {{list(X.columns)}}")
'''

    # 2. DECISION TREE (Classification - Iris Dataset)
    elif "tree" in topic_key or "decision" in topic_key:
        url = TOPIC_DATASETS["decision_tree"]["url"]
        target = TOPIC_DATASETS["decision_tree"]["target"]
        return f'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load dataset thực tế cho Decision Tree (Phân loại hoa Iris)
url = "{url}"
df = pd.read_csv(url)

# 2. Định nghĩa Features và Target
X = df.drop(columns=['{target}'])
y = df['{target}']

# 3. Chia Train/Test với seed={seed} động
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state={seed}, stratify=y
)

print(f"Dataset Loaded Successfully! Shape: {{df.shape}}")
print(f"Features: {{list(X.columns)}}")
'''

    # 3. K-MEANS (Clustering - Unsupervised)
    elif "kmeans" in topic_key or "k-means" in topic_key or "k_means" in topic_key:
        url = TOPIC_DATASETS["k_means"]["url"]
        return f'''import pandas as pd
import numpy as np

# 1. Load dataset thực tế cho K-Means Clustering (Phân cụm)
url = "{url}"
df = pd.read_csv(url)

# Loại bỏ cột nhãn nếu có để đảm bảo bài toán Unsupervised
if 'species' in df.columns:
    X = df.drop(columns=['species'])
elif 'class' in df.columns:
    X = df.drop(columns=['class'])
else:
    X = df.copy()

# Xáo trộn dữ liệu ngẫu nhiên dựa vào seed={seed}
X = X.sample(frac=1.0, random_state={seed}).reset_index(drop=True)

print(f"Dataset Loaded Successfully for K-Means! Shape: {{X.shape}}")
print(f"Features for Clustering: {{list(X.columns)}}")
'''

    # Fallback mặc định cho Logistic Regression nếu truyền topic lạ
    else:
        return get_dataset_code("logistic_regression", seed)