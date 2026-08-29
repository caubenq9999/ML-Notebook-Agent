---
topic: "Decision Tree"
subtopic: "Data Preprocessing, Encoding & Feature Importance"
level: "Intermediate"
doc_id: "dt_03"
source_url: "https://scikit-learn.org/stable/modules/tree.html"
key_concepts:
  - "Tính bất biến với thang đo (Scale Invariance)"
  - "Thiên vị do biến có nhiều giá trị (High Cardinality Bias)"
  - "So sánh Ordinal Encoding và One-Hot Encoding"
  - "Độ quan trọng Gini (Gini Importance / MDI)"
  - "Độ quan trọng theo hoán vị (Permutation Importance)"
  - "Rời rạc hóa nhãn mục tiêu (Target Binarization)"
  - "Kỹ thuật tạo đặc trưng (Feature Engineering dựa trên Domain Knowledge)"
---

# Hướng Dẫn Tiền Xử Lý Dữ Liệu & Kỹ Thuật Đặc Trưng

## 1. Tính Bất Biến Với Thang Đo (Scale Invariance)

Một trong những ưu điểm lớn nhất của Cây quyết định (Decision Tree) so với các thuật toán học máy khác là **hoàn toàn không yêu cầu chuẩn hóa dữ liệu** (Không cần `StandardScaler` hay `MinMaxScaler`).

### Tại sao Cây quyết định không quan tâm đến Thang đo?

- **Cơ chế phân tách đơn biến (Axis-aligned split):** Tại mỗi nút, cây chỉ xem xét **một đặc trưng đơn lẻ** $x_j$ và so sánh nó với một ngưỡng $t$ ($x_j \le t$).
- **Biến đổi đơn điệu (Monotonic Transformation):** Việc nhân một cột với $1000$ (chuyển từ Km sang m) hoặc lấy hàm $\log(x)$ không làm thay đổi thứ tự tương đối giữa các điểm dữ liệu. Do đó, điểm cắt ngưỡng $t$ tối ưu vẫn giữ nguyên vị trí phân loại của nó.

### So sánh yêu cầu Tiền xử lý dữ liệu:

| Thuật toán | Cần Chuẩn hóa Thang đo? | Lý do |
| --- | --- | --- |
| **Decision Tree / Random Forest** | **KHÔNG** | Phân tách dựa trên thứ tự xếp hạng (Rank-based) của từng cột độc lập. |
| **Logistic Regression / Neural Net** | **CÓ** | Dùng Gradient Descent; nếu thang đo lệch nhau, mặt mất mát sẽ bị méo. |
| **KNN / SVM / K-Means** | **CÓ** | Dựa trên khoảng cách không gian (Euclidean); cột có giá trị lớn sẽ áp đảo. |

---

## 2. Mã Hóa Biến Phân Loại & Thiên Vị Do High Cardinality

Mặc dù bất biến với thang đo, `DecisionTreeClassifier` trong Scikit-Learn vẫn yêu cầu đầu vào phải là dạng số (`float` hoặc `int`). Cách bạn biến đổi các biến phân loại (Categorical Features) thành dạng số sẽ ảnh hưởng trực tiếp đến chất lượng của cây.

### 2.1. One-Hot Encoding vs Ordinal Encoding

- **One-Hot Encoding (`OneHotEncoder`):** Tách 1 cột phân loại thành $N$ cột nhị phân ($0$ và $1$).
  - *Nhược điểm với Cây:* Tạo ra ma trận dữ liệu thưa thớt (sparse matrix). Cây phải duyệt qua rất nhiều cột phụ, làm tăng độ sâu của cây và làm suy giảm khả năng chọn đặc trưng gốc.
- **Ordinal Encoding (`OrdinalEncoder`):** Gán mỗi giá trị phân loại thành một số nguyên ($0, 1, 2, 3...$).
  - *Ưu điểm với Cây:* Giữ nguyên số lượng cột, cây vẫn có thể thực hiện phép so sánh $\le$ để chia nhóm các nhãn hiệu quả mà không làm "nổ" kích thước cây.

### 2.2. Hiện tượng Thiên vị Kích thước Tập giá trị (High Cardinality Bias)

**High Cardinality** là thuật ngữ chỉ các cột có quá nhiều giá trị duy nhất (ví dụ: `Mã_Bưu_Điện`, `ID_Nguoi_Dung`, `Ngay_Sinh`).

- **Nguyên nhân:** Một biến liên tục hoặc một biến phân loại có $100$ giá trị duy nhất sẽ cung cấp tới $99$ điểm cắt ngưỡng $t$ ứng viên cho thuật toán CART thử nghiệm. Trong khi đó, một biến nhị phân (Đúng/Sai) chỉ cung cấp đúng $1$ điểm cắt.
- **Hậu quả:** Thuật toán tham lam (Greedy Algorithm) sẽ **bị thiên vị**, luôn ưu tiên chọn các biến có High Cardinality để phân tách vì chúng dễ dàng tìm được một điểm cắt ngẫu nhiên làm giảm độ bất thuần (Impurity) trên tập Train. Điều này khiến mô hình đánh giá sai tầm quan trọng của đặc trưng và gây ra Overfitting nghiêm trọng.

---

## 3. Trích Xuất Độ Quan Trọng Đặc Trưng (Feature Importance)

Sau khi huấn luyện (`.fit()`), mô hình cung cấp thuộc tính `model.feature_importances_`. Đây chính là chỉ số **Gini Importance** (hay còn gọi là **Mean Decrease in Impurity - MDI**).

### 3.1. Cơ sở Toán học

Mức độ giảm độ bất thuần tại nút $m$ khi tách bằng đặc trưng $j$:

$$\Delta H(m) = H(m) - \left( \frac{N_L}{N_m} H(m_L) + \frac{N_R}{N_m} H(m_R) \right)$$

Độ quan trọng chưa chuẩn hóa của đặc trưng $j$ là tổng giảm Impurity trên **tất cả các nút** mà đặc trưng $j$ được chọn để tách, có trọng số là số lượng mẫu $N_m$ tại nút đó:

$$I(j) = \sum_{m \in \text{Nodes split on } j} N_m \cdot \Delta H(m)$$

Độ quan trọng chuẩn hóa (được trả về bởi `feature_importances_`):

$$\text{Importance}(j) = \frac{I(j)}{\sum_{k} I(k)} \quad \left(\text{Tổng tất cả các Feature Importances} = 1.0\right)$$

### 3.2. Hạn chế của Gini Importance (MDI) & Giải pháp

1. **Bị lệch bởi High Cardinality:** Như đã giải thích ở Mục 2.2, biến có nhiều giá trị duy nhất sẽ tự động có Gini Importance cao hơn thực tế.
2. **Tính toán trên tập Train:** MDI chỉ đo đạc mức độ giảm độ bất thuần trên tập dữ liệu huấn luyện, không phản ánh khả năng dự đoán trên tập dữ liệu mới (Test Set).
3. **Giải pháp thay thế:** Sử dụng **Permutation Importance** (`sklearn.inspection.permutation_importance`). Phương pháp này đo lường sự sụt giảm độ chính xác của mô hình trên tập Validation/Test khi bị xáo trộn ngẫu nhiên giá trị của một cột đặc trưng.

---

## 4. Mã Nguồn Minh Họa Cơ Bản

Đoạn mã dưới đây minh họa quy trình: Mã hóa dữ liệu phân loại, huấn luyện cây, trích xuất và vẽ biểu đồ độ quan trọng đặc trưng.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# Tập dữ liệu giả lập
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'Thu_Nhap': np.random.normal(15, 5, n_samples),
    'Tuoi': np.random.randint(18, 65, size=n_samples),
    'Trinh_Do': np.random.choice(['Bieu_Thong', 'Dai_Hoc', 'Thac_Si'], size=n_samples),
    'Khu_Vuc': np.random.choice(['Mien_Bac', 'Mien_Trung', 'Mien_Nam'], size=n_samples),
})
y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

# Tiền xử lý
categorical_cols = ['Trinh_Do', 'Khu_Vuc']
numerical_cols = ['Thu_Nhap', 'Tuoi']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)
X_processed = preprocessor.fit_transform(data)

# Huấn luyện
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
```

## 5. Áp Dụng Thực Tế: Hướng Dẫn Chi Tiết Tiền Xử Lý Dataset "Red Wine Quality"

Phần này sẽ áp dụng các lý thuyết ở trên vào tập dữ liệu thực tế Red Wine Quality. Tập dữ liệu này gồm 11 đặc trưng hóa học (toàn bộ là biến số liên tục) và 1 biến nhãn quality (điểm chất lượng từ 3 đến 8).

Đối với người mới học, chúng ta sẽ đi qua từng kỹ thuật tiền xử lý chuyên sâu và giải thích lý do tại sao lại làm như vậy.

### 5.1. Rời Rạc Hóa Nhãn Mục Tiêu (Target Binarization)

**Vấn đề:** Biến `quality` đang là các con số phân tán ($3, 4, 5, 6, 7, 8$). Việc yêu cầu Decision Tree đoán trúng phóc 1 con số cụ thể (Phân loại đa lớp) là rất khó và thực tế kinh doanh ít khi cần.

**Kỹ thuật xử lý:** Chuyển bài toán thành Phân loại nhị phân (Binary Classification). Ta tự định nghĩa ngưỡng:
- `quality >= 7`: Gán nhãn 1 (Rượu Ngon / High Quality).
- `quality < 7`: Gán nhãn 0 (Rượu Thường / Normal).

**Lợi ích:** Mô hình tập trung vào việc học "ranh giới" nào tạo nên một chai rượu ngon, giúp độ chính xác (Accuracy) cao hơn hẳn.

### 5.2. Kỹ Thuật Tạo Đặc Trưng (Feature Engineering)

**Vấn đề:** Đưa dữ liệu thô (raw data) vào mô hình đôi khi là chưa đủ. Máy học có thể bỏ lỡ các quy luật phức tạp nếu ta không "gợi ý" cho nó.

**Kỹ thuật xử lý:** Dựa vào kiến thức chuyên ngành (Domain Knowledge), ta tạo ra cột mới từ các cột cũ.

**Ví dụ 1:** Cột `free sulfur dioxide` (SO2 tự do) bảo vệ rượu khỏi vi khuẩn. Tuy nhiên, nó chỉ có ý nghĩa khi so sánh với tổng lượng SO2 (`total sulfur dioxide`).

**Hành động:** Ta tạo ra cột mới là Tỷ lệ SO2 bằng phép tính: `free_SO2_ratio = free sulfur dioxide / total sulfur dioxide`. Mô hình Cây sẽ dùng tỷ lệ này để phân nhánh dễ dàng hơn.

### 5.3. Không Cần Xóa Giá Trị Ngoại Lệ (Outliers)

**Vấn đề:** Cột `residual sugar` (lượng đường) phần lớn dao động quanh $2.0$. Nhưng có vài chai rượu vang ngọt vọt lên tận $15.0$. Với mô hình Hồi quy (Linear Regression), chai rượu này sẽ làm hỏng hoàn toàn đường thẳng dự đoán.

**Quy tắc với Tree:** KHÔNG CẦN XÓA! Vì Cây Quyết Định phân tách dựa trên điều kiện lớn hơn hoặc nhỏ hơn. Giá trị $15.0$ hay $5.0$ thì khi so sánh với ngưỡng cắt $t = 3.0$ (sugar > 3.0), chúng đều được gom chung vào một nhánh con. Đây chính là tính "kháng ngoại lệ" siêu việt của mô hình Cây.

### 5.4. Giải Thích Code Tiền Xử Lý Từng Bước Cụ Thể

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# --- BƯỚC 1: TẢI DỮ LIỆU ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";") # Data gốc dùng dấu chấm phẩy phân cách

# --- BƯỚC 2: TIỀN XỬ LÝ (PREPROCESSING) ---

# Kỹ thuật 1: Target Binarization (Chuyển nhãn đa lớp thành nhị phân)
# Cú pháp .astype(int) sẽ chuyển True/False thành 1/0.
df['target'] = (df['quality'] >= 7).astype(int) 

# Kỹ thuật 2: Feature Engineering (Tạo đặc trưng mới)
# Lưu ý: Cộng thêm 1e-5 (một số rất nhỏ) ở mẫu số để đề phòng lỗi "chia cho 0" (ZeroDivisionError)
df['free_SO2_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-5)
df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']

# Định nghĩa X (Đặc trưng) và y (Nhãn dự đoán)
# Ta phải bỏ đi cột 'quality' (nhãn gốc) và 'target' (nhãn đã biến đổi) khỏi X
X = df.drop(columns=['quality', 'target'])
y = df['target']

# --- BƯỚC 3: CHIA TẬP TRAIN / TEST ---
# Quan trọng cho người mới: Khi binarize (>=7), số chai vang Ngon rất ít (chỉ ~13%).
# Ta bắt buộc phải dùng tham số stratify=y để đảm bảo tập Train và Test
# đều duy trì tỷ lệ 13% này, nếu không mô hình sẽ học sai lệch.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)

# --- BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ---
# Không cần lệnh StandardScaler() như Logistic Regression
# max_depth=5 để cây không mọc quá sâu gây Overfitting
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# --- BƯỚC 5: ĐÁNH GIÁ ĐẶC TRƯNG TỐT NHẤT ---
# Thay vì dùng Gini Importance dễ bị thiên vị, ta dùng Permutation Importance
# Thuật toán sẽ thử xáo trộn ngẫu nhiên từng cột trên tập Test. 
# Cột nào bị xáo trộn mà làm độ chính xác (Accuracy) giảm thê thảm nhất -> Cột đó quan trọng nhất.
result = permutation_importance(dt_model, X_test, y_test, n_repeats=10, random_state=42)

# Trực quan hóa kết quả
sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], result.importances_mean[sorted_idx], color='teal')
plt.title("Độ quan trọng đặc trưng trên tập Test (Permutation Importance)")
plt.xlabel("Mức độ sụt giảm Accuracy khi bị xáo trộn (Càng cao càng quan trọng)")
plt.show()
```

## 6. Checklist Cốt Lõi Khi Làm Việc Với Mô Hình Cây

- Bỏ qua Scaling (Chuẩn hóa): Không tốn công viết code StandardScaler hay MinMaxScaler.
- Encoding hợp lý: Dùng OrdinalEncoder thay vì OneHotEncoder cho biến có thứ tự.
- Outliers (Ngoại lệ): Cứ để nguyên đó, Decision Tree tự biết cách gom chúng lại bằng các điểm cắt (Threshold).
- Feature Engineering: Luôn cố gắng suy nghĩ tạo ra các cột tính toán tỷ lệ, tổng, hiệu (như tỷ lệ SO2) thay vì chỉ nhét toàn bộ dữ liệu thô vào.
- Đo lường độ quan trọng: Hãy tập thói quen dùng permutation_importance trên tập Test thay vì tin hoàn toàn vào thuộc tính feature_importances_ mặc định của mô hình.
