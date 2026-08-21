"""eval/calibrate.py - HUY

Dò metric_threshold thật cho golden_set.py.

Vì sao cần: đề cương ghi rõ KHÔNG dùng số mặc định 0.90 của sklearn - đó là
accuracy trên toy dataset. Đặt ngưỡng 0.90 cho Heart Failure là 20/20 case
fail oan. Script này train model chuẩn (reference solution) trên dataset thật,
chạy nhiều seed, rồi đề xuất ngưỡng = trung bình trừ biên an toàn.

Chạy:
    python -m eval.calibrate            # dùng CSV trong data/
    python -m eval.calibrate --download # tự tải bộ nào có nguồn công khai

Dataset (theo đề cương phần 3), đặt trong thư mục data/:
    data/heart.csv    Heart Failure Prediction  - 918 dòng x 12 cột
                      kaggle.com/datasets/fedesoriano/heart-failure-prediction
    data/winequality-red.csv  Red Wine Quality  - 1599 dòng x 12 cột
                      tự tải được từ UCI bằng cờ --download
    data/Mall_Customers.csv   Mall Customer     - 200 dòng x 5 cột
                      kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SEEDS = [0, 1, 2, 3, 4]

# Biên an toàn trừ đi từ điểm trung bình. Ngưỡng phải thấp hơn điểm thật một
# chút, nếu không thì dao động giữa các seed cũng đủ làm case fail oan.
MARGIN = 0.05

UCI_WINE_ZIP = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


# ---------------------------------------------------------------------------
# Tải dữ liệu
# ---------------------------------------------------------------------------


def download_wine(dest: Path) -> Path:
    """Tải Red Wine Quality từ UCI - đúng nguồn gốc của bản trên Kaggle."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(UCI_WINE_ZIP, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith("winequality-red.csv"))
        dest.write_bytes(z.read(name))
    return dest


# ---------------------------------------------------------------------------
# Reference solution cho từng topic
# ---------------------------------------------------------------------------


def calibrate_logistic_regression(csv: Path) -> list[float]:
    """Heart Failure - phân loại nhị phân HeartDisease."""
    df = pd.read_csv(csv)

    # Làm sạch theo đúng ghi chú trong đề cương: Cholesterol = 0 là vô lý.
    df = df.copy()
    for col in ("Cholesterol", "RestingBP"):
        if col in df:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    y = df["HeartDisease"]
    X = df.drop(columns=["HeartDisease"])
    cat = X.select_dtypes(include="object").columns.tolist()
    num = [c for c in X.columns if c not in cat]

    scores = []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        # Scale SAU khi split - đây chính là chỗ notebook hay bị leakage.
        pipe = Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        pipe.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_te, pipe.predict(X_te)))
    return scores


def calibrate_decision_tree(csv: Path) -> list[float]:
    """Red Wine - gộp quality thành 3 nhóm (đề cương yêu cầu, vì nhãn gốc lệch)."""
    df = pd.read_csv(csv, sep=";")
    y = pd.cut(df["quality"], bins=[0, 4, 6, 10], labels=[0, 1, 2]).astype(int)
    X = df.drop(columns=["quality"])

    scores = []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        clf = DecisionTreeClassifier(max_depth=5, random_state=seed)
        clf.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_te, clf.predict(X_te)))
    return scores


def calibrate_kmeans(csv: Path) -> list[float]:
    """Mall Customer - phân cụm, chấm bằng silhouette."""
    df = pd.read_csv(csv)
    df = df.drop(columns=[c for c in df.columns if "CustomerID" in c])
    gender = [c for c in df.columns if "Gender" in c or "Genre" in c]
    for c in gender:
        df[c] = (df[c].astype(str).str.lower().str[0] == "m").astype(int)

    X = StandardScaler().fit_transform(df.select_dtypes(include="number"))
    scores = []
    for seed in SEEDS:
        km = KMeans(n_clusters=5, n_init=10, random_state=seed)
        scores.append(silhouette_score(X, km.fit_predict(X)))
    return scores


TOPICS = {
    "logistic_regression": ("accuracy", "heart.csv", calibrate_logistic_regression),
    "decision_tree": ("accuracy", "winequality-red.csv", calibrate_decision_tree),
    "kmeans": ("silhouette", "Mall_Customers.csv", calibrate_kmeans),
}


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--download", action="store_true",
                        help="tải bộ nào có nguồn công khai (hiện chỉ Red Wine)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.download:
        wine = data_dir / "winequality-red.csv"
        if not wine.exists():
            print(f"tải Red Wine từ UCI -> {wine}")
            download_wine(wine)

    print(f"{'topic':22} {'metric':11} {'n':>2}  {'trung bình':>10} {'thấp nhất':>10}"
          f" {'cao nhất':>9}   ĐỀ XUẤT")
    print("-" * 88)

    missing = []
    for topic, (metric, filename, fn) in TOPICS.items():
        csv = data_dir / filename
        if not csv.exists():
            missing.append((topic, filename))
            print(f"{topic:22} {metric:11}  -  {'thiếu ' + filename:>32}")
            continue
        s = fn(csv)
        lo, hi, mean = min(s), max(s), sum(s) / len(s)
        suggest = round(mean - MARGIN, 2)
        print(f"{topic:22} {metric:11} {len(s):>2}  {mean:>10.4f} {lo:>10.4f}"
              f" {hi:>9.4f}   {metric} >= {suggest}")

    if missing:
        print()
        print("Còn thiếu dataset, tải từ Kaggle rồi bỏ vào thư mục data/:")
        for topic, filename in missing:
            print(f"  {filename:24} cho {topic}")


if __name__ == "__main__":
    main()
