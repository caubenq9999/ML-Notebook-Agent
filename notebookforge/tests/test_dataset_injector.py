"""Dataset Injector - ba CSV local phải tồn tại và chạy từ Executor workdir."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tools.dataset_injector import get_dataset_code


DATASET_ROOT = PACKAGE_ROOT / "datasets"
EXECUTOR_WORKDIR = PACKAGE_ROOT / "output_notebooks"


def _execute_dataset_cells(topic: str) -> dict:
    previous = Path.cwd()
    scope: dict = {}
    try:
        os.chdir(EXECUTOR_WORKDIR)
        for cell in get_dataset_code(topic, seed=42):
            exec(cell["code"], scope)
    finally:
        os.chdir(previous)
    return scope


def test_three_required_csv_files_exist() -> None:
    """Ba dataset thống nhất của nhóm phải được đóng gói local."""
    expected = {"heart.csv", "winequality-red.csv", "Mall_Customers.csv"}
    actual = {path.name for path in DATASET_ROOT.glob("*.csv")}
    assert expected <= actual, f"Thiếu dataset: {sorted(expected - actual)}"


def test_logistic_dataset_runs_without_target_leakage() -> None:
    """Heart dataset chạy được và preprocessing không impute theo nhãn."""
    generated = "\n".join(
        cell["code"] for cell in get_dataset_code("logistic_regression", seed=42)
    )
    assert "groupby('HeartDisease')" not in generated
    scope = _execute_dataset_cells("logistic_regression")
    assert scope["df"].shape[0] == 918
    assert "HeartDisease" not in scope["X_train"].columns
    assert scope["X_train_scaled"].shape[1] == scope["X_train"].shape[1]


def test_decision_tree_dataset_runs_from_executor_workdir() -> None:
    """Wine Quality đọc đúng delimiter và tạo train/test đa lớp."""
    scope = _execute_dataset_cells("decision_tree")
    assert "quality" in scope["df"].columns
    assert "quality" not in scope["X_train"].columns
    assert scope["y_train"].nunique() >= 5


def test_kmeans_dataset_runs_from_executor_workdir() -> None:
    """Mall Customers bỏ ID, encode categorical và tạo X_scaled."""
    scope = _execute_dataset_cells("kmeans")
    assert scope["df"].shape == (200, 5)
    assert "CustomerID" not in scope["df_encoded"].columns
    assert scope["X_scaled"].shape[0] == 200


if __name__ == "__main__":
    from tests._runner import run_module

    raise SystemExit(run_module(__name__))
