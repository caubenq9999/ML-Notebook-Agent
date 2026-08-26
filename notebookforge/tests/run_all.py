"""
tests/run_all.py - chạy toàn bộ test của nhóm

    python tests/run_all.py

Không cần cài pytest. Exit code 0 = qua hết, 1 = có lỗi -> cắm thẳng vào
GitHub Actions được (đề cương mục 2.8).

KHÔNG gọi LLM, không tốn quota. Riêng test_executor có bật kernel thật nên
mất khoảng 1 phút.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODULES = [
    "tests.test_api_schemas",
    "tests.test_pipeline",
    "tests.test_notebook_gen",
    "tests.test_dataset_injector",
    "tests.test_ui_report_adapter",
    "tests.test_executor",
]


def main() -> int:
    from tests._runner import run_module

    failed = []
    for name in MODULES:
        importlib.import_module(name)
        if run_module(name) != 0:
            failed.append(name)

    print(f"\n{'=' * 66}")
    if failed:
        print(f"  HỎNG {len(failed)}/{len(MODULES)} nhóm test: {', '.join(failed)}")
        return 1
    print(f"  TẤT CẢ {len(MODULES)} NHÓM TEST ĐỀU QUA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
