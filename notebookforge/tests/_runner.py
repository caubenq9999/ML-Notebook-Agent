"""
tests/_runner.py - NotebookForge (Nhóm 19)
==========================================
Chủ sở hữu: HOÀNG. Bộ chạy test tối giản, KHÔNG cần cài pytest.

Mỗi file test_*.py chỉ cần khai các hàm tên `test_...`, rồi ở cuối file:

    if __name__ == "__main__":
        from tests._runner import run_module
        raise SystemExit(run_module(__name__))

Chạy:
    python tests/test_executor.py       # một file
    python tests/run_all.py             # tất cả

Vẫn tương thích pytest nếu ai cài: `pytest notebookforge/tests`.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def run_module(module_name: str) -> int:
    """Chạy mọi hàm test_* trong module. Trả 0 nếu qua hết, 1 nếu có lỗi."""
    module = sys.modules[module_name]
    tests = [
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("test_") and callable(obj)
    ]
    title = getattr(module, "__doc__", "") or module_name
    print(f"\n{'=' * 66}\n  {title.strip().splitlines()[0]}\n{'=' * 66}")

    passed, failed = 0, []
    for name, fn in tests:
        label = (fn.__doc__ or name).strip().splitlines()[0]
        try:
            fn()
            print(f"  [PASS] {label}")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {label}\n         {exc}")
            failed.append(name)
        except Exception:  # noqa: BLE001
            print(f"  [LỖI ] {label}")
            print("         " + traceback.format_exc().replace("\n", "\n         ")[:600])
            failed.append(name)

    print(f"\n  {passed}/{len(tests)} qua" + (f", HỎNG: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0
