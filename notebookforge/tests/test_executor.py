"""Executor - 5 yêu cầu của excRes (Sprint 2.2, đề cương mục 2.3 + brainstorm mục D)

    nbclient · timeout 120s · tắt mạng · timeout_hit · failed_cell_index + traceback

Chạy: python tests/test_executor.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor import CELL_TIMEOUT, run_notebook  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="nf-exec-test-"))


def _make_nb(sources: list[str]) -> str:
    """Ghi ra 1 notebook tạm gồm các code cell cho trước, trả đường dẫn."""
    nb = {
        "cells": [
            {"id": f"c{i}", "cell_type": "code", "execution_count": None,
             "metadata": {}, "outputs": [], "source": src}
            for i, src in enumerate(sources)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = _TMP / f"nb{len(list(_TMP.glob('*.ipynb')))}.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return str(path)


def test_timeout_mac_dinh_120s():
    """CELL_TIMEOUT mặc định = 120s (đề cương mục 2.3)"""
    assert CELL_TIMEOUT == 120, f"đề cương chốt 120s, đang là {CELL_TIMEOUT}"


def test_notebook_sach_chay_duoc():
    """Notebook sạch -> success=True, không lỗi, failed_cell_index=None"""
    r = run_notebook(_make_nb(["print('hello')\n", "x = 1 + 1\nassert x == 2\n"]))
    assert r.success is True, f"phải chạy sạch, lại có {r.errors}"
    assert r.executed_cells == r.total_cells == 2
    assert r.failed_cell_index is None
    assert r.timeout_hit is False


def test_timeout_set_co_timeout_hit():
    """Cell chạy quá giờ -> timeout_hit=True, đúng index cell bị treo"""
    r = run_notebook(
        _make_nb(["print('truoc')\n", "import time\ntime.sleep(60)\n", "print('sau')\n"]),
        timeout=5,
    )
    assert r.timeout_hit is True, "phải set timeout_hit=True"
    assert r.success is False
    assert r.failed_cell_index == 1, f"cell treo là số 1, báo {r.failed_cell_index}"
    assert r.duration_seconds < 30, "phải dừng ngay sau timeout, không chạy tiếp"


def test_gom_het_cell_loi_khong_dung_o_cell_dau():
    """allow_errors=True -> chạy hết notebook, gom MỌI cell lỗi trong 1 lượt"""
    r = run_notebook(_make_nb([
        "a = 1\n",
        "raise ValueError('loi co y')\n",
        "print('van chay tiep')\n",
        "undefined_name\n",
    ]))
    assert r.executed_cells == 4, "phải chạy hết 4 cell dù cell 1 lỗi"
    assert r.failed_cell_count == 2, f"phải bắt 2 lỗi, được {r.failed_cell_count}"
    assert r.failed_cell_index == 1, "index cell hỏng ĐẦU TIÊN"
    assert {e.ename for e in r.errors} == {"ValueError", "NameError"}


def test_moi_loi_deu_co_traceback():
    """Mỗi CellError phải kèm traceback (Huy cần để viết feedback)"""
    r = run_notebook(_make_nb(["raise RuntimeError('xyz')\n"]))
    assert r.errors, "phải có lỗi"
    err = r.errors[0]
    assert err.traceback_tail, "traceback_tail rỗng"
    assert "RuntimeError" in err.traceback_tail[-1]
    assert "\x1b[" not in "".join(err.traceback_tail), "phải bỏ mã màu ANSI"


def test_tat_mang():
    """Notebook không ra được internet (đề cương mục 2.3: 'tắt mạng')

    Lỗi này từng lọt: kernel_env= của NotebookClient bị nuốt im lặng, biến môi
    trường không vào tới kernel nên mạng vẫn thông suốt Sprint 2.
    """
    r = run_notebook(_make_nb([
        "import os\nassert os.environ.get('HTTP_PROXY'), 'kernel KHONG nhan duoc HTTP_PROXY'\n",
        "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=10)\n",
    ]), timeout=30)
    assert r.errors, "urlopen phải thất bại khi đã tắt mạng"
    assert r.errors[0].cell_index == 1, (
        f"cell 0 phải PASS (kernel nhận được biến), lỗi đầu tiên phải ở cell 1, "
        f"đang là cell {r.errors[0].cell_index}: {r.errors[0].evalue[:80]}"
    )


def test_luon_ghi_notebook_da_chay():
    """executed_nb_path luôn được ghi, kể cả khi notebook lỗi"""
    r = run_notebook(_make_nb(["raise ValueError('x')\n"]))
    assert r.executed_nb_path, "thiếu executed_nb_path"
    assert Path(r.executed_nb_path).is_file()


def test_file_khong_ton_tai_thi_raise():
    """File hỏng/không có -> raise rõ ràng, không trả ExcRes rỗng"""
    try:
        run_notebook(str(_TMP / "khong-co-that.ipynb"))
    except FileNotFoundError:
        return
    raise AssertionError("phải raise FileNotFoundError")


if __name__ == "__main__":
    from tests._runner import run_module

    raise SystemExit(run_module(__name__))
