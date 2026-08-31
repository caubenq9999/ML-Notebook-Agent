"""
executor.py - NotebookForge (Nhóm 19)
=====================================
Chủ sở hữu: HOÀNG. Chữ ký đã chốt:

    def run_notebook(nb_path: str) -> ExcRes

Chạy notebook bằng nbclient, KHÔNG dừng ở cell lỗi đầu tiên (allow_errors=True) -
vì Huy cần thấy TẤT CẢ cell hỏng để viết feedback một lần cho đủ, thay vì mỗi
vòng retry chỉ sửa được một lỗi.

Ba thứ trả về quan trọng:
  - errors[]          : danh sách cell hỏng (index + ename + evalue + traceback)
  - timeout_hit       : True nếu có cell chạy quá CELL_TIMEOUT giây
  - executed_nb_path  : notebook đã có output, phục vụ hard gate và Judge

Chạy tay để test:
    python executor.py path/to/notebook.ipynb

CẢNH BÁO: nbclient chạy code NGAY TRÊN MÁY MÌNH, không phải sandbox thật.
Notebook do LLM sinh ra -> đừng chạy khi chưa đọc qua. Sprint 3 nếu kịp thì
bọc thêm Docker; hiện tại chỉ có 3 lớp chặn: timeout mỗi cell, working dir
riêng, và biến môi trường tắt bớt truy cập mạng của thư viện.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import CellError, ExcRes  # noqa: E402

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------

for _stream in (sys.stdout, sys.stderr):  # console Windows cp1252 -> log tiếng Việt ra rác
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# giây/cell - đề cương mục 2.3 + 2.8: "timeout 120 giây, tắt mạng".
# Cho phép hạ xuống qua biến môi trường để test nhánh timeout mà không phải ngồi
# chờ đủ 120 giây. Chạy thật thì luôn để mặc định.
CELL_TIMEOUT = int(os.getenv("NOTEBOOKFORGE_CELL_TIMEOUT", "120"))
KERNEL_NAME = "python3"
TRACEBACK_TAIL_LINES = 8  # chỉ giữ vài dòng cuối, tránh nhồi rác vào prompt của Huy

# Notebook phải tự sinh dataset bằng sklearn (đề cương mục 2.3: dataset injection),
# không được tải từ mạng. Trỏ proxy vào cổng chết để mọi request HTTP fail ngay.
_OFFLINE_ENV = {
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "http_proxy": "http://127.0.0.1:9",   # urllib đọc bản chữ thường
    "https_proxy": "http://127.0.0.1:9",
    "NO_PROXY": "",
    "no_proxy": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "SKLEARN_NO_DOWNLOAD": "1",
    "MPLBACKEND": "Agg",  # matplotlib không mở cửa sổ khi chạy headless
}


@contextmanager
def _offline_env():
    """Đặt biến môi trường chặn mạng quanh lúc chạy kernel, xong trả lại như cũ.

    KHÔNG dùng tham số kernel_env= của NotebookClient: nó bị nuốt im lặng, biến
    không vào tới kernel (đã test: HTTP_PROXY = None bên trong notebook, và
    urlopen vẫn ra internet bình thường). Kernel là tiến trình CON nên cách chắc
    ăn là sửa môi trường của tiến trình cha ngay trước khi khởi động nó.
    """
    saved = {k: os.environ.get(k) for k in _OFFLINE_ENV}
    os.environ.update(_OFFLINE_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Traceback của IPython có mã màu ANSI - bỏ đi cho prompt sạch."""
    return _ANSI.sub("", text)


def _collect_errors(nb) -> list[CellError]:
    """Quét output của từng cell, nhặt ra cell nào có output kiểu 'error'."""
    errors: list[CellError] = []
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []) or []:
            if output.get("output_type") != "error":
                continue
            tb = [_strip_ansi(line) for line in output.get("traceback", []) or []]
            errors.append(
                CellError(
                    cell_index=idx,
                    ename=output.get("ename") or "UnknownError",
                    evalue=_strip_ansi(output.get("evalue") or "")[:500],
                    traceback_tail=tb[-TRACEBACK_TAIL_LINES:],
                )
            )
    return errors


def _last_started_cell(nb) -> int:
    """Index cell code CUỐI CÙNG đã bắt đầu chạy - chính là cell bị treo.

    CellTimeoutError của nbclient không mang theo cell_index, nên phải suy ra:
    nbclient gán execution_count ngay khi cell BẮT ĐẦU chạy, nên cell có
    execution_count lớn nhất là cell đang chạy dở lúc hết giờ.
    """
    last = 0
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
            last = idx
    return last


def _count_cells(nb) -> tuple[int, int]:
    """(tổng số code cell, số code cell đã thực sự chạy)."""
    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    executed = [c for c in code_cells if c.get("execution_count") is not None]
    return len(code_cells), len(executed)


def run_notebook(nb_path: str, attempt: int = 1, timeout: int | None = None) -> ExcRes:
    """Chạy notebook và trả về ExcRes.

    KHÔNG raise khi notebook lỗi - notebook lỗi là kết quả hợp lệ của pipeline,
    Huy sẽ chấm và feedback. Chỉ raise khi bản thân file hỏng (không đọc được).

    timeout: giây/cell, để None thì lấy CELL_TIMEOUT (120s theo đề cương).
    """
    cell_timeout = CELL_TIMEOUT if timeout is None else timeout
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellTimeoutError, DeadKernelError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Chưa cài nbclient/nbformat. Chạy: pip install -r requirements.txt") from exc

    source = Path(nb_path)
    if not source.is_file():
        raise FileNotFoundError(f"Không thấy notebook: {nb_path}")

    nb = nbformat.read(source, as_version=4)
    total_cells, _ = _count_cells(nb)

    workdir = source.parent
    client = NotebookClient(
        nb,
        timeout=cell_timeout,
        kernel_name=KERNEL_NAME,
        allow_errors=True,        # chạy hết notebook, gom mọi lỗi trong 1 lượt
        record_timing=True,
        resources={"metadata": {"path": str(workdir)}},
    )

    started = time.perf_counter()
    timeout_hit = False
    fatal: CellError | None = None

    try:
        # "tắt mạng" theo đề cương mục 2.3/2.8. Không phải sandbox thật (code vẫn
        # mở socket trực tiếp được), nhưng chặn được trường hợp hay gặp nhất:
        # LLM sinh code tải dataset từ internet.
        with _offline_env():
            client.execute()
    except CellTimeoutError as exc:
        # allow_errors không nuốt timeout - phải bắt riêng.
        timeout_hit = True
        fatal = CellError(
            cell_index=_last_started_cell(nb),
            ename="CellTimeoutError",
            evalue=f"Có cell chạy quá {cell_timeout}s",
            traceback_tail=[str(exc)[:500]],
        )
    except DeadKernelError as exc:
        fatal = CellError(
            cell_index=0,
            ename="DeadKernelError",
            evalue="Kernel chết giữa chừng (thường do hết RAM hoặc vòng lặp vô hạn)",
            traceback_tail=[str(exc)[:500]],
        )
    except Exception as exc:  # noqa: BLE001 - mọi lỗi khác vẫn phải thành ExcRes
        fatal = CellError(
            cell_index=0,
            ename=type(exc).__name__,
            evalue=str(exc)[:500],
            traceback_tail=[],
        )

    duration = time.perf_counter() - started
    errors = _collect_errors(nb)
    if fatal is not None:
        errors.append(fatal)

    # Luôn ghi bản đã chạy, kể cả khi lỗi - hard gate/Judge cần output.
    executed_path = source.with_suffix(".executed.ipynb")
    try:
        import nbformat as _nbf

        _nbf.write(nb, executed_path)
        executed_str: str | None = str(executed_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[executor] không ghi được notebook đã chạy: {exc}", file=sys.stderr)
        executed_str = None

    _, executed_cells = _count_cells(nb)

    return ExcRes(
        nb_path=str(source),
        attempt=attempt,
        success=not errors,
        total_cells=total_cells,
        executed_cells=executed_cells,
        errors=errors,
        duration_seconds=round(duration, 3),
        timeout_hit=timeout_hit,
        executed_nb_path=executed_str,
        cost_this_attempt=0.0,  # executor không gọi LLM - main.py điền lại
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy notebook và in ra ExcRes")
    parser.add_argument("nb_path", help="Đường dẫn tới file .ipynb")
    parser.add_argument("--json", action="store_true", help="In ra JSON thay vì tóm tắt")
    args = parser.parse_args(argv)

    exc = run_notebook(args.nb_path)

    if args.json:
        print(exc.model_dump_json(indent=2))
        return 0 if exc.success else 1

    status = "OK" if exc.success else "LỖI"
    print(f"[{status}] {exc.nb_path}")
    print(f"  cell chạy được : {exc.executed_cells}/{exc.total_cells}")
    print(f"  thời gian      : {exc.duration_seconds}s (timeout_hit={exc.timeout_hit})")
    print(f"  notebook output: {exc.executed_nb_path}")
    for err in exc.errors:
        print(f"  - cell {err.cell_index}: {err.ename}: {err.evalue}")
    return 0 if exc.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
