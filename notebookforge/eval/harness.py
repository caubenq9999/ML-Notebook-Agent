"""eval/harness.py - HUY

Chạy toàn bộ golden set qua pipeline, xuất bảng metric.
Chạy lại mỗi lần đổi prompt hoặc đổi model (GitHub Actions).

Các chỉ số cần xuất:
    - Execution Pass Rate      mục tiêu >= 80%
    - Model Performance        mục tiêu >= 80% đạt benchmark
    - Leakage Detection        báo cáo % case có rò rỉ dữ liệu
    - Average Rubric Score     mục tiêu >= 3.5/5
    - Cost & Runtime           <= $0.30 và <= 3 phút / case
    - Cohen's Kappa            >= 0.70 (chấm tay 10 case, 3 người - chạy riêng)
"""

from __future__ import annotations

import argparse

from eval.golden_set import GOLDEN_SET


def run_case(case: dict) -> dict:
    """Chạy 1 case qua pipeline, trả về dict metric của case đó."""
    raise NotImplementedError("TODO(Huy): gọi main.generate() + gom metric")


def run_all(cases: list[dict] | None = None) -> list[dict]:
    """Chạy toàn bộ golden set."""
    return [run_case(c) for c in (cases or GOLDEN_SET)]


def summarize(results: list[dict]) -> dict:
    """Tổng hợp thành bảng metric cuối cùng."""
    raise NotImplementedError("TODO(Huy): tính pass rate, avg score, cost...")


def report_markdown(summary: dict) -> str:
    """Xuất bảng metric dạng markdown để dán vào báo cáo."""
    raise NotImplementedError("TODO(Huy): render bảng markdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="chạy 1 case, vd GS-001")
    parser.add_argument("--no-verifier", action="store_true",
                        help="ablation: tắt Verifier để so sánh")
    args = parser.parse_args()

    cases = [c for c in GOLDEN_SET if c["id"] == args.case] if args.case else None
    print(report_markdown(summarize(run_all(cases))))
