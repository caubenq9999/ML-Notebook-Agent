"""eval/check_rules.py - HUY

Chấm rule_checks lên notebook CÓ SẴN, không cần chạy pipeline.

Dùng để soi 8 quy tắc bắt/bỏ cái gì trên notebook thật trước khi tin vào chúng -
sửa regex xong chạy lại file này là biết ngay có bắt oan không.

Khác harness.py: harness chạy cả pipeline (sinh + execute + chấm) trên golden
set, file này chỉ đọc .ipynb và chạy phần rule, không tốn tiền LLM.

Chạy từ thư mục notebookforge/:

    python -m eval.check_rules duong/dan.ipynb
    python -m eval.check_rules duong/dan/thu_muc          # quét đệ quy
    python -m eval.check_rules nb.ipynb --topic kmeans    # bài clustering
    python -m eval.check_rules thu_muc --level 2          # chấm min cell level 2
    python -m eval.check_rules thu_muc --detail           # in dòng bị rule 4 bắt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from agents.verifier import _HARDCODE_PATTERN, _strip_comments, count_asserts, rule_checks

RULES = [
    "has_instructions",
    "has_todo",
    "has_assert",
    "no_hardcoded_answers",
    "has_train_test_split",
    "has_visualization",
    "has_demo_per_module",
    "min_cells_by_level",
]


def collect(target: Path) -> list[Path]:
    """1 file .ipynb, hoặc mọi .ipynb trong thư mục (đệ quy)."""
    return [target] if target.is_file() else sorted(target.rglob("*.ipynb"))


def hardcode_hits(nb: dict) -> list[tuple[int, str]]:
    """Các dòng bị quy tắc 4 bắt, kèm số thứ tự cell - để soi bắt oan."""
    hits = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        for line in _strip_comments("".join(c["source"])).splitlines():
            if re.search(_HARDCODE_PATTERN, line, flags=re.IGNORECASE):
                hits.append((i, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("target", help="file .ipynb hoặc thư mục chứa .ipynb")
    parser.add_argument("--topic", help="topic của notebook, quy tắc 5 cần (vd kmeans)")
    parser.add_argument(
        "--level", type=int, choices=(1, 2, 3),
        help="level dùng cho min_cells_by_level (mặc định đọc metadata/beginner)",
    )
    parser.add_argument("--detail", action="store_true", help="in dòng bị quy tắc 4 bắt")
    args = parser.parse_args()

    # Console Windows mặc định cp1252, tên file / code tiếng Việt sẽ lỗi.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    target = Path(args.target)
    if not target.exists():
        print(f"Không thấy: {target}")
        return 1

    paths = collect(target)
    if not paths:
        print(f"Không có .ipynb nào trong {target}")
        return 1

    root = target if target.is_dir() else target.parent
    width = max(len(str(p.relative_to(root))) for p in paths)

    print(f"{'notebook':<{width}} " + " ".join(r[:9].rjust(9) for r in RULES) + "  asrt")
    print("-" * (width + len(RULES) * 10 + 6))

    failed = 0
    for p in paths:
        nb = json.loads(p.read_text(encoding="utf-8"))
        res = rule_checks(nb, topic=args.topic, level=args.level)
        failed += not all(res.values())
        row = " ".join(("PASS" if res[r] else "FAIL").rjust(9) for r in RULES)
        print(f"{str(p.relative_to(root)):<{width}} {row} {count_asserts(nb):5}")

        if args.detail:
            for cell_idx, line in hardcode_hits(nb):
                print(f"{'':<{width}}   [rule 4] cell {cell_idx}: {line[:70]}")

    print(f"\n{len(paths) - failed}/{len(paths)} notebook qua đủ {len(RULES)} quy tắc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
