"""agents/verifier.py - HUY

Verifier gồm 2 lớp:
    (a) rule_checks  - 5 quy tắc Python thuần, không cần LLM   [Sprint 2.1]
    (b) llm_scores   - Groq Llama 3.3 70B chấm 4 tiêu chí 1-5  [Sprint 2.2]

average_score = trung bình 4 tiêu chí LLM. < 3.5 -> fail -> retry.
"""

from __future__ import annotations

import json
import re

from schemas import (  # PASS_THRESHOLD lấy từ schemas, không tự khai lại
    PASS_THRESHOLD,
    ExcRes,
    ResearchBundle,
    VerifierReport,
)

# Tên biến kết quả đánh giá - gán thẳng số vào mấy biến này nghĩa là LLM ghi
# sẵn đáp án thay vì để người học tự tính.
_HARDCODE_PATTERN = (
    r"\b(accuracy|acc|score|f1|precision|recall|silhouette|inertia)"
    r"\s*=\s*[\d.]+"
)

# 'assert' phải đứng đầu câu lệnh mới tính. \b để 'assert_frame_equal(...)'
# không bị nhận nhầm là một assert.
_ASSERT_PATTERN = r"^[ \t]*assert\b"


def _strip_comments(text: str) -> str:
    """Bỏ phần comment của từng dòng.

    Cần thiết vì notebook do LLM sinh hay có code bị comment lại, ví dụ
    '# assert abs(sigmoid(0) - 0.5) < 1e-9'. Đó là câu nhắc, không phải test
    thật - tính vào là notebook không có assert nào vẫn qua cửa.

    Cắt thô ở dấu '#' đầu tiên: dấu '#' nằm trong chuỗi cũng bị cắt oan, chấp
    nhận được với rule check.
    """
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def count_asserts(nb: dict) -> int:
    """Đếm số câu lệnh assert THẬT trong notebook.

    Golden set có min_asserts cho từng case, harness dùng hàm này để đối chiếu.
    """
    return sum(
        len(re.findall(_ASSERT_PATTERN, _strip_comments("".join(c["source"])), re.M))
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )


def rule_checks(nb: dict) -> dict:
    """5 quy tắc, mỗi quy tắc pass hoặc fail."""
    results = {}
    cells = nb["cells"]

    # Quy tắc 1: phải có ít nhất 3 cell markdown (hướng dẫn)
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    results["has_instructions"] = len(md_cells) >= 3

    code_texts = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    # Quy tắc 3, 4, 5 chấm trên phần code đã bỏ comment. Riêng quy tắc 2 phải
    # đọc bản gốc, vì TODO vốn dĩ nằm trong comment.
    code_only = [_strip_comments(t) for t in code_texts]

    # Quy tắc 2: phải có ít nhất 1 code cell có TODO
    results["has_todo"] = any("TODO" in t for t in code_texts)

    # Quy tắc 3: phải có test cell với assert thật, không phải assert bị comment
    results["has_assert"] = any(re.search(_ASSERT_PATTERN, t, re.M) for t in code_only)

    # Quy tắc 4: không được hardcode đáp án.
    # Bắt theo TÊN BIẾN kết quả, không bắt theo hình dạng con số - bản cũ
    # dùng r"=\s*0\.\d{3,}" bắt oan cả siêu tham số hợp lệ (tol=0.0001,
    # learning_rate=0.001) mà lại để lọt 'acc = 0.85'.
    # 'acc = accuracy_score(...)' không khớp vì sau dấu = không phải chữ số.
    hardcoded = [
        m
        for t in code_only
        for m in re.findall(_HARDCODE_PATTERN, t, flags=re.IGNORECASE)
    ]
    results["no_hardcoded_answers"] = len(hardcoded) == 0

    # Quy tắc 5: dataset phải có train/test split
    results["has_train_test_split"] = any("train_test_split" in t for t in code_only)

    return results


def llm_judge(nb: dict, bundle: ResearchBundle) -> dict:
    """[Sprint 2.2] Groq chấm 4 tiêu chí, trả dict điểm 1-5 + feedback.

    Prompt: prompts/verifier.txt
    """
    raise NotImplementedError("TODO(Huy) Sprint 2.2: gọi judge_llm")


def run_verifier(nb_path: str, exc: ExcRes, bundle: ResearchBundle) -> VerifierReport:
    """Chấm điểm một notebook đã chạy.

    Args:
        nb_path: file .ipynb đã được Executor chạy.
        exc: kết quả thực thi - status, traceback, failed_cell_index.
            Notebook fail execution thì executability phải bị trừ nặng.
        bundle: nguồn gốc kiến thức, để chấm groundedness (nội dung có bám
            KB không hay LLM tự bịa).

    Returns:
        VerifierReport. Hoàng đọc average_score + decision để quyết định
        retry; feedback quay ngược thành prior_feedback của vòng sau nên
        phải cụ thể: "[CELL 7] lỗi X. FIX: cách sửa."
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    checks = rule_checks(nb)
    # TODO(Huy) Sprint 2.2: scores = llm_judge(nb, bundle) -> average_score
    # TODO(Huy) Sprint 2.2: retry_history nối thêm attempt trước, dùng luôn
    #     để xuất quality_report.md (Nam hiển thị cái này)
    raise NotImplementedError("TODO(Huy) Sprint 2.2: ghép rule_checks + llm_judge")


# --- Self-test: chạy `python -m agents.verifier` từ thư mục gốc ----------
# Notebook giả tối thiểu, để test rule_checks mà không cần chờ Hợp sinh
# notebook thật hay chờ Hoàng làm tests/mocks.py.
_SAMPLE_NB = {
    "cells": [
        {"cell_type": "markdown", "source": ["# Logistic Regression\n"]},
        {"cell_type": "markdown", "source": ["## Module 1: Giới thiệu\n"]},
        {"cell_type": "markdown", "source": ["## Module 2: Sigmoid\n"]},
        {
            "cell_type": "code",
            "source": [
                "from sklearn.model_selection import train_test_split\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
            ],
        },
        {"cell_type": "code", "source": ["# TODO: điền công thức sigmoid\n"]},
        {"cell_type": "code", "source": ["assert acc > 0.7\n"]},
    ]
}

if __name__ == "__main__":
    for rule, ok in rule_checks(_SAMPLE_NB).items():
        print(f"{'PASS' if ok else 'FAIL'}  {rule}")
