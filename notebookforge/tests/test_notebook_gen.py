"""Notebook Generator - local JSON parsing and one bounded regeneration."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import notebook_gen
import llm_client


def _valid_response() -> str:
    return json.dumps(
        {
            "cells": [
                {"cell_type": "markdown", "source": "# Tổng quan"},
                {
                    "cell_type": "code",
                    "source": "# DATASET_INJECTION_PLACEHOLDER",
                },
            ]
        },
        ensure_ascii=False,
    )


def test_notebook_gen_tat_json_mode_va_dung_budget_rieng():
    """NotebookGen không dùng JSON enforcement và giữ output budget riêng."""
    original = notebook_gen.call_text
    seen: list[dict] = []

    def fake_call_text(prompt, **kwargs):
        seen.append({"prompt": prompt, **kwargs})
        return _valid_response(), None

    notebook_gen.call_text = fake_call_text
    try:
        cells = notebook_gen._request_notebook_cells("prompt", "session-test")
    finally:
        notebook_gen.call_text = original

    assert len(cells) == 2
    assert seen[0]["json_mode"] is False
    assert seen[0]["max_tokens"] == notebook_gen.notebook_request_max_tokens("prompt")
    assert seen[0]["max_tokens"] == notebook_gen.NOTEBOOK_MAX_TOKENS
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[0]["include_reasoning"] is False
    assert seen[0]["session_id"] == "session-test"


def test_notebook_gen_retry_mot_lan_khi_json_hong():
    """JSON hỏng lần đầu phải regenerate; lần hai hợp lệ thì trả cells."""
    original = notebook_gen.call_text
    responses = iter(["{not-json", _valid_response()])
    prompts: list[str] = []

    def fake_call_text(prompt, **kwargs):
        prompts.append(prompt)
        return next(responses), None

    notebook_gen.call_text = fake_call_text
    try:
        cells = notebook_gen._request_notebook_cells("prompt gốc", "session-retry")
    finally:
        notebook_gen.call_text = original

    assert len(cells) == 2
    assert len(prompts) == 2
    assert prompts[0] == "prompt gốc"
    assert "LẦN TRƯỚC OUTPUT" in prompts[1]


def test_retry_feedback_duoc_gop_va_gioi_han_do_dai():
    """Hai feedback cùng lỗi/cell chỉ giữ bản sửa cụ thể ở phía sau."""
    feedback = (
        "[CELL 13] TypeError: unexpected multi_class. FIX: sửa theo traceback. "
        "[CELL 19] NameError: thiếu y_pred. FIX: định nghĩa biến. "
        "[CELL 13] TypeError: unexpected multi_class. FIX: xóa tham số multi_class."
    )

    compact = notebook_gen.compact_prior_feedback(feedback)

    assert compact.count("[CELL 13]") == 1
    assert "xóa tham số multi_class" in compact
    assert "[CELL 19]" in compact
    assert len(compact) <= notebook_gen.NOTEBOOK_FEEDBACK_MAX_CHARS


def test_tpm_constraint_chan_lai_request_8043_token():
    """Prompt retry test web phải được hạ output ceiling dưới budget 7600."""
    prompt = "x" * 10_531

    original_provider = notebook_gen.PROVIDER
    notebook_gen.PROVIDER = "groq"
    try:
        max_tokens = notebook_gen.notebook_request_max_tokens(prompt)
    finally:
        notebook_gen.PROVIDER = original_provider
    estimated_input = math.ceil(len(prompt) / notebook_gen.NOTEBOOK_CHARS_PER_TOKEN)

    assert max_tokens == 4_089
    assert max_tokens < notebook_gen.NOTEBOOK_MAX_TOKENS
    assert estimated_input + max_tokens == notebook_gen.NOTEBOOK_TPM_BUDGET


def test_preflight_sua_runtime_hazards_tu_test_8():
    """Chặn seaborn, DataFrame slicing và assert TODO làm notebook mẫu fail."""
    cells = [
        {
            "cell_type": "code",
            "source": (
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "feature = X_train[:, 0]\n"
                "cm = np.eye(2)\n"
                "sns.heatmap(cm, annot=True, fmt='d')"
            ),
        },
        {
            "cell_type": "code",
            "source": "answer = None\nassert answer is not None, 'chưa làm'",
        },
    ]

    prepared = notebook_gen.prepare_generated_cells(cells)

    demo = prepared[0]["source"]
    check = prepared[1]["source"]
    assert "seaborn" not in demo
    assert "plt.imshow(cm" in demo
    assert "X_train.iloc[:, 0]" in demo
    assert "except (AssertionError" in check
    compile(demo, "<demo>", "exec")
    compile(check, "<check>", "exec")


def test_preflight_sua_literal_newline_neu_no_lam_code_compile_duoc():
    """Chỉ đổi literal \\n khi việc đó sửa được syntax của cả cell."""
    cells = [
        {
            "cell_type": "code",
            "source": r"value = 1\nprint(value)\nassert value == 1",
        }
    ]

    source = notebook_gen.prepare_generated_cells(cells)[0]["source"]

    assert "\\n" not in source
    compile(source, "<repaired>", "exec")


def test_preflight_sua_newline_markdown_nhung_giu_latex():
    """Markdown xuống dòng/render math đúng; các lệnh LaTeX không bị cắt."""
    cells = [
        {
            "cell_type": "markdown",
            "source": (
                r"## Tổng quan\nNội dung chính\n\n**Công thức**: "
                r"\(\nabla f(x)\) và \(x \neq 0\). "
                r"\[\ell(y, p) = -y\log(p)\]"
            ),
        }
    ]

    source = notebook_gen.prepare_generated_cells(cells)[0]["source"]

    assert "## Tổng quan\nNội dung chính\n\n**Công thức**" in source
    assert r"$\nabla f(x)$" in source
    assert r"$x \neq 0$" in source
    assert r"$$\ell(y, p) = -y\log(p)$$" in source
    assert r"\(" not in source and r"\[" not in source


def test_preflight_khong_de_todo_ghi_de_bien_demo():
    """Biến starter trùng demo được đổi tên cùng cell assert, demo sau giữ nguyên."""
    cells = [
        {"cell_type": "code", "source": "model = object()\ny_pred = [0, 1]"},
        {
            "cell_type": "code",
            "source": "# TODO: huấn luyện\nmodel = None\ny_pred = None",
        },
        {"cell_type": "code", "source": "print(y_pred)"},
        {
            "cell_type": "code",
            "source": "assert model is not None\nassert y_pred is not None",
        },
    ]

    prepared = notebook_gen.prepare_generated_cells(cells)

    assert "model_exercise_1 = None" in prepared[1]["source"]
    assert "y_pred_exercise_1 = None" in prepared[1]["source"]
    assert prepared[2]["source"] == "print(y_pred)"
    assert "model_exercise_1 is not None" in prepared[3]["source"]
    assert "y_pred_exercise_1 is not None" in prepared[3]["source"]


def test_llm_client_forward_reasoning_cho_groq_gpt_oss():
    """llm_client chỉ forward reasoning controls cho đúng Groq GPT-OSS."""
    original_client = llm_client._client
    original_provider = llm_client.PROVIDER
    seen: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 20})()
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"usage": usage, "choices": [choice]})()

    fake_client = type(
        "FakeClient",
        (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    llm_client._client = fake_client
    llm_client.PROVIDER = "groq"
    try:
        text, usage = llm_client.call_text(
            "prompt",
            session_id="reasoning-forward-test",
            model="openai/gpt-oss-120b",
            reasoning_effort="low",
            include_reasoning=False,
        )
    finally:
        llm_client._client = original_client
        llm_client.PROVIDER = original_provider
        llm_client.reset_tracker("reasoning-forward-test")

    assert text == '{"ok": true}'
    assert usage.finish_reason == "stop"
    assert seen["reasoning_effort"] == "low"
    assert seen["extra_body"] == {"include_reasoning": False}


def test_llm_client_forward_non_thinking_cho_groq_qwen36():
    """Qwen 3.6 Judge nhận reasoning_effort=none và không dùng extra_body GPT-OSS."""
    original_client = llm_client._client
    original_provider = llm_client.PROVIDER
    seen: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 20})()
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"usage": usage, "choices": [choice]})()

    fake_client = type(
        "FakeClient",
        (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    llm_client._client = fake_client
    llm_client.PROVIDER = "groq"
    try:
        text, _usage = llm_client.call_text(
            "prompt",
            session_id="qwen-non-thinking-test",
            model="qwen/qwen3.6-27b",
            reasoning_effort="none",
        )
    finally:
        llm_client._client = original_client
        llm_client.PROVIDER = original_provider
        llm_client.reset_tracker("qwen-non-thinking-test")

    assert text == '{"ok": true}'
    assert seen["reasoning_effort"] == "none"
    assert "extra_body" not in seen


def test_llm_client_deepseek_tat_thinking_va_giu_temperature():
    """DeepSeek mặc định non-thinking để JSON ổn định và tiết kiệm."""
    original_client = llm_client._client
    original_provider = llm_client.PROVIDER
    original_thinking = llm_client.DEEPSEEK_THINKING
    seen: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            usage = type(
                "Usage",
                (),
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 80,
                    "prompt_cache_miss_tokens": 20,
                },
            )()
            message = type("Message", (), {"content": "OK"})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"usage": usage, "choices": [choice]})()

    llm_client._client = type(
        "FakeClient",
        (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    llm_client.PROVIDER = "deepseek"
    llm_client.DEEPSEEK_THINKING = "disabled"
    try:
        text, usage = llm_client.call_text(
            "prompt",
            session_id="deepseek-disabled-test",
            model="deepseek-v4-flash",
            temperature=0.2,
            reasoning_effort="low",
        )
    finally:
        llm_client._client = original_client
        llm_client.PROVIDER = original_provider
        llm_client.DEEPSEEK_THINKING = original_thinking
        llm_client.reset_tracker("deepseek-disabled-test")

    expected_cost = (80 * 0.014 + 20 * 0.44 + 20 * 1.32) / 1_000_000
    assert text == "OK"
    assert seen["temperature"] == 0.2
    assert seen["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in seen
    assert usage.cache_hit_tokens == 80
    assert usage.cache_miss_tokens == 20
    assert abs(usage.cost_usd - expected_cost) < 1e-12


def test_llm_client_deepseek_bat_thinking_va_bo_temperature():
    """Thinking mode forward effort nhưng bỏ temperature theo tài liệu DeepSeek."""
    original_client = llm_client._client
    original_provider = llm_client.PROVIDER
    original_thinking = llm_client.DEEPSEEK_THINKING
    seen: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 5})()
            message = type("Message", (), {"content": "OK"})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"usage": usage, "choices": [choice]})()

    llm_client._client = type(
        "FakeClient",
        (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    llm_client.PROVIDER = "deepseek"
    llm_client.DEEPSEEK_THINKING = "enabled"
    try:
        llm_client.call_text(
            "prompt",
            session_id="deepseek-enabled-test",
            model="deepseek-v4-flash",
            temperature=0.2,
            reasoning_effort="max",
        )
    finally:
        llm_client._client = original_client
        llm_client.PROVIDER = original_provider
        llm_client.DEEPSEEK_THINKING = original_thinking
        llm_client.reset_tracker("deepseek-enabled-test")

    assert "temperature" not in seen
    assert seen["extra_body"] == {"thinking": {"type": "enabled"}}
    assert seen["reasoning_effort"] == "max"


if __name__ == "__main__":
    from tests._runner import run_module

    raise SystemExit(run_module(__name__))
