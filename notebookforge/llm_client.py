"""
llm_client.py - NotebookForge (Nhóm 19)
=======================================
Chủ sở hữu: HOÀNG. Nơi DUY NHẤT được gọi LLM. Ba việc (đề cương mục 2.1, phần 2):
  1. Gọi model, retry với exponential backoff khi API lỗi.
  2. Ép output về đúng Pydantic schema, sai thì bắt model sửa.
  3. Đếm token và chi phí -> cost guard $0.30 mới có ý nghĩa.

NHÀ CUNG CẤP: Groq (đề cương mục 2.2). Groq đi theo chuẩn OpenAI chứ không phải
Anthropic, nên file này dùng SDK `openai` trỏ vào endpoint của Groq. Cùng cơ chế đó
cắm được Gemini, OpenRouter, DeepSeek và AgentRouter mà
không phải viết thêm backend - xem PROVIDERS bên dưới.

Cả nhóm dùng như sau (KHÔNG ai tự import openai/groq):

    from llm_client import call_json, call_text, get_tracker, MODEL_JUDGE

    # Trí / Hợp: cần LLM trả về đúng schema
    bundle, meta = call_json(
        prompt=open("prompts/research.txt").read().format(topic=topic),
        schema=ResearchBundle,
        session_id=profile.session_id,
    )

    # Huy: verifier phải dùng model KHÁC worker để giảm self-bias (đề cương 2.2)
    report, meta = call_json(..., schema=VerifierReport, model=MODEL_JUDGE, ...)

Tiền cộng dồn tự động theo session_id, main.py đọc lại bằng:
    get_tracker(session_id).total_usd
    get_tracker(session_id).cost_since(mark)   # tiền của riêng 1 attempt

API key: đặt biến tương ứng với provider trong .env ở gốc repo. KHÔNG commit
.env - đã có trong .gitignore.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

try:  # tuỳ chọn - không có cũng chạy, chỉ là phải tự export biến môi trường
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

T = TypeVar("T", bound=BaseModel)

for _stream in (sys.stdout, sys.stderr):  # console Windows cp1252 -> log tiếng Việt ra rác
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# ---------------------------------------------------------------------------
# Nhà cung cấp - tất cả đều nói chuẩn OpenAI nên dùng chung một SDK
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, tuple[str, str]] = {
    #  tên          (base_url,                                              biến môi trường chứa key)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    # Có thể override vì endpoint AgentRouter phụ thuộc cổng/tổ chức của tài khoản.
    "agentrouter": (
        os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1"),
        "AGENTROUTER_API_KEY",
    ),
}

PROVIDER = os.getenv("NOTEBOOKFORGE_PROVIDER", "groq")

# HAI MODEL, CỐ Ý KHÁC NHAU - đề cương mục 2.2:
#   Judge (Verifier) không được trùng model với Worker, nếu không nó tự chấm bài
#   của chính mình (self-bias). Đề cương chỉ đích danh Groq Llama 3.3 70B làm judge.
MODEL = os.getenv("NOTEBOOKFORGE_MODEL", "openai/gpt-oss-120b")
MODEL_JUDGE = os.getenv("NOTEBOOKFORGE_MODEL_JUDGE", "qwen/qwen3.6-27b")

MAX_TOKENS = int(os.getenv("NOTEBOOKFORGE_MAX_TOKENS", "16000"))
TEMPERATURE = float(os.getenv("NOTEBOOKFORGE_TEMPERATURE", "0.3"))
MAX_RETRIES = int(os.getenv("NOTEBOOKFORGE_LLM_RETRIES", "3"))
DEEPSEEK_THINKING = os.getenv(
    "NOTEBOOKFORGE_DEEPSEEK_THINKING", "disabled"
).strip().lower()
BASE_DELAY = 1.0  # giây, nhân đôi sau mỗi lần fail (exponential backoff)

# Giá USD / 1 TRIỆU token. DeepSeek dùng mức PEAK để cost guard luôn bảo thủ;
# off-peak bằng một nửa. Input trong PRICING là giá cache-miss.
PRICING: dict[str, tuple[float, float]] = {
    #  model id                    (input, output)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "deepseek-v4-flash": (0.44, 1.32),
    "deepseek-v4-pro": (1.32, 3.96),
    "deepseek-v4-flash-vision-exp": (0.44, 1.32),
}
# (cache-hit input, cache-miss input, output), đều là mức peak / 1M token.
DEEPSEEK_PEAK_PRICING: dict[str, tuple[float, float, float]] = {
    "deepseek-v4-flash": (0.014, 0.44, 1.32),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
    "deepseek-v4-flash-vision-exp": (0.014, 0.44, 1.32),
}
DEFAULT_PRICE = (0.59, 0.79)  # model lạ -> lấy giá model đắt nhất cho an toàn


# ---------------------------------------------------------------------------
# Lỗi riêng - để agents/ bắt đúng loại thay vì bắt Exception chung
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Gốc của mọi lỗi trong file này."""


class LLMSchemaError(LLMError):
    """Hết số lần retry mà output vẫn không parse được về schema."""


# ---------------------------------------------------------------------------
# Đếm token và tiền
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """Chi tiết một lần gọi LLM."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    api_calls: int = 1  # >1 nếu phải gọi lại vì output sai schema
    finish_reason: str | None = None
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Quy token ra USD theo bảng PRICING."""
    price_in, price_out = PRICING.get(model, DEFAULT_PRICE)
    million = 1_000_000
    return (input_tokens / million) * price_in + (output_tokens / million) * price_out


@dataclass
class CostTracker:
    """Cộng dồn tiền + token của một session. main.py so với COST_CAP_USD."""

    session_id: str
    total_usd: float = 0.0
    calls: list[Usage] = field(default_factory=list)

    def add(self, usage: Usage) -> None:
        self.total_usd += usage.cost_usd
        self.calls.append(usage)

    def mark(self) -> float:
        """Chụp mốc tiền hiện tại (gọi trước mỗi attempt)."""
        return self.total_usd

    def cost_since(self, mark: float) -> float:
        """Tiền tiêu từ mốc đó tới giờ = chi phí của riêng attempt này."""
        return round(self.total_usd - mark, 6)

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.calls)


_TRACKERS: dict[str, CostTracker] = {}


def get_tracker(session_id: str) -> CostTracker:
    """Lấy (hoặc tạo) bộ đếm của một session."""
    if session_id not in _TRACKERS:
        _TRACKERS[session_id] = CostTracker(session_id=session_id)
    return _TRACKERS[session_id]


def reset_tracker(session_id: str) -> None:
    _TRACKERS.pop(session_id, None)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Tạo client 1 lần rồi dùng lại (giữ connection pool)."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("Chưa cài SDK. Chạy: pip install -r requirements.txt") from exc

        if PROVIDER not in PROVIDERS:
            raise LLMError(
                f"NOTEBOOKFORGE_PROVIDER='{PROVIDER}' không hợp lệ. "
                f"Chọn: {list(PROVIDERS)}"
            )
        base_url, key_env = PROVIDERS[PROVIDER]
        api_key = os.getenv(key_env)
        if not api_key:
            raise LLMError(
                f"Thiếu {key_env}. Tạo file .env ở gốc repo (xem .env.example). "
                f"Lấy API key từ dashboard của provider đang chọn."
            )
        # SDK tự retry 429/5xx 2 lần; mình bọc thêm 1 lớp cho lỗi schema.
        _client = OpenAI(base_url=base_url, api_key=api_key, max_retries=2, timeout=180.0)
    return _client


def _log(msg: str) -> None:
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


def _usage_from_response(resp: Any, model: str) -> Usage:
    u = getattr(resp, "usage", None)
    inp = getattr(u, "prompt_tokens", 0) or 0
    out = getattr(u, "completion_tokens", 0) or 0
    cache_hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
    cache_miss = getattr(u, "prompt_cache_miss_tokens", 0) or 0
    finish = None
    if getattr(resp, "choices", None):
        finish = getattr(resp.choices[0], "finish_reason", None)
    cost = estimate_cost(model, inp, out)
    if PROVIDER == "deepseek" and model in DEEPSEEK_PEAK_PRICING:
        hit_price, miss_price, output_price = DEEPSEEK_PEAK_PRICING[model]
        # SDK/version cũ có thể chưa expose breakdown; khi đó coi toàn bộ là miss.
        if cache_hit == 0 and cache_miss == 0:
            cache_miss = inp
        cost = (
            cache_hit * hit_price + cache_miss * miss_price + out * output_price
        ) / 1_000_000
    return Usage(
        model=model,
        input_tokens=inp,
        output_tokens=out,
        cost_usd=cost,
        finish_reason=finish,
        cache_hit_tokens=cache_hit,
        cache_miss_tokens=cache_miss,
    )


def _is_retryable(exc: Exception) -> bool:
    """Lỗi mạng / quá tải / hết quota phút thì thử lại, lỗi sai request thì thôi."""
    name = type(exc).__name__
    if name in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff + jitter. Free tier Groq giới hạn theo phút nên chờ là qua."""
    delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
    _log(f"retry sau {delay:.1f}s ...")
    time.sleep(delay)


def _extract_model_ids(payload: Any) -> list[str]:
    """Đọc model ID từ cả OpenAI schema và các biến thể gateway phổ biến."""
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        model_ids: list[str] = []
        for item in payload:
            model_ids.extend(_extract_model_ids(item))
        return model_ids
    if not isinstance(payload, dict):
        return []

    for key in ("id", "model_id"):
        value = payload.get(key)
        if isinstance(value, str):
            return [value]

    for key in ("data", "models", "items", "result"):
        if key in payload:
            model_ids = _extract_model_ids(payload[key])
            if model_ids:
                return model_ids
    return []


def _list_available_models() -> list[str]:
    """Gọi REST trực tiếp để không phụ thuộc schema `/models` của OpenAI SDK."""
    if PROVIDER not in PROVIDERS:
        raise LLMError(
            f"NOTEBOOKFORGE_PROVIDER='{PROVIDER}' không hợp lệ. Chọn: {list(PROVIDERS)}"
        )
    base_url, key_env = PROVIDERS[PROVIDER]
    api_key = os.getenv(key_env)
    if not api_key:
        raise LLMError(f"Thiếu {key_env}.")

    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise LLMError("Chưa cài requests. Chạy: pip install -r requirements.txt") from exc

    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f" HTTP {status}" if status else ""
        raise LLMError(f"Gọi model catalog thất bại.{detail}") from exc
    except ValueError as exc:
        raise LLMError("Model catalog không trả về JSON hợp lệ.") from exc

    model_ids = sorted(set(_extract_model_ids(payload)))
    if not model_ids:
        raise LLMError("Không nhận diện được Model ID trong response catalog.")
    return model_ids


# ---------------------------------------------------------------------------
# API công khai cho cả nhóm
# ---------------------------------------------------------------------------


def call_text(
    prompt: str,
    *,
    session_id: str,
    system: str | None = None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    model: str = MODEL,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
    include_reasoning: bool | None = None,
) -> tuple[str, Usage]:
    """Gọi LLM, trả về (text, Usage). Tiền được cộng vào tracker của session."""
    client = _get_client()
    tracker = get_tracker(session_id)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # GPT-OSS và Qwen 3.6 dùng tập giá trị reasoning_effort khác nhau trên Groq.
    # Chỉ gửi các tham số riêng khi model hỗ trợ; model/provider khác tiếp tục
    # dùng request chuẩn OpenAI như trước.
    is_groq_gpt_oss = PROVIDER == "groq" and model.startswith("openai/gpt-oss-")
    is_groq_qwen36 = PROVIDER == "groq" and model.startswith("qwen/qwen3.6-")
    is_deepseek_v4 = PROVIDER == "deepseek" and model.startswith("deepseek-v4-")
    if reasoning_effort is not None and (is_groq_gpt_oss or is_groq_qwen36):
        allowed_efforts = (
            {"low", "medium", "high"} if is_groq_gpt_oss else {"none", "default"}
        )
        if reasoning_effort not in allowed_efforts:
            allowed_text = "/".join(sorted(allowed_efforts))
            raise ValueError(
                f"reasoning_effort cho {model} phải là {allowed_text}"
            )
        kwargs["reasoning_effort"] = reasoning_effort
    if include_reasoning is not None and is_groq_gpt_oss:
        kwargs["extra_body"] = {"include_reasoning": include_reasoning}
    if is_deepseek_v4:
        if DEEPSEEK_THINKING not in {"enabled", "disabled"}:
            raise ValueError(
                "NOTEBOOKFORGE_DEEPSEEK_THINKING phải là enabled hoặc disabled"
            )
        kwargs["extra_body"] = {"thinking": {"type": DEEPSEEK_THINKING}}
        if DEEPSEEK_THINKING == "enabled":
            kwargs.pop("temperature", None)  # thinking mode không dùng temperature
            effort = reasoning_effort or "low"
            if effort not in {"low", "medium", "high", "xhigh", "max"}:
                raise ValueError(
                    "reasoning_effort cho DeepSeek phải là low/medium/high/xhigh/max"
                )
            kwargs["reasoning_effort"] = effort

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - phân loại ngay bên dưới
            last_exc = exc
            if _is_retryable(exc) and attempt < MAX_RETRIES:
                _log(f"lỗi tạm thời ({type(exc).__name__}), lần {attempt}/{MAX_RETRIES}")
                _sleep_backoff(attempt)
                continue
            raise LLMError(f"Gọi LLM thất bại: {type(exc).__name__}: {exc}") from exc

        usage = _usage_from_response(resp, model)
        usage.api_calls = attempt
        tracker.add(usage)

        text = (resp.choices[0].message.content or "").strip()
        if usage.finish_reason == "length":
            _log(f"CẢNH BÁO: output bị cắt vì chạm max_tokens={max_tokens}")
        return text, usage

    raise LLMError(f"Hết {MAX_RETRIES} lần thử: {last_exc}")


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Lấy khối JSON ra khỏi text (model hay bọc trong ```json ... ```)."""
    fenced = _JSON_FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def call_json(
    prompt: str,
    schema: type[T],
    *,
    session_id: str,
    system: str | None = None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    model: str = MODEL,
    reasoning_effort: str | None = None,
    include_reasoning: bool | None = None,
) -> tuple[T, Usage]:
    """Gọi LLM và ép output về đúng `schema` (một class trong schemas.py).

    Sai schema thì tự gọi lại, kèm NGUYÊN VĂN lỗi validation để model tự sửa -
    tối đa MAX_RETRIES lần. Hết lần vẫn sai thì raise LLMSchemaError.

    Trả về (object đã validate, Usage của lần gọi cuối).
    """
    tracker = get_tracker(session_id)

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    base_system = ((system + "\n\n") if system else "") + (
        "Bạn phải trả lời DUY NHẤT một object JSON hợp lệ, khớp chính xác JSON Schema "
        "sau. Không thêm lời dẫn, không bọc trong markdown.\n\n"
        f"{schema_json}"
    )

    current_prompt = prompt
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        text, usage = call_text(
            current_prompt,
            session_id=session_id,
            system=base_system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            json_mode=True,  # Groq ép model trả JSON hợp lệ ngay từ tầng API
            reasoning_effort=reasoning_effort,
            include_reasoning=include_reasoning,
        )
        usage.api_calls = attempt
        try:
            obj = schema.model_validate_json(_extract_json(text))
            if attempt > 1:
                _log(f"{schema.__name__}: parse OK ở lần thử {attempt}")
            return obj, usage
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            _log(f"{schema.__name__}: output sai schema (lần {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                break
            # Đưa nguyên văn lỗi lại cho model - cách sửa hiệu quả nhất.
            current_prompt = (
                f"{prompt}\n\n"
                f"[LẦN TRƯỚC BẠN TRẢ VỀ JSON SAI SCHEMA]\n"
                f"Output cũ:\n{text[:2000]}\n\n"
                f"Lỗi validation:\n{last_error[:2000]}\n\n"
                f"Hãy trả lại JSON đã sửa, đúng schema, không kèm lời dẫn."
            )

    raise LLMSchemaError(
        f"{schema.__name__}: sau {MAX_RETRIES} lần vẫn không parse được.\n"
        f"Lỗi cuối: {last_error}\n"
        f"Tổng tiền session đã tiêu: ${tracker.total_usd:.4f}"
    )


__all__ = [
    "PROVIDER",
    "PROVIDERS",
    "MODEL",
    "MODEL_JUDGE",
    "PRICING",
    "DEEPSEEK_PEAK_PRICING",
    "Usage",
    "CostTracker",
    "LLMError",
    "LLMSchemaError",
    "estimate_cost",
    "get_tracker",
    "reset_tracker",
    "call_text",
    "call_json",
]


if __name__ == "__main__":
    # `python llm_client.py` -> in cấu hình + ước tính chi phí. KHÔNG gọi API.
    # `--list-models` là tiện ích tùy chọn; không phải provider nào cũng hỗ trợ.
    base_url, key_env = PROVIDERS.get(PROVIDER, ("?", "?"))
    print(f"provider     : {PROVIDER}  ({base_url})")
    print(f"model worker : {MODEL}")
    same = MODEL_JUDGE == MODEL
    print(f"model judge  : {MODEL_JUDGE}" + ("  <-- TRÙNG worker, có self-bias!" if same
                                             else "  (khác worker, đúng đề cương 2.2)"))
    print(f"có {key_env:<18}: {bool(os.getenv(key_env))}")
    print(f"temperature  : {TEMPERATURE}, max_tokens={MAX_TOKENS}, retries={MAX_RETRIES}")

    # Ước tính CẢ SESSION theo đúng pipeline đề cương, không phải 1 call lẻ:
    # research(1) + curriculum(1) + 2 x [notebook_gen + verifier]
    steps = [("research", 5_000, 2_000, 1), ("curriculum", 8_000, 3_000, 1),
             ("notebook_gen", 10_000, 8_000, 2), ("verifier", 15_000, 2_000, 2)]
    print("\nƯớc tính chi phí TRỌN 1 SESSION (2 vòng sinh, pipeline đề cương):")
    for name in PRICING:
        total = sum(estimate_cost(name, i, o) * n for _, i, o, n in steps)
        print(f"  {name:<26} ${total:.4f}   {'trong trần' if total <= 0.30 else 'VƯỢT trần'}")
    print("\nTrần đề cương = $0.30/notebook.")
    print("Bảng trên chỉ là ước lượng; đối chiếu dashboard của provider trước khi")
    print("đưa số chi phí vào báo cáo.")

    if "--list-models" in sys.argv:
        print("\nModel mà tài khoản hiện truy cập được:")
        try:
            available = _list_available_models()
        except Exception as exc:  # noqa: BLE001 - tiện ích chẩn đoán ở CLI
            raise SystemExit(
                f"Không lấy được model catalog: {type(exc).__name__}: {exc}"
            ) from exc
        for model_id in available:
            print(f"  {model_id}")

    if "--smoke" in sys.argv:
        print("\nSmoke test Chat Completions:")
        try:
            answer, smoke_usage = call_text(
                "Reply with exactly one word: OK",
                session_id="llm-client-smoke",
                model=MODEL,
                max_tokens=16,
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001 - tiện ích chẩn đoán ở CLI
            raise SystemExit(
                f"Smoke test thất bại: {type(exc).__name__}: {exc}"
            ) from exc
        print(f"response      : {answer}")
        print(
            f"usage         : input={smoke_usage.input_tokens}, "
            f"output={smoke_usage.output_tokens}, cost=${smoke_usage.cost_usd:.6f}"
        )
