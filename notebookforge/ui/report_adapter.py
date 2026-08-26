"""Chuẩn hóa payload FastAPI thành cấu trúc hiển thị ổn định cho Streamlit."""

from __future__ import annotations

from typing import Any


def normalize_report_payload(api_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Hỗ trợ cả GenerationResult mới và report phẳng của UI/mock cũ."""
    if not isinstance(api_payload, dict):
        return None

    if "decision" in api_payload and (
        "attempts_used" in api_payload or "retry_history" in api_payload
    ):
        generation = api_payload
    else:
        generation = api_payload.get("result") or api_payload.get("report")
        if not isinstance(generation, dict):
            generation = api_payload

    # Tương thích payload UI/mock cũ: {scores, status, feedback}.
    if isinstance(generation.get("scores"), dict):
        return generation

    verifier = generation.get("report")
    if not isinstance(verifier, dict):
        # Tương thích khi API trả thẳng VerifierReport.
        verifier = generation if isinstance(generation.get("llm_scores"), dict) else {}

    scores = verifier.get("llm_scores")
    if not isinstance(scores, dict):
        scores = {}

    return {
        "scores": scores,
        "rule_checks": verifier.get("rule_checks") or {},
        "feedback": verifier.get("feedback"),
        "status": generation.get("decision") or verifier.get("decision"),
        "attempts_used": generation.get("attempts_used"),
        "total_cost_usd": generation.get("total_cost_usd"),
        "retry_history": generation.get("retry_history") or [],
    }
