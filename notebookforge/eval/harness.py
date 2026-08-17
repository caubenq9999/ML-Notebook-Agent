"""Chạy golden set qua pipeline và xuất bảng benchmark Markdown."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from agents.verifier import count_asserts, rule_checks
from eval.golden_set import GOLDEN_SET
from schemas import LearnerProfile


GenerateFn = Callable[[LearnerProfile], Any]


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _nested(data: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = data
    for key in path:
        value = _as_dict(value).get(key, default)
        if value is default:
            break
    return value


def _load_generate() -> tuple[GenerateFn | None, str | None]:
    errors: list[str] = []
    for module_name in ("main", "notebookforge.main"):
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            errors.append(f"{module_name}: {error}")
            continue
        generate = getattr(module, "generate", None)
        if callable(generate):
            return generate, None
        errors.append(f"{module_name}: thiếu hàm generate")
    return None, "; ".join(errors)


def _profile(case: dict[str, Any]) -> LearnerProfile:
    digits = re.sub(r"\D", "", case["id"])
    seed = int(digits or "1")
    return LearnerProfile(
        topic=case["topic"],
        level_declared=case["level"],
        level_final=case["level"],
        quiz_score=4,
        constraints=case["constraints"],
        session_id=f"eval-{case['id'].lower()}",
        dataset_seed=seed,
    )


def _call_generate(
    generate: GenerateFn,
    profile: LearnerProfile,
    no_verifier: bool,
) -> Any:
    parameters = inspect.signature(generate).parameters
    kwargs: dict[str, Any] = {}
    if "use_verifier" in parameters:
        kwargs["use_verifier"] = not no_verifier
    elif "enable_verifier" in parameters:
        kwargs["enable_verifier"] = not no_verifier
    return generate(profile, **kwargs)


def _normalise_generation(value: Any) -> dict[str, Any]:
    data = _as_dict(value)
    if data:
        return data
    # Hợp đồng cũ trong tài liệu: (nb_path, learning_path, report).
    if isinstance(value, tuple) and len(value) >= 3:
        return {
            "notebook_path": value[0],
            "learning_path": _as_dict(value[1]),
            "report": _as_dict(value[2]),
        }
    return {}


def _notebook_path(result: dict[str, Any]) -> Path | None:
    candidates = (
        _nested(result, "report", "nb_path"),
        result.get("notebook_path"),
        result.get("nb_path"),
        result.get("executed_nb_path"),
    )
    for candidate in candidates:
        if candidate and Path(str(candidate)).exists():
            return Path(str(candidate))
    return None


def _output_text(output: dict[str, Any]) -> str:
    pieces: list[str] = []
    text = output.get("text")
    if isinstance(text, list):
        pieces.extend(map(str, text))
    elif text is not None:
        pieces.append(str(text))
    data = output.get("data") or {}
    for key in ("text/plain", "application/json"):
        value = data.get(key)
        if isinstance(value, list):
            pieces.extend(map(str, value))
        elif value is not None:
            pieces.append(json.dumps(value) if isinstance(value, dict) else str(value))
    return "\n".join(pieces)


def _execution_pass(nb: dict[str, Any], result: dict[str, Any]) -> bool | None:
    for path in (
        ("execution_success",),
        ("exc", "success"),
        ("execution", "success"),
    ):
        value = _nested(result, *path)
        if isinstance(value, bool):
            return value

    code_cells = [
        cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"
    ]
    if any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    ):
        return False
    counts = [cell.get("execution_count") for cell in code_cells]
    if counts and any(count is not None for count in counts):
        return all(count is not None for count in counts)
    return None


def _metric_value(nb: dict[str, Any], metric: str) -> float | None:
    metadata = nb.get("metadata") or {}
    forge = metadata.get("notebookforge") or {}
    for source in (forge.get("metrics"), metadata.get("metrics")):
        if isinstance(source, dict) and isinstance(source.get(metric), (int, float)):
            return float(source[metric])

    pattern = re.compile(
        rf"\b{re.escape(metric)}(?:\s+score)?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        re.I,
    )
    found: list[float] = []
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            found.extend(float(value) for value in pattern.findall(_output_text(output)))
    return found[-1] if found else None


def _content(nb: dict[str, Any]) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list)
        else str(cell.get("source", ""))
        for cell in nb.get("cells", [])
    ).lower()


def _coverage(terms: list[str], content: str) -> tuple[int, int, float | None]:
    if not terms:
        return 0, 0, None
    matched = sum(term.lower() in content for term in terms)
    return matched, len(terms), matched / len(terms)


def _call_arg(call: ast.Call, index: int) -> str:
    if len(call.args) <= index:
        return ""
    try:
        return ast.unparse(call.args[index]).lower()
    except Exception:
        return ""


def _leakage_detected(nb: dict[str, Any], topic: str) -> bool | None:
    if topic == "kmeans":
        return None
    fits: list[str] = []
    evaluations: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)
        try:
            parsed = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(parsed):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "fit":
                fits.append(_call_arg(node, 0))
            elif node.func.attr in {"predict", "score"}:
                evaluations.append(_call_arg(node, 0))
    if not fits:
        return None
    if any("train" in value for value in fits) and any(
        "test" in value or "val" in value for value in evaluations
    ):
        return False
    if any(value in {"x", "df", "data"} for value in fits) and any(
        value in fits for value in evaluations
    ):
        return True
    return None


def run_case(
    case: dict[str, Any],
    *,
    generate_fn: GenerateFn | None = None,
    no_verifier: bool = False,
) -> dict[str, Any]:
    """Chạy một golden case và gom metric; không che lỗi tích hợp."""
    generate = generate_fn
    if generate is None:
        generate, load_error = _load_generate()
        if generate is None:
            return {
                "id": case["id"],
                "topic": case["topic"],
                "level": case["level"],
                "status": "BLOCKED",
                "error": f"Không tải được main.generate: {load_error}",
            }

    started = time.perf_counter()
    try:
        raw = _call_generate(generate, _profile(case), no_verifier)
    except Exception as error:
        return {
            "id": case["id"],
            "topic": case["topic"],
            "level": case["level"],
            "status": "ERROR",
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }
    runtime = time.perf_counter() - started
    result = _normalise_generation(raw)
    path = _notebook_path(result)
    if path is None:
        return {
            "id": case["id"],
            "topic": case["topic"],
            "level": case["level"],
            "status": "ERROR",
            "runtime_seconds": runtime,
            "error": "Pipeline không trả đường dẫn notebook tồn tại",
        }
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "id": case["id"],
            "topic": case["topic"],
            "level": case["level"],
            "status": "ERROR",
            "runtime_seconds": runtime,
            "error": f"Không đọc được notebook: {error}",
        }

    report = _as_dict(result.get("report"))
    scores = _as_dict(report.get("llm_scores"))
    average = report.get("average_score")
    if not isinstance(average, (int, float)) and scores:
        values = [scores.get(field) for field in (
            "executability", "groundedness", "difficulty_fit", "pedagogical_order"
        )]
        if all(isinstance(value, (int, float)) for value in values):
            average = sum(values) / 4

    metric, threshold = next(iter(case["metric_threshold"].items()))
    metric_value = _metric_value(nb, metric)
    content = _content(nb)
    modules = _coverage(case.get("expected_modules", []), content)
    skills = _coverage(case.get("expected_skills", []), content)
    forbidden = [
        term for term in case.get("must_not_have", []) if term.lower() in content
    ]
    checks = rule_checks(nb, topic=case["topic"], level=case["level"])
    decision = result.get("decision") or report.get("decision")
    return {
        "id": case["id"],
        "topic": case["topic"],
        "level": case["level"],
        "status": "COMPLETED",
        "decision": str(decision) if decision is not None else None,
        "notebook_path": str(path),
        "execution_pass": _execution_pass(nb, result),
        "rule_checks": checks,
        "rules_all_passed": all(checks.values()),
        "assert_count": count_asserts(nb),
        "min_asserts_pass": count_asserts(nb) >= case.get("min_asserts", 0),
        "average_score": float(average) if isinstance(average, (int, float)) else None,
        "metric": metric,
        "metric_value": metric_value,
        "metric_threshold": threshold,
        "model_performance_pass": (
            metric_value >= threshold if metric_value is not None else None
        ),
        "leakage_detected": _leakage_detected(nb, case["topic"]),
        "expected_modules_found": modules[0],
        "expected_modules_total": modules[1],
        "expected_modules_coverage": modules[2],
        "expected_skills_found": skills[0],
        "expected_skills_total": skills[1],
        "expected_skills_coverage": skills[2],
        "forbidden_terms_found": forbidden,
        "cost_usd": float(result.get("total_cost_usd") or 0),
        "runtime_seconds": runtime,
        "error": result.get("error"),
    }


def run_all(
    cases: list[dict[str, Any]] | None = None,
    *,
    generate_fn: GenerateFn | None = None,
    no_verifier: bool = False,
) -> list[dict[str, Any]]:
    """Chạy danh sách case; mặc định là toàn bộ 20 golden cases."""
    selected = GOLDEN_SET if cases is None else cases
    return [
        run_case(case, generate_fn=generate_fn, no_verifier=no_verifier)
        for case in selected
    ]


def _rate(results: list[dict[str, Any]], key: str, positive: bool = True) -> float | None:
    values = [row.get(key) for row in results if isinstance(row.get(key), bool)]
    if not values:
        return None
    matched = sum(value is positive for value in values)
    return matched / len(values)


def _mean(results: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in results
        if isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
    ]
    return sum(values) / len(values) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp metric, giữ rõ số case chưa có dữ liệu thay vì coi là fail."""
    completed = [row for row in results if row.get("status") == "COMPLETED"]
    blocked = [row for row in results if row.get("status") == "BLOCKED"]
    errors = [row for row in results if row.get("status") == "ERROR"]
    execution_rate = _rate(completed, "execution_pass")
    performance_rate = _rate(completed, "model_performance_pass")
    leakage_rate = _rate(completed, "leakage_detected", positive=True)
    pass_rate = (
        sum(str(row.get("decision")) == "PASS" for row in completed) / len(completed)
        if completed
        else None
    )
    return {
        "total_cases": len(results),
        "completed_cases": len(completed),
        "blocked_cases": len(blocked),
        "error_cases": len(errors),
        "decision_pass_rate": pass_rate,
        "execution_pass_rate": execution_rate,
        "execution_cases_measured": sum(
            isinstance(row.get("execution_pass"), bool) for row in completed
        ),
        "model_performance_pass_rate": performance_rate,
        "model_performance_cases_measured": sum(
            isinstance(row.get("model_performance_pass"), bool) for row in completed
        ),
        "leakage_rate": leakage_rate,
        "leakage_cases_measured": sum(
            isinstance(row.get("leakage_detected"), bool) for row in completed
        ),
        "average_rubric_score": _mean(completed, "average_score"),
        "rules_pass_rate": _rate(completed, "rules_all_passed"),
        "min_asserts_pass_rate": _rate(completed, "min_asserts_pass"),
        "average_module_coverage": _mean(completed, "expected_modules_coverage"),
        "average_skill_coverage": _mean(completed, "expected_skills_coverage"),
        "total_cost_usd": sum(float(row.get("cost_usd") or 0) for row in completed),
        "average_cost_usd": _mean(completed, "cost_usd"),
        "average_runtime_seconds": _mean(completed, "runtime_seconds"),
        "results": results,
    }


def _fmt(value: Any, *, percent: bool = False, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    if percent:
        return f"{float(value) * 100:.1f}%"
    return f"{float(value):.{decimals}f}"


def report_markdown(summary: dict[str, Any]) -> str:
    """Xuất bảng benchmark Markdown."""
    lines = [
        "# NotebookForge Benchmark",
        "",
        "## Coverage",
        "",
        "| Tổng case | Hoàn thành | Bị chặn | Lỗi |",
        "| ---: | ---: | ---: | ---: |",
        (
            f"| {summary['total_cases']} | {summary['completed_cases']} | "
            f"{summary['blocked_cases']} | {summary['error_cases']} |"
        ),
        "",
        "## Metrics",
        "",
        "| Metric | Kết quả | Mục tiêu |",
        "| :--- | ---: | ---: |",
        f"| Execution Pass Rate | {_fmt(summary['execution_pass_rate'], percent=True)} | >= 80% |",
        f"| Model Performance Pass Rate | {_fmt(summary['model_performance_pass_rate'], percent=True)} | >= 80% |",
        f"| Leakage Rate | {_fmt(summary['leakage_rate'], percent=True)} | 0% |",
        f"| Average Rubric Score | {_fmt(summary['average_rubric_score'])} | >= 3.5 |",
        f"| Rule Pass Rate | {_fmt(summary['rules_pass_rate'], percent=True)} | báo cáo |",
        f"| Min-assert Pass Rate | {_fmt(summary['min_asserts_pass_rate'], percent=True)} | báo cáo |",
        f"| Average Cost / Case (USD) | {_fmt(summary['average_cost_usd'], decimals=4)} | <= 0.30 |",
        f"| Total Cost (USD) | {_fmt(summary['total_cost_usd'], decimals=4)} | báo cáo |",
        f"| Average Runtime (s) | {_fmt(summary['average_runtime_seconds'], decimals=2)} | <= 180 |",
        "",
        "## Cases",
        "",
        "| Case | Topic | Status | Decision | Execution | Model metric | Avg score | Runtime (s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | ---: | ---: |",
    ]
    for row in summary.get("results", []):
        metric = "N/A"
        if row.get("metric_value") is not None:
            metric = (
                f"{row.get('metric')}={row['metric_value']:.3f} "
                f"(>={row.get('metric_threshold')})"
            )
        execution = row.get("execution_pass")
        execution_text = "N/A" if execution is None else ("PASS" if execution else "FAIL")
        lines.append(
            "| {id} | {topic} | {status} | {decision} | {execution} | "
            "{metric} | {average} | {runtime} |".format(
                id=row.get("id", "?"),
                topic=row.get("topic", "?"),
                status=row.get("status", "?"),
                decision=row.get("decision") or "N/A",
                execution=execution_text,
                metric=metric,
                average=_fmt(row.get("average_score")),
                runtime=_fmt(row.get("runtime_seconds"), decimals=2),
            )
        )
    issues = [
        row for row in summary.get("results", []) if row.get("error")
    ]
    if issues:
        lines.extend(["", "## Blockers / Errors", ""])
        for row in issues:
            error = str(row["error"]).replace("\n", " ").replace("|", "\\|")
            lines.append(f"- {row.get('id', '?')}: {error}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="chạy 1 case, ví dụ GS-001")
    parser.add_argument(
        "--no-verifier",
        action="store_true",
        help="ablation: tắt Verifier nếu main.generate hỗ trợ flag",
    )
    parser.add_argument("--output", help="đường dẫn file Markdown kết quả")
    args = parser.parse_args()

    cases = (
        [case for case in GOLDEN_SET if case["id"] == args.case]
        if args.case
        else None
    )
    if args.case and not cases:
        parser.error(f"Không có case {args.case}")
    markdown = report_markdown(
        summarize(run_all(cases, no_verifier=args.no_verifier))
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
