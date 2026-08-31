"""Regression tests for Huy's Sprint 2.2 verifier/evaluation work.

Run from ``notebookforge/`` with::

    python -m unittest eval.test_sprint22 -v

The tests inject a fake ``llm_client.call_json``, so they never spend API credit.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.verifier import (
    build_execution_feedback,
    llm_judge,
    render_quality_report,
    rule_checks,
    run_verifier,
    update_retry_history,
)
from eval.golden_set import GOLDEN_SET
from eval.harness import report_markdown, run_all, run_case, summarize
from schemas import CellError, ExcRes, ResearchBundle


def _cell(cell_type: str, source: str, **metadata) -> dict:
    cell = {"cell_type": cell_type, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update(execution_count=1, outputs=[])
    cell["metadata"] = metadata
    return cell


def _notebook(cells: list[dict], level: int = 1) -> dict:
    return {
        "cells": cells,
        "metadata": {"notebookforge": {"level": level}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _passing_notebook() -> dict:
    cells = [
        _cell("markdown", "# Notebook"),
        _cell("markdown", "Tổng quan mô hình và mục tiêu học tập."),
        _cell("markdown", "Tổng quan dataset và biến mục tiêu."),
        _cell("markdown", "## Module 1: Chuẩn bị"),
        _cell("markdown", "Giải thích bước chuẩn bị dữ liệu."),
        _cell(
            "code",
            "import pandas as pd\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "print(X_train.shape)",
        ),
        _cell("markdown", "### Exercise 1: Thử thay đổi tham số"),
        _cell("code", "# TODO: thử đổi test_size\nassert len(X_train) > 0"),
        _cell("markdown", "## Module 2: Huấn luyện"),
        _cell("markdown", "Giải thích huấn luyện và đánh giá."),
        _cell(
            "code",
            "model.fit(X_train, y_train)\n"
            "print(model.score(X_test, y_test))\n"
            "plt.plot([0, 1], [0, 1])\n"
            "plt.scatter([0, 1], [0, 1])",
        ),
        _cell("code", "assert len(X_test) > 0"),
    ]
    return _notebook(cells)


class _FakeJudgeCall:
    def __init__(self, payload: dict, cost_usd: float = 0.0123):
        self.payload = payload
        self.cost_usd = cost_usd
        self.calls = []

    def __call__(self, prompt, schema, **kwargs):
        self.calls.append({"prompt": prompt, "schema": schema, **kwargs})
        return schema(**self.payload), SimpleNamespace(cost_usd=self.cost_usd)


class Sprint22RuleTests(unittest.TestCase):
    def test_assert_ignores_comments_and_strings(self):
        nb = _notebook([
            _cell("code", "# assert x == 1\nmessage = 'assert y == 2'"),
        ])
        self.assertFalse(rule_checks(nb)["has_assert"])
        nb["cells"].append(_cell("code", "if ready:\n    assert score > 0.7"))
        self.assertTrue(rule_checks(nb)["has_assert"])

    def test_split_requires_call_but_clustering_and_train_csv_are_valid(self):
        imported_only = _notebook([
            _cell("code", "from sklearn.model_selection import train_test_split"),
        ])
        self.assertFalse(rule_checks(imported_only)["has_train_test_split"])
        self.assertTrue(rule_checks(imported_only, topic="kmeans")["has_train_test_split"])

        reads_train = _notebook([_cell("code", "df = pd.read_csv('../input/train.csv')")])
        self.assertTrue(rule_checks(reads_train)["has_train_test_split"])
        writes_train = _notebook([_cell("code", "df.to_csv('train.csv')")])
        self.assertFalse(rule_checks(writes_train)["has_train_test_split"])

    def test_visualization_needs_two_plot_calls(self):
        imported = _notebook([_cell("code", "import matplotlib.pyplot as plt")])
        one_plot = _notebook([_cell("code", "plt.scatter(x, y)")])
        two_plots = _notebook([_cell("code", "plt.scatter(x, y)\nplt.hist(x)")])
        self.assertFalse(rule_checks(imported)["has_visualization"])
        self.assertFalse(rule_checks(one_plot)["has_visualization"])
        self.assertTrue(rule_checks(two_plots)["has_visualization"])

    def test_instruction_markdown_threshold_depends_on_level(self):
        beginner = _notebook([_cell("markdown", f"Phần {i}") for i in range(8)])
        intermediate = _notebook(
            [_cell("markdown", f"Phần {i}") for i in range(10)], level=2
        )
        self.assertTrue(rule_checks(beginner, level=1)["has_instructions"])
        self.assertFalse(rule_checks(beginner, level=2)["has_instructions"])
        self.assertTrue(rule_checks(intermediate, level=2)["has_instructions"])

    def test_level_three_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "chỉ hỗ trợ level 1"):
            rule_checks(_passing_notebook(), level=3)

    def test_every_module_needs_complete_demo_code(self):
        nb = _passing_notebook()
        self.assertTrue(rule_checks(nb)["has_demo_per_module"])
        nb["cells"][5]["source"] = ["# TODO: train model\n", "raise NotImplementedError\n"]
        nb["cells"][10]["source"] = ["# TODO: add visualization\n"]
        self.assertFalse(rule_checks(nb)["has_demo_per_module"])

    def test_minimum_cells_depends_on_level(self):
        nb = _passing_notebook()  # exactly 12 cells
        self.assertTrue(rule_checks(nb, level=1)["min_cells_by_level"])
        self.assertFalse(rule_checks(nb, level=2)["min_cells_by_level"])
        nb["cells"].extend(
            [_cell("markdown", "Bổ sung lý thuyết 1"), _cell("markdown", "Bổ sung lý thuyết 2")]
            + [_cell("code", f"print({i})") for i in range(4)]
        )
        self.assertEqual(len(nb["cells"]), 18)
        self.assertTrue(rule_checks(nb, level=2)["min_cells_by_level"])


class Sprint22JudgeTests(unittest.TestCase):
    def setUp(self):
        self.bundle = ResearchBundle(
            topic="logistic_regression",
            key_concepts=["sigmoid"],
            unresolved_concepts=[],
        )
        self.exc_ok = ExcRes(
            nb_path="notebook.ipynb",
            attempt=1,
            success=True,
            total_cells=12,
            executed_cells=12,
            duration_seconds=1.0,
        )

    def test_name_error_feedback_is_deterministic(self):
        exc = self.exc_ok.model_copy(
            update={
                "success": False,
                "errors": [
                    CellError(
                        cell_index=7,
                        ename="NameError",
                        evalue="name 'X_train' is not defined",
                    )
                ],
            }
        )
        feedback = build_execution_feedback(exc)
        self.assertIn("[CELL 7] NameError", feedback)
        self.assertIn("define 'X_train'", feedback)
        self.assertNotIn("train_test_split", feedback)

    def test_llm_judge_parses_structured_json(self):
        fake = _FakeJudgeCall(
            {
                "groundedness": 4,
                "difficulty_fit": 4,
                "pedagogical_order": 4,
                "content_completeness": 4,
                "learning_coverage": 5,
                "covered_concepts": ["sigmoid"],
                "shallow_concepts": [],
                "missing_concepts": [],
                "feedback": None,
                "ungrounded_claims": [],
            }
        )
        result = llm_judge(
            _passing_notebook(),
            self.bundle,
            exc=self.exc_ok,
            session_id="test-001",
            judge_call=fake,
            level=1,
        )
        self.assertEqual(result["learning_coverage"], 5)
        self.assertEqual(result["covered_concepts"], ["sigmoid"])
        self.assertEqual(result["ungrounded_claims"], [])
        self.assertEqual(result["judge_cost_usd"], 0.0123)
        call = fake.calls[0]
        self.assertEqual(call["session_id"], "test-001")
        self.assertEqual(call["temperature"], 0)

    def test_production_verifier_prompt_is_rendered(self):
        prompt_dir = Path(__file__).parents[1] / "prompts"
        payload = {
            "groundedness": 4,
            "difficulty_fit": 4,
            "pedagogical_order": 4,
            "content_completeness": 4,
            "learning_coverage": 5,
            "covered_concepts": ["sigmoid"],
            "shallow_concepts": [],
            "missing_concepts": [],
            "feedback": "[CELL 4] Cần giải thích rõ hơn. FIX: thêm giải thích trước code.",
            "ungrounded_claims": [],
        }
        fake = _FakeJudgeCall(payload)
        result = llm_judge(
            _passing_notebook(), self.bundle, exc=self.exc_ok,
            session_id="test-001", judge_call=fake, level=1,
            prompt_path=prompt_dir / "verifier.txt",
        )
        self.assertEqual(result["groundedness"], 4)
        rendered_prompt = fake.calls[0]["prompt"]
        for placeholder in ("{level}", "{research_summary}", "{content}"):
            self.assertNotIn(placeholder, rendered_prompt)
        self.assertIn("[CELL 0 | markdown]", rendered_prompt)
        self.assertTrue((prompt_dir / "verifier_old.txt").is_file())

    def test_production_judge_uses_llm_client_cost_tracker(self):
        payload = {
            "groundedness": 4,
            "difficulty_fit": 4,
            "pedagogical_order": 4,
            "content_completeness": 4,
            "learning_coverage": 5,
            "covered_concepts": ["sigmoid"],
            "shallow_concepts": [],
            "missing_concepts": [],
            "feedback": None,
            "ungrounded_claims": [],
        }
        calls = []

        class FakeTracker:
            def mark(self):
                calls.append("mark")
                return 0.2

            def cost_since(self, mark):
                calls.append(("cost_since", mark))
                return 0.0456

        def fake_call_json(prompt, schema, **kwargs):
            calls.append(("call_json", kwargs))
            return schema(**payload), SimpleNamespace(cost_usd=0.01)

        fake_module = SimpleNamespace(
            MODEL_JUDGE="judge-model",
            call_json=fake_call_json,
            get_tracker=lambda session_id: FakeTracker(),
        )
        with patch.dict(sys.modules, {"llm_client": fake_module}):
            result = llm_judge(
                _passing_notebook(),
                self.bundle,
                exc=self.exc_ok,
                session_id="test-001",
                level=1,
            )

        self.assertEqual(result["judge_cost_usd"], 0.0456)
        self.assertIn("mark", calls)
        call = next(item for item in calls if isinstance(item, tuple) and item[0] == "call_json")
        self.assertEqual(call[1]["session_id"], "test-001")
        self.assertEqual(call[1]["model"], "judge-model")
        self.assertIn(("cost_since", 0.2), calls)

    def test_run_verifier_merges_execution_rule_and_llm_feedback(self):
        nb = _passing_notebook()
        nb["cells"][6]["source"] = ["# TODO: complete this cell\n"]
        fake = _FakeJudgeCall(
            {
                "groundedness": 4,
                "difficulty_fit": 4,
                "pedagogical_order": 4,
                "content_completeness": 4,
                "learning_coverage": 5,
                "covered_concepts": ["sigmoid"],
                "shallow_concepts": [],
                "missing_concepts": [],
                "feedback": "[CELL 5] Thiếu giải thích metric. FIX: thêm giải thích accuracy.",
                "ungrounded_claims": [],
            }
        )
        exc = self.exc_ok.model_copy(
            update={
                "success": False,
                "errors": [
                    CellError(
                        cell_index=5,
                        ename="NameError",
                        evalue="name 'model' is not defined",
                    )
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempt1.ipynb"
            path.write_text(json.dumps(nb), encoding="utf-8")
            report = run_verifier(
                str(path),
                exc,
                self.bundle,
                session_id="test-001",
                judge_call=fake,
                level=1,
            )
        self.assertEqual(report.decision, "RETRY")
        self.assertEqual(report.llm_scores.learning_coverage, 5)
        self.assertEqual(exc.cost_this_attempt, 0.0123)
        self.assertIn("NameError", report.feedback or "")
        self.assertIn("Thiếu giải thích metric", report.feedback or "")


class Sprint22ReportTests(unittest.TestCase):
    def test_harness_checkpoint_resumes_completed_case(self):
        calls = []

        def should_not_run(profile):
            calls.append(profile.session_id)
            raise AssertionError("completed case must be resumed")

        saved = [{"id": "GS-001", "status": "COMPLETED", "decision": "PASS"}]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            checkpoint.write_text(json.dumps(saved), encoding="utf-8")
            results = run_all(
                [GOLDEN_SET[0]],
                generate_fn=should_not_run,
                checkpoint_path=checkpoint,
                resume=True,
            )
        self.assertEqual(results, saved)
        self.assertEqual(calls, [])

    def test_harness_runs_one_case_with_injected_pipeline(self):
        nb = _passing_notebook()
        nb["metadata"]["notebookforge"]["metrics"] = {"accuracy": 0.85}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "executed.ipynb"
            path.write_text(json.dumps(nb), encoding="utf-8")

            def fake_generate(profile):
                self.assertEqual(profile.topic, "logistic_regression")
                return {
                    "notebook_path": str(path),
                    "decision": "PASS",
                    "total_cost_usd": 0.1,
                    "report": {
                        "nb_path": str(path),
                        "decision": "PASS",
                        "average_score": 4.0,
                        "llm_scores": {
                            "groundedness": 4,
                            "difficulty_fit": 4,
                            "pedagogical_order": 4,
                            "content_completeness": 4,
                            "learning_coverage": 4,
                        },
                    },
                }

            result = run_case(GOLDEN_SET[0], generate_fn=fake_generate)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["execution_pass"])
        self.assertTrue(result["model_performance_pass"])
        self.assertEqual(result["average_score"], 4.0)

    def test_history_is_upserted_and_markdown_is_rendered(self):
        fake = _FakeJudgeCall(
            {
                "groundedness": 4,
                "difficulty_fit": 4,
                "pedagogical_order": 4,
                "content_completeness": 4,
                "learning_coverage": 5,
                "covered_concepts": ["sigmoid"],
                "shallow_concepts": [],
                "missing_concepts": [],
                "feedback": None,
                "ungrounded_claims": [],
            }
        )
        bundle = ResearchBundle(topic="logistic_regression", key_concepts=["sigmoid"])
        exc = ExcRes(
            nb_path="attempt1.ipynb",
            attempt=1,
            success=True,
            total_cells=8,
            executed_cells=8,
            duration_seconds=1.2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempt1.ipynb"
            path.write_text(json.dumps(_passing_notebook()), encoding="utf-8")
            report = run_verifier(
                str(path),
                exc,
                bundle,
                session_id="test-001",
                judge_call=fake,
                level=1,
            )
        history = update_retry_history([], report, exc)
        history = update_retry_history(history, report, exc)
        self.assertEqual(len(history), 1)
        markdown = render_quality_report(history)
        self.assertIn("| 1 |", markdown)
        self.assertIn("PASS", markdown)

    def test_harness_summary_handles_partial_benchmark(self):
        summary = summarize(
            [
                {
                    "id": "GS-001",
                    "status": "COMPLETED",
                    "execution_pass": True,
                    "model_performance_pass": True,
                    "leakage_detected": False,
                    "average_score": 4.0,
                    "cost_usd": 0.1,
                    "runtime_seconds": 10.0,
                    "decision": "PASS",
                },
                {"id": "GS-002", "status": "BLOCKED", "error": "main.generate missing"},
            ]
        )
        self.assertEqual(summary["completed_cases"], 1)
        self.assertEqual(summary["blocked_cases"], 1)
        self.assertEqual(summary["execution_pass_rate"], 1.0)
        self.assertIn("Benchmark", report_markdown(summary))


if __name__ == "__main__":
    unittest.main()
