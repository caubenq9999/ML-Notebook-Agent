"""Regression tests for Huy's Sprint 2.2 verifier/evaluation work.

Run from ``notebookforge/`` with::

    python -m unittest eval.test_sprint22 -v

The tests use a fake Groq client, so they never spend API credit.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agents.verifier import (
    build_execution_feedback,
    llm_judge,
    render_quality_report,
    rule_checks,
    run_verifier,
    update_retry_history,
)
from eval.golden_set import GOLDEN_SET
from eval.harness import report_markdown, run_case, summarize
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
        _cell("markdown", "## Module 1: Chuẩn bị"),
        _cell("code", "import pandas as pd\nX_train, X_test, y_train, y_test = train_test_split(X, y)"),
        _cell("code", "print(X_train.shape)"),
        _cell("markdown", "## Module 2: Huấn luyện"),
        _cell("code", "model.fit(X_train, y_train)\nprint(model.score(X_test, y_test))"),
        _cell("code", "# TODO: thử đổi max_iter\nassert len(X_train) > 0"),
        _cell("code", "import matplotlib.pyplot as plt\nplt.plot([0, 1], [0, 1])"),
    ]
    return _notebook(cells)


class _FakeCompletions:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps(self.payload, ensure_ascii=False)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeGroq:
    def __init__(self, payload: dict):
        self.chat = SimpleNamespace(completions=_FakeCompletions(payload))


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

    def test_visualization_needs_a_plot_call(self):
        imported = _notebook([_cell("code", "import matplotlib.pyplot as plt")])
        plotted = _notebook([_cell("code", "import matplotlib.pyplot as plt\nplt.scatter(x, y)")])
        self.assertFalse(rule_checks(imported)["has_visualization"])
        self.assertTrue(rule_checks(plotted)["has_visualization"])

    def test_every_module_needs_complete_demo_code(self):
        nb = _passing_notebook()
        self.assertTrue(rule_checks(nb)["has_demo_per_module"])
        nb["cells"][5]["source"] = ["# TODO: train model\n", "raise NotImplementedError\n"]
        nb["cells"][7]["source"] = ["# TODO: add visualization\n"]
        self.assertFalse(rule_checks(nb)["has_demo_per_module"])

    def test_minimum_cells_depends_on_level(self):
        nb = _passing_notebook()  # exactly 8 cells
        self.assertTrue(rule_checks(nb, level=1)["min_cells_by_level"])
        self.assertFalse(rule_checks(nb, level=2)["min_cells_by_level"])


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
            total_cells=8,
            executed_cells=8,
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
        fake = _FakeGroq(
            {
                "executability": 5,
                "groundedness": 4,
                "difficulty_fit": 4,
                "pedagogical_order": 4,
                "feedback": None,
                "ungrounded_claims": [],
            }
        )
        result = llm_judge(
            _passing_notebook(), self.bundle, exc=self.exc_ok, client=fake, level=1
        )
        self.assertEqual(result["executability"], 5)
        self.assertEqual(result["ungrounded_claims"], [])
        call = fake.chat.completions.calls[0]
        self.assertEqual(call["response_format"], {"type": "json_object"})

    def test_run_verifier_merges_execution_rule_and_llm_feedback(self):
        nb = _passing_notebook()
        nb["cells"][6]["source"] = ["# TODO: complete this cell\n"]
        fake = _FakeGroq(
            {
                "executability": 5,
                "groundedness": 4,
                "difficulty_fit": 4,
                "pedagogical_order": 4,
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
            report = run_verifier(str(path), exc, self.bundle, judge_client=fake, level=1)
        self.assertEqual(report.decision, "RETRY")
        self.assertLessEqual(report.llm_scores.executability, 2)
        self.assertIn("NameError", report.feedback or "")
        self.assertIn("Thiếu giải thích metric", report.feedback or "")


class Sprint22ReportTests(unittest.TestCase):
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
                            "executability": 4,
                            "groundedness": 4,
                            "difficulty_fit": 4,
                            "pedagogical_order": 4,
                        },
                    },
                }

            result = run_case(GOLDEN_SET[0], generate_fn=fake_generate)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["execution_pass"])
        self.assertTrue(result["model_performance_pass"])
        self.assertEqual(result["average_score"], 4.0)

    def test_history_is_upserted_and_markdown_is_rendered(self):
        fake = _FakeGroq(
            {
                "executability": 5,
                "groundedness": 4,
                "difficulty_fit": 4,
                "pedagogical_order": 4,
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
            report = run_verifier(str(path), exc, bundle, judge_client=fake, level=1)
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
