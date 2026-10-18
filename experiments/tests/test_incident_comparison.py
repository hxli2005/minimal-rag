"""Evaluation must preserve invalid citations and never reroll a judge result."""
import importlib.util
import json
from pathlib import Path

import pytest

from minimal_rag_experiments.evidence import CallBudget


spec = importlib.util.spec_from_file_location("incident_comparison_test", Path(__file__).resolve().parents[1] / "scripts/compare_incident_workflows.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_unknown_evidence_ids_are_not_silently_dropped_during_scoring():
    report = {"evidence": [{"id": "E1", "document_id": "doc"}, {"id": "E2", "document_id": "doc"}],
              "result": {"answer": "x", "sources": ["E1", "E2", "E999"]}}
    assert comparison.scoring_report(report)["result"]["sources"] == ["doc", "E999"]
    assert report["result"]["sources"] == ["E1", "E2", "E999"]


def test_partial_or_ungraded_cohort_does_not_publish_an_accuracy():
    row = {"status": "finished", "format_or_runtime_error": None, "unseen_citations": [],
           "tokens": {}, "graded": True, "correct": 1., "completeness": 1., "overall": 1.}
    partial = comparison.cohort_metrics([row], 2)
    assert partial["correct"] is None and not partial["quality_metrics_complete"]
    ungraded = comparison.cohort_metrics([row, {**row, "graded": False, "correct": None, "overall": None}], 2)
    assert ungraded["overall"] is None
    complete = comparison.cohort_metrics([row, row], 2)
    assert complete["correct"] == 1 and complete["quality_metrics_complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [True, False])
async def test_judge_has_one_attempt_including_valid_negative_and_malformed_result(tmp_path, monkeypatch, valid):
    calls = []
    class Fake:
        runs = []
        async def __call__(self, messages, tools):
            calls.append(1)
            if not valid:
                return "invalid JSON"
            return json.dumps({"text": json.dumps({"correct": False, "facts_supported": [False], "reason": "missing fact"}),
                               "sources": [], "tool_calls": []})
    monkeypatch.setattr(comparison, "CodexModel", lambda *a, **kw: Fake())
    budget = CallBudget(1)
    result = await comparison.judge_once(None, str(tmp_path), {"question": "q", "gold": "a", "facts": ["a"]}, "wrong", budget, tmp_path)
    assert len(calls) == 1 and budget.used == 1
    assert result["correct"] is False if valid else result is None
    stored = json.loads((tmp_path / "judgment.json").read_text())
    assert stored["status"] == ("graded" if valid else "ungraded")
