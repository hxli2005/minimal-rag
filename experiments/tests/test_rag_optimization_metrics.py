import pytest
from minimal_rag_experiments.rag_optimization_eval import aggregate, score, validate_evaluation_scope


def test_completed_answer_does_not_hide_recovered_tool_and_protocol_errors():
    case = {"id": "fixture", "category": "basic", "gold_ids": []}
    report = {"native_answer": True, "result": {"status": "finished", "answer": "A", "sources": []},
              "events": [{"role": "tool", "name": "search", "content": {"error": "InterfaceError"}},
                         {"role": "protocol_error"}], "judgement": {"correct": True, "facts_supported": [True]}}
    row = score(case, report, "enterprise", None)
    metrics = aggregate([row])
    assert metrics["errors"] == 0
    assert metrics["correct"] == 1.0
    assert metrics["tool_errors"] == metrics["cases_with_tool_errors"] == metrics["protocol_errors"] == 1


def test_new_observability_fields_can_aggregate_historical_case_records():
    case = {"id": "fixture", "category": "basic", "gold_ids": []}
    row = score(case, {"native_answer": True, "result": {"status": "finished", "answer": "A", "sources": []}}, "enterprise", None)
    del row["tool_errors"], row["protocol_errors"]
    metrics = aggregate([row])
    assert metrics["tool_errors"] == metrics["cases_with_tool_errors"] == metrics["protocol_errors"] == 0


def test_known_regression_requires_the_full_test_cohort_and_cannot_be_a_holdout():
    validate_evaluation_scope("test", False, True, None, None)
    with pytest.raises(ValueError):
        validate_evaluation_scope("test", True, True, None, None)
    with pytest.raises(ValueError):
        validate_evaluation_scope("test", False, True, 20, None)
    with pytest.raises(ValueError):
        validate_evaluation_scope("dev", False, True, None, None)
