"""Missing citation data is unavailable, not measured zero recall."""
import json
import pytest

from minimal_rag_experiments.rag_judge import assess, load_answers, summarize


def test_unpublished_source_ids_are_not_scored_as_zero_recall(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"question_id": "q1", "answer": "correct text", "document_ids": [], "citations_available": False}) + "\n")
    answers = load_answers(path)
    (tmp_path / "assessments").mkdir()
    (tmp_path / "assessments/q1.json").write_text(json.dumps({"assessment": {"correct": True, "facts_supported": [True], "reason": "supported"}}))
    cases = [{"id": "q1", "category": "test", "gold_ids": ["source1"]}]
    result = summarize(cases, answers, tmp_path)
    assert result["correct"] == 1
    assert result["document_recall"] is None
    answers["q1"]["citations_available"] = True
    assert summarize(cases, answers, tmp_path)["document_recall"] == 0


@pytest.mark.asyncio
async def test_first_valid_negative_grade_is_never_retried_even_after_resume():
    attempts, responses = [], iter(["invalid JSON", json.dumps({"text": json.dumps({"correct": False, "facts_supported": [False], "reason": "wrong"}), "tool_calls": [], "sources": []})])
    calls = []
    class Model:
        runs = []
        async def __call__(self, messages, tools):
            calls.append(1)
            return next(responses)
    answer, _ = await assess(Model, {"candidate_answer": "x"}, 1, attempts)
    assert answer["correct"] is False
    assert len(calls) == 2 and len(attempts) == 2
    same, _ = await assess(lambda: pytest.fail("valid grade must be reused"), {}, 1, attempts)
    assert same == answer
