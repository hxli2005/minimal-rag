"""评分和证据可见范围测试；不调用模型，不修改被冻结的RAG。"""
import json
import unittest

from minimal_rag_experiments.hotpot_benchmark import (
    HotpotKnowledgeBase, aggregate, exposure, load_scorer, parse_prediction, score_case,
)


def sample():
    return {"_id": "example", "question": "Where?", "answer": "Paris", "type": "bridge", "level": "hard",
            "context": [["Article", ["x" * 300, " Paris."]]], "supporting_facts": [["Article", 0], ["Article", 1]]}


class BenchmarkTests(unittest.TestCase):
    def test_official_answer_rules(self):
        scorer = load_scorer()
        self.assertTrue(scorer.exact_match_score("The Eiffel Tower!", "eiffel tower"))
        self.assertEqual(scorer.f1_score("yes indeed", "yes"), (0, 0, 0))
        self.assertAlmostEqual(scorer.f1_score("New York City", "New York")[0], 0.8)

    def test_short_answer_parse_is_deterministic(self):
        self.assertEqual(parse_prediction('```json\n{"answer":"Paris","sources":["D0"]}\n```'), ("Paris", ["D0"]))
        with self.assertRaises(ValueError):
            parse_prediction("The answer is Paris.")

    def test_real_passages_only_enter_adapter(self):
        case = sample()
        case["answer"] = "CANARY_GOLD_LABEL_NOT_IN_CONTEXT"
        kb = HotpotKnowledgeBase(case["_id"], case["context"])
        result = kb.read("D0")
        self.assertEqual(result["text"], "x" * 300 + " Paris.")
        self.assertNotIn("CANARY_GOLD", json.dumps(result))
        self.assertTrue(result["source"].startswith("hotpotqa://"))

    def test_title_hit_does_not_count_as_full_sentence_exposure(self):
        case = sample()
        kb = HotpotKnowledgeBase(case["_id"], case["context"])
        events = [{"role": "tool", "name": "search", "content": kb.search("Article")}]
        first = exposure(case, events)
        self.assertEqual(first["gold_document_exposure_recall"], 1)
        self.assertEqual(first["gold_sentence_exposure_recall"], 0)
        events.append({"role": "tool", "name": "read", "content": kb.read("D0", offset=240)})
        last = exposure(case, events)
        self.assertEqual(last["gold_sentence_exposure_recall"], 1)
        self.assertTrue(last["all_gold_sentences_exposed"])

    def test_failures_remain_in_denominator(self):
        case = sample()
        scorer = load_scorer()
        good = score_case(case, {"result": {"status": "finished", "answer": '{"answer":"Paris","sources":[]}'}, "seconds": 2}, scorer)
        bad = score_case(case, {"result": {"status": "runtime_error"}, "seconds": 4}, scorer)
        result = aggregate([good, bad])
        self.assertEqual(result["answer_em"], 0.5)
        self.assertEqual(result["answer_f1"], 0.5)
        self.assertEqual(result["format_or_runtime_failures"], 1)
        self.assertEqual(result["median_seconds"], 3)

    def test_invalid_annotation_is_reported_not_silently_removed(self):
        case = sample()
        case["supporting_facts"].append(["Article", 20])
        result = exposure(case, [])
        self.assertEqual(result["invalid_gold_annotations"], [["Article", 20]])
        self.assertEqual(result["gold_sentence_exposure_recall"], 0)


if __name__ == "__main__":
    unittest.main()
