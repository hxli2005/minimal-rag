import json
import unittest

from minimal_rag_experiments.hotpot_architecture_ablation import paired_stats, prepare_context, score_baseline
from minimal_rag_experiments.hotpot_benchmark import HotpotKnowledgeBase, load_scorer


def case():
    return {"_id": "sample", "question": "Where is needle?", "answer": "GOLD_CANARY",
            "type": "bridge", "level": "hard", "supporting_facts": [["doc6", 0]],
            "context": [[f"doc{i}", [f"needle entry {i}. " + "x" * 300]] for i in range(7)]}


class ArchitectureTests(unittest.TestCase):
    def test_full_context_keeps_all_candidates_and_complete_text(self):
        sample = case()
        messages, events, ids = prepare_context(sample, "full_context")
        self.assertEqual(ids, [f"D{i}" for i in range(7)])
        self.assertNotIn("GOLD_CANARY", json.dumps(messages))
        self.assertEqual([e["content"]["text"] for e in events], ["".join(s) for _, s in sample["context"]])
        self.assertTrue(all(e["origin"] == "fixed_pipeline" for e in events))

    def test_top5_uses_original_question_and_same_bm25_without_gold_filter(self):
        sample = case()
        expected = HotpotKnowledgeBase(sample["_id"], sample["context"]).search(sample["question"], limit=5)
        messages, events, ids = prepare_context(sample, "bm25_top5")
        self.assertEqual(ids, [r["document_id"] for r in expected["hits"]])
        self.assertEqual(len(ids), 5)
        self.assertEqual(json.loads(events[0]["arguments_json"])["query"], sample["question"])
        self.assertNotIn("GOLD_CANARY", json.dumps(messages))
        self.assertTrue(all(len(e["content"]["text"]) > 240 for e in events if e["name"] == "read"))

    def test_pipeline_io_not_counted_as_model_tool_calls(self):
        sample = case()
        _, events, ids = prepare_context(sample, "full_context")
        report = {"result": {"status": "finished", "answer": '{"answer":"unknown","sources":[]}'},
                  "events": events, "supplied_documents": ids, "supplied_characters": 2200}
        row = score_baseline(sample, report, load_scorer())
        self.assertEqual(row["tool_calls"], 0)
        self.assertEqual(row["pipeline_operations"], 7)
        self.assertEqual(row["gold_sentence_exposure_recall"], 1)

    def test_paired_stats_match_ids_not_row_order(self):
        left = [{"id": "a", "em": 1, "f1": 1}, {"id": "b", "em": 0, "f1": 0.5}]
        right = [{"id": "b", "em": 1, "f1": 1}, {"id": "a", "em": 0, "f1": 0}]
        result = paired_stats(left, right, bootstrap_samples=100)
        self.assertEqual(result["em_delta"], 0)
        self.assertEqual(result["f1_delta"], 0.25)
        self.assertEqual(result["candidate_only_exact"], 1)
        self.assertEqual(result["reference_only_exact"], 1)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1)

    def test_identical_arms_have_zero_interval(self):
        rows = [{"id": "a", "em": 1, "f1": 1}, {"id": "b", "em": 0, "f1": 0}]
        result = paired_stats(rows, rows, bootstrap_samples=50)
        self.assertEqual(result["paired_bootstrap_95_intervals"]["f1"], [0, 0])
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1)
        with self.assertRaises(ValueError):
            paired_stats(rows, rows[:1], bootstrap_samples=50)


if __name__ == "__main__":
    unittest.main()
