import unittest

from api_pipeline.reranker_preference import (
    build_pairwise_rows,
    ndcg,
    reranker_candidate,
    reranker_query,
)


QUERY = {
    "query_id": "test:7:vuln:h1",
    "idx": 7,
    "dataset_split": "train",
    "mechanism_claim": "unchecked length",
    "repair_sought": "check remaining bytes",
    "evidence_to_check": "read before check",
}
CANDIDATE = {
    "mechanism_observed": "short input",
    "repair_applied": "add length guard",
    "evidence_decisive": "guard precedes read",
}


class PreferenceDataTests(unittest.TestCase):
    def test_current_reranker_text_contract(self):
        self.assertEqual(
            reranker_query(QUERY),
            "retrieval_summary: unchecked length\n"
            "needed_example: check remaining bytes\n"
            "evidence_hint: read before check",
        )
        self.assertEqual(
            reranker_candidate(CANDIDATE),
            "mechanism_summary: short input\n"
            "solution_summary: add length guard\n"
            "evidence_summary: guard precedes read",
        )

    def test_pairs_are_within_query_and_weighted_by_label_gap(self):
        pools = [{"query": QUERY, "candidates": [
            {"candidate_idx": 10, "candidate_mre": CANDIDATE},
            {"candidate_idx": 11, "candidate_mre": CANDIDATE},
        ]}]
        labels = [
            {"pair_id": "test:7:vuln:h1::10", "status": "success", "reference_value_label": 2},
            {"pair_id": "test:7:vuln:h1::11", "status": "success", "reference_value_label": 0},
        ]
        rows = build_pairwise_rows(pools, labels)
        self.assertEqual(len(rows["train"]), 1)
        self.assertEqual(rows["train"][0]["weight"], 2)
        self.assertEqual(rows["dev"], [])
        self.assertEqual(rows["test"], [])

    def test_repair_record_must_not_cross_splits(self):
        pools = [
            {"query": QUERY, "candidates": []},
            {"query": {**QUERY, "query_id": "test:7:safe:h1", "dataset_split": "test"}, "candidates": []},
        ]
        with self.assertRaisesRegex(ValueError, "appears in both"):
            build_pairwise_rows(pools, [])

    def test_ndcg(self):
        self.assertEqual(ndcg([2, 1, 0], 3), 1.0)
        self.assertEqual(ndcg([0, 0], 3), 0.0)


if __name__ == "__main__":
    unittest.main()
