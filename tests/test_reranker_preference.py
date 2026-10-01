import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from api_pipeline import build_preference_pairs
from api_pipeline import build_reranker_distill, run_formal_hyporag
from api_pipeline.build_reranker_distill import select_pool_candidates, selection_score
from api_pipeline.reranker_preference import (
    build_pairwise_rows,
    ndcg,
    parse_teacher_output,
    reranker_candidate,
    reranker_query,
)
from prompts.reranker_preference_labeling import SYSTEM_PROMPT, build_messages


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

    def test_teacher_uses_six_mre_fields_not_family(self):
        messages = build_messages(
            {**QUERY, "mechanism_family": "FAMILY_NOT_SENT"},
            {**CANDIDATE, "mechanism_family": "CANDIDATE_FAMILY_NOT_SENT"},
        )
        self.assertEqual(
            hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
            "13c80f3ddbbf",
        )
        self.assertIn("unchecked length", messages[1]["content"])
        self.assertIn("add length guard", messages[1]["content"])
        self.assertNotIn("FAMILY_NOT_SENT", messages[1]["content"])
        self.assertNotIn("CANDIDATE_FAMILY_NOT_SENT", messages[1]["content"])
        self.assertEqual(
            parse_teacher_output('{"reference_value_label":2,"helpfulness_type":"confirm","reason":"Comparable length guard."}')["reference_value_label"],
            2,
        )

    def test_pool_has_retrieved_source_and_cross_family(self):
        query = {**QUERY, "mechanism_family": "CAPACITY_INDEX_EXTENT"}
        knowledge = {
            idx: {"mre": {**CANDIDATE, "mechanism_family": family}}
            for idx, family in [
                (7, "CAPACITY_INDEX_EXTENT"), (8, "CAPACITY_INDEX_EXTENT"),
                (9, "SIZE_OFFSET_ARITHMETIC"),
            ]
        }
        hits = {8: [{"route": "mechanism", "rank": 1}]}
        candidates = select_pool_candidates(query, hits, knowledge, [8, 9], 42, (1, 0, 0), 1)
        by_idx = {row["candidate_idx"]: row for row in candidates}
        self.assertEqual(by_idx[8]["candidate_source"], "retrieval")
        self.assertEqual(by_idx[7]["candidate_source"], "same_source")
        self.assertEqual(by_idx[9]["candidate_source"], "cross_family")
        self.assertEqual(selection_score(query["query_id"], 42), selection_score(query["query_id"], 42))

    def test_teacher_request_scores_text_without_family(self):
        candidate = {
            "candidate_idx": 10, "candidate_mre": {**CANDIDATE, "mechanism_family": "HIDDEN_FAMILY"},
            "candidate_source": "retrieval", "same_source": False,
        }
        args = SimpleNamespace(
            api_base="http://localhost:8000/v1", api_key="unit-test", model_name="test-model",
            temperature=0.0, top_p=1.0, max_new_tokens=8192, timeout=30,
            reasoning_effort="high", thinking_type="disabled", chat_template_family="default",
            retries=1,
        )
        client = unittest.mock.MagicMock()
        client.with_options.return_value = client
        response = '{"reference_value_label":2,"helpfulness_type":"confirm","reason":"Comparable length guard."}'
        with patch.object(build_reranker_distill, "build_client", return_value=client), \
             patch.object(build_reranker_distill, "call_chat_completion_with_usage", return_value=(response, {"prompt_tokens": 12})) as request:
            row = build_reranker_distill.label_one(
                {**QUERY, "mechanism_family": "QUERY_FAMILY"}, candidate, args, "frozen",
            )
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["reference_value_label"], 2)
        self.assertEqual(request.call_args.kwargs["thinking_type"], "disabled")
        self.assertNotIn("QUERY_FAMILY", request.call_args.kwargs["messages"][1]["content"])
        self.assertNotIn("HIDDEN_FAMILY", request.call_args.kwargs["messages"][1]["content"])

    def test_manifest_filters_to_complete_selected_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pools = [
                {"query": QUERY, "candidates": [
                    {"candidate_idx": 10, "candidate_mre": CANDIDATE},
                    {"candidate_idx": 11, "candidate_mre": CANDIDATE},
                ]},
                {"query": {**QUERY, "query_id": "train:8:safe:h1", "idx": 8}, "candidates": []},
            ]
            labels = [
                {"pair_id": f"{QUERY['query_id']}::{idx}", "query_id": QUERY["query_id"],
                 "status": "success", "reference_value_label": value,
                 "prompt_hash": "frozen"}
                for idx, value in ((10, 2), (11, 0))
            ]
            manifest = [{"query_id": QUERY["query_id"], "status": "complete", "prompt_hash": "frozen"}]
            for name, rows in (("pools.jsonl", pools), ("labels.jsonl", labels), ("manifest.jsonl", manifest)):
                (root / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
                )
            argv = ["build_preference_pairs.py", "--candidate-pools-path", str(root / "pools.jsonl"),
                    "--teacher-labels-path", str(root / "labels.jsonl"),
                    "--query-group-manifest-path", str(root / "manifest.jsonl"),
                    "--output-dir", str(root / "out")]
            with patch("sys.argv", argv):
                build_preference_pairs.main()
            self.assertEqual(len(build_preference_pairs.read_jsonl(root / "out" / "pairwise_train.jsonl")), 1)
            self.assertEqual(len(build_preference_pairs.read_jsonl(root / "out" / "teacher_labels_canonical.jsonl")), 2)

    def test_current_taxonomy_pool_to_teacher_to_pairs_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal_dir = root / "formal"
            output = formal_dir / "reranker_distill"
            output.mkdir(parents=True)
            records = [
                {"idx": idx, "project": f"project{idx}", "func_vuln": f"bad{idx}", "func_safe": f"good{idx}"}
                for idx in (1, 2)
            ]
            formal_manifest = {
                "taxonomy": {"name": run_formal_hyporag.DEFAULT_TAXONOMY},
                "train": {"records_sha256": run_formal_hyporag.stable_hash(records)},
            }
            run_formal_hyporag.write_json(formal_dir / "experiment_manifest.json", formal_manifest)
            encoder = root / "encoder"
            run_formal_hyporag.write_json(formal_dir / "knowledge" / "index_manifest.json", {
                "embedding_model_path": str(encoder), "knowledge_fingerprint": "unit-knowledge",
            })
            point = {**QUERY, "id": 1, "mechanism_family": "CAPACITY_INDEX_EXTENT"}
            s0_rows = [
                {"key": f"train-s0:{idx}:{side}", "idx": idx, "side": side,
                 "status": "success" if (idx, side) == (1, "vuln") else "excluded",
                 "input_hash": f"hash-{idx}-{side}",
                 "parsed": {"hypotheses": [point]} if (idx, side) == (1, "vuln") else None}
                for idx in (1, 2) for side in ("vuln", "safe")
            ]
            run_formal_hyporag.write_jsonl(output / "s0_train_hypotheses.jsonl", s0_rows)
            knowledge = {
                idx: {"idx": idx, "record": records[idx - 1], "mre": {
                    **CANDIDATE, "mechanism_family": "CAPACITY_INDEX_EXTENT",
                }}
                for idx in (1, 2)
            }
            build_args = SimpleNamespace(
                formal_output_dir=formal_dir, output_dir=output, train_path=root / "train.jsonl",
                exclusion_manifest=root / "excluded.json", embedding_model_path=encoder,
                embedding_max_length=1024, embedding_batch_size=4, retrieval_topk=20,
                rank_1_5=4, rank_6_10=3, rank_11_20=3,
                cross_family_negatives=2, clone_threshold=0.8,
                seed=42, split_seed=42, device="cpu",
            )
            vectors = {route: np.asarray([[1.0, 0.0], [0.5, 0.5]]) for route in ("mechanism", "repair", "evidence")}
            with patch.object(run_formal_hyporag, "selected_train", return_value=records), \
                 patch.object(run_formal_hyporag, "knowledge_units", return_value=knowledge), \
                 patch.object(run_formal_hyporag, "load_index_vectors", return_value=vectors), \
                 patch.object(run_formal_hyporag, "load_embedding_model", return_value=(object(), object())), \
                 patch.object(run_formal_hyporag, "encode_all", side_effect=lambda _t, _m, texts, _a, _d: np.asarray([[1.0, 0.0]] * len(texts))), \
                 patch.object(build_reranker_distill, "eligible_candidates", return_value=[1]):
                build_reranker_distill.build_pools(build_args)
                build_reranker_distill.build_pools(build_args)
            pools = build_reranker_distill.read_jsonl(output / "candidate_pools.jsonl")
            self.assertEqual(len(pools), 1)
            self.assertEqual(len(pools[0]["candidates"]), 2)

            build_reranker_distill.select_groups(SimpleNamespace(
                output_dir=output, selection_seed=42, selection_ratio=1.0,
            ))

            calls_by_candidate = {}

            def fake_label(query, candidate, _args, prompt_hash):
                idx = int(candidate["candidate_idx"])
                calls_by_candidate[idx] = calls_by_candidate.get(idx, 0) + 1
                return {
                    "pair_id": f"{query['query_id']}::{idx}", "query_id": query["query_id"],
                    "candidate_idx": idx,
                    "status": "failed" if idx == 2 and calls_by_candidate[idx] == 1 else "success",
                    "prompt_hash": prompt_hash,
                    "reference_value_label": 2 if idx == 1 else 0,
                    "helpfulness_type": "confirm" if idx == 1 else "none",
                }

            label_args = SimpleNamespace(
                output_dir=output, api_key="unit-test", max_query_groups=-1, max_workers=2,
            )
            with patch.object(build_reranker_distill, "label_one", side_effect=fake_label) as called:
                build_reranker_distill.label_groups(label_args)
                self.assertEqual(
                    build_reranker_distill.read_jsonl(output / "query_group_manifest.jsonl")[0]["status"],
                    "partial_failed",
                )
                build_reranker_distill.label_groups(label_args)
                build_reranker_distill.label_groups(label_args)
                self.assertEqual(called.call_count, 3)
            self.assertEqual(
                build_reranker_distill.read_jsonl(output / "query_group_manifest.jsonl")[0]["status"],
                "complete",
            )
            argv = ["build_preference_pairs.py", "--candidate-pools-path", str(output / "candidate_pools.jsonl"),
                    "--teacher-labels-path", str(output / "teacher_labels.jsonl"),
                    "--query-group-manifest-path", str(output / "query_group_manifest.jsonl"),
                    "--output-dir", str(output)]
            with patch("sys.argv", argv):
                build_preference_pairs.main()
            pair_rows = build_preference_pairs.read_jsonl(output / f"pairwise_{pools[0]['query']['dataset_split']}.jsonl")
            self.assertEqual(len(pair_rows), 1)


if __name__ == "__main__":
    unittest.main()
