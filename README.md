# HypoRAG

**Hypothesis-Guided Retrieval and Verification for LLM-Based Vulnerability Detection**

HypoRAG analyzes a function through localized, testable vulnerability hypotheses rather than representing the entire function as a single retrieval query. For each hypothesis, it retrieves repair knowledge addressing a relevant failure condition, ranks the cases by verification utility, and asks whether the proposed failure can actually occur in the target code. The final function label follows from the verified points.

This repository contains the current 15-family inference pipeline, a retrieval diagnostic with adjudicated labels, and a supplementary appendix. It supports repair-knowledge construction, indexing, hypothesis generation, retrieval, point verification, and pair-level evaluation. Benchmark source data and model weights are obtained separately.

## Method at a glance

```text
Training repair pairs                    Target function (one side only)
        |                                          |
        v                                          v
Neutral Repair Signature                 Up to three local hypotheses
        |                                + family and M / R / E queries
        +--> Retrieval package                      |
        |    family + M / R / E             Family-constrained dense retrieval
        |    --> three vector indexes             (top-10 per view)
        |                                          |
        +--> Verification guidance          Union and deduplicate by case ID
             four analysis fields                   |
                        |                    Utility-trained reranker
                        +-------------------> top-3 cases per hypothesis
                                                   |
                                            Point-level verification
                                                   |
                                           Deterministic Boolean OR
                                                   |
                                             Function prediction
```

**Offline knowledge construction.** Each vulnerable/fixed training pair first yields a neutral Repair Signature grounded in the observed code change. Two separate generations then produce (1) a retrieval package containing a mechanism family and three aligned views, and (2) a verification guide. The retrieval views are **M** (mechanism observed), **R** (repair applied), and **E** (evidence decisive). The guide contains `case_explanation`, `vulnerable_pattern`, `repaired_pattern`, and `verification_guidance`. Keeping retrieval text and verification guidance separate lets each serve its own task.

**Online hypothesis-guided retrieval.** The detector sees one target function, not its paired repair. The `recall_v2` generator proposes up to three hypotheses, each with a source anchor, a mechanism family, and corresponding M/R/E queries. Each view searches its own dense index within the assigned family; the frozen taxonomy is defined in [`config/taxonomy_catalog.json`](config/taxonomy_catalog.json). The three top-10 lists are merged by knowledge ID. `OTHER` and empty-family cases fall back to the eligible global pool.

**Reranking and verification.** A cross-encoder, trained from teacher-assigned verification-utility preferences, selects the top three cases for each hypothesis. Point verification receives the target function, the hypothesis, and only the four guidance fields of each selected case. It does **not** receive the historical code/diff or M/R/E retrieval text. A point can be `supported`, `unsupported`, or `insufficient_evidence`; the final function label is vulnerable if any point is `supported`, and non-vulnerable otherwise.

## Repository map

| Path | What it contains |
| --- | --- |
| [`api_pipeline/run_formal_hyporag.py`](api_pipeline/run_formal_hyporag.py) | Stage-oriented, resumable experiment from repair pairs to final predictions. |
| [`api_pipeline/`](api_pipeline/) | Current Repair Signature, taxonomy routing, retrieval, hypothesis generation, point verification, and diagnostic helpers. |
| [`api_pipeline/build_preference_pairs.py`](api_pipeline/build_preference_pairs.py) | Converts complete teacher-labeled M/R/E candidate pools into record-disjoint pairwise train/dev/test files. |
| [`api_pipeline/train_reranker.py`](api_pipeline/train_reranker.py) and [`api_pipeline/evaluate_reranker.py`](api_pipeline/evaluate_reranker.py) | Preference-weighted cross-encoder training and ranking evaluation. |
| [`prompts/point_judgment.py`](prompts/point_judgment.py) | Point-verification prompt variants, including the frozen formal prompt. |
| [`config/taxonomy_catalog.json`](config/taxonomy_catalog.json) | Frozen 15-family taxonomy and routing definitions. |
| [`config/excluded_records.json`](config/excluded_records.json) | Seven exceptionally long or unparseable training records excluded from the formal knowledge target. |
| [`retrieval_diagnostic/`](retrieval_diagnostic/) | Fixed queries, candidate IDs, selected pairs, functional summaries, final mechanism labels and rationales, and summary statistics. |
| [`scripts/`](scripts/) | Diagnostic recomputation, release validation, data export, and appendix generation. |
| [`appendix.pdf`](appendix.pdf) | Supplementary implementation, evaluation, and annotation details. |

The release does not bundle PrimeVul source records, generated full-run knowledge, trained reranker weights, or the complete set of online hypotheses. The **141 diagnostic hypotheses and their retrieval/audit records are included** so that the retrieval comparison remains inspectable.

## Data and model prerequisites

- **Benchmark:** [PrimeVul-v0.1](https://github.com/DLVulDet/PrimeVul), with paired training and test JSONL records. The runners expect stable `idx`, `func_vuln`, and `func_safe` fields; project/CVE metadata is needed for retrieval exclusions.
- **Dense encoder:** [BAAI/bge-code-v1](https://huggingface.co/BAAI/bge-code-v1) for code/semantic retrieval. The same local encoder must be used when building and querying the three indexes.
- **Cross-encoder:** [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) is the initialization for preference training. Formal inference expects a trained checkpoint supplied with `--reranker-model-path`; the initialization alone does not reproduce tuned-reranker results.
- **Inference:** an OpenAI-compatible chat endpoint serving the chosen LLM. Specify its base URL and served identifier at run time. The API credential is read from `LLM_API_KEY` by default; do not commit a credential.

Use a Python 3.11+ environment with the packages in [`requirements.txt`](requirements.txt). Install GPU-enabled PyTorch, vLLM, and model packages for the target CUDA environment. Diagnostic recomputation reads released JSONL files and does not call an LLM or run a model.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/recompute_retrieval_diagnostic.py
```

## Run the formal pipeline

The formal runner exposes the stages `prepare`, `signatures`, `retrieval-package`, `guidance-package`, `index`, `s0`, `retrieve`, `s5`, `aggregate`, `validate`, and `summarize`. `run` executes them in order. Knowledge generation and test-side hypothesis generation are independent until retrieval, so a scheduler can run those stages separately. Stage JSONL outputs retain successful tasks; restarting a stage in the same output directory fills missing or failed tasks. Use a **new output directory** when changing the model, taxonomy, prompt, inputs, or decoding configuration; `prepare` rejects a conflicting experiment manifest.

After obtaining the benchmark, encoder, trained reranker, and an inference endpoint:

```bash
export LLM_API_KEY="${LLM_API_KEY:?set LLM_API_KEY in your environment}"

python api_pipeline/run_formal_hyporag.py run \
  --train-path data/raw/primevul_train_merged.jsonl \
  --test-path data/raw/primevul_test_merged.jsonl \
  --output-dir data/formal_hyporag_run \
  --taxonomy-json config/taxonomy_catalog.json \
  --exclusion-manifest config/excluded_records.json \
  --embedding-model-path /path/to/bge-code-v1 \
  --reranker-model-path /path/to/trained-reranker \
  --api-base http://localhost:8000/v1 \
  --model-name YOUR_SERVED_MODEL \
  --s0-prompt-version recall_v2 \
  --dense-top-k 10 \
  --reranker-top-k 3
```

The default taxonomy key is `allocation_state_representation_other_absorbs_lifetime_error_protocol_15`; the default verification prompt is `case-mapped-local-arithmetic`. Set `--chat-template-family template_kwargs` if the server requires thinking parameters through chat-template kwargs, and set `--reasoning-effort` to a supported level. See `python api_pipeline/run_formal_hyporag.py --help` for context-budget, worker, device, and decoding options. A formal run writes an experiment manifest, append-only stage outputs, Chroma indexes, `formal_validation.json`, and `final_results.json` under its output directory. Report completed and excluded records separately: a requested test pair is not automatically a valid evaluated pair.

## Preference reranker

The reranker training objective is unchanged: the cross-encoder learns from teacher-labeled candidate preferences using a weighted pairwise loss. The release includes the pair builder, trainer, and evaluator, but not the teacher labels, training candidate pools, or trained weights. Those inputs must be generated separately for the desired training split; the Table 1 audit labels are not reranker-training labels. Each pool row has a `query` with `query_id`, repair-record `idx`, `dataset_split`, and hypothesis `mechanism_claim`, `repair_sought`, and `evidence_to_check`; each candidate has `candidate_idx` and `candidate_mre` with `mechanism_observed`, `repair_applied`, and `evidence_decisive`. Teacher rows use `pair_id` (`query_id::candidate_idx`), `status=success`, `reference_value_label` in `{0,1,2}`, and `helpfulness_type`. A pool must be fully labeled before pair construction. Assign both sides of a repair record to the same split.

```bash
python api_pipeline/build_preference_pairs.py \
  --candidate-pools-path data/reranker_distill/candidate_pools.jsonl \
  --teacher-labels-path data/reranker_distill/teacher_labels_canonical.jsonl \
  --output-dir data/reranker_distill

python api_pipeline/train_reranker.py \
  --reranker_model_path /path/to/bge-reranker-v2-m3 \
  --output_path models/reranker/tuned/preference-reranker \
  --train_batch_size 4 --gradient_accumulation_steps 8 \
  --epochs 3 --learning_rate 2e-5 --precision bf16

python api_pipeline/evaluate_reranker.py \
  --model_path models/reranker/tuned/preference-reranker \
  --dataset_split test
```

Use the same held-out pools, labels, and `pairwise_test.jsonl` to compare the base and tuned cross-encoders. Training data production and the exact historical tuned checkpoint are not bundled, so these commands document the released training path rather than reproducing the paper's weights from the repository alone. Loading the base cross-encoder in the formal pipeline changes the ranking experiment.

## Retrieval diagnostic (Table 1)

The fixed diagnostic asks whether whole-function resemblance retrieves the *same useful vulnerability mechanism*. Fifty vulnerable functions from the PrimeVul test split form the targets. Random, BM25-Code, Dense-Code, and Dense-LLM each return one training-side vulnerable case per **function**. HypoRAG returns one case per **hypothesis**: 141 hypotheses generated from those same 50 target functions. The diagnostic corpus is fixed by [`candidate_ids.json`](retrieval_diagnostic/candidate_ids.json) at 3,611 training cases. The function-level and hypothesis-level denominators differ and must not be interpreted as a single paired benchmark.

| Retriever | Query unit | Queries | Token Sim. | Target Callee Overlap | Mech. Match |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | Function | 50 | 0.0436 | 0.0000 | 4/50 (8.0%) |
| BM25-Code | Function | 50 | 0.0899 | 0.0494 | 10/50 (20.0%) |
| Dense-Code | Function | 50 | 0.0887 | 0.0516 | 6/50 (12.0%) |
| Dense-LLM | Function | 50 | 0.0846 | 0.0399 | 10/50 (20.0%) |
| Hypothesis-Guided | Hypothesis | 141 | 0.0621 | 0.0235 | 114/141 (80.9%) |

All five retrievers exclude candidates sharing the target's project or CVE, or whose vulnerable function has token Jaccard similarity `>= 0.80` to the query function. BM25-Code and Dense-Code retrieve against vulnerable-function code. Dense-LLM uses generated functional summaries; the released [`semantic_query_summaries.jsonl`](retrieval_diagnostic/semantic_query_summaries.jsonl) preserves 3,611 candidate and 50 query summaries. Hypothesis-Guided uses the hypothesis-level retrieval and tuned reranker, but this diagnostic reports **top-1**, not the top-3 passed to online point verification.

**Measures and annotation.** Token Sim. is lexical-token Jaccard similarity between the target and retrieved vulnerable functions. Target Callee Overlap is the fraction of the target function's callees appearing in the retrieved function. The audit uses `0` for unrelated, `1` for a transferable verification/repair principle, and `2` for the same decisive failed condition or missing defense. Table 1's Mech. Match counts `1` **or** `2`; strict mechanism match counts `2` only. A shared CWE or similar API name alone is not a match. The 200 function-level and 141 hypothesis-level labels in [`mechanism_audit.jsonl`](retrieval_diagnostic/mechanism_audit.jsonl) are **final human-adjudicated labels**: two annotators labeled independently and a third resolved disagreements. The [annotation protocol](retrieval_diagnostic/ANNOTATION_PROTOCOL.md) gives the decision rule and join keys.

Recompute the means and counts from individual records and check coverage and exclusion flags:

```bash
python scripts/recompute_retrieval_diagnostic.py
python scripts/verify_release.py
```

[`retrieval_pairs.jsonl`](retrieval_diagnostic/retrieval_pairs.jsonl) contains the 341 selected query-case pairs and stored similarity scores; `mechanism_audit.jsonl` contains labels, repair evidence, and rationales. [`manifest.json`](retrieval_diagnostic/manifest.json) fixes the target sample and exclusion threshold. `scripts/export_retrieval_audit.py --help` documents how to regenerate a release from local benchmark and run outputs. The release provides final labels, not individual pre-adjudication annotation sheets.

## Evaluation conventions

The formal evaluator keeps the vulnerable and fixed sides separate until both single-function predictions are complete. A valid pair falls into exactly one outcome:

| Outcome | Vulnerable side | Fixed side |
| --- | --- | --- |
| `Both-R` | vulnerable | non-vulnerable |
| `Both-W` | non-vulnerable | vulnerable |
| `Both-s` | non-vulnerable | non-vulnerable |
| `Both-v` | vulnerable | vulnerable |

These counts sum to the number of **valid** pairs, not necessarily all requested pairs. The summary also reports TP/FP/TN/FN, ACC, precision, recall, F1, and stage-level exclusions and token usage. For NPDS, pair outcomes must be interpreted using their valid-pair denominator; see [`appendix.pdf`](appendix.pdf) for metric and implementation details. Any excluded or failed pair should be reported alongside the scores rather than assigned a label after the fact.

## Scope and provenance

This repository is a research artifact, not a prepackaged model checkpoint or scanner service. The frozen taxonomy and exclusion manifest document the formal configuration; the released diagnostic preserves the selected inputs, outputs, and adjudicated mechanism labels needed to audit Table 1. The full PrimeVul records, inference service, encoder, and trained reranker are external prerequisites for a fresh end-to-end run. The 50-function diagnostic was used during method selection and is part of the reported test split; it is not an independent holdout. See the paper and [`appendix.pdf`](appendix.pdf) for the evaluation scope and limitations.
