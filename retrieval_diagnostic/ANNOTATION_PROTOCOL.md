# Mechanism-match audit protocol

Each row compares one target query with one top-1 historical repair case. The
query is either a complete vulnerable function (Random, BM25, Dense-Code,
Dense-LLM) or a localized vulnerability hypothesis (Hypothesis-Guided). The
annotator inspects the proposed failure condition, the historical failure,
and the vulnerable-to-fixed repair diff. Shared CWE or surface vocabulary
alone does not establish a match.

- `0`: no concrete shared failure condition or transferable verification rule.
- `1`: a concrete verification or repair principle transfers to the query,
  although the immediate failed condition differs.
- `2`: the decisive failed safety condition or missing defense is the same.

The paper's `Mech. Match` counts labels `1` and `2`. Strict mechanism match
counts only label `2`. The release stores final human-adjudicated labels and
rationales for all 200 function-level and 141 hypothesis-level pairs. Two
annotators independently applied this protocol, and a third adjudicated
disagreements.

Join `mechanism_audit.jsonl` and `retrieval_pairs.jsonl` by `(method,
query_idx, hypothesis_id, candidate_idx)`, using a null `hypothesis_id` for
function-level rows. The retrieval file contains both vulnerable function
bodies; the audit file contains the hypothesis when applicable, repair diff,
metadata, label, and explanation. Run
`python scripts/recompute_retrieval_diagnostic.py` from the repository root
to reproduce the displayed counts and mean scores.
