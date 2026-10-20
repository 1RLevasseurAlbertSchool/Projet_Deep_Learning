# eval/

Hand-built evaluation sets (versioned):

- `retrieval_queries.jsonl`: 28 draft lawyer questions for stream C, C3. Review candidates from dense, BM25,
  and hybrid search; fill `relevant_train_indices` with manually judged training-clause `idx` values and change
  `judgment_status` to `reviewed` before reporting manual-query metrics. Pending entries are not judgments;
- `counterfactuals/`: documentation for pairs generated from actual test clauses by `src/audit/perturb.py`
  (stream C, C7). Review them before interpreting results;
- `ood/clauses.jsonl`: a small, synthetic set of European-style English and French clauses for prediction/confidence
  inspection only; there are no gold labels and no accuracy claim (stream C, C7).
