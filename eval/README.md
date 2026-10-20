# eval/

Hand-built evaluation sets (versioned):

- `retrieval_queries.jsonl`: 28 draft lawyer questions for stream C, C3. Review candidates from dense, BM25,
  and hybrid search; fill `relevant_train_indices` with manually judged training-clause `idx` values and change
  `judgment_status` to `reviewed` before reporting manual-query metrics. Pending entries are not judgments;
- `counterfactuals/`: pairs (original clause, perturbed clause) for the audit (stream C, C7);
- `ood/`: out-of-distribution clauses (European style, French) for the audit.
