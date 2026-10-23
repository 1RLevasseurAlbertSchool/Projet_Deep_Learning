# eval/

Hand-built evaluation sets (versioned):

- `retrieval_queries.jsonl`: 28 draft lawyer questions for stream C, C3. They are retained as illustrative future
  queries only; Notebook 05 does not retrieve or score them because no domain-expert relevance judgments are
  available. Do not report manual-query metrics from these pending entries;
- `counterfactuals/`: documentation for pairs generated from actual test clauses by `src/audit/perturb.py`
  (stream C, C7). Review them before interpreting results;
- `ood/clauses.jsonl`: retained exploratory data, not used by the focused final audit; it has no gold labels and
  supports no accuracy or generalization claim.
