# Out-of-distribution probes

`clauses.jsonl` contains a small set of manually authored, synthetic examples
in English and French with European legal references. The paired texts are
illustrative paraphrases, not a validated translation set. They have no gold
LEDGAR labels, so use the model's predictions and confidence only; do not report
accuracy or generalize their results to all European or French contracts.

Notebook 06 writes their output to `results/audit/ood_predictions.csv`.
