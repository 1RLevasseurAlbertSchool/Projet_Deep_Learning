# Counterfactual audit inputs

Counterfactual pairs are generated from actual LEDGAR evaluation clauses by
`src.audit.perturb.generate_counterfactual_pairs()`. This avoids presenting
hand-written examples as if they were sampled from the dataset.

Only clauses containing a target phrase are transformed. Each generated pair
changes one feature at a time: a named party-role pair, `he`/`she`, New York
jurisdiction, or USD currency notation. The transformations are heuristic
stress tests, not legal redrafting; review generated text before interpreting
the flip rates.

The output pair table is written to `results/audit/counterfactual_pairs.csv`
when notebook 06 is run.
