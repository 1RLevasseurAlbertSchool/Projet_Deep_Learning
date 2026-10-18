"""Bias and robustness audit (stream C, Brian).

Input: results/predictions/<run_id>_test.parquet files (common format, see CLAUDE.md).
Outputs: per-slice performance, counterfactual tests, out-of-distribution tests, calibration.
"""
