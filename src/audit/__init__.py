"""Bias and robustness audit (stream C, Brian).

Input: results/predictions/<run_id>_test.parquet files (common format, see CLAUDE.md).
Outputs: per-slice performance, counterfactual tests, out-of-distribution tests, calibration.
"""

from src.audit.slices import build_audit_frame, slice_performance
from src.audit.perturb import (
    audit_counterfactuals,
    audit_ood_examples,
    generate_counterfactual_pairs,
)

__all__ = [
    "audit_counterfactuals",
    "audit_ood_examples",
    "build_audit_frame",
    "generate_counterfactual_pairs",
    "slice_performance",
]
