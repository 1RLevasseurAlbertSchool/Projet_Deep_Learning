"""Audit de biais et de robustesse (volet C, Brian).

Entrée : fichiers results/predictions/<run_id>_test.parquet (format commun, voir CLAUDE.md).
Sorties : performances par tranche, tests contrefactuels, hors distribution, calibration.
"""
