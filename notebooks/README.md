# notebooks/

Notebooks are read in order; one notebook = one owner (avoids Git conflicts).
Reusable logic lives in `src/`; notebooks tell the story and call `src/`.

| # | Notebook | Content | Owner | Milestone |
|---|---|---|---|---|
| 00 | `00_colab_setup.ipynb` | Clone, installation, GPU check | Robin | — |
| 01 | `01_eda_baseline.ipynb` | LEDGAR EDA, near-duplicates, TF-IDF + LogReg | Amandine | m1 |
| 02 | `02_retrieval_index.ipynb` | Sentence-transformer index, queries checked by hand | Brian | m2 |
| 03 | `03_error_analysis.ipynb` | Error analysis of Legal-BERT vs TF-IDF: per-class ΔF1 vs training frequency, top confusions, truncation at 256 tokens, 20 errors read (LSTM dropped, 25/10) | Robin | m3 |
| 04 | `04_transformer_finetuning.ipynb` | BERT-class fine-tuning, LoRA, paired tests, calibration | Robin | m4 |
| 05 | `05_retrieval_eval.ipynb` | Two dense encoders, BM25 and MiniLM+BM25 RRF on a fixed stratified 5% test sample; no manual lawyer-query evaluation | Brian | — |
| 06 | `06_bias_audit.ipynb` | Classifier slice metrics and counterfactual stress test bounded to 1,000 test clauses | Brian | audit |
