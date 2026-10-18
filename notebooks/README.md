# notebooks/

Notebooks are read in order; one notebook = one owner (avoids Git conflicts).
Reusable logic lives in `src/`; notebooks tell the story and call `src/`.

| # | Notebook | Content | Owner | Milestone |
|---|---|---|---|---|
| 00 | `00_colab_setup.ipynb` | Clone, installation, GPU check | Robin | — |
| 01 | `01_eda_baseline.ipynb` | LEDGAR EDA, near-duplicates, TF-IDF + LogReg | Amandine | m1 |
| 02 | `02_retrieval_index.ipynb` | Sentence-transformer index, queries checked by hand | Brian | m2 |
| 03 | `03_lstm_classifier.ipynb` | LSTM / BiLSTM, error analysis | Amandine | m3 |
| 04 | `04_transformer_finetuning.ipynb` | BERT-class fine-tuning, LoRA, paired tests, calibration | Robin | m4 |
| 05 | `05_retrieval_eval.ipynb` | Retrieval evaluation, encoders, hybrid BM25 + dense | Brian | — |
| 06 | `06_bias_audit.ipynb` | Per-slice audit, counterfactuals, out-of-distribution | Brian | audit |
