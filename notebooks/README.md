# notebooks/

Les notebooks se lisent dans l'ordre ; un notebook = un propriétaire (évite les conflits Git).
La logique réutilisable vit dans `src/`, les notebooks racontent l'histoire et appellent `src/`.

| # | Notebook | Contenu | Propriétaire | Jalon |
|---|---|---|---|---|
| 00 | `00_colab_setup.ipynb` | Clone, installation, vérification GPU | Robin | — |
| 01 | `01_eda_baseline.ipynb` | EDA LEDGAR, quasi-doublons, TF-IDF + LogReg | Amandine | m1 |
| 02 | `02_retrieval_index.ipynb` | Index sentence-transformer, requêtes vérifiées à la main | Brian | m2 |
| 03 | `03_lstm_classifier.ipynb` | LSTM / BiLSTM, analyse d'erreurs | Amandine | m3 |
| 04 | `04_transformer_finetuning.ipynb` | Fine-tuning BERT-class, LoRA, tests appariés, calibration | Robin | m4 |
| 05 | `05_retrieval_eval.ipynb` | Évaluation retrieval, encodeurs, hybride BM25 + dense | Brian | — |
| 06 | `06_bias_audit.ipynb` | Audit par tranche, contrefactuels, hors distribution | Brian | audit |
