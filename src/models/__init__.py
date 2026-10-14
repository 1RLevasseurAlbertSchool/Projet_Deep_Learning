"""Modèles de classification de clauses.

Volet B (Amandine) : baseline TF-IDF + régression logistique, LSTM / BiLSTM (packing, GloVe).
Volet A (Robin)    : fine-tuning Hugging Face (full) et LoRA.
Chaque modèle écrit ses runs et prédictions via src.utils.save_run / save_predictions.
"""
