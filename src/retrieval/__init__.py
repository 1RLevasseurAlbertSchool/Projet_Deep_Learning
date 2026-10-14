"""Recherche de clauses similaires (volet C, Brian).

API attendue : search(query: str, k: int = 5) -> list[dict] avec idx, text, label, label_name, score.
Banque = split train ; embeddings normalisés (cosinus = produit scalaire), mis en cache dans data/indexes/.
"""
