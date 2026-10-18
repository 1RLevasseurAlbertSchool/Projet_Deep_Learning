"""Similar-clause retrieval (stream C, Brian).

Expected API: search(query: str, k: int = 5) -> list[dict] with idx, text, label, label_name, score.
Bank = train split; normalised embeddings (cosine = dot product), cached in data/indexes/.
"""
