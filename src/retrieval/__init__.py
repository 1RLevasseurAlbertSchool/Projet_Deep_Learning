"""Similar-clause retrieval (stream C, Brian).

Build a :class:`DenseRetriever` from the train split with :func:`build_dense_index`,
then call ``retriever.search(query, k=5)``. Results contain ``idx``, ``text``,
``label``, ``label_name`` and cosine ``score``.
"""

from src.retrieval.dense import DenseRetriever, build_dense_index

__all__ = ["DenseRetriever", "build_dense_index"]
