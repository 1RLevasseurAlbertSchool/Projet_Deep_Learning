"""Similar-clause retrieval (stream C, Brian).

Build a :class:`DenseRetriever` from the train split with :func:`build_dense_index`,
then call ``retriever.search(query, k=5)``. Results contain ``idx``, ``text``,
``label``, ``label_name`` and cosine ``score``.
"""

from src.retrieval.dense import DenseRetriever, build_dense_index
from src.retrieval.hybrid import BM25Retriever, HybridRetriever

__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "build_dense_index"]
