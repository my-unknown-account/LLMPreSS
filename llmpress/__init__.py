"""LLMPreSS: LLM Pretraining Semantic Search.

Load an index once with SemanticIndex and reuse it for local searches or
passage retrieval. Corpus and encoder settings come from the index directory.
"""

from .semantic_search import SemanticIndex, SearchResult
from ._engine import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_URL,
    DEFAULT_TOP_K,
    PassageAPIClient,
    QueryEncoder,
    SearchClient,
    SemanticSearcher,
    report_search_errors,
)

__all__ = [
    "SemanticIndex",
    "SearchResult",
    "SemanticSearcher",
    "SearchClient",
    "QueryEncoder",
    "PassageAPIClient",
    "report_search_errors",
    "DEFAULT_API_URL",
    "DEFAULT_API_TIMEOUT",
    "DEFAULT_TOP_K",
]
