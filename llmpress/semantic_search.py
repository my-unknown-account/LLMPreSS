"""User-facing retrieval interface for LLMPreSS.

Load a SemanticIndex once, then reuse it for repeated queries. Search returns
ranked SearchResult objects; serialization and file saving belong to the caller.
Corpus, encoder and FAISS configuration are read from the index directory.

Examples
--------
>>> from llmpress import SemanticIndex
>>> index = SemanticIndex("RedPajama", edition="compact")  # doctest: +SKIP
>>> results = index.search("Hamiltonian paths in cubic graphs", top_k=20)  # doctest: +SKIP
>>> for result in results:  # doctest: +SKIP
...     print(result.passage_id, result.score)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from . import _engine
from ._index_files import _resolve_index_directory
from ._engine import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_URL,
    SemanticSearcher,
    main,
    report_search_errors,
)

__all__ = ["SemanticIndex", "SearchResult", "report_search_errors"]


@dataclass(frozen=True)
class SearchResult:
    """One ranked passage or local index match.

    Attributes
    ----------
    numeric_id : int
        Permanent numeric ID within the index's corpus.
    score : float
        Local similarity score. Higher is better; it is not a probability.
    rank : int
        One-based position in this result list.
    passage_id : str or None
        Original passage identifier, when supplied by the passage API.
    text : str or None
        Passage text. Populated by SemanticIndex.search; None for search_ids.
    corpus : str or None
        Corpus selected by index.json.
    source_rel_path, source_group : str or None
        Source location and group, when supplied by the API.
    source_record_index : int or None
        Zero-based original source line, when available.

    Notes
    -----
    Attributes cannot be reassigned. Use to_dict() for JSON serialization;
    additional API metadata is retained there. Long passage text is omitted
    from the object's repr to keep interactive output readable.
    """

    numeric_id: int
    score: float
    rank: int
    passage_id: str | None = None
    text: str | None = field(default=None, repr=False)
    corpus: str | None = None
    source_rel_path: str | None = field(default=None, repr=False)
    source_group: str | None = field(default=None, repr=False)
    source_record_index: int | None = field(default=None, repr=False)
    _extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, JSON-serializable passage dictionary.

        Unavailable optional fields are omitted. Extra API metadata and local
        approximate scores are preserved. No file is created.
        """
        result = deepcopy(self._extra)
        result.update({
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_") and getattr(self, item.name) is not None
        })
        return result

    @classmethod
    def _from_dict(cls, row: dict[str, Any], rank: int, corpus: str | None) -> SearchResult:
        """Adapt a validated engine result without exposing engine dictionaries."""
        names = {item.name for item in fields(cls) if not item.name.startswith("_")}
        values = {key: value for key, value in row.items() if key in names}
        values.update(rank=rank, corpus=row.get("corpus", corpus))
        extra = {key: deepcopy(value) for key, value in row.items() if key not in names}
        return cls(**values, _extra=extra)


class SemanticIndex:
    """Search a corpus index and retrieve passages using text queries.

    Parameters
    ----------
    corpus : str
        Corpus name, such as "RedPajama". No repository name or URL is needed.
    edition : {"compact", "quality"}, default="compact"
        Index edition to download or validate against a local directory.
    index_dir : str, pathlib.Path or None, default=None
        Directory to load from or download into. None uses
        ~/.cache/llmpress/<corpus>/<edition>. A missing or empty directory is
        populated automatically. Nonempty directories must contain a valid index
        for the requested corpus/edition; otherwise ValueError is raised.
    threads : int, default=32
        CPU threads for FAISS and numerical libraries.
    device : str, default="cpu"
        Encoder device: "cpu", "cuda" or "cuda:N". FAISS runs on CPU.
    verbose : bool, default=True
        Show client progress and status messages. False disables this client's
        progress output; third-party libraries retain their own output settings.
        Nonempty server notices remain visible even when verbose=False.
    api_url : str, default="https://api.jmozafari.com"
        Passage lookup endpoint used by search(). search_ids() makes no API calls.
    timeout : float, default=120
        Timeout in seconds for each HTTP request, including server queue waiting.
    nprobe : int or None, default=None
        IVF search breadth. None uses index.json; larger values search more lists.

    Notes
    -----
    Construction reuses a valid index directory or downloads into a missing/empty
    one, then loads the index and encoder. Hugging Face automatically downloads
    and caches the encoder specified by index.json and shared defaults.
    Construction does not call the passage API.
    Reuse the instance for successive queries. Searches are intended to run
    sequentially on one instance; native thread settings affect the process.
    Invalid configuration raises ValueError. Filesystem, model and HTTP errors
    propagate to callers; report_search_errors() offers concise terminal errors.
    """

    def __init__(
        self,
        corpus: str,
        edition: str = "compact",
        *,
        index_dir: str | Path | None = None,
        threads: int = 32,
        device: str = "cpu",
        verbose: bool = True,
        api_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_API_TIMEOUT,
        nprobe: int | None = None,
    ) -> None:
        # Validate the endpoint before spending time loading the model or index.
        self._api = _engine.PassageAPIClient(api_url, timeout)
        directory = _resolve_index_directory(
            corpus, edition, index_dir, verbose,
        )
        self._searcher = _engine.SemanticSearcher(
            index=directory, threads=threads, device=device,
            nprobe=nprobe, verbose=verbose,
        )
        self._directory = directory
        self._edition = edition

    @property
    def corpus(self) -> str | None:
        """Corpus name read from index.json."""
        return self._searcher.corpus

    @property
    def directory(self) -> Path:
        """Absolute path to the index directory."""
        return self._directory

    @property
    def edition(self) -> str:
        """Requested index edition, verified against index.json."""
        return self._edition

    def __repr__(self) -> str:
        return (f"SemanticIndex({self.corpus!r}, edition={self.edition!r}, "
                f"index_dir={str(self.directory)!r})")

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        candidates: int | None = None,
        refine: bool = True,
    ) -> list[SearchResult]:
        """Return ranked passages with text, identifiers and similarity scores.

        Parameters
        ----------
        query : str
            Nonempty query, within the encoder's token limit. Longer queries are
            rejected rather than silently truncated.
        top_k : int, default=20
            Maximum results, between 1 and min(100000, index vector count).
            Fewer matches may be returned if the probed lists contain fewer IDs.
        candidates : int or None, default=None
            Candidate budget before refinement. None uses
            min(100000, max(800, 4 * top_k)). Explicit values must be at least top_k.
        refine : bool, default=True
            Re-rank by cosine similarity of compressed-vector reconstructions.

        Returns
        -------
        list of SearchResult
            Highest scores first, with one-based ranks. Read result.passage_id,
            result.text and result.score, or call result.to_dict() to serialize.

        Notes
        -----
        API requests use at most 1,000 IDs each and are sent sequentially. Results
        are returned only after all requests succeed. Ranks and scores come from
        local search, not the API. No files are written.
        """
        response = self._searcher.search_passages(
            query, top_k=top_k, api_url=self._api.api_url, timeout=self._api.timeout,
            candidates=candidates, refine=refine,
        )
        return self._results(response["results"])

    def search_ids(
        self,
        query: str,
        top_k: int = 20,
        *,
        candidates: int | None = None,
        refine: bool = True,
    ) -> list[SearchResult]:
        """Return local numeric IDs and scores without contacting the passage API.

        Parameters match search(). Results have numeric_id, score, rank and corpus;
        text and passage_id are None because passage metadata has not been fetched.
        This is useful for offline retrieval experiments and evaluation.
        """
        rows = self._searcher.search(query, top_k=top_k, candidates=candidates, refine=refine)
        return self._results(rows)

    def _results(self, rows: list[dict[str, Any]]) -> list[SearchResult]:
        """Convert ordered engine output into the public result type."""
        return [
            SearchResult._from_dict(row, rank, self.corpus)
            for rank, row in enumerate(rows, start=1)
        ]
