#!/usr/bin/env python3
"""Search local FAISS indexes and retrieve their passages from an HTTP API.

Use :class:`SemanticSearcher` for text queries. It loads the index and encoder
once and can serve repeated searches. Corpus selection comes from index.json;
the same Python files work with different corpora produced by this pipeline.

Required files
--------------
index.faiss
    FAISS index containing the OPQ transform and IVF/PQ configuration.
lists.ivfdata
    On-disk inverted-list data. Keep it beside index.faiss when moving files.
index.json
    Corpus name, vector count, file sizes, index parameters and optional encoder
    overrides. Omitted encoder settings use the shared defaults in this module.

Notes
-----
The default encoder is BGE-M3 at a pinned Hugging Face revision. Model files
are downloaded into the Hugging Face cache on first use. FAISS runs on the CPU;
query encoding can use a supported CUDA device when requested.

Search methods return Python dictionaries and lists that json.dumps can
serialize. Terminal prompts and saving belong to the caller; example_search.py
shows that workflow. The module also provides an offline command-line interface
that prints numeric IDs and scores without retrieving passage text.

Examples
--------
Load an index directory and retrieve passages::

    from llmpress import SemanticSearcher

    searcher = SemanticSearcher(index="/path/to/index")
    result = searcher.search_passages("What is semantic search?", top_k=10)

Reuse the loaded index and model for another query::

    result = searcher.search_passages("How does vector search work?", top_k=5)

Search without contacting the passage API::

    matches = searcher.search("What is semantic search?", top_k=10)

See Also
--------
SemanticSearcher : Main interface for text queries.
SearchClient : Lower-level searches with precomputed query vectors.
QueryEncoder : Query-vector generation.
PassageAPIClient : Passage lookup by numeric ID.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
import sys
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import faiss
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

DEFAULT_API_URL = "https://api.jmozafari.com"
DEFAULT_API_TIMEOUT = 120.0
DEFAULT_TOP_K = 1000
_HTTP_USER_AGENT = "LLM-Search-Client/1.0"
_MIN_CANDIDATES = 800
_MAX_CANDIDATES = 100000


# Encoder defaults shared by indexes produced by this pipeline.
# An index.json encoder section can override individual settings.
_ENCODER_DEFAULTS = {
    "candidate_revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "dimension": 1024,
    "faiss_input": "float32 renormalized",
    "inference_dtype": "bfloat16",
    "max_model_tokens": 8192,
    "model": "BAAI/bge-m3",
    "normalization_dtype": "bfloat16",
    "normalize": True,
    "pooling": "CLS",
    "prompt": None,
    "storage_dtype": "float16",
}


# Index metadata and encoder configuration.


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON metadata file.

    Parameters
    ----------
    path : str or pathlib.Path
        Metadata file to read.

    Returns
    -------
    dict
        Parsed metadata; its schema is validated by the caller.

    Raises
    ------
    OSError
        If the file cannot be read.
    json.JSONDecodeError
        If its contents are not valid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_file(root: str | Path, name: str, expected: dict[str, Any]) -> Path:
    """Validate a metadata-declared file and return its resolved path.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing the index files.
    name : str
        Relative filename declared by the index metadata.
    expected : dict
        File description with a nonnegative integer ``bytes`` entry.

    Returns
    -------
    pathlib.Path
        Existing file inside root whose size matches the declaration.

    Raises
    ------
    ValueError
        If the declaration is invalid, the path leaves root, or the file is
        missing or has a different size.
    OSError
        If filesystem metadata cannot be read.
    """
    if (
        not isinstance(expected, dict)
        or type(expected.get("bytes")) is not int
        or expected["bytes"] < 0
    ):
        raise ValueError(f"Invalid file metadata: {name}")
    path = (Path(root) / name).resolve()
    if Path(name).is_absolute() or Path(root).resolve() not in path.parents:
        raise ValueError(f"Unsafe release filename: {name}")
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Missing/truncated release file: {name}")
    return path


def _encoder_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """Combine shared encoder defaults with per-index overrides.

    Parameters
    ----------
    config : dict or None
        Optional settings overriding the shared defaults.

    Returns
    -------
    dict
        New dictionary of validated, fully resolved encoder settings.

    Raises
    ------
    ValueError
        If overrides are not a dictionary or resolved settings are unsupported.
    """
    if config is not None and not isinstance(config, dict):
        raise ValueError("Encoder configuration must be an object")
    settings = {**_ENCODER_DEFAULTS, **(config or {})}
    _check_encoder_config(settings)
    return settings


def _check_encoder_config(config: dict[str, Any]) -> None:
    """Validate the encoder operations supported by this client.

    Parameters
    ----------
    config : dict
        Fully resolved encoder settings.

    Raises
    ------
    ValueError
        If a required value is missing, invalid or unsupported.
    """
    if not isinstance(config, dict):
        raise ValueError("Missing encoder configuration in index.json")
    for key in ("model", "candidate_revision"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing encoder setting: {key}")
    for key in ("dimension", "max_model_tokens"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid encoder setting: {key}")
    for key, expected in {
        "pooling": "CLS",
        "prompt": None,
        "normalize": True,
        "faiss_input": "float32 renormalized",
    }.items():
        if config.get(key) != expected:
            raise ValueError(f"Unsupported encoder setting: {key}")
    for key in ("inference_dtype", "normalization_dtype", "storage_dtype"):
        if config.get(key) not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"Unsupported encoder setting: {key}")


def _validate_metadata(manifest: dict[str, Any]) -> None:
    """Validate index metadata and resolve its encoder settings in place.

    Parameters
    ----------
    manifest : dict
        Parsed metadata. Its encoder entry is replaced with resolved settings.

    Raises
    ------
    ValueError
        If metadata is malformed, unsupported or internally inconsistent.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Index metadata must be a JSON object")
    if manifest.get("version") != 1 or not isinstance(manifest.get("edition"), str):
        raise ValueError("Unsupported index metadata")
    manifest["encoder"] = _encoder_settings(manifest.get("encoder"))
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Missing index parameters")
    for key in ("dimension", "pq_bytes", "nlist", "nprobe"):
        if type(parameters.get(key)) is not int or parameters[key] < 1:
            raise ValueError(f"Invalid index parameter: {key}")
    if manifest["encoder"]["dimension"] != parameters["dimension"]:
        raise ValueError("Encoder/index dimension mismatch")
    spec = manifest.get("index_spec", {})
    expected_spec = {
        "metric": "inner_product",
        "pq_nbits": 8,
        "direct_map": "NoMap",
        "storage": "ondisk_ivf",
        "opq_dimension": parameters["dimension"],
    }
    if spec != expected_spec:
        raise ValueError("Unsupported metric/OPQ/storage settings")
    if type(manifest.get("rows")) is not int or manifest["rows"] < 1:
        raise ValueError("Invalid vector count")
    identity = manifest.get("catalog_identity")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("Missing permanent ID namespace")

    if not isinstance(manifest.get("files"), dict):
        raise ValueError("Missing index file declarations")
    corpus = manifest.get("corpus")
    if corpus is not None and (not isinstance(corpus, str) or not corpus.strip()):
        raise ValueError("Corpus must be a nonempty string")


def load_manifest(directory: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read and validate the metadata belonging to an index directory.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing index.json, index.faiss and lists.ivfdata. A path
        to index.faiss is also accepted and selects its containing directory.

    Returns
    -------
    directory : pathlib.Path
        Directory containing the index and metadata.
    manifest : dict
        Validated metadata, including the corpus when declared, index parameters,
        vector count and encoder settings merged with the shared defaults.

    Raises
    ------
    ValueError
        If metadata is missing, unsupported or inconsistent, or a declared file
        is missing or has the wrong size.
    OSError
        If a metadata file or index-file information cannot be read.

    Notes
    -----
    This function checks metadata and file sizes without opening the FAISS index
    or loading a model. Older packages using release.json instead of index.json
    remain supported. Legacy encoder.json declarations are checked when present.
    """
    directory = Path(directory)
    if directory.is_file():
        if directory.name != "index.faiss":
            raise ValueError("Use an edition directory or index.faiss")
        directory = directory.parent

    metadata_path = directory / "index.json"
    if not metadata_path.is_file():
        metadata_path = directory / "release.json"
    if not metadata_path.is_file():
        raise ValueError("Missing index.json; generate it with the release packager")
    manifest = _read_json(metadata_path)
    _validate_metadata(manifest)
    for filename in ("index.faiss", "lists.ivfdata"):
        if filename not in manifest["files"]:
            raise ValueError(f"Missing file declaration: {filename}")
        checked_file(directory, filename, manifest["files"][filename])
    if "encoder.json" in manifest["files"]:
        checked_file(directory, "encoder.json", manifest["files"]["encoder.json"])
        config = _read_json(directory / "encoder.json")
        if any(
            config.get(key) != manifest["encoder"].get(key)
            for key in config
            if key in manifest["encoder"]
        ):
            raise ValueError("Incompatible encoder configuration")
    return directory, manifest


# Native output and CPU thread limits.


@contextmanager
def _diagnostic_stdout() -> Iterator[None]:
    """Redirect Python and native stdout diagnostics to stderr temporarily.

    Yields
    ------
    None
        Execute loading/search operations while reserving stdout for CLI results.

    Notes
    -----
    The process stdout file descriptor is restored on exit. This context is intended
    for the single-threaded command-line entry point, not concurrent library calls.
    """
    import ctypes

    def flush():
        sys.stdout.flush()
        try:
            ctypes.CDLL(None).fflush(None)
        except (AttributeError, OSError):
            pass

    flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield
        flush()
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def _limit_threads(threads: int) -> threadpool_limits:
    """Apply a positive thread budget to native numerical libraries.

    Parameters
    ----------
    threads : int
        Maximum native BLAS/OpenMP threads.

    Returns
    -------
    threadpoolctl.threadpool_limits
        Limit controller. Use as a context manager for temporary limits or retain
        it while a loaded index/encoder needs the limits.

    Raises
    ------
    ValueError
        If threads is not a positive integer.
    """
    if type(threads) is not int or threads < 1:
        raise ValueError("Positive native thread count required")
    return threadpool_limits(limits=threads)


# FAISS index validation and vector retrieval.


def _normalize_vectors(vectors: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
    """Normalize a finite matrix of near-unit vectors without changing its input.

    Parameters
    ----------
    vectors : numpy.ndarray
        Two-dimensional query-vector matrix.
    tolerance : float, default=0.02
        Absolute tolerance for the near-unit norm check; NumPy also applies its
        default relative tolerance.

    Returns
    -------
    numpy.ndarray
        Float32 matrix with unit-length rows.

    Raises
    ------
    ValueError
        If input is not a finite matrix or contains zero or non-unit vectors.
    """
    values = np.array(vectors, dtype="float32", order="C", copy=True)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite matrix")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0) or not np.allclose(norms, 1, atol=tolerance):
        raise ValueError("Vectors are not normalized")
    return values / norms[:, None]


def _load_index(path: str | Path) -> faiss.Index:
    """Open a read-only FAISS index using adjacent inverted-list data.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to index.faiss.

    Returns
    -------
    faiss.Index
        Index with on-disk IVF paths relocated beside index.faiss.

    Raises
    ------
    RuntimeError
        If FAISS cannot read the index or its inverted lists.
    """
    return faiss.read_index(
        str(path),
        faiss.IO_FLAG_ONDISK_SAME_DIR | faiss.IO_FLAG_READ_ONLY,
    )


def _select_profile(
    index: faiss.Index, config: dict[str, Any], nprobe: int | None = None
) -> dict[str, Any]:
    """Validate FAISS structure against declared parameters and select nprobe.

    Parameters
    ----------
    index : faiss.Index
        Loaded full-dimensional OPQ/IVF-PQ index.
    config : dict
        Declared dimension, PQ code bytes, IVF-list count and default nprobe.
    nprobe : int or None, default=None
        Optional replacement search probe count.

    Returns
    -------
    dict
        Copy of config containing the effective nprobe value.

    Raises
    ------
    ValueError
        If structure or the effective nprobe is incompatible.
    """
    config = dict(config)
    ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    if (
        index.d != config["dimension"]
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
        or ivf.nlist != config["nlist"]
        or not hasattr(ivf, "pq")
        or ivf.pq.nbits != 8
        or ivf.code_size != config["pq_bytes"]
        or ivf.pq.M != config["pq_bytes"]
    ):
        raise ValueError("Index does not match its declared parameters")
    if not isinstance(index, faiss.IndexPreTransform) or index.chain.size() != 1:
        raise ValueError("Expected one full-dimensional OPQ transform")
    transform = faiss.downcast_VectorTransform(index.chain.at(0))
    if (
        not isinstance(transform, faiss.LinearTransform)
        or transform.d_in != config["dimension"]
        or transform.d_out != config["dimension"]
    ):
        raise ValueError("Expected full-dimensional OPQ")
    if ivf.direct_map.type != faiss.DirectMap.NoMap:
        raise ValueError("Expected no global direct map")
    if nprobe is not None:
        config["nprobe"] = nprobe
    if not 1 <= config["nprobe"] <= ivf.nlist:
        raise ValueError("Invalid nprobe override")
    return config


def _retrieve(
    index: faiss.Index,
    vector: np.ndarray,
    top_k: int,
    nprobe: int,
    candidates: int = 800,
    refine: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve vector matches and optionally refine their PQ scores.

    Parameters
    ----------
    index : faiss.Index
        Inner-product IVF index without a global direct map.
    vector : numpy.ndarray
        Single near-unit query row with the index dimension.
    top_k : int
        Requested result count.
    nprobe : int
        Number of IVF lists to probe.
    candidates : int, default=800
        Approximate candidates considered when refinement is enabled.
    refine : bool, default=True
        Whether to rank normalized PQ reconstructions instead of raw scores.

    Returns
    -------
    results : list of dict
        Descending-score matches; numeric ID breaks ties.
    timing : dict
        Search/refinement durations and candidates_returned.

    Raises
    ------
    ValueError
        If limits, query vectors, reconstructed vectors or scores are invalid.
    RuntimeError
        If FAISS search or reconstruction fails.
    """
    if type(top_k) is not int or not 1 <= top_k <= index.ntotal:
        raise ValueError("top-k must be between 1 and the number of indexed vectors")
    if type(candidates) is not int or not top_k <= candidates <= _MAX_CANDIDATES:
        raise ValueError("candidates must be >= top-k and <= 100000")
    ivf = faiss.extract_index_ivf(index)
    if index.metric_type != faiss.METRIC_INNER_PRODUCT or not 1 <= nprobe <= ivf.nlist:
        raise ValueError("Expected inner-product index and valid nprobe")
    if ivf.direct_map.type != faiss.DirectMap.NoMap:
        raise ValueError("Refusing an index with a global direct map")
    query_vector = _normalize_vectors(vector)
    if query_vector.shape != (1, index.d):
        raise ValueError("Expected one query with index dimension")
    ivf.nprobe = nprobe
    search_started = time.perf_counter()
    if refine:
        distances, labels, reconstructed = index.search_and_reconstruct(
            query_vector,
            min(candidates, index.ntotal),
        )
    else:
        distances, labels = index.search(query_vector, top_k)
    search_seconds = time.perf_counter() - search_started
    refinement_started = time.perf_counter()
    valid = labels[0] >= 0
    numeric_ids = labels[0, valid]
    approximate_scores = distances[0, valid]
    if not np.isfinite(approximate_scores).all() or len(np.unique(numeric_ids)) != len(
        numeric_ids,
    ):
        raise ValueError("Invalid search results")
    if refine:
        reconstructed_vectors = reconstructed[0, valid]
        norms = np.linalg.norm(reconstructed_vectors, axis=1)
        if (
            not np.isfinite(reconstructed_vectors).all()
            or not np.isfinite(norms).all()
            or np.any(norms <= 0)
        ):
            raise ValueError("Invalid PQ reconstruction")
        scores = reconstructed_vectors @ query_vector[0] / norms
    else:
        scores = approximate_scores
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite similarity scores")
    order = np.lexsort((numeric_ids, -scores))[:top_k]
    results = [
        {
            "numeric_id": int(numeric_ids[position]),
            "score": float(scores[position]),
            "approximate_inner_product": float(approximate_scores[position]),
        }
        for position in order
    ]
    return (
        results,
        {
            "search_and_reconstruction_seconds": search_seconds,
            "refinement_seconds": time.perf_counter() - refinement_started,
            "candidates_returned": len(numeric_ids),
        },
    )


def _terminal_color() -> bool:
    """Use ANSI colors only on interactive terminals that allow them."""
    return sys.stderr.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def _styled(text: str, code: str) -> str:
    """Style terminal text while leaving redirected logs readable."""
    return f"\033[{code}m{text}\033[0m" if _terminal_color() else text


def _write_status(message: str, label: str = "INFO") -> None:
    """Write a labeled status line without interfering with active bars."""
    colors = {"INFO": "1;36", "WAIT": "1;33", "DONE": "1;32", "ERROR": "1;31",
              "WARNING": "1;33", "CRITICAL": "1;31"}
    prefix = _styled(f"[{label:5}]", colors[label])
    tqdm.write(f"{prefix} {message}", file=sys.stderr)


@contextmanager
def _report_operation(description: str, verbose: bool) -> Iterator[None]:
    """Report blocking work with a completion bar and elapsed time.

    Calls expose no fractional progress. Failures leave the bar incomplete and
    retain the original exception. Color is limited to interactive terminals.
    """
    started = time.perf_counter()
    if verbose:
        _write_status(description, "WAIT")
    try:
        with tqdm(
            total=1, desc=_styled(description, "1;36"), unit="operation",
            disable=not verbose, file=sys.stderr, leave=False, dynamic_ncols=True,
            colour="cyan" if _terminal_color() else None,
            bar_format="{desc} |{bar}| {percentage:3.0f}% [{elapsed}]",
        ) as progress:
            yield
            progress.update(1)
    except Exception:
        if verbose:
            _write_status(f"{description}: failed after {time.perf_counter() - started:.2f}s", "ERROR")
        raise
    else:
        if verbose:
            _write_status(f"{description}: completed in {time.perf_counter() - started:.2f}s", "DONE")


class SearchClient:
    """Open an on-disk FAISS index for repeated query-vector searches.

    Parameters
    ----------
    directory : str or pathlib.Path
        Index directory, or the path to its index.faiss file.
    threads : int, default=4
        Positive CPU thread budget for FAISS and native BLAS/OpenMP pools.
    nprobe : int or None, default=None
        Number of IVF lists to probe per search. None uses index.json. An explicit
        value must be between 1 and the index's number of IVF lists, inclusive.

    Attributes
    ----------
    directory : pathlib.Path
        Directory containing the loaded index.
    manifest : dict
        Validated metadata with resolved encoder settings.
    index : faiss.Index
        Read-only FAISS index referencing the adjacent on-disk inverted lists.
    config : dict
        Validated FAISS parameters with the effective nprobe value.
    open_seconds : float
        Time spent loading metadata, opening and validating the index, in seconds.

    Raises
    ------
    ValueError
        If metadata, thread settings or the FAISS structure is incompatible.
    OSError
        If metadata or index files cannot be accessed.
    RuntimeError
        If FAISS cannot open the index or its inverted lists.

    Notes
    -----
    Use SemanticSearcher for text queries. This class accepts already encoded
    vectors and does not load a tokenizer, download a model or call the API.
    Native thread limits apply within the process, not just to this instance.
    """

    def __init__(
        self,
        directory: str | Path,
        threads: int = 4,
        nprobe: int | None = None,
    ) -> None:
        """Load metadata and open the FAISS index once.

        See the class documentation for constructor parameters and exceptions.
        """
        self.threads = threads
        self._thread_limits = _limit_threads(threads)
        faiss.omp_set_num_threads(threads)
        started = time.perf_counter()
        self.directory, self.manifest = load_manifest(directory)
        self.index = _load_index(self.directory / "index.faiss")
        self.config = _select_profile(
            self.index,
            self.manifest["parameters"],
            nprobe,
        )
        if self.index.ntotal != self.manifest["rows"] or not self.index.is_trained:
            raise ValueError("Index/manifest vector count mismatch")
        ivf = faiss.downcast_index(faiss.extract_index_ivf(self.index))
        if not isinstance(
            faiss.downcast_InvertedLists(ivf.invlists),
            faiss.OnDiskInvertedLists,
        ):
            raise ValueError("Expected memory-mapped on-disk IVF data")
        self.open_seconds = time.perf_counter() - started

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 20,
        candidates: int = 800,
        refine: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search a normalized query vector and rank the matching numeric IDs.

        Parameters
        ----------
        vector : numpy.ndarray, shape (1, dimension)
            Finite, near-unit query vector with the index's dimension. A float32 copy
            is normalized before searching; the input array is not modified.
        top_k : int, default=20
            Maximum number of results, between 1 and the indexed vector count.
        candidates : int, default=800
            Number of approximate matches considered when refine is True. Must be
            at least top_k and at most 100000; retrieval is capped at the vector count.
        refine : bool, default=True
            Re-rank candidates by cosine similarity of their PQ reconstructions.
            False returns direct approximate inner-product matches.

        Returns
        -------
        results : list of dict
            Descending-score results containing numeric_id, score and
            approximate_inner_product. Tied scores are ordered by ascending numeric_id.
            Fewer than top_k results may be returned when probed lists contain fewer
            valid matches.
        timing : dict
            Search/reconstruction and refinement durations in seconds, plus the
            number of candidates returned by FAISS.

        Raises
        ------
        ValueError
            If the vector, result limits, reconstructions or returned numeric IDs
            are invalid.
        RuntimeError
            If FAISS fails to search or reconstruct candidates.

        Notes
        -----
        Refined scores use compressed PQ reconstructions, not original embeddings.
        They are similarity measures, not relevance probabilities.
        """
        with _limit_threads(self.threads):
            rows, timing = _retrieve(
                self.index,
                vector,
                top_k,
                self.config["nprobe"],
                candidates,
                refine,
            )
        if any(row["numeric_id"] >= self.manifest["rows"] for row in rows):
            raise ValueError("Out-of-range permanent numeric ID")
        return rows, timing


class QueryEncoder:
    """Encode text using the model and numerical settings of an index.

    Parameters
    ----------
    checkpoint : str, pathlib.Path or None, default=None
        Optional local model directory. When supplied, tokenizer and weights are
        loaded locally without downloading. Otherwise Hugging Face downloads or
        reuses the model and candidate_revision selected by the resolved settings.
    threads : int, default=4
        Positive CPU thread budget for Torch and native numerical libraries.
    device : str, default="cpu"
        Query-encoding device: "cpu", "cuda" or "cuda:N". CUDA must be available
        and support BF16 when the declared inference precision requires it.
    identity : dict or None, default=None
        Encoder overrides, normally read from index.json. None uses shared defaults.
        Settings select model, candidate_revision, dimension, max_model_tokens,
        inference_dtype, normalization_dtype and storage_dtype. The implemented
        encoding contract requires CLS pooling, normalization and no query prompt.

    Attributes
    ----------
    identity : dict
        Validated encoder settings after defaults and overrides are combined.
    threads : int
        Effective CPU thread budget.

    Raises
    ------
    ValueError
        If settings, device support or model dimensions are incompatible, or Torch
        inter-operation threads cannot be configured in the current process.
    OSError
        If model or tokenizer files cannot be downloaded or loaded.
    RuntimeError
        If model initialization fails on the requested device.

    Notes
    -----
    For normal use, SemanticSearcher constructs this encoder from index metadata.
    The shared defaults use BGE-M3, BF16 inference/normalization and FP16 storage
    rounding before conversion to normalized float32 FAISS input.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        threads: int = 4,
        device: str = "cpu",
        identity: dict[str, Any] | None = None,
    ) -> None:
        """Resolve encoder settings and load the tokenizer and model once.

        See the class documentation for constructor parameters and exceptions.
        """
        if threads < 1 or (device != "cpu" and not device.startswith("cuda")):
            raise ValueError("Use positive threads and cpu or cuda[:N]")
        identity = _encoder_settings(identity)
        self.identity = dict(identity)
        model_source = checkpoint if checkpoint is not None else identity["model"]
        download_options = (
            {"local_files_only": True}
            if checkpoint is not None
            else {"revision": identity["candidate_revision"]}
        )
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError as error:
                raise ValueError(
                    "Start a fresh search process to set Torch inter-op threads to one",
                ) from error
        self.threads = threads
        self._thread_limits = _limit_threads(threads)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self._torch = torch
        self._device = torch.device(device)
        if device.startswith("cuda"):
            with torch.cuda.device(self._device):
                if (
                    identity["inference_dtype"] == "bfloat16"
                    and not torch.cuda.is_bf16_supported()
                ):
                    raise ValueError(
                        "This checkpoint configuration requires BF16; use CPU on this "
                        "GPU",
                    )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            use_fast=True,
            **download_options,
        )
        self._model = (
            AutoModel.from_pretrained(
                model_source,
                use_safetensors=False,
                dtype=getattr(torch, identity["inference_dtype"]),
                **download_options,
            )
            .to(self._device)
            .eval()
        )
        if self._model.config.hidden_size != identity["dimension"]:
            raise ValueError("Model dimension differs from index.json")

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        """Encode one query without silently truncating its tokens.

        Parameters
        ----------
        text : str
            Nonempty query text. Its token count must not exceed max_model_tokens
            in the resolved encoder settings.

        Returns
        -------
        vector : numpy.ndarray, shape (1, dimension)
            Contiguous, unit-normalized float32 vector suitable for the FAISS index.
        input_tokens : int
            Token count produced by the tokenizer, including its special tokens.

        Raises
        ------
        ValueError
            If text is empty, exceeds the token limit, or yields an invalid vector.
        RuntimeError
            If model inference fails.
        """
        with _limit_threads(self.threads):
            return self._encode(text)

    def _encode(self, text: str) -> tuple[np.ndarray, int]:
        """Encode and normalize one query using the loaded tokenizer and model.

        Parameters
        ----------
        text : str
            Nonempty text within the configured token limit.

        Returns
        -------
        vector : numpy.ndarray
            One unit-normalized float32 vector.
        input_tokens : int
            Number of tokenizer input tokens.

        Raises
        ------
        ValueError
            If text, token count or the resulting vector is invalid.
        RuntimeError
            If model inference fails.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query must be nonempty text")
        inputs = self._tokenizer(text, return_tensors="pt", truncation=False)
        tokens = inputs["input_ids"].shape[1]
        if tokens > self.identity["max_model_tokens"]:
            raise ValueError(
                f"Query exceeds {self.identity['max_model_tokens']} tokens; refusing truncation"
            )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            hidden = self._model(**inputs).last_hidden_state[:, 0, :]
            vector = self._torch.nn.functional.normalize(
                hidden.to(getattr(self._torch, self.identity["normalization_dtype"])),
                p=2,
                dim=1,
            )
            vector = (
                vector.to(getattr(self._torch, self.identity["storage_dtype"]))
                .to(self._torch.float32)
                .cpu()
                .numpy()
            )
        if vector.shape != (1, self.identity["dimension"]):
            raise ValueError("Unexpected query embedding shape")
        return _normalize_vectors(vector), int(tokens)


class PassageAPIClient:
    """Retrieve passage text and provenance for ordered numeric IDs.

    Parameters
    ----------
    api_url : str, default="https://api.jmozafari.com"
        Full HTTP or HTTPS endpoint accepting a POST request for passage lookup.
    timeout : float, default=120.0
        Positive, finite timeout in seconds, applied separately to each HTTP request.
    batch_size : int, default=1000
        Maximum IDs per request, between 1 and 1000. This does not limit the
        total number of passages fetched; larger results use multiple requests.

    Attributes
    ----------
    api_url : str
        Passage endpoint.
    timeout : float
        Per-request timeout in seconds.
    batch_size : int
        Maximum IDs sent in a single request.

    Raises
    ------
    ValueError
        If the endpoint, timeout or batch size is invalid.

    Notes
    -----
    Each response must match the requested corpus, ID order and passage count.
    Catalog identity is checked when supplied. SemanticSearcher attaches local
    scores and ranks after all passage batches succeed.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_API_TIMEOUT,
        batch_size: int = 1000,
    ) -> None:
        """Validate HTTP connection settings without making a request.

        See the class documentation for constructor parameters and exceptions.
        """
        endpoint = urlsplit(api_url)
        if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
            raise ValueError("API URL must be an absolute HTTP or HTTPS URL")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("API timeout must be positive and finite")
        if type(batch_size) is not int or not 1 <= batch_size <= 1000:
            raise ValueError("API batch size must be an integer between 1 and 1000")
        self.batch_size = batch_size
        self.api_url = api_url
        self.timeout = timeout
        self._seen_messages: set[tuple[str, str]] = set()

    def fetch(
        self,
        numeric_ids: list[int],
        corpus: str,
        catalog_identity: str | None = None,
        *,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch passages in batches while preserving ID order and duplicates.

        Parameters
        ----------
        numeric_ids : list of int
            Nonnegative passage IDs in the desired output order. An empty list returns
            an empty list without contacting the server.
        corpus : str
            Name of the API corpus containing these numeric IDs.
        catalog_identity : str or None, default=None
            Expected numeric-ID namespace, usually read from index.json. When supplied,
            it is sent with each request and must match the response.
        verbose : bool, default=False
            Show passage progress and batch timings on stderr. The bar advances
            only after each HTTP response has been received and validated.

        Returns
        -------
        list of dict
            Passage records in exactly the requested order. Records include numeric_id,
            nonempty text and the provenance fields provided by the API. row_index and
            catalog_identity are omitted, including from nested metadata.

        Raises
        ------
        ValueError
            If an ID is invalid, the API rejects request validation (HTTP 422),
            or a response has incompatible corpus/identity, mismatched IDs,
            an incorrect passage count or missing text. HTTP 422 errors include
            field details, such as numeric_ids exceeding the 1000-item limit.
        urllib.error.HTTPError
            If the server returns an HTTP error other than request validation
            (HTTP 422), which is reported as ValueError.
        urllib.error.URLError
            If the endpoint cannot be reached or a connection fails.
        OSError
            If an HTTP operation fails, including a socket timeout.

        Notes
        -----
        No partial list is returned when a later batch fails. This method retrieves
        metadata and text; SemanticSearcher supplies the authoritative rank and score.
        Nonempty server notices are printed on stderr even when verbose=False.
        Identical notices are shown once per fetch rather than once per batch.
        """
        if any(type(number) is not int or number < 0 for number in numeric_ids):
            raise ValueError("Numeric IDs must be nonnegative integers")
        self._seen_messages.clear()
        passages = []
        batches = math.ceil(len(numeric_ids) / self.batch_size)
        with tqdm(
            total=len(numeric_ids), desc=_styled("Fetching passages", "1;37"),
            unit="passage", disable=not verbose, file=sys.stderr, dynamic_ncols=True,
            colour="white" if _terminal_color() else None,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as progress:
            for batch, start in enumerate(range(0, len(numeric_ids), self.batch_size), 1):
                ids = numeric_ids[start : start + self.batch_size]
                started = time.perf_counter()
                if verbose:
                    _write_status(
                        f"API batch {batch}/{batches}: requesting {len(ids)} passages "
                        f"from {self.api_url} (timeout {self.timeout:g}s)",
                        "WAIT",
                    )
                passages.extend(self._fetch_batch(ids, corpus, catalog_identity))
                progress.update(len(ids))
                if verbose:
                    _write_status(
                        f"API batch {batch}/{batches}: received {len(ids)} passages "
                        f"in {time.perf_counter() - started:.2f}s",
                        "DONE",
                    )
        return passages

    def _show_message(self, payload: Any) -> None:
        """Display each distinct server notice once per fetch, even in quiet mode."""
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict):
            return
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        text = text.strip()
        kind = message.get("type", "info")
        if kind not in ("info", "warning", "critical"):
            kind = "info"
        notice = (kind, text)
        if notice not in self._seen_messages:
            self._seen_messages.add(notice)
            _write_status(f"Server: {text}", kind.upper())

    def _fetch_batch(
        self,
        numeric_ids: list[int],
        corpus: str,
        catalog_identity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Send one passage lookup request and validate its response.

        Parameters
        ----------
        numeric_ids : list of int
            IDs within a single request batch.
        corpus : str
            Requested API corpus.
        catalog_identity : str or None, default=None
            Optional expected numeric-ID namespace.

        Returns
        -------
        list of dict
            Validated passage records in request order.

        Raises
        ------
        ValueError
            If the API rejects request validation (HTTP 422), or response data
            fails validation. Request errors include the server's field messages.
        urllib.error.URLError
            If an HTTP request fails; HTTPError is a subclass.
        """
        body = {"corpus": corpus, "numeric_ids": numeric_ids}
        if catalog_identity is not None:
            body["catalog_identity"] = catalog_identity
        request = Request(
            self.api_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": _HTTP_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as error:
            with error:
                raw = error.read(65536)
            body = raw.decode("utf-8", errors="replace")
            try:
                validation = json.loads(body)
            except ValueError:
                validation = {}
            self._show_message(validation)
            if error.code != 422:
                # Keep the body readable for callers after inspecting its notice.
                raise HTTPError(error.url, error.code, error.reason,
                                error.headers, BytesIO(raw)) from error
            detail = validation.get("detail") if isinstance(validation, dict) else None
            if isinstance(detail, list):
                messages = []
                for issue in detail:
                    if isinstance(issue, dict):
                        field = ".".join(
                            str(part) for part in issue.get("loc", []) if part != "body"
                        )
                        messages.append(f"{field}: {issue.get('msg', 'invalid value')}")
                detail = "; ".join(messages)
            explanation = (
                detail
                if isinstance(detail, str) and detail
                else body or str(error.reason)
            )
            raise ValueError(
                f"Passage API rejected the request (HTTP 422): {explanation}"
            ) from error
        self._show_message(payload)
        return self._validate_response(payload, numeric_ids, corpus, catalog_identity)

    @staticmethod
    def _validate_response(
        payload: Any,
        numeric_ids: list[int],
        corpus: str,
        catalog_identity: str | None,
    ) -> list[dict[str, Any]]:
        """Validate response alignment and exclude internal passage fields.

        Parameters
        ----------
        payload : object
            Decoded JSON response.
        numeric_ids : list of int
            Requested IDs in their expected order.
        corpus : str
            Expected corpus name.
        catalog_identity : str or None
            Expected namespace when supplied.

        Returns
        -------
        list of dict
            Validated records, with row_index and catalog_identity removed in place.

        Raises
        ------
        ValueError
            If corpus, identity, passage count, IDs or text do not match the request.
        """
        if (
            not isinstance(payload, dict)
            or (
                catalog_identity is not None
                and payload.get("catalog_identity") != catalog_identity
            )
            or payload.get("corpus") != corpus
        ):
            raise ValueError("API returned an incompatible passage-ID namespace")
        passages = payload.get("passages")
        if not isinstance(passages, list) or len(passages) != len(numeric_ids):
            raise ValueError("API did not return the requested number of passages")
        for expected, passage in zip(numeric_ids, passages):
            if (
                not isinstance(passage, dict)
                or type(passage.get("numeric_id")) is not int
                or passage["numeric_id"] != expected
                or not isinstance(passage.get("text"), str)
                or not passage["text"]
            ):
                raise ValueError(
                    "API passage IDs/text do not match the requested results",
                )
            passage.pop("row_index", None)
            passage.pop("catalog_identity", None)
            if isinstance(passage.get("metadata"), dict):
                passage["metadata"] = {
                    key: value
                    for key, value in passage["metadata"].items()
                    if key not in {"row_index", "catalog_identity"}
                }
        return passages


@contextmanager
def report_search_errors() -> Iterator[None]:
    """Report interactive-session errors while keeping library methods reusable.

    Yields
    ------
    None
        Execute terminal input, search and optional caller-managed saving inside
        the context manager.

    Raises
    ------
    SystemExit
        With exit code 1 after an HTTP, operating-system, value or runtime error.
        A concise error message is written to stderr before exiting.

    Notes
    -----
    EOFError and KeyboardInterrupt are suppressed to end an interactive session
    quietly. Outside this context, search methods propagate their exceptions.

    Examples
    --------
    >>> with report_search_errors():  # doctest: +SKIP
    ...     searcher = SemanticSearcher(index="/path/to/index")
    ...     result = searcher.search_passages(input("Query: "), top_k=10)
    """
    try:
        yield
    except (EOFError, KeyboardInterrupt):
        return
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        print(f"API error HTTP {error.code}: {detail}", file=sys.stderr)
        raise SystemExit(1) from error
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Search/API/save failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def _validate_query(query: str, top_k: int) -> None:
    """Validate text and result count before encoding or API requests.

    Parameters
    ----------
    query : str
        Nonempty query text.
    top_k : int
        Result count between 1 and 100000.

    Raises
    ------
    ValueError
        If query is empty or the result count is invalid.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty")
    if type(top_k) is not int or not 1 <= top_k <= _MAX_CANDIDATES:
        raise ValueError("top-k must be an integer between 1 and 100000")


def _search_candidates(query: str, top_k: int, candidates: int | None) -> int:
    """Resolve the candidate budget for a text query.

    Parameters
    ----------
    query : str
        Nonempty query text.
    top_k : int
        Maximum result count.
    candidates : int or None
        Explicit budget, or None for min(100000, max(800, 4 * top_k)).

    Returns
    -------
    int
        Validated budget between top_k and 100000.

    Raises
    ------
    ValueError
        If query, result count or candidate budget is invalid.
    """
    _validate_query(query, top_k)
    if candidates is None:
        candidates = min(_MAX_CANDIDATES, max(_MIN_CANDIDATES, 4 * top_k))
    if type(candidates) is not int or not top_k <= candidates <= _MAX_CANDIDATES:
        raise ValueError("Require 1 <= top-k <= candidates <= 100000")
    return candidates


class SemanticSearcher:
    """Load an index directory once and search its corpus using text queries.

    Parameters
    ----------
    index : str or pathlib.Path
        Directory containing index.json, index.faiss and lists.ivfdata. A path to
        index.faiss is also accepted. The corpus name is read from index.json;
        it is not a separate constructor argument.
    checkpoint : str, pathlib.Path or None, default=None
        Optional local encoder directory. None uses the Hugging Face model/revision
        selected by index.json and shared encoder defaults.
    threads : int, default=32
        Positive CPU thread budget for FAISS, Torch and native numerical libraries.
    device : str, default="cpu"
        Device used to encode queries: "cpu", "cuda" or "cuda:N". FAISS always
        searches on the CPU. CUDA precision support must match the encoder settings.
    nprobe : int or None, default=None
        Override the number of IVF lists probed per query. None uses index.json.
        Larger values generally search more of the index and increase work.
    profile : str or None, default=None
        Optional check that index.json declares this edition. This validates the
        selected index; it does not select a different index directory.
    encoder_factory : callable or None, default=None
        Optional replacement encoder constructor. It receives checkpoint, threads,
        device and resolved encoder settings as positional arguments. The returned
        callable must accept query text and return (vector, input_tokens). None
        constructs QueryEncoder; normal callers can leave this argument unset.
    verbose : bool, default=True
        Show progress bars, index/model details, token counts, API batch progress
        and elapsed times on stderr. False disables this module's progress and
        status messages. Server notices remain visible even with verbose=False.
        Third-party library output uses those libraries' settings.
        Interactive terminals use colored labels and bars; NO_COLOR disables
        colors. Redirected output uses plain text.
        Blocking loading, encoding and FAISS calls report completion rather than
        fractional progress; passage retrieval advances after each validated batch.

    Attributes
    ----------
    corpus : str or None
        API corpus name from index.json. Required for search_passages; offline
        search can run without it.
    client : SearchClient
        Loaded FAISS index, metadata and effective search configuration.
    encoder : callable
        Loaded query encoder, reused for successive searches.
    model_load_seconds : float
        Time spent constructing the encoder, including downloads if needed.
    last_input_tokens : int
        Token count of the most recently encoded query; initially zero.
    last_timing : dict
        Durations in seconds for the latest search's encoding, FAISS search and
        refinement, plus candidates_returned. Initially empty.

    Raises
    ------
    ValueError
        If metadata, files, index structure, encoder settings, thread budget,
        device or optional profile check is invalid.
    OSError
        If index metadata, model files or tokenizer files cannot be accessed.
    RuntimeError
        If FAISS or model loading fails.

    Notes
    -----
    Construction loads the model and index before any query is requested. Reuse
    one instance to avoid repeating that work. Model downloads are cached by
    Hugging Face. Native thread limits are process-wide; avoid concurrent instances
    with conflicting thread settings.

    Methods return JSON-serializable Python data, not a JSON string. This class
    neither prompts for terminal input nor saves output files.

    Examples
    --------
    >>> searcher = SemanticSearcher(index="/path/to/index")  # doctest: +SKIP
    >>> result = searcher.search_passages("What is semantic search?", top_k=10)  # doctest: +SKIP
    >>> matches = searcher.search("How do vector indexes work?", top_k=5)  # doctest: +SKIP
    >>> quiet = SemanticSearcher(index="/path/to/index", verbose=False)  # doctest: +SKIP
    """

    def __init__(
        self,
        index: str | Path,
        checkpoint: str | Path | None = None,
        threads: int = 32,
        device: str = "cpu",
        nprobe: int | None = None,
        profile: str | None = None,
        encoder_factory: Callable[..., Any] | None = None,
        verbose: bool = True,
    ) -> None:
        """Load the index and encoder using the class's constructor parameters.

        Read corpus and index settings from index.json. See the class documentation
        for parameter descriptions, defaults, exceptions and usage examples.
        """
        self._verbose = verbose
        with _report_operation(f"Loading index from {index}", verbose):
            self.client = SearchClient(index, threads=threads, nprobe=nprobe)
        self.corpus = self.client.manifest.get("corpus")
        if profile and profile != self.client.manifest["edition"]:
            raise ValueError("Requested profile differs from release")
        if encoder_factory is None:
            encoder_factory = QueryEncoder
        if verbose:
            _write_status(
                f"Corpus: {self.corpus}; CPU threads: {threads}; encoding device: {device}",
            )
        started = time.perf_counter()
        model = checkpoint or self.client.manifest["encoder"]["model"]
        with _report_operation(f"Loading query encoder {model}", verbose):
            self.encoder = encoder_factory(
                checkpoint,
                threads,
                device,
                self.client.manifest["encoder"],
            )
        self.model_load_seconds = time.perf_counter() - started
        self.last_timing = {}
        self.last_input_tokens = 0

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidates: int | None = None,
        refine: bool = True,
    ) -> list[dict[str, Any]]:
        """Search locally and return ranked numeric IDs with similarity scores.

        Parameters
        ----------
        query : str
            Nonempty text, within the encoder's configured token limit. Overlong
            queries are rejected rather than truncated.
        top_k : int, default=20
            Maximum result count, from 1 through 100000 and no larger than the number
            of indexed vectors.
        candidates : int or None, default=None
            Approximate matches considered before refinement. None uses
            min(100000, max(800, 4 * top_k)). An explicit value must be an integer
            between top_k and 100000. Retrieval is capped at the index vector count.
        refine : bool, default=True
            Re-rank candidates by normalized PQ-reconstruction cosine similarity.
            False uses direct approximate inner-product scores.

        Returns
        -------
        list of dict
            Results sorted by descending score, breaking ties by ascending numeric_id.
            Each record contains integer numeric_id, float score and float
            approximate_inner_product. Fewer than top_k matches may be returned when
            the probed lists do not contain enough candidates. No text is fetched.

        Raises
        ------
        ValueError
            If query, limits, vector dimensions, reconstructions or numeric IDs are
            invalid, or the query exceeds the encoder token limit.
        RuntimeError
            If query encoding or FAISS search fails.

        Notes
        -----
        This method does not call the passage API. It updates last_input_tokens and
        last_timing. Scores come from compressed representations rather than original
        embedding vectors and are not relevance probabilities.

        Examples
        --------
        >>> matches = searcher.search("What is semantic search?", top_k=10)  # doctest: +SKIP
        >>> best_id = matches[0]["numeric_id"]  # doctest: +SKIP
        """
        candidates = _search_candidates(query, top_k, candidates)
        started = time.perf_counter()
        verbose = getattr(self, "_verbose", False)
        with _report_operation("Encoding query", verbose):
            vector, self.last_input_tokens = self.encoder(query)
        encoding_seconds = time.perf_counter() - started
        if verbose:
            _write_status(
                f"Query tokens: {self.last_input_tokens}; top-k: {top_k}; "
                f"candidates: {candidates}; refinement: {refine}",
            )
        with _report_operation("Searching FAISS index", verbose):
            rows, timing = self.client.search(vector, top_k, candidates, refine)
        if verbose:
            _write_status(f"Found {len(rows)} matches", "DONE")
        self.last_timing = {"encoding_seconds": encoding_seconds, **timing}
        return rows

    def search_passages(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        api_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_API_TIMEOUT,
        candidates: int | None = None,
        refine: bool = True,
    ) -> dict[str, Any]:
        """Search locally and return ranked passages as JSON-serializable data.

        Parameters
        ----------
        query : str
            Nonempty text within the configured encoder token limit.
        top_k : int, default=1000
            Maximum number of passages. Must be between 1 and 100000 and no larger
            than the number of indexed vectors. Returned matches may be fewer.
        api_url : str, default="https://api.jmozafari.com"
            Keyword-only HTTP or HTTPS endpoint used to resolve numeric IDs.
        timeout : float, default=120.0
            Keyword-only timeout in seconds for each API request, not the total search.
        candidates : int or None, default=None
            Candidate budget passed to local search. None uses its default budget.
        refine : bool, default=True
            Re-rank local candidates by compressed-vector cosine similarity.

        Returns
        -------
        dict
            A dictionary with query (the original query string) and results (a ranked
            list of passage dictionaries). Each result contains numeric_id, text,
            one-based rank and local similarity score. Additional fields such as
            passage_id, corpus, source_rel_path, source_group and source_record_index
            are retained when supplied by the API. row_index and catalog_identity
            are excluded, including from nested metadata.

        Raises
        ------
        ValueError
            If query, top_k, API settings or search data is invalid; index.json lacks
            a corpus; or an API response fails corpus, identity, ID or text validation.
        urllib.error.HTTPError
            If the passage server returns an HTTP error other than HTTP 422.
            Request-validation errors are reported as ValueError with details.
        urllib.error.URLError
            If the passage API cannot be reached.
        OSError
            If a network operation fails, including a socket timeout.
        RuntimeError
            If model inference or FAISS search fails.

        Notes
        -----
        Ranks and scores come from local search; the API supplies text and provenance.
        Requests are automatically batched into at most 1000 IDs. The full result is
        returned only after every batch succeeds. Nothing is written to disk.

        Examples
        --------
        >>> result = searcher.search_passages("What is semantic search?", top_k=10)  # doctest: +SKIP
        >>> import json
        >>> output_json = json.dumps(result, ensure_ascii=False, indent=2)  # doctest: +SKIP
        """
        _validate_query(query, top_k)
        if not self.corpus:
            raise ValueError(
                "Missing corpus in index.json; provide metadata for this index",
            )
        api = PassageAPIClient(api_url, timeout)
        matches = self.search(query, top_k=top_k, candidates=candidates, refine=refine)
        identity = self.client.manifest.get("catalog_identity")
        passages = api.fetch(
            [row["numeric_id"] for row in matches],
            self.corpus,
            identity,
            verbose=getattr(self, "_verbose", False),
        )
        results = [
            dict(passage, rank=rank, score=match["score"])
            for rank, (match, passage) in enumerate(zip(matches, passages), start=1)
        ]
        return {"query": query, "results": results}


def search(
    index: faiss.Index,
    vector: np.ndarray,
    top_n: int = 10,
    nprobe: int = 128,
) -> list[dict[str, Any]]:
    """Search an already-open FAISS index using approximate inner products.

    Parameters
    ----------
    index : faiss.Index
        Inner-product IVF index with no global direct map.
    vector : numpy.ndarray, shape (1, index.d)
        Finite, near-unit query vector; normalized as a float32 copy for search.
    top_n : int, default=10
        Maximum number of results, from 1 through min(100000, index.ntotal).
    nprobe : int, default=128
        IVF lists to probe, from 1 through the index's number of lists.

    Returns
    -------
    list of dict
        Ranked numeric_id, score and approximate_inner_product records. This
        helper performs no reconstruction refinement, so both scores are equal.

    Raises
    ------
    ValueError
        If vector, index metric, direct-map mode, limits or returned scores are
        invalid.
    RuntimeError
        If FAISS search fails.

    See Also
    --------
    SemanticSearcher.search : Encode a text query and optionally refine matches.
    """
    rows, _ = _retrieve(
        index, vector, top_n, nprobe, max(top_n, _MIN_CANDIDATES), refine=False
    )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    """Build the offline search argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for index selection, search settings and output formatting.
    """
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--index",
        type=Path,
        default=Path.cwd(),
        help="Index directory containing index.json (default: current working directory)",
    )
    argument_parser.add_argument("--query", required=True)
    argument_parser.add_argument(
        "--top-k",
        "--top-n",
        dest="top_k",
        type=int,
        default=20,
    )
    argument_parser.add_argument(
        "--candidates",
        type=int,
        help="Default max(800, 4 * top-k), capped at 100000",
    )
    argument_parser.add_argument("--nprobe", type=int)
    argument_parser.add_argument("--threads", type=int, default=32)
    argument_parser.add_argument(
        "--device",
        default="cpu",
        help="Query encoder only; FAISS stays on CPU",
    )
    argument_parser.add_argument(
        "--profile",
        help="Optional explicit manifest cross-check",
    )
    argument_parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Return direct approximate inner products",
    )
    argument_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable output",
    )
    return argument_parser


def _execute(args: argparse.Namespace, encoder_factory=None) -> dict[str, Any]:
    """Run an offline CLI query and assemble its diagnostic result.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments produced by _build_parser.
    encoder_factory : callable or None, default=None
        Optional encoder constructor passed to SemanticSearcher.

    Returns
    -------
    dict
        Query results, index/search settings, token count and timing diagnostics.

    Raises
    ------
    ValueError
        If settings or search inputs are invalid.
    OSError
        If files or encoder artifacts cannot be accessed.
    RuntimeError
        If encoder or FAISS operations fail.
    """
    candidates = _search_candidates(args.query, args.top_k, args.candidates)
    started = time.perf_counter()
    engine = SemanticSearcher(
        index=args.index,
        threads=args.threads,
        device=args.device,
        nprobe=args.nprobe,
        profile=args.profile,
        encoder_factory=encoder_factory,
    )
    client = engine.client
    results = engine.search(args.query, args.top_k, candidates, not args.no_refine)
    return {
        "edition": client.manifest["edition"],
        "index_vectors": int(client.index.ntotal),
        "query": args.query,
        "input_tokens": engine.last_input_tokens,
        "top_k": args.top_k,
        "returned": len(results),
        "nprobe": client.config["nprobe"],
        "candidates": candidates,
        "refinement": not args.no_refine,
        "score_kind": (
            "cosine_of_PQ_reconstruction"
            if not args.no_refine
            else "approximate_inner_product"
        ),
        "deduplication": False,
        "text_access": False,
        "results": results,
        "timing": {
            "index_open_seconds": client.open_seconds,
            "model_load_seconds": engine.model_load_seconds,
            **engine.last_timing,
            "total_including_startup_seconds": time.perf_counter() - started,
        },
    }


def main(argv: list[str] | None = None) -> None:
    """Run the offline search command-line interface.

    Parameters
    ----------
    argv : list of str or None, default=None
        Command-line arguments, excluding the program name. None uses sys.argv.

    Raises
    ------
    SystemExit
        With code 0 for help, or code 2 for invalid arguments and handled loading
        or search failures.

    Notes
    -----
    Prints numeric IDs and scores to stdout, or a result/diagnostic dictionary
    when --json is supplied. Native loading diagnostics are redirected to stderr.
    This command does not fetch passage text or save an output file.

    Examples
    --------
    Run from a terminal::

        python semantic_search.py --index /path/to/index --query "semantic search" --top-k 10 --json
    """
    argument_parser = _build_parser()
    args = argument_parser.parse_args(argv)
    try:
        with _diagnostic_stdout():
            result = _execute(args)
    except (ValueError, OSError, KeyError, RuntimeError, TypeError) as error:
        argument_parser.exit(2, f"semantic-search: {error}\n")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    else:
        print(
            f"{result['edition']}: {result['returned']} results ("
            f"{result['score_kind']})",
        )
        for rank, row in enumerate(result["results"], 1):
            print(f"{rank}\t{row['numeric_id']}\t{row['score']:.8f}")


if __name__ == "__main__":
    main()
