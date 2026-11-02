"""Resolve and validate local or automatically downloaded index files."""

from pathlib import Path
import re

from huggingface_hub import snapshot_download

from . import _engine

_INDEX_FILES = ("index.faiss", "index.json", "lists.ivfdata")
_NAMESPACE = "JamshidJDMY"


def _validate_directory(directory: Path, corpus: str, edition: str) -> Path:
    """Reject missing files, invalid metadata and mismatched corpus/edition."""
    expected = f"{corpus}/{edition}"
    if not directory.is_dir():
        raise ValueError(f"Index path for {expected} is not an existing directory: {directory}")
    missing = [name for name in _INDEX_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Invalid index directory for {expected} at {directory}: "
                         f"missing {', '.join(missing)}")
    try:
        _, metadata = _engine.load_manifest(directory)
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid index directory for {expected} at {directory}: {error}") from error
    actual = f"{metadata.get('corpus', 'unspecified')}/{metadata.get('edition', 'unspecified')}"
    if metadata.get("corpus") != corpus or metadata.get("edition") != edition:
        raise ValueError(f"Index directory {directory} contains {actual}; expected {expected}.")
    return directory


def _resolve_index_directory(
    corpus: str,
    edition: str,
    index_dir: str | Path | None,
    verbose: bool,
) -> Path:
    """Reuse a valid directory or download into a missing/empty destination."""
    if not isinstance(corpus, str) or re.fullmatch(r"[A-Za-z0-9_-]+", corpus) is None:
        raise ValueError("Use a corpus name such as 'RedPajama'; pass local paths with index_dir=.")
    if edition not in ("compact", "quality"):
        raise ValueError("edition must be 'compact' or 'quality'")
    directory = (
        Path(index_dir).expanduser() if index_dir is not None
        else Path.home() / ".cache" / "llmpress" / corpus / edition
    ).resolve()
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"Index path for {corpus}/{edition} is not a directory: {directory}")
        if any(directory.iterdir()):
            return _validate_directory(directory, corpus, edition)
    repo_id = f"{_NAMESPACE}/LLMPreSS_{corpus}_{edition}"
    if verbose:
        _engine._write_status(f"Preparing {corpus}/{edition} in {directory}")
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=directory,
        allow_patterns=list(_INDEX_FILES),
    )
    return _validate_directory(directory, corpus, edition)
