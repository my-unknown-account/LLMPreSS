"""Public retrieval API tests without model downloads or external requests."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmpress import SearchResult, SemanticIndex
from llmpress import _engine, semantic_search


@pytest.fixture
def engine(monkeypatch):
    calls = []
    rows = [{
        "numeric_id": 7, "score": 0.9, "rank": 99,
        "passage_id": "Example:7", "text": "A passage.",
        "source_rel_path": "source.jsonl.zst", "source_group": "pretraining",
        "source_record_index": 12, "metadata": {"language": "en"},
    }]

    def passages(query, **options):
        calls.append(("passages", query, options))
        return {"query": query, "results": rows}

    def ids(query, **options):
        calls.append(("ids", query, options))
        return [{"numeric_id": 7, "score": 0.9, "approximate_inner_product": 0.8}]

    loaded = SimpleNamespace(corpus="Example", search_passages=passages, search=ids)

    def load(**options):
        calls.append(("load", options))
        return loaded

    monkeypatch.setattr(_engine, "SemanticSearcher", load)
    monkeypatch.setattr(semantic_search, "_resolve_index_directory",
                        lambda corpus, edition, index_dir, *args: Path(index_dir or '.').resolve())
    return calls, rows, loaded


def test_search_returns_ranked_result_objects_and_serializable_metadata(engine, tmp_path):
    calls, rows, _ = engine
    index = SemanticIndex("Example", index_dir=tmp_path, verbose=False)
    results = index.search("a query", top_k=20)
    assert isinstance(results, list) and len(results) == 1
    hit = results[0]
    assert isinstance(hit, SearchResult)
    assert (hit.passage_id, hit.score, hit.text, hit.numeric_id) == ("Example:7", 0.9, "A passage.", 7)
    assert hit.rank == 1 and hit.corpus == index.corpus == "Example"
    assert hit.source_rel_path == "source.jsonl.zst"
    assert hit.source_record_index == 12
    payload = hit.to_dict()
    assert payload["metadata"] == {"language": "en"}
    assert "_extra" not in payload
    payload["metadata"]["language"] = "changed"
    assert hit.to_dict()["metadata"]["language"] == rows[0]["metadata"]["language"] == "en"
    with pytest.raises(FrozenInstanceError):
        hit.score = 0
    assert "A passage." not in repr(hit)
    assert index.directory == tmp_path
    assert "SemanticIndex(" in repr(index)
    assert [call[0] for call in calls] == ["load", "passages"]


def test_index_is_loaded_once_and_constructor_config_is_reused(engine):
    calls, _, _ = engine
    index = SemanticIndex(
        "Example", index_dir="indexes/compact", threads=4, device="cuda:0", verbose=False,
        nprobe=8, api_url="https://example.test/", timeout=30,
    )
    index.search("first", top_k=3, candidates=40, refine=False)
    index.search("second", top_k=5)
    assert [call[0] for call in calls] == ["load", "passages", "passages"]
    assert calls[0][1] == {
        "index": Path("indexes/compact").resolve(), "threads": 4, "device": "cuda:0",
        "verbose": False, "nprobe": 8,
    }
    assert calls[1][2] == {
        "top_k": 3, "api_url": "https://example.test/", "timeout": 30,
        "candidates": 40, "refine": False,
    }
    assert calls[2][2]["api_url"] == "https://example.test/"


def test_search_ids_does_not_request_passages(engine):
    calls, _, loaded = engine
    def forbidden(*args, **kwargs):
        pytest.fail("Offline search contacted the passage API")
    loaded.search_passages = forbidden
    index = SemanticIndex("Example", index_dir="indexes/compact")
    hits = index.search_ids("query", top_k=5, candidates=50, refine=False)
    assert [call[0] for call in calls] == ["load", "ids"]
    assert (hits[0].numeric_id, hits[0].score, hits[0].rank) == (7, 0.9, 1)
    assert hits[0].text is hits[0].passage_id is None
    assert hits[0].to_dict()["approximate_inner_product"] == 0.8


@pytest.mark.parametrize("settings", [{"api_url": "invalid"}, {"timeout": 0}])
def test_bad_api_settings_fail_before_loading_index(engine, settings):
    calls, _, _ = engine
    with pytest.raises(ValueError):
        SemanticIndex("Example", index_dir="indexes/compact", **settings)
    assert calls == []


def test_retrieval_errors_propagate_without_partial_results(engine):
    _, _, loaded = engine
    def fail(*args, **kwargs):
        raise OSError("passage API unavailable")
    loaded.search_passages = fail
    index = SemanticIndex("Example", index_dir="indexes/compact")
    with pytest.raises(OSError, match="passage API unavailable"):
        index.search("query")


def test_index_directory_is_resolved(engine, tmp_path):
    assert SemanticIndex("Example", index_dir=tmp_path).directory == tmp_path
