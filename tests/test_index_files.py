"""Corpus/edition resolution and download tests with no real Hub transfers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmpress import SemanticIndex, _engine, _index_files


def write_index(directory, corpus='RedPajama', edition='compact'):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'index.faiss').write_bytes(b'index')
    (directory / 'lists.ivfdata').write_bytes(b'lists')
    dimension = _engine._ENCODER_DEFAULTS['dimension']
    metadata = {
        'version': 1, 'corpus': corpus, 'edition': edition, 'rows': 1,
        'catalog_identity': 'a' * 64,
        'parameters': {'dimension': dimension, 'pq_bytes': 32, 'nlist': 65536, 'nprobe': 128},
        'index_spec': {'metric': 'inner_product', 'pq_nbits': 8, 'direct_map': 'NoMap',
                       'storage': 'ondisk_ivf', 'opq_dimension': dimension},
        'files': {'index.faiss': {'bytes': 5}, 'lists.ivfdata': {'bytes': 5}},
    }
    (directory / 'index.json').write_text(json.dumps(metadata))


@pytest.fixture
def loaded(monkeypatch):
    calls = []
    def load(**settings):
        calls.append(settings)
        metadata = json.loads((settings['index'] / 'index.json').read_text())
        return SimpleNamespace(corpus=metadata['corpus'])
    monkeypatch.setattr(_engine, 'SemanticSearcher', load)
    return calls


def test_default_download_needs_only_corpus_and_edition(tmp_path, monkeypatch, loaded):
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    calls = []
    def download(**settings):
        calls.append(settings)
        write_index(settings['local_dir'], edition='quality')
        return str(settings['local_dir'])
    monkeypatch.setattr(_index_files, 'snapshot_download', download)
    index = SemanticIndex('RedPajama', edition='quality', verbose=False)
    expected = tmp_path / '.cache/llmpress/RedPajama/quality'
    assert index.directory == loaded[0]['index'] == expected
    assert index.corpus == 'RedPajama' and index.edition == 'quality'
    assert calls == [{
        'repo_id': 'JamshidJDMY/LLMPreSS_RedPajama_quality', 'repo_type': 'dataset',
        'local_dir': expected, 'allow_patterns': ['index.faiss', 'index.json', 'lists.ivfdata'],
    }]


def test_existing_index_is_reused_without_hub_calls(tmp_path, monkeypatch, loaded):
    write_index(tmp_path)
    original = (tmp_path / 'lists.ivfdata').stat().st_mtime_ns
    def forbidden(**settings):
        pytest.fail('A valid index must be reused without contacting Hugging Face')
    monkeypatch.setattr(_index_files, 'snapshot_download', forbidden)
    first = SemanticIndex('RedPajama', index_dir=tmp_path, verbose=False)
    second = SemanticIndex('RedPajama', index_dir=tmp_path, verbose=False)
    assert first.directory == second.directory == tmp_path
    assert len(loaded) == 2
    assert (tmp_path / 'lists.ivfdata').stat().st_mtime_ns == original


def test_explicit_valid_directory_never_downloads(tmp_path, monkeypatch, loaded):
    write_index(tmp_path, corpus='OtherCorpus', edition='quality')
    def forbidden(**settings):
        pytest.fail('Explicit paths must not trigger downloads')
    monkeypatch.setattr(_index_files, 'snapshot_download', forbidden)
    index = SemanticIndex('OtherCorpus', 'quality', index_dir=tmp_path, verbose=False)
    assert index.directory == loaded[0]['index'] == tmp_path
    assert index.corpus == 'OtherCorpus' and index.edition == 'quality'


@pytest.mark.parametrize('problem', ['unrelated_directory', 'missing_file', 'missing_metadata', 'invalid_json',
                                    'wrong_corpus', 'wrong_edition', 'truncated_file', 'file_path'])
def test_invalid_local_path_rejected_before_loading_without_download_fallback(
    tmp_path, monkeypatch, loaded, problem,
):
    directory = tmp_path / 'chosen-index'
    if problem != 'unrelated_directory':
        write_index(directory, corpus='OtherCorpus' if problem == 'wrong_corpus' else 'RedPajama',
                    edition='quality' if problem == 'wrong_edition' else 'compact')
    if problem == 'unrelated_directory':
        directory.mkdir()
        (directory / 'README.txt').write_text('not an index')
    elif problem == 'missing_file':
        (directory / 'lists.ivfdata').unlink()
    elif problem == 'missing_metadata':
        (directory / 'index.json').unlink()
    elif problem == 'invalid_json':
        (directory / 'index.json').write_text('{bad')
    elif problem == 'truncated_file':
        (directory / 'lists.ivfdata').write_bytes(b'x')
    elif problem == 'file_path':
        directory = directory / 'index.faiss'
    def forbidden(**settings):
        pytest.fail('A bad user path must not fall back to a download')
    monkeypatch.setattr(_index_files, 'snapshot_download', forbidden)
    with pytest.raises(ValueError, match='RedPajama/compact') as failure:
        SemanticIndex('RedPajama', index_dir=directory, verbose=False)
    assert str(directory) in str(failure.value)
    assert loaded == []


@pytest.mark.parametrize('corpus,edition', [('bad/path', 'compact'), ('../RedPajama', 'compact'),
                                           ('RedPajama', 'other'), ('', 'quality')])
def test_invalid_corpus_or_edition_does_not_download(monkeypatch, loaded, corpus, edition):
    def forbidden(**settings):
        pytest.fail('Invalid selection must not trigger a download')
    monkeypatch.setattr(_index_files, 'snapshot_download', forbidden)
    with pytest.raises(ValueError):
        SemanticIndex(corpus, edition=edition, verbose=False)
    assert loaded == []


def test_downloaded_metadata_must_match_requested_corpus_and_edition(tmp_path, monkeypatch, loaded):
    def wrong_download(**settings):
        write_index(settings['local_dir'], corpus='OtherCorpus', edition='quality')
    monkeypatch.setattr(_index_files, 'snapshot_download', wrong_download)
    with pytest.raises(ValueError, match='contains OtherCorpus/quality; expected RedPajama/compact'):
        SemanticIndex('RedPajama', index_dir=tmp_path, verbose=False)
    assert loaded == []


@pytest.mark.parametrize('empty_directory', [False, True])
def test_custom_directory_is_the_download_destination(tmp_path, monkeypatch, loaded, empty_directory):
    directory = tmp_path / 'chosen-destination'
    if empty_directory:
        directory.mkdir()
    calls = []
    def download(**settings):
        calls.append(settings)
        write_index(settings['local_dir'])
    monkeypatch.setattr(_index_files, 'snapshot_download', download)
    index = SemanticIndex('RedPajama', index_dir=directory, verbose=False)
    assert calls[0]['local_dir'] == directory
    assert index.directory == directory
    assert loaded[0]['index'] == directory

