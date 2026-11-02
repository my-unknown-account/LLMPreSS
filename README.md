# LLMPreSS — LLM Pretraining Semantic Search

LLMPreSS searches a local FAISS index and retrieves matching passage text from
an HTTP API. Its Python package name is **`llmpress`**.

## Install

Use Python 3.10 or newer. Run these commands from the project directory
containing this README and `pyproject.toml`:

```bash
git clone https://github.com/my-unknown-account/LLMPreSS.git
cd LLMPreSS
python -m pip install .
```

For development, use an editable installation instead:

```bash
python -m pip install -e .
```

Dependencies are installed from this project's `requirements.txt`. The default
encoder runs on CPU. For GPU query encoding, install a CUDA-compatible PyTorch
build for your machine and pass `device="cuda:0"`; FAISS search still uses CPU.

## Search an index

Specify the corpus and edition; repository names and URLs are handled internally:

```python
from llmpress import SemanticIndex

index = SemanticIndex("RedPajama", edition="compact", threads=32, verbose=True)
results = index.search("Hamiltonian paths in cubic graphs", top_k=20)

for result in results:
    print(result.passage_id, result.score)
    print(result.text)
```

`index_dir` is the single location used for both loading and downloading. When
omitted, it resolves to `~/.cache/llmpress/<corpus>/<edition>/`. Choose a different
destination with the same parameter:

```python
index = SemanticIndex(
    "RedPajama", edition="quality", index_dir="/data/indexes/quality",
    threads=32, verbose=False,
)
```

The client checks that directory first:

- **Valid index:** load it directly without contacting the Hub.
- **Missing or empty directory:** download `index.faiss`, `index.json` and
  `lists.ivfdata` into it from the matching published repository, then validate.
- **Nonempty but incompatible directory:** raise a clear `ValueError` without
  downloading or replacing its contents. Required files, metadata, declared file
  sizes, corpus and edition must be correct.

Each chosen directory holds one corpus and edition; the default location
separates them automatically. Automatic downloads require the selected index
to be available on Hugging Face.

The BGE encoder is downloaded and cached automatically by Hugging Face, using the
model settings and pinned revision selected by the index. No checkpoint path or
separate download switch is required. Reuse the loaded index for successive queries.

`search()` returns a ranked list of `SearchResult` objects. Each exposes
`numeric_id`, `passage_id`, `score`, `rank`, `text`, `corpus`, `source_rel_path`,
`source_group` and `source_record_index`. Scores come from local vector search;
higher is better. Unavailable metadata is `None`.

The default passage endpoint is `https://api.jmozafari.com`. Configure it once in
the constructor with `api_url="https://your-api/"` and `timeout=120` if needed.
Requests exceeding 1,000 IDs are split into sequential batches and combined in
order. No partial result list is returned if an API request fails.

To save results, convert the objects to dictionaries in your own code:

```python
import json
from pathlib import Path

payload = {"query": "Hamiltonian paths in cubic graphs",
           "results": [result.to_dict() for result in results]}
Path("results.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
```

`to_dict()` preserves extra API metadata and omits unavailable optional fields.
The library does not prompt for input or save files.

For local IDs and scores without passage API calls, use `search_ids()`:

```python
matches = index.search_ids("Hamiltonian paths in cubic graphs", top_k=20)
for match in matches:
    print(match.numeric_id, match.score)
```

These are also `SearchResult` objects, with `text` and `passage_id` set to `None`.
Both search methods accept `candidates` and `refine` to control re-ranking.
Set `nprobe` in the constructor to override the index's search breadth.

Set `verbose=False` in the constructor to disable client progress and status
messages. Nonempty service notices (`info`, `warning`, `critical`) are still
shown on stderr, even with `verbose=False`. Identical notices appear once per
search rather than once per API batch. `NO_COLOR=1` disables terminal colors. Third-party libraries retain
their own output settings. The encoder uses the Hugging Face cache and downloads
the pinned model on first use; model and index files are not part of the package.

You can also run offline search from a terminal:

```bash
python -m llmpress --index /path/to/index --query "What is semantic search?" --top-k 10 --json
```

The equivalent installed command is `llmpress`. The index defaults to the current
working directory when `--index` is omitted. Use `python -m llmpress --help` or
`help(SemanticIndex)` for parameters and documentation.

`SemanticIndex` is the main retrieval interface. The lower-level
`SemanticSearcher` interface is also available for advanced integrations.

## Example

After installing the package, run an interactive search from the repository root:

```bash
python example_search.py
```

The example downloads the RedPajama compact index on first use. Edit the corpus,
edition or index directory in the script to use a different index.

## Development

The unit tests use mocked encoders and downloads; they do not require index files,
models or a passage server:

```bash
python -m pip install --no-deps -e .
python -m pip install pytest wheel
python -m pytest
```

## Build a distribution

From the repository root:

```bash
python -m pip install build
python -m build
```

The wheel and source archive are written to `dist/`. They include the `llmpress`
library and package metadata. The example is excluded from distributions;
tests may be included in the source archive.


## Project layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── llmpress/
│   ├── __init__.py
│   ├── __main__.py
│   ├── semantic_search.py
│   ├── _index_files.py
│   └── _engine.py
├── example_search.py
└── tests/
```
