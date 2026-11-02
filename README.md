<div align="center">

# LLMPreSS

### LLM Pretraining Semantic Search

**Find relevant passages in pretraining corpora with a few lines of Python.**

Search local FAISS indexes. Retrieve passage text. Keep querying with one reusable client.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/Search-FAISS-6366F1)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-14B8A6)

[Quick start](#quick-start) · [How it works](#how-it-works) · [Usage](#usage) · [Configuration](#configuration) · [Development](#development)

</div>

---

## Why LLMPreSS?

| | What you get |
| :--- | :--- |
| **Semantic retrieval** | Search passages by meaning using BGE query encoding and a local FAISS index. |
| **Automatic setup** | Select a corpus and edition; the client resolves and downloads the published index. |
| **Passage text** | Retrieve matching passages and source metadata through an HTTP API. |
| **Reusable client** | Load the index and encoder once, then run successive queries. |
| **Flexible output** | Work with Python result objects, export JSON, or retrieve only local IDs and scores. |

## Quick start

Requires **Python 3.10+**.

```bash
git clone https://github.com/my-unknown-account/LLMPreSS.git
cd LLMPreSS
python -m pip install .
```

```python
from llmpress import SemanticIndex

index = SemanticIndex("RedPajama", edition="compact")
results = index.search("Hamiltonian paths in cubic graphs", top_k=20)

for result in results:
    print(result.passage_id, result.score)
    print(result.text)
```

> [!NOTE]
> On first use, the client downloads the selected index and the encoder model.
> Automatic index downloads require that edition to be published on Hugging Face.
> Model and index files are separate from the Python package.

Prefer an interactive example? Run [`python example_search.py`](example_search.py)
from the repository root after installation.

## How it works

```mermaid
flowchart LR
    A[Your query] --> B[BGE encoder]
    B --> C[Local FAISS index]
    C --> D[Ranked IDs and scores]
    D --> E[Passage HTTP API]
    E --> F[Passage text and metadata]
```

Vector search runs locally on the CPU. Query encoding also uses the CPU by default,
with optional GPU support. `search()` retrieves the matching text from the passage
API; `search_ids()` returns local matches without making passage API calls.

## Usage

### Choose where indexes live

By default, indexes are cached under:

```text
~/.cache/llmpress/<corpus>/<edition>/
```

Use `index_dir` to choose a directory for both loading and downloading:

```python
index = SemanticIndex(
    "RedPajama",
    edition="quality",
    index_dir="/data/indexes/quality",
    threads=32,
    verbose=False,
)
```

| Directory state | Client behavior |
| :--- | :--- |
| Valid index | Loads it directly without contacting the Hub. |
| Missing or empty | Downloads `index.faiss`, `index.json`, and `lists.ivfdata`, then validates them. |
| Nonempty but incompatible | Raises `ValueError` without downloading or replacing its contents. |

Each directory holds one corpus and edition. Validation checks required files,
metadata, declared file sizes, corpus, and edition. The encoder uses the model
settings and pinned revision selected by the index and is cached by Hugging Face.

### Work with results

`search()` returns a ranked list of `SearchResult` objects:

| Field | Meaning |
| :--- | :--- |
| `numeric_id` | Numeric passage identifier within the corpus |
| `passage_id` | Original passage identifier, when available |
| `score` | Local vector search score; higher is better |
| `rank` | One-based position in the results |
| `text` | Retrieved passage text |
| `corpus` | Corpus name |
| `source_rel_path` | Source file path, when available |
| `source_group` | Source group, when available |
| `source_record_index` | Source record index, when available |

Unavailable metadata is `None`. Use `to_dict()` to preserve extra API metadata
and omit unavailable optional fields when exporting:

```python
import json
from pathlib import Path

payload = {
    "query": "Hamiltonian paths in cubic graphs",
    "results": [result.to_dict() for result in results],
}
Path("results.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

The library leaves prompting and saving files to your application.

### Retrieve only IDs and scores

```python
matches = index.search_ids("Hamiltonian paths in cubic graphs", top_k=20)

for match in matches:
    print(match.numeric_id, match.score)
```

These are also `SearchResult` objects, with `text` and `passage_id` set to `None`.
The index and encoder must already be available locally for fully offline use.

### Search from the terminal

Use the CLI to search a local index and print IDs and scores:

```bash
python -m llmpress \
  --index /path/to/index \
  --query "What is semantic search?" \
  --top-k 10 \
  --json
```

The installed `llmpress` command is equivalent to `python -m llmpress`. If `--index`
is omitted, the CLI uses the current working directory.

## Configuration

| Option | Purpose |
| :--- | :--- |
| `index_dir` | Directory used to load or download an index |
| `device` | Encoder device; defaults to CPU, or use `"cuda:0"` with compatible PyTorch |
| `threads` | CPU search thread count |
| `nprobe` | Override the index's search breadth |
| `api_url` | Passage service URL; defaults to `https://api.jmozafari.com` |
| `timeout` | Passage API request timeout in seconds |
| `verbose` | Enable or disable client progress and status messages |

Both search methods accept `candidates` and `refine` to control re-ranking.
For full parameter documentation, use `help(SemanticIndex)` or
`python -m llmpress --help`. The lower-level `SemanticSearcher` is also available
for advanced integrations.

<details>
<summary><strong>GPU query encoding</strong></summary>

Install a CUDA-compatible PyTorch build for your machine, then select the encoder
device when constructing the index:

```python
index = SemanticIndex("RedPajama", edition="compact", device="cuda:0")
```

FAISS search continues to run on the CPU.

</details>

<details>
<summary><strong>Passage API behavior and service notices</strong></summary>

Requests exceeding 1,000 IDs are split into sequential batches and combined in
order. If an API request fails, the client does not return a partial result list.

Service notices (`info`, `warning`, `critical`) appear on stderr even when
`verbose=False`. Identical notices appear once per search. Set `NO_COLOR=1` to
disable terminal colors. Third-party libraries retain their own output settings.

</details>

## Development

Install an editable copy:

```bash
python -m pip install -e .
```

### Build a distribution

```bash
python -m pip install build
python -m build
```

Distributions are written to `dist/`. The wheel contains the library and package
metadata. The interactive example is excluded from distributions.

<details>
<summary><strong>Repository layout</strong></summary>

```text
.
├── llmpress/
│   ├── __init__.py
│   ├── __main__.py
│   ├── semantic_search.py
│   ├── _index_files.py
│   └── _engine.py
├── example_search.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

</details>

---

[Report an issue](https://github.com/my-unknown-account/LLMPreSS/issues)
