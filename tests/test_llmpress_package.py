"""Build and install the client independently without models or network access."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


def test_client_distributions_exclude_tools_and_install_independently(tmp_path):
    source = Path(__file__).parents[1]
    project = tmp_path / "project"
    shutil.copytree(source, project, ignore=shutil.ignore_patterns(
        "__pycache__", "*.egg-info", "build", "dist",
    ))
    result = subprocess.run(
        [sys.executable, "-c", (
            "from setuptools.build_meta import build_sdist, build_wheel; "
            "build_sdist('dist'); build_wheel('dist')"
        )], cwd=project, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((project / "dist").glob("*.whl"))
    archive = next((project / "dist").glob("*.tar.gz"))
    excluded = {"example_search.py", "simulate_search.py", "queries.txt"}
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
        assert "llmpress/__init__.py" in names
        assert "llmpress/semantic_search.py" in names
        assert "llmpress/_engine.py" in names
        assert "llmpress/_index_files.py" in names
        assert "llmpress/__main__.py" in names
        assert not any(Path(name).name in excluded for name in names)
        metadata = bundle.read(next(name for name in names if name.endswith('/METADATA'))).decode()
        assert "Name: llmpress" in metadata
        assert "Requires-Dist: tqdm" in metadata
        assert "Requires-Dist: torch>=2.6.0" in metadata
    with tarfile.open(archive) as bundle:
        assert not any(Path(name).name in excluded for name in bundle.getnames())
    installed = tmp_path / "installed"
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
        "--target", str(installed), str(wheel),
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    environment = os.environ.copy()
    # Avoid importing this repository's source rather than the installed wheel.
    environment["PYTHONPATH"] = str(installed)
    code = (
        "import json, llmpress; from llmpress import SemanticIndex, SearchResult; "
        "from llmpress.semantic_search import SemanticIndex as Internal; "
        "assert SemanticIndex is Internal; assert SearchResult(1, 0.5, 1).score == 0.5; print(json.dumps(llmpress.__file__))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)).is_relative_to(installed)
    result = subprocess.run([sys.executable, "-m", "llmpress", "--help"],
                            cwd=tmp_path, env=environment, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--index" in result.stdout and "--query" in result.stdout
    command = installed / ("Scripts/llmpress.exe" if os.name == "nt" else "bin/llmpress")
    assert command.exists()
