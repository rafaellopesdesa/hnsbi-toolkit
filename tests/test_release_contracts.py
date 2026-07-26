"""Distribution and executable-example release contracts."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = ROOT / "examples" / "notebooks"
NOTEBOOKS = tuple(sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb")))
FREQUENTIST_NOTEBOOKS = (
    NOTEBOOK_DIRECTORY / "hybrid_reference_flow_and_density_ratios.ipynb",
    NOTEBOOK_DIRECTORY / "neural_importance_sampling_asimov.ipynb",
)
UPSTREAM_REQUIREMENT = (
    "nsbi-common-utils @ "
    "git+https://github.com/iris-hep/nsbi-lhc-toolkit.git@main"
)


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )


def test_lhc_extra_tracks_the_canonical_upstream_library() -> None:
    requirements = metadata.requires("hnsbi-toolkit") or []
    assert any(UPSTREAM_REQUIREMENT in requirement for requirement in requirements)


def test_colab_notebooks_use_the_new_clean_workspace() -> None:
    assert len(NOTEBOOKS) == 5
    for path in NOTEBOOKS:
        notebook = _load_notebook(path)
        source = _source(notebook)
        assert notebook["nbformat"] == 4
        assert "/content/drive/MyDrive/hsbi-toolkit" in source
        assert "Colab Notebooks/ml4hep_tifr_colab" not in source
        assert "rafaellopesdesa/nsbi-lhc-toolkit" not in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs", []) == []


def test_frequentist_notebooks_install_upstream_through_lhc_extra() -> None:
    for path in FREQUENTIST_NOTEBOOKS:
        source = _source(_load_notebook(path))
        assert 'f"{REPO_DIR}[data,flows,lhc,plots]"' in source
        assert "nsbi_common_utils" in source
        assert "nsbi-common-utils @" not in source
