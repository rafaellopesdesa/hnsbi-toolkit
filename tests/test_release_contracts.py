"""Distribution and executable-example release contracts."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import yaml

from hnsbi.config import ToolkitConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = ROOT / "examples" / "notebooks"
NOTEBOOKS = tuple(sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb")))
DOCUMENTATION_FILES = (
    ROOT / "README.md",
    *sorted((ROOT / "docs").rglob("*.md")),
    *sorted((ROOT / "examples").rglob("*.md")),
)
FREQUENTIST_NOTEBOOKS = (
    NOTEBOOK_DIRECTORY / "hybrid_reference_flow_and_density_ratios.ipynb",
    NOTEBOOK_DIRECTORY / "neural_importance_sampling_asimov.ipynb",
)
YAML_ENTRYPOINTS = {
    ROOT / "examples" / "lhc_analysis" / "analysis.yaml": "frequentist",
    ROOT / "examples" / "dingo_bbh" / "dual.yaml": "bayesian",
    ROOT / "examples" / "dingo_bns" / "dual.yaml": "bayesian",
}


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])


def _markdown_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )


def _project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _all_declared_requirements(project: dict) -> tuple[str, ...]:
    optional = project["project"]["optional-dependencies"]
    return (
        *project["project"]["dependencies"],
        *(requirement for values in optional.values() for requirement in values),
    )


def test_release_version_is_synchronized_across_artifacts() -> None:
    project_version = _project_metadata()["project"]["version"]
    module_version = runpy.run_path(ROOT / "src" / "hnsbi" / "_version.py")[
        "__version__"
    ]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_project = next(
        package for package in lock["package"] if package["name"] == "hnsbi-toolkit"
    )

    assert project_version == "0.2.0"
    assert module_version == project_version
    assert citation["version"] == project_version
    assert str(citation["date-released"]) == "2026-07-27"
    assert locked_project["version"] == project_version
    assert locked_project["source"] == {"editable": "."}


def test_lhc_extra_is_self_contained_and_includes_pyhf() -> None:
    project = _project_metadata()
    core = project["project"]["dependencies"]
    requirements = project["project"]["optional-dependencies"]["lhc"]

    assert "PyYAML>=6" in core
    assert "numpy>=1.23" in core
    assert "numpy>=1.23" in requirements
    assert "scipy>=1.11.4" in requirements
    assert any(requirement.startswith("pyhf") for requirement in requirements)
    assert not any("nsbi-common-utils" in requirement for requirement in requirements)
    assert not any("git+" in requirement for requirement in requirements)


def test_adapted_code_license_notices_ship_in_the_wheel() -> None:
    project = _project_metadata()
    wheel_files = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    assert wheel_files["NOTICE"] == "hnsbi/NOTICE"
    assert "Permission is hereby granted, free of charge" in notice
    assert "BSD 3-Clause License" in notice
    assert "Copyright (c) 2026, Davide Valsecchi" in notice


def test_lock_and_build_metadata_have_no_removed_upstream_dependency() -> None:
    project = _project_metadata()
    requirements = _all_declared_requirements(project)
    forbidden = (
        "nsbi-common-utils",
        "nsbi_common_utils",
        "nsbi_lhc_toolkit",
        "nsbi-lhc-toolkit",
    )
    assert not any("git+" in requirement for requirement in requirements)
    assert not any(
        token in requirement.lower()
        for requirement in requirements
        for token in forbidden
    )

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert not any(
        package["name"] in {"nsbi-common-utils", "nsbi-lhc-toolkit"}
        for package in lock["package"]
    )
    assert not any(
        isinstance(package.get("source"), dict) and "git" in package["source"]
        for package in lock["package"]
    )

    requirement_files = tuple((ROOT / "requirements").glob("**/*"))
    assert not any(path.is_file() for path in requirement_files)
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "nsbi-common-utils" not in workflow
    assert "requirements/lhc-upstream.txt" not in workflow
    assert "tests/test_native_inference.py" in workflow


def test_primary_frequentist_and_bayesian_interfaces_are_yaml() -> None:
    for path, section in YAML_ENTRYPOINTS.items():
        assert path.is_file()
        assert path.suffix == ".yaml"
        configuration = ToolkitConfig.load(path)
        assert configuration.raw["schema_version"] == "2.0"
        assert getattr(configuration, section) is not None

    lhc_runner = (ROOT / "examples" / "lhc_analysis" / "run_analysis.py").read_text(
        encoding="utf-8"
    )
    assert 'with_name("analysis.yaml")' in lhc_runner


def test_colab_notebooks_use_the_new_clean_workspace() -> None:
    assert len(NOTEBOOKS) == 5
    for path in NOTEBOOKS:
        notebook = _load_notebook(path)
        source = _source(notebook)
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] >= 5
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(cell_ids), path
        assert len(cell_ids) == len(set(cell_ids)), path
        assert "/content/drive/MyDrive/hsbi-toolkit" in source
        assert "Colab Notebooks/ml4hep_tifr_colab" not in source
        assert "rafaellopesdesa/nsbi-lhc-toolkit" not in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs", []) == []


def test_frequentist_notebooks_use_only_the_native_lhc_extra() -> None:
    for path in FREQUENTIST_NOTEBOOKS:
        source = _source(_load_notebook(path))
        assert "[lhc,flows]" in source
        assert "nsbi_common_utils" not in source
        assert "nsbi-common-utils @" not in source


def test_frequentist_colab_setup_keeps_numpy_and_jax_binary_stacks_coherent() -> None:
    for path in FREQUENTIST_NOTEBOOKS:
        source = _source(_load_notebook(path))
        assert "loaded_versions" in source
        assert "jaxlib_version" in source
        assert "jax-cuda12-plugin" in source
        assert "jax-cuda13-plugin" in source
        assert "jax[{extra}]" in source
        assert "signal.SIGKILL" in source
        assert "load a consistent NumPy/JAX stack" in source
        assert "'pull', '--ff-only', 'origin', 'main'" in source


def test_package_has_no_runtime_import_of_the_removed_toolkit() -> None:
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "nsbi_common_utils" not in source, path


def test_documentation_uses_dollar_math_delimiters() -> None:
    legacy_delimiters = (r"\(", r"\)", r"\[", r"\]")

    for path in DOCUMENTATION_FILES:
        source = path.read_text(encoding="utf-8")
        assert not any(delimiter in source for delimiter in legacy_delimiters), path

    for path in NOTEBOOKS:
        source = _markdown_source(_load_notebook(path))
        assert not any(delimiter in source for delimiter in legacy_delimiters), path


def test_documentation_logo_is_configured() -> None:
    conf = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    logo = ROOT / "docs" / "_static" / "hnsbi-logo.png"

    assert 'html_logo = "_static/hnsbi-logo.png"' in conf
    assert 'html_static_path = ["_static"]' in conf
    assert '"sidebar_hide_name": True' in conf
    assert logo.is_file()
    assert logo.stat().st_size > 0
