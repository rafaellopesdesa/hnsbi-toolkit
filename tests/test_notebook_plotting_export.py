from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pyplot = pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]


def _plotting_module():
    path = ROOT / "examples" / "notebooks" / "utils_plotting.py"
    specification = importlib.util.spec_from_file_location(
        "hnsbi_notebook_plotting",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_standalone_export_reconstructs_image_and_colorbar(tmp_path) -> None:
    plotting = _plotting_module()
    figure, axis = pyplot.subplots()
    image = axis.imshow(np.arange(16, dtype=np.float64).reshape(4, 4))
    figure.colorbar(image, ax=axis)

    script = plotting.export_standalone_figure_script(
        figure,
        "image_with_colorbar",
        output_dir=tmp_path,
    )
    pyplot.close(figure)
    assert ".imshow(" in script.read_text(encoding="utf-8")

    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert script.with_suffix(".png").is_file()
