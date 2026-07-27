from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "examples" / "notebooks"


def _notebook(name: str) -> dict:
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])


def test_hybrid_notebook_covers_exercise_five_plot_families() -> None:
    notebook = _notebook("hybrid_reference_flow_and_density_ratios.ipynb")
    cell_ids = {cell["id"] for cell in notebook["cells"]}
    assert {
        "hybrid-reference-closure",
        "hybrid-reference-truth",
        "hybrid-ratio-validation",
        "hybrid-weighted-reference",
        "hybrid-density-truth",
        "hybrid-nominal-scan",
        "hybrid-fit",
        "hybrid-compressed-toys",
        "hybrid-asimov-toys",
    }.issubset(cell_ids)

    source = _source(notebook)
    for required in (
        "plot_flow_pair_closure",
        "plot_log_prob_closure",
        "plot_log_prob_cdf_closure",
        "plot_ratio_validation_calibration",
        "plot_ratio_validation_reweighting",
        "plot_log_density_truth_scatter",
        "plot_log_density_truth_binned",
        "hybrid_vs_analytic_profile_likelihood",
        "full_systematics_profile_scan",
        "toy_test_statistic_distributions",
        "toy_signal_strength_distributions",
        "asimov_prediction_vs_toys",
    ):
        assert required in source
    assert "TOY_PROFILE = 'quick'  # 'quick' or 'paper'" in source
    assert "write_reuse_provenance" in source


def test_nis_notebook_covers_exercise_six_plot_families() -> None:
    notebook = _notebook("neural_importance_sampling_asimov.ipynb")
    source = _source(notebook)
    for required in (
        "proposal_target_closure",
        "importance_reweighted_reference_closure",
        "asimov_benchmark_scan",
        "nis_asimov_convergence",
        "nis_repeated_small_asimov_scans",
        "nis_q0_equal_cost",
    ):
        assert required in source
    assert "STUDY_PROFILE = 'quick'  # 'quick' or 'paper'" in source
    assert "TARGET_ROWS" in source
    assert "Loading verified Exercise-5 nominal artifacts." in source
    assert "verify_reuse_provenance" in source
    assert "result.reference_weights" in source
    assert "plot_nis_feature_closure(feature_closure, columns=5)" in source
    assert "len(showcase_scans[method]) < 8" in source
    assert (
        "axis.set_xticks([1, 2], ['Direct reference', 'Neural importance'])"
        in source
    )


def test_nis_design_points_cover_the_notebook_scan() -> None:
    configuration = yaml.safe_load(
        (ROOT / "examples" / "lhc_analysis" / "analysis.yaml").read_text(
            encoding="utf-8"
        )
    )
    points = configuration["frequentist"]["nis"]["design_points"]
    assert [point["mu"] for point in points] == [index / 4 for index in range(13)]
    assert all(
        point[name] == 0.0
        for point in points
        for name in ("response", "resolution", "theory")
    )
