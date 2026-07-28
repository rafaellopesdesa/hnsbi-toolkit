from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "lhc_analysis"
RAW_SAMPLE_NAMES = {
    "signal",
    "background",
    "signal_response_up",
    "signal_response_down",
    "signal_resolution_up",
    "signal_resolution_down",
    "background_response_up",
    "background_response_down",
    "background_resolution_up",
    "background_resolution_down",
    "signal_theory_up",
    "signal_theory_down",
    "reference",
}
REQUESTED_EVENTS = {
    "signal": 12_000,
    "background": 40_000,
    "reference": 8_000,
}


def _load_example_module(filename: str, module_name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        module_name,
        EXAMPLE / filename,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    example_path = str(EXAMPLE)
    inserted = example_path not in sys.path
    if inserted:
        sys.path.insert(0, example_path)
    try:
        specification.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(example_path)
    return module


PRESELECTION = _load_example_module(
    "preselection.py",
    "hnsbi_lhc_preselection_tests",
)
GENERATOR = _load_example_module(
    "generate_distributions.py",
    "hnsbi_lhc_generator_preselection_tests",
)


def _selector() -> object:
    return PRESELECTION.GaussianMixtureRatioSelector(
        features=GENERATOR.FEATURES,
        signal_components=PRESELECTION.reconstructed_components(
            GENERATOR.signal_components(),
            scale=GENERATOR.SCALE,
            resolution=GENERATOR.RESOLUTION,
        ),
        background_components=PRESELECTION.reconstructed_components(
            GENERATOR.background_components(),
            scale=GENERATOR.SCALE,
            resolution=GENERATOR.RESOLUTION,
        ),
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("lhc-preselection")
    paths = GENERATOR.generate(
        output,
        signal_events=REQUESTED_EVENTS["signal"],
        background_events=REQUESTED_EVENTS["background"],
        reference_events=REQUESTED_EVENTS["reference"],
        seed=20260811,
    )
    return output, paths


def _manifest(generated) -> dict:
    _, paths = generated
    return json.loads(Path(paths["preselection_manifest"]).read_text(encoding="utf-8"))


def _assert_frame_rows_preserved(
    raw,
    selected,
    *,
    value_columns: tuple[str, ...],
) -> None:
    assert selected["event_id"].is_unique
    assert set(selected["event_id"]).issubset(set(raw["event_id"]))
    assert set(raw.columns).issubset(selected.columns)
    raw_by_id = raw.set_index("event_id")
    selected_by_id = selected.set_index("event_id")
    aligned = raw_by_id.loc[selected_by_id.index]
    np.testing.assert_array_equal(
        aligned["split"].to_numpy(),
        selected_by_id["split"].to_numpy(),
    )
    np.testing.assert_array_equal(
        aligned.loc[:, value_columns].to_numpy(),
        selected_by_id.loc[:, value_columns].to_numpy(),
    )
    raw_weights = aligned["weight"].to_numpy(dtype=np.float64)
    selected_weights = selected_by_id["weight"].to_numpy(dtype=np.float64)
    scale = selected_weights / raw_weights
    np.testing.assert_allclose(scale, scale[0], rtol=1.0e-12, atol=1.0e-12)


def test_generation_writes_thirteen_raw_and_selected_parquets(generated) -> None:
    output, paths = generated
    assert set(paths) == {
        *RAW_SAMPLE_NAMES,
        *(f"{name}_presel" for name in RAW_SAMPLE_NAMES),
        "preselection_manifest",
    }
    raw_files = {
        path.stem
        for path in output.glob("*.parquet")
        if not path.stem.endswith("_presel")
    }
    selected_files = {
        path.stem.removesuffix("_presel")
        for path in output.glob("*_presel.parquet")
    }
    assert raw_files == RAW_SAMPLE_NAMES
    assert selected_files == RAW_SAMPLE_NAMES
    assert len(list(output.glob("*.parquet"))) == 26
    assert Path(paths["preselection_manifest"]).name == (
        PRESELECTION.MANIFEST_NAME
    )


def test_selected_process_rows_are_preserved_and_pass_one_fixed_cut(
    generated,
) -> None:
    _, paths = generated
    manifest = _manifest(generated)
    selector = _selector()
    assert manifest["selector"]["fingerprint"] == selector.fingerprint
    ratio_cut = float(manifest["cut"]["ratio"])
    config = PRESELECTION.PreselectionConfig(**manifest["config"])
    preserved_columns = (*GENERATOR.LATENT, *GENERATOR.FEATURES)

    for name in sorted(RAW_SAMPLE_NAMES - {"reference"}):
        raw = pd.read_parquet(paths[name])
        selected = pd.read_parquet(paths[f"{name}_presel"])
        _assert_frame_rows_preserved(
            raw,
            selected,
            value_columns=preserved_columns,
        )
        assert set(selected["preselection_partition"]) <= {
            "flow_train",
            "evaluation",
        }
        assert set(selected["preselection_split"]) <= {
            "train",
            "validation",
            "holdout",
        }
        assert (
            selected.loc[
                selected["preselection_partition"] == "evaluation",
                "preselection_split",
            ]
            == "holdout"
        ).all()
        assert (
            selected.loc[
                selected["preselection_partition"] == "flow_train",
                "preselection_split",
            ]
            != "holdout"
        ).all()
        ratios = selector(selected.loc[:, GENERATOR.FEATURES])
        assert np.all(ratios >= ratio_cut)
        diagnostics = manifest["samples"][name]
        assert len(selected) == diagnostics["selected_events"]
        assert float(selected["weight"].sum()) == pytest.approx(
            diagnostics["selected_weight"],
            rel=1.0e-12,
            abs=1.0e-12,
        )

        partitions = PRESELECTION.legacy_partition_labels(
            len(raw),
            config=config,
        )
        expected = (
            (partitions != "preselection")
            & (selector(raw.loc[:, GENERATOR.FEATURES]) >= ratio_cut)
        )
        assert set(selected["event_id"]) == set(raw.loc[expected, "event_id"])


def test_nominal_selection_reaches_target_and_variations_migrate(
    generated,
) -> None:
    _, paths = generated
    manifest = _manifest(generated)
    selector = _selector()
    ratio_cut = float(manifest["cut"]["ratio"])
    target = float(manifest["config"]["target_background_to_signal"])
    cut_ratio = float(manifest["cut"]["background_to_signal"])
    assert 0.90 * target < cut_ratio <= target

    selected_yields = {
        process: float(
            pd.read_parquet(paths[f"{process}_presel"])["weight"].sum()
        )
        for process in ("signal", "background")
    }
    assert selected_yields["signal"] == pytest.approx(
        manifest["cut"]["signal_yield"],
        rel=1.0e-10,
        abs=1.0e-10,
    )
    assert selected_yields["background"] == pytest.approx(
        manifest["cut"]["background_yield"],
        rel=1.0e-10,
        abs=1.0e-10,
    )
    observed_ratio = selected_yields["background"] / selected_yields["signal"]
    assert observed_ratio == pytest.approx(target, rel=0.05)

    config = PRESELECTION.PreselectionConfig(**manifest["config"])
    for process in ("signal", "background"):
        nominal = pd.read_parquet(paths[process])
        nominal_partition = (
            PRESELECTION.legacy_partition_labels(len(nominal), config=config)
            != "preselection"
        )
        nominal_pass = (
            selector(nominal.loc[:, GENERATOR.FEATURES]) >= ratio_cut
        ) & nominal_partition
        nominal_ids = set(nominal.loc[nominal_pass, "event_id"])
        nominal_yield = selected_yields[process]
        response_yields: dict[str, float] = {}

        for direction in ("down", "up"):
            name = f"{process}_response_{direction}"
            varied = pd.read_parquet(paths[name])
            varied_partition = (
                PRESELECTION.legacy_partition_labels(len(varied), config=config)
                != "preselection"
            )
            varied_pass = (
                selector(varied.loc[:, GENERATOR.FEATURES]) >= ratio_cut
            ) & varied_partition
            varied_ids = set(varied.loc[varied_pass, "event_id"])
            assert varied_ids != nominal_ids
            response_yields[direction] = float(
                pd.read_parquet(paths[f"{name}_presel"])["weight"].sum()
            )
            assert response_yields[direction] == pytest.approx(
                manifest["samples"][name]["selected_weight"],
                rel=1.0e-12,
                abs=1.0e-12,
            )

        assert response_yields["down"] < nominal_yield
        assert response_yields["up"] > nominal_yield

        variation_kinds = ["resolution"]
        if process == "signal":
            variation_kinds.append("theory")
        for kind in variation_kinds:
            for direction in ("down", "up"):
                name = f"{process}_{kind}_{direction}"
                varied = pd.read_parquet(paths[name])
                varied_partition = (
                    PRESELECTION.legacy_partition_labels(
                        len(varied),
                        config=config,
                    )
                    != "preselection"
                )
                varied_pass = (
                    selector(varied.loc[:, GENERATOR.FEATURES]) >= ratio_cut
                ) & varied_partition
                varied_ids = set(varied.loc[varied_pass, "event_id"])
                assert varied_ids != nominal_ids
                selected_yield = float(
                    pd.read_parquet(paths[f"{name}_presel"])["weight"].sum()
                )
                assert selected_yield == pytest.approx(
                    manifest["samples"][name]["selected_weight"],
                    rel=1.0e-12,
                    abs=1.0e-12,
                )


def test_selected_reference_is_exactly_balanced_and_normalized(
    generated,
) -> None:
    _, paths = generated
    manifest = _manifest(generated)
    selector = _selector()
    ratio_cut = float(manifest["cut"]["ratio"])
    raw = pd.read_parquet(paths["reference"])
    selected = pd.read_parquet(paths["reference_presel"])
    assert set(raw.columns).issubset(selected.columns)
    assert len(selected) == REQUESTED_EVENTS["reference"]
    assert selected["event_id"].is_unique
    assert np.isfinite(
        selected.loc[:, (*GENERATOR.LATENT, *GENERATOR.FEATURES)]
        .to_numpy(dtype=np.float64)
    ).all()
    assert set(selected["reference_component"]) == {"signal", "background"}
    component_weights = selected.groupby("reference_component")["weight"].sum()
    assert component_weights["signal"] == pytest.approx(0.5, abs=1.0e-12)
    assert component_weights["background"] == pytest.approx(0.5, abs=1.0e-12)
    assert float(selected["weight"].sum()) == pytest.approx(1.0, abs=1.0e-12)
    assert np.all(
        selector(selected.loc[:, GENERATOR.FEATURES]) >= ratio_cut
    )
    assert manifest["reference"]["component_weights"] == pytest.approx(
        {"background": 0.5, "signal": 0.5},
        abs=1.0e-12,
    )


def test_manifest_verification_detects_selected_file_tampering(
    generated,
    tmp_path,
) -> None:
    output, _ = generated
    copied = tmp_path / "copied"
    shutil.copytree(output, copied)
    valid, reasons = PRESELECTION.verify_preselection_manifest(
        copied,
        requested_events=REQUESTED_EVENTS,
        generation_seed=20260811,
        config=PRESELECTION.PreselectionConfig(),
        selector_fingerprint=_selector().fingerprint,
    )
    assert valid
    assert reasons == ()

    mismatched_counts = {**REQUESTED_EVENTS, "signal": 12_001}
    valid, reasons = PRESELECTION.verify_preselection_manifest(
        copied,
        requested_events=mismatched_counts,
    )
    assert not valid
    assert reasons == ("requested generation counts changed",)

    valid, reasons = PRESELECTION.verify_preselection_manifest(
        copied,
        requested_events=REQUESTED_EVENTS,
        generation_seed=20260812,
    )
    assert not valid
    assert reasons == ("generation seed changed",)

    changed_config = PRESELECTION.PreselectionConfig(
        target_background_to_signal=200.0
    )
    valid, reasons = PRESELECTION.verify_preselection_manifest(
        copied,
        config=changed_config,
        selector_fingerprint="not-the-generated-selector",
    )
    assert not valid
    assert reasons == (
        "preselection configuration changed",
        "preselection selector changed",
    )

    malicious = tmp_path / "malicious"
    shutil.copytree(output, malicious)
    manifest_path = malicious / PRESELECTION.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for inventory in ("raw", "selected"):
        manifest["files"][inventory].pop("signal")
        manifest["files"][inventory]["decoy"] = dict(
            manifest["files"][inventory]["background"]
        )
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (malicious / "signal_presel.parquet").open("ab") as stream:
        stream.write(b"unlisted tamper")
    valid, reasons = PRESELECTION.verify_preselection_manifest(
        malicious,
        requested_events=REQUESTED_EVENTS,
        generation_seed=20260811,
        config=PRESELECTION.PreselectionConfig(),
        selector_fingerprint=_selector().fingerprint,
    )
    assert not valid
    assert "preselection file inventory is incomplete" in reasons

    redirected = tmp_path / "redirected"
    shutil.copytree(output, redirected)
    manifest_path = redirected / PRESELECTION.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["selected"]["signal"] = dict(
        manifest["files"]["selected"]["background"]
    )
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    valid, reasons = PRESELECTION.verify_preselection_manifest(redirected)
    assert not valid
    assert (
        "selected file 'signal' does not use its canonical path" in reasons
    )

    linked = tmp_path / "linked"
    shutil.copytree(output, linked)
    external = tmp_path / "external-signal-presel.parquet"
    shutil.copy2(linked / "signal_presel.parquet", external)
    (linked / "signal_presel.parquet").unlink()
    (linked / "signal_presel.parquet").symlink_to(external)
    valid, reasons = PRESELECTION.verify_preselection_manifest(linked)
    assert not valid
    assert "selected file 'signal' is a symbolic link" in reasons

    with (copied / "signal_presel.parquet").open("ab") as stream:
        stream.write(b"tampered")
    valid, reasons = PRESELECTION.verify_preselection_manifest(
        copied,
        requested_events=REQUESTED_EVENTS,
    )
    assert not valid
    assert any(
        "selected file" in reason and "changed" in reason
        for reason in reasons
    )


def test_generation_is_deterministic(tmp_path) -> None:
    requested = {
        "signal": 3_000,
        "background": 9_000,
        "reference": 2_400,
    }
    outputs = [tmp_path / "first", tmp_path / "second"]
    generated_paths = [
        GENERATOR.generate(
            output,
            signal_events=requested["signal"],
            background_events=requested["background"],
            reference_events=requested["reference"],
            seed=314159,
        )
        for output in outputs
    ]
    manifests: list[Mapping[str, object]] = [
        json.loads(
            Path(paths["preselection_manifest"]).read_text(encoding="utf-8")
        )
        for paths in generated_paths
    ]
    for key in (
        "config",
        "cut",
        "generation",
        "reference",
        "requested_events",
        "samples",
        "selector",
    ):
        assert manifests[0][key] == manifests[1][key]
    for inventory in ("raw", "selected"):
        first = manifests[0]["files"][inventory]
        second = manifests[1]["files"][inventory]
        assert set(first) == set(second)
        for name in first:
            assert first[name]["sha256"] == second[name]["sha256"]
            assert first[name]["size_bytes"] == second[name]["size_bytes"]
