"""Run the YAML-driven native hNSBI analysis and pyhf CLs projection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hnsbi import Project
from hnsbi.diagnostics import json_safe
from hnsbi.impacts import (
    compute_pulls,
    covariance_impacts,
    global_observable_impacts,
    plot_impacts,
    plot_pulls,
)
from hnsbi.inference import MinuitInference
from hnsbi.pyhf_tools import covariance_impacts as pyhf_covariance_impacts
from hnsbi.pyhf_tools import fit as pyhf_fit
from hnsbi.pyhf_tools import global_observable_impacts as pyhf_global_impacts
from hnsbi.pyhf_tools import hypotest, upper_limit
from hnsbi.pyhf_tools import pulls as pyhf_pulls


def _histogram(
    path: Path,
    observable: str,
    *,
    bins: int,
    value_range: tuple[float, float],
) -> list[float]:
    frame = pd.read_parquet(path, columns=[observable, "weight"])
    values = np.histogram(
        frame[observable],
        bins=bins,
        range=value_range,
        weights=frame["weight"],
    )[0]
    return np.maximum(values, 0.0).tolist()


def build_pyhf_projection(
    project: Project,
    *,
    asimov_values: np.ndarray,
    asimov_weights: np.ndarray,
) -> tuple[Any, Any, Any]:
    """Build a binned HistFactory projection from the same YAML samples."""

    import pyhf

    frequentist = project.config.frequentist
    assert frequentist is not None
    settings = frequentist["inference"]["pyhf_projection"]
    observable = settings["observable"]
    observable_index = project.config.features.index(observable)
    bins = int(settings["bins"])
    value_range = tuple(map(float, settings["range"]))
    samples = {sample["name"]: sample for sample in frequentist["samples"]}
    systematics: dict[str, dict[str, dict[str, Path]]] = {}
    for systematic in frequentist["systematics"]:
        for variation in systematic["variations"]:
            systematics.setdefault(variation["sample"], {})[systematic["parameter"]] = {
                direction: project.resolve_path(variation[direction]["path"])
                for direction in ("up", "down")
            }
    projected_samples = []
    for name, sample in samples.items():
        source = project.resolve_path(sample["source"]["path"])
        modifiers: list[dict[str, Any]] = []
        if name == "signal":
            modifiers.append({"name": "mu", "type": "normfactor", "data": None})
        for parameter, paths in systematics.get(name, {}).items():
            modifiers.append(
                {
                    "name": parameter,
                    "type": "histosys",
                    "data": {
                        "hi_data": _histogram(
                            paths["up"],
                            observable,
                            bins=bins,
                            value_range=value_range,
                        ),
                        "lo_data": _histogram(
                            paths["down"],
                            observable,
                            bins=bins,
                            value_range=value_range,
                        ),
                    },
                }
            )
        projected_samples.append(
            {
                "name": name,
                "data": _histogram(
                    source,
                    observable,
                    bins=bins,
                    value_range=value_range,
                ),
                "modifiers": modifiers,
            }
        )
    observation = np.histogram(
        asimov_values[:, observable_index],
        bins=bins,
        range=value_range,
        weights=asimov_weights,
    )[0]
    specification = {
        "channels": [
            {
                "name": frequentist["workspace"]["channel"],
                "samples": projected_samples,
            }
        ],
        "observations": [
            {
                "name": frequentist["workspace"]["channel"],
                "data": np.maximum(observation, 0.0).tolist(),
            }
        ],
        "measurements": [
            {
                "name": frequentist["workspace"]["measurement"],
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {
                            "name": "mu",
                            "inits": [1.0],
                            "bounds": [[0.0, 5.0]],
                        }
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }
    workspace = pyhf.Workspace(specification)
    model = workspace.model()
    data = workspace.data(model)
    return workspace, model, data


def _write_json(path: Path, value: Any) -> None:
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        payload = asdict(value)
    else:
        payload = value
    path.write_text(
        json.dumps(json_safe(payload), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def run(
    config_path: Path,
    *,
    pyhf_toys: bool = False,
    native_toys: bool = False,
    nis: bool = False,
) -> None:
    project = Project.load(config_path)
    project.output_directory.mkdir(parents=True, exist_ok=True)
    reference_artifacts = project.train_reference()
    reference = reference_artifacts.training.flow
    ratios = project.train_ratios(
        reference,
        normalization_events=50_000,
        seed=20260729,
    )
    systematic_training = project.train_systematics()
    runtime_systematics = project.build_runtime_systematics(systematic_training)
    asimov = project.build_configured_asimov(
        reference=reference,
        ratios=ratios.evaluators,
        normalizer=ratios.normalizer,
        systematics=runtime_systematics,
    )
    workspace = project.write_configured_workspace(
        asimov,
        reference_manifest=reference_artifacts.checkpoint_manifest,
        ratio_manifests={
            name: training.manifest_path for name, training in ratios.training.items()
        },
    )
    likelihood = project.workspace_runtime(workspace.path)
    fit = MinuitInference(likelihood).fit()
    pulls = compute_pulls(likelihood, fit)
    global_impacts = global_observable_impacts(
        likelihood,
        "mu",
        fit=fit,
        fit_kwargs={"backend": "minuit"},
    )
    local_impacts = covariance_impacts(likelihood, "mu", fit=fit)
    _write_json(project.output_directory / "fit.json", fit)
    _write_json(project.output_directory / "pulls.json", pulls)
    _write_json(project.output_directory / "impacts_global.json", global_impacts)
    _write_json(project.output_directory / "impacts_covariance.json", local_impacts)
    for name, figure in (
        ("pulls.png", plot_pulls(pulls)[0]),
        ("impacts_global.png", plot_impacts(global_impacts)[0]),
        ("impacts_covariance.png", plot_impacts(local_impacts)[0]),
    ):
        figure.savefig(project.output_directory / name, bbox_inches="tight")

    import pyhf

    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer())
    pyhf_workspace, model, data = build_pyhf_projection(
        project,
        asimov_values=asimov.events.values,
        asimov_weights=asimov.events.weights,
    )
    pyhf_workspace_path = project.output_directory / "pyhf_workspace.json"
    pyhf_workspace_path.write_text(
        json.dumps(dict(pyhf_workspace), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pyhf_fit_result = pyhf_fit(data, model)
    pyhf_pull_result = pyhf_pulls(
        data,
        model,
        fit_result=pyhf_fit_result,
    )
    pyhf_global_result = pyhf_global_impacts(
        data,
        model,
        fit_result=pyhf_fit_result,
    )
    pyhf_covariance_result = pyhf_covariance_impacts(
        data,
        model,
        fit_result=pyhf_fit_result,
    )
    _write_json(project.output_directory / "pyhf_fit.json", pyhf_fit_result)
    _write_json(project.output_directory / "pyhf_pulls.json", pyhf_pull_result)
    _write_json(
        project.output_directory / "pyhf_impacts_global.json",
        pyhf_global_result,
    )
    _write_json(
        project.output_directory / "pyhf_impacts_covariance.json",
        pyhf_covariance_result,
    )
    for name, figure in (
        ("pyhf_pulls.png", plot_pulls(pyhf_pull_result)[0]),
        ("pyhf_impacts_global.png", plot_impacts(pyhf_global_result)[0]),
        (
            "pyhf_impacts_covariance.png",
            plot_impacts(pyhf_covariance_result)[0],
        ),
    ):
        figure.savefig(project.output_directory / name, bbox_inches="tight")
    inference = project.config.frequentist["inference"]["pyhf_projection"]
    scan = inference["scan"]
    asymptotic = upper_limit(
        data,
        model,
        scan=scan,
        calctype="asymptotics",
    )
    _write_json(
        project.output_directory / "limit_asymptotic_asimov.json",
        asymptotic,
    )
    _write_json(
        project.output_directory / "cls_mu1_asymptotic.json",
        hypotest(1.0, data, model, calctype="asymptotics"),
    )
    if pyhf_toys:
        ntoys = int(inference.get("toys", 2000))
        toy_limit = upper_limit(
            data,
            model,
            scan=scan,
            calctype="toybased",
            ntoys=ntoys,
            track_progress=False,
        )
        _write_json(project.output_directory / "limit_toys.json", toy_limit)
    if nis:
        nis_artifacts = project.train_nis_asimov(
            reference=reference,
            ratios=ratios.evaluators,
            systematics=runtime_systematics,
        )
        _write_json(
            project.output_directory / "nis_summary.json",
            {
                "asimov_path": str(nis_artifacts.asimov_path),
                "effective_sample_size": nis_artifacts.asimov.ess,
                "raw_events": nis_artifacts.asimov.raw_events,
                "validation_report": str(nis_artifacts.validation_report),
            },
        )
    if native_toys:
        toy_records = project.generate_configured_toys(
            reference=reference,
            ratios=ratios.evaluators,
            normalizer=ratios.normalizer,
            systematics=runtime_systematics,
            seed=20260731,
        )
        _write_json(
            project.output_directory / "native_toys_summary.json",
            [
                {
                    "point_index": record["point_index"],
                    "toy_index": record["toy_index"],
                    "seed": record["seed"],
                    "path": str(record["path"]),
                    "total_events": record["result"].observed_count,
                }
                for record in toy_records
            ],
        )
    print(f"Analysis products written to {project.output_directory}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("analysis.yaml"),
    )
    parser.add_argument(
        "--pyhf-toys",
        "--toys",
        dest="pyhf_toys",
        action="store_true",
        help="Also run the configured pyhf toy-based CLs limit.",
    )
    parser.add_argument(
        "--native-toys",
        action="store_true",
        help="Also generate the configured unbinned hNSBI toy campaign.",
    )
    parser.add_argument(
        "--nis",
        action="store_true",
        help="Also train the configured NIS proposal and efficient Asimov sample.",
    )
    arguments = parser.parse_args()
    run(
        arguments.config,
        pyhf_toys=arguments.pyhf_toys,
        native_toys=arguments.native_toys,
        nis=arguments.nis,
    )


if __name__ == "__main__":
    main()
