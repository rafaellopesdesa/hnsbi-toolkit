# Getting started

Install the capabilities needed by your workflow:

```bash
python -m pip install \
  "hnsbi-toolkit[lhc,flows] @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
python -m pip install \
  "hnsbi-toolkit[bayes] @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
```

For development, a source checkout can install everything with
`python -m pip install -e ".[data,flows,lhc,bayes,plots,test,docs]"`.
The LHC extra includes JAX, iminuit, and pyhf. It does not install
`nsbi-lhc-toolkit`.

## Start with YAML

YAML is the primary user interface. It is strict: duplicate keys, non-finite
numbers, unknown properties, and unsupported schema versions fail before
training.

```yaml
schema_version: "2.0"
features: [x, y, z]
output_dir: artifacts

frequentist:
  reference:
    kind: parquet
    path: data/reference.parquet
  samples:
    - name: signal
      source: {kind: parquet, path: data/signal.parquet}
      nominal_yield: 100.0
      multiplier: mu
  flow:
    architecture: realnvp
    n_coupling_layers: 6
    hidden_features: 128
    hidden_layers: 2
    training:
      epochs: 100
      batch_size: 2048
      learning_rate: 0.0003
  ratios:
    backend: native
    ensemble_size: 5
    normalization: independent_reference_mean
    calibration: true
    calibration_type: isotonic
    training:
      epochs: 100
      batch_size: 2048
      learning_rate: 0.001
      hidden_layers: 3
      neurons: 128
  parameters:
    - name: mu
      role: poi
      nominal: 1.0
      bounds: [0.0, 5.0]
  workspace:
    backend: native
    measurement: signal_strength
    channel: signal_region
    output_path: artifacts/workspace.json
```

Load and validate the same file through either public entry point:

```python
from hnsbi import Project, ToolkitConfig

config = ToolkitConfig.load("analysis.yaml")
project = Project.load("analysis.yaml")
```

Paths are resolved relative to the YAML file. Parquet and Arrow IPC are
file-backed sources; in-memory PyArrow and Awkward objects use a
`registry_key`.

The complete frequentist sequence remains explicit:

```python
reference_artifacts = project.train_reference()
reference = reference_artifacts.training.flow

ratio_artifacts = project.train_ratios(reference=reference)
asimov = project.build_configured_asimov(
    reference=reference,
    ratios=ratio_artifacts.evaluators,
    normalizer=ratio_artifacts.normalizer,
)
workspace = project.write_configured_workspace(
    asimov,
    reference_manifest=reference_artifacts.onnx_bundle.manifest_path,
    ratio_manifests={
        name: result.manifest_path
        for name, result in ratio_artifacts.training.items()
    },
)
likelihood = project.workspace_runtime(workspace.path)
fit = likelihood.fit(backend="minuit", use_jax=True)
scan = likelihood.profile_scan("mu", [0.0, 0.5, 1.0, 1.5, 2.0])
```

Native ratio training writes reproducible train/validation/holdout splits,
embedded-preprocessing ONNX models, calibration state, diagnostics, and
independent $E_q[r]$ normalization.

## Bayesian YAML

The Bayesian YAML names samples already drawn under the $\rho$, $\nu$, and
$\kappa$ designs. Each source should contain explicit `train`, `validation`,
and `holdout` rows:

```python
project = Project.load("examples/dingo_bbh/dual.yaml")
training_data = project.dual_training_data()
dual_artifacts = project.train_dual()
```

See {doc}`bayesian/training` for the five training stages and
{doc}`examples/dingo` for two complete configurations.

## JSON remains available

JSON and Python dictionaries use the same schema. JSON is the canonical
serialization for resolved configurations, workspaces, manifests, training
history, and diagnostics:

```python
config.dump_json("resolved-analysis.json")
```

## Before a production run

1. Reserve independent samples for training, validation, normalization, and
   final closure.
2. Freeze a proposal flow before training its residual classifier.
3. Verify checksums, ordered feature signatures, and ONNX/native parity.
4. Inspect calibration, reweighting closure, ratio tails, normalization, and
   effective sample size.
5. Reproduce an analytic or simulator-based closure before interpreting a fit
   or exclusion.
