# Getting started

The NumPy-only core loads configurations and provides the backend-independent
data and intensity contracts. Until a package release is published, install
from GitHub:

```bash
python -m pip install \
  "hnsbi-toolkit @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
```

Install only the capabilities needed by the workflow:

```bash
python -m pip install \
  "hnsbi-toolkit[data,flows,plots] @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
python -m pip install \
  "hnsbi-toolkit[bayes] @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
```

The LHC workflow installs the canonical upstream `nsbi-lhc-toolkit` `main`
branch through the `lhc` extra:

```bash
python -m pip install \
  "hnsbi-toolkit[lhc] @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
```

In a source checkout, `python -m pip install -e ".[lhc]"` does the same.
`requirements/lhc-upstream.txt` records the canonical direct requirement for
CI and environment tooling. The upstream repository is consumed as a Python
library and is not copied or cloned by hNSBI workflows.

Load a JSON file or an equivalent Python dictionary with the same function:

```python
from hnsbi import Project, ToolkitConfig

config = ToolkitConfig.load("analysis.json")
print(config.features)
print(config.output_dir)

project = Project.load("analysis.json")
intensity = project.intensity_model()
```

Complete configurations are provided in `examples/configs/`. Paths in a file
configuration are interpreted relative to that file by workflow runners. An
in-memory PyArrow table or Awkward array is associated with a `registry_key`
rather than serialized into JSON. `Project.train_reference()` writes both a
trusted native checkpoint and the portable ONNX log-density/inverse bundle;
`Project.train_ratios()` delegates training and diagnostics to
`nsbi-common-utils` and converts every scaler to ONNX.

The complete frequentist sequence is deliberately explicit:

```python
# For paper-level closure, pass a disjoint DataSource as validation_source.
reference_artifacts = project.train_reference()

# Bind the frozen reference through either its native object or ONNX bundle.
reference = reference_artifacts.training.flow
ratio_artifacts = project.train_ratios(reference=reference)
print(ratio_artifacts.normalizer_manifest)

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
runtime = project.workspace_runtime(workspace.path)
```

When `frequentist.workspace.base_config` is present, the project first invokes
the upstream `WorkspaceBuilder` and then replaces the single configured
unbinned channel and measurement with the verified hNSBI arrays and metadata.
Without `base_config`, it writes the minimal compatible skeleton directly.

For a Bayesian project whose Parquet datasets were already sampled under
$\rho$, $\nu$, and $\kappa$, the native five-stage trainer is one call:

```python
dual_artifacts = project.train_dual()
dual_model = dual_artifacts.model
```

The resulting `dual_model.manifest.json` can later be verified and loaded as a
lazy ONNX model with `hnsbi.bayes.load_dual_model()`.

## Before a production run

1. Reserve independent samples for training, ratio normalization, and final
   closure.
2. Freeze a flow before generating the denominator class for its residual
   classifier.
3. Export the complete deployable graph, including preprocessing, to ONNX.
4. Verify artifact checksums and ordered feature signatures.
5. Inspect ratio normalization, ESS, and tail diagnostics.
6. Reproduce a known analytic or simulator-based closure before fitting data.
