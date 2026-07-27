# Workspaces, fits, and scans

`Project.write_configured_workspace()` writes a native, self-contained JSON
workspace. JSON is the stable runtime representation even when the analysis
was authored in YAML.

The workspace records:

- observation and reference-quadrature arrays with checksums;
- one normalized component ratio per physics sample;
- restricted multiplier formulas such as `mu` or `mu * exp(alpha)`;
- parameter roles, initial and nominal values, bounds, and Gaussian
  constraints;
- optional normalized shape/rate systematic anchors;
- relative references to flow, ratio, FNF, and array manifests.

Python `eval`, attribute access, indexing, imports, and arbitrary callables are
not accepted. Every component multiplier is checked for finite,
nonnegative values at runtime.

## Native likelihood

`load_workspace_model()` verifies the schema, paths, manifests, feature order,
intensity fingerprint, and array semantics. `ExtendedUnbinnedLikelihood`
separates the event NLL from Gaussian constraint terms and can replace
auxiliary observations without changing the nominal model metadata.

```python
from hnsbi import Project
from hnsbi.workspace import load_workspace_model

model = load_workspace_model("artifacts/workspace.json")
project = Project.load("analysis.yaml")
likelihood = project.workspace_runtime("artifacts/workspace.json")
```

For every declared FNF, the runtime reconstructs the nominal process density
from the checked reference-flow and ratio manifests and binds the residual
automatically. A portable FNF workspace cannot be written without those
nominal-density manifests. Missing dependencies fail explicitly, and injected
`fnf_systematics` overrides are rejected because their base density cannot be
authenticated against the serialized model.
The workspace also records whether its Asimov arrays were generated with each
FNF; a non-nominal FNF point cannot be serialized from nominal-only weights.

## JAX and iminuit

`JaxLikelihood` implements the same formula and systematic interpolation in
JAX and exposes values, gradients, and Hessians. `MinuitInference` minimizes
the NLL with MIGRAD, runs HESSE, and returns named values, uncertainties,
covariance, correlation, EDM, and parameters at their bounds.

```python
from hnsbi.inference import MinuitInference

inference = MinuitInference(likelihood, use_jax=True)
fit = inference.fit()
profile = inference.profile_scan("mu", [0.0, 0.5, 1.0, 1.5, 2.0])
test = inference.test_statistic_scan("mu", [0.0, 0.5, 1.0, 1.5, 2.0])
```

The objective is the NLL, so iminuit uses `errordef=0.5`. The
test-statistic scan reports the one-sided profiled statistic relative to the
global fit. Profile and test-statistic scans raise `RuntimeError` if the
global fit or any fixed-point fit is unsuccessful or has a non-finite NLL;
failed minimizations are never converted into finite-looking scan points.

## Several workspaces

`CombinedLikelihood` sums independent channel likelihoods. Parameters with
the same name are shared, and each shared Gaussian constraint is counted once.
Each channel retains its own observation, reference support, ratio
normalization, and optional FNF or up/down systematic model.
