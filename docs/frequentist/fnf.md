# Factorizable systematic flows

Factorizable normalizing flows (FNFs) provide a normalized alternative to
endpoint density-ratio interpolation for continuous systematic variations.
The implementation follows *Factorizable Normalizing Flows for
parameter-dependent density morphing*,
[arXiv:2606.30489](https://arxiv.org/abs/2606.30489), and records that
provenance in every saved model.

## Density construction

Let $p_0(x)$ be a frozen, normalized nominal process density. An FNF learns an
invertible map from an observed nuisance-dependent event to nominal
coordinates,

$$
x_0 = T_\nu(x).
$$

The varied density is

$$
p(x\mid\nu)
=
p_0\!\left(T_\nu(x)\right)
\left|\det\frac{\partial T_\nu(x)}{\partial x}\right|.
$$

Each autoregressive layer transforms feature $j$ as

$$
x_{0,j}
=
x_j\exp s_j(x_{<j},\nu)+t_j(x_{<j},\nu).
$$

For standardized nuisance coordinates

$$
u_k=\frac{\nu_k-\nu_{0,k}}{\sigma_k},
$$

the scale and shift contain independent linear and quadratic contributions
from every nuisance,

$$
s_j
=
\sum_k\left[u_k\alpha_j^k+u_k^2\beta_j^k\right]
+
\sum_{k<l}u_ku_l\phi_j^{kl},
$$

with the same form for $t_j$. Each nuisance has its own coefficient networks.
Optional pairwise networks model effects that cannot be reconstructed from the
axis variations alone.

All coefficient output layers start at zero. Consequently
$T_{\nu_0}(x)=x$ and the log determinant is exactly zero at the nominal point,
independently of the learned weights. Feature-order permutations are applied
by conjugating each residual layer, so stacked layers retain the same exact
identity.

## Configure and train

The primary interface declares one FNF per sample in the project YAML:

```yaml
frequentist:
  fnf:
    models:
      - name: signal_detector
        sample: signal
        nuisances: [response, resolution, theory]
        centers: {response: 0.0, resolution: 0.0, theory: 0.0}
        scales: {response: 1.0, resolution: 1.0, theory: 1.0}
        interactions:
          - [response, resolution]
        architecture:
          num_layers: 2
          hidden_features: [128, 128]
          log_scale_clip: 1.5
        training:
          epochs: 100
          batch_size: 1024
          learning_rate: 0.0001
          validation_fraction: 0.2
          holdout_fraction: 0.1
        anchors:
          - name: response-down
            point: {response: -1.0}
            source: {kind: parquet, path: data/signal_response_down.parquet}
          - name: response-up
            point: {response: 1.0}
            source: {kind: parquet, path: data/signal_response_up.parquet}
          # Add axis anchors for resolution and theory and joint anchors
          # for every configured interaction.
        yield_anchors:
          response: [0.97, 1.04]
        output_path: artifacts/fnf/signal.manifest.json
```

After freezing a differentiable nominal density for each configured sample,
train and save every model in one call:

```python
from hnsbi import Project

project = Project.load("analysis.yaml")
artifacts = project.train_fnf_systematics(
    reference=reference_training.training.flow,
    ratios=ratio_artifacts,
    reference_manifest=reference_training.checkpoint_manifest,
)
```

The project composes the normalized nominal process density
$p_s=q\,r_s/E_q[r_s]$ from the frozen reference flow and the trained native
ratio ensemble. It reconstructs both models from their checksum-verified
checkpoints and verifies that any same-process live objects are identical to
those artifacts; cross-binding a flow or ratio from another training run is an
error. Advanced callers may instead pass a
`base_torch_log_probs={"signal": callable}` mapping, but such residuals have
no authenticated portable base-density provenance and therefore cannot be
attached to a workspace.

`Project.load_fnf_systematics()` reloads and verifies the configured
manifests. Bind the residuals to their nominal process densities once for
likelihood evaluation, Asimov construction, and toy generation:

```python
fnf_systematics = project.build_fnf_runtime_systematics(
    reference_density=reference_training.training.flow,
    ratios=ratio_artifacts,
    reference_manifest=reference_training.checkpoint_manifest,
)
```

When `frequentist.fnf` is present, Project Asimov and toy helpers require this
mapping explicitly. They raise instead of silently generating data from the
nominal model.

The lower-level API remains independent of file loading:

```python
from hnsbi.fnf import (
    FNFAnchor,
    FNFResidualConfig,
    FNFTrainer,
    FNFTrainingConfig,
)

residual_config = FNFResidualConfig(
    n_features=3,
    nuisance_names=("scale", "resolution", "theory"),
    num_layers=2,
    hidden_features=(128, 128),
    interactions=(("scale", "resolution"),),
    log_scale_clip=1.5,
)

training_config = FNFTrainingConfig(
    epochs=100,
    batch_size=1024,
    learning_rate=1e-4,
    validation_fraction=0.2,
    holdout_fraction=0.1,
)

anchors = [
    FNFAnchor(
        values=scale_down_events,
        point={"scale": -1.0},
        weights=scale_down_weights,
        groups=scale_down_event_ids,
        name="scale-down",
    ),
    FNFAnchor(
        values=scale_up_events,
        point={"scale": 1.0},
        weights=scale_up_weights,
        groups=scale_up_event_ids,
        name="scale-up",
    ),
    # Axis anchors for resolution and theory, followed by joint anchors
    # in which both scale and resolution are non-nominal.
]

result = FNFTrainer(residual_config, training_config).fit(
    anchors,
    base_torch_log_prob=nominal_density.torch_log_prob,
    features=("x", "y", "z"),
    seed=7,
)
```

The nominal density must remain frozen, but its log-density evaluation must
retain gradients with respect to its input. This permits the residual
transform to learn through the nominal reference-flow and density-ratio
models.

Training minimizes the equal-anchor weighted likelihood

$$
\mathcal L
=
-\frac{1}{|A|}
\sum_{a\in A}
\frac{\sum_{i\in D_a}w_i\log p(x_i\mid\nu_a)}
     {\sum_{i\in D_a}w_i}.
$$

Thus a large variation file does not dominate a smaller one. Training weights
must be finite and nonnegative; a signed Monte Carlo measure is not a
probability density and is rejected. Correlated nominal and varied views
should carry the same `groups` or event identifiers. A stable hash assigns
each group to exactly one of training, validation, and holdout partitions,
also across different anchors.

Every configured nuisance needs at least one non-nominal axis anchor. A
pairwise interaction additionally requires data where both parameters are
non-nominal. Axis-only data cannot identify an explicit cross term, and the
trainer rejects that configuration.

## Density interface

Attach the fitted residual to any nominal density with `log_prob` and `sample`
methods:

```python
density = result.density(
    nominal_density,
    base_torch_log_prob=nominal_density.torch_log_prob,
)

log_density = density.log_prob(events, {"scale": 0.4})
log_ratio = density.log_ratio(events, {"scale": 0.4})
varied_samples = density.sample(
    100_000,
    {"scale": 0.4},
    rng=numpy.random.default_rng(17),
)
```

`log_ratio` returns the log-density ratio to the nominal point unless an
explicit `reference_point` is supplied. `to_nominal` and `from_nominal`
expose both directions of the residual map. The Torch-native
`torch_log_prob` path remains differentiable for likelihood minimization with
automatic derivatives.

The FNF describes a normalized **shape**. Expected selected yields are a
separate model. `LogQuadraticYieldMorph` gives a positive interpolation
through down, nominal, and up yield anchors:

```python
from hnsbi.fnf import LogQuadraticYieldMorph

yield_model = LogQuadraticYieldMorph.from_anchors(
    nominal_yield=1000.0,
    anchors={
        "scale": (970.0, 1040.0),
        "resolution": (990.0, 1015.0),
    },
)
expected = yield_model.expected_yield({"scale": 0.5})
```

For one nuisance the interpolation is

$$
Y(u)=Y_0\exp(au+bu^2),
$$

where $a$ and $b$ are determined exactly by the two positive anchors.
Independent nuisance factors multiply.

## Fits, Asimov samples, and toys

`FNFSystematic` exposes one common runtime contract:
`shape_factor(values, point)` and `yield_factor(point)`. The likelihood,
direct Asimov builder, and toy generator all use the same calculation. On a
finite support with nominal process ratio $\widetilde r_0$, the shape is

$$
\widetilde r_\nu(x_i)
=
\frac{\widetilde r_0(x_i)\,p_\nu(x_i)/p_0(x_i)}
{\sum_j\omega_j\widetilde r_0(x_j)\,p_\nu(x_j)/p_0(x_j)}.
$$

The denominator is a finite-support quadrature correction. It enforces
component normalization for an Asimov or likelihood support and is recorded
as a diagnostic; it does not alter the trained FNF or hide the independent
normalization diagnostic from `diagnose_fnf`.

```python
asimov = project.build_configured_asimov(
    reference=reference_training.training.flow,
    ratios=ratio_artifacts.evaluators,
    normalizer=ratio_artifacts.normalizer,
    fnf_systematics=fnf_systematics,
)

toys = project.generate_configured_toys(
    reference=reference_training.training.flow,
    ratios=ratio_artifacts.evaluators,
    normalizer=ratio_artifacts.normalizer,
    fnf_systematics=fnf_systematics,
)
```

At a non-nominal nuisance point, the FNF yield factor changes each
component's Asimov integral and Poisson expectation, while the FNF shape
factor changes the event distribution. `AsimovResult.fnf_components` and the
event metadata record which models were applied.

`Project.write_configured_workspace()` attaches the configured FNF manifests.
It refuses to serialize a non-nominal FNF point if the supplied Asimov was
built without that FNF. Workspace likelihoods and
`ToyGenerator.from_workspace()` reconstruct every declared FNF from the
checked reference-flow, native-ratio, and residual manifests; portable FNF
workspaces require all three. Explicit runtime overrides are rejected because
their nominal process density cannot be authenticated against the serialized
model. The loaders never fall back to the nominal model.

```python
workspace = project.write_configured_workspace(
    asimov,
    reference_manifest=reference_training.checkpoint_manifest,
    ratio_manifests={
        name: result.manifest_path
        for name, result in ratio_artifacts.training.items()
    },
)
```

FNF evaluation currently runs through Torch or ONNX rather than JAX. Use
`MinuitInference(..., use_jax=False)` for FNF workspaces.

## Several channels and workspaces

The safe default is one FNF for each `(channel, sample)` domain. Channels may
have different selections, feature sets, nominal reference densities, and
responses to a shared nuisance. Equal nuisance names correlate parameter
values across workspaces; they do not require coefficient-network sharing.

Conditional sharing can be added by a higher-level project interface only
when domains have identical feature ordering and preprocessing. The FNF paper
uses class-conditioned density sectors, which motivates this extension, but
does not itself define a multi-workspace statistical likelihood.

Each channel must keep its own normalization support for finite-reference
Asimov and observed-data likelihood evaluation. A finite-support correction
is one scalar per channel and parameter point; it must not be shared between
workspaces or learned into the FNF shape.

## Artifacts and diagnostics

`result.save(directory)` writes:

- a native PyTorch checkpoint;
- a backend-neutral NumPy state archive;
- JSON architecture, nuisance-basis, preprocessing, and provenance metadata;
- a checksum manifest covering all three files.

Loading uses the portable state after checking the manifest. A dependency
bundle at project level should additionally record the hashes of the frozen
nominal reference flow, ratio model, scaler, feature signature, selection,
and normalization constant.

`diagnose_fnf` reports:

- exact nominal identity and zero nominal log determinant;
- forward/inverse and log-determinant closure at each requested point;
- raw normalization from $E_{p_0}[p_\nu/p_0]$, including its Monte Carlo
  standard error and importance-sampling effective sample size;
- optional analytic-versus-autograd Jacobian agreement.

```python
from hnsbi.fnf import diagnose_fnf

report = diagnose_fnf(
    density,
    nominal_validation_samples,
    points=(
        {"scale": -1.0},
        {"scale": -0.5},
        {"scale": 0.5},
        {"scale": 1.0},
    ),
    check_jacobian=True,
)
```

The normalization diagnostic deliberately uses the uncorrected density.
Applying a finite-support likelihood correction here would hide genuine
normalization failures. Physics validation should additionally compare
weighted distributions and excess negative log likelihood at every input
anchor, unseen interpolation points, and held-out joint nuisance points.

The mathematical design and parts of the architecture are adapted from the
[reference FNF implementation](https://github.com/valsdav/factorizable-normalizing-flow),
distributed under the BSD 3-Clause license.
