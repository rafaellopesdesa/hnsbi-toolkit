# Reduced synthetic BBH example

This is a small, self-contained gravitational-wave-shaped problem for the
dual hNPE--hNDE workflow. It is **inspired by** DINGO's early five-parameter
binary-black-hole example and its neural-importance-sampling work. It does not
run DINGO, reproduce a DINGO result, or provide a waveform model suitable for
gravitational-wave analysis.

The example uses five parameters,

$$
\theta=({\cal M},q,\phi_c,t_c,d_L),
$$

and twelve complex, whitened frequency coefficients. The deterministic mean
contains Newtonian and 1PN non-spinning inspiral phase terms. Every real and
imaginary coefficient receives independent unit Gaussian noise. Consequently,
the reduced likelihood is known exactly:

$$
\log p(x\mid\theta)
=-\frac{1}{2}\lVert x-h(\theta)\rVert^2
 -\frac{D}{2}\log(2\pi).
$$

`generate_data.py` exposes this reference as `exact_log_likelihood()`. The
learned hNDE must still infer the likelihood from simulations; the exact
function is only a validation oracle.

[`dingo_bbh_dual.ipynb`](dingo_bbh_dual.ipynb) is a complete opt-in Colab
workflow. Its single `RUN_FULL_WORKFLOW` switch selects either a fast smoke
check or quick-profile generation, real five-stage training, verified ONNX
reload, both posterior routes, dual consensus, normalization/ESS/bridge
diagnostics, and comparison with the exact reduced likelihood.

## Run

From the repository root:

```bash
python -m pip install -e ".[bayes]"
python examples/dingo_bbh/generate_data.py \
  --output-dir examples/dingo_bbh/data \
  --profile quick
hnsbi validate-config examples/dingo_bbh/dual.yaml
```

Inspect the YAML-first handoff without starting training:

```python
from hnsbi import Project

project = Project.load("examples/dingo_bbh/dual.yaml")
training_data = project.dual_training_data()
print(training_data.rho_flow.theta.shape)
print(training_data.nu_flow.observation.shape)
```

Train all five dual artifacts with:

```python
artifacts = project.train_dual()
print(artifacts.manifest_path)
```

The notebook then reloads that manifest with `load_dual_model()`. It evaluates
`hnpe_log_weights()` and `hnde_log_weights()` on the same proposal points,
forms `geometric_consensus()`, and reports route-specific and dual ESS.
`posterior_normalization_diagnostic()`,
`conditional_normalization_diagnostic()`, `route_diagnostic()`, and
`bridge_diagnostic()` expose complementary normalization and agreement
checks.

Because this reduced simulator has an exact normalized Gaussian likelihood,
the notebook also constructs exact posterior importance weights on those same
points. It compares learned and exact log likelihoods, posterior means, weight
affinities, and all five one-dimensional marginals.

The default `quick` profile writes 12,000 rows for each of the four independent
training sources and 2,000 additional validation rows. `smoke` is intended only
for software checks. `publication` is a higher-statistics starting point, not
a promise of publication-quality closure for every random seed.

## What to validate

- held-out simulator/reference-flow closure for $q_\eta(x\mid\theta)$;
- `posterior_normalization_diagnostic()` near one within Monte Carlo error;
- corrected conditional normalization from
  `conditional_normalization_diagnostic()`;
- hNPE/hNDE agreement through `route_diagnostic()` and
  `bridge_diagnostic()`;
- raw NPE, hNPE, hNDE, and dual posterior samples against the exact reduced
  likelihood;
- importance ESS and dominant-weight fractions;
- evidence against numerical integration on a fixed low-dimensional slice;
- simulation-based calibration on independent simulator draws.

The configured design is uniform in distance. An analysis prior proportional
to $d_L^2$ can be applied afterward with
`update_posterior_weights()` to demonstrate a nuisance/prior update without
retraining.

## Relationship to DINGO

The original five-dimensional demonstration used
$(m_1,m_2,\phi_c,t_c,d_L)$, IMRPhenomPv2 waveforms, one-second time series, a
fixed Advanced-LIGO PSD, and a million waveform/parameter pairs. DINGO's
current toy tutorial uses IMRPhenomD, H1 and L1, and eleven inferred
parameters. This example intentionally replaces those expensive ingredients
with a reduced analytic simulator so both sides of the dual construction can
be trained in a Colab-scale session.

Primary sources:

- S. R. Green, C. Simpson, and J. Gair,
  [“Gravitational-wave parameter estimation with autoregressive neural
  network flows”](https://arxiv.org/abs/2002.07656).
- M. Dax et al.,
  [“Real-time gravitational-wave science with neural posterior
  estimation”](https://arxiv.org/abs/2106.12594).
- M. Dax et al.,
  [“Neural Importance Sampling for Rapid and Reliable Gravitational-Wave
  Inference”](https://arxiv.org/abs/2210.05686).
- The official
  [DINGO toy tutorial](https://github.com/dingo-gw/dingo/blob/a608acebd22cb1150478fcd6f2f97e9651457745/docs/source/example_toy_npe_model.md)
  and
  [settings](https://github.com/dingo-gw/dingo/tree/a608acebd22cb1150478fcd6f2f97e9651457745/examples/toy_npe_model).

DINGO is MIT-licensed. No DINGO source code, pretrained model, detector data,
or waveform asset is copied or redistributed here.
