# Reduced synthetic BNS example

This is a small, self-contained gravitational-wave-shaped problem for the
dual hNPE--hNDE workflow. It is **inspired by** the public DINGO-BNS GW170817
example. It does not run DINGO, reproduce the GW170817 analysis, or provide a
waveform or detector model suitable for gravitational-wave analysis.

The example fixes the chirp mass to the DINGO-BNS demonstration value
$1.19786\,M_\odot$ and infers five parameters,

$$
\theta=(q,\tilde\Lambda,\phi_c,t_c,d_L).
$$

Six complex frequency coefficients are observed through three fixed toy
detector responses labelled H1, L1, and V1. The source phase contains
point-particle terms through 2PN and a leading effective tidal term. Every
real and imaginary coefficient receives independent unit Gaussian noise, so
the exact reduced likelihood is

$$
\log p(x\mid\theta)
=-\frac{1}{2}\lVert x-h(\theta)\rVert^2
-\frac{D}{2}\log(2\pi).
$$

`generate_data.py` exposes this reference as `exact_log_likelihood()`. The
learned hNDE must infer the likelihood from simulations; the exact function is
only a validation oracle.

[`dingo_bns_dual.ipynb`](dingo_bns_dual.ipynb) is a complete opt-in Colab
workflow. Its single `RUN_FULL_WORKFLOW` switch selects either a fast smoke
check or quick-profile generation, real five-stage training, verified ONNX
reload, both posterior routes, dual consensus, normalization/ESS/bridge
diagnostics, and comparison with the exact reduced likelihood.

## Run

From the repository root:

```bash
python -m pip install -e ".[bayes]"
python examples/dingo_bns/generate_data.py \
  --output-dir examples/dingo_bns/data \
  --profile quick
hnsbi validate-config examples/dingo_bns/dual.yaml
```

Inspect the YAML-first handoff without starting training:

```python
from hnsbi import Project

project = Project.load("examples/dingo_bns/dual.yaml")
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
- coverage and simulation-based calibration on independent simulator draws;
- the $\tilde\Lambda$ marginal against numerical integration on fixed
  low-dimensional slices.

The configured design is uniform in distance. A volume prior proportional to
$d_L^2$ can be applied afterward with `update_posterior_weights()` to
demonstrate prior updating without retraining.

## Relationship to DINGO-BNS

DINGO-BNS performs amortized inference for real compact-binary data with a
frequency-domain waveform approximant, detector noise PSD conditioning,
additional intrinsic and extrinsic parameters, and importance-sampling
validation. The official GW170817 example uses real strain and PSD data,
pretrained networks, and production waveform software. This example replaces
all those expensive ingredients with a reduced analytic simulator so both
sides of the dual construction can be trained in a Colab-scale session.

Primary sources:

- M. Dax et al.,
  [“Real-time gravitational-wave science with neural posterior
  estimation”](https://arxiv.org/abs/2106.12594).
- M. Dax et al.,
  [“Neural Importance Sampling for Rapid and Reliable Gravitational-Wave
  Inference”](https://arxiv.org/abs/2210.05686).
- A. Kolmus et al.,
  [“DINGO-BNS: Fast neural Bayesian inference for binary neutron star
  mergers”](https://arxiv.org/abs/2407.09602).
- The official
  [DINGO-BNS GW170817 demonstration](https://github.com/dingo-gw/binary-neutron-star-demo/tree/e87244b8715c6a881056beb597311a0734746210/GW170817).

DINGO and the DINGO-BNS demonstration are MIT-licensed. No DINGO source code,
pretrained model, detector data, or waveform asset is copied or redistributed
here.
