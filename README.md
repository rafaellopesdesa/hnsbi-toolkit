# hnsbi-toolkit

`hnsbi-toolkit` is a self-contained implementation of hybrid neural
density estimation (hNDE) and dual hNPE--hNDE workflows.

The frequentist toolkit provides:

- RealNVP and rational-quadratic-spline reference flows;
- native weighted density-ratio ensembles, calibration, and diagnostics;
- ONNX export with preprocessing embedded in each deployable graph;
- JSON workspaces for formula-based unbinned intensity models;
- JAX automatic differentiation, iminuit MIGRAD/HESSE fits, profile scans,
  and test-statistic scans;
- arbitrary-parameter toys, direct Asimov samples, and defensive
  neural-importance-sampling (NIS) Asimov samples;
- up/down ratio systematics and multi-nuisance factorizable normalizing flows
  (FNFs);
- pull plots, global-observable and covariance impacts, and pyhf CLs limits
  with asymptotic or toy-based calculators.

The Bayesian toolkit trains and freezes the five objects in the dual
construction,

$$
\{q_\phi,\widehat r_{\rm P},q_\eta,\widehat r_{\rm C},
\widehat Z_{\rm C}\},
$$

then exposes posterior and likelihood routes, dual-consensus weights,
nuisance-prior and auxiliary-likelihood updates, evidence,
posterior-predictive generation, and selection integrals.

## Installation

Until a package release is published, install from GitHub:

```bash
python -m pip install \
  "hnsbi-toolkit[lhc,flows] @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
```

Use the `bayes` extra for native dual training, or install every development
capability from a source checkout:

```bash
python -m pip install -e ".[data,flows,lhc,bayes,plots,test,docs]"
```

`hnsbi-toolkit` does **not** install, import, or clone
`nsbi-lhc-toolkit`. The native implementation incorporates and extends
scientific and software patterns developed there; provenance and licenses are
recorded in [NOTICE](NOTICE).

## YAML-first workflows

YAML is the primary human interface. The full LHC example supplies every
nominal sample and detector/theory variation in one file:

```python
from hnsbi import Project

project = Project.load("examples/lhc_analysis/analysis.yaml")
reference = project.train_reference()
ratios = project.train_ratios(reference=reference.training.flow)
```

The Bayesian interface uses the same loader. Its YAML names the samples drawn
under the $\rho$, $\nu$, and $\kappa$ designs:

```python
project = Project.load("examples/dingo_bbh/dual.yaml")
dual = project.train_dual()
```

JSON remains the canonical machine serialization for configurations,
manifests, diagnostics, and workspaces:

```python
config = project.config
config.dump_json("resolved-analysis.json")
```

See the [documentation](https://hnsbi-toolkit.readthedocs.io), the
[complete LHC example](examples/lhc_analysis/), and the
[DINGO-inspired BBH and BNS examples](docs/examples/dingo.md).

## Scientific scope

Learned surrogates remain approximate. Independent closure, normalization,
effective-sample-size, calibration, tail, route-agreement, and ONNX-parity
checks are part of the inference contract, not optional post-processing.
