# hnsbi-toolkit

`hnsbi-toolkit` turns the hybrid neural density-estimation (hNDE) and dual
hNPE--hNDE constructions into reusable software.

The frequentist workflow combines a normalized reference flow with the
classifier density-ratio training and statistical-model machinery from
[`nsbi-lhc-toolkit`](https://github.com/iris-hep/nsbi-lhc-toolkit).
It provides:

- RealNVP and rational-quadratic-spline reference flows;
- ONNX artifact bundles for neural networks and preprocessing;
- flow-specific closure diagnostics, while delegating ratio diagnostics to
  `nsbi-lhc-toolkit`;
- intensity models, unbinned workspace export, fits and profile-scan adapters;
- pseudo-experiments at arbitrary parameter points;
- reference-normalized weighted Asimov samples with effective sample size;
- defensive neural-importance-sampling (NIS) Asimov samples;
- normalized shape-systematic interfaces.

The Bayesian workflow includes a native, configuration-driven trainer for the
five frozen objects of the dual construction,

$$
\{q_\phi,\widehat r_{\rm P},q_\eta,\widehat r_{\rm C},
\widehat Z_{\rm C}\},
$$

and a portable, checksummed ONNX manifest. Its inference API implements the
posterior and likelihood routes, dual consensus weights, nuisance-prior and
auxiliary-likelihood updates, evidence, posterior-predictive generation, and
selection integrals.

## Installation

Until a package release is published, install the small NumPy-only core
directly from this repository:

```bash
python -m pip install \
  "hnsbi-toolkit @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
```

The `lhc` extra installs the canonical upstream `main` branch as the
`nsbi_common_utils` library:

```bash
python -m pip install \
  "hnsbi-toolkit[lhc] @ git+https://github.com/rafaellopesdesa/hnsbi-toolkit.git@main"
```

For native dual hNPE--hNDE training, install
the `bayes` extra from the same Git source. A source checkout can install all
development capabilities with:

```bash
python -m pip install -e ".[data,flows,lhc,bayes,plots,test,docs]"
```

`nsbi-lhc-toolkit` is used as a normal runtime library; it is neither cloned
by user code nor vendored into this repository. The equivalent direct
requirement is recorded in `requirements/lhc-upstream.txt` for CI and
environment tooling.

## Minimal configuration

Both JSON files and Python dictionaries are accepted:

```python
from hnsbi import Project, ToolkitConfig

config = ToolkitConfig.load("analysis.json")
project = Project.load("analysis.json")
intensity = project.intensity_model()
```

`Project.train_reference()`, `Project.train_ratios()`,
`Project.train_nis_asimov()`, and `Project.train_dual()` provide the
configuration-first training paths; the lower-level classes remain available
for custom workflows.
See `examples/configs/` and the
[documentation](https://hnsbi-toolkit.readthedocs.io) for complete
frequentist and Bayesian configurations. The paper notebooks are preserved
under `examples/notebooks/` with their legacy `utils_*` helper closure and
Google Colab entry points. Their setup cells install the canonical upstream
library through the `lhc` extra.

## Status

This is the initial source release. Learned surrogates remain approximate:
independent simulator closure, normalization, effective-sample-size, and
ratio-tail diagnostics are part of the intended workflow rather than optional
post-processing.
