# DINGO-inspired dual examples

Two self-contained examples adapt ideas from the DINGO gravitational-wave
inference literature to the dual hNPE--hNDE interface. They are deliberately
small synthetic benchmarks: neither example runs DINGO, analyzes detector
data, nor reproduces a published DINGO result.

| Example | Inferred parameters | Reduced observation | Exact validation |
| --- | --- | --- | --- |
| `examples/dingo_bbh` | $({\cal M},q,\phi_c,t_c,d_L)$ | 12 complex whitened frequency coefficients | Normalized Gaussian likelihood |
| `examples/dingo_bns` | $(q,\tilde\Lambda,\phi_c,t_c,d_L)$ at fixed chirp mass | 6 complex coefficients in each of 3 toy detectors | Normalized Gaussian likelihood |

This reduction is intentional. Production DINGO conditions on high-dimensional
strain and PSD information and uses production waveform software. The current
dual toolkit presents the configured observation columns directly to
$q_\eta(x\mid\theta)$, so these examples use a low-dimensional observation
that can be trained and independently checked in a modest Colab session.

## YAML-first workflow

Each example contains:

- `generate_data.py`, a deterministic NumPy simulator and Parquet writer;
- `dual.yaml`, the complete dual hNPE--hNDE training configuration;
- a complete opt-in Colab workflow for training, verified model reload,
  inference, diagnostics, and exact-likelihood comparison;
- `README.md`, assumptions, commands, validation targets, and limitations.

Generate either benchmark from the repository root:

```bash
python -m pip install -e ".[bayes]"
python examples/dingo_bbh/generate_data.py \
  --output-dir examples/dingo_bbh/data --profile quick
hnsbi validate-config examples/dingo_bbh/dual.yaml
```

Replace `dingo_bbh` with `dingo_bns` for the tidal example. The generator
creates physically disjoint `rho`, `rho_residual`, `nu`, `kappa`, and
independent-validation files. The four training sources each contain explicit
`train`, `validation`, and `holdout` rows. The YAML can then be handed directly
to the public API:

```python
from hnsbi import Project

project = Project.load("examples/dingo_bbh/dual.yaml")
training_data = project.dual_training_data()
artifacts = project.train_dual()
```

Use `--profile smoke` to check data/configuration plumbing without training.
The `quick` and `publication` sizes are starting points; they do not waive the
normalization, route-agreement, coverage, or importance-weight diagnostics.

## Notebook execution switch

Each notebook has one switch:

```python
RUN_FULL_WORKFLOW = False
```

The default `False` path generates the smoke profile and validates the YAML
handoff in seconds. Set it to `True` and run all cells to:

1. generate the quick independent $\rho$, $\nu$, and $\kappa$ datasets;
2. call `Project.train_dual()` for the real five-stage training workflow;
3. verify and reload the saved ONNX manifest with `load_dual_model()`;
4. evaluate hNPE and hNDE weights on the same proposal points;
5. form the normalized geometric dual consensus;
6. report posterior and conditional normalization, ESS/tails,
   route agreement, evidence, and bridge diagnostics;
7. compare learned likelihoods and all three posterior results with the exact
   reduced Gaussian benchmark.

The notebook reports numerical diagnostics without imposing universal pass
thresholds. The quick profile is an execution demonstration; scientific
claims require higher statistics, independent seeds, and problem-specific
coverage criteria.

## Independent likelihood oracle

Both generators expose:

```python
waveform_mean(theta)
exact_log_likelihood(theta, observation)
```

For $D$ real-valued, unit-whitened observation components, the oracle is

$$
\log p(x\mid\theta)
=-\frac{1}{2}\left[
  \lVert x-h(\theta)\rVert^2+D\log(2\pi)
\right].
$$

It is not used as a training target. It supports posterior-slice comparisons,
normalization checks, simulation-based calibration, and tests of hNPE/hNDE
route agreement after training.

## Scientific scope and provenance

The BBH benchmark is motivated by the original five-parameter neural-flow
demonstration and later DINGO/DINGO-IS work:

- [Green, Simpson, and Gair (2020)](https://arxiv.org/abs/2002.07656)
- [Dax et al. (2021)](https://arxiv.org/abs/2106.12594)
- [Dax et al. (2022), DINGO-IS](https://arxiv.org/abs/2210.05686)
- [Official DINGO toy tutorial](https://github.com/dingo-gw/dingo/blob/a608acebd22cb1150478fcd6f2f97e9651457745/docs/source/example_toy_npe_model.md)

The BNS benchmark also follows the fixed-chirp-mass conditioning idea in the
public GW170817 demonstration:

- [Kolmus et al. (2024), DINGO-BNS](https://arxiv.org/abs/2407.09602)
- [Official DINGO-BNS GW170817 example](https://github.com/dingo-gw/binary-neutron-star-demo/tree/e87244b8715c6a881056beb597311a0734746210/GW170817)

DINGO and the binary-neutron-star demonstration are MIT-licensed. These
examples copy no DINGO source, pretrained model, waveform, PSD, or detector
asset. Their analytic phase expressions are pedagogical surrogates and must
not be interpreted as production gravitational-wave likelihoods.
