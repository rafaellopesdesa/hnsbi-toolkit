# Examples

The examples are organized by scientific workflow rather than by exercise
number. YAML is the primary human interface:

- `lhc_analysis/analysis.yaml` drives the full frequentist analysis;
- `dingo_bbh/dual.yaml` and `dingo_bns/dual.yaml` drive the dual Bayesian
  examples.

The JSON files in `configs/` demonstrate the equivalent machine
serialization:

- `frequentist_complete.json` covers native reference-flow and ratio training,
  a JSON workspace, toys, a normalized weighted Asimov sample, NIS, and
  up/down shape variations.
- `dual_complete.json` describes the five frozen objects in the dual
  hNPE--hNDE construction using independent data drawn under the
  $\rho$, $\nu$, and $\kappa$ designs.

Paths are resolved relative to the configuration file by workflow runners.
For a Python dictionary backed by an in-memory `pyarrow.Table` or
`awkward.Array`, use a `registry_key` in place of a path and register the
object before starting the workflow.

Colab-ready notebooks are available in `notebooks/`:

- `hybrid_reference_flow_and_density_ratios.ipynb`
- `neural_importance_sampling_asimov.ipynb`
- `dual_hnpe_hnde.ipynb`
- `sbibm_slcp_benchmark.ipynb`
- `sbibm_two_moons_benchmark.ipynb`

The two frequentist notebooks now use the self-contained native runtime. The
Bayesian benchmark notebooks preserve the paper workflows.

Additional examples are:

- `lhc_analysis/`: full YAML-driven synthetic analysis with three
  systematics, pulls, impacts, and pyhf CLs limits;
- `dingo_bbh/` and `dingo_bns/`: reduced synthetic DINGO-inspired, opt-in
  end-to-end dual hNPE--hNDE studies with exact likelihood oracles.

The full paper settings remain the reproducibility target and are intentionally
too expensive for pull-request CI. Package tests use separate synthetic smoke
fixtures; they do not rewrite the seeds, model definitions, or numerical
settings stored in these notebooks.
