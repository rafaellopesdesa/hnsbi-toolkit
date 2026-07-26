# Examples

The examples are organized by scientific workflow rather than by exercise
number. The complete configurations in `configs/` are intended as readable
templates:

- `frequentist_complete.json` covers reference-flow training, density ratios,
  an `nsbi-common-utils` workspace, toys, a normalized weighted Asimov sample,
  NIS, and up/down shape variations.
- `dual_complete.json` describes the five frozen objects in the dual
  hNPE--hNDE construction using independent data drawn under the
  \(\rho\), \(\nu\), and \(\kappa\) designs.

Paths are resolved relative to the configuration file by workflow runners.
For a Python dictionary backed by an in-memory `pyarrow.Table` or
`awkward.Array`, use a `registry_key` in place of a path and register the
object before starting the workflow.

The preserved paper notebooks are available in `notebooks/` with descriptive
names and Google Colab badges:

- `hybrid_reference_flow_and_density_ratios.ipynb`
- `neural_importance_sampling_asimov.ipynb`
- `dual_hnpe_hnde.ipynb`
- `sbibm_slcp_benchmark.ipynb`
- `sbibm_two_moons_benchmark.ipynb`

These are archival, end-to-end reproductions from the paper development
repository. Their scientific cells, stored outputs, and legacy `utils_*`
helper dependency closure are intentionally preserved. Only filenames, Colab
badges, environment bootstrap cells, and schema-only notebook metadata were
adapted for this repository; the notebooks are not migrations to the package
API.

The full paper settings remain the reproducibility target and are intentionally
too expensive for pull-request CI. Package tests use separate synthetic smoke
fixtures; they do not rewrite the seeds, model definitions, or numerical
settings stored in these notebooks.
