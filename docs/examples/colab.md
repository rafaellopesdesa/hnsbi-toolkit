# Colab examples

The paper workflows remain available as descriptive notebooks:

| Notebook | Source exercise | Purpose |
|---|---|---|
| `hybrid_reference_flow_and_density_ratios.ipynb` | Exercise 5 | reference flow, ratios, diagnostics, fits, toys |
| `neural_importance_sampling_asimov.ipynb` | Exercise 6 | defensive NIS and efficient Asimov |
| `dual_hnpe_hnde.ipynb` | Exercise 9 | five dual artifacts and reusable calculations |
| `sbibm_slcp_benchmark.ipynb` | Exercise 10 | SLCP benchmark |
| `sbibm_two_moons_benchmark.ipynb` | Exercise 10 Two Moons | multimodal benchmark |

The notebooks and their required `utils_*` helpers preserve the scientific
cells developed for the paper exercises. The descriptive filenames, Colab
badges, environment bootstrap cells, and schema-only notebook metadata were
adapted for this repository. Outputs were cleared when the workspace moved so
the notebooks can be rerun cleanly.

The frequentist setup cells install
[`iris-hep/nsbi-lhc-toolkit@main`](https://github.com/iris-hep/nsbi-lhc-toolkit/tree/main)
through the hNSBI `lhc` extra and import `nsbi_common_utils` as a library. They
do not clone or import code from a fork.

Each checked-in notebook displays a Google Colab badge pointing to its actual
repository path. The setup cell installs the toolkit and preserved helper
dependencies. With `USE_DRIVE = True`, source checkouts and generated
artifacts live under:

`MyDrive/hsbi-toolkit`

The hNSBI source checkout is in its `hnsbi-toolkit/` child directory, while
generated data, checkpoints, and plots are shared through `paper-examples/`.

Full paper runs are too expensive for pull-request CI. CI validates the package
with separate compact tests; it does not claim to execute every paper
notebook or reproduce its complete numerical result on every change.

`sbibm==1.1.0` has historical algorithm dependencies that conflict with newer
scientific stacks. The benchmark environment installs only the task package
and the explicitly required runtime dependencies; SBIBM is not part of the
toolkit core.
