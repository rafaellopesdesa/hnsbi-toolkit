# Colab examples

The paper workflows remain available as descriptive notebooks:

| Notebook | Source exercise | Purpose |
|---|---|---|
| `hybrid_reference_flow_and_density_ratios.ipynb` | Exercise 5 | reference flow, ratios, diagnostics, workspace, and fit |
| `neural_importance_sampling_asimov.ipynb` | Exercise 6 | defensive NIS and efficient Asimov |
| `dual_hnpe_hnde.ipynb` | Exercise 9 | five dual artifacts and reusable calculations |
| `sbibm_slcp_benchmark.ipynb` | Exercise 10 | SLCP benchmark |
| `sbibm_two_moons_benchmark.ipynb` | Exercise 10 Two Moons | multimodal benchmark |

The three Bayesian notebooks and their required `utils_*` helpers retain the
established paper calculations. The two frequentist notebooks are compact
native rewrites of Exercises 5 and 6. Descriptive filenames, Colab badges,
environment bootstrap cells, and notebook metadata were adapted for this
repository. Outputs were cleared when the workspace moved so the notebooks
can be rerun cleanly.

The frequentist setup cells install `hnsbi-toolkit[lhc,flows]` and use its
self-contained native ratio, workspace, and inference stack. They do not
clone or import another NSBI package.

The frequentist generator retains raw parquets and creates `_presel.parquet`
inputs before either notebook begins its method-specific work. It preserves
the legacy disjoint 50% / 46% / 4% selector, downstream-training, and
evaluation partition; derives one nominal $p_S/p_B$ cut; applies that fixed
cut to every variation; and constructs an equal signal/background reference
after selection. Selected weight sums are the corresponding post-selection
expected yields.

The historical learned PRESEL checkpoint is unavailable, so the controlled
Gaussian-mixture example uses its known reconstructed densities as a
deterministic, legacy-equivalent analytic selector. It does not claim to
reproduce that learned network. A `preselection.manifest.json` binds the
selector, cut, requested counts, yields, and checksums of all raw and selected
files; the setup reuses persistent inputs only after this manifest verifies.
The notebooks therefore remain focused on hNSBI rather than repeating the
preselection exercise.

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
