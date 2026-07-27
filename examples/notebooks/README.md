# Paper notebooks

These Colab-ready notebooks cover the paper methods. The two frequentist
workflows were rewritten to use the self-contained native hNSBI runtime; the
three Bayesian notebooks retain the established paper calculations.

The three Bayesian notebooks are archival reproductions of the retained paper
calculations; the two frequentist notebooks are package-native replacements
for their source exercises. For additional package-native starting points,
use the compact configurations in
[`../configs`](../configs/) and the workflows in the main documentation.

| Notebook | Source notebook | Purpose |
| --- | --- | --- |
| [`hybrid_reference_flow_and_density_ratios.ipynb`](hybrid_reference_flow_and_density_ratios.ipynb) | `Exercise_5_Hybrid_NormalizingFlow_DensityRatio.ipynb` | Native reference flow, ratio diagnostics, ONNX, workspace, and Minuit fit |
| [`neural_importance_sampling_asimov.ipynb`](neural_importance_sampling_asimov.ipynb) | `Exercise_6_NeuralImportanceSampling_Asimov.ipynb` | Native NIS and efficient Asimov construction |
| [`dual_hnpe_hnde.ipynb`](dual_hnpe_hnde.ipynb) | `Exercise_9_Hybrid_NPE_NDE.ipynb` | Dual hNPE–hNDE inference, diagnostics, consensus weights, and prior updates |
| [`sbibm_slcp_benchmark.ipynb`](sbibm_slcp_benchmark.ipynb) | `Exercise_10_SBIBM_Hybrid_Benchmark.ipynb` | `sbibm` SLCP benchmark |
| [`sbibm_two_moons_benchmark.ipynb`](sbibm_two_moons_benchmark.ipynb) | `Exercise_10_SBIBM_Hybrid_Benchmark_TwoMoons.ipynb` | `sbibm` Two Moons benchmark |

## Self-contained runtime

The frequentist notebooks install only `hnsbi-toolkit[lhc,flows]`. They do not
clone or import another NSBI package. Algorithms adapted from
`nsbi-lhc-toolkit` are acknowledged in the repository NOTICE and
documentation.

The retained Bayesian helper closure is:

`generate_distributions.py`, `utils.py`, `utils_benchmark.py`,
`utils_distributions.py`, `utils_dual_hnde.py`, `utils_hnpe.py`, `utils_nf.py`,
and `utils_plotting.py`. Notebook metadata and cell IDs are normalized so all
files pass strict `nbformat` validation.

## Running in Colab

Open a notebook and use its **Open in Colab** badge. The setup cell checks out
the `main` branch of `hnsbi-toolkit`, installs the package and required extras,
and imports the preserved helpers from this directory.

With `USE_DRIVE = True`, generated data, checkpoints, and plots use the shared
Drive workspace:

`MyDrive/hsbi-toolkit/paper-examples`

The paper profiles are intentionally computationally expensive. After the
first run, artifacts in that workspace are reused when the corresponding
`LOAD_IF_AVAILABLE` setting is enabled.
