# Paper notebooks

These are the notebooks used to develop and validate the methods in the paper.
Their scientific cells and configurations are preserved. The filenames,
Colab badges, environment bootstrap cells, and schema-only notebook metadata
repairs were adapted for
`rafaellopesdesa/hnsbi-toolkit`; scientific code was not rewritten. Outputs
were cleared when the workspace moved so the notebooks can be rerun cleanly.

The notebooks are archival, end-to-end reproductions. For package-native
starting points, use the compact configurations in
[`../configs`](../configs/) and the workflows in the main documentation.

| Notebook | Preserved source notebook | Purpose |
| --- | --- | --- |
| [`hybrid_reference_flow_and_density_ratios.ipynb`](hybrid_reference_flow_and_density_ratios.ipynb) | `Exercise_5_Hybrid_NormalizingFlow_DensityRatio.ipynb` | Reference normalizing flow, component-to-reference ratios, and frequentist diagnostics |
| [`neural_importance_sampling_asimov.ipynb`](neural_importance_sampling_asimov.ipynb) | `Exercise_6_NeuralImportanceSampling_Asimov.ipynb` | Neural importance sampling and efficient Asimov construction |
| [`dual_hnpe_hnde.ipynb`](dual_hnpe_hnde.ipynb) | `Exercise_9_Hybrid_NPE_NDE.ipynb` | Dual hNPE–hNDE inference, diagnostics, consensus weights, and prior updates |
| [`sbibm_slcp_benchmark.ipynb`](sbibm_slcp_benchmark.ipynb) | `Exercise_10_SBIBM_Hybrid_Benchmark.ipynb` | `sbibm` SLCP benchmark |
| [`sbibm_two_moons_benchmark.ipynb`](sbibm_two_moons_benchmark.ipynb) | `Exercise_10_SBIBM_Hybrid_Benchmark_TwoMoons.ipynb` | `sbibm` Two Moons benchmark |

## Upstream library

The frequentist examples install the canonical
[`iris-hep/nsbi-lhc-toolkit`](https://github.com/iris-hep/nsbi-lhc-toolkit/tree/main)
`main` branch through `hnsbi-toolkit[lhc]` and import
`nsbi_common_utils` as a library. The upstream repository is not cloned into
the notebook workspace, and no fork is used as a runtime dependency.

The retained helper dependency closure is:

`generate_distributions.py`, `utils.py`, `utils_benchmark.py`,
`utils_distributions.py`, `utils_dual_hnde.py`, `utils_hnpe.py`, `utils_nf.py`,
and `utils_plotting.py`. The metadata-only repairs add a missing cell ID and
remove invalid `metadata` keys from stream outputs so all files pass strict
`nbformat` validation.

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
