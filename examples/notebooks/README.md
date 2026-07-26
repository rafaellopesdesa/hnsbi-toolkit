# Paper notebooks

These are the executed notebooks used to develop and validate the methods in
the paper. Their scientific cells, configurations, and stored results are
preserved. Only the filenames, Colab badges, environment bootstrap cells, and
schema-only notebook metadata repairs were adapted for
`rafaellopesdesa/hnsbi-toolkit`; scientific code and stored output payloads
were not rewritten.

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

## Provenance

All five notebooks and their required legacy helpers were copied from:

- repository: `rafaellopesdesa/nsbi-lhc-toolkit`
- commit: `e1249eb90e78b9fcbf24bf39cb9575fa3b621785`
- path: `workshops/ml4hep_tifr_colab/`

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

With `USE_DRIVE = True`, generated data, checkpoints, and plots continue to use
the paper notebooks' backward-compatible Drive workspace:

`MyDrive/Colab Notebooks/ml4hep_tifr_colab/nsbi-lhc-toolkit/workshops/ml4hep_tifr`

The paper profiles are intentionally computationally expensive. Existing
artifacts in that workspace are reused when the corresponding
`LOAD_IF_AVAILABLE` setting is enabled.
