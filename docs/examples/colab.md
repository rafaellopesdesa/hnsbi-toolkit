# Colab examples

The paper workflows remain available as descriptive notebooks:

| Notebook | Source exercise | Purpose |
|---|---|---|
| `hybrid_reference_flow_and_density_ratios.ipynb` | Exercise 5 | reference flow, ratios, diagnostics, fits, toys |
| `neural_importance_sampling_asimov.ipynb` | Exercise 6 | defensive NIS and efficient Asimov |
| `dual_hnpe_hnde.ipynb` | Exercise 9 | five dual artifacts and reusable calculations |
| `sbibm_slcp_benchmark.ipynb` | Exercise 10 | SLCP benchmark |
| `sbibm_two_moons_benchmark.ipynb` | Exercise 10 Two Moons | multimodal benchmark |

The notebooks and their required `utils_*` helpers are preserved from
`rafaellopesdesa/nsbi-lhc-toolkit` commit
`e1249eb90e78b9fcbf24bf39cb9575fa3b621785`, under
`workshops/ml4hep_tifr_colab/`. Only the descriptive filenames, Colab badges,
environment bootstrap cells, and schema-only notebook metadata were adapted.
They intentionally retain the paper code and stored output payloads rather
than pretending to be package-API migrations.

Each checked-in notebook displays a Google Colab badge pointing to its actual
repository path. The setup cell installs the toolkit and preserved helper
dependencies. With `USE_DRIVE = True`, the historical output directory remains:

`MyDrive/Colab Notebooks/ml4hep_tifr_colab/nsbi-lhc-toolkit/workshops/ml4hep_tifr`

Full paper runs are too expensive for pull-request CI. CI validates the package
with separate compact tests; it does not claim to execute every archival
notebook or reproduce its complete numerical result on every change.

`sbibm==1.1.0` has historical algorithm dependencies that conflict with newer
scientific stacks. The benchmark environment installs only the task package
and the explicitly required runtime dependencies; SBIBM is not part of the
toolkit core.
