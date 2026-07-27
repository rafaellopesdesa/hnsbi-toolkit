# Complete LHC analysis

`examples/lhc_analysis/analysis.yaml` is the primary end-to-end frequentist
example. Its generator follows the established correlated Gaussian-mixture
exercise and adds three nuisance families:

| Nuisance | Variation |
|---|---|
| response | detector response/scale shifted up and down |
| resolution | event-level Gaussian residual widened or narrowed |
| theory | centers of the narrow signal Gaussians shifted in latent $z$ |

Run the full workflow from the repository root:

```bash
python -m pip install -e ".[lhc,flows]"
python examples/lhc_analysis/generate_distributions.py \
  --output examples/lhc_analysis/data
hnsbi validate-config examples/lhc_analysis/analysis.yaml
python examples/lhc_analysis/run_analysis.py \
  examples/lhc_analysis/analysis.yaml
```

The runner trains the reference flow and native ratio ensembles, produces
flow and ratio diagnostics, builds a normalized Asimov sample and JSON
workspace, fits with JAX derivatives and iminuit, writes pull and impact
plots, projects the templates into a pyhf HistFactory model, and calculates
asymptotic CLs exclusions.

The default runs native unbinned fitting/impacts and the asymptotic pyhf
limit. Optional expensive stages are controlled separately:

| Flag | Additional work |
|---|---|
| `--pyhf-toys` | YAML-configured pyhf toy-based CLs scan |
| `--toys` | alias for `--pyhf-toys` |
| `--native-toys` | native unbinned hNSBI toy campaign |
| `--nis` | NIS proposal training and efficient Asimov construction |

```bash
python examples/lhc_analysis/run_analysis.py \
  examples/lhc_analysis/analysis.yaml \
  --pyhf-toys --native-toys --nis
```

The pyhf projection is intentionally explicit, **binned, and lossy**: it
projects one configured observable and the corresponding modeled
sample/Asimov weights into HistFactory. It is a standard CLs cross-check, not
an implicit or lossless conversion of the high-dimensional native unbinned
likelihood. Native fits, scans, impacts, and pseudo-experiments continue to
use the unbinned model.

The checked-in settings target a GPU-backed Colab or workstation. Reduce
event counts and epochs in a copy of the YAML for a smoke run.
