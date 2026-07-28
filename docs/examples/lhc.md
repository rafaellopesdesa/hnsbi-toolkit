# Complete LHC analysis

`examples/lhc_analysis/analysis.yaml` is the primary end-to-end frequentist
example. Its generator follows the established correlated Gaussian-mixture
exercise and adds three nuisance families:

| Nuisance | Variation |
|---|---|
| response | detector response/scale shifted up and down |
| resolution | event-level Gaussian residual widened or narrowed |
| theory | centers of the narrow signal Gaussians shifted in latent $z$ |

## Generated and selected samples

Generation produces both the raw `<sample>.parquet` files and
`<sample>_presel.parquet` analysis inputs. The selected weight sum is the
physical expected yield after selection, including the variation-specific
acceptance for each systematic anchor. The checked-in YAML consumes these
selected files; the raw files are retained for validation. Its
`nominal_yield: {kind: source_weight_sum}` entries ask `Project` to stream
each selected source once and cache that physical yield.

The original ML4HEP exercise used a learned signal/background preselection,
but its trained checkpoint is unavailable. Because this synthetic example has
known reconstructed Gaussian-mixture densities, the generator evaluates the
same $p_S(x)/p_B(x)$ variable analytically. This deterministic
Gaussian-mixture selector is legacy-equivalent for the example; it does not
claim to reproduce the unpublished learned checkpoint.

One nominal cut is chosen to meet the configured $B/S\leq250$ target and then
applied unchanged to nominal, response, resolution, and theory samples.
Original rows are assigned with the historical SplitMix64 rule to disjoint
50% selector, 46% downstream-training, and 4% evaluation partitions. The
reference sample is balanced only after selection,

$$
p_{\mathrm{ref}}^{\mathrm{sel}}(x)
=\tfrac12 p_S(x\mid\mathrm{pass})
+\tfrac12 p_B(x\mid\mathrm{pass}),
$$

so it remains a genuine equal conditional mixture.

`data/preselection.manifest.json` binds the selector fingerprint, cut,
partition settings, requested counts, selected-yield diagnostics, and
SHA-256 checksums of every raw and `_presel` parquet. Existing outputs are
reused only when manifest verification succeeds.

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
asymptotic CLs exclusions. Preselection is intentionally complete before the
runner starts, keeping the analysis workflow focused on hNSBI methods.

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
