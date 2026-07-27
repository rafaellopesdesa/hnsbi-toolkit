# Complete YAML-driven LHC analysis

This example starts from the same correlated Gaussian-mixture construction as
the ML4HEP-TIF `generate_distributions.py` exercise and runs the complete
native toolkit:

1. generate nominal signal, background, and reference samples;
2. generate detector-response, detector-resolution, and signal-theory
   variations;
3. train the reference flow and native density-ratio ensembles;
4. run every ratio and flow diagnostic and export ONNX graphs;
5. construct a finite-quadrature Asimov sample and native JSON workspace;
6. fit with JAX gradients and iminuit MIGRAD/HESSE;
7. make pull and impact plots using both shifted global observables and the
   correlation matrix;
8. project the same templates and Asimov sample to HistFactory and calculate
   CLs exclusions with pyhf, asymptotically or with toys.

The response anchors change the detector mean by ±10%. Resolution anchors
scale the event-level Gaussian residual by 1.25 or 0.75. The theory anchors
move the centers of the two narrow signal Gaussians in latent
$(z_1,\ldots,z_5)$ while leaving the broad support component fixed.

```bash
pip install -e '.[lhc,flows]'
python examples/lhc_analysis/generate_distributions.py \
  --output examples/lhc_analysis/data
hnsbi validate-config examples/lhc_analysis/analysis.yaml
python examples/lhc_analysis/run_analysis.py \
  examples/lhc_analysis/analysis.yaml
```

The default performs the native unbinned fit/impact workflow and a
deterministic asymptotic pyhf CLs scan. Expensive optional stages have
explicit flags:

| Flag | Additional work |
|---|---|
| `--pyhf-toys` | configured 2,000-toy pyhf CLs scan |
| `--toys` | backward-compatible alias for `--pyhf-toys` |
| `--native-toys` | configured unbinned hNSBI pseudo-experiment campaign |
| `--nis` | configured NIS proposal training and efficient Asimov sample |

For example:

```bash
python examples/lhc_analysis/run_analysis.py \
  examples/lhc_analysis/analysis.yaml \
  --pyhf-toys --native-toys --nis
```

The configuration is intentionally the first interface. Paths are resolved
relative to `analysis.yaml`; learned artifacts and workspaces remain JSON/ONNX
for stable machine serialization.

The pyhf result is intentionally a **binned, lossy projection** of one
configured observable into HistFactory. It is useful for standard CLs
cross-checks, but it does not replace the native high-dimensional unbinned
likelihood, fit, scans, or toys. Agreement between the two views is a
validation result; exact equality is not expected after binning.

This run uses up/down ratio anchors for its three systematics. The native FNF
API is an alternative when the same variation samples should train one
normalized multi-nuisance morph. Train one FNF per `(channel, sample)` domain,
share nuisance names across workspaces, and keep the yield morph separate from
the normalized FNF shape. See the
[FNF documentation](../../docs/frequentist/fnf.md).

For a quick smoke run, reduce the generated event counts and the flow/ratio
epochs in a copy of the YAML. The checked-in values are meant for a GPU-backed
Colab or workstation and are not CI defaults.
