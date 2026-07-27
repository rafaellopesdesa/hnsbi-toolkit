# Native frequentist backend

Version 0.2 is self-contained. `hnsbi-toolkit` does not install, import,
clone, or dynamically discover `nsbi-lhc-toolkit`. The `lhc` extra installs
only the scientific libraries used by the native implementation, including
JAX, iminuit, and pyhf.

## Density ratios and diagnostics

`NativeRatioBackend` trains weighted Torch classifiers with reproducible
train/validation/holdout splits. It supports classifier ensembles, early
stopping, isotonic or histogram calibration, and physical-input ONNX export
with the fitted standardizer embedded in the graph.

`diagnose_ratio` covers balanced loss, train/holdout overtraining, score and
ratio calibration, weighted AUC, all-feature reweighting closure,
$E_q[r]$ normalization, and ratio-tail summaries. Diagnostics and plots are
serialized independently of the training checkpoint.

## Workspaces and inference

The native JSON workspace stores:

- observation arrays and reference quadrature weights;
- component ratios and independent normalization constants;
- restricted sample-multiplier expressions;
- parameter initial values, bounds, roles, and Gaussian constraints;
- normalized shape-systematic anchors;
- checksummed references to flow, ratio, and array artifacts.

`ExtendedUnbinnedLikelihood` is the NumPy runtime.
`JaxLikelihood` evaluates the same intensity and constraints with automatic
differentiation. `MinuitInference` uses MIGRAD and HESSE and exposes profile
and test-statistic scans. `CombinedLikelihood` sums independent channel
workspaces while sharing named parameters and counting each shared Gaussian
constraint once.

## Provenance

The classifier-ratio, diagnostic, workspace, and fitting design incorporates
ideas and implementation patterns from the MIT-licensed
[`iris-hep/nsbi-lhc-toolkit`](https://github.com/iris-hep/nsbi-lhc-toolkit).
The native implementation is maintained here and has no upstream runtime
dependency. See the repository `NOTICE` for the exact copyright and license
attribution.
