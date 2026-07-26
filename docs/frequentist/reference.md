# Reference flow

The reference dataset defines the proposal density \(q(x)\). It need not have
the physical signal/background composition, but it must cover every region in
which a process ratio will be evaluated.

Two architectures are part of the stable configuration:

- `realnvp`: affine coupling transforms, useful as a robust baseline;
- `quadratic_spline`: autoregressive rational-quadratic-spline transforms,
  usually more expressive for the same depth and also used by the conditional
  Bayesian stages.

The flow training split must be disjoint from:

- samples used for final closure;
- reference events used to estimate \(\mathbb E_q[\widehat r_s]\);
- events used to report paper-level performance.

## Deployment bundle

A reference bundle contains:

- `<prefix>.log_prob.onnx`, mapping original ordered features to
  \(\log q(x)\);
- `<prefix>.base_to_data.onnx`, mapping a supplied latent normal draw to
  observation space;
- `<prefix>.data_to_base.onnx`, mapping observations back to latent space;
- a `<prefix>.manifest.json` bundle with feature order, context signature,
  architecture, embedded-scaler declaration, and file checksums;
- a separate native checkpoint and checksum sidecar for trusted restarts;
- ONNX-parity results and a checksummed diagnostics directory.

The fitted affine standardizer is embedded in all three graphs. There is no
separate `preprocess.onnx` in the current artifact contract.

Sampling randomness remains in the caller: the ONNX inverse graph is
deterministic for a supplied latent value. This permits reproducible toys.

## Diagnostics

At minimum, compare held-out data and flow samples through:

- one-dimensional weighted densities and CDFs;
- pairwise projections and correlations;
- classifier two-sample tests on independent data;
- tail occupancy and finite log-density checks;
- Torch/ONNX numerical parity.

A closure plot is evidence about the tested projections, not a proof that the
complete high-dimensional density is correct.

`diagnose_flow()` returns all of these checks in one JSON-serializable result.
Its classifier two-sample test uses a deterministic held-out split and
reference event weights. If scikit-learn is not installed, the report records
an unavailable C2ST with a reason instead of making flow diagnostics unusable.
Non-finite counts, 0.1%/1%/99%/99.9% quantiles, and generated occupancy beyond
the reference 1%--99% interval are recorded for every feature and for
log-density values.

For every feature pair, the report includes correlation closure, a common-bin
2D total-variation distance, and Jensen--Shannon divergence.
`plot_pairwise_closure()` renders reference, generated, and residual 2D
densities; `FlowDiagnosticResult.save()` includes that plot in the checksummed
bundle whenever the model has at least two features.

`Project.train_reference(validation_source=...)` runs these diagnostics on the
supplied `DataSource`. Without that argument, it uses the exact internal
validation holdout selected by `FlowTrainer` when `validation_fraction` is
positive; if the fraction is zero, it necessarily falls back to the training
sample. The manifest records which source was used. The internal holdout
participates in early stopping, so it is a training diagnostic—not independent
paper-level validation. Supply a disjoint simulator sample for final closure
and performance claims.
