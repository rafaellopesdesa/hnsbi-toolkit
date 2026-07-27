# Neural-importance-sampling Asimov

NIS learns a proposal $g_\varphi(x)$ concentrated where the chosen
collection of Asimov integrands is influential. The user supplies design
points in parameter space; those points define the target, rather than being
hidden in notebook code.

To protect coverage, sampling uses the defensive mixture

$$
q_\epsilon(x)
= (1-\epsilon)g_\varphi(x)+\epsilon q(x),
\qquad 0<\epsilon\leq 1.
$$

The final importance denominator is the actual mixture $q_\epsilon$.
Process ratios remain normalized with respect to the original reference
$q$, because that normalization defines the physical component densities.

## Workflow

1. Draw a high-statistics pilot sample from $q$.
2. Evaluate normalized ratios and influence amplitudes at every design point.
3. Train the NIS flow on the resulting nonnegative pilot weights.
4. Freeze and export the proposal.
5. Draw reproducibly from both mixture components.
6. Evaluate $q$, $g_\varphi$, and $q_\epsilon$ exactly for every draw.
7. Return the efficient Asimov sample with ESS and training diagnostics.

The returned NIS Asimov uses normalized $q/q_\epsilon$ quadrature weights
and fits every process-ratio normalizer on that same weighted support. It
therefore has exact finite-support total-yield closure, while the independent
direct-reference normalizer remains the appropriate learned-ratio validation.
Metadata records the defensive bound $q/q_\epsilon\leq1/\epsilon$, its
observed maximum, reference-weight diagnostics, raw count, ESS, and intensity
fingerprint.

Runtime systematics can be passed to both `NISProposalTrainer` and
`NISAsimovBuilder`. They enter the influence target at every nuisance design
point and the final efficient quadrature uses the same normalized
`SystematicRatioEvaluator` morph as the direct Asimov and native likelihood.
This gives systematic-adjusted yield and shape closure at nonzero generating
points.

## Required validation

- proposal versus influence-target marginals and pair projections;
- centered $\log(g_\varphi/q)$ behavior and finite density checks;
- repeated small-sample scans;
- convergence of scan error, fitted minimum, and ESS versus raw count;
- comparison with a high-statistics direct-reference benchmark;
- sensitivity to the defensive $\epsilon$.

An ESS gain is useful only if the scan remains unbiased at all supplied design
points.

## Reusable numerical validation

The `hnsbi.nis_diagnostics` module keeps the validation calculations separate
from the proposal implementation:

- `summarize_nis_log_weights()` reports finite and exact-zero counts, stabilized
  ESS, maximum normalized weight, dynamic range, Pareto tail shape, and the log
  mean weight;
- `compare_nis_proposal()` compares those quantities with a direct-reference
  benchmark;
- `nis_prefix_convergence()` evaluates normalization, ESS, concentration, and
  optional self-normalized observables on deterministic event prefixes;
- `compare_nis_epsilons()` evaluates user-supplied defensive-mixture choices
  with the same metric definitions.

Use `nis_prefix_convergence()` for raw-$N$ convergence and
`compare_nis_epsilons()` for the defensive-mixture scan. For likelihood-scan
closure, run `ExtendedUnbinnedLikelihood.profile_scan()` or
`MinuitInference.profile_scan()` on deterministic prefixes and independent
repeated samples; there is deliberately no hidden scan inside proposal
training.

Each result has `to_dict()` for strict JSON serialization: numerical NaN values
such as an unavailable optional Pareto fit become JSON `null`. Plot helpers for
log-weight samples, prefix convergence, and epsilon comparisons import
Matplotlib only when called.

## Configuration-first run

`Project.train_nis_asimov()` reads the configured design points, defensive
epsilon, pilot size, target size, and flow architecture. It returns the pilot
influence target, native training history, checksummed ONNX bundle, a
`FlowDiagnosticResult` usable with the flow plotting functions, ONNX parity,
the exact defensive proposal, and the efficient weighted Asimov sample. That
automatic `FlowDiagnosticResult` compares generated events with the weighted
pilot target used to train the proposal. It is useful for optimization
debugging but is not an independent validation sample. Apply `diagnose_flow()`
to a separately supplied pilot/target sample and use the explicit
`hnsbi.nis_diagnostics` helpers above for paper-level validation.
Its ONNX parity rows and flow-closure reference are the exact internal
validation subset excluded from flow optimization. Both manifests record the
split seed, validation fraction, and train/validation row counts.
