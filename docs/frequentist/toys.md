# Pseudo-experiments

At parameter point $\theta_0$, a toy from an extended model first draws the
event count

$$
N\sim\operatorname{Poisson}(\nu(\theta_0)),
$$

then draws event locations from
$\lambda(x\mid\theta_0)/\nu(\theta_0)$.

If direct component simulators are unavailable, the toolkit can use the
reference flow as a proposal and perform self-normalized importance
resampling with the configured normalized ratios. This is an approximation
whose accuracy depends on proposal coverage and the size of the candidate
pool. Each toy records the generating point and, per component, the sampling
method, final proposal-pool size, pool ESS, and number of growth rounds in
`events.metadata["component_sampling_diagnostics"]`.

For every constrained nuisance, the result also draws and records one Gaussian
auxiliary observation centered on the generating nuisance value with the
configured constraint width. It is available as
`ToyResult.constraint_observations` and in the persisted event metadata.

`ToyGenerator` draws one independent Poisson count per intensity component and
then fills that component either from its direct sampler or from a fresh
reference-flow importance pool. The pool grows until a minimum ESS condition
is met or fails explicitly; low-support pools are not silently accepted.

`ToyGenerator.from_workspace()` reconstructs the exact intensity formulas,
parameter declarations, feature order, ratio normalizers, and systematic
metadata after verifying the workspace manifests. Runtime reference/ratio
objects remain explicit arguments, so callers can choose native or ONNX
execution. A workspace with systematics additionally requires one matching
`RuntimeSystematic` per `(component, parameter)`; yield morphing is applied to
the Poisson expectation and shape morphing to the component draw.

The result should retain:

- parameter point and random seed;
- generated count and expected count;
- feature table and process labels when sampled component-wise;
- per-component counts and expectations;
- Gaussian constraint observations;

`ToyGenerator` only generates the pseudo-data and auxiliary observations. Run
`ExtendedUnbinnedLikelihood.fit()` / `.profile_scan()` or the upstream
inference adapter explicitly on the generated result when a fit or scan is
part of the study.

For validation, check the Poisson count mean and variance, fitted-parameter
closure, boundary pile-up, and the toy test-statistic distribution. A large
number of toys does not remove bias in the learned intensity.
