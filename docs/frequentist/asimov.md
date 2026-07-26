# Weighted unbinned Asimov samples

For \(N\) independent points \(x_i\sim q\), the direct weighted Asimov sample
at \(\theta_A\) uses

$$
w_i(\theta_A)
= \frac{1}{N}\sum_s
  \nu_s(\theta_A)\widetilde r_s(x_i;\theta_A).
$$

This construction approximates integrals under the physical intensity while
retaining a controllable raw point count. The returned object reports both
that raw count and

$$
N_{\rm eff}
= \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.
$$

Raw count and ESS answer different questions: increasing the raw count does
not repair inadequate reference support.

## Closure requirements

For a signal-strength parameter, independently normalized ratios ensure
\(\mathbb E_q[\widetilde r_s]=1\). The expected extended score then vanishes at
the generating signal strength in the population limit. The implementation
supports two deliberately different normalization modes:

- `normalization="sample"` fits each \(C_s\) on the same reference support
  used by the returned quadrature. This enforces exact finite-sample component
  normalization and total-yield closure in yield-only directions.
- `normalization="fixed"` uses a supplied `RatioNormalizer`, normally fitted
  on an independent reference sample. This is the honest validation/deployment
  mode: finite Monte Carlo error remains visible and closure is asymptotic.

`AsimovResult` always exposes the raw count, ESS, per-component ESS, raw and
normalized ratios, reference weights, parameter point, normalizer means and
their standard errors. Its event metadata records expected yield, total
weight, closure residual, normalization mode, and the exact intensity
fingerprint. `write_nsbi_arrays()` checksums those arrays for workspace use.

## Nonzero systematic points

Pass the trained runtime morphs through `AsimovBuilder(systematics=...)` (or
`Project.build_configured_asimov(systematics=...)`). On the sampled support,
the builder normalizes every up/down conditional-shape anchor under its
nominal process measure. It then uses `SystematicRatioEvaluator` to apply the
same anchor interpolation, separated yield morph, joint shape product, and
pointwise support normalization as `ExtendedUnbinnedLikelihood`.

The resulting component shapes each integrate to one on the Asimov
quadrature, including at intermediate nuisance values. Therefore the event
weights close to the systematic-adjusted expected yield at the generating
point. The support-bound anchors are retained in `AsimovResult` and
`write_nsbi_workspace()` writes them automatically; do not supply a second
set of workspace modifiers for such a result.

For constrained parameters, an Asimov result also records the auxiliary
observation at the generating truth. The native likelihood uses that
observation as the Gaussian constraint center, so a nonzero nuisance Asimov
has zero constraint score at its truth. The upstream runtime fixes
unit-Gaussian auxiliary data at zero, so these off-center workspaces are
explicitly routed to the native likelihood.

Use same-support normalization when constructing a deterministic numerical
Asimov quadrature. Use an independent fixed normalizer to assess whether the
learned ratio itself closes. Never shift the completed likelihood curve by
hand to place its minimum at the truth; a displaced minimum is a diagnostic
result.
