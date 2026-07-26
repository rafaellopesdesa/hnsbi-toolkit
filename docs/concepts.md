# Concepts

## A flow plus a residual ratio

Let \(q(x)\) be a normalized reference density represented by a normalizing
flow and let \(p_s(x)\) be the density of physics sample \(s\). A classifier
can estimate the residual ratio

$$
r_s(x) = \frac{p_s(x)}{q(x)}.
$$

The hybrid estimate is \(q(x)\widehat r_s(x)\). A finite classifier does not
guarantee that this product integrates to one, so the deployable ratio is

$$
\widetilde r_s(x) =
\frac{\widehat r_s(x)}
{\widehat{\mathbb E}_{x\sim q}[\widehat r_s(x)]},
$$

where the denominator is estimated using an independent reference sample.
This normalization is essential in signal-strength directions: without it,
shape error can masquerade as a yield change and displace the Asimov minimum.

## An intensity, not only a density

For expected yields \(\nu_s(\theta)\), the event intensity is

$$
\lambda(x\mid\theta)
= q(x)\sum_s \nu_s(\theta)\widetilde r_s(x;\theta).
$$

The extended likelihood contains both the event term and the integral
\(\nu(\theta)=\int\lambda(x\mid\theta)\,dx\). Sample multipliers in a serialized
configuration therefore use a restricted expression language and must remain
nonnegative over the fitted parameter domain.

## The dual Bayesian construction

The posterior side starts with a conditional parameter flow
\(q_\phi(\theta\mid x)\) and a correction
\(\widehat r_{\rm P}(\theta;x)\). The observation side starts with a
conditional flow \(q_\eta(x\mid\theta)\), a residual ratio
\(\widehat r_{\rm C}(x;\theta)\), and a parameter-dependent normalizer

$$
\widehat Z_{\rm C}(\theta)
= \mathbb E_{x\sim q_\eta(\cdot\mid\theta)}
  [\widehat r_{\rm C}(x;\theta)].
$$

The normalized conditional likelihood is

$$
\widehat L_{\rm C}(x\mid\theta)
= \frac{q_\eta(x\mid\theta)\widehat r_{\rm C}(x;\theta)}
       {\widehat Z_{\rm C}(\theta)}.
$$

The posterior and likelihood routes should agree after normalization. Their
pointwise bridge is an unusually useful internal validation because the two
corrections are trained independently.

## Validation is not optional

Hybrid correction reduces approximation error only where the training designs
have support. Honest validation requires independent simulator samples,
group-safe splitting of matched classifier pairs, conditional normalization
checks, and inspection of extreme importance weights. Posterior-predictive
generation from the learned likelihood tests the surrogate; only fresh
simulator data can reveal misspecification shared by all learned components.
