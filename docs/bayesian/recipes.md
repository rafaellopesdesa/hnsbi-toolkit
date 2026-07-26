# Bayesian capability recipes

This page maps every capability in the paper's comparison table to a concrete
dual-model calculation. All weights should be computed in log space and
returned with ESS and tail diagnostics.

## Capability-to-API map

| Capability | Toolkit API |
|---|---|
| draw corrected hNPE, hNDE, or dual posterior samples | `sample_posterior()` with route `hnpe`, `hnde`, or `dual` |
| evaluate posterior-side weights at supplied points | `hnpe_log_weights()` |
| evaluate likelihood-side weights at supplied points | `hnde_log_weights()` |
| combine independently normalized routes | `geometric_consensus()` |
| update a nuisance prior or auxiliary likelihood | `prior_auxiliary_log_update()` or `update_posterior_weights()` |
| evaluate the normalized generative likelihood | `DualModel.log_likelihood()` |
| estimate absolute evidence | `estimate_evidence()` |
| generate a simulator-free posterior predictive | `posterior_predictive()` |
| estimate fixed-parameter selection efficiency | `selection_integral()` |
| integrate a population selection function | `population_selection()` |
| diagnose route agreement and the bridge identity | `route_diagnostic()` and `bridge_diagnostic()` |
| diagnose posterior/conditional normalization | `posterior_normalization_diagnostic()` and `conditional_normalization_diagnostic()` |

All functions above are exported from `hnsbi.bayes`. `WeightedSamples`
preserves both normalized weights and normalized log weights and reports ESS
and weight-tail summaries.

## Direct parameter proposal

Draw $\theta_j\sim q_\phi(\cdot\mid x_o)$, or from its defensive mixture with
$\rho$. This supplies efficient points but is not by itself the corrected
analysis posterior.

## Classifier correction of the posterior proposal

Let $d(\theta\mid x_o)$ denote the exact denominator against which
$\widehat r_{\rm P}$ was trained: either $q_\phi$ or its recorded defensive
mixture with $\rho$. For analysis prior $\pi$, design $\rho$, actual
inference proposal $g$, new auxiliary likelihood $f$, and optional
baseline auxiliary likelihood $f_0$, use

$$
u_j^{\rm P}
= d(\theta_j\mid x_o)\widehat r_{\rm P}(\theta_j;x_o)
  \frac{\pi(\theta_j)}{\rho(\theta_j)}
  \frac{f(a_o\mid\alpha_j)}{f_0(a_o\mid\alpha_j)}
  \frac{1}{g(\theta_j\mid x_o)}.
$$

`hnpe_log_weights()` applies this accounting. When its `proposal_log_prob` is
omitted, the supplied points are assumed to have been drawn from $d$, so the
corresponding denominator/proposal terms cancel numerically.

## Normalized posterior

Set $w_j=u_j/\sum_k u_k$. Expectations are weighted sums over the parameter
proposal. Report

$$
N_{\rm eff}=\frac{(\sum_j u_j)^2}{\sum_j u_j^2}
$$

and inspect dominant weights and support boundaries.

## Normalized generative likelihood

Evaluate

$$
\widehat L_{\rm C}(x\mid\theta)
=q_\eta(x\mid\theta)
 \widehat r_{\rm C}(x;\theta)
 /\widehat Z_{\rm C}(\theta).
$$

Generation draws candidates from $q_\eta$, weights or resamples them with
$\widehat r_{\rm C}$, and retains the conditional-normalization diagnostics.

## Likelihood-side posterior and dual consensus

For any parameter proposal $g$,

$$
u_j^{\rm L}
=\frac{\pi(\theta_j)[f(a_o\mid\alpha_j)/f_0(a_o\mid\alpha_j)]
        \widehat L_{\rm C}(x_o\mid\theta_j)}
       {g(\theta_j\mid x_o,a_o)}.
$$

Normalize posterior-side and likelihood-side weights separately. The dual
consensus uses their geometric mean followed by one final normalization. Do
not take a geometric mean of unnormalized weights with arbitrary constants.

`geometric_consensus()` performs exactly this normalized construction.
`sample_posterior(route="dual")` applies it internally when both routes share
the same sampled parameter points.

## Absolute evidence

For design prior $\rho$,

$$
p_\rho(x_o)
=\int \rho(\theta)\widehat L_{\rm C}(x_o\mid\theta)\,d\theta.
$$

Estimate the integral by Monte Carlo or another declared quadrature and return
its Monte Carlo uncertainty. For a different prior, replace $\rho$ only when
the integration proposal has adequate support.

## Posterior-predictive generation without the simulator

Draw or resample $\theta$ from the corrected posterior. For each parameter,
draw $x_{\rm rep}\sim q_\eta(\cdot\mid\theta)$, correct with
$\widehat r_{\rm C}(x_{\rm rep};\theta)$, and resample. These replicas support
checks and decisions but do not replace validation on independent simulator
replicas.

## Selection integrals without the simulator

For selection indicator $I_{\rm det}(x)$,

$$
\beta(\theta)
=\frac{
  \mathbb E_{q_\eta(\cdot\mid\theta)}
  [I_{\rm det}(x)\widehat r_{\rm C}(x;\theta)]
}{\widehat Z_{\rm C}(\theta)}.
$$

Population selection functions follow by integrating $\beta$ against the
population model. `SelectionEstimate` returns the ordinary and self-normalized
Monte Carlo estimates, the sampled reference normalization, and conditional
ESS. The current API does not attach an uncertainty estimate; quantify Monte
Carlo error with independent repeats or an analysis-appropriate resampling
procedure rather than interpreting ESS itself as a standard error.

## Nuisance-prior and auxiliary-likelihood updates

Keep nuisances explicit in $\theta$. Reweight existing parameter proposals by
the ratio of the new prior to the design prior and by the new auxiliary
likelihood. No neural retraining is required when support is adequate. A low
ESS, large Pareto-$k$, or missing support is a failed update, not a reason to
clip weights silently.

`prior_auxiliary_log_update()` returns the additive log factor
$\log(\pi/\rho)+\log(f/f_0)$. `update_posterior_weights()` applies this factor
to existing log weights and returns newly normalized linear weights. The
optional baseline $f_0$ makes it possible to replace, rather than merely
append, an auxiliary likelihood.
