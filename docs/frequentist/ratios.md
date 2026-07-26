# Density ratios

For nominal sample $s$, train a balanced classifier between $p_s$ and the
frozen reference $q$. With equal class normalization, optimal classifier
odds estimate $r_s=p_s/q$.

The `nsbi_common_utils` backend owns the classifier training and established
diagnostics. `hnsbi-toolkit` supplies data adapters, artifact packaging, and
the additional normalization needed by the hybrid density.

## Ensembles

The tutorial convention is the arithmetic mean of member ratios,

$$
\widehat r_{\rm ens}(x)
= \frac{1}{M}\sum_{m=1}^{M}\widehat r_m(x),
$$

not the mean classifier score and not the exponential of the mean log ratio.
The aggregation rule is recorded in the artifact manifest.

## Independent normalization

Estimate

$$
C_s=\mathbb E_q[\widehat r_{\rm ens}(x)]
$$

on reference events not used for classifier training or member selection, and
deploy $\widetilde r_s=\widehat r_{\rm ens}/C_s$. Record the uncertainty,
sample size, ESS, and ratio-tail summary. `Project.train_ratios()` performs
this independent draw, stores the means, Monte Carlo standard errors, row
count, and normalization-weight ESS in a checksummed
`ratio_normalization.json`. Reusing the weighted Asimov sample itself for this
validation can hide the error being measured.

## Diagnostics delegated to the backend

- training/holdout overfit comparisons;
- score and ratio calibration;
- reference-to-target reweighting in all features;
- normalization closure.

The adapter additionally enforces ONNX/native parity and finite probability or
log-ratio outputs before accepting each member. `RatioEnsemble.member_ratios()`
and `standard_deviation()` expose member agreement; `weight_summary()` and
`ratio_normalization_report()` provide common tail and normalization summaries
for independent validation samples.
