# Density ratios

For nominal sample $s$, train a balanced classifier between $p_s$ and the
frozen reference $q$. With equal class normalization, the optimal classifier
odds estimate $r_s=p_s/q$.

`NativeRatioBackend` performs weighted Torch training, early stopping, and
physical-input ONNX export. Its standardizer is fitted only on the training
partition and embedded in the exported graph. Validation and holdout
partitions remain distinct.

When a sample source declares `split_column`, `Project.train_ratios()` honors
the exact `train`, `validation`, and `holdout` labels. A configured
`group_column`, or `event_id_column` when no group column is present, keeps all
rows with the same identifier in one deterministic partition. Equal group
identifiers are coordinated across numerator and denominator samples, and
conflicting explicit labels fail before training. With neither option, the
backend retains its seeded row-wise random split.

## Ensembles and calibration

The ensemble convention is the arithmetic mean of member ratios,

$$
\widehat r_{\rm ens}(x)
= \frac{1}{M}\sum_{m=1}^{M}\widehat r_m(x),
$$

not the mean classifier score and not the exponential of the mean log ratio.
The reduction is recorded in the artifact manifest.

Optional isotonic or histogram calibration is fitted on validation scores and
serialized as finite piecewise-linear state. Native and ONNX outputs must
agree before a member is accepted.

## Independent normalization

Estimate

$$
C_s=\mathbb E_q[\widehat r_{\rm ens}(x)]
$$

on reference events not used for classifier training or model selection, and
deploy $\widetilde r_s=\widehat r_{\rm ens}/C_s$. Record the uncertainty,
sample size, ESS, and ratio-tail summary. `Project.train_ratios()` writes the
means, Monte Carlo standard errors, row count, and normalization-weight ESS
to checksummed JSON.

## Diagnostics

`diagnose_ratio` produces:

- weighted training and holdout loss;
- weighted train/holdout KS checks for both classes;
- score calibration and empirical MC log-density-ratio calibration on the
  training and independent holdout partitions;
- classifier-score saturation flags and ratio-tail summaries;
- weighted AUC;
- train/holdout reference-to-target closure for every feature;
- $E_q[r]$ normalization and Monte Carlo uncertainty;
- finite-output checks.

The four YAML switches under `frequentist.ratios.diagnostics` independently
enable `overtraining`, `calibration`, `reweighting`, and `normalization`.
`RatioDiagnosticReport.write()` records the enabled checks in its strict-JSON,
checksummed manifest and writes optional Matplotlib plots. Reusing the final
weighted Asimov sample for these checks can hide the error being measured;
production closure needs independent simulator data.
