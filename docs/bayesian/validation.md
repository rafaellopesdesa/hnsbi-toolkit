# Dual-model validation

Validate each learned object separately before relying on agreement between
routes.

## Minimum suite

- posterior-flow closure and simulation-based calibration where appropriate;
- posterior residual member validation and
  $\mathbb E_g[\widehat r_{\rm P}]$ checks;
- conditional observation-flow classifier two-sample tests;
- likelihood residual closure at held-out $\theta$ values;
- direct Monte Carlo checks of $\widehat Z_{\rm C}(\theta)$;
- member-wise extreme-weight and Pareto-$k$ summaries;
- posterior-side versus likelihood-side normalized weights;
- the pointwise bridge between $\widehat r_{\rm P}$ and the normalized
  conditional likelihood;
- independent simulator posterior-predictive validation.

Validation splits must be made at the simulator-pair or matched-group level.
Putting a positive row in training and its matched negative row in validation
leaks the context and produces optimistic classifier diagnostics.

Use exact `train`, `validation`, and `holdout` values in each configured
Bayesian `split_column`. The native trainer partitions simulator rows first
and carries the label onto both classifier members. `validation` controls
checkpoint selection; `holdout` never does. An unlabeled
`datasets.validation` file supplies an independent validation set unless the
stage source already has one, in which case it is treated as holdout.
If no holdout exists, deployment parity and the final ratio summary reuse the
validation rows and record `holdout_reuses_validation: true`; this is not a
claim of an independent holdout test.

## Support failures

Defensive mixing reduces but does not eliminate support risk. Always report
the design density, actual proposal density, ESS, maximum normalized weight,
and the parameter range represented after reweighting. Prior updates and
evidence calculations should fail clearly when the configured density cannot
be evaluated at a proposal point.
