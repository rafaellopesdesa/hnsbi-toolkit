# Training the five artifacts

Users provide simulator datasets already drawn under the $\rho$, $\nu$,
and $\kappa$ parameter designs. Separate files or explicit independent
splits are strongly preferred.

`Project.train_dual()` provides the complete native route. It materializes the
configured Parquet/Arrow sources into `ProposalDataset` objects, preserves
simulation identifiers and optional split/design-log-density columns, trains
the five objects in the statistically required order, checks native/ONNX
parity, and writes `dual_model.manifest.json`.

When a source declares `split_column`, its exact `train`, `validation`, and
`holdout` labels control the scientific partition. Conditional-flow scalers
and weights see only `train`; native ratio early stopping sees
`validation`; final ratio reports and graph parity prefer `holdout`. Matched
classifier rows are constructed only after this partition is fixed. See
{doc}`../specifications/data` for the complete contract and the fallback
behavior of `datasets.validation`.

```python
from hnsbi import Project

project = Project.load("examples/configs/dual_complete.json")
artifacts = project.train_dual()

model = artifacts.model
print(artifacts.manifest_path)
```

The default conditional-density stages are quadratic-spline flows. The default
`backend: native` ratio stages are Torch classifiers with preprocessing
embedded in their ONNX graphs. A caller can inject established ratio backends
for `r_p` and/or `r_c` through `ratio_backends`; the final dual graph contract
remains the same.

An injected object implementing `BayesianTrainingBackend` is still accepted by
`Project.train_dual(backend=...)` for research-specific training. That path
returns the backend's `DualModel`; portable packaging is then the backend's
responsibility.

## Posterior flow and correction

Train $q_\phi(\theta\mid x)$ on
$\theta\sim\rho,\ x\sim p(x\mid\theta)$, then freeze it. On an independent
matched-$x$ sample, compare:

$$
j_1(\theta,x)=\rho(\theta)p(x\mid\theta)
$$

with rows in which the same $x$ is paired with
$\widetilde\theta\sim q_\phi(\cdot\mid x)$. Split by the matched observation
group before constructing classifier rows.

When `defensive_epsilon > 0`, the negative parameter is instead drawn from
$(1-\epsilon)q_\phi(\theta\mid x)+\epsilon\rho(\theta)$. The exact
denominator and $\epsilon$ are recorded in the dual manifest and reused by
posterior inference.

## Conditional likelihood and residual

Train $q_\eta(x\mid\theta)$ using
$\theta\sim\nu,\ x\sim p(x\mid\theta)$, then freeze it. For
$\theta\sim\kappa$, create matched rows

$$
x^+\sim p(\cdot\mid\theta),
\qquad
x^-\sim q_\eta(\cdot\mid\theta).
$$

Keep each matched-$\theta$ pair in one split. Using different parameter
distributions in the positive and negative classes introduces their density
ratio into the classifier odds.

## Conditional normalization

At many contexts $\theta$, draw from the frozen $q_\eta$ and estimate

$$
\log \widehat Z_{\rm C}(\theta)
= \log\left[
  \frac{1}{M}\sum_{m=1}^{M}
  \widehat r_{\rm C}(x_m;\theta)
\right].
$$

A smooth regressor amortizes this calculation as the fifth stored object,
$\widehat Z_{\rm C}$, and requires independent conditional checks.

The native backend draws the configured number of $q_\eta$ observations at
each configured context, constructs Monte Carlo $\log Z_{\rm C}$ targets,
and trains an MLP regressor. It retains the targets and training history next
to the ONNX graph so validation is reproducible. Configured validation
contexts receive independent Monte Carlo targets for checkpoint selection;
holdout contexts receive a separate bias/RMSE report. The ratio stages
similarly write `validation.json`, including balanced logistic loss,
classification accuracy, and split counts. These reports are produced even
when an injected established ratio backend manages its own internal training
split.
