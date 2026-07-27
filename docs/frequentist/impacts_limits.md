# Pulls, impacts, and CLs limits

The frequentist diagnostics distinguish a fitted nuisance parameter from its
auxiliary or *global* observation. This distinction matters: the recommended
refit-based impact varies the global observation and leaves the nuisance
parameter floating.

## Pulls

For a Gaussian constraint with observed auxiliary value $t_k$ and pre-fit
width $\sigma_{t_k}$, the nuisance pull and normalized post-fit uncertainty
are

$$
p_k = \frac{\hat{\theta}_k-t_k}{\sigma_{t_k}},
\qquad
c_k = \frac{\sqrt{V_{kk}}}{\sigma_{t_k}}.
$$

Only constrained parameters are pulls. An unconstrained parameter estimate
must not be displayed as though it had a pre-fit standard-deviation band.

```python
from hnsbi.impacts import compute_pulls, plot_pulls

fit = likelihood.fit()
pulls = compute_pulls(likelihood, fit)
figure, axis = plot_pulls(pulls)
figure.savefig("pulls.pdf")
```

The result objects provide `to_dict()` and `to_json()` methods. Plot functions
return the Matplotlib figure and axes and never call `show()`.

## Global-observable impacts

For every constrained nuisance $k$, the toolkit repeats the fully profiled fit
after changing only its auxiliary observation:

$$
t_k \longrightarrow t_k \pm \sigma_{t_k},
\qquad
\Delta\mu_k^\pm =
  \hat{\mu}(t_k\pm\sigma_{t_k})-\hat{\mu}(t_k).
$$

Both signed responses are retained, so nonlinear or asymmetric responses are
visible. The reported symmetric ranking magnitude is

$$
I_k = \frac{|\Delta\mu_k^+|+|\Delta\mu_k^-|}{2}.
$$

```python
from hnsbi.impacts import global_observable_impacts, plot_impacts

impacts = global_observable_impacts(
    likelihood,
    "mu",
    groups={"detector": ["scale", "resolution"]},
)
figure, axis = plot_impacts(impacts)
```

No nuisance parameter is fixed in these refits. Passing fixed parameters to
the impact fit is rejected, preventing accidental use of the legacy
shift-and-fix procedure.

## Covariance impacts

In the local Gaussian and linear regime, the same one-standard-deviation
response can be obtained from the post-fit covariance:

$$
\Delta\mu_k^+ = \frac{V_{\mu k}}{\sigma_{t_k}},
\qquad
\Delta\mu_k^- = -\frac{V_{\mu k}}{\sigma_{t_k}}.
$$

For independent Gaussian constraint sources, the systematic and residual
statistical components are

$$
\sigma_{\mu,\mathrm{syst}} =
\sqrt{\sum_k\left(\frac{V_{\mu k}}{\sigma_{t_k}}\right)^2},
\qquad
\sigma_{\mu,\mathrm{stat}} =
\sqrt{V_{\mu\mu}-\sigma_{\mu,\mathrm{syst}}^2}.
$$

```python
from hnsbi.impacts import covariance_impacts

impacts = covariance_impacts(likelihood, "mu", fit=fit)
```

This decomposition requires a named, positive-semidefinite covariance from a
valid HESSE calculation. It is not reliable at a hard parameter boundary, for
a singular fit, or when the likelihood is substantially non-Gaussian.
Correlated auxiliary sources require an explicit source covariance rather than
the independent-source quadrature above.

## Direct pyhf inference

`hnsbi.pyhf_tools` wraps the public pyhf APIs for an ordinary HistFactory
`pyhf.Model`. It does not convert an unbinned hnsbi likelihood into a binned
HistFactory model.

For a fit with uncertainties and correlations, configure pyhf's Minuit
optimizer first:

```python
import pyhf
from hnsbi.pyhf_tools import fit

pyhf.set_backend(
    pyhf.tensorlib,
    pyhf.optimize.minuit_optimizer(),
)
fit_result = fit(data, model)
```

The wrapper requests HESSE uncertainties and correlations and returns named
values, errors, covariance, correlation, and twice the negative
log-likelihood.

The same pull and impact definitions are available directly for pyhf models:

```python
from hnsbi.pyhf_tools import (
    covariance_impacts as pyhf_covariance_impacts,
    global_observable_impacts as pyhf_global_observable_impacts,
    pulls as pyhf_pulls,
)

pull_result = pyhf_pulls(data, model, fit_result=fit_result)
refit_impacts = pyhf_global_observable_impacts(
    data,
    model,
    fit_result=fit_result,
    groups={"detector": ["scale", "resolution"]},
)
local_impacts = pyhf_covariance_impacts(
    data,
    model,
    fit_result=fit_result,
)
```

`PyhfLikelihoodAdapter` reads the HistFactory auxiliary-data order and
parameter-set metadata. It supports normal constraints and maps Poisson
auxiliary counts into the nuisance coordinate using the pyhf parameter-set
factor. The global-observable method shifts that auxiliary coordinate and
refits every model parameter; it never shifts or fixes the nuisance
parameter itself. Unsupported constraint distributions fail explicitly.

At a tested signal strength $\mu$, pyhf evaluates

$$
\mathrm{CL_s} = \frac{\mathrm{CL}_{s+b}}{\mathrm{CL}_b}.
$$

Both asymptotic and toy-based calculators are available:

```python
from hnsbi.pyhf_tools import hypotest, upper_limit

point = hypotest(1.0, data, model, calctype="asymptotics")
asymptotic_limit = upper_limit(data, model, scan=None)

toy_limit = upper_limit(
    data,
    model,
    scan=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    calctype="toybased",
    ntoys=10_000,
    track_progress=False,
)
```

The pyhf asymptotic calculator constructs its conditional background Asimov
dataset internally. Toy calculations use pyhf's conditional plug-in nuisance
points. A fixed increasing scan is mandatory for toy upper limits because
stochastic CLs evaluations are unsuitable for deterministic root finding.
Every fixed scan must bracket a downward crossing of the requested CLs level
for the observed curve and all five expected bands; otherwise the wrapper
raises instead of reporting a clipped scan endpoint as a limit.
The returned upper-limit object retains the observed and five expected limits,
the evaluated scan, and its observed/expected CLs curves.

## Method references

The global-observable and covariance decompositions follow
[Pinto et al., *Uncertainty components in profile likelihood fits*](https://arxiv.org/abs/2307.04007).
The CLs wrappers use the public
[pyhf inference API](https://scikit-hep.org/pyhf/api.html#inference). pyhf is
Copyright 2018 pyhf Developers and distributed under the Apache License 2.0;
see `NOTICE`.
