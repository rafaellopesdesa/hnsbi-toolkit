# Frequentist workflow

The YAML-first LHC workflow has seven reusable stages:

1. train a normalized reference flow $q(x)$;
2. train native $p_s/q$ classifier ensembles for every nominal sample;
3. validate and independently normalize each ratio;
4. model systematics with up/down ratio anchors or a multi-nuisance FNF;
5. build a direct or NIS Asimov sample and native JSON workspace;
6. fit and scan with iminuit, optionally using JAX derivatives;
7. report pulls, impacts, and CLs exclusions.

The toolkit accepts Parquet paths and in-memory PyArrow or Awkward objects.
The YAML supplies all nominal samples, variations, parameters, and multiplier
formulas; learned models and workspaces can be reused without retraining.

`hnsbi-toolkit` implements this stack natively. It does not import or install
`nsbi-lhc-toolkit`. The ratio, diagnostic, workspace, and minimization design
incorporates work from that MIT-licensed project; see `NOTICE`.

## Two systematic models

Up/down density ratios are a direct choice when each nuisance has explicit
anchor samples. The likelihood separates rate and shape, interpolates each,
and renormalizes the shape on its reference quadrature.

An FNF is the normalized alternative when continuous multi-nuisance morphing
is needed. It learns an invertible residual relative to the frozen nominal
density, is exactly the identity at the nominal point, supports optional
pairwise nuisance interactions, and can be trained separately for each
channel/sample workspace.

## Statistical inference

`ExtendedUnbinnedLikelihood` evaluates the native workspace.
`JaxLikelihood` provides gradients and Hessians for the same model, while
`MinuitInference` performs MIGRAD/HESSE fits, profile scans, and one-sided
test-statistic scans. `CombinedLikelihood` combines independent workspaces
with shared named parameters.

Pulls and impacts use the full post-fit covariance. The refit-based impact
shifts the global observable and leaves the nuisance floating; the toolkit
does not implement the nuisance-shift-and-fix method. A separate pyhf
projection supports asymptotic and toy-based CLs limits.

## Physical constraints

The intensity must be nonnegative over the fitted domain. Signed Monte Carlo
weights may pass through the data layer, but a negative measure cannot be
trained as an ordinary probability density or sampled as a Poisson point
process. Such cases need an explicit signed-component construction.
