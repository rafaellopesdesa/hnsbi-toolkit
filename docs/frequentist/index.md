# Frequentist workflow

The frequentist workflow follows five separable stages:

1. train a normalized flow $q(x)$ on a user-supplied reference dataset;
2. estimate $p_s/q$ for every nominal physics sample, delegating classifier
   training and its established diagnostics to `nsbi-common-utils`;
3. estimate each ratio normalization on independent reference data;
4. connect the resulting intensity to an LHC-style workspace, fit, and scan;
5. generate pseudo-experiments or direct/NIS weighted Asimov samples at
   explicit parameter points.

Keeping these stages separate makes artifacts reusable. A new scan does not
retrain a density ratio, and changing the number of Asimov points does not
change the statistical model.

The toolkit accepts Parquet paths and in-memory PyArrow or Awkward objects.
Arrow record batches are the streaming interchange representation; conversion
to pandas occurs only at the `nsbi-common-utils` adapter boundary.

The model must have a nonnegative total intensity over the relevant parameter
domain. Signed Monte Carlo weights are preserved by the data layer, but a
negative measure cannot be trained as an ordinary probability density or
sampled as a Poisson point process. Such cases require an explicit signed
component construction rather than silent absolute weights.
