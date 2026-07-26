# Systematic variations

The initial interface supports the `NormPlusShape` pattern already used by
`nsbi-common-utils`: each affected nominal sample has explicit up and down
datasets, and the backend trains the corresponding density ratios and supplies
its standard diagnostics.

The configuration names:

- the nuisance parameter and its constraint;
- every affected physics sample;
- up/down data sources;
- optional up/down total-yield anchors;
- the interpolation convention used between templates.

`Project.train_systematics()` uses the same configured ratio trainer for
`up/nominal` and `down/nominal`. `Project.build_systematic_modifiers()` then
evaluates both ensembles on the exact workspace reference support, separates
their integrated yield factors from their shape ratios, and writes checksummed
`SystematicAnchor` arrays. The toolkit does not implement the FNF exercise in
the initial scope.

`yield_up` and `yield_down`, when present on a variation, are multiplicative
total-yield factors relative to that sample's nominal yield. They override
rate information inferred from the event tables. Both must be finite and
nonnegative; `nsbi_code4p` requires strictly positive anchors because its
extrapolation is logarithmic. A zero anchor is therefore supported only with
`linear` interpolation.

When either anchor is omitted, `Project.train_systematics()` falls back to

$$
y_{\rm up/down}
=
\frac{\sum_{i\in{\rm up/down}} w_i}
     {\sum_{j\in{\rm nominal}} w_j}.
$$

This fallback is a physical-normalization contract, not a row-count
convention: event weights must integrate to the corresponding physical sample
yields. If training is truncated with `max_events`, the retained weights must
already be rescaled to preserve those integrals; otherwise configure the yield
anchors explicitly. The training result records each direction's source as
`configured` or `integrated_mc_weights`.

## Normalization

The native hNSBI likelihood treats normalization and shape separately. It
interpolates the up/down yield factors and pointwise shape anchors, then
renormalizes the joint interpolated shape under the fixed reference quadrature
at every nuisance value. This preserves the configured extended yield even
between anchors. Both `linear` and the upstream-compatible `nsbi_code4p`
(HistFactory strategy 5) anchor interpolations are available.

This intermediate-point normalization is not performed by the pinned
`nsbi-common-utils` runtime. Consequently, every workspace containing a
`normplusshape` modifier is marked `hnsbi.upstream_compatible=false` and
`Project.workspace_runtime()` selects `ExtendedUnbinnedLikelihood`. It never
silently sends such a workspace through an upstream runtime with different
rate semantics.

`RuntimeSystematic` exposes the same interpolation and separated yield/shape
contract to `ToyGenerator`, `AsimovBuilder`, and `NISAsimovBuilder`; callers
supply callable up/down ratio evaluators. `Project.build_runtime_systematics()`
turns the output of `Project.train_systematics()` into this shared runtime
mapping. `SystematicRatioEvaluator` is the support-bound implementation used
by Asimov construction and `ExtendedUnbinnedLikelihood`, preventing the two
paths from drifting in their anchor or intermediate-point normalization.

Validate:

- up/down reweighting closure for every affected process;
- total-yield behavior across the nuisance scan;
- ratio normalization at nominal, anchor, and intermediate nuisance points;
- the two-dimensional likelihood surface in parameter-of-interest and
  nuisance directions;
- the Asimov minimum at the generating point.
