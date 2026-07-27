# Systematic variations

The YAML interface names the nuisance parameter, every affected nominal
sample, its up/down data sources, optional yield anchors, and interpolation
convention. `Project.train_systematics()` uses the same native classifier
stack and diagnostics as nominal ratio training.

Variation and nominal sources use the same leakage-safe partition contract as
nominal ratio training. Configure `split_column` for fixed scientific
partitions and `group_column` (or `event_id_column`) for correlated rows.
Shared identifiers in nominal/up/down samples are assigned to the same
partition, and inconsistent explicit labels are rejected.

`Project.build_systematic_modifiers()` evaluates both ensembles on the exact
workspace reference support, separates their integrated yield factors from
their shape ratios, and writes checksummed `SystematicAnchor` arrays.

`yield_up` and `yield_down` are multiplicative factors relative to the
sample's nominal yield. They override rate information inferred from event
tables. If omitted, the toolkit uses

$$
y_{\rm up/down}
=
\frac{\sum_{i\in{\rm up/down}} w_i}
     {\sum_{j\in{\rm nominal}} w_j}.
$$

That fallback requires weights which preserve physical sample normalization.
Use explicit anchors for shape-only inputs. `nsbi_code4p` requires strictly
positive anchors; `linear` is the only interpolation that permits zero.

## Shape normalization

The likelihood interpolates rate and shape separately, then renormalizes the
joint shape under the fixed reference quadrature at every nuisance value.
This preserves the configured extended yield at nominal, anchor, and
intermediate points. `ToyGenerator`, `AsimovBuilder`,
`NISAsimovBuilder`, and `ExtendedUnbinnedLikelihood` all use the same
`SystematicRatioEvaluator`.

## FNF alternative

Use a factorizable normalizing flow when a continuous normalized
multi-nuisance density is preferable to independent endpoint interpolation.
The FNF is exactly the identity at the nominal point and keeps the yield morph
separate from the normalized shape. See {doc}`fnf`.

For either representation, validate:

- up/down closure for every affected process;
- total yield across the nuisance scan;
- shape normalization at anchors and interpolation points;
- POI--nuisance likelihood surfaces;
- the Asimov minimum at the generating point.
