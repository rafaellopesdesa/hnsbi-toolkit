# Workspaces, fits, and scans

The `nsbi_common_utils` backend supplies its pyhf-like workspace builder,
JAX statistical model, and iminuit fit/scan interface. `hnsbi-toolkit` adds a
namespaced extension that identifies portable reference and ratio bundles,
feature signatures, normalization constants, and sample multiplier
expressions.

The integration does not copy the upstream serializer. When
`frequentist.workspace.base_config` is configured,
`Project.write_configured_workspace()` invokes the upstream
`WorkspaceBuilder` and then decorates its result. The base workspace must
contain exactly the configured channel and measurement, and its sample-name
set must match the hNSBI intensity. The decorator preserves unrelated
top-level metadata, replaces that channel and measurement with verified
unbinned arrays/modifiers, and adds the `hnsbi` extension.

When no `base_config` is supplied, the same method writes a minimal one-channel
workspace skeleton directly. In either case the integration:

1. validates a configured or existing upstream workspace;
2. resolves artifact bundle paths relative to the workspace;
3. translates simple multiplicative expressions into upstream norm-factor
   modifiers where possible;
4. uses the hNSBI intensity evaluator when a valid expression cannot be
   represented by the current upstream model;
5. delegates minimization and profile scans to the backend when the resulting
   workspace is semantically compatible.

## Multiplier expressions

Expressions such as `mu`, `mu * exp(alpha)`, or `1` use a restricted grammar.
Python `eval`, attribute access, indexing, imports, and arbitrary callables are
not accepted from JSON. Parameter names must be declared, and every component
multiplier is checked for finite, nonnegative values at runtime.

## Portable paths

Workspace JSON stores the pre-evaluated `.npy` arrays required by the upstream
runtime and, in the namespaced `hnsbi` extension, optional relative paths to
the reference and ratio bundle manifests. A workspace and its artifact
directory can therefore move together and still pass checksum and
feature-order verification.

`NsbiCommonUtilsInference.from_workspace()` resolves all upstream array paths
relative to the workspace file before constructing the upstream JAX model and
iminuit engine. Its `perform_fit()` and `profile_scan()` methods delegate
directly to `nsbi-common-utils`. General multiplier formulas use
`ExtendedUnbinnedLikelihood`; they are never silently approximated as an
upstream norm-factor product.

The backend currently has restrictions on nonlinear parameterization and
cross-channel sample structure. See the dedicated backend caveats before
interpreting a successful serialization as model validation.

`hnsbi.upstream_compatible` is false for general formulas, nonstandard or
off-center Gaussian constraints, and normalized shape-systematic workspaces.
`Project.workspace_runtime()` routes those cases to the toolkit's
`ExtendedUnbinnedLikelihood`; compatible models use
`NsbiCommonUtilsInference`. The workspace's intensity fingerprint and all
array/model manifests are rechecked before evaluation.

When an Asimov is generated away from a constrained nuisance's nominal
constraint mean, `hnsbi.auxiliary_observations` records the auxiliary
measurement at the generating truth. The native likelihood loads it as the
Gaussian constraint center while retaining the model's nominal constraint
metadata and intensity fingerprint. This makes the full event-plus-constraint
score close at the truth; the upstream runtime is not used because it cannot
represent that auxiliary observation.
