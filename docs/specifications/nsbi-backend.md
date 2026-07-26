# `nsbi-common-utils` backend

`hnsbi-toolkit` deliberately reuses `nsbi-common-utils` for LHC density-ratio
training and diagnostics, workspace construction, the JAX likelihood model,
and iminuit fitting/scanning.

Upstream repository:
[`iris-hep/nsbi-lhc-toolkit`](https://github.com/iris-hep/nsbi-lhc-toolkit)

Tracked branch: `main`

The dependency currently has no released package version. Its repository
package metadata declares only NumPy even though the runtime uses pandas,
uproot, Awkward, PyYAML, Torch, Lightning, scikit-learn, joblib, ONNX,
ONNX Runtime, JAX, iminuit, and plotting libraries. The hNSBI `lhc` extra
therefore supplies the complete runtime stack and installs the upstream
package with a PEP 508 direct reference. CI resolves and tests the current
upstream `main` on every run. A frozen production release may additionally
record the resolved commit in its environment lockfile.

## Adapter boundary

Only one private adapter imports backend implementation names such as
`density_ratio_trainer`, `WorkspaceBuilder`, `sbi_parametric_model`, and
`inference`. User code should never rely on those lowercase classes through
the `hnsbi` namespace.

The adapter:

- converts bounded Arrow batches into the pandas representation expected by
  the trainer;
- delegates established ratio diagnostics;
- converts backend ONNX plus fitted preprocessing into a portable hNSBI
  bundle;
- validates and decorates an upstream workspace;
- translates backend fit/scan output into stable result objects.

## Current caveats

- the upstream dataset helper is ROOT/pandas-specific;
- configuration schema validation is incomplete;
- its unbinned workspace convention points to pre-evaluated `.npy` ratios and
  Asimov weights;
- nonlinear sample multipliers are not generally supported;
- some assumptions are derived from the first channel or common sample list;
- joblib preprocessing is sensitive to scikit-learn versions;
- API compatibility is not guaranteed by a release.

Integration tests against upstream `main` are therefore mandatory. If a
required provider hook is missing, prefer a small upstream contribution over
copying the workspace or training implementation into this repository.

Workspace delegation is conditional, not automatic. General multiplier
formulas, normalized shape-systematic workspaces, and constraints the upstream
runtime cannot represent are marked incompatible and evaluated through
`ExtendedUnbinnedLikelihood`.
