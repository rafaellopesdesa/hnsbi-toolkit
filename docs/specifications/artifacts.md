# Artifact contracts

The toolkit emits several small, purpose-specific manifests rather than one
universal directory named `manifest.json`. Every manifest contains portable
relative paths, file sizes, SHA-256 digests, its artifact type, schema and
package versions, and type-specific metadata. Loaders verify both bytes and
semantic roles before constructing a runtime.

## Reference flow

`Project.train_reference()` writes a restartable native checkpoint, three
physical-input ONNX graphs, parity results, and flow diagnostics. With the
default prefix the relevant files are:

```text
reference/
├── reference_flow.pt
├── reference_flow.pt.manifest.json
├── reference_flow.log_prob.onnx
├── reference_flow.log_prob.onnx.manifest.json
├── reference_flow.base_to_data.onnx
├── reference_flow.base_to_data.onnx.manifest.json
├── reference_flow.data_to_base.onnx
├── reference_flow.data_to_base.onnx.manifest.json
├── reference_flow.onnx_parity.json
├── reference_flow.manifest.json
└── diagnostics/
```

The bundle manifest has artifact type `reference-flow-onnx-bundle`. It records
the ordered observation features, architecture, conditional-context signature,
and that the affine standardizer is embedded in each graph. The graph roles
are:

| Role | Meaning |
|---|---|
| `log-prob-onnx` | physical features $\rightarrow \log q(x)$ |
| `base-to-data-onnx` | caller-supplied standard-normal noise $\rightarrow x$ |
| `data-to-base-onnx` | physical features $\rightarrow$ latent code |

Sampling randomness intentionally remains in the caller. This makes ONNX
generation reproducible for a fixed latent array.

The native `reference_flow.pt` is a PyTorch checkpoint. Its checksum sidecar is
mandatory under the default verified loader, and the restricted
`weights_only=True` path is used. Loading an old unrestricted pickle requires
the explicit `allow_unsafe_pickle=True` trust decision.

## Density-ratio ensemble

Each configured physics sample receives its own directory:

```text
ratios/signal/
├── ratio_ensemble.manifest.json
├── member_000/
│   ├── model0.onnx
│   ├── model0.onnx.manifest.json
│   ├── model_scaler0.onnx
│   ├── model_scaler0.onnx.manifest.json
│   ├── model_scaler0.bin
│   ├── member.manifest.json
│   ├── onnx_parity.json
│   └── diagnostics/
└── member_001/
    └── ...
```

The root artifact type is `density-ratio-ensemble`; each member has type
`nsbi-common-utils-ratio-member`. The root manifest records the numerator and
denominator names, ordered features, exact training configuration, backend,
independent class-weight normalization, and
`arithmetic-mean-of-ratios` reduction. Member manifests bind the ONNX scaler
and classifier to those identities and record the native artifacts retained
for provenance.

Portable inference uses `scaler ONNX -> classifier ONNX -> ratio`. The joblib
scaler exists because the upstream trainer writes it, but the default loader
does not deserialize an untrusted pickle. Loading a serialized calibrator
likewise requires an explicit unsafe-pickle opt-in.

The independent process normalizers
$C_s=\mathbb E_q[\widehat r_s]$ are returned in the
`RatioSetTrainingArtifacts.normalizer` object and written as
`ratios/ratio_normalization.json` with the
`ratio_normalization.json.manifest.json` sidecar of artifact type
`density-ratio-normalizer`. They are also copied into each workspace's JSON
extension and Asimov metadata; they are not hidden inside a classifier graph.

## Asimov and workspace arrays

`AsimovResult.write_nsbi_arrays()` writes:

```text
arrays/
├── asimov_weights.npy
├── reference_weights.npy
├── ratio_signal.npy
├── ratio_background.npy
├── asimov_metadata.json
└── asimov_arrays.manifest.json
```

The `asimov-array-bundle` manifest records the feature signature, row count,
sample names, and intensity fingerprint. The workspace loader verifies this
manifest before using any NumPy array. NumPy object arrays and pickle loading
are never enabled.

## Systematic anchors

For component `signal` and nuisance `alpha`, a
`SystematicAnchor.write_workspace_modifier()` call writes:

```text
signal_alpha_up.npy
signal_alpha_down.npy
signal_alpha.manifest.json
```

The `systematic-anchor` manifest binds the arrays to the component, nuisance,
interpolation convention, row count, and up/down yield factors. Workspace
construction verifies that its modifier paths and metadata refer to exactly
those files.

## Native dual hNPE--hNDE bundle

`Project.train_dual()` with no injected backend trains and packages all five
objects under `bayesian.output_bundle`:

```text
dual_model/
├── dual_model.manifest.json
├── q_phi/
│   ├── q_phi.pt
│   ├── q_phi.log_prob.onnx
│   ├── q_phi.base_to_data.onnx
│   ├── q_phi.data_to_base.onnx
│   ├── onnx_parity.json
│   └── artifact.manifest.json
├── r_p/
│   ├── training/
│   │   ├── ratio_ensemble.manifest.json
│   │   └── member_000/log_ratio.onnx
│   ├── onnx_parity.json
│   └── artifact.manifest.json
├── q_eta/
│   └── ...
├── r_c/
│   └── ...
└── z_c/
    ├── log_normalization.onnx
    ├── mc_targets.npz
    ├── training_history.json
    ├── onnx_parity.json
    └── artifact.manifest.json
```

Ensemble member filenames can differ when an established ratio backend is
injected; the manifest roles, rather than guessed filenames, are the stable
contract. `dual_model.manifest.json` records the ordered $\theta$ and
observation signatures, graph tensor names, transforms and Jacobian
conventions, ensemble reductions, posterior-ratio denominator provenance, and
source/configuration provenance. `verify_dual_artifact_manifest()` checks all
five specifications and graph digests. `load_dual_model()` then creates lazy
ONNX Runtime adapters.

## Reproducibility and trust

Do not casually commit large learned bundles. Publish them in a versioned
release or data archive and retain their manifests beside the files. Record
seeds, group-safe split identifiers, source identifiers, and software versions
without embedding private absolute paths.

Checksums establish integrity, not scientific validity. A verified artifact
still requires independent closure, tail, normalization, and ONNX-parity
evidence appropriate to its role.
