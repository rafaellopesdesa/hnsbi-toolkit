# Changelog

This project follows Semantic Versioning for the Python API. Configuration and
artifact formats have their own explicit schema versions.

## Unreleased

### Added

- Generation-time, legacy-equivalent LHC preselection with raw and selected
  Parquet bundles, fixed-cut provenance, and an exactly balanced conditional
  reference sample.
- Source-weight-derived nominal yields through
  `nominal_yield: {kind: source_weight_sum}`.

### Changed

- The hybrid reference-flow and neural-importance-sampling notebooks consume
  only the prepared selected samples and use conditional analytical truth.

## 0.2.0 - 2026-07-27

### Added

- Native weighted density-ratio training, calibration, diagnostics, workspace
  serialization, JAX autodifferentiation, iminuit fitting, and profile scans.
- YAML-first frequentist and dual hNPE--hNDE configuration, while preserving
  JSON as the serialized artifact format.
- Multi-nuisance FNF systematic morphing, including diagnostics and portable
  checkpoints.
- Pulls, global-observable and covariance impacts, and pyhf asymptotic,
  Asimov, and toy-based CLs limits.
- A complete generated LHC analysis and two reduced, DINGO-inspired
  gravitational-wave examples.

### Changed

- The LHC stack is self-contained and no longer has a runtime dependency on
  `nsbi-lhc-toolkit`.
- Portable FNF workspaces bind residuals to the exact checked reference and
  ratio manifests used in training, auto-load that stack for toys, and reject
  unauthenticated runtime overrides or FNF-component samplers.
- FNF training and runtime reconstruction reject live reference or ratio
  objects cross-bound to checked artifacts from a different training run.
- Fixed pyhf scans must bracket the observed and all five expected CLs
  crossings; clipped scan endpoints are never reported as limits.
- Colab examples use `MyDrive/hsbi-toolkit` as their shared root and have
  cleared outputs for clean reruns in the new workspace.

## 0.1.0 - 2026-07-26

### Added

- Initial NumPy-based data, configuration, expression, intensity, and
  diagnostic contracts.
- Version 1.0 JSON configuration schema.
- RealNVP and rational-quadratic-spline reference flows with restartable
  checkpoints, physical-input ONNX graphs, dynamic-batch parity checks, and
  closure/C2ST/tail diagnostics.
- Density-ratio training and diagnostics delegated to a commit-pinned
  `nsbi-common-utils` integration, with every deployed classifier and scaler
  available through verified ONNX.
- Direct and defensive-NIS Asimov construction with explicit ratio
  normalization, raw count, ESS, tail summaries, and checksummed workspace
  arrays.
- Systematic-aware direct and NIS Asimovs with a shared normalized morph
  evaluator, automatic workspace anchors, nonzero-nuisance auxiliary
  observations, and holdout-only NIS flow diagnostics.
- Formula-based intensity models, component-wise Poisson toys, normalized
  up/down shape systematics, workspace export, upstream JAX/Minuit routing,
  and native formula likelihood fits and scans.
- Native training, ONNX packaging, and lazy loading for all five dual
  hNPE--hNDE objects, plus posterior, evidence, prior-update, predictive, and
  selection calculations.
- Leakage-safe Bayesian `train`/`validation`/`holdout` labels that control
  flow, residual-ratio, and conditional-normalizer training and diagnostics,
  including independent validation data across all five stages.
- Complete frequentist and dual hNPE--hNDE configuration examples.
- Five Colab-ready archival paper notebooks with their exact helper closure
  and preserved scientific cells.
- Sphinx/Furo documentation for frequentist and Bayesian workflows, artifact
  and data contracts, backend integration, and Colab examples.
- Core Python 3.10--3.12, optional flow/Bayesian/upstream integration, wheel,
  and warnings-as-errors documentation workflows.

### Integration notes

- This historical release used `nsbi-common-utils` as an optional,
  commit-pinned backend. Version 0.2.0 replaces that runtime integration with
  native implementations.
- Learned deployment artifacts use ONNX as their portable contract; trusted
  training checkpoints may be retained only for resumption or migration.
