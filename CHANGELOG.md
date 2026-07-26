# Changelog

This project follows Semantic Versioning for the Python API. Configuration and
artifact formats have their own explicit schema versions.

## Unreleased

### Changed

- The `lhc` extra now installs `nsbi-common-utils` from the canonical
  `iris-hep/nsbi-lhc-toolkit` `main` branch; no user fork is used.
- Colab examples now use `MyDrive/hsbi-toolkit` as their shared root and have
  cleared outputs for clean reruns in the new workspace.

## 0.1.0 - 2026-07-26

### Added

- Initial NumPy-based data, configuration, expression, intensity, and
  diagnostic contracts.
- Version 1.0 JSON configuration schema.
- RealNVP and rational-quadratic-spline reference flows with restartable
  checkpoints, physical-input ONNX graphs, dynamic-batch parity checks, and
  closure/C2ST/tail diagnostics.
- Commit-pinned `nsbi-common-utils` ratio training and diagnostic delegation,
  with every deployed classifier and scaler available through verified ONNX.
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

- `nsbi-common-utils` remains an optional, commit-pinned backend rather than a
  vendored implementation.
- Learned deployment artifacts use ONNX as their portable contract; trusted
  training checkpoints may be retained only for resumption or migration.
