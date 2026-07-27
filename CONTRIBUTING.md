# Contributing

Thank you for helping make `hnsbi-toolkit` reliable scientific software.

## Development setup

Use a supported Python version (3.10--3.12) and install the core test and
documentation dependencies:

```bash
python -m pip install -e '.[test,docs]'
python -m pip install jsonschema
```

Optional training stacks are deliberately separate:

```bash
python -m pip install -e '.[data,flows,plots]'
python -m pip install -e '.[lhc]'
python -m pip install -e '.[bayes]'
```

## Checks

Before opening a pull request, run:

```bash
python -m pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The complete example configurations must validate against
`schemas/toolkit.schema.json`. If a configuration field changes, update the
schema, loader checks, both relevant examples, documentation, and tests in the
same pull request.

## Scientific changes

A change to a likelihood, normalization, weight, interpolation, proposal, or
sampling rule needs:

- the mathematical target in the docstring or documentation;
- an analytic or simulator-backed regression test;
- independent train/validation data where a learned object is involved;
- explicit behavior for nonfinite values, negative measures, and low ESS;
- a note in `CHANGELOG.md`.

Never repair an *independent validation* closure by shifting a fitted curve or
refitting its normalizer on that validation sample. Same-support normalization
is an intentional finite-quadrature construction for Asimov samples, but its
exact identity must not be presented as independent surrogate validation.

## LHC implementation boundary

Keep ratio training, diagnostics, workspace construction, and fitting within
the versioned `hnsbi` contracts. The package must remain installable without a
Git checkout or a runtime dependency on `nsbi-lhc-toolkit`. When adapting an
algorithm from another project, preserve its license notice, document the
provenance, and add an independent regression test for the local
implementation.

## Public interfaces and compatibility

- Keep top-level imports small and backend independent.
- Use keyword-only arguments for new optional behavior.
- Return typed result objects with numerical diagnostics.
- Version configuration and artifact schemas independently.
- Emit deprecation warnings before removing a documented name or field.
- Do not load untrusted pickle, joblib, or Torch checkpoint files through the
  default inference path.

## Pull-request checklist

- [ ] Tests cover the new behavior and failure modes.
- [ ] Documentation builds with warnings treated as errors.
- [ ] User-facing YAML and serialized JSON examples remain valid.
- [ ] ONNX changes include native/ONNX parity tests.
- [ ] Seeds and split provenance are recorded.
- [ ] No generated data, model bundle, credential, or private path was added.
- [ ] The changelog describes user-visible behavior.
