# hnsbi-toolkit

`hnsbi-toolkit` packages self-contained hybrid neural density-estimation
(hNDE) and dual hNPE--hNDE workflows. YAML describes the human-authored
analysis; JSON, ONNX, NumPy, and native checkpoints provide stable machine
artifacts.

The frequentist path targets unbinned LHC-style intensity models. The Bayesian
path freezes five complementary learned objects and supports both
posterior-side and likelihood-side inference. In both cases, learned
surrogates remain approximate: normalization, closure, effective sample size,
and tail diagnostics are part of the inference contract.

```{toctree}
:maxdepth: 2
:caption: Start here

getting-started
concepts
```

```{toctree}
:maxdepth: 2
:caption: Frequentist workflows

frequentist/index
frequentist/reference
frequentist/ratios
frequentist/workspaces
frequentist/toys
frequentist/asimov
frequentist/nis
frequentist/systematics
frequentist/fnf
frequentist/impacts_limits
```

```{toctree}
:maxdepth: 2
:caption: Bayesian workflows

bayesian/index
bayesian/training
bayesian/recipes
bayesian/validation
```

```{toctree}
:maxdepth: 2
:caption: Specifications and examples

specifications/data
specifications/configuration
specifications/artifacts
specifications/native-backend
examples/colab
examples/lhc
examples/dingo
```

```{toctree}
:maxdepth: 2
:caption: Python API

api/index
```
