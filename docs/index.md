# hnsbi-toolkit

`hnsbi-toolkit` packages the hybrid neural density-estimation (hNDE) and dual
hNPE--hNDE workflows used in the accompanying paper and tutorials. It combines
a normalized, generative flow with classifier density ratios while keeping
normalization and validation explicit.

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
specifications/nsbi-backend
examples/colab
```

```{toctree}
:maxdepth: 2
:caption: Python API

api/index
```
