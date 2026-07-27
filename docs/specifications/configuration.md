# Configuration specification

YAML is the primary human-authored interface. JSON and Python mappings remain
fully supported, and all three representations validate against
`schemas/toolkit.schema.json` (JSON Schema Draft 2020-12).
`schema_version` is required and currently equals `"2.0"`.

The top-level fields are:

| Field | Required | Meaning |
|---|---:|---|
| `schema_version` | yes | serialized contract version |
| `features` | yes | ordered, unique observation features |
| `output_dir` | no | default artifact root |
| `frequentist` | one workflow | LHC-style intensity workflow |
| `bayesian` | one workflow | dual hNPE--hNDE workflow |

At least one workflow section is required; a project may contain both.
Unknown properties and duplicate YAML or JSON keys are rejected so spelling
errors cannot become ignored settings. NaN and infinity are rejected in every
representation.

```python
from hnsbi.config import ToolkitConfig

from_yaml = ToolkitConfig.load("analysis.yaml")
from_mapping = ToolkitConfig.load(
    {
        "schema_version": "2.0",
        "features": ["x1", "x2"],
        "bayesian": {
            "...": "see examples/dingo_bbh/dual.yaml"
        },
    }
)
```

The abbreviated mapping illustrates the API only; it is not a complete valid
workflow.

`ToolkitConfig.dump("analysis.yaml")` writes YAML and
`ToolkitConfig.dump_json("analysis.json")` writes strict JSON. The file suffix
selects the format. Workspaces and artifact manifests remain JSON even when
their originating analysis was YAML.

## Frequentist input contract

The YAML supplies the reference dataset, every nominal physics sample, all
up/down variations, parameter declarations, yield multipliers, Asimov or NIS
settings, and workspace destination. `frequentist.ratios.backend` is
`native`; classifier preprocessing and calibration are part of the saved
artifact.

The full contract is exercised by
`examples/lhc_analysis/analysis.yaml`, including response, resolution, and
theory variations and the pyhf projection used for CLs limits.
`frequentist.fnf.models` provides the YAML alternative for sample-specific,
multi-nuisance normalized morphs; see {doc}`../frequentist/fnf`.

## Bayesian input contract

The YAML supplies `theta_features`, the data drawn under the $\rho$, $\nu$,
and $\kappa$ designs, the corresponding design distributions, and training
configuration for $q_\phi$, $\widehat r_{\rm P}$, $q_\eta$,
$\widehat r_{\rm C}$, and $\widehat Z_{\rm C}$. See the BBH and BNS
`dual.yaml` examples for complete contracts.

## Versioning

Configuration and artifact versions evolve independently of the Python
package. A loader may migrate an older known schema with a deprecation
warning, but must reject a newer unknown major schema. Migrations are
deterministic and must not infer scientific choices such as an NIS design or
prior.

## Expression safety

Sample multipliers use the toolkit expression parser. Configuration files
never execute Python code. Names, operators, and functions outside the
allow-list fail at configuration load or model construction.

## Systematic rate anchors

Each `frequentist.systematics[].variations[]` object may set `yield_up` and
`yield_down`. These are dimensionless factors relative to the affected
sample's nominal yield, not absolute event counts. Omitted anchors are
inferred from the ratio of integrated variation and nominal MC weights. Use
explicit anchors when input tables carry shape-only weights.

Both anchors must be finite and nonnegative. The `nsbi_code4p` interpolation
requires strictly positive anchors; `linear` is the only interpolation that
permits a zero-rate anchor.
