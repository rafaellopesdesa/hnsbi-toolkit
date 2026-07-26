# Configuration specification

The canonical schema is `schemas/toolkit.schema.json`, using JSON Schema Draft
2020-12. `schema_version` is required and currently equals `"1.0"`.

The top-level fields are:

| Field | Required | Meaning |
|---|---:|---|
| `schema_version` | yes | serialized contract version |
| `features` | yes | ordered, unique observation features |
| `output_dir` | no | default artifact root |
| `frequentist` | one workflow | LHC-style intensity workflow |
| `bayesian` | one workflow | dual hNPE--hNDE workflow |

At least one workflow section is required; a project may contain both.
Unknown properties are rejected so spelling errors do not become ignored
settings.

Load either a path or a dictionary:

```python
from hnsbi.config import ToolkitConfig

from_file = ToolkitConfig.load("analysis.json")
from_dict = ToolkitConfig.load(
    {
        "schema_version": "1.0",
        "features": ["x1", "x2"],
        "bayesian": {
            "...": "see examples/configs/dual_complete.json"
        },
    }
)
```

The second abbreviated object illustrates the API only and is not a valid
complete workflow.

## Versioning

Configuration and artifact versions evolve independently of the Python
package. A loader may migrate an older known schema with a deprecation warning,
but must reject a newer unknown major schema. Migrations are deterministic and
must not infer scientific choices such as an NIS design or prior.

## Expression safety

Sample multipliers use the toolkit expression parser. JSON never executes
Python code. Names, operators, and functions outside the allow-list fail at
configuration load or model construction.

## Systematic rate anchors

Each `frequentist.systematics[].variations[]` object may set `yield_up` and
`yield_down`. These are dimensionless yield factors relative to the affected
sample's nominal yield, not absolute event counts. Omitted anchors are inferred
from the ratio of integrated variation and nominal MC weights. For that
fallback to be physical, the input event weights—including any weights in a
bounded training subset—must preserve the samples' integrated normalization.
Use explicit anchors when the training tables contain shape-only weights.

Both anchors must be finite and nonnegative. The `nsbi_code4p` interpolation
requires them to be strictly positive; `linear` is the only interpolation that
permits a zero-rate anchor.
