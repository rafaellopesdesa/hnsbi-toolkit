"""Hybrid neural simulation-based inference.

The top-level namespace intentionally exposes the small, backend-independent
objects. Training backends are imported lazily from their respective modules.
"""

from ._version import __version__
from .asimov import AsimovBuilder, AsimovResult
from .config import ConfigError, ToolkitConfig, load_config
from .data import DataSource, EventBatch, WeightedEvents
from .diagnostics import effective_sample_size, weight_summary
from .intensity import Component, IntensityModel, Parameter, RatioNormalizer
from .likelihood import (
    ExtendedUnbinnedLikelihood,
    FitResult,
    GaussianConstraint,
    ProfileScanResult,
)
from .project import (
    NISWorkflowArtifacts,
    Project,
    RatioSetTrainingArtifacts,
    ReferenceTrainingArtifacts,
)
from .systematics import (
    RuntimeSystematic,
    SystematicAnchor,
    SystematicEvaluation,
    SystematicRatioEvaluator,
)
from .toys import ToyGenerator, ToyResult
from .workspace import WorkspaceModel, load_workspace_model

__all__ = [
    "AsimovBuilder",
    "AsimovResult",
    "Component",
    "ConfigError",
    "DataSource",
    "EventBatch",
    "ExtendedUnbinnedLikelihood",
    "FitResult",
    "GaussianConstraint",
    "IntensityModel",
    "NISWorkflowArtifacts",
    "Parameter",
    "ProfileScanResult",
    "Project",
    "RatioSetTrainingArtifacts",
    "RatioNormalizer",
    "ReferenceTrainingArtifacts",
    "RuntimeSystematic",
    "SystematicAnchor",
    "SystematicEvaluation",
    "SystematicRatioEvaluator",
    "ToolkitConfig",
    "ToyGenerator",
    "ToyResult",
    "WeightedEvents",
    "WorkspaceModel",
    "__version__",
    "effective_sample_size",
    "load_config",
    "load_workspace_model",
    "weight_summary",
]
