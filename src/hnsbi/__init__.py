"""Hybrid neural simulation-based inference.

The top-level namespace intentionally exposes the small, backend-independent
objects. Training backends are imported lazily from their respective modules.
"""

from ._version import __version__
from .asimov import AsimovBuilder, AsimovResult
from .config import ConfigError, ToolkitConfig, load_config
from .data import DataSource, EventBatch, WeightedEvents
from .diagnostics import effective_sample_size, weight_summary
from .fnf import (
    FactorizableDensity,
    FactorizableResidualStack,
    FNFAnchor,
    FNFArtifact,
    FNFDiagnosticReport,
    FNFResidualConfig,
    FNFStandardizer,
    FNFTrainer,
    FNFTrainingConfig,
    FNFTrainingResult,
    LogQuadraticYieldMorph,
    diagnose_fnf,
)
from .fnf_runtime import (
    FNFSystematic,
    NativeProcessDensity,
    load_workspace_fnf_systematics,
)
from .impacts import (
    ImpactEntry,
    ImpactResult,
    PullEntry,
    PullResult,
    compute_pulls,
    covariance_impacts,
    global_observable_impacts,
    plot_impacts,
    plot_pulls,
)
from .inference import (
    JaxLikelihood,
    MinuitInference,
    TestStatisticPoint,
    TestStatisticScan,
)
from .intensity import Component, IntensityModel, Parameter, RatioNormalizer
from .likelihood import (
    ExtendedUnbinnedLikelihood,
    FitResult,
    GaussianConstraint,
    ProfileScanResult,
)
from .multi_workspace import CombinedLikelihood
from .native_ratios import (
    NativeRatioBackend,
    NativeRatioEvaluator,
    OnnxNativeRatioMember,
    PiecewiseLinearCalibrator,
    load_native_ratio_ensemble,
)
from .project import (
    FNFModelArtifacts,
    NISWorkflowArtifacts,
    Project,
    RatioSetTrainingArtifacts,
    ReferenceTrainingArtifacts,
)
from .ratio_diagnostics import (
    RatioDiagnosticReport,
    diagnose_ratio,
    weighted_auc,
    weighted_ks,
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
    "CombinedLikelihood",
    "ConfigError",
    "DataSource",
    "EventBatch",
    "ExtendedUnbinnedLikelihood",
    "FNFAnchor",
    "FNFArtifact",
    "FNFDiagnosticReport",
    "FNFModelArtifacts",
    "FNFResidualConfig",
    "FNFSystematic",
    "FNFStandardizer",
    "FNFTrainer",
    "FNFTrainingConfig",
    "FNFTrainingResult",
    "FactorizableDensity",
    "FactorizableResidualStack",
    "FitResult",
    "GaussianConstraint",
    "ImpactEntry",
    "ImpactResult",
    "IntensityModel",
    "JaxLikelihood",
    "LogQuadraticYieldMorph",
    "NISWorkflowArtifacts",
    "NativeProcessDensity",
    "NativeRatioBackend",
    "NativeRatioEvaluator",
    "OnnxNativeRatioMember",
    "Parameter",
    "PiecewiseLinearCalibrator",
    "ProfileScanResult",
    "Project",
    "RatioSetTrainingArtifacts",
    "RatioDiagnosticReport",
    "RatioNormalizer",
    "ReferenceTrainingArtifacts",
    "RuntimeSystematic",
    "SystematicAnchor",
    "SystematicEvaluation",
    "SystematicRatioEvaluator",
    "ToolkitConfig",
    "TestStatisticPoint",
    "TestStatisticScan",
    "ToyGenerator",
    "ToyResult",
    "WeightedEvents",
    "WorkspaceModel",
    "MinuitInference",
    "PullEntry",
    "PullResult",
    "__version__",
    "effective_sample_size",
    "diagnose_fnf",
    "diagnose_ratio",
    "compute_pulls",
    "covariance_impacts",
    "global_observable_impacts",
    "load_native_ratio_ensemble",
    "load_config",
    "load_workspace_fnf_systematics",
    "load_workspace_model",
    "weight_summary",
    "weighted_auc",
    "weighted_ks",
    "plot_impacts",
    "plot_pulls",
]
