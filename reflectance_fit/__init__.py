"""受控材质反射拟合器。"""

from .params import (
    PARAM_SPEC,
    COLOR_SPACE,
    fit_params_to_physical,
    physical_to_fit_params,
    default_parameters,
    clip_physical,
)
from .brdf import brdf_rgb, shade_rgb, microfacet_components
from .geometry import Observation, make_observation_set, vec
from .fitter import (
    FITTER_IMPLEMENTATION_VERSION,
    FitResult,
    fit_brdf,
    fit_brdf_legacy,
    fit_reflection,
)
from .optimization import (
    DEFAULT_BACKEND_ID,
    LEGACY_BACKEND_ID,
    OPTIMIZER_CONTRACT_VERSION,
    minimize_bounded,
    normalize_backend,
)
from .evidence import (
    MigrationDecision,
    accept_migration,
    boundary_report,
    check_parameter_bounds,
    highlight_diameter,
    highlight_size,
    measure_highlight,
    rerender_rmse_evidence,
)
from .diagnostics import (
    analyze_flat_valley,
    analyze_jacobian_valley,
    validate_material_preview,
    validate_reflection_curve,
)
from .lighting import (
    render_lighting_preview,
    select_next_angle,
    select_next_observation,
    suggest_next_observation,
)
from .project import (
    MaterialProject,
    load_material_asset,
    load_material_project,
    load_project,
    save_project,
)
from .asset_editor import AssetEditor, MaterialAssetEditor, MaterialEditor
from .migration import (
    OPTIMIZER_MIGRATION_VERSION,
    compare_backends,
    compare_optimizer_backends,
    migration_version,
)

__version__ = "0.2.0"
__all__ = [
    "PARAM_SPEC",
    "COLOR_SPACE",
    "fit_params_to_physical",
    "physical_to_fit_params",
    "default_parameters",
    "clip_physical",
    "brdf_rgb",
    "shade_rgb",
    "microfacet_components",
    "Observation",
    "make_observation_set",
    "vec",
    "FitResult",
    "fit_brdf",
    "fit_brdf_legacy",
    "fit_reflection",
    "FITTER_IMPLEMENTATION_VERSION",
    "DEFAULT_BACKEND_ID",
    "LEGACY_BACKEND_ID",
    "OPTIMIZER_CONTRACT_VERSION",
    "minimize_bounded",
    "normalize_backend",
    "MigrationDecision",
    "accept_migration",
    "boundary_report",
    "check_parameter_bounds",
    "highlight_size",
    "highlight_diameter",
    "measure_highlight",
    "rerender_rmse_evidence",
    "analyze_flat_valley",
    "analyze_jacobian_valley",
    "validate_reflection_curve",
    "validate_material_preview",
    "render_lighting_preview",
    "select_next_observation",
    "select_next_angle",
    "suggest_next_observation",
    "MaterialProject",
    "load_project",
    "load_material_project",
    "load_material_asset",
    "save_project",
    "MaterialAssetEditor",
    "AssetEditor",
    "MaterialEditor",
    "OPTIMIZER_MIGRATION_VERSION",
    "compare_optimizer_backends",
    "compare_backends",
    "migration_version",
    "__version__",
]
