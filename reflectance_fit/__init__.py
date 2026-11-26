"""受控材质反射拟合器。

模块概览
--------
- ``brdf``       : GGX Cook-Torrance 反射求值（真正的 BRDF，不是经验表）。
- ``params``     : 固定的参数空间、取值范围与颜色空间约定。
- ``geometry``   : 已知几何/相机/光照的观测条件与向量运算。
- ``observation``: 由真值参数渲染多角度观测，并注入噪声与标定误差。
- ``fitter``     : 留出角度的优化拟合、歧义检测与可行参数范围。
- ``render``     : 材质球预览渲染与反射曲线计算。
- ``gui``        : 反射拟合图形界面（区域选取、拟合、曲线与预览）。
"""

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
from .fitter import FitResult, fit_brdf

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
    "FitResult",
    "fit_brdf",
    "vec",
]
