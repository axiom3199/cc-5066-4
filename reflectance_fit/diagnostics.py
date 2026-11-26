"""拟合诊断：平坦参数谷方向、反射曲线和材质预览有效性。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .brdf import shade_rgb
from .geometry import Observation
from .optimization import physical_bounds
from .params import PARAM_SPEC


@dataclass
class DirectionReport:
    index: int
    singular_value: float
    kind: str
    vector: Dict[str, float]
    dominant_parameters: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IdentifiabilityReport:
    singular_values: List[float]
    tolerance: float
    rank: int
    condition_number: float
    directions: List[DirectionReport] = field(default_factory=list)
    identifiable_parameters: List[str] = field(default_factory=list)
    flat_directions: List[Dict[str, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "singular_values": self.singular_values,
            "tolerance": self.tolerance,
            "rank": self.rank,
            "condition_number": self.condition_number,
            "directions": [d.to_dict() for d in self.directions],
            "identifiable_parameters": self.identifiable_parameters,
            "flat_directions": self.flat_directions,
            "notes": self.notes,
        }


def scaled_radiance_jacobian(
    params: Mapping[str, float],
    observations: Sequence[Observation],
    active: Sequence[str],
    *,
    rel_step: float = 1e-6,
) -> np.ndarray:
    """计算辐射度对归一化参数 u=(p-low)/(high-low) 的 Jacobian。

    归一化后每个参数跨越相同的 [0,1] 尺度，SVD 得到的耦合方向不会被
    roughness/albedo 等参数的物理单位和范围宽度扭曲。
    """
    active = tuple(active)
    base = dict(params)
    p0 = np.array([float(base[k]) for k in active], dtype=np.float64)
    widths = physical_bounds(active).width

    def radiance_vec(p: np.ndarray) -> np.ndarray:
        candidate = dict(base)
        candidate.update({k: float(v) for k, v in zip(active, p)})
        return shade_rgb(candidate, observations).reshape(-1)

    f0 = radiance_vec(p0)
    jac_physical = np.zeros((f0.size, len(active)), dtype=np.float64)
    for j, width in enumerate(widths):
        h = rel_step * max(1.0, abs(p0[j]), width)
        pp = p0.copy()
        pp[j] += h
        jac_physical[:, j] = (radiance_vec(pp) - f0) / h
    return jac_physical * widths[None, :]


def analyze_jacobian_valley(
    jacobian: np.ndarray,
    names: Sequence[str],
    *,
    rank_rtol: float = 1e-5,
) -> IdentifiabilityReport:
    """对尺度化 Jacobian 做 SVD，标出可辨方向和平坦/耦合谷方向。"""
    j = np.asarray(jacobian, dtype=np.float64)
    if j.ndim != 2:
        raise ValueError("jacobian 必须是二维数组")
    names = tuple(names)
    if j.shape[1] != len(names):
        raise ValueError("参数名数量与 Jacobian 列数不一致")
    if j.size == 0:
        return IdentifiabilityReport(
            singular_values=[], tolerance=0.0, rank=0,
            condition_number=float("inf"),
            notes=["空 Jacobian，无法核验可辨方向"],
        )
    _, singular, vt = np.linalg.svd(j, full_matrices=False)
    top = float(singular[0]) if singular.size else 0.0
    tol = max(j.shape) * np.finfo(float).eps * max(top, 1.0)
    tol = max(tol, rank_rtol * top)
    rank = int(np.sum(singular > tol))
    cond = float(top / singular[-1]) if singular[-1] > tol else float("inf")

    directions: List[DirectionReport] = []
    flat_directions: List[Dict[str, float]] = []
    for i, s in enumerate(singular):
        vector = {name: float(vt[i, j]) for j, name in enumerate(names)}
        order = sorted(names, key=lambda k: abs(vector[k]), reverse=True)
        dominant = [k for k in order if abs(vector[k]) >= 0.25][:3]
        kind = "identifiable" if s > tol else "flat_valley"
        report = DirectionReport(i, float(s), kind, vector, dominant)
        directions.append(report)
        if kind == "flat_valley":
            flat_directions.append(vector)

    identifiable_parameters: List[str] = []
    for d in directions:
        if d.kind == "identifiable":
            for key in d.dominant_parameters:
                if key not in identifiable_parameters:
                    identifiable_parameters.append(key)
    notes: List[str] = []
    if flat_directions:
        notes.append("存在平坦参数谷；沿最小右奇异向量的参数组合不可辨，"
                     "应报告可行范围而不是唯一单点。")
    if not np.isfinite(cond):
        notes.append("尺度化 Jacobian 秩不足或条件数无穷大。")
    return IdentifiabilityReport(
        singular_values=[float(x) for x in singular],
        tolerance=float(tol),
        rank=rank,
        condition_number=cond,
        directions=directions,
        identifiable_parameters=identifiable_parameters,
        flat_directions=flat_directions,
        notes=notes,
    )


def analyze_flat_valley(
    sample: Optional[Mapping[str, float] | Sequence[Observation] | np.ndarray] = None,
    observations: Optional[Sequence[Observation] | Sequence[str]] = None,
    params: Optional[Mapping[str, float]] = None,
    active: Optional[Sequence[str]] = None,
    *,
    jacobian: Optional[np.ndarray] = None,
    names: Optional[Sequence[str]] = None,
    objective: Optional[Callable[[np.ndarray], float]] = None,
    x0: Optional[Sequence[float]] = None,
    rank_rtol: float = 1e-5,
) -> IdentifiabilityReport:
    """核验平坦参数谷的可辨方向。

    支持直接给 ``jacobian``；给反射观测和参数；或给一个通用 ``objective``
    与 ``x0`` 做前向差分。位置参数也兼容 ``(jacobian, names)``、
    ``(params, observations)`` 和 ``(observations, params)`` 三种自然顺序。
    """
    if jacobian is None and isinstance(sample, np.ndarray):
        jacobian = sample
        if names is None and isinstance(observations, Sequence) and (
            not observations or isinstance(observations[0], str)
        ):
            names = observations  # type: ignore[assignment]
    if jacobian is not None:
        if names is None:
            raise ValueError("直接分析 Jacobian 时必须提供 names")
        return analyze_jacobian_valley(jacobian, names, rank_rtol=rank_rtol)

    if params is None and isinstance(sample, Mapping):
        params = sample
        if isinstance(observations, Sequence) and observations and \
                isinstance(observations[0], Observation):
            obs = observations
        else:
            obs = None
    else:
        obs = observations if not (
            isinstance(observations, Sequence) and observations and
            isinstance(observations[0], str)
        ) else None

    if obs is None and isinstance(sample, Sequence) and sample and \
            isinstance(sample[0], Observation):
        obs = sample  # type: ignore[assignment]
    if params is None and isinstance(observations, Mapping):
        params = observations  # type: ignore[assignment]

    if obs is not None and params is not None:
        if active is not None:
            keys = tuple(active)
        else:
            keys = tuple(k for k in PARAM_SPEC if k in params) or tuple(params.keys())
        jac = scaled_radiance_jacobian(params, obs, keys)
        return analyze_jacobian_valley(jac, keys, rank_rtol=rank_rtol)

    if objective is not None and x0 is not None:
        x = np.asarray(x0, dtype=np.float64)
        f0 = float(objective(x))
        h = 1e-6
        jac = np.zeros((1, x.size))
        for j in range(x.size):
            xp = x.copy()
            xp[j] += h
            jac[0, j] = (float(objective(xp)) - f0) / h
        parameter_names = tuple(names or [f"x{j}" for j in range(x.size)])
        return analyze_jacobian_valley(jac, parameter_names, rank_rtol=rank_rtol)

    raise ValueError("必须提供 jacobian、observations+params 或 objective+x0")


# 常见语义别名。
check_identifiable_directions = analyze_flat_valley
sensitivity_analysis = analyze_flat_valley


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def validate_reflection_curve(
    curve: Mapping[str, np.ndarray],
    *,
    min_peak: float = 1e-8,
) -> List[ValidationIssue]:
    """校验反射曲线数据完整、有限、非负且存在有效镜面峰。"""
    issues: List[ValidationIssue] = []
    required = ("theta_deg", "luminance", "diffuse_luminance",
                "specular_luminance", "specular_peak_angle")
    for key in required:
        if key not in curve:
            issues.append(ValidationIssue(
                f"curve_missing_{key}", "error", f"反射曲线缺少 {key}"))
    if issues:
        return issues
    for key in ("luminance", "diffuse_luminance", "specular_luminance"):
        arr = np.asarray(curve[key])
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            issues.append(ValidationIssue(
                f"curve_non_finite_{key}", "error",
                f"反射曲线 {key} 为空或包含非有限值"))
        elif np.min(arr) < -1e-10:
            issues.append(ValidationIssue(
                f"curve_negative_{key}", "error",
                f"反射曲线 {key} 出现非物理负值"))
    peak = float(np.max(curve["specular_luminance"]))
    if peak < min_peak:
        issues.append(ValidationIssue(
            "curve_specular_peak_missing", "warning",
            "镜面曲线几乎全黑；近镜面参数可能位于边界或拟合无效"))
    angle = float(curve["specular_peak_angle"])
    if not (-80.0 <= angle <= 80.0) or not np.isfinite(angle):
        issues.append(ValidationIssue(
            "curve_peak_angle_invalid", "error",
            f"镜面峰角度 {angle} 超出扫描范围或非有限"))
    return issues


validate_curve = validate_reflection_curve
check_reflection_curve = validate_reflection_curve


def validate_material_preview(
    image: np.ndarray,
    *,
    min_spatial_std: float = 1e-5,
    min_bright_fraction: float = 1e-4,
) -> List[ValidationIssue]:
    """校验材质预览是有效重渲染图，而不是空图或平铺纯色。"""
    issues: List[ValidationIssue] = []
    img = np.asarray(image, dtype=np.float64)
    if img.ndim not in (2, 3) or img.size == 0:
        return [ValidationIssue("preview_invalid_shape", "error",
                                "材质预览图像为空或维度无效")]
    if not np.all(np.isfinite(img)):
        issues.append(ValidationIssue(
            "preview_non_finite", "error", "材质预览包含 NaN/Inf"))
    lum = img if img.ndim == 2 else img[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    active = lum[lum > 1e-10]
    if active.size / lum.size < min_bright_fraction:
        issues.append(ValidationIssue(
            "preview_empty", "error", "材质预览几乎全黑，不能作为有效材质证据"))
    elif float(active.std()) < min_spatial_std:
        issues.append(ValidationIssue(
            "preview_flat_color", "warning",
            "材质预览缺少空间变化/高光，可能是平铺纯色或参数异常"))
    return issues


validate_preview = validate_material_preview
check_material_preview = validate_material_preview


def issues_to_marks(issues: Sequence[ValidationIssue]) -> List[Dict[str, str]]:
    return [issue.to_dict() for issue in issues]
