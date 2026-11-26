"""把未确定的粗糙度范围交给灯光预览，并按最大误差选择下一观测角度。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .evidence import measure_highlight
from .geometry import Observation
from .params import default_parameters
from .render import render_sphere

DEFAULT_CANDIDATE_LIGHT_ANGLES_DEG: Tuple[float, ...] = (
    10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0
)


def peak_highlight_measure(
    image: np.ndarray, relative_threshold: float = 0.45
) -> Dict[str, Any]:
    """以峰值比例为阈值测量高光核心，适合比较不同粗糙度的高光尺寸。"""
    lum = np.asarray(image, dtype=np.float64)
    if lum.ndim == 3:
        lum = lum[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    peak = float(np.max(lum, initial=0.0))
    if peak <= 1e-12:
        threshold = 1.0
    else:
        threshold = peak * relative_threshold
    measure = measure_highlight(lum, threshold=threshold)
    return {
        **measure.to_dict(),
        "peak": peak,
        "relative_threshold": relative_threshold,
    }


def _light_vector(theta_deg: float, phi_deg: float = 0.0) -> np.ndarray:
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


@dataclass
class LightingPreviewPoint:
    angle_deg: float
    phi_deg: float
    nominal_highlight: Dict[str, Any]
    low_roughness_highlight: Optional[Dict[str, Any]] = None
    high_roughness_highlight: Optional[Dict[str, Any]] = None
    roughness_highlight_span: float = 0.0
    uncertainty: float = 0.0
    error: Optional[float] = None
    images: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        out = {
            "angle_deg": self.angle_deg,
            "phi_deg": self.phi_deg,
            "nominal_highlight": self.nominal_highlight,
            "low_roughness_highlight": self.low_roughness_highlight,
            "high_roughness_highlight": self.high_roughness_highlight,
            "roughness_highlight_span": self.roughness_highlight_span,
            "uncertainty": self.uncertainty,
            "error": self.error,
        }
        if include_images:
            out["images"] = self.images
        return out


def render_lighting_preview(
    params: Optional[Mapping[str, float]] = None,
    *,
    roughness_range: Optional[Tuple[float, float]] = None,
    angles_deg: Sequence[float] = DEFAULT_CANDIDATE_LIGHT_ANGLES_DEG,
    phi_deg: float = 0.0,
    size: int = 96,
    errors_by_angle: Optional[Mapping[float, float]] = None,
    include_images: bool = True,
) -> Dict[str, Any]:
    """渲染各灯光角度下的材质球，并给出粗糙度范围导致的高光尺寸区间。"""
    base = default_parameters()
    if params:
        base.update({k: float(v) for k, v in params.items()
                     if k in base or k.endswith("_r") or k.endswith("_g") or k.endswith("_b")})
    if roughness_range is None:
        roughness_range = (float(base["roughness"]), float(base["roughness"]))
    r_lo, r_hi = sorted((float(roughness_range[0]), float(roughness_range[1])))
    nominal_r = float(params.get("roughness", 0.5)) if params else 0.5
    nominal_r = min(max(nominal_r, r_lo), r_hi)

    points = []
    for angle in angles_deg:
        light = _light_vector(float(angle), phi_deg)
        nominal_img = render_sphere({**base, "roughness": nominal_r},
                                    size=size, light=light)
        nominal_h = peak_highlight_measure(nominal_img)
        images: Dict[str, np.ndarray] = {"nominal": nominal_img}
        low_h = high_h = None
        span = 0.0
        if r_hi - r_lo > 1e-8:
            low_img = render_sphere({**base, "roughness": r_lo},
                                    size=size, light=light)
            high_img = render_sphere({**base, "roughness": r_hi},
                                     size=size, light=light)
            low_h = peak_highlight_measure(low_img)
            high_h = peak_highlight_measure(high_img)
            # 粗糙端点的高光核心通常更大/更弥散；同时保留面积和等效直径差异。
            span = max(
                abs(low_h["area_fraction"] - high_h["area_fraction"]),
                abs(low_h["equivalent_diameter_fraction"]
                    - high_h["equivalent_diameter_fraction"]),
            )
            if include_images:
                images.update({"roughness_low": low_img,
                               "roughness_high": high_img})
        err = None
        if errors_by_angle is not None:
            err = _lookup_angle_error(float(angle), errors_by_angle)
        point = LightingPreviewPoint(
            angle_deg=float(angle),
            phi_deg=float(phi_deg),
            nominal_highlight=nominal_h,
            low_roughness_highlight=low_h,
            high_roughness_highlight=high_h,
            roughness_highlight_span=float(span),
            uncertainty=float(span),
            error=err,
            images=images if include_images else {},
        )
        points.append(point)
    return {
        "roughness_range": (r_lo, r_hi),
        "angles_deg": [float(a) for a in angles_deg],
        "points": points,
    }


# 常见命名别名。
lighting_preview = render_lighting_preview
create_lighting_preview = render_lighting_preview


def _lookup_angle_error(angle: float, errors_by_angle: Mapping[float, float]) -> Optional[float]:
    if angle in errors_by_angle:
        return float(errors_by_angle[angle])
    closest = min(errors_by_angle, key=lambda a: abs(float(a) - angle), default=None)
    if closest is not None and abs(float(closest) - angle) <= 0.51:
        return float(errors_by_angle[closest])
    return None


def observation_angle_errors(
    observations: Sequence[Observation],
    params: Mapping[str, float],
) -> Dict[float, float]:
    """计算每个光位的重渲染 RMSE，并以灯光极角为键。"""
    from .fitter import residuals

    residual = residuals(params, observations)
    errors: Dict[float, float] = {}
    for obs, res in zip(observations, residual):
        theta = float(np.degrees(np.arccos(np.clip(float(obs.light[2]), -1, 1))))
        rmse = float(np.sqrt(np.mean(res ** 2)))
        # 同角度多次观测时保留最大误差，符合“误差最大的照明角度”。
        errors[theta] = max(rmse, errors.get(theta, -np.inf))
    return errors


class NextObservationRecommendation(float):
    """数值上等于推荐角度，同时携带选择证据。"""

    def __new__(
        cls,
        angle_deg: float,
        *,
        error: Optional[float] = None,
        reason: str = "max_rerender_error",
        candidates: Optional[Mapping[float, float]] = None,
    ) -> "NextObservationRecommendation":
        obj = float.__new__(cls, float(angle_deg))
        obj.angle_deg = float(angle_deg)  # type: ignore[attr-defined]
        obj.error = error  # type: ignore[attr-defined]
        obj.reason = reason  # type: ignore[attr-defined]
        obj.candidates = dict(candidates or {})  # type: ignore[attr-defined]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angle_deg": self.angle_deg,
            "error": self.error,
            "reason": self.reason,
            "candidates": self.candidates,
        }


def _coerce_error_map(
    errors: Any,
) -> Dict[float, float]:
    if isinstance(errors, Mapping):
        return {float(k): float(v) for k, v in errors.items()}
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        out: Dict[float, float] = {}
        for item in errors:
            if isinstance(item, Observation):
                raise TypeError("Observation 序列请配合 params 使用")
            if isinstance(item, Mapping):
                angle = item.get("angle_deg", item.get("angle", item.get("theta_light_deg")))
                value = item.get("error", item.get("rmse", item.get("max_error")))
            else:
                angle, value = item[0], item[1]
            out[float(angle)] = max(float(value), out.get(float(angle), -np.inf))
        return out
    raise TypeError("errors 必须是映射、(angle,error) 序列或 Observation 序列")


def select_next_observation(
    errors: Any = None,
    *,
    observations: Optional[Sequence[Observation]] = None,
    params: Optional[Mapping[str, float]] = None,
    error_map: Optional[Mapping[float, float]] = None,
    candidates: Optional[Sequence[float]] = None,
    roughness_range: Optional[Tuple[float, float]] = None,
) -> NextObservationRecommendation:
    """选择误差最大的照明角度作为下一张观测。

    可直接传 ``{angle: error}``、``[(angle, error), ...]``，或传 observations
    与 params 现场重渲染计算误差。没有误差但给出粗糙度范围时，用灯光预览中
    的高光尺寸不确定性选点。
    """
    mapping: Dict[float, float] = {}
    if error_map is not None:
        mapping.update({float(k): float(v) for k, v in error_map.items()})
    if errors is not None:
        if isinstance(errors, Sequence) and errors and isinstance(errors[0], Observation):
            if params is None:
                raise ValueError("从观测计算角度误差时必须提供 params")
            mapping.update(observation_angle_errors(errors, params))
        else:
            mapping.update(_coerce_error_map(errors))
    if observations is not None:
        if params is None:
            raise ValueError("从观测计算角度误差时必须提供 params")
        mapping.update(observation_angle_errors(observations, params))

    if mapping:
        if candidates is not None:
            candidate_set = {round(float(a), 6) for a in candidates}
            filtered = {a: e for a, e in mapping.items()
                        if any(abs(a - c) <= 0.51 for c in candidate_set)}
            mapping = filtered or mapping
        angle = max(mapping, key=lambda a: mapping[a])
        return NextObservationRecommendation(
            angle, error=mapping[angle], candidates=mapping,
            reason="largest_rerender_error_light_angle",
        )

    preview = render_lighting_preview(
        params, roughness_range=roughness_range,
        angles_deg=candidates or DEFAULT_CANDIDATE_LIGHT_ANGLES_DEG,
        include_images=False,
    )
    by_uncertainty = {
        p.angle_deg: float(p.uncertainty) for p in preview["points"]
    }
    angle = max(by_uncertainty, key=lambda a: by_uncertainty[a])
    return NextObservationRecommendation(
        angle, error=by_uncertainty[angle], candidates=by_uncertainty,
        reason="largest_roughness_highlight_uncertainty",
    )


select_next_angle = select_next_observation
suggest_next_observation = select_next_observation
recommend_next_light = select_next_observation
