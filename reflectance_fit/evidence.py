"""真实重渲染评价、边界检查与高光尺寸证据。

优化器的 ``success`` 只说明求解过程是否正常停止；材质是否可接受必须由
独立的重渲染误差、参数边界和留出光位高光尺寸决定。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .brdf import shade_rgb
from .geometry import Observation
from .params import PARAM_SPEC


@dataclass(frozen=True)
class BoundaryStatus:
    key: str
    value: float
    low: float
    high: float
    valid: bool
    location: str
    normalized: float
    distance_to_bound: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_parameter_bounds(
    params: Mapping[str, float],
    active: Optional[Sequence[str]] = None,
    *,
    tolerance: float = 1e-7,
    near_tolerance: float = 1e-3,
) -> Dict[str, BoundaryStatus]:
    """检查参数是否在固定硬范围内，并标记贴边/越界。"""
    keys = tuple(active) if active is not None else tuple(PARAM_SPEC)
    out: Dict[str, BoundaryStatus] = {}
    for key in keys:
        if key not in PARAM_SPEC:
            raise KeyError(f"未知参数: {key}")
        pr = PARAM_SPEC[key]
        value = float(params[key])
        width = max(pr.high - pr.low, 1e-12)
        normalized = (value - pr.low) / width
        if value < pr.low - tolerance or value > pr.high + tolerance:
            location = "outside"
            valid = False
        elif abs(value - pr.low) <= max(tolerance, near_tolerance * width):
            location = "lower_boundary"
            valid = True
        elif abs(value - pr.high) <= max(tolerance, near_tolerance * width):
            location = "upper_boundary"
            valid = True
        else:
            location = "interior"
            valid = True
        out[key] = BoundaryStatus(
            key=key,
            value=value,
            low=pr.low,
            high=pr.high,
            valid=valid,
            location=location,
            normalized=float(normalized),
            distance_to_bound=float(min(abs(value - pr.low), abs(value - pr.high))),
        )
    return out


def boundary_report(
    params: Mapping[str, float], active: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    statuses = check_parameter_bounds(params, active)
    invalid = [key for key, st in statuses.items() if not st.valid]
    at_boundary = [key for key, st in statuses.items()
                   if st.location in {"lower_boundary", "upper_boundary"}]
    parameter_reports: Dict[str, Any] = {}
    for key, st in statuses.items():
        item = st.to_dict()
        item["within_bounds"] = st.valid
        item["status"] = st.location
        parameter_reports[key] = item
    valid = not invalid
    return {
        "all_within_bounds": valid,
        "within_bounds": valid,
        "valid": valid,
        "ok": valid,
        "invalid_parameters": invalid,
        "out_of_bounds": invalid,
        "violations": invalid,
        "at_boundary": at_boundary,
        "parameters": parameter_reports,
    }


def rerender_rmse_evidence(
    params: Mapping[str, float],
    observations: Sequence[Observation],
    *,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """用同一条 BRDF 渲染路径重新计算 RMSE，作为执行证据。"""
    from .fitter import weighted_rmse

    if not observations:
        return {
            "available": False,
            "reason": "no_observations",
            "rmse": None,
            "n_observations": 0,
        }
    predicted = shade_rgb(params, observations)
    measured = np.stack([o.radiance for o in observations])
    if weights is None:
        weights = np.array([o.weight for o in observations], dtype=np.float64)
    rmse = weighted_rmse(predicted, measured, weights)
    per_observation = np.sqrt(np.mean((predicted - measured) ** 2, axis=1))
    return {
        "available": True,
        "reason": "rerendered_with_shade_rgb",
        "rmse": float(rmse),
        "n_observations": int(len(observations)),
        "per_observation_rmse": [float(x) for x in per_observation],
        "max_observation_rmse": float(np.max(per_observation)),
        "predicted_shape": list(predicted.shape),
    }


def _luminance(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        lum = img
    else:
        if img.shape[-1] == 4:
            img = img[..., :3]
        lum = img @ np.array([0.2126, 0.7152, 0.0722])
    if lum.max(initial=0.0) > 1.5:
        raw = np.asarray(image, dtype=np.float64)
        if raw.ndim == 3:
            raw = raw[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        lum = raw / 255.0
    return lum


@dataclass(frozen=True)
class HighlightMeasure:
    pixel_count: int
    total_pixels: int
    area_fraction: float
    width_fraction: float
    height_fraction: float
    equivalent_diameter_fraction: float
    centroid_xy: Tuple[float, float]
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["centroid_xy"] = list(self.centroid_xy)
        return d


def measure_highlight(
    image: np.ndarray,
    *,
    threshold: Optional[float] = None,
    relative_peak: float = 0.45,
    percentile: Optional[float] = None,
    min_fraction: float = 1e-5,
) -> HighlightMeasure:
    """测量图像中亮高光区域的尺寸。

    默认阈值为图像峰值的 ``relative_peak`` 倍，因此不同粗糙度下高光弥散
    程度会真实改变面积；若显式提供 ``percentile``，则改用百分位阈值。
    输入可以是线性 float、sRGB float 或 uint8。未检测到高光时返回零尺寸，
    而不是伪造一个可接受的高光。
    """
    lum = _luminance(image)
    h, w = lum.shape
    peak = float(np.max(lum, initial=0.0))
    if threshold is None:
        if percentile is not None:
            positive = lum[lum > 1e-12]
            if positive.size < max(8, int(0.01 * h * w)):
                threshold = peak + 1e-12
            else:
                threshold = float(np.percentile(lum, percentile))
        else:
            threshold = peak * relative_peak if peak > 1e-12 else peak + 1e-12
    mask = lum >= threshold
    count = int(mask.sum())
    if count < max(1, int(min_fraction * h * w)):
        return HighlightMeasure(0, h * w, 0.0, 0.0, 0.0, 0.0,
                                (float("nan"), float("nan")), float(threshold))
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area = count / float(h * w)
    equiv = 2.0 * np.sqrt(count / np.pi) / max(min(h, w), 1)
    return HighlightMeasure(
        pixel_count=count,
        total_pixels=int(h * w),
        area_fraction=float(area),
        width_fraction=float((x1 - x0 + 1) / w),
        height_fraction=float((y1 - y0 + 1) / h),
        equivalent_diameter_fraction=float(equiv),
        centroid_xy=(float(xs.mean()), float(ys.mean())),
        threshold=float(threshold),
    )


def highlight_size(image: np.ndarray, **kwargs: Any) -> float:
    """高光面积占比；需要完整证据时使用 :func:`measure_highlight`。"""
    return measure_highlight(image, **kwargs).area_fraction


# 语义别名：部分调用方把高光尺寸理解为等效直径。
def highlight_diameter(image: np.ndarray, **kwargs: Any) -> float:
    return measure_highlight(image, **kwargs).equivalent_diameter_fraction


highlight_footprint = measure_highlight


@dataclass
class MigrationDecision:
    accepted: bool
    reason: str
    train_rmse: Optional[float]
    holdout_rmse: Optional[float]
    rerender: Dict[str, Any] = field(default_factory=dict)
    bounds: Dict[str, Any] = field(default_factory=dict)
    highlight: Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def accept_migration(
    params_or_result: Mapping[str, float] | Any,
    *,
    train_observations: Sequence[Observation] | None = None,
    holdout_observations: Sequence[Observation] | None = None,
    train_rmse_limit: float = 0.05,
    holdout_rmse_limit: float = 0.03,
    measured_highlight_image: Optional[np.ndarray] = None,
    rendered_highlight_image: Optional[np.ndarray] = None,
    measured_highlight_size: Optional[float] = None,
    rendered_highlight_size: Optional[float] = None,
    actual_highlight_size: Optional[float] = None,
    predicted_highlight_size: Optional[float] = None,
    actual_highlight_diameter: Optional[float] = None,
    predicted_highlight_diameter: Optional[float] = None,
    highlight_size_tolerance: float = 0.35,
    optimizer_evidence: Optional[Mapping[str, Any]] = None,
    active: Optional[Sequence[str]] = None,
    **extra_highlight: Any,
) -> MigrationDecision:
    """根据训练解释、留出重渲染、边界和实际高光尺寸决定是否接受迁移材质。

    优化器证据只进入报告；即使 ``optimizer_success=True``，若留出重渲染或
    高光尺寸不达标也不会接受。高光证据可以是实测/重渲染图像，也可以是调用
    方已经测量好的面积或等效直径标量。缺少任一侧证据时返回
    ``rejected_missing_actual_highlight_evidence`` 并拒绝自动接受，避免伪报。
    """
    if hasattr(params_or_result, "parameters") and hasattr(params_or_result, "execution_evidence"):
        result = params_or_result
        params = dict(result.parameters)
        if optimizer_evidence is None:
            optimizer_evidence = getattr(result, "optimizer_evidence", None)
        if active is None:
            active = getattr(result, "active_params", None)
    else:
        params = params_or_result

    diagnostics: list[str] = []
    bound_info = boundary_report(params, active)
    rerender_train = rerender_rmse_evidence(params, train_observations or [])
    rerender_hold = rerender_rmse_evidence(params, holdout_observations or [])

    train_rmse = rerender_train.get("rmse")
    holdout_rmse = rerender_hold.get("rmse")
    if train_rmse is not None and train_rmse > train_rmse_limit:
        diagnostics.append(f"训练重渲染 RMSE={train_rmse:.5f} 超过 {train_rmse_limit}")
    if holdout_rmse is not None and holdout_rmse > holdout_rmse_limit:
        diagnostics.append(f"留出重渲染 RMSE={holdout_rmse:.5f} 超过 {holdout_rmse_limit}")
    if not bound_info["all_within_bounds"]:
        diagnostics.append("存在越界参数: " + ",".join(bound_info["invalid_parameters"]))

    measured_scalar = _first_not_none(
        measured_highlight_size,
        actual_highlight_size,
        actual_highlight_diameter,
        _scalar_from_extra(extra_highlight, (
            "actual_highlight_area", "measured_highlight_area",
            "holdout_highlight_size", "actual_highlight",
            "measured_highlight", "true_highlight_size",
            "actual_highlight_footprint",
        )),
    )
    rendered_scalar = _first_not_none(
        rendered_highlight_size,
        predicted_highlight_size,
        predicted_highlight_diameter,
        _scalar_from_extra(extra_highlight, (
            "rendered_highlight_area", "simulated_highlight_size",
            "candidate_highlight_size", "rendered_highlight",
            "predicted_highlight", "migrated_highlight_size",
            "simulated_highlight",
        )),
    )

    highlight_info: Dict[str, Any] = {"available": False}
    if measured_highlight_image is not None and rendered_highlight_image is not None:
        measured_h = measure_highlight(measured_highlight_image)
        rendered_h = measure_highlight(rendered_highlight_image)
        denom = max(measured_h.area_fraction, 1e-8)
        size_rel_error = abs(rendered_h.area_fraction - measured_h.area_fraction) / denom
        diameter_error = abs(
            rendered_h.equivalent_diameter_fraction
            - measured_h.equivalent_diameter_fraction
        ) / max(measured_h.equivalent_diameter_fraction, 1e-8)
        accepted_h = size_rel_error <= highlight_size_tolerance
        highlight_info = {
            "available": True,
            "source": "images",
            "measured": measured_h.to_dict(),
            "rendered": rendered_h.to_dict(),
            "area_relative_error": float(size_rel_error),
            "diameter_relative_error": float(diameter_error),
            "tolerance": float(highlight_size_tolerance),
            "size_match": bool(accepted_h),
        }
        if not accepted_h:
            diagnostics.append(
                f"留出光位高光面积相对误差={size_rel_error:.3f} 超过 "
                f"{highlight_size_tolerance:.3f}"
            )
    elif measured_scalar is not None and rendered_scalar is not None:
        metric = "diameter" if (
            actual_highlight_diameter is not None
            or predicted_highlight_diameter is not None
        ) else "area_or_size"
        rel_error = abs(float(rendered_scalar) - float(measured_scalar)) / max(
            abs(float(measured_scalar)), 1e-8
        )
        accepted_h = rel_error <= highlight_size_tolerance
        highlight_info = {
            "available": True,
            "source": "scalar_sizes",
            "metric": metric,
            "measured_size": float(measured_scalar),
            "rendered_size": float(rendered_scalar),
            "relative_error": float(rel_error),
            "tolerance": float(highlight_size_tolerance),
            "size_match": bool(accepted_h),
        }
        if not accepted_h:
            diagnostics.append(
                f"留出光位实际高光尺寸相对误差={rel_error:.3f} 超过 "
                f"{highlight_size_tolerance:.3f}"
            )
    else:
        highlight_info["reason"] = "highlight_evidence_missing"
        diagnostics.append("缺少留出光位的实测/重渲染高光尺寸证据，不能仅凭训练误差接受迁移")

    accepted = (
        bound_info["all_within_bounds"]
        and not diagnostics
        and highlight_info.get("size_match", False)
        and (train_rmse is None or train_rmse <= train_rmse_limit)
        and (holdout_rmse is None or holdout_rmse <= holdout_rmse_limit)
    )
    if accepted:
        reason = "accepted_by_holdout_rerender_and_highlight_size"
    elif highlight_info.get("reason") == "highlight_evidence_missing":
        reason = "rejected_missing_actual_highlight_evidence"
    else:
        reason = "rejected_by_independent_rerender_evidence"
    return MigrationDecision(
        accepted=accepted,
        reason=reason,
        train_rmse=None if train_rmse is None else float(train_rmse),
        holdout_rmse=None if holdout_rmse is None else float(holdout_rmse),
        rerender={"train": rerender_train, "holdout": rerender_hold},
        bounds=bound_info,
        highlight=highlight_info,
        optimizer=dict(optimizer_evidence or {}),
        diagnostics=diagnostics,
    )


def _first_not_none(*values: Any) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def _scalar_from_extra(extra: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = extra.get(key)
        if isinstance(value, Mapping):
            value = value.get(
                "area_fraction",
                value.get("equivalent_diameter_fraction", value.get("size")),
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None
