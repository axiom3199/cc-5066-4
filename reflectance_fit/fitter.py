"""反射拟合优化器。

设计要点（对应受控拟合的方法论约束）
====================================

1. **目标函数**：训练视角上的加权 RMSE，比较的是完整 BRDF 重渲染辐射度
   与观测辐射度（线性 sRGB），而非平均颜色。

2. **优化/验证视角分离**：多起点优化只看训练集；每个候选解都计算留出
   视角误差。粗糙度 alpha 与镜面强度 specular 在近轴方向互相补偿，训练
   误差几乎相同的候选解可能在掠射角差异巨大——用留出误差选解，而不是
   选训练误差最低者。

3. **可辨识性判定**：当侧视/掠射观测不足时（``grazing_fraction`` 低于
   阈值），不宣称唯一恢复，而是对每个参数做轮廓扫描（profile scan），
   输出满足"留出误差不超过噪声容差"的可行参数范围。

4. **尺度固定**：优化向量只含材质参数（见 :mod:`params`），曝光与光强
   不参与优化，因此不存在曝光×光强×材质能量的尺度歧义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .brdf import shade_rgb
from .geometry import Observation, grazing_fraction
from .params import (
    PARAM_SPEC,
    clip_physical,
    default_parameters,
    fit_params_to_physical,
    physical_to_fit_params,
)

#: 留出误差相对最优解可接受的放宽倍数（噪声容差的工程取值）。
HOLDOUT_TOLERANCE = 1.5
#: 侧视观测占比低于该值时判定为可辨识性不足。
GRAZING_THRESHOLD = 0.25
#: 可行域判定阈值：holdout RMSE 不超过 best * 该倍数（并含噪声地板）。
FEASIBLE_FACTOR = 2.0
NOISE_FLOOR_RMSE = 0.01


# --------------------------------------------------------------------------
# 目标函数
# --------------------------------------------------------------------------

def _stack(obs: Sequence[Observation]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    measured = np.stack([o.radiance for o in obs])
    weights = np.array([o.weight for o in obs], dtype=np.float64)
    v = np.stack([o.view for o in obs])
    l = np.stack([o.light for o in obs])
    n = np.stack([o.normal for o in obs])
    return measured, weights, v, l, n


def weighted_rmse(predicted: np.ndarray, measured: np.ndarray,
                  weights: np.ndarray) -> float:
    residual = predicted - measured
    per_obs = np.sqrt(np.mean(residual ** 2, axis=1))
    w = weights / max(np.sum(weights), 1e-12)
    return float(np.sqrt(np.sum(w * per_obs ** 2)))


def residuals(params: Mapping[str, float],
              observations: Sequence[Observation]) -> np.ndarray:
    """逐观测、逐通道残差（重渲染 - 测量），形状 (N,3)，供界面展示。"""
    predicted = shade_rgb(params, observations)
    measured = np.stack([o.radiance for o in observations])
    return predicted - measured


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    params: Dict[str, float]
    train_rmse: float
    holdout_rmse: float


@dataclass
class FitResult:
    parameters: Dict[str, float]
    train_rmse: float
    holdout_rmse: float
    identifiable: bool
    status: str
    message: str
    candidates: List[Candidate] = field(default_factory=list)
    feasible_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    active_params: Tuple[str, ...] = tuple(PARAM_SPEC.keys())
    objective_trace: List[float] = field(default_factory=list)
    n_train: int = 0
    n_holdout: int = 0
    grazing_fraction: float = 0.0

    def summary(self) -> str:
        lines = [
            self.message,
            f"状态: {self.status} | 可辨识: {'是' if self.identifiable else '否'}",
            f"训练视角: {self.n_train}  留出视角: {self.n_holdout}  "
            f"侧视占比: {self.grazing_fraction:.2f}",
            f"训练 RMSE: {self.train_rmse:.5f}  留出 RMSE: {self.holdout_rmse:.5f}",
            "恢复参数:",
        ]
        for key in self.active_params:
            rng = self.feasible_ranges.get(key)
            value = self.parameters[key]
            if rng is not None and rng[1] - rng[0] > 1e-3:
                lines.append(f"  {key:10s} = {value:8.4f}  可行范围 "
                             f"[{rng[0]:.4f}, {rng[1]:.4f}]")
            else:
                lines.append(f"  {key:10s} = {value:8.4f}")
        if not self.identifiable:
            lines.append("注意: 多个参数组合可在容差内解释同样观测，"
                         "上述范围而非单点才是诚实的恢复结果。")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 多起点优化
# --------------------------------------------------------------------------

def _initial_points(active: Tuple[str, ...]) -> List[Dict[str, float]]:
    base = default_parameters()
    starts = []
    rough_grid = (0.15, 0.45, 0.8)
    spec_grid = (0.2, 0.5, 0.8)
    for r in rough_grid:
        for s in spec_grid:
            p = dict(base)
            p["roughness"], p["specular"] = r, s
            p["albedo_r"] = p["albedo_g"] = p["albedo_b"] = 0.45
            starts.append(p)
    # 金属材质起点
    for m in (0.0, 0.9):
        p = dict(base)
        p["metallic"], p["roughness"] = m, 0.35
        starts.append(p)
    return [{k: v for k, v in p.items() if k in active} for p in starts]


def fit_brdf(
    train_obs: Sequence[Observation],
    holdout_obs: Sequence[Observation],
    *,
    fixed: Mapping[str, float] | None = None,
    grazing_threshold: float = GRAZING_THRESHOLD,
    maxiter: int = 200,
) -> FitResult:
    """拟合并按留出误差选解。

    ``fixed`` 中的参数被锁定（如已知为介电质时 ``{"metallic": 0.0}``），
    其余 :data:`PARAM_SPEC` 参数进入优化向量。
    """
    if not train_obs:
        raise ValueError("训练观测不能为空")
    fixed = dict(fixed or {})
    for key in fixed:
        if key not in PARAM_SPEC:
            raise ValueError(f"未知固定参数: {key}")
    active = tuple(k for k in PARAM_SPEC if k not in fixed)

    measured, weights, _, _, _ = _stack(train_obs)
    if holdout_obs:
        hold_measured, hold_weights, _, _, _ = _stack(holdout_obs)
    else:
        hold_measured = np.zeros((0, 3))
        hold_weights = np.zeros(0)

    trace: List[float] = []

    def objective(z: np.ndarray) -> float:
        params = fit_params_to_physical(z, active)
        params.update(fixed)
        predicted = shade_rgb(params, train_obs)
        rmse = weighted_rmse(predicted, measured, weights)
        trace.append(rmse)
        return rmse

    candidates: List[Candidate] = []
    for start in _initial_points(active):
        merged = {**default_parameters(), **fixed, **start}
        z0 = physical_to_fit_params(merged, active)
        result = minimize(
            objective, z0, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8},
        )
        params = fit_params_to_physical(result.x, active)
        params.update(fixed)
        params = clip_physical(params)
        train_rmse = weighted_rmse(shade_rgb(params, train_obs), measured, weights)
        if holdout_obs:
            hold_rmse = weighted_rmse(
                shade_rgb(params, holdout_obs), hold_measured, hold_weights
            )
        else:
            hold_rmse = train_rmse
        candidates.append(Candidate(params, train_rmse, hold_rmse))

    # --- 留出选解：训练误差处于最优带内的候选中，选留出误差最低者 -------
    best_train = min(c.train_rmse for c in candidates)
    band = max(best_train * HOLDOUT_TOLERANCE, NOISE_FLOOR_RMSE)
    in_band = [c for c in candidates if c.train_rmse <= band] or \
              [min(candidates, key=lambda c: c.train_rmse)]
    chosen = min(in_band, key=lambda c: c.holdout_rmse)

    all_obs = list(train_obs) + list(holdout_obs)
    gfrac = grazing_fraction(all_obs)
    identifiable = gfrac >= grazing_threshold

    fit = FitResult(
        parameters=dict(chosen.params),
        train_rmse=chosen.train_rmse,
        holdout_rmse=chosen.holdout_rmse,
        identifiable=identifiable,
        status="identifiable" if identifiable else "underconstrained",
        message=("", ""),  # 占位，下面统一填写
        candidates=sorted(candidates, key=lambda c: c.holdout_rmse),
        active_params=active,
        objective_trace=trace,
        n_train=len(train_obs),
        n_holdout=len(holdout_obs),
        grazing_fraction=gfrac,
    )

    if identifiable:
        fit.feasible_ranges = {k: (chosen.params[k], chosen.params[k]) for k in active}
        fit.message = (
            "侧视观测充分，按留出角度误差选定唯一候选解；"
            "粗糙度与镜面强度的补偿歧义已由留出视角打破。"
        )
    else:
        fit.feasible_ranges = _profile_scan(
            chosen.params, active, fixed, train_obs, holdout_obs,
            measured, weights, hold_measured, hold_weights,
        )
        fit.message = (
            "侧视/掠射观测不足，粗糙度与镜面强度在当前数据上可互相补偿。"
            "已输出各参数的可行范围而非宣称唯一恢复；补充大角度观测可缩小范围。"
        )
    return fit


# --------------------------------------------------------------------------
# 轮廓扫描：固定某参数扫描，其余参数重优化，判断留出容差内的可行区间
# --------------------------------------------------------------------------

def _profile_scan(
    chosen: Dict[str, float],
    active: Tuple[str, ...],
    fixed: Mapping[str, float],
    train_obs: Sequence[Observation],
    holdout_obs: Sequence[Observation],
    measured: np.ndarray,
    weights: np.ndarray,
    hold_measured: np.ndarray,
    hold_weights: np.ndarray,
    n_grid: int = 13,
) -> Dict[str, Tuple[float, float]]:
    holdout_ref = _safe_holdout(
        chosen, holdout_obs, hold_measured, hold_weights, train_obs, measured, weights
    )
    tolerance = max(holdout_ref * FEASIBLE_FACTOR, NOISE_FLOOR_RMSE)

    ranges: Dict[str, Tuple[float, float]] = {}
    for key in active:
        pr = PARAM_SPEC[key]
        grid = np.linspace(pr.low + 1e-3, pr.high - 1e-3, n_grid)
        feasible_values: List[float] = []
        sub_active = tuple(k for k in active if k != key)
        for value in grid:
            locked = {**fixed, key: float(value)}
            sub_measured, sub_weights, _, _, _ = _stack(train_obs)
            z0 = physical_to_fit_params({**chosen, **locked}, sub_active)

            def obj(z: np.ndarray, _locked=locked) -> float:
                params = fit_params_to_physical(z, sub_active)
                params.update(_locked)
                return weighted_rmse(shade_rgb(params, train_obs),
                                     sub_measured, sub_weights)

            res = minimize(obj, z0, method="L-BFGS-B",
                           options={"maxiter": 120, "ftol": 1e-9})
            params = fit_params_to_physical(res.x, sub_active)
            params.update(locked)
            if holdout_obs:
                rmse = weighted_rmse(shade_rgb(params, holdout_obs),
                                     hold_measured, hold_weights)
            else:
                rmse = weighted_rmse(shade_rgb(params, train_obs),
                                     measured, weights)
            if rmse <= tolerance:
                feasible_values.append(float(value))
        if feasible_values:
            ranges[key] = (min(feasible_values), max(feasible_values))
        else:
            ranges[key] = (chosen[key], chosen[key])
    return ranges


def _safe_holdout(params, holdout_obs, hold_measured, hold_weights,
                  train_obs, measured, weights) -> float:
    if holdout_obs:
        return weighted_rmse(shade_rgb(params, holdout_obs),
                             hold_measured, hold_weights)
    return weighted_rmse(shade_rgb(params, train_obs), measured, weights)
