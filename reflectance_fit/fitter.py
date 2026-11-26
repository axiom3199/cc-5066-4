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
   输出满足"留出误差不超过噪声容差"的可行参数范围。尺度化 Jacobian 的
   SVD 另外标出平坦参数谷对应的参数组合。

4. **尺度固定**：优化向量只含材质参数（见 :mod:`params`），曝光与光强
   不参与优化，因此不存在曝光×光强×材质能量的尺度歧义。

5. **后端迁移证据**：优化器状态只用于诊断；最终 RMSE 会在求解结束后用
   :func:`reflectance_fit.brdf.shade_rgb` 独立重算，并记录边界检查。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .brdf import shade_rgb
from .diagnostics import analyze_flat_valley
from .evidence import boundary_report, rerender_rmse_evidence
from .geometry import Observation, grazing_fraction
from .optimization import (
    DEFAULT_BACKEND_ID,
    LEGACY_BACKEND_ID,
    OPTIMIZER_CONTRACT_VERSION,
    minimize_bounded,
    normalize_backend,
)
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

FITTER_IMPLEMENTATION_VERSION = "reflection-fit-new-path/v2"


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
    optimizer_success: bool = True
    stopping_reason: str = "not_recorded"
    backend: str = DEFAULT_BACKEND_ID
    optimizer_evidence: Dict[str, Any] = field(default_factory=dict)


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
    implementation: str = FITTER_IMPLEMENTATION_VERSION
    backend: str = DEFAULT_BACKEND_ID
    contract_version: str = OPTIMIZER_CONTRACT_VERSION
    optimizer_success: bool = True
    stopping_reason: str = ""
    optimizer_evidence: Dict[str, Any] = field(default_factory=dict)
    execution_evidence: Dict[str, Any] = field(default_factory=dict)
    boundary_checks: Dict[str, Any] = field(default_factory=dict)
    direction_diagnostics: Dict[str, Any] = field(default_factory=dict)
    validation_issues: List[Dict[str, str]] = field(default_factory=list)

    @property
    def new_path(self) -> bool:
        return self.implementation == FITTER_IMPLEMENTATION_VERSION

    @property
    def evidence(self) -> Dict[str, Any]:
        return self.execution_evidence

    @property
    def backend_version(self) -> str:
        return self.contract_version

    @property
    def boundary_valid(self) -> bool:
        return bool(self.boundary_checks.get("all_within_bounds", True))

    @property
    def rerender_error(self) -> float:
        """留出重渲染 RMSE；无留出时回退到训练重渲染 RMSE。"""
        rerender = self.execution_evidence.get("rerender", {})
        holdout = rerender.get("holdout", {}) if isinstance(rerender, dict) else {}
        if holdout.get("available"):
            return float(holdout["rmse"])
        return float(self.holdout_rmse)

    def summary(self) -> str:
        lines = [
            self.message,
            f"状态: {self.status} | 可辨识: {'是' if self.identifiable else '否'}",
            f"拟合路径: {self.implementation} | 后端: {self.backend}",
            f"停止条件: {self.stopping_reason or '未记录'} | "
            f"优化器成功标志: {self.optimizer_success}",
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
        rerender = self.execution_evidence.get("rerender", {})
        for split in ("train", "holdout"):
            info = rerender.get(split, {}) if isinstance(rerender, dict) else {}
            if info.get("available"):
                lines.append(
                    f"独立重渲染核验 {split}: RMSE={info['rmse']:.5f} "
                    f"({info['reason']})"
                )
        invalid = self.boundary_checks.get("invalid_parameters", [])
        if invalid:
            lines.append("边界异常: " + ", ".join(invalid))
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


def _validate_fixed(fixed: Mapping[str, float] | None) -> Dict[str, float]:
    fixed = dict(fixed or {})
    for key, value in fixed.items():
        if key not in PARAM_SPEC:
            raise ValueError(f"未知固定参数: {key}")
        pr = PARAM_SPEC[key]
        if not pr.low <= float(value) <= pr.high:
            raise ValueError(
                f"固定参数 {key}={value} 超出物理范围 [{pr.low}, {pr.high}]"
            )
    return fixed


def fit_brdf(
    train_obs: Sequence[Observation],
    holdout_obs: Sequence[Observation],
    *,
    fixed: Mapping[str, float] | None = None,
    grazing_threshold: float = GRAZING_THRESHOLD,
    maxiter: int = 200,
    backend: str = DEFAULT_BACKEND_ID,
    explicit_gradient: bool = False,
) -> FitResult:
    """拟合并按留出误差选解。

    ``fixed`` 中的参数被锁定（如已知为介电质时 ``{"metallic": 0.0}``），
    其余 :data:`PARAM_SPEC` 参数进入优化向量。``backend`` 设为
    :data:`LEGACY_BACKEND_ID` 时保留迁移前的直接 SciPy 调用路径，供新旧
    契约对照。
    """
    backend = normalize_backend(backend)
    if backend == LEGACY_BACKEND_ID:
        return fit_brdf_legacy(
            train_obs, holdout_obs, fixed=fixed,
            grazing_threshold=grazing_threshold, maxiter=maxiter,
        )
    if not train_obs:
        raise ValueError("训练观测不能为空")
    fixed = _validate_fixed(fixed)
    active = tuple(k for k in PARAM_SPEC if k not in fixed)

    measured, weights, _, _, _ = _stack(train_obs)
    if holdout_obs:
        hold_measured, hold_weights, _, _, _ = _stack(holdout_obs)
    else:
        hold_measured = np.zeros((0, 3))
        hold_weights = np.zeros(0)

    trace: List[float] = []

    def physical_objective(x: np.ndarray) -> float:
        params = {key: float(value) for key, value in zip(active, x)}
        params.update(fixed)
        predicted = shade_rgb(params, train_obs)
        rmse = weighted_rmse(predicted, measured, weights)
        trace.append(rmse)
        return rmse

    candidates: List[Candidate] = []
    for start in _initial_points(active):
        merged = {**default_parameters(), **fixed, **start}
        x0 = np.array([merged[key] for key in active], dtype=np.float64)
        solved = minimize_bounded(
            physical_objective,
            x0,
            active,
            maxiter=maxiter,
            backend=backend,
            finite_gradient=explicit_gradient,
        )
        params = {**default_parameters(), **solved.x, **fixed}
        params = clip_physical(params)
        train_rmse = weighted_rmse(shade_rgb(params, train_obs), measured, weights)
        if holdout_obs:
            hold_rmse = weighted_rmse(
                shade_rgb(params, holdout_obs), hold_measured, hold_weights
            )
        else:
            hold_rmse = train_rmse
        evidence = solved.to_evidence()
        candidates.append(Candidate(
            params=params,
            train_rmse=train_rmse,
            holdout_rmse=hold_rmse,
            optimizer_success=solved.optimizer_success,
            stopping_reason=solved.stopping_reason,
            backend=solved.backend,
            optimizer_evidence=evidence,
        ))

    chosen = _choose_candidate(candidates)
    all_obs = list(train_obs) + list(holdout_obs)
    gfrac = grazing_fraction(all_obs)
    identifiable = gfrac >= grazing_threshold

    boundary = boundary_report(chosen.params, active)
    rerender_evidence = {
        "train": rerender_rmse_evidence(chosen.params, train_obs, weights=weights),
        "holdout": rerender_rmse_evidence(
            chosen.params, holdout_obs,
            weights=(hold_weights if holdout_obs else None),
        ),
    }
    try:
        directions = analyze_flat_valley(
            params=chosen.params, observations=all_obs or list(train_obs),
            active=active,
        ).to_dict()
    except Exception as exc:  # 诊断失败不能伪造成可辨，也不应让拟合结果丢失。
        directions = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    fit = FitResult(
        parameters=dict(chosen.params),
        train_rmse=chosen.train_rmse,
        holdout_rmse=chosen.holdout_rmse,
        identifiable=identifiable,
        status="identifiable" if identifiable else "underconstrained",
        message="",
        candidates=sorted(candidates, key=lambda c: c.holdout_rmse),
        active_params=active,
        objective_trace=trace,
        n_train=len(train_obs),
        n_holdout=len(holdout_obs),
        grazing_fraction=gfrac,
        backend=chosen.backend,
        optimizer_success=chosen.optimizer_success,
        stopping_reason=chosen.stopping_reason,
        optimizer_evidence=chosen.optimizer_evidence,
        boundary_checks=boundary,
        direction_diagnostics=directions,
    )
    fit.execution_evidence = {
        "contract_version": OPTIMIZER_CONTRACT_VERSION,
        "implementation": FITTER_IMPLEMENTATION_VERSION,
        "optimizer": chosen.optimizer_evidence,
        "rerender": rerender_evidence,
        "train_rerender_rmse": rerender_evidence["train"].get("rmse"),
        "holdout_rerender_rmse": rerender_evidence["holdout"].get("rmse"),
        "boundary": boundary,
        "boundary_valid": boundary["all_within_bounds"],
        "direction_diagnostics": directions,
        "acceptance_basis": "independent_rerender_and_boundary_checks_not_optimizer_flag",
    }

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


def _choose_candidate(candidates: Sequence[Candidate]) -> Candidate:
    # --- 留出选解：训练误差处于最优带内的候选中，选留出误差最低者 -------
    best_train = min(c.train_rmse for c in candidates)
    band = max(best_train * HOLDOUT_TOLERANCE, NOISE_FLOOR_RMSE)
    in_band = [c for c in candidates if c.train_rmse <= band] or \
              [min(candidates, key=lambda c: c.train_rmse)]
    return min(in_band, key=lambda c: c.holdout_rmse)


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
        sub_measured, sub_weights, _, _, _ = _stack(train_obs)
        for value in grid:
            locked = {**fixed, key: float(value)}
            x0 = np.array([{**chosen, **locked}[k] for k in sub_active],
                          dtype=np.float64)

            def obj(x: np.ndarray, _locked=locked) -> float:
                params = {k: float(v) for k, v in zip(sub_active, x)}
                params.update(_locked)
                return weighted_rmse(shade_rgb(params, train_obs),
                                     sub_measured, sub_weights)

            solved = minimize_bounded(
                obj, x0, sub_active, maxiter=120, ftol=1e-9,
                gtol=1e-5,
            )
            params = {**default_parameters(), **solved.x, **locked}
            params = clip_physical(params)
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


# --------------------------------------------------------------------------
# 旧后端：迁移前的直接 logit + SciPy L-BFGS-B 契约，保留用于对照和回滚。
# --------------------------------------------------------------------------

def fit_brdf_legacy(
    train_obs: Sequence[Observation],
    holdout_obs: Sequence[Observation],
    *,
    fixed: Mapping[str, float] | None = None,
    grazing_threshold: float = GRAZING_THRESHOLD,
    maxiter: int = 200,
) -> FitResult:
    if not train_obs:
        raise ValueError("训练观测不能为空")
    fixed = _validate_fixed(fixed)
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
        hold_rmse = (
            weighted_rmse(shade_rgb(params, holdout_obs), hold_measured, hold_weights)
            if holdout_obs else train_rmse
        )
        candidates.append(Candidate(
            params, train_rmse, hold_rmse,
            optimizer_success=bool(result.success),
            stopping_reason=str(result.message),
            backend=LEGACY_BACKEND_ID,
            optimizer_evidence={"optimizer_success": bool(result.success)},
        ))

    chosen = _choose_candidate(candidates)
    all_obs = list(train_obs) + list(holdout_obs)
    gfrac = grazing_fraction(all_obs)
    identifiable = gfrac >= grazing_threshold
    fit = FitResult(
        parameters=dict(chosen.params),
        train_rmse=chosen.train_rmse,
        holdout_rmse=chosen.holdout_rmse,
        identifiable=identifiable,
        status="identifiable" if identifiable else "underconstrained",
        message="",
        candidates=sorted(candidates, key=lambda c: c.holdout_rmse),
        active_params=active,
        objective_trace=trace,
        n_train=len(train_obs),
        n_holdout=len(holdout_obs),
        grazing_fraction=gfrac,
        implementation=LEGACY_BACKEND_ID,
        backend=LEGACY_BACKEND_ID,
        contract_version=LEGACY_BACKEND_ID,
        optimizer_success=chosen.optimizer_success,
        stopping_reason=chosen.stopping_reason,
        optimizer_evidence=chosen.optimizer_evidence,
    )
    fit.boundary_checks = boundary_report(chosen.params, active)
    rerender_evidence = {
        "train": rerender_rmse_evidence(chosen.params, train_obs, weights=weights),
        "holdout": rerender_rmse_evidence(
            chosen.params, holdout_obs,
            weights=(hold_weights if holdout_obs else None)),
    }
    try:
        directions = analyze_flat_valley(
            params=chosen.params, observations=all_obs or list(train_obs),
            active=active,
        ).to_dict()
    except Exception as exc:
        directions = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    fit.direction_diagnostics = directions
    fit.execution_evidence = {
        "contract_version": LEGACY_BACKEND_ID,
        "implementation": LEGACY_BACKEND_ID,
        "optimizer": chosen.optimizer_evidence,
        "rerender": rerender_evidence,
        "train_rerender_rmse": rerender_evidence["train"].get("rmse"),
        "holdout_rerender_rmse": rerender_evidence["holdout"].get("rmse"),
        "boundary": fit.boundary_checks,
        "boundary_valid": fit.boundary_checks["all_within_bounds"],
        "direction_diagnostics": directions,
        "acceptance_basis": "independent_rerender_and_boundary_checks_not_optimizer_flag",
    }
    if identifiable:
        fit.feasible_ranges = {k: (chosen.params[k], chosen.params[k]) for k in active}
        fit.message = (
            "侧视观测充分，按留出角度误差选定唯一候选解；"
            "粗糙度与镜面强度的补偿歧义已由留出视角打破。"
        )
    else:
        fit.feasible_ranges = _profile_scan_legacy(
            chosen.params, active, fixed, train_obs, holdout_obs,
            measured, weights, hold_measured, hold_weights,
        )
        fit.message = (
            "侧视/掠射观测不足，粗糙度与镜面强度在当前数据上可互相补偿。"
            "已输出各参数的可行范围而非宣称唯一恢复；补充大角度观测可缩小范围。"
        )
    return fit


def _profile_scan_legacy(
    chosen, active, fixed, train_obs, holdout_obs,
    measured, weights, hold_measured, hold_weights, n_grid=13,
):
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
        ranges[key] = (min(feasible_values), max(feasible_values)) \
            if feasible_values else (chosen[key], chosen[key])
    return ranges


def fit_reflection(*args, **kwargs) -> FitResult:
    """纹理制作界面/新代码使用的反射拟合新实现入口。"""
    return fit_brdf(*args, **kwargs)
