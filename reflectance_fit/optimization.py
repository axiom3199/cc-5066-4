"""反射拟合的数值优化后端适配层。

反射拟合对外仍使用物理参数；不同求解器对边界和梯度尺度的契约不同。
本模块把这些转换集中起来，避免迁移后端时把物理梯度误当成无界变量梯度。

默认后端仍使用 SciPy L-BFGS-B，但优化变量是由固定范围映射得到的无界
logit 变量，因此不会在硬边界上产生零梯度假收敛。需要显式约束的后端可直接
使用 :func:`physical_bounds`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .params import PARAM_SPEC, ParamRange

OPTIMIZER_CONTRACT_VERSION = "reflection-fit-optimizer-migration/v2"
DEFAULT_BACKEND_ID = "scipy-lbfgsb-logit-v2"
LEGACY_BACKEND_ID = "scipy-lbfgsb-logit-v1"
PHYSICAL_BOUNDS_BACKEND_ID = "scipy-lbfgsb-physical-bounds-v2"

BACKEND_ALIASES = {
    "new": DEFAULT_BACKEND_ID,
    "v2": DEFAULT_BACKEND_ID,
    "default": DEFAULT_BACKEND_ID,
    "logit": DEFAULT_BACKEND_ID,
    DEFAULT_BACKEND_ID: DEFAULT_BACKEND_ID,
    "legacy": LEGACY_BACKEND_ID,
    "old": LEGACY_BACKEND_ID,
    "v1": LEGACY_BACKEND_ID,
    LEGACY_BACKEND_ID: LEGACY_BACKEND_ID,
    "physical": PHYSICAL_BOUNDS_BACKEND_ID,
    "bounds": PHYSICAL_BOUNDS_BACKEND_ID,
    "explicit-bounds": PHYSICAL_BOUNDS_BACKEND_ID,
    PHYSICAL_BOUNDS_BACKEND_ID: PHYSICAL_BOUNDS_BACKEND_ID,
}


def normalize_backend(backend: str) -> str:
    try:
        return BACKEND_ALIASES[backend]
    except KeyError as exc:
        raise ValueError(f"未知优化后端: {backend!r}") from exc

Objective = Callable[[np.ndarray], float]
PhysicalGradient = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Bounds:
    low: np.ndarray
    high: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.high - self.low

    def as_scipy(self) -> list[tuple[float, float]]:
        return [(float(lo), float(hi)) for lo, hi in zip(self.low, self.high)]

    def contains(self, x: np.ndarray, tolerance: float = 1e-8) -> bool:
        x = np.asarray(x, dtype=np.float64)
        return bool(np.all(x >= self.low - tolerance)
                    and np.all(x <= self.high + tolerance))


def spec_for_keys(active: Sequence[str]) -> Tuple[ParamRange, ...]:
    return tuple(PARAM_SPEC[key] for key in active)


def physical_bounds(active: Sequence[str]) -> Bounds:
    spec = spec_for_keys(active)
    return Bounds(
        np.array([pr.low for pr in spec], dtype=np.float64),
        np.array([pr.high for pr in spec], dtype=np.float64),
    )


def _sigmoid_array(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    e_pos = np.exp(-np.minimum(z[pos], 700.0))
    out[pos] = 1.0 / (1.0 + e_pos)
    e_neg = np.exp(np.maximum(z[~pos], -700.0))
    out[~pos] = e_neg / (1.0 + e_neg)
    return out


def physical_to_internal(
    x: np.ndarray, active: Sequence[str], *, eps: float = 1e-4
) -> np.ndarray:
    """物理向量 -> 无界 logit 向量，并在端点保留与旧实现一致的截断。"""
    bounds = physical_bounds(active)
    u = np.clip((np.asarray(x, dtype=np.float64) - bounds.low) / bounds.width,
                eps, 1.0 - eps)
    return np.log(u / (1.0 - u))


def internal_to_physical(z: np.ndarray, active: Sequence[str]) -> np.ndarray:
    bounds = physical_bounds(active)
    u = _sigmoid_array(np.asarray(z, dtype=np.float64))
    return bounds.low + bounds.width * u


def physical_to_unit(x: np.ndarray, active: Sequence[str]) -> np.ndarray:
    bounds = physical_bounds(active)
    return (np.asarray(x, dtype=np.float64) - bounds.low) / bounds.width


def unit_to_physical(u: np.ndarray, active: Sequence[str]) -> np.ndarray:
    bounds = physical_bounds(active)
    return bounds.low + bounds.width * np.asarray(u, dtype=np.float64)


def logit_derivative_factor(
    x: np.ndarray, active: Sequence[str]
) -> np.ndarray:
    """Return dp/dz for p=low+(high-low)*sigmoid(z)."""
    bounds = physical_bounds(active)
    x = np.asarray(x, dtype=np.float64)
    u = np.clip((x - bounds.low) / bounds.width, 1e-12, 1.0 - 1e-12)
    return bounds.width * u * (1.0 - u)


def convert_physical_gradient_to_logit(
    grad_physical: np.ndarray,
    x_physical: np.ndarray,
    active: Sequence[str],
) -> np.ndarray:
    """dL/dz = dL/dp * dp/dz。

    新后端若在无界 logit 空间请求梯度，必须使用这个换算；直接把物理梯度
    传给 L-BFGS-B 会使窄范围/饱和方向的步长尺度错误。
    """
    return np.asarray(grad_physical, dtype=np.float64) * \
        logit_derivative_factor(x_physical, active)


def convert_physical_gradient_to_unit(
    grad_physical: np.ndarray, active: Sequence[str]
) -> np.ndarray:
    """p=low+width*u，因此 dL/du=width*dL/dp。"""
    return physical_bounds(active).width * np.asarray(grad_physical, dtype=np.float64)


def finite_difference_gradient(
    fun: Objective,
    x: np.ndarray,
    *,
    rel_step: float = 1e-6,
    min_step: float = 1e-5,
) -> np.ndarray:
    """前向差分物理空间梯度；步长按参数尺度缩放。"""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    f0 = float(fun(x))
    for i in range(x.size):
        h = rel_step * max(min_step, abs(float(x[i])))
        xp = x.copy()
        xp[i] += h
        grad[i] = (float(fun(xp)) - f0) / h
    return grad


@dataclass
class SolveResult:
    x: Dict[str, float]
    x_vector: np.ndarray
    internal_x: np.ndarray
    success: bool
    optimizer_success: bool
    status: str
    stopping_reason: str
    message: str
    fun: float
    backend: str = DEFAULT_BACKEND_ID
    contract_version: str = OPTIMIZER_CONTRACT_VERSION
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    maxiter: int = 0
    bounds: Optional[Bounds] = None
    raw: object = field(default=None, repr=False)

    def to_evidence(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "contract_version": self.contract_version,
            "optimizer_success": bool(self.optimizer_success),
            "success": bool(self.success),
            "status": self.status,
            "stopping_reason": self.stopping_reason,
            "message": self.message,
            "objective": float(self.fun),
            "iterations": int(self.nit),
            "function_evaluations": int(self.nfev),
            "gradient_evaluations": int(self.njev),
            "maxiter": int(self.maxiter),
        }


def _stable_stop_reason(result: object, maxiter: int) -> Tuple[str, str]:
    info = getattr(result, "message", "")
    message = info.decode() if isinstance(info, bytes) else str(info)
    status = int(getattr(result, "status", -1))
    nit = int(getattr(result, "nit", 0))
    text = message.lower()
    if "gtol" in text or "projected gradient" in text:
        reason = "converged_gradient_tolerance"
    elif "ftol" in text or "relative reduction" in text:
        reason = "converged_objective_tolerance"
    elif nit >= maxiter or "maxiter" in text or "iterations reached" in text:
        reason = "iteration_limit"
    elif status == 0:
        reason = "converged"
    elif "precision loss" in text or "abnormal" in text:
        reason = "precision_loss_or_constraint_conflict"
    else:
        reason = f"optimizer_status_{status}"
    return reason, message


def minimize_bounded(
    fun_physical: Objective,
    x0: np.ndarray,
    active: Sequence[str],
    *,
    grad_physical: Optional[PhysicalGradient] = None,
    maxiter: int = 200,
    ftol: float = 1e-10,
    gtol: float = 1e-8,
    backend: str = DEFAULT_BACKEND_ID,
    finite_gradient: bool = False,
) -> SolveResult:
    """在固定参数范围内最小化目标。

    ``fun_physical`` 接收物理向量。默认把它包装到无界 logit 空间；若提供
    物理梯度或开启有限差分梯度，会显式换算梯度尺度。
    """
    from scipy.optimize import minimize

    backend = normalize_backend(backend)
    active = tuple(active)
    bounds = physical_bounds(active)
    x0 = np.clip(np.asarray(x0, dtype=np.float64), bounds.low, bounds.high)
    z0 = physical_to_internal(x0, active)

    def z_to_x(z: np.ndarray) -> np.ndarray:
        return internal_to_physical(z, active)

    use_jac = grad_physical is not None or finite_gradient

    def objective(z: np.ndarray):
        x = z_to_x(z)
        if not use_jac:
            return float(fun_physical(x))
        value = float(fun_physical(x))
        if grad_physical is not None:
            gp = np.asarray(grad_physical(x), dtype=np.float64)
        else:
            gp = finite_difference_gradient(fun_physical, x)
        gz = convert_physical_gradient_to_logit(gp, x, active)
        return value, gz

    options = {"maxiter": int(maxiter), "ftol": ftol, "gtol": gtol}
    if backend == PHYSICAL_BOUNDS_BACKEND_ID:
        # 显式约束后端路径：SciPy 在物理空间处理 bounds，因此 jac 也必须是
        # 物理空间梯度，不能返回 logit 梯度。
        if use_jac:
            def physical_objective(x: np.ndarray):
                value = float(fun_physical(x))
                if grad_physical is not None:
                    grad = np.asarray(grad_physical(x), dtype=np.float64)
                else:
                    grad = finite_difference_gradient(fun_physical, x)
                return value, grad

            result = minimize(
                physical_objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds.as_scipy(),
                jac=True,
                options=options,
            )
        else:
            result = minimize(
                fun_physical,
                x0,
                method="L-BFGS-B",
                bounds=bounds.as_scipy(),
                options=options,
            )
        x = np.clip(np.asarray(result.x, dtype=np.float64),
                    bounds.low, bounds.high)
        z = physical_to_internal(x, active)
    else:
        result = minimize(
            objective,
            z0,
            method="L-BFGS-B",
            jac=use_jac,
            options=options,
        )
        z = np.asarray(result.x, dtype=np.float64)
        x = internal_to_physical(z, active)

    params = {key: float(value) for key, value in zip(active, x)}
    stop_reason, message = _stable_stop_reason(result, maxiter)
    optimizer_success = bool(getattr(result, "success", False))
    # 达到迭代上限但得到有限结果时，后端标志可能为 False；这仍不是伪成功，
    # 调用方必须继续做独立重渲染评价。
    finite = bool(np.all(np.isfinite(x)) and np.isfinite(result.fun))
    return SolveResult(
        x=params,
        x_vector=x,
        internal_x=z,
        success=finite,
        optimizer_success=optimizer_success,
        status="ok" if finite else "non_finite",
        stopping_reason=stop_reason,
        message=message,
        fun=float(result.fun if np.isfinite(result.fun) else np.inf),
        backend=backend,
        nit=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)),
        maxiter=int(maxiter),
        bounds=bounds,
        raw=result,
    )
