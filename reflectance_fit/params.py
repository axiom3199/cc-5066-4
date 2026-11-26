"""固定的参数空间、取值范围与颜色空间约定。

尺度歧义约定（重要）
====================
多角度反射拟合存在天然的尺度歧义：曝光、光强与材质能量三者同时缩放可以
得到完全相同的图像。本项目通过约定**固定**其中两个自由度来消除歧义：

* 光照辐照度固定为 ``LIGHT_IRRADIANCE = 1.0``（线性空间单位）；
* 相机曝光固定为 ``EXPOSURE = 1.0``，输出不做增益拟合；
* 只有材质参数（反照率、粗糙度、镜面强度、金属度）允许变化。

优化器**不允许**把曝光或光强当作自由参数。标定误差仅作为观测噪声注入，
不会通过调整曝光来吸收。

颜色空间
========
全部 BRDF 运算在线性 sRGB（linearized sRGB, EOTF 近似 gamma 2.4 分段）中
进行；仅在屏幕显示或保存 PNG 时才通过 ``linear_to_srgb`` 编码回 sRGB。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

# --- 固定尺度约定 ---------------------------------------------------------
LIGHT_IRRADIANCE: float = 1.0
EXPOSURE: float = 1.0
COLOR_SPACE: str = "linear-sRGB (CIE 1931 sRGB primaries, D65)"

# --- 参数范围（物理空间，全部为硬约束） -----------------------------------
@dataclass(frozen=True)
class ParamRange:
    low: float
    high: float

    def clip(self, x: float) -> float:
        return float(min(self.high, max(self.low, x)))


ALBEDO_RANGE = ParamRange(0.005, 0.99)
ROUGHNESS_RANGE = ParamRange(0.02, 1.0)      # GGX alpha = roughness^2 中的 roughness
SPECULAR_RANGE = ParamRange(0.0, 1.0)       # 介电质 F0 = 0.08 * specular
METALLIC_RANGE = ParamRange(0.0, 1.0)

#: 支持的全部可调参数及其范围。顺序即优化向量的顺序。
PARAM_SPEC: Dict[str, ParamRange] = {
    "albedo_r": ALBEDO_RANGE,
    "albedo_g": ALBEDO_RANGE,
    "albedo_b": ALBEDO_RANGE,
    "roughness": ROUGHNESS_RANGE,
    "specular": SPECULAR_RANGE,
    "metallic": METALLIC_RANGE,
}

#: 可由调用方关闭的参数（关闭后锁定为给定默认值，不进入优化向量）。
OPTIONAL_PARAMS = ("metallic",)


def default_parameters() -> Dict[str, float]:
    """返回一组保守的默认材质参数。"""
    return {
        "albedo_r": 0.5,
        "albedo_g": 0.5,
        "albedo_b": 0.5,
        "roughness": 0.5,
        "specular": 0.5,
        "metallic": 0.0,
    }


def clip_physical(params: Mapping[str, float]) -> Dict[str, float]:
    """把物理参数逐分量裁剪到固定范围内。"""
    out = dict(default_parameters())
    for key, value in params.items():
        if key in PARAM_SPEC:
            out[key] = PARAM_SPEC[key].clip(float(value))
    return out


# --- 优化空间 <-> 物理空间 ------------------------------------------------
# 优化在无界空间进行：对有界物理量使用 logit 变换，保证约束严格满足，
# 而不是依赖 SLSQP 的边界（边界处梯度为零会造成假收敛）。

_EPS = 1e-4


def _logit(p: float, low: float, high: float) -> float:
    p = min(max((p - low) / (high - low), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float, low: float, high: float) -> float:
    # 数值稳定：大正/大负参数下直接饱和，避免 math.exp 溢出。
    if z >= 0:
        e = math.exp(-min(z, 700.0))
        p = 1.0 / (1.0 + e)
    else:
        e = math.exp(max(z, -700.0))
        p = e / (1.0 + e)
    return low + (high - low) * p


def physical_to_fit_params(
    physical: Mapping[str, float], active: tuple[str, ...] | None = None
) -> np.ndarray:
    """物理参数 -> 无界优化向量。"""
    physical = clip_physical(physical)
    keys = active or tuple(PARAM_SPEC.keys())
    return np.array(
        [_logit(physical[k], PARAM_SPEC[k].low, PARAM_SPEC[k].high) for k in keys],
        dtype=np.float64,
    )


def fit_params_to_physical(
    z: np.ndarray, active: tuple[str, ...] | None = None
) -> Dict[str, float]:
    """无界优化向量 -> 物理参数字典（自动补全未激活参数的默认值）。"""
    keys = active or tuple(PARAM_SPEC.keys())
    out = default_parameters()
    for key, value in zip(keys, z):
        out[key] = _sigmoid(float(value), PARAM_SPEC[key].low, PARAM_SPEC[key].high)
    return out


# --- sRGB 编解码 -----------------------------------------------------------

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
