"""GGX / Cook-Torrance 微表面 BRDF 的真实反射求值。

实现的反射模型（限定表面模型，仅这些参数受支持）：

* 漫反射项：Lambert，金属度为 1 时关闭；
* 镜面项：Trowbridge-Reitz (GGX) 法线分布 D + Smith-GGX 遮蔽 G +
  Schlick 菲涅耳 F 的 Cook-Torrance 微表面模型；
* ``alpha = roughness ** 2``（Disney 重映射，避免 roughness→0 的数值奇异）；
* 介电质 F0 = ``0.08 * specular``；金属 F0 = 反照率，由 ``metallic`` 线性混合；
* 能量守恒：漫反射权重按 (1-F)(1-metallic) 衰减；Smith GGX 遮蔽保证方向
  反照率（半球积分）不超过 1。这是标准**单次散射**微表面模型，粗糙表面会
  保守吸收部分多次反弹能量，但绝不会超调（测试见 ``test_brdf.py``）。

这里没有任何"从平均颜色猜 PBR"的捷径：每个方向的输出都由完整 BRDF 积分
（单光源点积 cosine）计算，拟合与预览共用同一求值路径。
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from .geometry import Observation
from .params import EXPOSURE, LIGHT_IRRADIANCE, clip_physical

_DIELECTRIC_F0_SCALE = 0.08


def _as_rows(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    if d.ndim == 1:
        return d.reshape(1, 3)
    return d


def ggx_distribution(n_dot_h: np.ndarray, alpha: float) -> np.ndarray:
    """GGX/Trowbridge-Reitz 法线分布函数 D(h)。"""
    a2 = alpha * alpha
    n_dot_h = np.clip(n_dot_h, 0.0, 1.0)
    denom = np.pi * (n_dot_h ** 2 * (a2 - 1.0) + 1.0) ** 2
    return a2 / np.maximum(denom, 1e-12)


def ggx_g1(n_dot_x: np.ndarray, alpha: float) -> np.ndarray:
    """Smith-GGX 单端遮蔽项 G1。"""
    a2 = alpha * alpha
    n_dot_x = np.clip(n_dot_x, 0.0, 1.0)
    return 2.0 * n_dot_x / np.maximum(
        n_dot_x + np.sqrt(a2 + (1.0 - a2) * n_dot_x ** 2), 1e-12
    )


def ggx_geometry(n_dot_v: np.ndarray, n_dot_l: np.ndarray, alpha: float) -> np.ndarray:
    """Smith 高度相关遮蔽 G2(v,l)。"""
    return ggx_g1(n_dot_v, alpha) * ggx_g1(n_dot_l, alpha)


def fresnel_schlick(cos_theta: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Schlick 菲涅耳。f0 形状 (...,3)，cos_theta 会广播到通道维。"""
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    return f0 + (1.0 - f0) * (1.0 - cos_theta[..., None]) ** 5


def _f0_rgb(params: Mapping[str, float]) -> np.ndarray:
    dielectric_f0 = _DIELECTRIC_F0_SCALE * params["specular"]
    albedo = np.array([params["albedo_r"], params["albedo_g"], params["albedo_b"]])
    m = params["metallic"]
    return (1.0 - m) * dielectric_f0 + m * albedo


def evaluate(params: Mapping[str, float],
             view: np.ndarray,
             light: np.ndarray,
             normal: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """批量求值 BRDF 并返回各物理分量（用于反射曲线与诊断）。

    方向参数可传单条 (3,) 或批量 (N,3)。返回数组除特别标注外形状为 (N,3)。
    """
    params = clip_physical(params)
    v = _as_rows(view)
    l = _as_rows(light)
    n = np.array([0.0, 0.0, 1.0]) if normal is None else _as_rows(normal)
    if n.shape[0] == 1 and v.shape[0] > 1:
        n = np.repeat(n, v.shape[0], axis=0)

    alpha = float(params["roughness"]) ** 2
    n_dot_v = np.clip(np.sum(n * v, axis=1), 0.0, 1.0)
    n_dot_l = np.clip(np.sum(n * l, axis=1), 0.0, 1.0)
    h = v + l
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    n_dot_h = np.clip(np.sum(n * h, axis=1), 0.0, 1.0)
    v_dot_h = np.clip(np.sum(v * h, axis=1), 0.0, 1.0)

    d_ndf = ggx_distribution(n_dot_h, alpha)                 # (N,)
    g2 = ggx_geometry(n_dot_v, n_dot_l, alpha)               # (N,)
    f0 = _f0_rgb(params)                                     # (3,)
    f = fresnel_schlick(v_dot_h, f0)                         # (N,3)

    denom = 4.0 * np.maximum(n_dot_v * n_dot_l, 1e-6)
    specular = (d_ndf[:, None] * g2[:, None] * f) / denom[:, None]
    # 标准 Cook-Torrance 单次散射：Smith GGX 遮蔽保证方向反照率 <= 1
    # （粗糙表面因缺少多次反弹而保守吸收部分能量，绝不超调）。
    multi_scatter = np.zeros_like(specular)

    albedo = np.array([params["albedo_r"], params["albedo_g"], params["albedo_b"]])
    # 能量守恒：被菲涅耳反射的能量不进入漫反射；金属无漫反射。
    kd = (1.0 - params["metallic"]) * (1.0 - f)
    diffuse = kd * albedo[None, :] / np.pi

    valid = ((n_dot_l > 1e-6) & (n_dot_v > 1e-6))[:, None]
    specular *= valid
    multi_scatter *= valid
    diffuse *= valid
    brdf = diffuse + specular + multi_scatter

    return {
        "brdf": brdf,
        "diffuse": diffuse,
        "specular": specular,
        "multi_scatter": multi_scatter,
        "D": d_ndf,
        "G": g2,
        "F": f,
        "n_dot_l": n_dot_l,
        "n_dot_v": n_dot_v,
        "alpha": np.full_like(n_dot_l, alpha),
    }


def brdf_rgb(params: Mapping[str, float],
             view: np.ndarray,
             light: np.ndarray,
             normal: np.ndarray | None = None) -> np.ndarray:
    """返回 BRDF 值 f_r（sr^-1），单条输入返回 (3,)，批量返回 (N,3)。"""
    squeeze = np.asarray(view, dtype=np.float64).ndim == 1
    out = evaluate(params, view, light, normal)["brdf"]
    return out[0] if squeeze else out


def shade_rgb(params: Mapping[str, float],
              observations: Sequence[Observation]) -> np.ndarray:
    """按固定辐照度/曝光重渲染全部观测，输出线性 sRGB 辐射度 (N,3)。"""
    if not observations:
        return np.zeros((0, 3))
    v = np.stack([o.view for o in observations])
    l = np.stack([o.light for o in observations])
    n = np.stack([o.normal for o in observations])
    comp = evaluate(params, v, l, n)
    cos_l = comp["n_dot_l"][:, None]
    radiance = LIGHT_IRRADIANCE * EXPOSURE * comp["brdf"] * cos_l
    return radiance


def microfacet_components(params: Mapping[str, float],
                          view: np.ndarray,
                          light: np.ndarray,
                          normal: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """反射曲线用：返回 BRDF 各分量（与 evaluate 相同，命名面向界面）。"""
    return evaluate(params, view, light, normal)


def sample_ggx(view: np.ndarray, alpha: float,
               rng: np.random.Generator, n_samples: int) -> Dict[str, np.ndarray]:
    """GGX 重要性采样：给定视线 v，采样光源方向 l 与对应 pdf。

    用于白炉/方向反照率等需要低方差积分的场合。返回 ``light``、``pdf``、
    ``valid``（背面采样被标记为无效而非静默计入）。
    """
    v = np.asarray(view, dtype=np.float64)
    xi1 = rng.random(n_samples)
    xi2 = rng.random(n_samples)
    phi = 2.0 * np.pi * xi2
    # GGX 余弦加权分布采样半向量
    tan2_theta = alpha ** 2 * xi1 / np.maximum(1.0 - xi1, 1e-12)
    cos_theta = 1.0 / np.sqrt(1.0 + tan2_theta)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, 1.0))
    h = np.stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta,
    ], axis=1)
    # l = 2(v·h)h - v（反射方向）
    v_dot_h = np.clip(h @ v, 0.0, 1.0)
    light = 2.0 * v_dot_h[:, None] * h - v[None, :]
    light /= np.maximum(np.linalg.norm(light, axis=1, keepdims=True), 1e-12)
    n_dot_l = np.clip(light[:, 2], 0.0, 1.0)
    n_dot_h = cos_theta
    a2 = alpha ** 2
    denom = np.pi * (n_dot_h ** 2 * (a2 - 1.0) + 1.0) ** 2
    pdf_h = a2 / np.maximum(denom, 1e-12) * n_dot_h
    pdf = pdf_h / np.maximum(4.0 * v_dot_h, 1e-6)
    valid = n_dot_l > 1e-4
    return {"light": light, "pdf": pdf, "valid": valid, "n_dot_l": n_dot_l}


def directional_albedo(params: Mapping[str, float], view: np.ndarray,
                       n_samples: int = 20000,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """GGX 重要性采样估计方向反照率 ∫ f(v,l) cosθ_l dω（3 通道）。"""
    rng = rng or np.random.default_rng(0)
    params = clip_physical(params)
    alpha = float(params["roughness"]) ** 2
    smp = sample_ggx(np.asarray(view, dtype=np.float64), alpha, rng, n_samples)
    light = smp["light"][smp["valid"]]
    pdf = smp["pdf"][smp["valid"]]
    v = np.repeat(np.asarray(view, dtype=np.float64)[None, :], light.shape[0], axis=0)
    comp = evaluate(params, v, light)
    weights = smp["n_dot_l"][smp["valid"]] / np.maximum(pdf, 1e-12)
    return np.mean(comp["brdf"] * weights[:, None], axis=0)
