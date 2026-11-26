"""反射曲线与材质球预览——全部由 :mod:`brdf` 真实求值驱动。

* :func:`reflection_curve` 在固定入射方向下扫描出射角，输出 BRDF 各分量；
* :func:`render_sphere` 对半球法线网格做着色，生成材质球预览（线性 sRGB）；
* :func:`to_display_srgb` 负责线性->sRGB 编码与简单曝光裁剪。

预览中相机固定在 +Z 轴，主光从固定方向打向球体，球体上每一点的法线由
几何计算得到——因此这是真正的重渲染，不是把拟合颜色平铺成一个圆。
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import numpy as np

from .brdf import evaluate
from .params import EXPOSURE, LIGHT_IRRADIANCE, linear_to_srgb

PREVIEW_LIGHT = np.array([0.45, -0.55, 0.70])
PREVIEW_LIGHT = PREVIEW_LIGHT / np.linalg.norm(PREVIEW_LIGHT)
PREVIEW_VIEW = np.array([0.0, 0.0, 1.0])


def reflection_curve(
    params: Mapping[str, float],
    *,
    theta_light_deg: float = 45.0,
    phi_light_deg: float = 0.0,
    n_samples: int = 121,
) -> Dict[str, np.ndarray]:
    """固定光源，扫描出射极角 -80°..80°（镜面反射方向附近最有信息量）。

    返回 theta_deg、总 BRDF、漫反射、镜面（各为 (N,3) 线性值）及
    标量亮度曲线（按 Rec.709 加权）。
    """
    tl = math.radians(theta_light_deg)
    pl = math.radians(phi_light_deg)
    light = np.array([
        math.sin(tl) * math.cos(pl),
        math.sin(tl) * math.sin(pl),
        math.cos(tl),
    ])
    thetas = np.linspace(-80.0, 80.0, n_samples)
    views = []
    for t in thetas:
        tr = math.radians(abs(t))
        # 镜面反射方向与光源水平分量相反（v_mirror 的方位 = 光源方位 +π），
        # 因此正角度取对侧方位以让镜像峰出现在 +入射角 一侧。
        phi = pl + math.pi if t >= 0 else pl
        views.append([
            math.sin(tr) * math.cos(phi),
            math.sin(tr) * math.sin(phi),
            math.cos(tr),
        ])
    views = np.asarray(views)
    lights = np.repeat(light[None, :], n_samples, axis=0)
    comp = evaluate(params, views, lights)
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    return {
        "theta_deg": thetas,
        "brdf": comp["brdf"],
        "diffuse": comp["diffuse"],
        "specular": comp["specular"],
        "luminance": comp["brdf"] @ lum_w,
        "diffuse_luminance": comp["diffuse"] @ lum_w,
        "specular_luminance": comp["specular"] @ lum_w,
        "specular_peak_angle": float(thetas[np.argmax(comp["specular"] @ lum_w)]),
    }


def render_sphere(
    params: Mapping[str, float],
    *,
    size: int = 160,
    light: np.ndarray | None = None,
) -> np.ndarray:
    """渲染正交投影材质球，返回线性 sRGB 浮点图 (size,size,3)。"""
    light = PREVIEW_LIGHT if light is None else light
    light = np.asarray(light, dtype=np.float64)
    light = light / np.linalg.norm(light)

    axis = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(axis, -axis)  # 屏幕 y 向下
    rr2 = xx ** 2 + yy ** 2
    mask = rr2 <= 1.0
    zz = np.sqrt(np.clip(1.0 - rr2, 0.0, 1.0))
    normals = np.stack([xx, yy, zz], axis=-1)            # (S,S,3)
    n_flat = normals[mask]
    v_flat = np.repeat(PREVIEW_VIEW[None, :], n_flat.shape[0], axis=0)
    l_flat = np.repeat(light[None, :], n_flat.shape[0], axis=0)
    comp = evaluate(params, v_flat, l_flat, n_flat)
    radiance = (
        LIGHT_IRRADIANCE * EXPOSURE * comp["brdf"]
        * comp["n_dot_l"][:, None]
    )

    img = np.zeros((size, size, 3), dtype=np.float64)
    img[mask] = radiance
    # 边缘暗化（球面朝向的自然结果已含 cos 项，这里加轻微环境遮挡感不另造假光）
    img[~mask] = 0.0
    return img


def to_display_srgb(linear_image: np.ndarray,
                    *,
                    gain: float = 1.0) -> np.ndarray:
    """线性 HDR 图 -> 可显示的 8-bit sRGB（裁剪高光，不改变拟合尺度）。"""
    from .params import linear_to_srgb
    scaled = np.clip(linear_image * gain, 0.0, None)
    srgb = linear_to_srgb(scaled)
    return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def preview_gain_for(linear_image: np.ndarray,
                     percentile: float = 99.0) -> float:
    """为显示选择增益：使给定百分位亮度映射到 sRGB 中上段，仅用于显示。"""
    vals = linear_image[linear_image.sum(axis=-1) > 0]
    if vals.size == 0:
        return 1.0
    target = np.percentile(vals @ np.array([0.2126, 0.7152, 0.0722]),
                           percentile)
    if target <= 1e-8:
        return 1.0
    return float(min(20.0, max(0.5, 0.5 / target)))


def fit_gain_consistency_check(
    train_radiance: np.ndarray, measured_radiance: np.ndarray
) -> Tuple[float, float]:
    """尺度一致性诊断：若最优单通道增益显著偏离 1，说明存在未建模尺度问题。

    返回 (最优最小二乘增益, 相对偏差)。正常受控拟合该增益应接近 1。
    """
    a = train_radiance.reshape(-1)
    b = measured_radiance.reshape(-1)
    denom = float(a @ a)
    if denom < 1e-12:
        return 1.0, 0.0
    gain = float(a @ b) / denom
    return gain, abs(gain - 1.0)
