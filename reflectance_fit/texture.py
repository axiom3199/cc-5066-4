"""材质区域选取与按区域聚合的观测构建（纹理制作界面的后端）。

受控采集通常得到一组对齐的多视角图像。本模块负责：

* 保存每张图像对应的已知观测条件（视角/光向/区域掩膜）；
* 在图像上以矩形或多边形选取材质区域（ROI），对区域内像素做稳健平均，
  得到该角度下的一条 BRDF 观测；
* 生成区域纹理预览（sRGB PNG）供 GUI 显示。

注意：区域平均只是把多个像素的**测量值**聚合为一条观测，后续拟合仍然走
完整 BRDF 重渲染比较，不存在"平均颜色直接出 PBR 参数"的路径。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import Observation, direction_from_angles
from .params import linear_to_srgb, srgb_to_linear


@dataclass
class ViewImage:
    """一张受控采集图像及其已知几何。"""
    image_srgb: np.ndarray          # (H,W,3) uint8 或 float sRGB
    theta_view_deg: float
    phi_view_deg: float
    theta_light_deg: float
    phi_light_deg: float
    name: str = ""
    roi: Optional[Tuple[int, int, int, int]] = None  # (x0, y0, x1, y1)

    def linear_image(self) -> np.ndarray:
        img = self.image_srgb.astype(np.float64)
        if img.max() > 1.5:
            img = img / 255.0
        return srgb_to_linear(img)


def set_roi(view: ViewImage, x0: int, y0: int, x1: int, y1: int) -> ViewImage:
    h, w = view.image_srgb.shape[:2]
    x0, x1 = sorted((int(max(0, min(w - 1, x0))), int(max(0, min(w - 1, x1)))))
    y0, y1 = sorted((int(max(0, min(h - 1, y0))), int(max(0, min(h - 1, y1)))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("ROI 至少需要 2x2 像素")
    view.roi = (x0, y0, x1, y1)
    return view


def _robust_mean(pixels_linear: np.ndarray) -> np.ndarray:
    """截尾均值：丢弃最亮 5%（高光饱和像素）与最暗 5%，抑制异常点。"""
    if pixels_linear.shape[0] <= 4:
        return pixels_linear.mean(axis=0)
    lum = pixels_linear @ np.array([0.2126, 0.7152, 0.0722])
    lo, hi = np.quantile(lum, (0.05, 0.95))
    keep = (lum >= lo) & (lum <= hi)
    selected = pixels_linear[keep] if keep.sum() >= 3 else pixels_linear
    return selected.mean(axis=0)


def views_to_observations(views: Sequence[ViewImage]) -> List[Observation]:
    """把带 ROI 的多视角图像聚合为 BRDF 观测列表。"""
    obs: List[Observation] = []
    for idx, view in enumerate(views):
        if view.roi is None:
            raise ValueError(f"视图 {view.name or idx} 尚未选取材质区域")
        x0, y0, x1, y1 = view.roi
        patch = view.linear_image()[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3)
        radiance = _robust_mean(patch)
        obs.append(Observation(
            view=direction_from_angles(math.radians(view.theta_view_deg),
                                       math.radians(view.phi_view_deg)),
            light=direction_from_angles(math.radians(view.theta_light_deg),
                                        math.radians(view.phi_light_deg)),
            radiance=radiance,
            weight=float(patch.shape[0]),
            name=view.name or f"view{idx:02d}",
        ))
    return obs


def make_swatch_texture(linear_rgb: np.ndarray, size: int = 64) -> np.ndarray:
    """把拟合反照率做成 sRGB 色块纹理（供材质库/预览使用）。"""
    rgb = np.clip(linear_to_srgb(np.asarray(linear_rgb, dtype=np.float64)),
                  0, 1)
    return (np.repeat(rgb[None, :], size, axis=0)
              .reshape(1, -1).repeat(size, axis=0).reshape(size, size, 3)
            * 255).astype(np.uint8)


def synthetic_view_image(linear_radiance: np.ndarray, h: int = 96,
                         w: int = 96) -> np.ndarray:
    """把均匀线性辐射度转成 sRGB 测试图（无真实相机时的合成入口）。"""
    img = np.repeat(linear_radiance[None, None, :], h, axis=0)
    img = np.repeat(img, w, axis=1)
    return (np.clip(linear_to_srgb(img), 0, 1) * 255).astype(np.uint8)
