"""已知几何：相机方向、光照方向与观测数据。

拟合的前提是**受控**采集——几何关系已知。每个观测包含：

* 表面法线 ``n``（本项目平面样品默认 +Z，接口保留逐观测法线以支持曲面区域）；
* 指向相机的单位向量 ``v`` 与指向光源的单位向量 ``l``；
* 该方向测得的线性 sRGB 辐射度 ``radiance``（3 通道）；
* 可选权重 ``weight``（如像素数量、置信度）。

约定：所有方向向量指向**离开表面**的半空间，单位长度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .params import srgb_to_linear


def vec(x: float, y: float, z: float) -> np.ndarray:
    v = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("零向量无法归一化为方向")
    return v / norm


def direction_from_angles(theta: float, phi: float) -> np.ndarray:
    """极角 theta（与 +Z 法线夹角，弧度）、方位角 phi -> 单位方向。"""
    st = math.sin(theta)
    return np.array([st * math.cos(phi), st * math.sin(phi), math.cos(theta)],
                    dtype=np.float64)


@dataclass
class Observation:
    view: np.ndarray          # v：指向相机
    light: np.ndarray         # l：指向光源
    radiance: np.ndarray      # 线性 sRGB，shape (3,)
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    weight: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        self.view = np.asarray(self.view, dtype=np.float64)
        self.light = np.asarray(self.light, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        self.radiance = np.asarray(self.radiance, dtype=np.float64).reshape(3)
        for name_, d in (("view", self.view), ("light", self.light), ("normal", self.normal)):
            if abs(np.linalg.norm(d) - 1.0) > 1e-6:
                raise ValueError(f"{name_} 方向必须为单位向量")
        if float(np.dot(self.normal, self.view)) <= 1e-6:
            raise ValueError(f"观测 {self.name!r} 的视线位于表面背面，无法采样平面 BRDF")
        if float(np.dot(self.normal, self.light)) <= 1e-6:
            raise ValueError(f"观测 {self.name!r} 的光源位于表面背面")


def make_observation_set(
    angles: Sequence[tuple[float, float, float, float]],
    radiances: Optional[Iterable[np.ndarray]] = None,
    *,
    radiance_is_srgb: bool = False,
    weights: Optional[Sequence[float]] = None,
) -> List[Observation]:
    """从角度元组批量构造观测。

    ``angles`` 每项为 ``(theta_view, phi_view, theta_light, phi_light)``，弧度。
    ``radiances`` 为 None 时先放零值，后续由 ``observation.render_observations``
    用真值材质填充。
    """
    rad_iter = iter(radiances) if radiances is not None else None
    obs: List[Observation] = []
    for i, (tv, pv, tl, pl) in enumerate(angles):
        radiance = next(rad_iter) if rad_iter is not None else np.zeros(3)
        radiance = np.asarray(radiance, dtype=np.float64)
        if radiance_is_srgb:
            radiance = srgb_to_linear(radiance)
        obs.append(
            Observation(
                view=direction_from_angles(tv, pv),
                light=direction_from_angles(tl, pl),
                radiance=radiance,
                weight=1.0 if weights is None else float(weights[i]),
                name=f"obs{i:02d}",
            )
        )
    return obs


def grazing_fraction(observations: Sequence[Observation],
                     threshold_deg: float = 45.0) -> float:
    """侧视（极角大于阈值）观测占比，用于判断可辨识性。"""
    if not observations:
        return 0.0
    n_obs = len(observations)
    grazing = 0
    for o in observations:
        theta_v = math.acos(np.clip(float(np.dot(o.normal, o.view)), -1, 1))
        theta_l = math.acos(np.clip(float(np.dot(o.normal, o.light)), -1, 1))
        if math.degrees(max(theta_v, theta_l)) > threshold_deg:
            grazing += 1
    return grazing / n_obs
