"""合成观测生成与受控采集协议。

用途
====
1. 用**自定义真值参数**经真实 BRDF 渲染多角度观测（不是贴图复制）；
2. 注入测量噪声与轻微标定误差，用于检验参数恢复的稳定性；
3. 把观测划分为优化集与留出验证集——两者视角严格分开，防止优化器
   "只记住某一张高光图"。

噪声模型（线性空间）
====================
* 散粒噪声近似：高斯，标准差 = ``noise_sigma * sqrt(radiance) + floor``；
* 标定误差：对每个方向施加小角度旋转扰动（角度噪声），并对光源方向
  施加固定的小幅系统偏置（模拟标定偏差）。拟合器拿到的是**扰动后**的
  几何，但不知道具体偏置——这正是稳定性测试要考察的内容。
"""

from __future__ import annotations

import math
from typing import List, Mapping, Sequence, Tuple

import numpy as np

from .brdf import shade_rgb
from .geometry import Observation, direction_from_angles
from .params import clip_physical

#: 标准受控采集协议：(theta_view_deg, phi_view_deg, theta_light_deg, phi_light_deg)
#: 同时覆盖近轴、45° 与掠入射，保证粗糙度/镜面强度有被区分的几何信息。
DEFAULT_PROTOCOL_DEG: List[Tuple[float, float, float, float]] = [
    (0.0,   0.0,  15.0,   0.0),
    (0.0,   0.0,  30.0,   0.0),
    (0.0,   0.0,  45.0,   0.0),
    (0.0,   0.0,  60.0,   0.0),
    (0.0,   0.0,  75.0,   0.0),
    (30.0,  0.0,  30.0,   0.0),
    (45.0,  0.0,  45.0,   0.0),
    (45.0, 90.0,  45.0,  90.0),
    (60.0,  0.0,  60.0,   0.0),
    (60.0,  0.0,  30.0, 180.0),
    (75.0,  0.0,  75.0,   0.0),
    (75.0,  0.0,  45.0, 180.0),
]

#: 视角贫乏协议：全部为近轴小角度（无 >25° 侧视），用于触发"可行参数范围"
#: 输出分支。仍保留 5 个角度以满足留出划分的最低数据量。
POOR_PROTOCOL_DEG: List[Tuple[float, float, float, float]] = [
    (0.0, 0.0, 8.0, 0.0),
    (0.0, 0.0, 15.0, 0.0),
    (0.0, 0.0, 22.0, 0.0),
    (8.0, 0.0, 12.0, 0.0),
    (15.0, 0.0, 20.0, 0.0),
]


def protocol_to_radians(
    protocol_deg: Sequence[Tuple[float, float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    return [tuple(math.radians(a) for a in row) for row in protocol_deg]


def _rotated(direction: np.ndarray, d_theta: float, d_phi: float) -> np.ndarray:
    theta = math.acos(np.clip(direction[2], -1.0, 1.0)) + d_theta
    phi = math.atan2(direction[1], direction[0]) + d_phi
    return direction_from_angles(theta, phi)


def render_observations(
    ground_truth: Mapping[str, float],
    protocol_deg: Sequence[Tuple[float, float, float, float]] = DEFAULT_PROTOCOL_DEG,
) -> List[Observation]:
    """用真值参数渲染干净的多角度观测。"""
    ground_truth = clip_physical(ground_truth)
    obs: List[Observation] = []
    for i, (tv, pv, tl, pl) in enumerate(protocol_deg):
        o = Observation(
            view=direction_from_angles(math.radians(tv), math.radians(pv)),
            light=direction_from_angles(math.radians(tl), math.radians(pl)),
            radiance=np.zeros(3),
            name=f"obs{i:02d}_v{tv:.0f}_l{tl:.0f}",
        )
        obs.append(o)
    radiance = shade_rgb(ground_truth, obs)
    for o, r in zip(obs, radiance):
        o.radiance = r
    return obs


def perturb_observations(
    observations: Sequence[Observation],
    *,
    noise_sigma: float = 0.005,
    noise_floor: float = 0.001,
    angle_jitter_deg: float = 0.3,
    light_bias_deg: float = 0.3,
    rng: np.random.Generator | None = None,
) -> List[Observation]:
    """返回带噪声与标定误差的新观测列表（不修改输入）。

    参数
    ----
    noise_sigma:
        散粒噪声相对强度（标准差 = sigma*sqrt(辐射度)），零均值、不改变
        期望能量，因此不与"固定曝光/光强"的尺度约定冲突。
    noise_floor:
        加性暗噪声标准差（零均值）。
    angle_jitter_deg:
        每个方向独立的角度随机扰动（标定噪声）。
    light_bias_deg:
        光源方向的系统性偏置幅度（所有观测共享同一符号的极角偏移）。

    注意：这里**不**注入逐观测乘性增益噪声，因为那等价于自由改变曝光，
    会破坏本项目固定尺度的可辨识性前提（见 :mod:`reflectance_fit.params`）。
    """
    rng = rng or np.random.default_rng(0)
    jitter = math.radians(angle_jitter_deg)
    bias = math.radians(light_bias_deg)
    out: List[Observation] = []
    for o in observations:
        v_pert = _rotated(
            o.view,
            rng.normal(0.0, jitter),
            rng.normal(0.0, jitter),
        )
        l_pert = _rotated(
            o.light,
            bias + rng.normal(0.0, jitter),
            rng.normal(0.0, jitter),
        )
        clean = o.radiance
        shot = noise_sigma * np.sqrt(np.maximum(clean, 0.0)) \
            * rng.standard_normal(3)
        dark = rng.normal(0.0, noise_floor, size=3)
        noisy = np.maximum(clean + shot + dark, 0.0)
        out.append(
            Observation(
                view=v_pert,
                light=l_pert,
                normal=o.normal.copy(),
                radiance=noisy,
                weight=o.weight,
                name=o.name,
            )
        )
    return out


def split_holdout(
    observations: Sequence[Observation],
    holdout_fraction: float = 0.25,
    *,
    rng: np.random.Generator | None = None,
) -> Tuple[List[Observation], List[Observation]]:
    """按视角分层留出：近轴/侧视/掠射各层都至少有一条进入验证集。"""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction 必须位于 (0,1)")
    rng = rng or np.random.default_rng(42)
    order = list(range(len(observations)))
    rng.shuffle(order)
    n_hold = max(1, round(len(observations) * holdout_fraction))

    # 分层：按最大极角排序后等距抽取，保证留出集覆盖角度范围。
    def max_angle_deg(o: Observation) -> float:
        tv = math.degrees(math.acos(np.clip(float(np.dot(o.normal, o.view)), -1, 1)))
        tl = math.degrees(math.acos(np.clip(float(np.dot(o.normal, o.light)), -1, 1)))
        return max(tv, tl)

    ranked = sorted(order, key=lambda i: max_angle_deg(observations[i]))
    hold_idx = set(ranked[:: max(1, len(ranked) // n_hold)][:n_hold])
    train = [observations[i] for i in order if i not in hold_idx]
    hold = [observations[i] for i in order if i in hold_idx]
    if not train or not hold:  # 极端小样本兜底
        hold_idx = {ranked[-1]}
        train = [observations[i] for i in order if i not in hold_idx]
        hold = [observations[ranked[-1]]]
    return train, hold


# 灯光选点函数也放在观测模块命名空间，方便旧调用方按“下一观测”导入。
from .lighting import (  # noqa: E402
    render_lighting_preview,
    select_next_observation,
)

__all__ = [
    "DEFAULT_PROTOCOL_DEG",
    "POOR_PROTOCOL_DEG",
    "protocol_to_radians",
    "render_observations",
    "perturb_observations",
    "split_holdout",
    "render_lighting_preview",
    "select_next_observation",
]
