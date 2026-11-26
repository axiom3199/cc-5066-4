"""反射模型物理性质测试。"""

import numpy as np

from reflectance_fit.brdf import (
    brdf_rgb,
    directional_albedo,
    evaluate,
    fresnel_schlick,
    ggx_distribution,
    ggx_geometry,
    shade_rgb,
)
from reflectance_fit.geometry import Observation, direction_from_angles
from reflectance_fit.params import EXPOSURE, LIGHT_IRRADIANCE


def _params(**over):
    p = {"albedo_r": 0.5, "albedo_g": 0.4, "albedo_b": 0.3,
         "roughness": 0.4, "specular": 0.5, "metallic": 0.0}
    p.update(over)
    return p


def test_ggx_normalizes_over_hemisphere():
    alpha = 0.35
    rng = np.random.default_rng(1)
    n = 60000
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2])
    cos_t = d[:, 2]
    # 均匀球面采样：积分 D*cos 用 2π * mean(D*cos)
    ndf = ggx_distribution(cos_t, alpha)
    integral = 2 * np.pi * np.mean(ndf * cos_t)
    assert abs(integral - 1.0) < 0.03


def test_brdf_non_negative_and_reciprocal():
    v = np.array([0.3, 0.1, 0.95])
    l = np.array([-0.2, 0.4, 0.89])
    v /= np.linalg.norm(v)
    l /= np.linalg.norm(l)
    p = _params()
    f1 = brdf_rgb(p, v, l)
    f2 = brdf_rgb(p, l, v)
    assert np.all(f1 >= 0)
    assert np.allclose(f1, f2, atol=1e-10)


def test_white_furnace_energy_upper_bound_metallic():
    # 标准 Cook-Torrance 单次散射在 Smith 遮蔽下方向反照率 <= 1
    # （能量守恒上界）。GGX 重要性采样低方差估计。
    v0 = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(2)
    for rough in (0.05, 0.5, 1.0):
        p = _params(roughness=rough, metallic=1.0, specular=1.0,
                    albedo_r=0.99, albedo_g=0.99, albedo_b=0.99)
        energy = directional_albedo(p, v0, n_samples=30000, rng=rng)
        assert np.all(energy <= 1.0 + 0.02), energy
        assert np.all(energy > 0.3), energy


def test_smooth_surface_near_unit_albedo():
    # 光滑导体单次散射损失很小，方向反照率应接近 1（白炉近似闭合）。
    v0 = np.array([0.0, 0.0, 1.0])
    p = _params(roughness=0.05, metallic=1.0, specular=1.0,
                albedo_r=0.99, albedo_g=0.99, albedo_b=0.99)
    energy = directional_albedo(p, v0, n_samples=30000,
                                rng=np.random.default_rng(4))
    assert np.all(energy > 0.9)


def test_dielectric_energy_conservation():
    v0 = np.array([0.0, 0.0, 1.0])
    p = _params(roughness=0.3, metallic=0.0, specular=1.0,
                albedo_r=0.9, albedo_g=0.9, albedo_b=0.9)
    energy = directional_albedo(p, v0, n_samples=30000,
                                rng=np.random.default_rng(3))
    # 介电质吸收部分能量，但绝不超过 1
    assert np.all(energy <= 1.0 + 1e-6)
    assert np.all(energy > 0.5)


def test_fresnel_endpoints():
    f0 = np.array([[0.04, 0.04, 0.04]])
    f_normal = fresnel_schlick(np.array([1.0]), f0)
    f_grazing = fresnel_schlick(np.array([0.0]), f0)
    assert np.allclose(f_normal, f0, atol=1e-6)
    assert np.allclose(f_grazing, 1.0, atol=1e-6)


def test_smith_geometry_bounded():
    c = np.linspace(0.01, 1.0, 50)
    g = ggx_geometry(c, c, 0.4)
    assert np.all(g >= 0) and np.all(g <= 1.0 + 1e-9)


def test_shade_uses_fixed_scale_only():
    obs = [Observation(
        view=direction_from_angles(0.0, 0.0),
        light=direction_from_angles(np.radians(30), 0.0),
        radiance=np.zeros(3),
    )]
    p = _params()
    rad = shade_rgb(p, obs)
    comp = evaluate(p, obs[0].view[None], obs[0].light[None])
    expected = LIGHT_IRRADIANCE * EXPOSURE * comp["brdf"][0] * comp["n_dot_l"][0]
    assert np.allclose(rad[0], expected)


def test_metallic_has_no_diffuse():
    p = _params(metallic=1.0)
    v = direction_from_angles(np.radians(20), 0.0)
    l = direction_from_angles(np.radians(40), 0.0)
    comp = evaluate(p, v[None], l[None])
    assert np.allclose(comp["diffuse"], 0.0, atol=1e-12)
