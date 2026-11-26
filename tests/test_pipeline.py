"""参数空间、ROI 纹理管线与渲染模块测试。"""

import math
import os

import numpy as np
import pytest

from reflectance_fit.params import (
    PARAM_SPEC,
    clip_physical,
    fit_params_to_physical,
    linear_to_srgb,
    physical_to_fit_params,
    srgb_to_linear,
)
from reflectance_fit.render import (
    reflection_curve,
    render_sphere,
    to_display_srgb,
)
from reflectance_fit.texture import (
    ViewImage,
    set_roi,
    synthetic_view_image,
    views_to_observations,
)
from reflectance_fit.brdf import shade_rgb
from reflectance_fit.observation import render_observations


def test_logit_sigmoid_roundtrip_and_bounds():
    # 内点可严格往返；边界点经 logit 截断到 _EPS，单独验证裁剪行为。
    p = {"albedo_r": 0.62, "albedo_g": 0.31, "albedo_b": 0.18,
         "roughness": 0.32, "specular": 0.6, "metallic": 0.25}
    z = physical_to_fit_params(p)
    back = fit_params_to_physical(z)
    for key, value in p.items():
        assert abs(back[key] - value) < 1e-6
    # 极端无界输入也必须落在物理范围内且不溢出
    extreme = fit_params_to_physical(np.full(len(PARAM_SPEC), 1e4))
    for key, pr in PARAM_SPEC.items():
        assert pr.low <= extreme[key] <= pr.high
    # 物理边界往返：恢复值被安全地夹在范围内，不报错
    boundary = fit_params_to_physical(
        physical_to_fit_params({k: pr.low for k, pr in PARAM_SPEC.items()})
    )
    for key, pr in PARAM_SPEC.items():
        assert pr.low <= boundary[key] < pr.low + 1e-3


def test_clip_enforces_fixed_ranges():
    bad = {"albedo_r": 5.0, "albedo_g": -2.0, "albedo_b": 0.5,
           "roughness": 0.0, "specular": 9.0, "metallic": 0.5}
    clipped = clip_physical(bad)
    assert clipped["albedo_r"] == PARAM_SPEC["albedo_r"].high
    assert clipped["albedo_g"] == PARAM_SPEC["albedo_g"].low
    assert clipped["roughness"] == PARAM_SPEC["roughness"].low
    assert clipped["specular"] == PARAM_SPEC["specular"].high


def test_srgb_roundtrip():
    linear = np.array([0.0, 0.002, 0.18, 0.5, 0.95])
    back = srgb_to_linear(linear_to_srgb(linear))
    assert np.allclose(back, linear, atol=2e-3)


def test_roi_pipeline_aggregates_to_observations():
    gt = {"albedo_r": 0.5, "albedo_g": 0.4, "albedo_b": 0.3,
          "roughness": 0.4, "specular": 0.5, "metallic": 0.0}
    clean = render_observations(gt)
    views = []
    for obs in clean:
        img = synthetic_view_image(obs.radiance, h=40, w=40)
        tv = math.degrees(math.acos(obs.view[2]))
        tl = math.degrees(math.acos(obs.light[2]))
        views.append(ViewImage(img, tv, 0.0, tl, 0.0, name=obs.name))
    for v in views:
        set_roi(v, 8, 8, 32, 32)
    observations = views_to_observations(views)
    assert len(observations) == len(clean)
    # 聚合值应接近真值渲染（sRGB 往返 + 截尾均值带来小误差）
    for agg, ref in zip(observations, clean):
        assert np.allclose(agg.radiance, ref.radiance, atol=0.02)
        assert agg.weight > 0


def test_roi_rejects_tiny_region():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    v = ViewImage(img, 0, 0, 30, 0)
    with pytest.raises(ValueError):
        set_roi(v, 5, 5, 5, 6)


def test_views_to_observations_requires_roi():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    v = ViewImage(img, 0, 0, 30, 0)
    with pytest.raises(ValueError):
        views_to_observations([v])


def test_reflection_curve_components_and_peak():
    params = {"albedo_r": 0.6, "albedo_g": 0.3, "albedo_b": 0.2,
              "roughness": 0.2, "specular": 0.7, "metallic": 0.0}
    curve = reflection_curve(params, theta_light_deg=45.0)
    assert curve["theta_deg"].shape == curve["luminance"].shape
    assert np.all(curve["diffuse_luminance"] >= 0)
    assert np.all(curve["specular_luminance"] >= 0)
    # 镜面峰应在镜面反射方向附近（入射角 45° -> 出射 ~45°）
    assert 25.0 < curve["specular_peak_angle"] < 65.0
    # 更粗糙的表面峰值更低
    rough_curve = reflection_curve({**params, "roughness": 0.8})
    assert rough_curve["specular_luminance"].max() < \
        curve["specular_luminance"].max()


def test_render_sphere_is_real_shading_not_flat_color():
    params = {"albedo_r": 0.5, "albedo_g": 0.5, "albedo_b": 0.5,
              "roughness": 0.3, "specular": 0.8, "metallic": 0.0}
    img = render_sphere(params, size=80)
    assert img.shape == (80, 80, 3)
    inside = img[img.sum(axis=-1) > 0]
    # 球面着色必须存在空间变化（高光），不能是平铺纯色
    assert inside.std(axis=0).max() > 1e-4
    display = to_display_srgb(img)
    assert display.dtype == np.uint8
    assert display.max() <= 255


def test_load_view_directory_manifest_validation(tmp_path):
    from reflectance_fit.io_texture import load_view_directory
    with pytest.raises(FileNotFoundError):
        load_view_directory(str(tmp_path))
    # 列不全的清单
    manifest = tmp_path / "views.csv"
    manifest.write_text("name,file\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_view_directory(str(tmp_path))
