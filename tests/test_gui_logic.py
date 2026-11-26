"""GUI 后端业务逻辑测试（无需真实显示器）。

完整 Qt 窗口需要系统图形库（libEGL 等），在无头 CI 中不可用；这里覆盖
GUI 调用的全部纯逻辑：合成数据集、ROI 聚合、固定参数校验、拟合与报告。
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from reflectance_fit.gui import (
    FitAppError,
    build_report_figure,
    build_synthetic_dataset,
    run_fit,
)
from reflectance_fit.observation import render_observations
from reflectance_fit.texture import (
    ViewImage,
    set_roi,
    synthetic_view_image,
    views_to_observations,
)
import math

GT = {"albedo_r": 0.62, "albedo_g": 0.31, "albedo_b": 0.18,
      "roughness": 0.32, "specular": 0.6, "metallic": 0.0}


def _views_from_obs():
    clean = render_observations(GT)
    views = []
    for obs in clean:
        tv = math.degrees(math.acos(obs.view[2]))
        tl = math.degrees(math.acos(obs.light[2]))
        img = synthetic_view_image(obs.radiance, h=48, w=48)
        views.append(ViewImage(img, tv, 0.0, tl, 0.0, name=obs.name))
    return views


def test_full_gui_backend_workflow():
    views = _views_from_obs()
    for v in views:
        set_roi(v, 10, 10, 38, 38)
    observations = views_to_observations(views)
    result = run_fit(observations)
    assert result.n_train + result.n_holdout == len(observations)
    fig = build_report_figure(GT, result, observations)
    # 残差、曲线、预览、参数表四个轴
    assert len(fig.axes) >= 4


def test_run_fit_rejects_missing_roi_via_aggregation():
    views = _views_from_obs()
    # 不设置任何 ROI
    with pytest.raises(ValueError):
        views_to_observations(views)


def test_synthetic_poor_dataset_flags_underconstrained():
    obs = build_synthetic_dataset(GT, poor=True, seed=3)
    result = run_fit(obs, holdout_fraction=0.2)
    assert not result.identifiable
    assert result.feasible_ranges["specular"][1] > \
        result.feasible_ranges["specular"][0]


def test_run_fit_too_few_observations_raises_fitapperror():
    obs = render_observations(GT)[0:3]
    with pytest.raises(FitAppError):
        run_fit(obs)


def test_report_handles_missing_ground_truth():
    obs = build_synthetic_dataset(GT, seed=5)
    result = run_fit(obs)
    fig = build_report_figure(None, result, obs)
    assert len(fig.axes) >= 4
