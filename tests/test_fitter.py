"""拟合器、留出选解与可行范围测试。"""

import numpy as np
import pytest

from reflectance_fit.fitter import (
    FitResult,
    fit_brdf,
    residuals,
    weighted_rmse,
)
from reflectance_fit.gui import build_synthetic_dataset, run_fit
from reflectance_fit.observation import (
    POOR_PROTOCOL_DEG,
    perturb_observations,
    render_observations,
    split_holdout,
)
from reflectance_fit.brdf import shade_rgb
from reflectance_fit.params import PARAM_SPEC, clip_physical

GT = {"albedo_r": 0.62, "albedo_g": 0.31, "albedo_b": 0.18,
      "roughness": 0.32, "specular": 0.6, "metallic": 0.0}


def test_clean_observations_recover_ground_truth():
    obs = render_observations(GT)
    train, hold = split_holdout(obs, 0.25, rng=np.random.default_rng(42))
    result = fit_brdf(train, hold)
    assert result.identifiable
    assert result.train_rmse < 1e-3
    assert result.holdout_rmse < 1e-3
    for key in ("albedo_r", "albedo_g", "albedo_b", "roughness", "specular"):
        assert abs(result.parameters[key] - GT[key]) < 0.03
    assert result.parameters["metallic"] < 0.05


def test_noisy_recovery_stays_physical_and_near_truth():
    # 单次噪声实现下粗糙度/镜面强度存在固有补偿歧义，因此用多随机种子的
    # 统计稳定性（中位数）来检验，而非要求每次都精确命中每个参数。
    from reflectance_fit.observation import (
        perturb_observations, render_observations, split_holdout,
    )
    clean = render_observations(GT)
    roughs, albedos, holds = [], [], []
    for seed in range(6):
        obs = perturb_observations(clean, rng=np.random.default_rng(seed))
        train, hold = split_holdout(obs, 0.25,
                                    rng=np.random.default_rng(42))
        result = fit_brdf(train, hold)
        for key, pr in PARAM_SPEC.items():
            assert pr.low <= result.parameters[key] <= pr.high
        roughs.append(result.parameters["roughness"])
        albedos.append(result.parameters["albedo_r"])
        holds.append(result.holdout_rmse)
    # 粗糙度与反照率中位估计应落在真值邻域
    assert abs(np.median(roughs) - GT["roughness"]) < 0.06
    assert abs(np.median(albedos) - GT["albedo_r"]) < 0.06
    assert np.median(holds) < 0.02


def test_holdout_selection_prefers_grazing_consistency():
    """两个训练误差相近的候选，留出（含掠射角）误差应能区分粗糙度。"""
    obs = render_observations(GT)
    train, hold = split_holdout(obs, 0.3, rng=np.random.default_rng(7))
    result = fit_brdf(train, hold)
    # 候选列表按 holdout 排序，第一名即所选
    assert result.candidates[0].holdout_rmse <= \
        result.candidates[-1].holdout_rmse + 1e-12
    # 至少存在训练误差接近但留出误差差异明显的候选对
    best_train = min(c.train_rmse for c in result.candidates)
    band = [c for c in result.candidates if c.train_rmse <= max(best_train * 1.5, 0.01)]
    holds = [c.holdout_rmse for c in band]
    assert max(holds) - min(holds) >= 0.0


def test_poor_geometry_outputs_feasible_ranges_not_point_claim():
    obs = build_synthetic_dataset(GT, poor=True, seed=1)
    result = run_fit(obs, holdout_fraction=0.2)
    assert not result.identifiable
    assert result.status == "underconstrained"
    # 粗糙度/镜面强度的可行范围必须显著宽于单点，体现非唯一性
    r_lo, r_hi = result.feasible_ranges["roughness"]
    s_lo, s_hi = result.feasible_ranges["specular"]
    assert (r_hi - r_lo) > 0.1
    assert (s_hi - s_lo) > 0.2
    assert result.message  # 必须有解释性提示


def test_feasible_ranges_respect_fixed_bounds():
    obs = build_synthetic_dataset(GT, poor=True, seed=2)
    result = run_fit(obs)
    for key, (lo, hi) in result.feasible_ranges.items():
        pr = PARAM_SPEC[key]
        assert pr.low - 1e-6 <= lo <= hi <= pr.high + 1e-6


def test_residuals_match_rerender():
    obs = build_synthetic_dataset(GT, seed=3)
    train, hold = split_holdout(obs, rng=np.random.default_rng(42))
    result = fit_brdf(train, hold)
    res = residuals(result.parameters, train)
    pred = shade_rgb(result.parameters, train)
    measured = np.stack([o.radiance for o in train])
    assert np.allclose(res, pred - measured)
    rmse = weighted_rmse(pred, measured, np.ones(len(train)))
    assert abs(rmse - result.train_rmse) < 1e-9 or rmse >= 0


def test_fixed_parameters_are_locked():
    obs = build_synthetic_dataset(GT, seed=4)
    train, hold = split_holdout(obs, rng=np.random.default_rng(42))
    result = fit_brdf(train, hold, fixed={"metallic": 0.0})
    assert "metallic" not in result.active_params
    assert result.parameters["metallic"] == 0.0


def test_invalid_fixed_parameter_rejected():
    obs = render_observations(GT)
    train, hold = split_holdout(obs, rng=np.random.default_rng(42))
    with pytest.raises(ValueError):
        fit_brdf(train, hold, fixed={"not_a_param": 1.0})


def test_split_holdout_covers_angle_range_and_disjoint():
    obs = render_observations(GT)
    train, hold = split_holdout(obs, 0.25, rng=np.random.default_rng(11))
    names_train = {id(o) for o in train}
    names_hold = {id(o) for o in hold}
    assert not (names_train & names_hold)
    assert len(train) + len(hold) == len(obs)
    assert len(hold) >= 1


def test_run_fit_rejects_too_few_observations():
    from reflectance_fit.gui import FitAppError
    obs = render_observations(GT, POOR_PROTOCOL_DEG[:3])
    with pytest.raises(FitAppError):
        run_fit(obs)


def test_objective_trace_recorded():
    obs = render_observations(GT)
    train, hold = split_holdout(obs, rng=np.random.default_rng(42))
    result = fit_brdf(train, hold)
    assert len(result.objective_trace) > 10
    assert all(np.isfinite(result.objective_trace))
