"""命令行辅助工具。

注意：反射拟合的**主入口是图形界面**（``python -m reflectance_fit.gui``），
CLI 仅用于脚本化演示、无头测试与报告导出。

子命令
======
* ``demo``    : 用自定义真值生成多角度观测，加噪声与标定误差后拟合，
                打印恢复参数/可行范围并可导出报告图；
* ``fit-dir`` : 从含 ``views.csv`` 的图像目录读取 ROI 观测并拟合；
* ``check``   : 只做 BRDF 自检（能量守恒、白炉测试、互易性）。
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np

from .brdf import brdf_rgb, evaluate
from .gui import build_report_figure, build_synthetic_dataset, run_fit
from .io_texture import load_view_directory
from .params import PARAM_SPEC, clip_physical
from .texture import set_roi, views_to_observations


def _parse_gt(text: str | None) -> Dict[str, float]:
    gt = {
        "albedo_r": 0.62, "albedo_g": 0.31, "albedo_b": 0.18,
        "roughness": 0.32, "specular": 0.6, "metallic": 0.0,
    }
    if text:
        overrides = json.loads(text)
        for key, value in overrides.items():
            if key not in PARAM_SPEC:
                raise SystemExit(f"未知参数: {key}")
            gt[key] = float(value)
    return clip_physical(gt)


def _print_result(result, gt: Dict[str, float] | None) -> None:
    print(result.summary())
    if gt is not None:
        print("\n真值对比:")
        for key in result.active_params:
            err = abs(result.parameters[key] - gt[key])
            print(f"  {key:10s} 真值={gt[key]:.4f} 拟合={result.parameters[key]:.4f} "
                  f"绝对偏差={err:.4f}")


def _result_payload(result, gt: Dict[str, float] | None = None) -> Dict[str, object]:
    return {
        "parameters": result.parameters,
        "feasible_ranges": result.feasible_ranges,
        "train_rmse": result.train_rmse,
        "holdout_rmse": result.holdout_rmse,
        "identifiable": result.identifiable,
        "status": result.status,
        "implementation": result.implementation,
        "backend": result.backend,
        "contract_version": result.contract_version,
        "optimizer_success": result.optimizer_success,
        "stopping_reason": result.stopping_reason,
        "boundary_checks": result.boundary_checks,
        "execution_evidence": result.execution_evidence,
        "validation_issues": result.validation_issues,
        "ground_truth": gt,
    }


def _write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"JSON 结果已写入: {path}")


def cmd_demo(args: argparse.Namespace) -> int:
    gt = _parse_gt(args.ground_truth)
    print("真值参数:", json.dumps(gt, ensure_ascii=False))
    observations = build_synthetic_dataset(
        gt, poor=args.poor,
        noise_sigma=args.noise,
        angle_jitter_deg=args.jitter,
        light_bias_deg=args.bias,
        seed=args.seed,
    )
    fixed = json.loads(args.fixed) if args.fixed else None
    result = run_fit(observations, holdout_fraction=args.holdout,
                     fixed=fixed, seed=args.seed)
    _print_result(result, None if args.poor else gt)

    if args.report:
        import matplotlib
        matplotlib.use("Agg")
        fig = build_report_figure(None if args.poor else gt, result, observations)
        fig.savefig(args.report, dpi=130)
        print(f"\n报告图已写入: {args.report}")
    if args.json_out:
        _write_json(args.json_out, _result_payload(result, gt))
    return 0


def cmd_fit_dir(args: argparse.Namespace) -> int:
    views = load_view_directory(args.directory)
    for view in views:
        h, w = view.image_srgb.shape[:2]
        set_roi(view, w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    observations = views_to_observations(views)
    result = run_fit(observations, holdout_fraction=args.holdout)
    _print_result(result, None)
    if args.report:
        import matplotlib
        matplotlib.use("Agg")
        fig = build_report_figure(None, result, observations)
        fig.savefig(args.report, dpi=130)
        print(f"\n报告图已写入: {args.report}")
    if args.json_out:
        _write_json(args.json_out, _result_payload(result))
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    failures: List[str] = []

    # 1. 互易性 f(wi,wo) == f(wo,wi)
    v = np.array([0.3, 0.1, 0.949])
    l = np.array([-0.2, 0.4, 0.894])
    v /= np.linalg.norm(v)
    l /= np.linalg.norm(l)
    params = {"albedo_r": 0.5, "albedo_g": 0.4, "albedo_b": 0.3,
              "roughness": 0.4, "specular": 0.5, "metallic": 0.0}
    f_ab = brdf_rgb(params, v, l)
    f_ba = brdf_rgb(params, l, v)
    if not np.allclose(f_ab, f_ba, atol=1e-10):
        failures.append("互易性失败")

    # 2. 非负
    if np.any(evaluate(params, v[None], l[None])["brdf"] < -1e-12):
        failures.append("BRDF 出现负值")

    # 3. 能量守恒上界：GGX 重要性采样积分方向反照率，单次散射必须 <= 1
    from .brdf import directional_albedo
    rng = np.random.default_rng(0)
    fixed_n = np.array([0.0, 0.0, 1.0])
    for roughness in (0.05, 0.5, 1.0):
        p = dict(params, roughness=roughness, metallic=1.0,
                 albedo_r=0.99, albedo_g=0.99, albedo_b=0.99, specular=1.0)
        energy = directional_albedo(p, fixed_n, n_samples=30000, rng=rng)
        if np.any(energy > 1.02) or np.any(energy <= 0.0):
            failures.append(f"能量守恒上界失败 roughness={roughness}: {energy}")

    # 4. 粗糙度单调性：镜面峰高度随 roughness 增大而降低（真实反射曲线）
    from .render import reflection_curve
    peaks = []
    for r in (0.1, 0.5, 0.9):
        curve = reflection_curve(dict(params, roughness=r), theta_light_deg=45.0)
        peaks.append(float(curve["specular_luminance"].max()))
    if not (peaks[0] > peaks[1] > peaks[2]):
        failures.append("粗糙度-镜面峰值单调性异常")

    if failures:
        print("BRDF 自检失败:")
        for f in failures:
            print("  -", f)
        return 1
    print("BRDF 自检通过：互易性、非负、能量守恒上界、粗糙度单调性均满足。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflectance-fit",
        description="受控材质反射拟合 CLI（图形界面才是主入口："
                    "python -m reflectance_fit.gui）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="合成真值观测并拟合（含噪声/标定误差）")
    p_demo.add_argument("--ground-truth", help="JSON 覆盖真值参数")
    p_demo.add_argument("--poor", action="store_true", help="使用贫乏视角协议")
    p_demo.add_argument("--noise", type=float, default=0.005)
    p_demo.add_argument("--jitter", type=float, default=0.3,
                        help="角度随机扰动（度）")
    p_demo.add_argument("--bias", type=float, default=0.3,
                        help="光源系统偏置（度）")
    p_demo.add_argument("--holdout", type=float, default=0.25)
    p_demo.add_argument("--seed", type=int, default=1)
    p_demo.add_argument("--fixed", help="JSON 固定参数，如 '{\"metallic\":0}'")
    p_demo.add_argument("--report", help="导出报告图 PNG 路径")
    p_demo.add_argument("--json", dest="json_out", help="导出结果 JSON")
    p_demo.set_defaults(func=cmd_demo)

    p_dir = sub.add_parser("fit-dir", help="从图像目录（含 views.csv）拟合")
    p_dir.add_argument("directory")
    p_dir.add_argument("--holdout", type=float, default=0.25)
    p_dir.add_argument("--report")
    p_dir.add_argument("--json", dest="json_out", help="导出结果与执行证据 JSON")
    p_dir.set_defaults(func=cmd_fit_dir)

    p_check = sub.add_parser("check", help="BRDF 物理性质自检")
    p_check.set_defaults(func=cmd_check)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
