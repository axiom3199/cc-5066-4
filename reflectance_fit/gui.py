"""反射拟合图形界面（纹理制作工作台）。

图形入口
========
    python -m reflectance_fit.gui
    或安装后：reflectance-gui

界面流程（与受控采集协议一一对应）
==================================
1. 「观测」区：载入/合成多角度图像，在任一视图上框选材质区域（ROI）；
2. 「拟合」区：选择激活参数、固定已知参数、设置留出比例后执行拟合；
3. 结果区同时展示：
   - 目标函数（训练/留出 RMSE）与逐观测残差；
   - 反射曲线（BRDF 总/漫反射/镜面分量，真实求值）；
   - 材质球预览（真实重渲染）与恢复参数、可行范围；
4. 异常（ROI 缺失、观测不足、数值失败等）统一在状态栏与对话框提示，
   不会静默产生一套"看起来差不多"的 PBR 数值。

依赖
====
GUI 需要 PySide6（``pip install PySide6``）。无显示环境下可用
:func:`build_report_figure` 在 Agg 后端生成同样的报告图，供测试与
命令行 ``--report`` 使用。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .brdf import shade_rgb
from .fitter import FitResult, fit_brdf
from .geometry import Observation
from .observation import (
    DEFAULT_PROTOCOL_DEG,
    POOR_PROTOCOL_DEG,
    perturb_observations,
    render_observations,
    split_holdout,
)
from .params import PARAM_SPEC, clip_physical
from .render import (
    preview_gain_for,
    reflection_curve,
    render_sphere,
    to_display_srgb,
)
from .texture import ViewImage, set_roi, views_to_observations


class FitAppError(RuntimeError):
    """界面层显式抛出的受控异常，统一在状态栏提示。"""


# --------------------------------------------------------------------------
# 无头可用的纯逻辑：合成数据集 + 拟合 + 报告数据
# --------------------------------------------------------------------------

def build_synthetic_dataset(
    ground_truth: Dict[str, float],
    *,
    poor: bool = False,
    noise_sigma: float = 0.005,
    angle_jitter_deg: float = 0.3,
    light_bias_deg: float = 0.3,
    seed: int = 1,
) -> List[Observation]:
    """供界面「合成数据」按钮与测试共用：真值渲染 + 噪声 + 标定误差。"""
    protocol = POOR_PROTOCOL_DEG if poor else DEFAULT_PROTOCOL_DEG
    clean = render_observations(ground_truth, protocol)
    rng = np.random.default_rng(seed)
    return perturb_observations(
        clean,
        noise_sigma=noise_sigma,
        angle_jitter_deg=angle_jitter_deg,
        light_bias_deg=light_bias_deg,
        rng=rng,
    )


def run_fit(
    observations: Sequence[Observation],
    *,
    holdout_fraction: float = 0.25,
    fixed: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> FitResult:
    """界面/CLI 共用的拟合入口，包含视角分层留出与输入校验。"""
    if len(observations) < 4:
        raise FitAppError(
            f"至少需要 4 个角度的观测（当前 {len(observations)} 个），"
            "否则粗糙度与镜面强度无法区分。"
        )
    train, hold = split_holdout(list(observations), holdout_fraction,
                                rng=np.random.default_rng(seed))
    return fit_brdf(train, hold, fixed=fixed)


def build_report_figure(gt_params: Optional[Dict[str, float]],
                        fit: FitResult,
                        observations: Sequence[Observation],
                        *,
                        figsize=(12, 9),
                        target_figure=None):
    """构造四块报告图。

    布局：左上目标函数/残差，右上反射曲线，左下材质球对比，右下参数表。
    传入 ``target_figure`` 时直接在该 Figure 上重绘（供 Qt 画布复用），
    否则新建 Figure（Agg 下可保存）。
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig = target_figure if target_figure is not None else plt.figure(figsize=figsize)
    fig.clear()
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.1, 1.0],
        hspace=0.32, wspace=0.22,
        left=0.07, right=0.97, top=0.90, bottom=0.08,
    )
    ax_res = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_sphere = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")

    # --- 残差 -------------------------------------------------------------
    predicted = shade_rgb(fit.parameters, observations)
    measured = np.stack([o.radiance for o in observations])
    residual = predicted - measured
    names = [o.name for o in observations]
    x = np.arange(len(names))
    colors = ("r", "g", "b")
    for c in range(3):
        ax_res.bar(x + (c - 1) * 0.25, residual[:, c], width=0.24,
                   label=f"channel {'RGB'[c]}", color=colors[c], alpha=0.8)
    ax_res.axhline(0.0, color="k", lw=0.8)
    ax_res.set_xticks(x)
    ax_res.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax_res.set_ylabel("residual (rerender - measured)")
    ax_res.set_title(
        f"Objective  train RMSE={fit.train_rmse:.4f}  "
        f"holdout RMSE={fit.holdout_rmse:.4f}\nstatus: {fit.status}"
    )
    ax_res.legend(fontsize=8)

    # --- 反射曲线 ---------------------------------------------------------
    curve = reflection_curve(fit.parameters)
    ax_curve.plot(curve["theta_deg"], curve["diffuse_luminance"],
                  "--", label="diffuse", color="tab:orange")
    ax_curve.plot(curve["theta_deg"], curve["specular_luminance"],
                  "--", label="specular", color="tab:blue")
    ax_curve.plot(curve["theta_deg"], curve["luminance"],
                  "-", lw=2, label="total BRDF", color="k")
    if gt_params is not None:
        gt_curve = reflection_curve(gt_params)
        ax_curve.plot(gt_curve["theta_deg"], gt_curve["luminance"],
                      ":", lw=2, color="tab:red", label="ground truth")
    ax_curve.set_xlabel("outgoing polar angle (deg)")
    ax_curve.set_ylabel("BRDF luminance (sr$^{-1}$)")
    ax_curve.set_title("Reflection curve (light fixed at 45 deg, true BRDF)")
    ax_curve.legend(fontsize=8)
    ax_curve.grid(alpha=0.3)

    # --- 材质球预览 -------------------------------------------------------
    sphere = render_sphere(fit.parameters, size=140)
    gain = preview_gain_for(sphere)
    ax_sphere.imshow(to_display_srgb(sphere, gain=gain))
    ax_sphere.set_title("Material preview (re-rendered)")
    ax_sphere.set_aspect("equal")
    ax_sphere.axis("off")

    # --- 参数/可行范围表 --------------------------------------------------
    rows = []
    for key in fit.active_params:
        value = fit.parameters[key]
        rng = fit.feasible_ranges.get(key, (value, value))
        bound = PARAM_SPEC[key]
        gt = gt_params.get(key) if gt_params else None
        rows.append([
            key,
            f"{value:.4f}",
            f"[{bound.low:g},{bound.high:g}]",
            f"[{rng[0]:.3f},{rng[1]:.3f}]",
            "-" if gt is None else f"{gt:.4f}",
        ])
    table = ax_table.table(
        cellText=rows,
        colLabels=["param", "fitted", "bound", "feasible range", "truth"],
        loc="center",
        cellLoc="center",
        colWidths=[0.2, 0.16, 0.16, 0.26, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)
    ax_table.set_title("Recovered parameters and feasible ranges", pad=14)

    fig.suptitle(
        "Controlled material reflectance fit report  |  "
        f"grazing fraction={fit.grazing_fraction:.2f}  "
        f"identifiable={fit.identifiable}",
        fontsize=12, y=0.98,
    )
    return fig


# --------------------------------------------------------------------------
# PySide6 交互界面
# --------------------------------------------------------------------------

def launch_gui() -> int:
    try:
        from PySide6 import QtCore, QtWidgets
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
    except Exception as exc:  # pragma: no cover - 环境相关
        raise FitAppError(
            "GUI 需要 PySide6 与 matplotlib Qt 支持，请执行 "
            "`pip install PySide6` 后重试。无显示环境可使用 "
            "`python -m reflectance_fit.cli demo --report report.png`。"
        ) from exc

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("受控材质反射拟合器 — 纹理制作工作台")
            self.resize(1280, 860)
            self.views: List[ViewImage] = []
            self.observations: List[Observation] = []
            self.ground_truth: Optional[Dict[str, float]] = None
            self.fit_result: Optional[FitResult] = None
            self._build_ui()

        # -- UI 构造 -------------------------------------------------------
        def _build_ui(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)

            # 左：控制面板
            left = QtWidgets.QVBoxLayout()
            root.addLayout(left, 0)

            box_data = QtWidgets.QGroupBox("1. 观测数据")
            form_data = QtWidgets.QGridLayout(box_data)
            self.btn_synth = QtWidgets.QPushButton("合成多角度观测（标准协议）")
            self.btn_poor = QtWidgets.QPushButton("合成贫乏视角观测（演示歧义）")
            self.btn_load = QtWidgets.QPushButton("载入图像目录…")
            form_data.addWidget(self.btn_synth, 0, 0, 1, 2)
            form_data.addWidget(self.btn_poor, 1, 0, 1, 2)
            form_data.addWidget(self.btn_load, 2, 0, 1, 2)
            left.addWidget(box_data)

            box_roi = QtWidgets.QGroupBox("2. 材质区域（在右图框选）")
            form_roi = QtWidgets.QVBoxLayout(box_roi)
            self.view_selector = QtWidgets.QComboBox()
            form_roi.addWidget(self.view_selector)
            self.lbl_roi = QtWidgets.QLabel("ROI：未选取")
            form_roi.addWidget(self.lbl_roi)
            btn_roi_all = QtWidgets.QPushButton("对全部视图使用中心区域")
            form_roi.addWidget(btn_roi_all)
            left.addWidget(box_roi)

            box_fit = QtWidgets.QGroupBox("3. 拟合设置")
            form_fit = QtWidgets.QGridLayout(box_fit)
            self.holdout_spin = QtWidgets.QDoubleSpinBox()
            self.holdout_spin.setRange(0.1, 0.5)
            self.holdout_spin.setSingleStep(0.05)
            self.holdout_spin.setValue(0.25)
            form_fit.addWidget(QtWidgets.QLabel("留出比例"), 0, 0)
            form_fit.addWidget(self.holdout_spin, 0, 1)
            self.param_checks: Dict[str, QtWidgets.QCheckBox] = {}
            self.fixed_edits: Dict[str, QtWidgets.QLineEdit] = {}
            row = 1
            for key, pr in PARAM_SPEC.items():
                chk = QtWidgets.QCheckBox(key)
                chk.setChecked(True)
                edit = QtWidgets.QLineEdit()
                edit.setPlaceholderText(f"固定值 [{pr.low:g},{pr.high:g}]")
                edit.setEnabled(False)
                chk.toggled.connect(edit.setDisabled)
                form_fit.addWidget(chk, row, 0)
                form_fit.addWidget(edit, row, 1)
                self.param_checks[key] = chk
                self.fixed_edits[key] = edit
                row += 1
            self.btn_fit = QtWidgets.QPushButton("执行反射拟合")
            self.btn_fit.setStyleSheet("font-weight:bold")
            form_fit.addWidget(self.btn_fit, row, 0, 1, 2)
            left.addWidget(box_fit)

            box_obj = QtWidgets.QGroupBox("目标函数与参数")
            v = QtWidgets.QVBoxLayout(box_obj)
            self.objective_text = QtWidgets.QPlainTextEdit()
            self.objective_text.setReadOnly(True)
            self.objective_text.setMaximumBlockCount(2000)
            v.addWidget(self.objective_text)
            left.addWidget(box_obj, 1)

            # 右：图像/报告区
            right = QtWidgets.QVBoxLayout()
            root.addLayout(right, 1)
            self.tabs = QtWidgets.QTabWidget()
            right.addWidget(self.tabs, 1)

            self.roi_figure = Figure(figsize=(5, 4))
            self.roi_canvas = FigureCanvasQTAgg(self.roi_figure)
            self.tabs.addTab(self.roi_canvas, "观测图像 / 区域选取")
            self.roi_canvas.mpl_connect("button_press_event", self._on_press)
            self.roi_canvas.mpl_connect("button_release_event", self._on_release)
            self._roi_start = None

            self.report_figure = Figure(figsize=(8, 8))
            self.report_canvas = FigureCanvasQTAgg(self.report_figure)
            self.tabs.addTab(self.report_canvas, "拟合结果 / 反射曲线 / 预览")

            self.status = self.statusBar()
            self.status.showMessage("就绪：先合成或载入多角度观测。")

            self.btn_synth.clicked.connect(self._on_synth)
            self.btn_poor.clicked.connect(self._on_synth_poor)
            self.btn_load.clicked.connect(self._on_load)
            self.btn_fit.clicked.connect(self._on_fit)
            btn_roi_all.clicked.connect(self._on_roi_all)
            self.view_selector.currentIndexChanged.connect(self._draw_roi_view)

        # -- 数据载入 ------------------------------------------------------
        def _on_synth(self) -> None:
            self._load_dataset(poor=False)

        def _on_synth_poor(self) -> None:
            self._load_dataset(poor=True)

        def _load_dataset(self, poor: bool) -> None:
            try:
                self.ground_truth = clip_physical({
                    "albedo_r": 0.62, "albedo_g": 0.31, "albedo_b": 0.18,
                    "roughness": 0.32, "specular": 0.6, "metallic": 0.0,
                })
                self.observations = build_synthetic_dataset(
                    self.ground_truth, poor=poor
                )
                from .texture import synthetic_view_image
                self.views = []
                protocol = POOR_PROTOCOL_DEG if poor else DEFAULT_PROTOCOL_DEG
                for obs, angles in zip(self.observations, protocol):
                    tv, pv, tl, pl = angles
                    img = synthetic_view_image(obs.radiance)
                    self.views.append(ViewImage(
                        img, tv, pv, tl, pl, name=obs.name
                    ))
                self.view_selector.clear()
                self.view_selector.addItems([v.name for v in self.views])
                self._on_roi_all()
                self.status.showMessage(
                    f"已合成 {len(self.views)} 个角度观测"
                    + ("（贫乏视角，预期触发可行范围输出）" if poor else "")
                )
            except Exception as exc:
                self._show_error(exc)

        def _on_load(self) -> None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "选择多视角图像目录"
            )
            if not directory:
                return
            try:
                from .io_texture import load_view_directory
                self.views = load_view_directory(directory)
                self.observations = []
                self.ground_truth = None
                self.view_selector.clear()
                self.view_selector.addItems([v.name for v in self.views])
                self.status.showMessage(
                    f"载入 {len(self.views)} 张图像，请框选材质区域。"
                )
                self._draw_roi_view()
            except Exception as exc:
                self._show_error(exc)

        # -- ROI 交互 ------------------------------------------------------
        def _current_view(self) -> Optional[ViewImage]:
            i = self.view_selector.currentIndex()
            return self.views[i] if 0 <= i < len(self.views) else None

        def _on_press(self, event) -> None:
            if event.inaxes is not None and event.xdata is not None:
                self._roi_start = (event.xdata, event.ydata)

        def _on_release(self, event) -> None:
            if self._roi_start is None or event.inaxes is None:
                return
            x0, y0 = self._roi_start
            x1, y1 = event.xdata, event.ydata
            self._roi_start = None
            view = self._current_view()
            if view is None:
                return
            try:
                set_roi(view, int(round(x0)), int(round(y0)),
                        int(round(x1)), int(round(y1)))
                self.lbl_roi.setText(f"ROI：{view.roi}")
                self._draw_roi_view()
            except Exception as exc:
                self._show_error(exc)

        def _on_roi_all(self) -> None:
            try:
                for view in self.views:
                    h, w = view.image_srgb.shape[:2]
                    set_roi(view, w // 4, h // 4, 3 * w // 4, 3 * h // 4)
                if self.views:
                    self.lbl_roi.setText(f"ROI：{self._current_view().roi}")
                    self._draw_roi_view()
            except Exception as exc:
                self._show_error(exc)

        def _draw_roi_view(self) -> None:
            view = self._current_view()
            self.roi_figure.clear()
            ax = self.roi_figure.add_subplot(111)
            if view is None:
                ax.text(0.5, 0.5, "无观测图像", ha="center")
            else:
                ax.imshow(view.image_srgb)
                if view.roi is not None:
                    from matplotlib.patches import Rectangle
                    x0, y0, x1, y1 = view.roi
                    ax.add_patch(Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        fill=False, edgecolor="lime", lw=2,
                    ))
                ax.set_title(
                    f"{view.name}  视角 θ={view.theta_view_deg}° "
                    f"光照 θ={view.theta_light_deg}°（拖拽框选区域）"
                )
                ax.axis("off")
            self.roi_canvas.draw_idle()

        # -- 拟合 ----------------------------------------------------------
        def _collect_fixed(self) -> Dict[str, float]:
            fixed: Dict[str, float] = {}
            for key, chk in self.param_checks.items():
                if not chk.isChecked():
                    text = self.fixed_edits[key].text().strip()
                    if not text:
                        raise FitAppError(f"参数 {key} 被固定但未填写固定值")
                    value = float(text)
                    pr = PARAM_SPEC[key]
                    if not pr.low <= value <= pr.high:
                        raise FitAppError(
                            f"{key}={value} 超出固定范围 [{pr.low},{pr.high}]"
                        )
                    fixed[key] = value
            return fixed

        def _on_fit(self) -> None:
            try:
                if not self.views:
                    raise FitAppError("请先合成或载入观测图像。")
                if any(v.roi is None for v in self.views):
                    raise FitAppError("存在未选取 ROI 的视图，请先框选材质区域。")
                self.observations = views_to_observations(self.views)
                fixed = self._collect_fixed()
                result = run_fit(
                    self.observations,
                    holdout_fraction=self.holdout_spin.value(),
                    fixed=fixed,
                )
                self.fit_result = result
                self._present_result(result)
            except Exception as exc:
                self._show_error(exc)

        def _present_result(self, result: FitResult) -> None:
            self.objective_text.clear()
            self.objective_text.appendPlainText(result.summary())
            self.objective_text.appendPlainText("")
            self.objective_text.appendPlainText(
                f"目标函数评估次数（含多起点）: {len(result.objective_trace)}"
            )
            self.objective_text.appendPlainText(
                "候选解（按留出 RMSE 排序，前 5）:"
            )
            for c in result.candidates[:5]:
                self.objective_text.appendPlainText(
                    f"  train={c.train_rmse:.5f} holdout={c.holdout_rmse:.5f} "
                    f"rough={c.params['roughness']:.3f} "
                    f"spec={c.params['specular']:.3f} "
                    f"metal={c.params['metallic']:.3f}"
                )
            self.report_figure.clear()
            build_report_figure(
                self.ground_truth, result, self.observations,
                target_figure=self.report_figure,
            )
            self.report_canvas.draw_idle()
            self.tabs.setCurrentIndex(1)
            if result.identifiable:
                self.status.showMessage("拟合完成：侧视充分，已按留出角度选解。")
            else:
                self.status.showMessage(
                    "拟合完成但可辨识性不足：请查看参数可行范围，避免误读为唯一解。"
                )

        # -- 工具 ----------------------------------------------------------
        def _show_error(self, exc: Exception) -> None:
            self.status.showMessage(f"异常：{exc}")
            QtWidgets.QMessageBox.warning(self, "反射拟合异常", str(exc))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(launch_gui())
