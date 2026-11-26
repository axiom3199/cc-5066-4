# 受控材质反射拟合器（Controlled Material Reflectance Fitter）

在**已知几何、相机与光照**的受控采集条件下，用多角度观测拟合一个**限定表面
模型**的材质参数（反照率颜色、粗糙度、镜面强度、金属度），并通过**实际重渲染
与观测比较**来检验恢复质量。

本项目刻意避免两类常见误用：

* 不会从一张图的平均颜色"猜"一套随意的 PBR 数值；
* 不会因为单张预览看起来相似就宣称材质被唯一恢复——当多个参数组合同样能
  解释观测时，会明确输出**可行参数范围**并说明歧义来源。

---

## 1. 图形入口（主要使用方式）

反射拟合的主入口是**图形界面（纹理制作工作台）**：

```bash
# 先安装 GUI 依赖
pip install PySide6

# 或安装 v0.2.0 验证过的完整锁定环境（含 GUI 与 pytest）
python -m pip install -r requirements-lock.txt

# 启动反射拟合图形界面
python -m reflectance_fit.gui

# 安装本包后也可使用同一图形入口
reflectance-gui
```

> 无显示器/无 PySide6 的环境无法启动窗口，此时可用命令行 `demo --report`
> 生成完全相同的报告图（见第 4 节），但日常使用请走图形入口。

### 界面工作流

1. **观测区**：点击「合成多角度观测」用自定义真值生成带噪声/标定误差的数据，
   或「载入图像目录」读取真实受控采集（目录需含 `views.csv` 几何清单）。
2. **材质区域（ROI）**：在任一视图图像上拖拽框选材质区域，或一键对全部视图
   使用中心区域。ROI 内像素做截尾均值聚合为该角度的一条 BRDF 观测。
3. **拟合设置**：勾选参与优化的参数、为已知参数填写固定值、设置留出比例。
4. **执行反射拟合**：结果区同时展示
   * 目标函数（训练/留出 RMSE）与逐观测逐通道**残差**；
   * **反射曲线**（漫反射/镜面/总 BRDF 分量，真实 BRDF 求值）；
   * **材质球预览**（逐法线重渲染，不是平铺纯色）；
   * 恢复参数表、固定范围与（可辨识性不足时的）**可行范围**。
5. 任何异常（缺 ROI、观测不足、参数越界、数值失败）都在状态栏与对话框提示，
   不会静默产生不可信结果。

### 图像目录清单格式 `views.csv`

```csv
name,file,theta_view,phi_view,theta_light,phi_light
view00,view_00.png,0,0,15,0
view01,view_01.png,0,0,30,0
...
```

角度单位为度。几何必须显式提供——几何未知时拒绝拟合。

---

## 2. 反射模型与固定约定

### 限定表面模型（仅这些参数受支持）

| 参数 | 含义 | 固定范围 |
|------|------|----------|
| `albedo_r/g/b` | 线性 sRGB 漫反射反照率 | `[0.005, 0.99]` |
| `roughness` | GGX 粗糙度（`alpha = roughness²`） | `[0.02, 1.0]` |
| `specular` | 介电质镜面强度（`F0 = 0.08·specular`） | `[0.0, 1.0]` |
| `metallic` | 金属度（金属 F0=反照率、关闭漫反射） | `[0.0, 1.0]` |

反射项为 **GGX/Trowbridge-Reitz 法线分布 D + Smith-GGX 遮蔽 G + Schlick
菲涅耳 F** 的 Cook-Torrance 微表面模型，漫反射为 Fresnel 加权 Lambert。
模型满足互易性、非负，以及方向反照率不超过 1 的能量守恒上界。

### 颜色空间

全部 BRDF 运算在**线性 sRGB** 中进行；仅在显示/保存时编码回 sRGB
（`srgb_to_linear` / `linear_to_srgb`）。

### 尺度歧义如何被消除

曝光、光强、材质能量三者同时自由缩放会产生完全相同的图像（尺度歧义）。
本项目**固定**其中两个自由度，只让材质参数变化：

* 光照辐照度固定 `LIGHT_IRRADIANCE = 1.0`；
* 相机曝光固定 `EXPOSURE = 1.0`，不做增益拟合；
* 因此观测噪声模型也**不包含逐观测乘性增益**（那等价于偷偷调曝光）。

---

## 3. 拟合方法：留出角度选解与可行范围

1. **目标函数**：训练视角上的加权 RMSE，比较的是完整 BRDF **重渲染辐射度**
   与观测（线性空间），不是平均颜色。
2. **优化/验证视角严格分离**：多起点 L-BFGS-B 优化只看训练集；候选解按
   **留出视角误差**选择。粗糙度 `alpha` 与镜面强度 `specular` 在近轴方向
   互相补偿、训练误差几乎相同，但在掠射角表现不同——留出集专门打破这种歧义，
   避免"只记住某一张高光图"。
3. **可辨识性判定**：侧视/掠射观测占比低于阈值时，不宣称唯一恢复，而是对每个
   参数做轮廓扫描（固定该参数、其余重优化），输出留出容差内的**可行参数范围**。
4. **多起点**：覆盖粗糙度×镜面强度网格与金属/介电起点，避免局部最优。

诚实性说明：即便视角充分，粗糙度与镜面强度在有限噪声下仍可能有一定补偿
（见下方稳定性测试）；报告中的候选解列表与可行范围就是为了让这种不确定性
可见，而不是用一个小数点后四位的单点制造"精确恢复"的假象。

### 3.1 优化后端迁移版本（v0.2.0）

反射拟合的新路径版本契约为
`reflection-optimizer-backend-migration/v2.0`，适配层位于
`reflectance_fit/optimization.py`：

* 默认后端 ID：`scipy-lbfgsb-logit-v2`。物理范围在适配层显式转换为无界
  logit 变量；若后端使用物理空间梯度，则按
  `dL/dz = dL/dp · dp/dz` 转换梯度尺度。需要显式边界的求解器可直接使用
  `physical_bounds()`。
* 旧路径保留为 `scipy-lbfgsb-logit-v1`，可通过
  `fit_brdf(..., backend=LEGACY_BACKEND_ID)` 或 `fit_brdf_legacy(...)`
  运行，用于旧样例和新旧契约对照。
* `FitResult.execution_evidence` 保存后端 ID、停止原因、迭代/求值次数、
  边界检查和求解后独立重渲染 RMSE。优化器的 `success` 只作为证据之一，
  不会被当成材质可接受的充分条件。
* 尺度化 Jacobian 的 SVD 诊断位于 `diagnostics.analyze_flat_valley()`；
  平坦最小奇异方向对应粗糙度↔镜面等耦合参数谷，仍按原逻辑输出可行范围。
* 旧材质 JSON 工程由 `project.load_project()` 兼容装载；材质资产编辑器
  `asset_editor.MaterialAssetEditor` 会把未确定的粗糙度范围交给
  `lighting.render_lighting_preview()`，并用
  `select_next_observation()` 选择重渲染误差最大的照明角度。
* 当新旧后端都能解释训练样片时，`migration.compare_optimizer_backends()`
  仍要求留出光位的实测/重渲染高光尺寸证据；缺少图像时返回
  `rejected_missing_actual_highlight_evidence`，不会伪报迁移成功。

GUI 拟合期间会显示“拟合状态：运行中”；拟合完成后会实际校验反射曲线和
材质球预览，并在图上用红色 `[异常: ...]` 与状态标签标出空曲线、非有限值、
负 BRDF、全黑预览或平铺纯色等问题。

---

## 4. 命令行（仅辅助）

CLI 用于脚本化演示、无头测试与报告导出，**不是主入口**。

```bash
# BRDF 物理性质自检（互易性、非负、能量守恒上界、粗糙度单调性）
python -m reflectance_fit.cli check

# 用自定义真值合成多角度观测 + 噪声/标定误差，拟合并导出报告
python -m reflectance_fit.cli demo --report report.png --json result.json

# 视角贫乏协议：演示"可行参数范围"输出分支
python -m reflectance_fit.cli demo --poor --report poor.png

# 从含 views.csv 的图像目录拟合
python -m reflectance_fit.cli fit-dir /path/to/capture \
  --report out.png --json fit-result.json
```

`demo` 可用参数：`--ground-truth '{"roughness":0.2}'`、`--noise`、
`--jitter`、`--bias`、`--holdout`、`--fixed '{"metallic":0}'`、`--seed`。

---

## 5. 稳定性验证（合成真值 + 噪声 + 标定误差）

测试用自定义真值经真实 BRDF 渲染，再注入：

* 零均值散粒噪声（`sigma·sqrt(radiance)`）与暗噪声；
* 每视角独立的小角度随机扰动；
* 所有观测共享的光源方向系统偏置。

结论（`tests/` 中可复现）：

* **无噪声基线**：参数恢复到 ~1e-4 量级；
* **含噪声/标定误差**：粗糙度与反照率的多种子中位估计落在真值邻域；镜面强度
  因粗糙度↔镜面补偿而方差更大，但其**中位数仍接近真值**，且全部候选都满足
  物理范围；
* **贫乏视角**：正确触发可行范围输出，镜面强度范围显著展开，明确提示非唯一。

运行测试：

```bash
pip install -r requirements-lock.txt  # 或至少 pip install pytest
python -m pytest -q
```

---

## 6. 模块结构

| 模块 | 职责 |
|------|------|
| `brdf.py` | GGX Cook-Torrance 真实反射求值、重要性采样、方向反照率 |
| `params.py` | 固定参数空间/范围、logit 有界变换、线性 sRGB 编解码、尺度约定 |
| `geometry.py` | 已知相机/光照/法线的观测数据结构与角度工具 |
| `observation.py` | 真值渲染、噪声/标定误差注入、分层留出划分、采集协议 |
| `optimization.py` | v2 优化后端契约、参数边界/logit/梯度尺度转换 |
| `fitter.py` | 新旧后端、多起点优化、留出选解、轮廓扫描可行范围、执行证据 |
| `evidence.py` | 边界检查、独立重渲染 RMSE、高光尺寸与迁移接受证据 |
| `diagnostics.py` | SVD 平坦参数谷、反射曲线与材质预览校验 |
| `lighting.py` | 粗糙度范围灯光预览、按最大误差光位推荐下一观测 |
| `project.py` / `asset_editor.py` | 旧材质工程兼容迁移、材质资产编辑器门面 |
| `migration.py` | 新旧优化后端对照与留出高光迁移验收 |
| `render.py` | 反射曲线与材质球重渲染预览 |
| `texture.py` | ROI 区域选取、截尾均值聚合、色块纹理 |
| `io_texture.py` | `views.csv` 图像目录载入与校验 |
| `gui.py` | 反射拟合图形界面与无头报告图（共用业务函数） |
| `cli.py` | 命令行辅助入口 |
