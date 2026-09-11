# 二维光声 FEM 前端（Python）

这是一个独立的物理正向模型，用来建立“二维组织 + 正则化激光点源 + 瞬态压力声波 + 有限孔径换能器接收”的最小闭环。它对应 `Photoacoustic_FEM_Method_CN.md` 的阶段 B，不实现论文中的随机 SLM/DMD 编码、编码声学孔径、压缩采样或图像重建。

## 已实现的物理链路

```text
Phi(x) [J/m²]
  -> H0 = eta_th * mu_a * Phi [J/m³ = Pa]
  -> p0 = Gamma * H0 [Pa]
  -> (1/K) p_tt - div((1/rho) grad(p)) = 0
  -> M p¨ + C_ABC p˙ + K p = 0 (Newmark beta=1/4, gamma=1/2)
  -> r^T p / A_sensor (有限线孔径平均)
  -> causal Butterworth band-pass (默认 1 MHz, 80% 工程假设)
  -> waveform.npz
```

所有进入求解器的长度均为 m；压力为 Pa；光能注量为 J/m²；吸收系数为 1/m。Gmsh 的 Physical IDs 全局唯一：`water=101`、`tissue=102`、`absorber=103`、`sensor=201`、`outer_absorbing=202`。组织和吸收体在几何上是 OCC fragment 后的相容曲面；吸收体只产生光学对比，首版声速和密度与组织相同。

光声源不是节点 Dirac 力，而是有限宽度高斯注量：

\[
\Phi=\Phi_{pk}\exp[-|x-x_s|^2/(2\sigma_s^2)],\qquad
p_0=\Gamma\eta_{th}\mu_a\Phi.
\]

`p0` 使用几何 L2 右端并进行正值守恒的 mass-lumped 投影，避免吸收体/组织的尖锐 `mu_a` 跳变在连续 P1 一致投影中造成负压 Gibbs overshoot；一致投影的负节点数仍写入报告。传播阶段没有再次施加激光时间载荷，因此不会重复注入能量。

## 安装

建议 Python 3.11+，当前已在 Python 3.11/3.12 上验证。依赖版本记录在 `requirements.txt`，核心链是 Gmsh 4.15.2、meshio 5.3.5、scikit-fem 12.0.2；因 Python 轮子兼容性，Python 3.11 固定 NumPy 2.4.6/SciPy 1.17.1，Python 3.12+ 固定 NumPy 2.5.3/SciPy 1.18.1。

```bash
git clone https://github.com/341151719/photoacoustic-fem-forward.git
cd photoacoustic-fem-forward
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

Gmsh 的 Python wheel 即使无图形运行也会动态依赖 `libGLU.so.1`。优先在 Linux 系统中安装 `libglu1-mesa`；代码也支持仓库或父目录 `.system-libs` 中的本地 fallback，以及 `PA_FEM_GLU_LIBRARY` 指定的兼容动态库。代码不会静默退回手写网格。

在当前工作区也可以使用：

```bash
source scripts/activate.sh
```

该脚本优先激活仓库内的 `.venv`（并兼容父目录 `.venv`），同时设置 `PYTHONPATH` 与本地 GLU fallback 的
`LD_LIBRARY_PATH`；不会修改原始压缩包或计划文档。

## 运行

最小真实 Gmsh/scikit-fem 冒烟：

```bash
source scripts/activate.sh
python -m pa_fem.cli self-test
```

调试基线（约 0.20 mm 全局尺寸，故报告会明确标记 `reference_resolution=false`）：

```bash
python -m pa_fem.cli solve --config configs/debug.json \
  --outdir results/debug_1MHz
```

推荐 1 MHz 参考网格（约 40 µm，运行时间和内存明显增加）：

```bash
python -m pa_fem.cli solve --config configs/reference.json \
  --outdir results/reference_1MHz
```

仅检查网格：

```bash
python -m pa_fem.cli mesh --config configs/debug.json --out /tmp/pa_domain.msh
python -m pa_fem.cli audit /tmp/pa_domain.msh --config configs/debug.json
```

`paper_linked_10MHz.json` 仅提供与论文检测频率建立联系的高频配置示例，不代表论文换能器参数，也不建议在普通工作站上直接运行；10 MHz 宽带 FEM 需要约 10 µm 级网格和 2 ns 时间步。

默认求解会做一次真实的 `2*p0` 第二次 Newmark 推进并比较整条接收波形，验证线性；若只需要单次运行，可加 `--no-linearity-check`。可加 `--no-plots` 关闭 PNG 输出。

## 输出

每个输出目录包含：

- `mesh.msh`：带 Physical Groups 的原始 Gmsh 4.1 网格；
- `snapshots.vtu`：`p0` 和若干时刻的节点压力，可用 ParaView 打开；
- `waveform.npz`：`time_s`、有限孔径 `s_raw_pa`、带宽后 `s_meas_pa`、可选 ADC 重采样及能量；
- `metadata.json`：SI 配置、版本、命令、网格 SHA256、矩阵/时间步/源/接收器信息；
- `validation.json`：Physical 标签、面积和质量、接口拓扑、矩阵对称性、源积分、到达时间诊断、ABC 能量和实际线性检验；
- `mesh.png`、`initial_pressure.png`、`waveform.png`：快速检查图（均标有单位）。

到达时间是高斯源有限空间宽度下的 `abs(raw)` 因果阈值诊断，不是严格波前；不使用会由有限记录末端向前泄漏的 Hilbert 全时域包络。二维压力模型对应纸面外无限延伸的线源/条形接收器，不能直接当三维实验幅值或二维/三维反平方扩散结果。

## 测试与验收

```bash
source scripts/activate.sh
pytest -q
```

测试覆盖配置约束、Gmsh Physical 标签、重复坐标与材料界面共享拓扑、有限孔径长度、`M/K/C` 对称性、正值守恒初压、ABC 能量不增长、实际二倍源线性和因果带通输出。debug 配置故意采用较粗网格/时间步，因此只用于 API/I/O/物理链闭环；`validation.json` 中 `reference_resolution` 和 `time_resolution_ok` 会明确给出精度警告。推荐网格依据源带宽 `f_max,src = 3c/(2πσ)` 与探头带宽上沿共同计算。

## 综合可视化验证

`results/visual_validation` 包含材料/网格/初压、瞬态波场、接收波形/频谱、
ABC 能量、网格质量、空间/时间收敛图、传播 GIF 和中文报告。收敛算例是控制变量计算：

- 80/40/30 µm 网格均使用 5 ns，检查空间细化；
- 40 µm 网格使用 10/5 ns，检查时间步减半。

可重新生成整套图件：

```bash
python tools/make_visual_validation.py \
  --reference results/reference_1MHz \
  --spatial-coarse results/convergence_spatial_80um \
  --spatial-fine results/convergence_spatial_30um \
  --time-coarse results/convergence_time_10ns \
  --debug results/debug_1MHz \
  --outdir results/visual_validation
```

原始宽带 FEM 波形保留了探头带外高频，因此其网格收敛慢于 1 MHz 带限测量。
当前 40 vs 30 µm 的 1 MHz 波形 L2 差为 1.35%，10 vs 5 ns 的差为 0.88%；
有效性结论因而限定于已建模的 1 MHz 接收带宽。

## 严格 Stage A/B 与 10 MHz 验收

除单算例内部自检外，仓库提供带独立参照的严格验收：

```bash
source scripts/activate.sh
python -m pa_fem.cli strict-validate --outdir results/strict_validation
```

该命令运行二维域内嵌的一维 d'Alembert 平面高斯解析解、封闭域能量守恒、
小域/大域 ABC 对照、解析平面界面反射系数、有限孔径解析积分，以及真正的
10 MHz 空间/时间细化。严格命令只在全部硬检查通过时返回 0；详细方法、阈值
和基准结果见 `docs/STRICT_VALIDATION_CN.md`。10 MHz 项是保持相应声学尺度的
缩小标准算例，用于验证数值参数，不冒充论文没有公开的实验几何或探头响应。

## 与参考文件的边界

原始建模计划已保存在 `docs/reference`。论文原始包与参考 Python FEM 包作为 GitHub Release 附件发布，避免膨胀 Git 历史；来源、许可说明与 SHA256 见 `THIRD_PARTY_NOTICES.md`。参考扬声器工程只用于代码组织与 Gmsh/meshio/scikit-fem 工程实践，其多物理方程没有移植。当前项目不包含压缩重建算法。
