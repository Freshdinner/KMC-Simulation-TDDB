# KMC 漏电流仿真

基于动力学蒙特卡洛（KMC，Gillespie direct method）的 MIM（金属-绝缘体-金属）
电容器漏电流仿真。电极/介质材料、器件尺寸、缺陷分布均可配置，不绑定单一材料
体系；覆盖全部输运机制（内部编号 R1–R8），输出 I-V / I-T 曲线以及按机制、按
缺陷位置的电流分解。

物理规格见 `docs/physics_model.md` 与 `docs/rate_equations.md`——那两份文档是
本项目的规格来源，本 README 只讲怎么用和实现上做了哪些取舍。

---

## 环境

- Python 3.8
- numpy ≤ 1.24.x、scipy ≤ 1.10.x、pytest ≤ 8.3.x（1.25 / 1.11 / 8.4 起不再支持
  3.8，`pyproject.toml` 里锁了上界，不要让依赖自动升级）
- 其余：matplotlib、typer、PyYAML；pytest 作为基础依赖一并安装

```bash
pip install -e .          # 在 Python 3.8 环境下，含 pytest
```

> **在更新的解释器上开发时**：`pyproject.toml` 的版本上界是按 3.8 锁的，如果本机
> 是 Python 3.10+ / numpy 2.x，`pip install -e .` 会试图把 numpy 降级到没有 wheel
> 的版本。这种情况下不要安装，直接用源码路径：
> ```bash
> pytest tests/ -v                              # pyproject 里配了 pythonpath = ["src"]
> PYTHONPATH=src python -m kmc_leakage run ...  # 命令行
> ```
> 代码本身对 numpy 1.24 与 2.x 都兼容（例如一律用 `scipy.integrate.trapezoid`，
> 不用在两个版本间变动的 `np.trapz` / `np.trapezoid`）。

---

## 快速开始

```bash
# 单点仿真
python -m kmc_leakage run --config examples/single_run.yaml

# I-V / I-T 扫描，结果写 CSV
python -m kmc_leakage sweep --config examples/iv_sweep.yaml --out results.csv

# 扫描顺手出曲线图：<前缀>_iv.png（横轴电压，一条曲线一个温度）
#                  <前缀>_it.png（横轴温度，一条曲线一个电压）
# 两张图上都只画总电流；另落 <前缀>_defects.png 与 <前缀>_defects_clusters.png：
# 系综里随机一个种子的缺陷分布及其强耦合簇标注（单层、叠层都适用，所用 seed 打印在终端）
python -m kmc_leakage sweep --config examples/iv_sweep.yaml --out results.csv --plot-prefix scan

# 出图：缺陷分布 / 能带结构。defects 同时落 <out 去扩展名>_clusters.png，标出强耦合簇
# （ADR-022，按配置里 simulation.voltage / temperature 这一工作点识别）
python -m kmc_leakage plot --config examples/single_run.yaml --kind defects --out defects.png
python -m kmc_leakage plot --config examples/single_run.yaml --kind bands   --out bands.png

# ZrO₂/Al₂O₃/ZrO₂ 叠层的 t₂ 结构对照：固定总厚 8 nm，t₂ ∈ {0,2,4} nm 各一份
# 配置，逐份跑同一条 sweep 命令再比对三个 CSV（t₂=0 是单层退化点）
python -m kmc_leakage sweep --config examples/stack_t2_2nm.yaml --out t2_2nm.csv
python -m kmc_leakage plot  --config examples/stack_t2_2nm.yaml --kind bands --out t2_2nm_bands.png

# TDDB：ZrO₂ 单层、50×50×8 nm 盒子（规格见 PROJECT_CONTEXT_TDDB.md）。run / sweep 给初始
# 缺陷集合下的稳态本底；tddb 跑一条恒压应力下的 I-t 轨迹（热反馈 + 贯穿通路击穿判据，一个缺陷实现，
# --seed 换实现），落轨迹 CSV 与 I-t（双对数）/ 温度演化图
python -m kmc_leakage run   --config examples/zro2_tddb.yaml
python -m kmc_leakage sweep --config examples/zro2_tddb.yaml --out tddb_background.csv
python -m kmc_leakage tddb  --config examples/zro2_tddb.yaml --seed 1 --out it.csv --plot it.png
```

`run` 的输出长这样（示例配置带 `ensemble` 段，逐个实现跑完再取算术平均）：

```
U = 2.00 V, T = 25.0 °C   系综：5 个缺陷实现
  [ 1/5 ] seed 12345    J = 1.7051e-08 A/cm²
  [ 2/5 ] seed 23456    J = 2.1561e-08 A/cm²
  [ 3/5 ] seed 34567    J = 9.9426e-09 A/cm²
  [ 4/5 ] seed 45678    J = 1.5389e-08 A/cm²
  [ 5/5 ] seed 56789    J = 1.8833e-08 A/cm²
系综平均（M = 5 个缺陷实现）：J = 1.6555e-08 A/cm²   标准误 ±1.944e-09（相对 11.7%）
  各实现散布：min 9.9426e-09   max 2.1561e-08 A/cm²   σ(log₁₀J) = 0.128
  按机制分解（系综平均）：
    直接/FN隧穿      J =   1.5557e-18 A/cm²   占比   0.000%   标准误 ±0.000e+00
    热电子发射        J =   4.1553e-23 A/cm²   占比   0.000%   标准误 ±0.000e+00
    TAT          J =   3.0153e-10 A/cm²   占比   1.821%   标准误 ±3.630e-11
    PF发射         J =   1.6254e-08 A/cm²   占比  98.179%   标准误 ±1.908e-09
```

去掉 `ensemble` 段就退回单实现的详细报告——多出按缺陷位置的分解与三个估计量的
交叉检验，那些量是单次实现才有的。

**耗时参考**（在改成 8 nm 默认之前的 50×50×9 nm 器件上实测：3×10¹⁸ cm⁻³ → 68 个缺陷，
`grid` = 32/32/36；现默认 50×50×8 nm 为 60 个缺陷，未重测）：
每个 KMC 事件约 1.9 ms，外加每个实现 0.2 s 的静电预计算（N+1 次泊松求解）。

| 跑什么 | 单个实现 | 5 个实现 |
|---|---|---|
| 单点 20000 事件 | 38 s | **3.1 分钟** |
| `examples/iv_sweep.yaml`（15 个工作点 × 5000 事件） | 2.3 分钟 | 12 分钟 |
| 默认全范围（16 电压 × 6 温度 = 96 点 × 5000 事件） | 15 分钟 | 1.3 小时 |

机时对 M 是严格线性的。浓度是比尺寸更敏感的成本变量：同样 50×50×9 nm，
3×10¹⁹ cm⁻³ 下 N=675，单个事件涨到 19 ms（电极↔陷阱那部分的 WKB 积分是
O(N)，占大头），单点 20000 事件约 6.3 分钟。

### 作为库使用

```python
import dataclasses
import numpy as np
from kmc_leakage.io import load_config
from kmc_leakage.kmc import run_single

config = load_config("examples/single_run.yaml")
config = dataclasses.replace(
    config, simulation=dataclasses.replace(config.simulation, voltage=3.0)
)

result = run_single(config)
print(result.current_density)                 # A/m²（内部一律 SI）
print(result.by_mechanism)                    # 四项之和 == result.current
print(result.by_defect)                       # 每个缺陷的贡献，之和 + R1 == current
top = np.argsort(result.by_defect)[::-1][:3]  # 贡献最大的三个缺陷
print(result.defects.positions[top, 2])
```

`SimulationResult` 的字段：

| 字段 | 含义 |
|---|---|
| `current` / `current_density` / `area` | 总电流 [A] / 电流密度 [A/m²] / 电极面积 [m²] |
| `by_mechanism` | 四类机制的电流，四项之和恒等于 `current` |
| `by_defect` | 每个缺陷的电流贡献，之和 + R1 两项恒等于 `current` |
| `occupancy` | KMC 采样的时间平均占据概率 |
| `occupancy_mean_field` | 条件稳态占据 p̂ 的时间平均，与上一项对照即可判断估计量前提是否成立 |
| `current_direct` / `current_counts_total` / `n_positive_events` | 交叉检验用的另外两个估计量（见下文）；`current_counts_total` = 正极净事件电流 `current_counts` + 快陷阱电流 `current_fast` + R1 |
| `n_fast_traps` / `current_fast` / `fast_exchange` | 快陷阱个数、它们的正极净电流 [A]、快→慢跃迁的净次数（未启用 `fast_trap_rate_s` 时全为 0） |
| `elapsed_time` | 统计窗口时长 [s] |

扫描：

```python
from kmc_leakage.sweep import run_sweep, write_csv

sweep = run_sweep(config)          # 复用同一套缺陷实现与静电预计算
write_csv(sweep, "results.csv")    # 输出边界换算成 A/cm² 与 °C
```

系综平均（M 个缺陷实现取算术平均，见下文"系综平均 `ensemble`"）：

```python
from kmc_leakage.ensemble import run_ensemble, run_ensemble_sweep, write_ensemble_csv

point = run_ensemble(config)                    # 逐个种子跑单点
print(point.current_density, point.current_density_sem)  # 平均值与标准误 [A/m²]
print(point.samples)                            # 平均前每个实现的原始值
print(point.results[0].by_defect)               # 单个实现的缺陷分解仍在手边

curves = run_ensemble_sweep(config)             # 逐个种子跑完整扫描
write_ensemble_csv(curves, "ensemble.csv")      # 平均 + 标准误 + 各实现原始值
```

`EnsembleResult` / `EnsembleSweepResult` 的字段（前者是标量，后者是 (温度点, 电压点)
矩阵，字段名相同）：

| 字段 | 含义 |
|---|---|
| `current_density` | M 个实现的算术平均 [A/m²] |
| `current_density_sem` | 平均值的标准误 σ/√M [A/m²]，M=1 时为 0 |
| `by_mechanism` / `by_mechanism_sem` | 四类机制的平均电流密度与标准误，四项之和恒等于 `current_density` |
| `samples` | 平均前的原始电流密度，形状 `(M,)` / `(M, 温度点, 电压点)` |
| `results` / `sweeps` | 每个实现的完整结果，顺序与 `seeds` 一致 |
| `seeds` / `n_realizations` | 种子列表与实现个数 M |

---

## 配置文件

完整示例见 `examples/single_run.yaml`。YAML 用器件物理惯用单位（eV / nm / cm⁻³ /
cm⁻² / °C），**换算到 SI 只发生在 `io.py` 的读入边界**，换算系数一律取自
`scipy.constants`；下游模块只见 SI。

介质结构采用"有序介质层 + 材料库 + 显式界面"表示（ADR-003/015）：顶层
`materials`（按名字典）给材料物性，`layers`（有序层）各按名引用材料、给厚度、配
该层体缺陷，可选的 `interfaces`（数量 = 层数-1）携带带阶 / 面电荷 / 界面缺陷。
**`Lz` 由各层 `thickness_nm` 求和，不再是配置项**；单层退化就是 `layers` 只含一个
元素。旧 schema 的顶层 `dielectric` / `defects` 与 `device.Lz_nm` 会直接报错并提示
新写法（不做运行时兼容层）。

```yaml
materials:                     # 材料库，按名字典；layers 按名引用
  reference-preset:
    kappa: 38.0                # 静态介电常数——只进泊松方程
    eps_opt: 5.6               # 光学介电常数——只进 PF 发射 (R8)
    band_gap_eV: 5.4
    electron_affinity_eV: 2.86
    m_cb_me: 1.2               # 裸带边质量，Weinberg 关系的输入
    m_vb_me: 2.5
    m_di_me: 1.2               # 隧穿质量，速率公式实际使用（现数值上 = m_cb）
    phonon_energy_eV: 0.04
    huang_rhys_S: 1.0
    attempt_frequency_s: 1.0e13  # ν0，R6/R7/R8 共用同一项

layers:                        # 有序介质层；Lz = 各层 thickness_nm 之和
  - material: reference-preset  # 引用 materials 里的材料名
    thickness_nm: 8.0
    defects:                   # 该层体缺陷（单层退化：只有这一层）
      concentration_cm3: 3.0e18
      charge_state_Z: 1.0
      min_separation_nm: 0.5
      level:   {mode: fixed,   ED_eV: 1.15, sigma_eV: 0.15}   # fixed | gaussian
      spatial: {mode: uniform, enrichment: 5.0, decay_length_nm: 1.0}  # uniform | interface

# interfaces:                  # 可选，数量 = 层数-1；单层不配。多层时携带：
#   - delta_Ec_eV: 1.46        # 导带带阶 ΔE_C 覆盖（省略则电子亲和能对齐）
#     sigma_C_per_cm2: 0.0     # 固定界面面电荷
#     defects: {surface_density_cm2: 1.0e12, level: {mode: fixed, ED_eV: 1.15}}

device:
  Lx_nm: 50.0
  Ly_nm: 50.0
  barrier_height_eV: 1.74      # 可选；省略则由 Φ_m - χ 推算
  # Lz 不再是配置项——由各层 thickness_nm 求和得到

electrodes:
  negative: &electrode
    name: TiN
    work_function_eV: 4.6
    effective_mass_me: 1.0
    fermi_energy_eV: 5.0       # 费米能相对该电极导带底的高度
  positive: *electrode

simulation:
  voltage_V: 2.0
  temperature_C: 25.0
  n_events: 20000
  burn_in_fraction: 0.2
  seed: 12345                 # 只在没有 ensemble 段时生效
  grid: {nx: 32, ny: 32, nz: 80}   # hx = Lx/nx，格点数必须跟着器件尺寸走
  # fast_trap_rate_s: 1.0e9   # 可选：与电极交换电子 ≥ 此速率 [s⁻¹] 的陷阱移出事件表（见"实现要点"）

sweep:                          # 整段省略则用默认范围
  voltage_V:     {start: 1.0, stop: 4.0, step: 0.2}    # 默认 I-V：16 点
  temperature_C: {start: 0.0, stop: 125.0, step: 25.0} # 默认 I-T：6 点

ensemble:                       # 整段省略则退化为 simulation.seed 一个实现
  count: 5                       # 实现个数 M，每次运行随机抽取 count 个互异种子
```

### 叠层结构与不同层厚的设置

介质由 `layers` 有序列表描述，每层 `material` 引用材料库、`thickness_nm` 给层厚，
**总厚 Lz = 各层 thickness_nm 之和**（不再有独立的 `device.Lz_nm`）。要换叠层结构，
改的就是层数、每层引用哪种材料、以及各层的 `thickness_nm`。

`examples/stack_t2_{0,2,4}nm.yaml` 是一组现成的 ZrO₂/Al₂O₃/ZrO₂ 结构对照：**固定
总厚 8 nm，中间 Al₂O₃ 层厚 t₂ ∈ {0,2,4} nm，两侧 ZrO₂ 对称补偿 t₁=t₃=(8−t₂)/2**。
三份配置只有层结构不同，材料、缺陷、器件、扫描范围、系综全部相同，所以比对三份结果
就是在看 t₂ 这一个变量的影响。要自己设别的层厚，照这组改 `thickness_nm` 即可：

```yaml
layers:
  - material: ZrO2      # 负极侧 ZrO₂，t₁ = (8−t₂)/2
    thickness_nm: 3.0
    defects: {...}      # 体缺陷挂在层上
  - material: Al2O3     # 阻挡层：不配 defects 段即体相无缺陷
    thickness_nm: 2.0
  - material: ZrO2      # 正极侧 ZrO₂，t₃ = t₁
    thickness_nm: 3.0
    defects: {...}

interfaces:             # 数量 = 层数−1（三层 → 两个界面），按 z 从小到大
  - delta_Ec_eV: 1.46   # 省略则按电子亲和能对齐自动推算带阶
    sigma_C_per_cm2: 0.0
    defects: {surface_density_cm2: 1.0e12, ...}  # 界面缺陷按面密度 Nₛ 配
  - {...}               # 第二个界面
```

改层厚时有两条硬约束，违反会在构造泊松求解器时直接报错：

- **每层 `thickness_nm` 必须是 hz = Lz/nz 的整数倍**（变 κ 泊松要求界面落在网格节点上）。
  默认 nz=80、Lz=8 nm，hz = 0.1 nm，所以 4/0/4、3/2/3、2/4/2 这几组都对齐。
  选层厚前先确认能被 hz 整除
- **改总厚 Lz 时 `grid.nz` 要跟着改**——`hz = Lz/nz` 由厚度和格点个数推出，固定 nz 改
  总厚会改变 hz 和 CIC 自作用势（∝ 1/h），测到的就成了网格假象（见"几个约定"与
  `docs/physics_model.md`）

**单层退化**：`layers` 只写一个元素、不写 `interfaces` 段即可（`io` 拒绝 0 厚层，
所以 t₂=0 那份是写成单层 8 nm ZrO₂，而不是留一个 `thickness_nm: 0` 的层）。单层时
所有叠层专属逻辑（分段 WKB、界面台阶、界面缺陷）逐位退回单层口径。

设好每份配置后，下面三件事都指向同一个 `--config`，换配置就是换叠层：

**① I-V / I-T 扫描**——扫描范围取自配置的 `sweep` 段（示例是 2–4 V 步长 0.5、
25–125 °C 步长 50），有 `ensemble` 段就逐个种子扫完取算术平均：

```bash
# 单份：写 CSV，同时落 t2_2nm_iv.png（横轴电压）与 t2_2nm_it.png（横轴温度）
python -m kmc_leakage sweep --config examples/stack_t2_2nm.yaml --out t2_2nm.csv --plot-prefix t2_2nm

# 三份逐个跑同一条命令，再比对三个 CSV / 三组曲线看 t₂ 的影响（t2_0nm 是单层退化点）
python -m kmc_leakage sweep --config examples/stack_t2_0nm.yaml --out t2_0nm.csv --plot-prefix t2_0nm
python -m kmc_leakage sweep --config examples/stack_t2_4nm.yaml --out t2_4nm.csv --plot-prefix t2_4nm
```

曲线图上只画总电流（深浅分行），带 ±σ/√M 误差棒；机制分解看 CSV，标准误同时走
CSV 的 `J_sem_A_cm2` 一列。

**② 缺陷分布图**——三维散点（按能级着色）+ 沿 z 的数密度直方图；叠层时两幅都叠画
层边界与界面位置，界面缺陷单独标色（直方图柱子只统计体缺陷）：

```bash
python -m kmc_leakage plot --config examples/stack_t2_2nm.yaml --kind defects --out t2_2nm_defects.png
```

**③ 能带图**——沿 z 的能带剖面，界面处画真实台阶（高度 = ΔE_C，不连斜坡），缺陷
能级按 z 散点标出：

```bash
python -m kmc_leakage plot --config examples/stack_t2_2nm.yaml --kind bands --out t2_2nm_bands.png
```

`plot` 画的是配置抽到的第一个种子对应的那一次缺陷实现（有 `ensemble` 段时是随机抽的
头一个，否则用 `simulation.seed`），换种子即换一次实现。

> **叠层材料是占位值**：示例里 Al₂O₃ 的物性与界面缺陷面密度 Nₛ 都是待标定的占位值
> （ADR-004/008），结构对照能看趋势，得出定量材料结论前要先换成真实数据。

### 最容易混淆的参数

| 参数 | 用在哪 | 不能用在哪 |
|---|---|---|
| `kappa`（静态介电常数） | 泊松方程 `electrostatics.py` | 任何速率公式 |
| `eps_opt`（光学介电常数） | **只**在 PF 发射 R8 | 其他速率公式里出现即为实现错误 |
| `m_cb` / `m_vb`（裸带边质量） | Weinberg 关系的输入；`m_cb` 另用于有效理查森常数 A\* | R1–R7 的速率公式 |
| `m_di`（隧穿质量） | R1–R7 的全部速率公式与 WKB 积分 | —— |
| `m_e`（自由电子质量） | R1 前置因子（发射极电子质量） | —— |

`m_di` 当前数值上等于 `m_cb`，但在代码里是两个独立配置项——以后单独调整 `m_di`
（比如真正实现 Weinberg 色散关系）不需要改接口。

### 缺陷分布的配置

体缺陷现在挂在层上（`layers[i].defects`，单层就是唯一那层）。下面 `defects` 子块里的
能级分布 `level` 与空间分布 `spatial` 是两个互相独立的子块，各有两种模式，四种组合都
可用。两者都只决定"能级怎么散开""位置往哪儿堆"——**缺陷个数由 `concentration_cm3 ×
体积` 单独决定**（四舍五入取整），换分布不改变个数。

#### 能级分布 `defects.level`

| 键 | 含义 | 说明 |
|---|---|---|
| `mode` | `fixed`（单一能级）或 `gaussian`（正态分散） | 缺省 `fixed` |
| `ED_eV` | 能级深度 E_D，相对介质导带底 | 必填；`gaussian` 模式下是均值 |
| `sigma_eV` | 高斯标准差 σ_E | 只在 `gaussian` 模式生效，`fixed` 模式下忽略 |

```yaml
defects:
  level:
    mode: fixed          # 全部缺陷同一能级
    ED_eV: 1.15
```

```yaml
defects:
  level:
    mode: gaussian       # 能级按 N(1.15, 0.15²) eV 分散
    ED_eV: 1.15
    sigma_eV: 0.15
```

- `mode: gaussian` 必须给出正的 `sigma_eV`，否则 `io.py` 在读入时就报错
- 高斯采样对 E_D ≤ 0 重采样（截断高斯）：E_D 非正会让局域化半径
  r_D = ℏ/√(2·m_di·E_D) 发散。σ_E 逼近 E_D 时重采样 1000 轮仍失败会抛
  `RuntimeError`，把 `sigma_eV` 保持在 `ED_eV` 的三分之一以内就不会碰到
- **r_D 按每个缺陷自己的能级算**，`gaussian` 模式下 r_D 随之分散，直接进 R6/R7 的
  缺陷间跃迁速率；`fixed` 模式下全部缺陷共用同一个 r_D
- 能级同时进 R2–R5、R8 的指数项：室温下 σ_E = 0.15 eV 对应 `exp(−E_D/k_BT)` 有约
  350 倍的 1σ 分散，选 σ 时按这个量级估计影响

#### 空间分布 `defects.spatial`

xy 方向始终均匀随机（周期边界），`mode` 只控制 z 方向（膜厚方向）的轮廓。

| 键 | 含义 | 说明 |
|---|---|---|
| `mode` | `uniform`（体内均匀）或 `interface`（界面富集） | 缺省 `uniform` |
| `enrichment` | 富集因子 g，界面处相对体内平台的密度倍数 | 只在 `interface` 生效，必须为正 |
| `decay_length_nm` | 特征衰减长度 λ | 同上 |

```yaml
defects:
  spatial:
    mode: uniform        # 体内均匀随机
```

```yaml
defects:
  spatial:
    mode: interface      # 双端指数富集
    enrichment: 5.0
    decay_length_nm: 1.0
```

界面富集轮廓：

    n(z) ∝ 1 + (g−1)·[exp(−z/λ) + exp(−(Lz−z)/λ)]

轮廓在采样时归一化，**缺陷总数始终 = 浓度 × 体积**——浓度控制数量、轮廓只控制形状，
这样不同轮廓下的 I-V 才是在同一缺陷数下的比较。

- 两种模式的 z 都限制在可采样窗口 `[d, Lz−d]`，d = `min_separation_nm`（贴在电极上的
  缺陷会让 KMC 时间步塌缩，见下文"几个约定"）；`interface` 模式在这个区间上对归一化轮廓
  做数值逆变换采样
- **轮廓锚定到可采样窗口**：峰落在窗口边界 z=d（距界面 d 处）、值仍 ≈ g，**不被 d 削掉**——
  d 内按数值稳定性要求本就不放缺陷，把峰锚在最靠近界面的可放置平面上，否则 λ ≲ d 时峰
  整个落进禁区、采样窗口上近乎平坦，加大 g、减小 λ 都救不回来（早期版本正是这么丢掉富集的）
- `enrichment: 1.0` 在数学上退化为均匀分布，但 `interface` 模式要求 g 与 λ 都为正；
  要均匀分布直接写 `mode: uniform`
- 例：Lz=8 nm、g=5、λ=1 nm，看距两侧可采样边界各 λ 以内的缺陷占比——d=0 时轮廓给
  44%、均匀分布给 25%（`tests/test_defects.py` 拿采样结果与这个解析积分对过）；d=0.5 nm
  时窗口变窄，两数变成 47% 与 29%，峰值仍 ≈ g（不被削）

#### 确认配置是否落到位

```bash
python -m kmc_leakage plot --config examples/single_run.yaml --kind defects --out defects.png
```

左图是按能级着色的三维散点（`gaussian` 模式的能级分散一眼可见），画其中的**代表实现**
（每次运行随机抽取的那批种子里的第一个）。右图是沿 z 的缺陷数密度直方图（核对
`uniform` / `interface` 轮廓）：**汇总系综 M 个实现再除以 M**——默认 N≈68、每 bin 才几个
缺陷，单实现的直方图基本是泊松噪声，看不出富集，汇总后 σ∝1/√(MN) 才压得下来。横轴只画
可采样窗口 `[standoff, Lz−standoff]`：两端 standoff 内本就不放缺陷，裁掉后图上不留空、
边缘柱恰好落在富集峰所在的可放置平面上。缺陷实现由
种子决定（没有 `ensemble` 段时用 `simulation.seed`），换种子即换一批实现。

在 Python 里改分布用 `dataclasses.replace`。缺陷现在挂在层上、`config.defects` 是只读
属性，改的是 `stack.layers[i].defects`。注意 `config.py` 的字段一律是 SI（能级是 J、长度
是 m），eV/nm 换算只发生在 YAML 读入边界：

```python
import dataclasses
from scipy import constants
from kmc_leakage.config import DefectLevelConfig, DefectSpatialConfig

EV = constants.electron_volt
stack = config.device.stack
layer = dataclasses.replace(stack.layers[0], defects=dataclasses.replace(
    stack.layers[0].defects,
    level=DefectLevelConfig(mode="gaussian", ED=1.15 * EV, sigma=0.15 * EV),
    spatial=DefectSpatialConfig(mode="interface", enrichment=5.0, decay_length=1.0e-9),
))
stack = dataclasses.replace(stack, layers=(layer,) + stack.layers[1:])
config = dataclasses.replace(config, device=dataclasses.replace(config.device, stack=stack))
```

### 系综平均 `ensemble`

一次 KMC 只看到一套具体的缺陷坐标，而漏电流对这套坐标敏感。要给出可与实验比绝对
值的 J，得跑 M 个独立实现再取**算术平均**——电流是可加量，实验测到的也是整片电极上
大量局部通道之和，所以取算术平均而不是几何平均。

```yaml
ensemble:
  count: 5   # M = 5，每次运行随机抽取 5 个互异种子
```

| 行为 | 说明 |
|---|---|
| 一个种子 = 一次实现 | 种子同时驱动缺陷生成与 KMC 轨迹，换种子换掉的是"缺陷摆在哪、能级多深"加上轨迹噪声 |
| 种子每次运行随机抽取 | `count` 给出实现个数 M，加载配置时从系统熵源抽 M 个互异种子；同一份配置每次跑都是新的一批，抽到的种子随 CLI 进度与 CSV 落盘，要复现照抄 |
| 实现之间完全独立 | 每个实现自己做一遍静电预计算（缺陷位置变了，耦合矩阵就得重算），没有共享状态 |
| 实现内部仍是同一个器件 | 扫描时一个实现跑遍所有工作点，曲线形状由单个器件决定，系综只在工作点之间纵向平均 |
| 误差范围 | 平均值的标准误 σ/√M（σ 为实现间样本标准差，`ddof=1`）；M=1 时按 0 处理。走 CLI 摘要与 CSV，曲线图上画成误差棒 |
| 整段省略 | 退化为 `simulation.seed` 一个实现，行为与加系综之前完全一致 |

`ensemble.count` 必须是 ≥ 1 的整数，否则读入时报错。

`sweep --out` 在有系综时写的是另一套表头：`J_mean_A_cm2`、`J_sem_A_cm2`、各机制平均，
外加每个实现的原始值 `J_seed<种子>_A_cm2`。原始值留在 CSV 里，换一种平均口径（比如
改看几何平均或中位数）不必重跑。

**成本是线性的**：M 个实现就是 M 倍机时，实现之间彼此独立、没有任何复用。压涨落的
另一条路是扩大侧向尺寸（缺陷数 N ∝ Lx·Ly，单个实现内部自平均更充分）。两条路**机时
上等价**——实测把 4 V 的相对标准误压到 10%，50×50 nm 需 M≈27、100×100 nm 需 M≈8，
而单实现机时正好差 3.8 倍。

内存上原先扩面积也是 O(N²)（`pair_distances` 的 (N,N,3) 与静电耦合的 (3,N,N) 都是
稠密矩阵），陷阱间跃迁改成近邻表之后只剩静电耦合一项。**当前默认取 50×50×8 nm
（N=60），涨落靠种子压**；要用更大的侧向尺寸，先看静电耦合矩阵的 24N² 字节
（N=2700 时 222 MiB）够不够，那是现在唯一的 O(N²) 内存项。

实测的实现间散布（16 个实现，n_events=2000，参考预设，Lz=8 nm）：

| 侧向尺寸 | 缺陷数 N | σ/mean @2 V | σ(log₁₀J) @2 V | σ/mean @4 V | σ(log₁₀J) @4 V |
|---|---|---|---|---|---|
| 50 × 50 nm | 60 | 38.8% | 0.190 | 52.0% | 0.329 |
| 100 × 100 nm | 240 | 16.9% | 0.080 | 27.9% | 0.138 |

缺陷数翻 4 倍把 σ 砍掉约一半，正是 σ ∝ 1/√N；两种尺寸的均值在各自标准误内一致
（2 V 差 8%，4 V 差 0.8%），侧向尺寸没有引入系统性偏移。改成 8 nm 默认之前的 50×50×9 nm
（N=68，20000 事件）实测 M=5 给出 2 V 下 11.7% 的相对标准误、σ(log₁₀J)=0.128。

### TDDB 配置 `examples/zro2_tddb.yaml`

TDDB 阶段（规格见 `PROJECT_CONTEXT_TDDB.md`，决策记录见 `docs/adr/tddb-decisions.md`）的
配置。材料（名为 `ZrO2`）、缺陷段（`min_separation_nm` 之外）、电极、接触势垒、层厚、侧向尺寸（50 × 50 nm，60 个初始缺陷）与
`single_run.yaml` 逐项相同，t=0 的初始缺陷按这份缺陷段生成，应力生成的 O 空位也用它。与默认盒子不同的有三处：

| 项 | 取值 | 原因 |
|---|---|---|
| `grid` | 200/200/80（hx = hy = 0.25 nm，hz = 0.1 nm） | 0.5–0.9 nm 间距空位对的库仑能与 hx = 0.2 nm 相差 ≤ 3.6%，默认盒子的 1.56 nm 步长算低 30–57%；t=0 本底的系综均值变化 < 1%（ADR-032） |
| `ensemble.count` | 30 | I-t 系综平均与 t_BD 的 Weibull 拟合用（Phase 5）；原按 20 × 20 nm 小盒子的散布定（ADR-026） |
| `min_separation_nm` | 0.26 | ZrO₂ 的 O–O 间距；前驱位点按它铺满，r_c = 0.3 nm 时紧挨着的空位才能连通（ADR-055）。hx = 0.25 nm 网格上 0.26 nm 间距空位对的库仑能单对相差 −9% ～ +21%（0.5 nm 以上 ≤ 3.5%） |

侧向尺寸原为 20 × 20 nm 小盒子（10 个初始缺陷）。小盒子里 t=0 电流几乎由一个贴负极的缺陷承载，3.6 V 下
σ(log₁₀J) = 1.18，50 × 50 nm 下 0.35（ADR-050），默认盒子因此改为 50 × 50 nm（ADR-051）。

顶层 `tddb` 段放 TDDB 缺陷生成、热反馈与击穿判据参数，由 `io.py` 读成 `Config.tddb`（SI 单位），
整段省略即纯漏电流仿真（`Config.tddb = None`）。字段在用到它的阶段逐个加入，现在有九个，除 k_TH
外都必须给出（前驱位点不设浓度，写了 `precursor_concentration_cm3` 直接报错，ADR-055）：

```yaml
tddb:
  activation_energy_eV: 1.30             # E_A：无场下形成 Frenkel 对的激活能，全部前驱位点相同（式 T2，必须为正，ADR-053）
  precursor_ionization_energy_eV: 1.15   # E_t1：前驱能级相对局部导带底的深度（必须为正）
  precursor_relaxation_energy_eV: 1.0    # E_rel1：第一次电子俘获导致键断裂、O 空位形成的弛豫能（必须为正）
  dipole_moment_eA: 1.0                  # p₀：前驱有效偶极矩均值 [e·Å]（必须 ≥ 0，取 0 即关掉场加速）
  field_averaging_width_nm: 0.5          # P_bb 宏观场的 Gaussian 平均窗口 σ（必须 ≥ 0，取 0 即局域场，ADR-043）
  percolation_distance_nm: 0.3           # r_c：导电通路连通距离，负极侧与正极侧导电簇连通即击穿（必须为正，ADR-054/055）
  thermal_conductivity_W_mK: 2.0         # k_TH：热导率（式 T14，必须为正）；省略即等温
  max_generated_vacancies: 3000          # 未击穿时生成空位数达到此值即停（机时上限）
  stop_time_s: 1.0e4                     # 应力时间上限 [s]
```

恒压应力取 `simulation` 段的 `voltage_V`，环境温度取 `temperature_C`，不另设字段（ADR-033）。

E_t1 取与 O 空位预设相同的值。E_rel1 是第一次电子俘获导致 O-Zr-O 键断裂、O 空位形成的
弛豫能（论文 SiO₂ 里对应 Si-O 键，取 1.5 eV），只用于空位生成的计算——E_rel1/ℏω₀ 作为 R_1e
多声子线型的参数——与电子输运用的材料 `huang_rhys_S` 无关，现在暂取 1.0 eV（ADR-030）。p₀ = 1 eÅ 时
3.6 V / 8 nm 下场加速项 p₀·(2+κ)/3·E 压低 0.6 eV，有效势垒对电压的斜率 0.167 eV/V（25 °C 下 P_bb 每伏约 2.8 个数量级），
只由 p₀ 定。E_A = 1.30 eV（全部前驱相同，ADR-053）是初值，标定待定。k_TH 取 2 W/(m·K)（ADR-040）。前驱位点不设浓度：按缺陷段的
`min_separation_nm`（0.26 nm）在可采样窗口里铺满，约 4.3×10²² cm⁻³，50 × 50 nm 盒子约 80 万个（ADR-055）。r_c 取 0.3 nm（ADR-054/055），
只有紧挨着的空位连通。
生成率场加速项里的相对介电常数 k 直接取材料的 `kappa`，前驱的 ℏω₀ 取材料的 `phonon_energy_eV`，都不另设
配置项。

生成的 O 空位与初始缺陷是同一物种，电荷按施主约定：出生时被一个电子占据，电子发射后带 +1，缺陷段的
`charge_state_Z: 1.0` 就是电子离开后的电荷（ADR-035）。生成反应"前驱 + e⁻ → V_O + O_i⁻"按电荷守恒处理（ADR-048）：
前驱俘获的那个电子的电荷留在 O 间隙离子上，O_i⁻ 是新空位的固定电荷 q_fix = −e（`DefectEnsemble.fixed_charge`），与空位
重合放置（`fixed_charge_offset` = 0）。缺陷净电荷 q·(Z − n) + q_fix，新空位连同 O_i⁻ 出生时带 −e、放出电子后为中性；
初始缺陷 q_fix = 0。

生成速率的纯函数在 `rates/generation.py`（式 T1–T4，ADR-027/039）：前驱单步俘获速率 R_1e = Σ_两侧电极 (R2 + R4)
+ Σ_近旁陷阱 p·R7。`single_electron_capture_rate` 给电极项，与引擎处理真实陷阱的口径逐项一致；
`trap_precursor_capture_rate` 给陷阱项，是引擎的陷阱间通道把目标换成前驱、按源陷阱占据加权，`tddb.py` 在每个前驱位点上
对已占据陷阱求和（ADR-052）；`tddb.py` 里电极项在 (z, 前驱能级, 温度) 的插值表上求（ADR-055）；
`bond_break_probability` 给断键概率 P_bb（有效势垒 ≤ 0 时饱和为 1）。P_bb 里的场是宏观场：泊松解的场在 σ 的 Gaussian
窗口上的平均（`PoissonSolver.gaussian_average`，ADR-043），Lorentz 因子 (2+k)/3 只对这种在约 1 nm 上近似均匀的场成立；
`bond_break_generation_rate` 给单个前驱的生成速率 G = R_1e·P_bb，俘获速率作为参数传入。

TDDB 外层每生成一个空位都要重求一次输运稳态，内层默认用**簇级平均场**求解器
（`mean_field.py`，ADR-031/038）：不做时间推进，直接求占据的不动点 p = p̂(p)。p̂ 就是 KMC 电流
估计量里的条件稳态占据——孤立陷阱按邻居的平均占据算跃迁供给，强耦合簇（ADR-022）内部枚举
全部组态、按真实电荷分布精确求解，簇外取平均电荷作环境。陷阱间跃迁的速率每条只算一次、两端共用：
簇内对与 KMC 同口径（MA 因子含 −U_ab），两端都按平均场处理的对取两端平均场能级之差，簇–孤立对的
成员一端按簇内组态求值、孤立一端按簇内稳态分布求平均。这样不动点处电流连续、零偏压下没有环流
（簇–簇边界留一个小环流，ADR-038）。电流是不动点处同一个估计量公式给出的正极净通量，分解口径
与 KMC 相同：

```python
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.mean_field import solve_mean_field

sim = KMCSimulation(config)                      # 缺陷实现、静电预计算、强耦合簇
result = solve_mean_field(sim)                   # 从全空出发
print(result.current_density, result.converged)  # A/m²；未收敛时 converged=False，不得当稳态交付
print(result.by_mechanism, result.occupancy)     # 分解口径同 KMC；不动点占据
again = solve_mean_field(sim, occupancy=result.occupancy)  # 热启动：外层下一步从上一步的占据出发
```

不动点在 ln p 上用 Anderson 混合求解，收敛判据是占据的相对残差 ≤ 1e-10（占据只有 1e-10 量级、
经快跃迁供给邻居的陷阱也要收敛到位）；50 次内没收敛就改用牛顿法（有限差分 Jacobian + Broyden 更新 + 回溯线搜索，
`result.newton_steps` 记步数，ADR-048）。膜体内致密的空位网络里 Anderson 会在不动点附近游走，牛顿法收敛。Jacobian 按列分组求
（ADR-058）：两两相距 > 1.25·L_z 的缺陷一组、一次 F 求一整组列，50 × 50 nm 盒子 360 个缺陷时 64 组（逐列要 360 次 F），一次求解
由 266 s 降到 86 s，结果不变。20 × 20 nm 盒子（N=10）下 5–7 次迭代、约 0.01 s 收敛，同一实现
20000 事件的 KMC 要 30–50 s；与开快陷阱的长窗口 KMC 在 2–4 V、25/125 °C 六个工作点上系综均值一致
到 5×10⁻⁵。浓度提高到 3×10¹⁹ cm⁻³（N=96）时 10–11 次、约 0.3 s，与长窗口 KMC 之比 0.997，在统计误差
以内（对拍见 ADR-031，逐实现数据在 `results/9.29/tddb_phase2_mf_vs_kmc.csv`）。

**TDDB 瞬态运行**（`tddb.py`，ADR-033/036/052/055）：`run_tddb(config)` 跑一条恒压应力下的轨迹。t = 0 在可采样窗口里按最小间隔
min_separation 铺满离散前驱位点（`place_precursors`，最大随机堆积：位点两两之间、与初始缺陷之间都 ≥ min_separation，且再也放不下
一个位点，ADR-055；E_A 全部相同，ADR-053）。外层在前驱位点上
做 Gillespie：各缺陷净电荷 q·(Z − p) + q_fix 与外加偏压解泊松方程（接电极的导电簇与该电极等电位，ADR-054；悬空导电簇自身等电位、
总电荷不变，ADR-056），在每个位点上求
G_i = R_1e·P_bb（`generation_rates`）：R_1e = 两侧电极 R2 + R4 + 近旁已占据陷阱的 Σ p·R7（接电极导电簇的空位按电极费米能级处占满的
电子源计入），取法同引擎处理真实陷阱；电极项在 (z, 前驱能级, 温度) 的插值表上求（ADR-055）。抽中的位点原地变成新空位，只用掉
这一个位点。新空位与初始缺陷同一物种、出生时被一个电子占据、带着重合的 O_i⁻。内层每步按当前缺陷集合重建 `KMCSimulation`、从上一步占据热启动求平均场
稳态，静电耦合增量扩展（每个新空位 1 次正向 + 3 次伴随泊松求解），近邻表与强耦合簇从头算。

**热反馈**（`thermal.py`，式 T13/T14）：`heat_release` 在内层稳态上逐事件累加电子放给晶格的能量 ΔE·R
（弹性隧穿在陷阱处不放热，多声子俘获 / 逃逸放出 / 吸收 p·ℏω，陷阱间跃迁放出能级差，PF 发射在陷阱处
吸热），介质内的功率解稳态热方程 k_TH·∇²T = −P（两电极恒温，直接调用 `PoissonSolver.solve`）。温度场在
前驱位点上的值进下一次的生成率，缺陷处按放热加权的平均温度作为下一步内层输运的全局标量温度 T(t)。
输运与热每步只走一遍、不迭代自洽，所以记录里的电流比温度滞后一次生成。省略 `thermal_conductivity_W_mK`
即等温运行。

**击穿判据**（§6.3，ADR-054）：O 空位间距 < r_c（`percolation_distance_nm`）连成导电簇，离电极平面 < r_c 的接到该电极；负极侧
与正极侧导电簇连通即击穿，轨迹在这一刻结束，`end_time` 就是 t_BD。这条记录的电流加上细丝按一个量子电导导通的 G₀·V
（G₀ = 2e²/h，3.6 V 下 0.28 mA），不再求生成率。接电极的导电簇在生成率里是电极的延伸：与电极等电位、按电极费米能级处占满的
电子源给近旁前驱送电子，尖端前方的生成因此加快。没接电极的导电簇（≥ 2 个空位）也是导体：成员等电位、总电荷不变，电压集中到它两端
的缺口上（ADR-056）。内层输运不变。论文的"25 个空位、间距 ≤ 6 Å"团簇只记录，不作判据。

```python
from kmc_leakage.io import load_config
from kmc_leakage.tddb import fit_transient, run_tddb

result = run_tddb(load_config("examples/zro2_tddb.yaml"))  # 一个缺陷实现（simulation.seed）
result.time, result.current_density        # 第 k 个状态从 time[k] 保持到 time[k+1]（最后一个到 end_time）
result.stop_reason, result.breakdown_time  # "breakdown" / "max_generated" / "stop_time" / ...；击穿时的 t_BD
result.transport_current, result.filament_current, result.conductor_gap  # 内层输运电流、细丝电流（击穿记录为 G₀·V）、导体间隙
result.power, result.effective_temperature, result.peak_temperature   # 介质功率、缺陷处加权温度、温度场最高值
result.dense_cluster_size                                             # 最大 6 Å 致密团簇
result.generation_rate, result.n_precursors                           # ΣG、剩余前驱位点数
result.space_charge, result.empty_generated                           # 净空间电荷 Σ[q·(Z − p) + q_fix]、空态生成空位数
result.converged, result.newton_steps                                 # 平均场收敛、牛顿法步数（0 = Anderson 收敛）
result.max_cluster_size, result.n_split                               # 簇大小与切分数
fit, times = fit_transient(result)         # 按 log t 等间距取样后对内层输运电流做双对数拟合：fit.exponent、fit.r_squared
```

**停跑后续跑**（ADR-057）：`run_tddb` 的 `checkpoint(状态)` 每步开始时拿到一次全部状态（`TDDBState`：缺陷集合、出生时刻、时刻、平均场占据、
剩余前驱位点、输运温度、随机数流），`save_state` 存成 npz（50 × 50 nm 约 20 MB），之后 `run_tddb(config, start=load_state(path))` 接着跑。
续跑的结果只含起跑之后的记录；静电耦合从头算，与不停跑的同一条轨迹只差舍入误差。

```python
from kmc_leakage.tddb import load_state, run_tddb, save_state

run_tddb(config, checkpoint=lambda state: save_state(state, "state.npz"))  # 每步覆盖存一次（实际可每 N 个空位存一次）
result = run_tddb(config, start=load_state("state.npz"))                   # 停跑后从存下的那一步接着跑
```

生成率为 0 时（例如 E_A 取得极大），电流逐位等于同一实现的稳态平均场结果、保持到 `stop_time_s`（开热反馈也成立）。
ADR-055 之后 50 × 50 nm 盒子约 80 万个前驱位点：铺点一次约 4.5 min（峰值工作集约 1.5 GB），生成率一步约 7 s、整步约 9 s。以下是 ADR-051 时的数据：单进程峰值工作集约 1.2 GB、t = 0 附近每步约 41 s（ADR-050）；4 条并发跑到击穿（71–123 个空位）时每步 50–92 s，单条 1.6–2.3 h
（ADR-051）。工作集与横向面积成正比：20 × 20 nm 盒子约 0.25 GB，每步约 9 s
（缺陷少时）到 30–40 s（约 400 个缺陷），一条 400 个空位的轨迹约 2.2–2.8 h。热反馈每步多约 10%。

**当前模型与参数下 I-t 不是双对数直线**（ADR-034、ADR-037、ADR-041、ADR-045–048）。ADR-048 之前生成反应每次丢掉一个 −e
（O 间隙离子不带电）：负极侧空态空位积累净正电荷，经 p₀·(2+κ)/3 放大后让生成在负极侧自加速，3 V 下 ΣG 一直升到 10⁴–10⁵ s⁻¹，
I-t 是"台阶 → 陡升"，击穿落在陡升段、84–99% 是负极侧空位层的 PF 发射（ADR-046）。ADR-048 按电荷守恒处理（O_i⁻ 固定电荷）之后，
20 × 20 nm 盒子 3.6 V / 25 °C 的 seed 1–4（`results/10.09/accept/`）：ΣG 不再自加速（全程 362–442 s⁻¹），平均场每条记录收敛，击穿时电流
全部是 TAT；但新空位仍在负极侧（z 中位数 1.1–1.2 nm），第 21–34 个空位、t_BD = 0.040–0.083 s 即击穿（实验约 1000 s），单条 I-t
仍是台阶状，从第一次生成起双对数拟合 n = 0.27–0.96、R² = 0.40–0.94。击穿前生成次数太少，t = 0 的 J 在种子之间就差 5 个数量级，
单条轨迹看不出幂律。默认盒子改为 50 × 50 nm 之后（ADR-051，`results/10.09/accept50/`），同一工况 seed 1–4 的 t = 0 电流只差 2.4 倍，
单个空位最多把 J 抬高 2.8 倍，4 条 I-t 几乎重合、不再是台阶状；但双对数向上弯（从第一次生成起拟合 n = 0.37–0.57、R² = 0.56–0.85，
从第 25 个空位起 n = 1.3–2.5），第 71–123 个空位、t_BD = 0.032–0.053 s 即击穿。热反馈仍可忽略（升温 ≤ 4×10⁻³ K）。
原因是模型 C 的 ΣG 全程不变（前驱不消耗），加上 t = 0 本底 J₀ 离击穿判据只差 15–36 倍、空位带来的电流对空位数超线性。ADR-052
改为离散前驱位点（断键一次即用掉、E_A 按高斯分布）之后（`results/10.11/discrete/`）：生成随前驱用完变慢（N ∝ t^0.2–0.34），10⁴ s 之前
I-t 双对数接近直线（n = 0.15–0.28、R² = 0.92–0.97）；10⁴ s 之后 I-t 变平、J 缓慢越过 10 A/cm²，E_A = 1.30 eV 时 t_BD = 3.2×10⁴ s 到
10⁷ s 以上，种子之间差两个数量级以上，模型里还没有让电流突变的机制。ADR-053 去掉 E_A 分布（全部前驱 1.30 eV，
`results/10.12/fixed_ea/`）之后，seed 1 / 2 / 4 的 t_BD = 0.69–1.85×10⁴ s、只差 2.7 倍，但 I-t 双对数重新向上弯（n = 0.35–0.56），
seed 3 的 J 停在 6.95 A/cm²、到 1.1×10⁷ s 未击穿。ADR-054 把击穿判据改为贯穿通路（接电极的导电簇在生成率里与电极等电位，通路形成时加
细丝电流 G₀·V）之后，N_T = 10²¹ cm⁻³、r_c = 1.0 nm 的 seed 1–4（`results/10.12/percolation/`）都以突变方式击穿：t_BD = 90–273 s，击穿前
TAT 电流 2.8–9.5 A/cm²，接通时跳到 1.12×10⁷ A/cm²，最后一段在 0.010–0.59 s 内由 6–16 个空位接通；I-t 双对数仍向上弯（n = 0.21–0.81）。
ADR-055 取消 N_T、前驱位点按 0.26 nm 铺满（约 80 万个）、r_c = 0.3 nm 之后，seed 1–3（`results/10.13/dense_sites/`）跑到 412–429 个空位（约 13 h，
t = 5.5–6.2 s）都没有击穿：J 从 0.28–0.90 升到 93–103 A/cm²，但接负极的导电簇只有贴界面的 1–5 个空位，新空位在 z ≈ 1–1.5 nm 铺成一层、
再长出到 6.2–6.9 nm 的竖直细柱，柱里空位间距多在 0.3 nm 以上；I-t 双对数向上弯（n = 0.65–1.34）。ADR-056 让悬空导电簇也等电位之后，seed 1
（`results/10.14/floating/`）在 r_c = 0.4 nm 下一根细柱先填上它与负极之间的缺口、接上负极，再在 10⁻⁹ s 内长到 6.66 nm（停时间隙 1.34 nm）；
r_c = 0.3 / 0.35 nm 到 349–400 个空位仍没有接上负极，结果对 r_c 不稳定。
Phase 3 的 15 条轨迹在 `results/9.30/tddb_phase3_*.csv`，ADR-037 的轨迹在 `results/9.30/phase4/`，ADR-041 的日志与 seed 2 的
CSV / npz 在 `results/9.30/phase4_rerun/`，ADR-045 的逐步 CSV、快照、抽查与汇总图在 `results/10.03/rerun/`，ADR-046 的在
`results/10.03/ext1000/`，ADR-048 的在 `results/10.09/`，ADR-051 的在 `results/10.09/accept50/`，ADR-055 的在 `results/10.13/dense_sites/`。

Phase 0 的 20 × 20 nm 小盒子与默认盒子的稳态本底一致：2–4 V、25/125 °C 六个工作点上，小盒子（M=128）与
默认盒子（M=32）的系综均值都在合成标准误的 2 倍以内；3–4 V 下 TAT 占总电流 92% 以上。
小盒子的单条轨迹之间可差 5–6 个数量级（完整对比表见 ADR-026，逐种子原始数据在
`results/9.29/tddb_phase0_box{20,50}nm.csv`）。

### 几个约定

- **能量零点 = 负极费米能级**，正极费米能级在 −eU
- **不考虑镜像力势垒降低**：所有势垒按零场高度 E_B 处理，界面处能带保持直线，
  WKB 积分在未经圆化的直线/分段线性势垒上做
- `min_separation_nm` 同时作用于两处：缺陷两两之间的最小距离，以及缺陷到两侧
  电极界面的最小距离（贴在电极上的缺陷 P(E)→1、速率发散到 1e16 s⁻¹ 量级，会让
  KMC 时间步长塌缩）

---

## 物理模型

| 编号 | 机制 | 统计口径 |
|---|---|---|
| R1（阈值以下） | 电极↔电极 直接/FN 隧穿 | `direct_fn` |
| R1（阈值以上） | 电极↔电极 热激发越过势垒 | `thermionic` |
| R2 / R3 | 电极→陷阱 / 陷阱→电极 弹性隧穿 | `tat` |
| R4 / R5 | 电极→陷阱 / 陷阱→电极 非弹性（多声子） | `tat` |
| R6 / R7 | 陷阱→陷阱 弹性 / 非弹性（Miller-Abrahams） | `tat` |
| R8 | 陷阱→导带 PF 热发射 | `pf` |

统计按"电子抵达收集电极（或进入导带后被收集）的最后一步事件类型"归类。R1 的两
部分以势垒沿隧穿路径的最高点能量为阈值切分。

`rates/` 下每个函数都是纯函数：输入物理量与参数、返回速率数值，不依赖全局状态、
不做 I/O，docstring 标注对应的 R 编号。

---

## 实现要点

这几条是规格文档里没有、但影响结果解读的工程决定。

### 1. 电流估计量

稳态净电流是正极侧两个大通量之差（正极→陷阱俘获 vs 陷阱→正极逃逸），二者在稳态
下几乎精确抵消。**直接数到达正极的净电子数在本项目参数下不可用**：噪声比信号大
2–3 个数量级，低偏压下常常给出 0 或负电流。

主输出改用条件期望（Rao-Blackwell）估计量：把单个陷阱的条件稳态占据
p̂ = C/(C+E) 代入净通量表达式后，相消项解析约掉

```
flux_i = [(C_c + H_in)·E_a − (E_c + H_out)·C_a] / (C + E)
```

（C_c/C_a 为负极/正极侧俘获总速率，E_c/E_a 为向负极/正极的逃逸总速率，H_in/H_out
为陷阱间跃迁的进出速率。）KMC 仍然在做实事——采样到的组态决定静电环境、MA 能量
因子与跃迁的 Pauli 阻塞。

**强耦合簇**（ADR-022，`clusters.py`）：上式只把陷阱自身的占据解析平均掉，H_in/H_out
里的邻居占据仍是采样值。两个陷阱间的跃迁与它们自身的电极交换一样快时，这会出两种错：
短测量窗口里邻居的稀有填充恰好落进来就高估（I-V 曲线上的尖峰），没落进来就漏算；即便
窗口无限长，也会丢掉两者占据之间的关联（强耦合对上实测只剩精确值的 1–7%）。所以每个
工作点初始化时，把跃迁速率 ≥ 1e-3 × 两端电极交换速率较小者的陷阱对连成簇（上限 8 个
陷阱，超限的在内部提高阈值再切分），测量时对每个簇枚举簇内 2^k 个组态、按各组态的
真实电荷分布重算速率，解簇内主方程稳态，簇外占据作为环境。孤立簇上结果与主方程精确解
一致，且与轨迹无关；单陷阱仍走上式，逐位不变。

**快陷阱移出事件预算**（ADR-023，可选，`simulation.fast_trap_rate_s`）：贴电极的陷阱与
电极交换电子可达 1e9–1e10 s⁻¹，Gillespie 的事件几乎全花在它们的来回翻转上，4000 个事件
只推进几十纳秒。配置 R_cut 后，全空组态下与电极交换总速率 ≥ R_cut 的陷阱退出事件表，
占据改用与当前慢组态自洽的条件稳态 p̂（进静电叠加与跃迁的 Pauli 阻塞），快↔慢跃迁仍是
事件、执行时只改慢陷阱一端。估计量不受影响：快陷阱的电流本来就由上式给出，照样可以是
强耦合簇成员；近似只在于慢陷阱感受到的是快陷阱的平均电荷，所以要求快陷阱比剩余事件
快得多（min Γ_F 与测量窗口内平均事件速率之比 < 10 判未通过）。t2_2nm 叠层实测
R_cut=1e9 把 4000 事件的窗口从 38 ns 延长到 24 μs，电流与不移出时相差 0.1%。不配置时
逐位退回原算法。

结果里同时给出另外两个估计量作交叉检验：`current_direct`（瞬时通量的时间平均）与
`current_counts_total`（数事件 + 快陷阱电流 + R1）。这三个估计量是 F04（ADR-021）要求的
口径。每个工作点
带一个**验证状态**（`validated` + `validation_notes`）：只在正向收集（U+V_bi>0）、测量
窗口充分、KMC 采样占据与条件稳态 p̂ 接近、批次间电流一致、（启用快陷阱时）时间尺度分离
足够、且（正极事件够多时）净计数交叉校验通过时才判"通过验证"，否则标"未通过验证"、
不得当可信稳态电流交付。批次间一致性
用批次均值法检验：测量窗口按事件数等分成 10 批，电流相对标准误超过 0.05 即判未通过——
它是强耦合簇之外剩余采样问题的保险。净事件计数在本项目低
漏电流下噪声大（比信号低 2–3 个量级），只有统计够用才纳入门槛。窗口内还核对全局电荷
守恒 (I_C−I_A)·Δt = q·(ΔN_inside − fast_exchange)（逐事件记账的恒等式；快↔慢跃迁不经过
电极，未启用快陷阱时 fast_exchange=0）。系综层面另给出成对差
ΔJ = J_条件 − J_数事件（`current_counts_total`）及其跨种子标准误。条件通量估计量本身用**小状态空间主方程
稳态解**独立对照（`master_equation.py` / `test_master_equation.py`，复现 review 的 25%
反例与一致极限）——耦合网络里它不自动无偏，所以要独立验证。

**分层归因**（`by_layer`）：把逐缺陷通量按 layer_id / interface_id 分组得到端口电流贡献，
两个界面各一栏、界面缺陷只计一次、R1 单列，满足 Σ_{ℓ,Γ} I_{ℓ,Γ} = I_TAT+I_PF、
I_total = Σ I_{ℓ,Γ} + I_R1，从单点贯通到 sweep / ensemble / CSV。

### 2. 泊松方程只在初始化时求解

泊松方程是线性的、缺陷位置在一次实现里固定，所以初始化时解 N+1 次（外加偏压 1 次
+ 每个缺陷单位电荷各 1 次），之后任一占据组态的电势/电场都由这些解按占据数线性
叠加得到，主循环里退化为矩阵-向量乘。这是精确的，不是近似。缺陷自身电荷在自己
位置产生的电势（CIC 自作用）就是耦合矩阵的对角元，整项扣除——实测自作用势约
3×q/(4πε₀κh)，对应自能约 0.24 eV ≈ 9 k_BT，并非可忽略的小量。

xy 方向周期 → FFT；每个傅里叶模式在 z 方向退化为三对角方程组，用 Thomas 算法直接
求解，三对角系数只依赖网格与各层 κ，在构造时预先分解、每步复用。**叠层里 κ 按单元
取值**（单元落在哪层取该层的静态 κ），界面对齐到网格节点（层厚必须是 hz 的整数倍，
否则构造求解器时报错）；横向项用相邻两单元 κ 的算术平均。单层退化时各单元 κ 相同，
退回常系数解。所有吃场的机制（R4/R5 前置因子、R8）经统一的场读取接口取局域场——
体缺陷用中心差分，界面缺陷的 E_z 用单侧差分（界面上 E_z 两侧差 κ 比，中心差分会取到
错误的平均）。

### 3. R1 按公式求值一次，不进主循环

电极↔电极隧穿不改变占据组态，放进事件表只会产生大量"空转"事件。它走的是**理想
平面势垒**的物理图像——Richardson-Schottky 对照式也是——所以势垒剖面取零空间电荷的
外加偏压解（`E_C(z) = E_B − e·(U+V_bi)·z/Lz`，无电荷时泊松解严格线性），在初始化时
按公式求值一次，`run()` 末尾把 `I_R1 = q·R1` 直接加进 `by_mechanism`。

可观察的后果：`direct_fn` / `thermionic` 两项**不随 seed、`n_events`、缺陷实现变化**，
也没有采样噪声；势垒顶恒为 E_B，两个分支的切分阈值与 RS 对照式的 φ_B 严格同口径。
参考预设下 R1 占总电流 10⁻¹¹–10⁻⁸，总电流数字不受影响。

### 4. R6 / R7 合并为一条通道

R7 = R6 × Θ，而 E_b ≤ E_a 时 Θ = 1 即 R7 ≡ R6，两条并列会把下坡跃迁速率精确地
算成两倍。本阶段陷阱间跃迁一律走 R7；R6 仍作为独立纯函数实现并单独测试。等 R7
换成量子模型之后两者才需要分开计入。

### 5. 系综平均是最外层循环

`ensemble.py` 只做一件事：把种子逐个换掉、把结果按算术平均合起来。每个实现从缺陷
生成开始重来一遍（新坐标 → 新的静电耦合矩阵 → 新轨迹），实现之间没有任何复用，
所以机时严格是 M 倍。好处是这一层天然可并行——真要跑大 M 时，把 `run_ensemble`
的循环换成进程池即可，不用动 KMC 内部；本项目发的是串行版本。

一个能直接看到的后果：`direct_fn` 与 `thermionic` 两项在系综里**标准误恒为 0**。
R1 按构造与缺陷实现无关（见上文第 3 条），M 个实现给出逐位相同的值。这条写成了
测试，同时也是"系综代码没有把实现之间串味"的检查。

---

## 测试

```bash
pytest tests/ -v      # 268 个测试，约 110 s
```

| 文件 | 覆盖内容 |
|---|---|
| `test_rates_limits.py` | R1–R8 的解析极限与物理约束：WKB 对矩形/三角势垒的解析解、R1 零偏压归零与玻尔兹曼极限、R2/R3 占据因子互补、**R4/R5 细致平衡**（R5/R4 = exp((E−E_F)/k_BT)，与 N_C、P、L_p 取值无关）、声子权重的低温泊松极限、R7 上下坡行为、**R8 的零场与高场两个极限** |
| `test_generation.py` | TDDB 生成速率（式 T1–T4，ADR-027）：**G 的零场极限** = R_1e·exp(−E_A/k_BT)、**ln G 对场严格线性**（斜率 p₀·(2+κ)/(3k_BT)）、有效势垒 ≤ 0 时 P_bb 饱和为 1、场加速项用静态 κ（3 V/8 nm 下每 1 eÅ 压低 0.5 eV）、**R_1e 的电极项等于引擎里同一参数陷阱的两侧 R2 + R4 之和**（正反偏压各一次，rtol 1e-12），拆成弹性部分与 F² × 多声子部分后各自等于两侧 R2、R4（ADR-044）；**陷阱项 = p·R7**（下坡只剩几何因子、上坡乘 exp(−ΔE/k_BT)、源陷阱空着为 0，ADR-039） |
| `test_richardson.py` | R1 热发射子部分 vs Richardson-Schottky 解析式的**独立检验**：单层里比值恒为 m_e/m_cb 且不随温度/偏压漂移，三个缺陷浓度下逐位相同（R1 按构造与缺陷无关）；另配一例用体内鼓包把势垒顶推离界面；**叠层里以 E_max（分段势垒顶，落在 Al₂O₃ 内）为参考的 R1 供给函数玻尔兹曼导出式对照**（ADR-012） |
| `test_electrostatics.py` | 解代回离散泊松方程的残差、均匀电荷的解析抛物线、**线性叠加成立**、CIC 电荷守恒与插值精确性、**单点电荷入盒测试**（近场库仑标度 + 自作用势量级）、**Gaussian 窗口平均**（均匀场不变；点电荷平均后等于 Gaussian 电荷云的解析场、与泊松网格无关，ADR-043） |
| `test_stack_physics.py` | **变 κ 泊松 + 分段能带 + 界面单侧场**（Phase 3）：无电荷解析分压、串联电容 C/A=ε0/S_κ、面电荷电位移跳变 = σ、变系数离散残差、界面对齐校验、同材料多层退回单层求解器、响应互易；E_C 真实台阶与显式 ΔE_C 覆盖、V_bi = (B_c+ΣΔE_C−B_a)/q；界面单侧场按 κ 比跳变、`field_vectors_at` 按 field_side 选侧、体缺陷回归锚点 |
| `test_kmc_consistency.py` | **各分量之和 = 总电流**（机制分解、缺陷分解两条）、可复现性、三个估计量互相印证、零偏压电流可忽略、跃迁通道只走 R7、**近邻表在保留对上与稠密 (N,N) 矩阵逐位相同且截掉的总速率在双精度噪声以下**、**R1 两项等于理想梯形势垒上的公式值且与 seed / 事件数无关** |
| `test_defects.py` | 缺陷数 = 浓度×体积、最小距离（含到界面）、r_D 公式、高斯能级统计、界面富集轮廓与解析积分对比、近邻表与"稠密矩阵+掩码"逐位一致（强制走多块路径）、截断半径命中给定容差 |
| `test_config_io.py` | 单位换算逐项核对、两种有效质量保持独立、势垒显式优先于推算、内建电势、默认扫描范围、两个示例配置的缺陷段同构；**新 schema 结构校验**（材料引用存在、层厚为正、interfaces 数量 = 层数-1）与**旧键拒绝**（顶层 `dielectric` / `defects`、`device.Lz_nm` 各自报错提示新写法）；**`tddb` 段**可省略、E_A / E_t1 / E_rel1 / p₀ / P_bb 平均窗口 σ / 空位数上限 / 应力时间上限 / r_c / k_TH（可省略即等温）的换算、取值检查与缺项报错，写了已取消的 `precursor_concentration_cm3` 报错，**`zro2_tddb.yaml` 与 `single_run.yaml` 的材料/缺陷/电极逐项相同**（缺陷的 min_separation 取 0.26 nm 除外）、横向网格 0.25 nm、前驱 E_t1 与 O 空位预设相同、E_rel1 单独取值、p₀ = 1 eÅ / E_A = 1.30 eV / k_TH = 2 W/(m·K) / r_c = 0.3 nm（ADR-024/028/030/032/035/040/053/054/055） |
| `test_stack_schema.py` | `DielectricStack` 统一查询接口（`Lz` 求和、`single_material` / `bulk_defects` 在多材料/多层时抛错）、**单元素 layers 退化用例**（YAML 解析与直接构造的单层叠层逐位相同：确定性量 rtol ≤ 1e-12、同种子轨迹 rtol ≤ 1e-9）、ZAZ 三层 schema 完整解析、**ZAZ 端到端跑通并给出机制分解**（Phase 4 打通，各分量之和 = 总电流、按缺陷分解 + R1 = 总电流） |
| `test_stack_rates.py` | **分段 WKB + 按材料取速率参数 + 分段 G_ab + 截断重推**（Phase 4）：WKB 逐段 m_di、界面双值台阶不插斜坡、R1 势垒顶落在 Al₂O₃、分段 G_ab 同质退回闭式且跨层用 Al₂O₃ 低质量（路径传播质量，F08）、ZAZ 里 R7 正反细致平衡（F02 事件能量差不变）、截断半径按最慢材料推（ADR-013/F05）、R8 逐缺陷取 ε_opt/ν、**三层同材料 + ΔE_C=0 的速率表（rtol ≤ 1e-12）与轨迹（rtol ≤ 1e-9）退回单层**（ADR-018） |
| `test_pf_collection.py` | **F03 PF 收集概率**（Phase 5）：带偏压 Al₂O₃ 势垒（矩形/梯形/三角）的 WKB 与解析闭式一致、0≤P_A≤1 且随偏压单调、阻挡层检测、**位置感知**（负极侧穿层 P_A<1、正极侧立即收集 P_A=1，界面缺陷按宿主侧判定）、反偏收集电极翻转、R8_eff=P_A·R8 折进事件表、单层 P_A≡1 逐位退回 |
| `test_current_validation.py` | **F04 三估计量与验证 + 分层归因**（Phase 5）：全局电荷守恒 (I_C−I_A)Δt=q·ΔN_inside、正向充分点判通过验证 / 零偏压判未通过验证 / R1-only 判通过、by_layer 单层与 ZAZ（层0/层2/界面0/界面1 各一栏）闭合 Σ=TAT+PF、贯通到 sweep/ensemble/CSV、块 4 确认 R8_eff 沿用界面单侧场 |
| `test_stack_examples.py` | **t₂ 结构对照配置与退化用例全套**（Phase 6，ADR-006/018/019）：三份示例（t₂ ∈ {0,2,4} nm）固定总厚 8 nm、ZrO₂ 两侧对称、层厚对齐到 hz 整数倍、除层结构外逐项相同；t₂>0 有阻挡层 + Al₂O₃ 体内为空 + 两界面各 25 个界面缺陷；**t₂=0 与单层基准 `single_run.yaml` 同一器件且数值逐位退回**（速率表 / R1 / φ / E_C(z) rtol ≤ 1e-12，同种子轨迹 rtol ≤ 1e-9）、三层同材料 + ΔE_C=0 的场量退回单层 |
| `test_master_equation.py` | **条件通量的独立主方程基准**（Phase 5，ADR-021 F04）：复现 review 4 态反例（条件通量比真实稳态低 25%）、单陷阱下条件通量 = 真实稳态、π 是合法分布且 πQ=0、**强耦合对连成簇后 KMC 估计量 = 精确稳态**（rtol 1e-10，单陷阱估计量在同一体系上只有约 1.5%）且短轨迹、换种子不变、长 KMC 轨迹的条件通量收敛到主方程期望 |
| `test_fast_traps.py` | **快陷阱移出事件预算**（ADR-023）：没有陷阱达到 R_cut 时与不配置逐位相同、三陷阱小体系开启后电流仍等于主方程精确稳态（rtol 1e-4）且窗口延长 6 个数量级以上、全局守恒计入快↔慢跃迁 (I_C−I_A)Δt=q(ΔN−fast_exchange)、全部陷阱都快时判未通过、时间尺度分离不足判未通过；配置项 `fast_trap_rate_s` 的解析与取值检查在 `test_config_io.py` |
| `test_mean_field.py` | **簇级平均场内层求解器**（TDDB Phase 2，ADR-031/038）：孤立强耦合簇上 = 主方程精确稳态（rtol 1e-10）、簇外还有弱耦合陷阱时差 <1e-9 而逐陷阱平均场高 2.4%、无强耦合时逐位退回逐陷阱平均场（MA 因子取两端平均场能级之差）且与独立的阻尼迭代同一不动点、占据 9e-11 的供给陷阱按相对残差收敛到位、不动点与初值无关且从不动点热启动一次即收敛、未收敛如实标出、分解之和 = 总电流、无缺陷只剩 R1；**强耦合分量被切开时负极净注入 = 正极净收集**；**零偏压下逐陷阱平均场逐对平衡**（KMC 的 −U_ab 口径净通量达总通量的 19–33%）、**簇–孤立边界没有净通量**；**Anderson 走不动时牛顿法接手**（ADR-048）、收敛到同一不动点，ADR-046 what-if 里 Anderson 200 次不收敛的状态（`tests/data/whatif_s1_n29.npz`）收敛；**分组有限差分 Jacobian**（ADR-058）：组内间距 > 1.25·L_z，与逐列有限差分相对差 ≤ 5%、作牛顿迭代矩阵的收敛因子 ≤ 0.1 |
| `test_tddb.py` | **TDDB 瞬态驱动**（Phase 3/4，ADR-033/036）：增量静电耦合与从头计算一致（初始块逐位相同）、**新空位出生时被一个电子占据、带重合的 O_i⁻（q_fix = −e）** 且与初始缺陷同一物种、出生前后其余缺陷处的电势/电场变化就是出生位置一个 −e 的贡献、**每次出生介质总电荷正好多一个 −e**（ADR-048）、外层从上一步不动点热启动；**G = 0 时 I(t) 恒定且逐位等于同一实现的稳态平均场结果**；每步的强耦合簇与新建 `KMCSimulation` 相同、终态电流与从头求解一致；**离散前驱位点**（ADR-052/055）：最小间隔 min_separation 的最大随机堆积（都在窗口内，两两之间、与初始缺陷之间 ≥ min_separation，随机点离位点或初始缺陷都 < min_separation，同一随机数流结果相同，min_separation = 0 时报错）；新空位落在上一步的某个位点上，只用掉这一个位点、其余原样保留；逐点求值的电极项与引擎在该点放前驱参数探针的两侧 R2 + R4 一致（含空间电荷与固定电荷），插值表与逐点求值按生成率加权的相对误差 ≤ 10⁻³，陷阱项等于对全部已占据陷阱的直接求和、源陷阱全空时为 0，P_bb 取配置的 E_A 与局域场或 Gaussian 平均场；P_bb 未饱和时 E_A 整体平移只改时间标尺；Gillespie 按速率选位点；幂律拟合还原 n、带噪声时拟合优度合理；CSV 与 CLI；**热反馈**：第 k 步的输运温度 = 第 k−1 步功率给出的缺陷处加权温度、功率与温度记录与按该步稳态重算的逐位相同、生成率按局域温度求值（电极项、陷阱项与 P_bb 都取位点处的温度）、开热反馈时 G = 0 仍逐位等于稳态结果；**导电簇与击穿判据**（ADR-054）：导电簇按间距 < r_c 连边、离电极 < r_c 接到电极（跨周期边界、断链），导体间隙；接电极的导电簇在生成率的电势里与电极等电位、悬空导电簇成员电势相同且总电荷守恒（ADR-056）、陷阱项把接电极的成员当作电极费米能级处占满的电子源、P_bb 取同一份电势；初始缺陷已贯穿时停在 t = 0，否则停在第一条贯穿记录上，击穿记录电流 = 内层输运电流 + G₀·V、之前细丝电流为 0，I-t 拟合只用内层输运电流；**续跑**（ADR-057）：状态存盘后逐位还原，从中途状态续跑与不停跑的轨迹只差舍入误差；致密团簇标志的几何团簇跨周期边界连边 |
| `test_thermal.py` | **TDDB 功率与热流**（式 T13/T14，ADR-036）：均匀功率 + 两端恒温的温度剖面逐点等于解析抛物线、零功率退回环境温度、点热源附近最热且电极恒温、功率分到网格后积分守恒；**能量守恒**：介质 + 两侧电极的放热 = U·I_陷阱（孤立陷阱、整体成簇、簇与簇外混合、强耦合分量被切开都严格成立，ADR-038）；零偏压热平衡占据下陷阱处的声子放热相消、只剩 R8 的吸热；输运温度是缺陷处按放热加权的温度 |
| `test_clusters.py` | **强耦合簇的纯函数**（ADR-022）：耦合强度与分簇、超限分量在最弱处切开、GTH 稳态分布与线性解一致且在速率跨 1e-20…1e14 时逐分量相对精度 ≤1e-12、簇内条件稳态在 4 态反例上给出精确 0.4、生成矩阵事件口径与 KMC 一致、无耦合时逐成员退回单陷阱公式（含贴正极 1e13 s⁻¹ 交换、净通量 0.4 s⁻¹ 的成员） |
| `test_numerics_convergence.py` | `kmc.py` 里各积分点数的收敛性依据 |
| `test_ensemble.py` | 平均确实是**算术平均**、每个种子确实换了一整套缺陷、M=1 退化为普通单点仿真、标准误 = σ/√M、**R1 分量在系综里零散布**（按构造与缺陷无关）、CSV 保留各实现原始值 |
| `test_sweep.py` / `test_visualization.py` | 扫描形状与单调性、缺陷实现复用、**高斯能级与界面富集在扫描里同样生效**（扫描用的缺陷与按同一配置同一种子直接生成的逐位相同）、CSV 往返、`--plot-prefix` 落两张图；绘图只返回 Figure、不落盘不 show，I-V/I-T 曲线的数据换算（A/cm²、°C）、TDDB 的 I-t 图（双对数阶梯 + 本底虚线 + 拟合直线、没有生成事件时只画本底）与应力图（击穿时终点标记、下图温度阶梯）、**给了标准误就画 J±SEM 误差棒**、**只画总电流**（不接受机制分解参数、图例无机制条目、行序在蓝色系里逐行变深且最浅档仍可见）；**叠层叠画**（ADR-016 ①②）：缺陷图两幅都叠层边界与界面位置、界面缺陷单独标色且柱子只统计体缺陷，能带图界面双值台阶（高度 = ΔE_C、两侧值就是各层 B_ℓ−eφ），单层两幅图逐位退回 |

---

## 已知局限与待确认项

- **ZrO₂/Al₂O₃/ZrO₂ 叠层结果是"明确标注近似的 v1"，不是定量材料预测**：
  `examples/stack_t2_*.yaml` 里 Al₂O₃ 的物性（κ=9、ε_opt=3.0、Eg=6.4 eV、
  χ=1.4 eV、m_cb=m_di=0.3 m_e、ℏω=90 meV）与界面缺陷面密度（Nₛ=1×10¹² cm⁻²）
  都是**占位值、待标定**（ADR-004/008）。带阶按电子亲和能对齐给出 ΔE_C=1.46 eV，
  界面固定面电荷取 0（ADR-007）。要得出定量结论必须先换成真实材料数据
- **PF 发射没有反向通道**（导带→陷阱俘获未在机制表里），零场下 R8 = ν·exp(−E_D/k_BT)
  仍非零，留下一个伪零偏压电流。实测比工作点电流小约 14 个数量级，已写进测试钉住量级
- **系综平均已落地**（`ensemble.py`）：`ensemble.count` 指定实现个数 M，每次运行随机
  抽取 M 个种子、各跑一次独立实现，取算术平均。改成 8 nm 默认之前的 50×50×9 nm（N=68）实测 M=5 在 2 V 给出 11.7% 的相对标准误；
  100×100 nm（N=240）下实现间 σ/mean = 16.9%（2 V）至 27.9%（4 V），M=5 对应
  8%–13%。侧向尺寸每翻 4 倍把 σ 砍掉约一半（σ ∝ 1/√N），均值在标准误内不变
- **陷阱间跃迁按近邻表存储**：R6/R7 的几何因子在同质材料里 e 折长度只有 r_D/2（参考
  预设下 0.083 nm），截断半径外的缺陷对速率在双精度下对事件表没有影响——实测与稠密
  (N,N) 实现给出逐位相同的 J 与机制分解。叠层里几何因子沿路径按材料分段衰减（ADR-010），
  截断半径按所有材料里最慢的衰减推、判据用被丢弃的总速率上界 < 机制表最小速率
  （`kmc._hop_cutoff_radius`，ADR-013/F05），不是写死的长度；单一材料叠层逐位退回闭式
- **TDDB 参考工况下生成贴着负极**（ADR-034、ADR-037、ADR-041）：生成被限制在离负极约 2.7 nm 内（1000 个空位时
  团簇最高点 2.35 nm），I-t 是"台阶 → 陡升 → 铺满后缓慢上升"而不是双对数直线，热失控不出现；前驱从
  近旁陷阱俘获的通道（ADR-039）按源陷阱积分算对之后（ADR-042），新空位仍落在负极侧 2 nm 以内，改变不了这一点。
  ADR-052 起生成改在离散前驱位点上求值，没有生成网格；泊松网格 80³ → 160³ 时致密层 ΣG 变 8–11%（前驱能级用的局域电势），
  ADR-041 及更早轨迹的后段时间标尺要重跑
- **内层平均场的簇–簇边界按常数速率处理**（ADR-038）：电流连续与式 T13 的能量守恒都精确，但零偏压下留一个
  小环流（Phase 4 终态 ≤ 1.8% 的 3 V 电功率），3 V 介质功率在个别终态偏负到 U·I 的 −26%。精确处理要按两个簇
  稳态分布的乘积平均，没做
- **电极→陷阱的势垒含陷阱自身电荷**（ADR-038 记录，未改）：WKB 势垒取横向平均剖面，同一陷阱的俘获（空、带 +e）
  与逃逸（占据、中性）走的势垒略有不同，单条电极通道的俘获/逃逸之比偏离 exp(−E/k_BT)。KMC 与平均场的强耦合簇
  都是这个口径，零偏压下在簇内留 1e-3 量级的非平衡；改它会改变稳态 KMC 的结果
- **电子跳跃轨迹输出未实现**——规格里标的是"可选"，且对应绘图项列在"后续可能添加"
- **侧向胞元尺寸的收敛性检查标准未定**。现在的 50 nm 与之前的 100 nm 都是按成本选的，
  不是收敛性结论；两者的均值在各自标准误内一致，但"多大才算收敛"仍未测。网格分辨率
  同理：无空间电荷时电势严格线性、结果与网格无关，网格只影响空间电荷的分辨能力
- **R7 是 Miller-Abrahams 经典跃迁模型**的简化实现，后续可能升级为量子模型；届时只需
  改 R7 内部实现与"R6/R7 合并"这一条
- 绘图文字用英文：matplotlib 在没有配置中文字体的机器上会把中文渲染成方框，这里不去
  改全局 rcParams，也就不依赖任何字体

---

## 目录结构

```
src/kmc_leakage/
  config.py            配置数据结构（全部 SI 单位）：materials / layers / interfaces + DielectricStack 查询接口
  io.py                YAML 读入 + 单位换算（唯一的换算边界）+ schema 校验、旧键拒绝
  defects.py           缺陷生成、界面富集轮廓、r_D、最小镜像距离、R6/R7 近邻表
  electrostatics.py    变 κ 泊松方程（FFT + Thomas，κ 按单元）、CIC 分配与插值、界面单侧场读取
  bands.py             能带剖面、费米分布、电极态密度
  rates/               R1–R8、WKB 与 TDDB 生成速率，每个函数一个纯函数
    wkb.py                 WKB 透射系数
    electrode_tunneling.py R1 + Richardson-Schottky 对照式
    elastic_capture.py     R2 / R3
    multiphonon.py         R4 / R5 + 声子线型
    trap_hopping.py        R6 / R7
    pf_emission.py         R8
    generation.py          TDDB 式 T1–T4：前驱单步俘获速率 R_1e、断键概率 P_bb、生成速率 G
  kmc.py               Gillespie 主循环、PF 收集概率 P_A、电流估计量、分层归因与验证状态
  clusters.py          强耦合陷阱簇：分簇、GTH 稳态分布、簇内条件稳态（纯函数）
  master_equation.py   小状态空间主方程稳态解（条件通量的独立基准，仅验证用）
  mean_field.py        簇级平均场内层求解器：占据不动点 + 与 KMC 同一估计量的电流（TDDB 内层默认路径）
  tddb.py              TDDB 瞬态驱动：外层生成事件 Gillespie（离散前驱位点）+ 内层平均场、增量静电耦合、热反馈、导电簇与贯穿通路击穿判据、细丝电流、I-t 幂律拟合、轨迹 CSV
  thermal.py           TDDB 式 T13/T14：逐事件放热 ΔE·R、功率分到网格、稳态热方程（直接调用泊松求解器）、输运用的加权温度
  sweep.py             I-V / I-T 扫描 + CSV 输出
  ensemble.py          多个缺陷实现的算术平均 + CSV 输出
  visualization.py     缺陷分布、能带结构、I-V / I-T 曲线、TDDB 的 I-t 与温度演化（只返回 Figure）
  __main__.py          命令行入口
tests/                 286 个测试
examples/              single_run.yaml、iv_sweep.yaml、stack_t2_{0,2,4}nm.yaml（t₂ 结构对照）、stack_t2_2nm_custom.yaml（不对称自定义算例）、zro2_tddb.yaml（TDDB 起点，50×50×8 nm 盒子）
docs/                  物理规格（改 rates/ 或 electrostatics.py 前先读）
```

## 可视化模块

`src/kmc_leakage/visualization.py` 与仿真主流程解耦：绘图函数只接受**已经算好的
数据**，返回 matplotlib 的 `Figure`，内部不调用 `plt.show()`、不保存文件、不触发
仿真——保存还是显示由调用方决定。图上文字一律用英文（不依赖中文字体）。

| 函数 | 画什么 | 输入 |
|---|---|---|
| `plot_defect_distribution` | 三维缺陷散点（按能级着色，画代表实现）+ 沿 z 的数密度直方图（汇总系综 M 个实现除以 M，横轴只画可采样窗口）；叠层时两幅都叠画层边界与界面位置、界面缺陷单独标色 | `DeviceConfig`、`DefectEnsemble`、可选 `realizations`（系综 M 个实现） |
| `plot_band_diagram` | 沿 z 的能带剖面（不画镜像力圆化）；叠层时导带/价带在界面处画真实台阶（界面双值，不连斜坡） | `BandProfile`，可选缺陷 z 与能级 |
| `plot_iv_curves` | I-V：横轴电压，一条曲线一个温度；只画总电流 | 电压 [V]、温度 [K]、J 矩阵 [A/m²] |
| `plot_it_curves` | I-T：横轴温度（°C），一条曲线一个电压；只画总电流 | 同上（同一份数据按电压切） |
| `plot_current_transient` | TDDB 恒压应力下的 J(t)，双对数：t = 0 本底画水平虚线，之后的状态画阶梯（最后一段到终点），可叠加幂律拟合直线并标出 n | `TDDBResult.time`、`current_density` [A/m²]、`end_time`，可选 (n, A, 取样时刻) |
| `plot_stress_transient` | 上图同 `plot_current_transient`，击穿时在终点画标记并写出时刻；下图是各状态的温度（°C），横轴共用。CLI `tddb --plot` 用它 | 同上，外加温度 [K]（`TDDBResult.effective_temperature`）与 `stop_reason` |

两条曲线函数吃的就是 `SweepResult` / `EnsembleSweepResult` 的
`voltages`、`temperatures`、`current_density`（矩阵形状 `(温度点数, 电压点数)`）：

```python
from kmc_leakage.ensemble import run_ensemble_sweep
from kmc_leakage.visualization import plot_iv_curves

result = run_ensemble_sweep(config)
figure = plot_iv_curves(
    result.voltages,
    result.temperatures,
    result.current_density,
)
figure.savefig("iv.png", dpi=150)
```

- 输入是内部 SI 单位，换算在绘图这个输出边界做：纵轴 J 换成 A/cm²、横轴温度换成 °C，
  与 CSV 输出同口径
- 纵轴取对数（漏电流跨几个数量级）
- **画 ±σ/√M 误差棒**（系综扫描时；`plot_iv_curves` / `plot_it_curves` 的可选参数
  `current_density_sem`，不传就不画，单种子扫描即如此）。数值同时走 CLI 摘要与 CSV 的
  `J_sem_A_cm2` 一列
- **只画总电流**，不画分机制的电流曲线；四项机制分解看 CSV
- 行（温度 / 电压）用 `Blues` 色系在 0.50–0.95 区间上按行数等分的深浅区分（浅 =
  低温/低压、深 = 高温/高压）。下限 0.50 是为了最浅那档在白底上仍有 2.1:1 以上的
  对比度；上限 0.95 是为了最深那档不糊成黑色。**行数超过 5 档时深浅分辨会变紧**
  ——默认 I-T 图是 11 个电压点，靠图例与"越深越高"的单调方向读，相邻两档本身分不开
- CLI 里 `sweep --plot-prefix scan` 就是这两个函数加一次 `savefig`，落
  `scan_iv.png` 与 `scan_it.png`，外加随机一个系综种子的缺陷分布图 `scan_defects.png`
  及其簇标注图 `scan_defects_clusters.png`
- **缺陷簇标注图**（`plot_defect_clusters`）：强耦合簇（`KMCSimulation.cluster_members`，
  按配置的单点工况识别）用缺陷本身的颜色区分——三维散点不按能级着色、不画 colorbar，
  非簇缺陷画黑色，同一个簇的成员同色（tab10 去掉与柱子同色的蓝、与黑易混的灰，8 色
  轮换），界面缺陷仍以菱形区分；深度剖面柱子与原图相同，底部每簇一行同色圆点标出成员
  的 z。簇常跨界面，所以不画 z 范围竖带
- **叠层的两处叠画**（ADR-016，函数签名不变，靠传进来的 `DeviceConfig` /
  `BandProfile` 自己认出层结构）：缺陷图在三维盒子上把每个界面画成一圈虚线、
  在深度剖面上画层边界竖线与层材料名，**界面缺陷单独标色**（深色菱形 + 界面处
  的个数标注）——它们按面密度 Nₛ 配置，与体缺陷的体密度不是一个量纲，所以直方图
  的柱子只统计体缺陷；能带图在界面处画**真实台阶**，同一个 z 上取两侧各一个值
  （`BandProfile.stepped`），不把台阶连成斜坡。单层没有内部界面，两幅图与叠层
  改造前逐位相同
