# 变更记录

本文件记录每次逻辑完整的修改：改了什么、为什么、涉及哪个 R 编号或物理文档、
pytest 是否通过。格式与约定见 `CLAUDE.md`。

## [未发布]

### 变更

- **删去 E_rel = S·ℏω₀，T3 乘 ½，E_rel1 取 1.0 eV**（新增 ADR-030，ADR-027 的对应写法与 ADR-029
  标 superseded；`PROJECT_CONTEXT_TDDB.md` 删去原式 T12、T3 改为 p₀ = ½·n·Valence·e·d、§2 第 5 条
  与参数表改写；`zro2_tddb.yaml` 的 E_rel1 0.04 → 1.0 eV；`config.py`、`rates/generation.py`、
  README 的注释；`test_generation.py` 改用 E_rel1 = 0.12 eV 直接传参，`test_config_io.py` 核对
  1.0 eV）。**原因**：按用户要求——E_rel = S·ℏω₀ 里的 S 与输运用的 Huang-Rhys 因子同名，写在文档里
  会让 E_rel1 被读成输运 S 的另一种写法；E_rel1 是第一次电子俘获导致 O-Zr-O 键断裂、O 空位形成的
  弛豫能，只用于空位生成。E_rel1 进 R_1e 的方式不变（E_rel1/ℏω₀ 作前驱线型 L(m) 的参数），代码
  逻辑不变，只把中间变量改名为 `lineshape_parameter`。T3 只在文档里出现，配置直接给 p₀，
  p₀ = 7.1 eÅ 不受影响。**核对**：E_rel1 = 1.0 eV 下 p₀ 推算仍是 7.145 eÅ；R4 要约 100 个声子项
  （80 项误差 1×10⁻⁴），Phase 3 调用 R_1e 取 n_phonon ≥ 100。涉及 T3/T10/T12，R1–R8 无改动。
  `pytest tests/ -v` 245 passed。

- **E_rel1 的含义写清：氧空位生成的弛豫能，与输运的 Huang-Rhys 因子 S 无关**（新增 ADR-029，
  ADR-028 的 E_rel1 部分标 superseded；`PROJECT_CONTEXT_TDDB.md` §2 新增第 5 条、§3 参数表；
  `zro2_tddb.yaml`、`config.TDDBConfig`、`rates/generation.py` 的注释；README；
  `test_config_io.py` 去掉"E_rel1 = 材料 S·ℏω₀"的断言，改为直接核对 0.04 eV）。
  **原因**：按用户说明，E_rel1（论文 1.5 eV 是 Si-O 键断裂的弛豫能，本项目是 ZrO₂ 中氧空位生成
  的弛豫能）只用于 TDDB 空位生成的计算——经 S₁ = E_rel1/ℏω₀ 进 R_1e 的多声子线型——而
  Phase 1 的配置注释、测试与 ADR-028 把它写成"与 O 空位预设相同"、等于输运侧 S·ℏω₀，把两个
  物理量绑在了一起。代码本来就各走各的（R_1e 不读材料的 S），计算结果不变；0.04 eV 保留为
  待定占位值。另测：0.04 eV 换成 1.5 eV 时 p₀ 推算仍是 7.145 eÅ（ADR-029）。不涉及 R 编号，
  涉及 T10/T12。`pytest tests/ -v` 245 passed。

- **TDDB Phase 1：生成速率纯函数与前驱参数配置**（新增 `rates/generation.py`：
  `single_electron_capture_rate`（式 T4）、`effective_activation_energy` / `bond_break_probability`
  （式 T2）、`bond_break_generation_rate`（式 T1）、`CaptureElectrode`；`config.TDDBConfig` 与
  `io._tddb` 新增 E_t1、E_rel1、p₀ 三个必填字段，`io.py` 新增 e·Å → C·m 换算系数
  `E_ANGSTROM`；`examples/zro2_tddb.yaml` 填入取值；新增 `tests/test_generation.py`（6 项），
  改写 `test_config_io.py` 的两项；新增 ADR-027/028；同步 README、`PROJECT_CONTEXT_TDDB.md`
  §3 参数表与 §8.1）。

  **做法**：R_1e = Σ_两侧电极 (R2 + R4)，直接调用引擎的同一组速率原语，前驱用自己的 E_t1
  （能级、r_D、R2 前置因子）与 S₁ = E_rel1/ℏω₀（式 T12，多声子线型），`multiphonon.py` 不改；
  R4 的声子项数由调用方给（S₁ = 1 时 12 项已收敛，S₁ 在 10–40 时要约 200 项）。P_bb 在有效势垒
  ≤ 0 时饱和为 1；G 的俘获速率作为参数传入，后期换 R_2e 只改调用方；场加速项的 k 取静态 κ
  （ADR-027）。前驱参数按用户确认取 E_t1 = 1.15 eV、E_rel1 = 0.04 eV（与 O 空位预设相同）；
  p₀ 按"3 V / 25 °C 下 t_BD ≈ 100 s"推算（暂借 N_T = 4×10¹⁹ cm⁻³、击穿约需 1.5×10³ 个空位），
  解出 7.145 eÅ，取 7.1 eÅ（ADR-028）。

  **结果**：p₀ = 7.1 eÅ 时有效势垒在 3 V 为 0.45 eV、3.38 V 降到 0，斜率 1.18 eV/V，25 °C 下
  P_bb 每伏变化约 20 个数量级。这个斜率由 E_A = 4 eV 决定、与 p₀ 怎么取无关，标定电压加速因子时
  先核对（ADR-028、§8.1）。R_1e 的空间平均几乎全部来自弹性通道 R2，量级由 E_t1 决定：3 V、
  25 °C 下 ⟨R_1e⟩ = 2.3×10⁶ s⁻¹。

  **验证**（§9 三条解析极限）：G 的零场极限 = R_1e·exp(−E_A/k_BT)；ln G 对 E 严格线性、斜率
  p₀·(2+κ)/(3k_BT)；R_1e 与引擎里同一参数陷阱的两侧 R2 + R4 之和一致（rtol 1e-12，正反偏压）。
  另测 P_bb 饱和、κ = 38 下每 1 eÅ 压低 0.5 eV。涉及 T1/T2/T4（T5–T12 经 R2/R4 复用），R1–R8
  无改动。`pytest tests/ -v`（`.venv`，Python 3.8）245 passed。

- **TDDB Phase 0：ZrO₂ 单层 TDDB 配置与小盒子稳态本底验证**（新增 `examples/zro2_tddb.yaml`；
  `config.py` 新增 `TDDBConfig` 与 `Config.tddb`，`io.py` 新增 `_tddb`，解析可选的顶层 `tddb`
  段；新增 `docs/adr/tddb-decisions.md`（ADR-024–026）；`tests/test_config_io.py` 新增 2 个测试；
  逐种子原始数据存入 `results/9.29/tddb_phase0_box20nm.csv` 与 `tddb_phase0_box50nm.csv`；同步
  README、`PROJECT_CONTEXT_TDDB.md` §8.3）。

  **做法**：材料（名为 `ZrO2`）、缺陷段、电极、接触势垒、层厚与 `single_run.yaml` 逐项相同，由
  测试保证；只把侧向尺寸改成 20×20 nm 小盒子（§6.3，10 个初始缺陷），横向格点取与默认步长
  1.5625 nm 最接近的 13（hx ≈ 1.538 nm，ADR-025），`ensemble.count = 30`（ADR-026）。`tddb` 段的
  字段在用到它的 Phase 逐个加入，现在只有 E_A = 4 eV（式 T2，`activation_energy_eV`，必须为正），
  p₀、N_T、E_t1/E_rel1、k_TH 等待定参数以注释占位；G 场加速项的 k 直接取材料 `kappa`（ADR-024）。

  **验证**（§10 Phase 0 判据）：小盒子 M=128、默认盒子 M=32，每个实现 20000 事件扫
  {2,3,4} V × {25,125} °C（口径同 `run_ensemble_sweep`，脚本并行跑种子）。六个工作点的系综均值
  之差都在合成标准误的 2 倍以内，比值 0.62–1.30；4 V 两点小盒子偏高 20–30%（1.4σ 与 1.7σ），
  这两点共用同一批实现，是否系统偏高要更大的 M 才能判断。3 V、25 °C 小盒子
  J = 6.66e-3 ± 0.88e-3 A/cm²，TAT 占 99.9%，R1 为 2e-12 A/cm²，I-t 从 TAT 本底起步。小盒子单个
  实现的散布 σ(log₁₀J) = 0.9–1.4（默认盒子 0.28–0.58），单条轨迹之间可差 5–6 个数量级；同一
  标准误下两种盒子的总机时相当。2 V 下两种盒子都有部分实现未通过验证（占据与 p̂ 偏差超限、
  计数交叉校验超差），3–4 V 下几乎全部通过。`plot --kind defects|bands` 在新配置上跑通；CLI `run`
  （新抽 30 个种子，3 V、25 °C）给出 J = 4.80e-3 ± 1.52e-3 A/cm²、TAT 占 99.9%，28/30 通过验证，
  未通过的两个是占据与 p̂ 偏差 0.114、0.120（门槛 0.1）。
  涉及 T2（E_A 配置项），R1–R8 无改动。`pytest tests/ -v`（`.venv`，Python 3.8）239 passed。

- **TDDB 阶段初始状态**：新增 `PROJECT_CONTEXT_TDDB.md`（TDDB 搭建规格，取代已删除的
  `CLAUDE_TDDB.md`）；`CLAUDE.md` 改为指向它，并补充输出风格、环境、单位约定与 TDDB 常用
  命令；原 `CLAUDE.md` 里的代码规范、测试要求、提交规范移入 `PROJECT_CONTEXT.md`。
  `kmc.py`（`r1_electrode_current`、`_blocking_layer_indices`）与 `rates/wkb.py`
  （`wkb_transmission`）精简 docstring，逻辑不变。`results/` 下旧的 stack_t2_* 扫描结果与
  t2_1nm 图删除，新结果按日期归档到 `results/9.28/`、`results/9.29/`，`results/t2_2nm.csv`
  更新。`examples/stack_t2_2nm.yaml` 保持预设值。原因：作为 TDDB 开发的起点。
  pytest：本次提交未运行。

- **簇标注图改为用缺陷本身的颜色区分簇**（`visualization.plot_defect_clusters`）：三维散点不再
  按能级着色、不画 colorbar，非簇缺陷画黑色，同簇成员同色，取消原先套在簇成员外的彩色空心圈；
  界面缺陷仍以菱形区分（图例改用黑色代理句柄）。簇配色取 tab10 去掉蓝（与深度剖面柱子同色、
  圆点看不见）和灰（与黑易混），8 色轮换。实现上给私有 `_draw_defect_distribution` 加可选的
  逐缺陷 `point_colors`，缺省时原图逐位不变。原因：一眼看出哪些缺陷同属一个簇。测试改为逐缺陷
  核对颜色（matplotlib 3.7 的三维散点出图后 `get_facecolor()` 是按深度排序且少一个的，改读原始
  输入颜色）。同步 README 与 docs/physics_model.md。pytest：test_visualization.py 全过；全量 235
  过、2 败（test_stack_examples 那两项，来自工作区里 examples/stack_t2_2nm.yaml 未提交的修改，
  与本改动无关）。

- **缺陷分布图每次出两张：原图 + 强耦合簇标注图**（新增 `visualization.plot_defect_clusters`、
  `KMCSimulation.cluster_members`）。`plot --kind defects` 另落 `<out 去扩展名>_clusters.png`，
  `sweep --plot-prefix` 另落 `<前缀>_defects_clusters.png`。簇 = ADR-022 的强耦合簇，按配置里的
  单点工况（simulation.voltage / temperature）只做 KMC 初始化识别、不跑事件；三维散点上每簇
  一圈同色空心圆，深度剖面底部每簇一行成员 z 圆点（簇常跨界面，z 范围竖带会盖住整层，弃用）。
  原图函数签名与输出不变，底图抽成私有 `_draw_defect_distribution` 共用。原因：直接看出簇在
  哪、跨不跨界面。同步 README 与 docs/physics_model.md"可视化接口"。pytest：新增单层/叠层簇标注、
  无簇退化三项通过；全量 235 过、2 败（test_stack_examples 那两项，来自工作区里
  examples/stack_t2_2nm.yaml 未提交的修改，与本改动无关）。

- **I-V / I-T 曲线图的误差棒两端加小横线**（`visualization._current_curves` 的 `capsize` 0 → 3）：
  原先只有竖线段，棒端位置不好读。端帽是额外的 Line2D、会混进 `axes.lines`，测试改为按
  `axes.containers` 逐条取主线/端帽/竖线段，并核对两个端帽恰在 J ± SEM。不涉及 R 编号。
  pytest：test_visualization.py 全过；全量 232 过、2 败（仍是 test_stack_examples 那两项，
  来自工作区里 examples/stack_t2_2nm.yaml 未提交的修改，与本改动无关）。

- **`sweep --plot-prefix` 另出一张缺陷分布图 `<前缀>_defects.png`**（`__main__._write_sweep_defect_plot`）：
  从系综种子里随机挑一个，按扫描同样的 `default_rng(seed)` 重建该实现的缺陷，复用
  `plot_defect_distribution`，单层、叠层都支持；所用 seed 打印在终端。原因：扫描完直接能看到
  实际参与计算的缺陷长什么样。不涉及 R 编号。pytest：新增 test_sweep.py 单层/叠层两项通过；
  全量 232 过、2 败（test_stack_examples 的两项，因为工作区里 examples/stack_t2_2nm.yaml
  有未提交的修改，与本改动无关）。

- **I-V / I-T 曲线图画 ±σ/√M 误差棒**（`visualization.plot_iv_curves` / `plot_it_curves`
  新增可选参数 `current_density_sem`，系综扫描的 `--plot-prefix` 传入
  `EnsembleSweepResult.current_density_sem`；单种子扫描不传、不画）。原因：让图上直接看出
  系综涨落。同步改 docs/physics_model.md"可视化接口"与 README 的对应口径；原测试
  "不带误差棒"改为核对棒端 = J ± SEM。pytest：test_visualization.py 19 项全过；全量
  230 过、2 挂（`test_stack_examples.py` 的 t₂ 系列两项，来自工作区里本次之前就未提交的
  `examples/stack_t2_2nm.yaml` 改动，与本次无关）
- **新增可选的快陷阱移出事件预算，测量窗口延长 80–900 倍**（`config.SimulationConfig` 新增
  `fast_trap_rate`、`io._simulation` 解析 `simulation.fast_trap_rate_s`；`kmc.py`：`_prepare`
  按全空组态速率分出快陷阱，主循环里快陷阱的块事件移出、占据取每步自洽的 p̂、快↔慢跃迁
  按 p̂ 加权并只改慢陷阱一端，`SimulationResult` 新增 `n_fast_traps`/`current_fast`/
  `fast_exchange` 与属性 `current_counts_total`，`_validate_result` 新增时间尺度分离判据；
  `ensemble.py` 的 ΔJ、`sweep.py` 的 CSV、`__main__.py` 的摘要改用 `current_counts_total`，
  删去随之无用的 `ensemble._r1`；新增 `tests/test_fast_traps.py` 6 个测试、`test_config_io.py`
  1 个；同步 README、`docs/glossary.md`、`docs/leakage_current_statistics.md` §3.3/§4、ADR-023）。

  **问题**：贴电极的陷阱与电极交换电子达 1e9–3e10 s⁻¹，4000 事件在 t2_2nm 叠层 125 °C/4.3 V
  下只推进 38 ns，事件预算几乎全花在它们的来回翻转上。

  **做法**：Jegert 等人 R_cutoff 方法的变体。全空组态下电极交换总速率 ≥ R_cut 的陷阱退出
  事件表，占据取与当前慢组态自洽的条件稳态 p̂（迭代容差 1e-3），估计量不变（快陷阱照样
  可以是强耦合簇成员）。省略该配置项时逐位退回原算法。

  **结果**：同一实现 4000 事件，125 °C/4.3 V 下 R_cut=1e10/1e9 把窗口从 38 ns 延长到 3.0/24 μs，
  25 °C/3.3 V 下从 0.54 μs 延长到 70/490 μs，电流与不启用相差 ≤0.23%；三陷阱小体系开启后电流
  仍等于主方程精确稳态（rtol 1e-4）。R_cut=1e8 时 125 °C/4.3 V 分离比 4.1，判未通过。无 R 编号
  （KMC 采样，ADR-021 F04 的延续）。`pytest tests/ -v`（`.venv`，Python 3.8）230 passed、2 failed；
  两个失败同上一条，来自工作区未提交的 `examples/stack_t2_2nm.yaml` 改动，与本次无关。

- **电流估计量对强耦合陷阱簇改用簇内条件稳态，消除 I-V 尖峰与跃迁供给通路的漏算**
  （新增 `clusters.py`：`coupling_strength`、`coupled_clusters`、`stationary_distribution_gth`、
  `cluster_generator`、`cluster_steady_state`；`kmc.py`：`_build_clusters`/`_cluster_plan` 在工作点
  初始化时分簇，`_rates` 拆出 `_path_barriers`/`_electrode_blocks` 以便按"缺陷×组态"行求值，
  新增 `_electrostatics_batch`、`_trap_flux`、`_cluster_flux`，主循环改调 `_trap_flux`；
  `master_equation.steady_state_flux` 的条件通量一栏改走同一估计量，`enumerate_occupations`
  移到 `clusters.py`；新增 `tests/test_clusters.py` 8 个测试、`tests/test_master_equation.py` 2 个；
  同步 README、`docs/glossary.md`、`docs/leakage_current_statistics.md` §6.5、ADR-022）。

  **问题**：单陷阱条件通量只把陷阱自身的占据解析平均掉，陷阱间跃迁的供给项用的是邻居的
  采样占据。t2_2nm 叠层 seed 1522848646 在 125 °C/4.3 V 得 1.12e-1 A/cm²，只换轨迹随机数就
  回落到 1.57e-2：阴极侧 #71 以 3e6 s⁻¹ 填充、以 3.4e7 s⁻¹ 跳给 #51、#51 以 1.2e8 s⁻¹ 放给
  阳极，而测量窗口只有 38 ns，#71 在窗口里填充 0 次（漏算）或 1 次（高估）。即使窗口无限长，
  单陷阱估计量也丢掉两者占据之间的关联：这一对偏低 9%，更强耦合的对上只剩精确值的 1–7%。

  **做法**：跃迁速率 ≥ 1e-3 × 两端电极交换速率较小者的陷阱对连成簇（上限 8 个陷阱），测量期
  每步对每个簇枚举 2^k 个组态、按各组态真实电荷分布重算簇成员速率，解簇内主方程稳态（GTH
  状态约化，逐分量相对精度），簇外占据作为环境。单陷阱仍走原公式、逐位不变。

  **结果**：上述两个尖峰点改后三条轨迹一致：125 °C/4.3 V 为 2.169e-2 / 2.170e-2 / 2.170e-2，
  100 °C/3.6 V 为 2.247e-3 / 2.242e-3 / 2.247e-3（旧代码 5.55e-3 / 1.86e-3 / 2.07e-3）；孤立强耦合
  对上 KMC 估计量 = 主方程精确稳态（rtol 1e-10）。代价：4000 事件单点从 27–28 s 增至 55–81 s
  （簇越大越慢）。无 R 编号（统计/估计量，ADR-021 F04 的延续）。
  `pytest tests/ -v`（`.venv`，Python 3.8）223 passed、2 failed；两个失败
  （`test_stack_examples.py::test_t2_series_fixes_total_thickness_and_keeps_zro2_symmetric`、
  `::test_t2_series_differs_only_in_layer_structure`）来自工作区里未提交的
  `examples/stack_t2_2nm.yaml` 改动（层厚 2.5/2.5/3.0、浓度 6e18），只读配置，与本次无关。

- **z 向网格加密：hz 从 0.2 nm 改为 0.1 nm（nz 从 40 改为 80）**（改 `examples/` 下全部 6 份配置的
  `grid.nz`，同步 `docs/physics_model.md`"已确认"一节的默认网格、README 的配置示例与层厚约束
  说明）。按用户要求执行。层厚现在只需是 0.1 nm 的整数倍，上一条里"t₂ 取奇数对不上网格"的
  限制不再成立（系列仍保持 {0,2,4}）；`stack_t2_2nm_custom.yaml` 的 2.5/2/3.5 nm 现在已对齐，
  可以直接运行。CIC 自作用势 ∝ 1/h 会随 hz 变化，但 KMC 引擎按静电耦合矩阵对角元整项扣除，
  不需要额外处理。机时方面，单实现静电预计算从 0.24 s 增加到 0.50 s，每个事件的耗时在噪声
  范围内不变（2000 事件实测）。无 R 编号。`pytest tests/ -v` 215 passed（在只含本次改动的
  暂存区快照上运行；工作区里 `stack_t2_2nm.yaml` 另有未提交的用户改动，未纳入本次提交）。

- **默认器件总厚改为 8 nm、网格 nz=40（hz=0.2 nm）；t₂ 系列同步改为总厚 8 nm、t₂ ∈ {0,2,4} nm**
  （改 `examples/single_run.yaml`、`iv_sweep.yaml`、`stack_t2_0nm.yaml`；`stack_t2_2nm.yaml`
  改为 3/2/3 nm，缺陷与扫描参数回到系列口径；`stack_t2_3nm.yaml` 改名为 `stack_t2_4nm.yaml`，
  层厚 2/4/2 nm；删除 `stack_t2_1nm.yaml`）。按用户要求调整默认器件。t₂ 取奇数时
  t₁=t₃=3.5、2.5 nm 不是 0.2 nm 的整数倍，界面对不上网格（ADR-011），所以系列只取偶数 t₂。
  原 `stack_t2_2nm.yaml` 的自定义版本（1e19 cm⁻³、富集因子 10、阴极侧 2.5 nm 不对称等）
  另存为 `examples/stack_t2_2nm_custom.yaml`，不纳入系列测试。它的层厚目前不是 0.2 nm 的
  整数倍，运行前需要调整。缺陷数从 68 变为 60。
  **顺带修复界面节点归属的浮点舍入问题**（改 `config.py::DielectricStack.layer_indices_of` /
  `layer_index_of`、`bands.py::BandProfile.stepped`）：在 8 nm/nz=40 网格下，界面节点坐标
  k·hz 会比层厚累加值小 1 ulp（例如 3e-9 < 3.0000000000000004e-9），节点因此落回下侧层。
  `kmc._band_on_path_steps` 默认界面节点取的是上侧 E_C，会把 ΔE_C 叠加两次，导致分段 WKB 的
  势垒剖面出错。修法是比较时加 Lz·1e-9 的容差。9 nm/nz=36 下舍入方向碰巧有利，所以一直没有
  暴露，结果不受影响。涉及 ADR-011/012、`docs/physics_model.md` 的器件默认值（已同步），
  README 同步命令、配置示例与 t₂ 系列说明；9 nm 下实测的耗时与系综标准误数字保留，并注明
  是 9 nm 实测。新增 `test_stack_physics.py::test_interface_nodes_belong_to_upper_layer_despite_rounding`
  （修复前 2/4/2 用例失败）。`pytest tests/ -v` 215 passed。原先失败的
  `test_t2_series_differs_only_in_layer_structure` 随 t₂=2 配置回到系列口径后通过。

- **I-V / I-T 曲线图只画总电流，不再画 TAT 与 PF 分机制曲线**（改 `visualization.py`：
  `plot_iv_curves` / `plot_it_curves` 去掉 `by_mechanism` 参数，删除 `_Component`、
  `_components`、`_COMPONENT_STYLES`、`_log_safe` 与两段式图例；行序仍按 Blues 色系深浅区分。
  `__main__._write_sweep_plots` 同步去掉该参数。同步 `docs/physics_model.md`"可视化接口"
  第 3 项与 `README.md` 的出图说明）。按用户要求简化曲线图；机制分解的数值出口（CLI 摘要、
  CSV）不变。无 R 编号。`tests/test_visualization.py` 删掉分机制相关测试，新增"只画总电流"
  与"蓝色系逐行变深"两项；`tests/test_sweep.py` 同步调用签名。`pytest tests/ -v`
  213 passed、1 failed，失败为已知的
  `test_stack_examples.py::test_t2_series_differs_only_in_layer_structure`，与本次无关。

- **pytest 从可选依赖 `[test]` 移入基础依赖，锁 `pytest>=7.0,<8.4`**（改 `pyproject.toml`；
  同步 `README.md` 环境一节与 `PROJECT_CONTEXT.md`）。原先 `pip install -e .` 不装 pytest，
  项目 `.venv` 里一直没有（review 2026-09-15 记录过"未能执行"），"合并前必须跑通
  `pytest tests/ -v`"这条要求在项目环境里落不了地。上界同 numpy/scipy 的理由：pytest 8.4
  起不再支持 Python 3.8。无 R 编号；项目 `.venv`（Python 3.8.4、numpy 1.24.4、pytest 8.3.5）
  下 `pytest tests/ -v` 224 passed、1 failed，失败为已知的
  `test_stack_examples.py::test_t2_series_differs_only_in_layer_structure`（见下条），与本次无关；
  上条在 numpy 2.x 下失败的 `test_expected_depth_density_conserves_count_and_tracks_profile`
  在项目环境下通过。

- **验证状态新增"批次间电流一致"硬门槛（批次均值法）**（改 `kmc.py`：新增纯函数
  `batch_relative_error`，主循环把测量窗口按事件数等分成 10 批、分批累计缺陷通量与时长，
  `_validate_result` 在相对标准误 > 0.05 时判未通过；新增 `tests/test_current_validation.py`
  里 4 个测试；同步 `README.md` 与 `docs/glossary.md` 的验证判据描述）。

  **问题**：`t2_2nm` 叠层 seed 243366264 的 I-V 扫描在 2.0/3.3/3.7/4.0 V 出现电流尖峰
  （如 3.3 V 为 5.6e-4 A/cm²，相邻点约 2e-5），却全部判"通过验证"。诊断表明尖峰来自
  采样不足：阴极→缺陷 65→97→界面缺陷 24→阳极这条陷阱间跃迁供给通路，只在 97 被占据时
  导电，而 4000 事件的测量窗口只有约 0.3 μs（事件预算被与电流无关的快速电极往返占掉），
  窗口里该通路被填充 0 次或 1 次——0 次时漏算，1 次时高估。固定缺陷、只换轨迹随机数，
  3.3 V 在 5.7e-4 与 2.3e-5 之间跳变。条件稳态估计量只解析平均了单个陷阱自身的占据，
  跃迁供给依赖 KMC 采到的邻居占据；原判据里占据偏差取全部缺陷的平均值、一两个缺陷偏离
  被平均掉，净计数校验又因阳极事件不足 50 次被跳过，所以拦不住。

  **标定**：正常点（叠层与单层默认配置各若干）相对标准误 ≤ 0.001，尖峰点 0.19–0.26，
  阈值取 0.05。**局限**：窗口里从未被填充的情况没有迹象，这一条查不出来（那些点偏低），
  需要延长物理时长或减少无关快事件，另行处理。无 R 编号（统计/估计量，ADR-021 F04）；
  `pytest tests/ -v`（Python 3.14 环境）223 passed、2 failed，两个失败在本次改动前即存在、
  与本次无关：`test_defects.py::test_expected_depth_density_conserves_count_and_tracks_profile`
  （该环境 numpy 为 2.x，与项目要求的 ≤1.24 不符）、`test_stack_examples.py::
  test_t2_series_differs_only_in_layer_structure`（`examples/stack_t2_2nm.yaml` 的缺陷浓度与
  空间分布与 t₂ 系列其余三份不一致）。

- **缺陷深度直方图取消解析轮廓叠画与图例**（改 `visualization.plot_defect_distribution`：
  删掉 `expected_depth_density` 叠画的黑色虚线、柱子的 `sampled (M=)` 标签与 `axes.legend`，
  只留柱状图；相应更新单层与汇总两个测试的断言）。图面回归纯直方图。`expected_depth_density`
  与其测试保留（不再被绘图调用，作为采样口径的独立解析对照仍可用）。无 R 编号；
  `pytest tests/ -v` 221 passed。

- **缺陷深度直方图横轴改为只画可采样窗口 [standoff, Lz−standoff]，图上不留空**
  （改 `visualization.plot_defect_distribution`：分箱范围与 xlim 从 [0,Lz] 收到
  [standoff, Lz−standoff]，standoff 取 `defects._global_standoff`；相应改两个测试的 bin
  宽按窗口宽算）。两端 standoff 内按数值稳定性要求本就不放缺陷，画进来只是空白；裁掉后
  边缘那根柱子恰好落在富集峰所在的可放置平面上、不再被死区体积摊低。无 R 编号；
  `pytest tests/ -v` 221 passed。

- **缺陷深度直方图汇总系综 M 个实现 + 叠画解析轮廓，修正单实现看不出富集的显示问题**
  （新增 `defects.expected_depth_density`，改 `visualization.plot_defect_distribution`
  （加可选 `realizations` 参数）与 `__main__.py` 的 `plot` 命令；新增
  `tests/test_defects.py::test_expected_depth_density_conserves_count_and_tracks_profile`、
  `tests/test_visualization.py::test_defect_plot_pools_realizations_and_overlays_expected_profile`）。

  **问题**：`plot --kind defects` 原先只画一个缺陷实现，默认 N≈68、每 bin 才几个缺陷，
  单实现的深度直方图基本是泊松噪声（每 bin 计数 0～13 乱跳），界面富集淹没在噪声里看不出来；
  且边缘 bin 因 standoff 死区体积按构造为空、密度却除以整格体积，最左/最右柱子恒偏低，
  视觉上反像"界面处更少"。

  **改法**：① `plot` 生成系综全部 M 个实现，三维散点仍画代表实现，深度直方图**汇总 M 个再除以 M**
  （每实现平均密度，σ∝1/√(MN)），压低噪声；② 新增 `expected_depth_density` 按 `_sample_z` 同口径
  复算期望深度密度 ρ(z)（数量守恒 ∫ρ dz=浓度·层厚、standoff 死区 ρ=0），直方图**叠画这条解析轮廓**——
  贴合即证明采样器照轮廓在放缺陷、偏离纯是有限 M 涨落，死区里轮廓为 0 也解释了边缘柱为何偏低。
  `realizations` 缺省时退回单实现（M=1），既有单实现调用逐位不变。无 R 编号；`pytest tests/ -v` 221 passed。

- **界面富集轮廓锚定到可采样窗口，修正 λ ≲ min_separation 时富集在直方图里
  消失**（改 `src/kmc_leakage/defects.py::_sample_z` 及调用点，新增
  `tests/test_defects.py::test_interface_enrichment_survives_standoff_shorter_than_lambda`；
  同步 `config.py`、`docs/physics_model.md` 的轮廓说明）。

  **问题**：`n(z) ∝ 1+(g-1)[exp(-z/λ)+exp(-(Lz-z)/λ)]` 的峰在界面 z=0/Lz 上，但体缺陷
  采样区间是层边界各让出 standoff（=min_separation，电极/内部界面避让的数值稳定性约束）
  后的窗口。λ ≲ standoff 时富集峰整个落进被禁区，采样窗口上近乎平坦——加大 g、减小 λ
  都救不回来（减小 λ 反而把峰更往被禁区推）。默认 λ=1.0/standoff=0.5 只剩 ~1.3× 边缘对比，
  再被 N≈68 的泊松噪声淹没。

  **改法**：`_sample_z` 把轮廓锚定到可采样窗口 [z_min, z_max]（z 相对窗口起点、Lz 取窗口
  厚度），富集峰因此落在距界面 standoff 处、即最靠近界面的可放置平面上。实测同参数下
  边缘/中心密度比从 1.34→1.82（λ=1.0）、从 0.56→1.24（λ=0.2），g=20/λ=0.2 从 0.79→4.0，
  富集重新可见且随 g 单调增强。缺陷总数仍 = 浓度 × 体积（轮廓只改形状），min_separation=0
  时逐位退回旧行为（既有富集测试不变）。无 R 编号；`pytest tests/ -v` 219 passed。

- **README: 新增"叠层结构与不同层厚的设置"小节**（只改 `README.md`，无代码改动）。
  把原先散落在"快速开始"与"配置文件"里的叠层用法归拢成一节：如何用 `layers` 的
  `thickness_nm` 设不同层厚（以 `examples/stack_t2_{0,1,2,3}nm.yaml` 的 t₁=t₃=(9−t₂)/2
  对称分配为例）、两条硬约束（层厚须为 hz=Lz/nz 整数倍、改 Lz 要同步改 grid.nz）、单层
  退化写法，并给出对同一份配置跑 I-V/I-T 扫描（`sweep --plot-prefix`）、缺陷分布图与能带图
  （`plot --kind defects|bands`）的完整命令。纯文档，无 R 编号、无测试改动。

- **t₂ 结构对照配置 + 叠层可视化 + 退化用例全套 + 首次叠层 J（Phase 6 = ADR-006/016/018/019）**
  （新增 `examples/stack_t2_{0,1,2,3}nm.yaml`、`tests/test_stack_examples.py`；
  改 `src/kmc_leakage/bands.py`、`visualization.py`、`config.py`、`tests/test_visualization.py`；
  同步 `README.md`）

  **背景**：Phase 5 闭合了 F03（PF 收集概率）与 F04（条件通量独立验证），叠层电流的口径已经具备
  交付条件。Phase 6 补齐结构对照的四份配置、ADR-016 的两项可视化与 ADR-018 的退化判据全套，并首次
  跑出 ZrO₂/Al₂O₃/ZrO₂ 的 I-V/I-T 与分解统计。

  **四份 t₂ 配置（ADR-006/019）**：固定总厚 9 nm、ZrO₂ 两侧对称分配 t₁=t₃=(9−t₂)/2，
  t₂ ∈ {0,1,2,3} nm 各一份（层厚 4.5/0/4.5、4/1/4、3.5/2/3.5、3/3/3），四份除层结构外逐项相同
  （材料、体缺陷、器件、电极、扫描范围、事件数、M=5）。层厚都是 hz=0.25 nm 的整数倍（nz=36），
  界面对齐校验通过。t₂=0 是单层退化点：无 Al₂O₃、无 ZA 界面、界面缺陷同时关闭，写成只含一个
  9 nm ZrO₂ 层（io 拒绝 0 厚层）。t₂>0 的三份：Al₂O₃ 体内无缺陷（ADR-005），两个界面对称、各
  Nₛ=1×10¹² cm⁻²（ADR-008，实测每界面 25 个），σ_interface=0（ADR-007），带阶按电子亲和能对齐
  给出 ΔE_C=1.46 eV（ADR-004）。**Al₂O₃ 物性与 Nₛ 都是占位值、配置里已标注"占位/待标定"**。
  不新增扫描机制，四份各跑一次现有 `sweep` 命令再比对 CSV。

  **可视化 ①②（ADR-016，函数签名不变）**：`plot_defect_distribution` 在三维盒子上把每个界面画成
  一圈虚线、在深度剖面上画层边界竖线与层材料名，**界面缺陷单独标色**（深色菱形 + 界面处个数标注）；
  界面缺陷按面密度配置、与体密度不同量纲，所以直方图的柱子只统计体缺陷。`plot_band_diagram` 在界面处
  画**真实台阶**：`bands.band_profile` 另算出每个界面下侧的 E_C/E_V（新增 `BandProfile.interface_z`/
  `E_C_lower`/`E_V_lower` 与 `stepped` 属性），同一个 z 上存两侧各一个值，台阶不被连成斜坡。
  `config.py` 相应加 `stack.layer_band_gap`。不做 J–t₂ 曲线、分层电势/电场图、界面电荷积累图。

  **退化用例全套（ADR-018）**：新增 `tests/test_stack_examples.py`（10 例）。t₂=0 与单层基准
  `examples/single_run.yaml` 是同一个器件（材料名之外逐项相同），且数值逐位退回：同种子生成的缺陷
  位置/能级逐位相同、固定组态下完整速率表与 R1 两分量、φ(z)、E_C(z) 全部 rtol ≤ 1e-12，同种子完整
  轨迹的 J 与机制分解 rtol ≤ 1e-9；无阻挡层 → P_A ≡ 1。三层同材料 + ΔE_C=0 的场量（φ、E_C(z)）同样
  rtol ≤ 1e-12 退回单层（速率表与轨迹已在 `test_stack_rates.py`）。可视化侧的退化：单层两幅图没有
  界面双值点、没有层边界/层名/界面计数，与叠层改造前逐位相同。

  **首次叠层 J（M=5、4000 事件/点、5 电压 × 3 温度）**：3 V / 25 °C 下

  | t₂ | J [A/cm²] | 相对标准误 | TAT 占比 | PF 占比 | 阴极侧 P_A | Al₂O₃ 分压 |
  |---|---|---|---|---|---|---|
  | 0 nm | 3.79e-04 | 35% | 99.76% | 0.24% | — | — |
  | 1 nm | 1.11e-05 | 53% | ~100% | 7.8e-5 % | 4.6e-3 | 1.04 V |
  | 2 nm | 2.25e-06 | 42% | ~100% | 2.9e-7 % | 3.2e-4 | 1.64 V |
  | 3 nm | 3.68e-07 | 42% | ~100% | 6.5e-8 % | 6.0e-5 | 2.04 V |

  J 随 t₂ 单调下降（2.5 V 以上各温度都单调），3 V 下每加 1 nm Al₂O₃ 压低 5–34 倍；I-T 侧四份都随温度
  上升。机制上叠层里 TAT 近乎独占，**PF 被 F03 的收集概率压掉 3–6 个量级**（阴极侧陷阱穿阻挡层，
  阳极侧 P_A=1 但那边陷阱与阳极近乎平衡）；R1 两项仍比总电流低 10–14 个量级。by_layer 显示电流几乎
  全部挂在阴极侧 ZrO₂（`layer_0`）——贡献最大的是 z≈2 nm 的少数几个陷阱，`layer_2` 数值上为零
  （阳极侧陷阱与阳极近乎平衡、净通量抵消），t₂=3 nm 时 `interface_0` 占到 13%。
  **未通过验证的工作点如实记入**：t₂=0/1/2/3 四份分别有 71/75、74/75、60/75、47/75 个实现通过验证，
  失败全部落在低偏压（2–2.5 V）与 t₂ 较大的点上，唯一触发的判据是**净计数交叉校验**
  （ΔJ/J≈1，即 J_数事件+R1 在噪声里本身≈0）。这些点不作为可信稳态电流交付。**占位参数下这是
  "明确标注近似的 v1"，不是定量材料预测。**

  **涉及编号**：不改任何速率公式（`rates/` 未动）；R8 侧只是把 Phase 5 的 P_A 放到真实叠层上量化，
  统计侧沿用 by_layer 与验证状态。文档侧同步 `README.md`（四份示例配置、叠层出图命令、两个绘图函数的
  叠层叠画、测试计数 205→218、新增 `test_stack_examples.py` 行、已知局限里新增"叠层结果是占位参数下
  的 v1"一条）。

  **pytest**：`pytest tests/ -v` 全绿，218 passed（205 + `test_stack_examples.py` 10 + `test_visualization.py`
  新增 3）。

- **PF 收集概率（F03）+ 条件通量独立验证（F04）+ 分层归因（Phase 5 = ADR-009/014/017，按 ADR-021 F03/F04 修订）**
  （改 `src/kmc_leakage/kmc.py`、`config.py`、`sweep.py`、`ensemble.py`、`__main__.py`；
  新增 `src/kmc_leakage/master_equation.py`、`tests/test_pf_collection.py`、`tests/test_current_validation.py`、
  `tests/test_master_equation.py`；改 `tests/test_stack_rates.py` 一例；订正 `docs/glossary.md`、`README.md`）

  **背景**：Phase 4 让 ZAZ 端到端跑通，但条件通量还是"局部稳态近似的候选估计量"、PF 发射的电子按
  "立即被阳极收集"处理。Phase 5 闭合这两条，让叠层电流"可信"，为 Phase 6 首次叠层 J 铺路。设计定义
  按 ADR-021 先冻结再落数值。

  **F03 —— PF 收集概率 P_A（ADR-021 F03 覆盖 ADR-009 矩形示例、采纳 ADR-017 §1 入射能量）**：
  `R8_eff = P_A·R8`。只有与收集电极之间隔着阻挡层（导带基准高于两侧的内部层，即 Al₂O₃）的陷阱用
  P_A，同侧立即收集 P_A=1（位置感知，按缺陷宿主层判定，界面缺陷按势阱侧）。P_A = min(1, T_WKB +
  exp(−ΔE_C/k_BT))，T_WKB 在**带偏压**的 Al₂O₃ 梯形/三角势垒 `V(x)=ΔE_C−q·F_Al·x` 上做，
  `F_Al=Δφ_Al/t_Al`、`Δφ_Al=(U+V_bi)·(t_Al/κ_Al)/Σ(t_ℓ/κ_ℓ)`，入射能量取热化到阻挡层入口处势阱侧
  局部导带底（ADR-017 §1）。与 R1 同口径：零空间电荷的静态叠层能带上算、不随 KMC 组态重算，`_prepare`
  按缺陷预计算一次（`_compute_pf_collection`、`pf_collection_transmission`）。收集方向随驱动电势
  U+V_bi 定（>0 阳极、<0 阴极）；非正向工况标"未通过验证"。挡回原陷阱是空事件、不单列，直接把 R8
  换成 R8_eff 进事件表与 `_conditional_flux`。R8_eff 沿用 Phase 3 的界面单侧场（ADR-017 §2，不另取场）。
  单层无阻挡层 → P_A≡1、R8_eff=R8，逐位退回原路径。

  **F04 —— 条件通量独立验证（ADR-021 F04 覆盖 ADR-014"自动作为主估计量"）**：每个工作点算三个估计量
  （条件通量 `current`、瞬时通量时间积分 `current_direct`、阳极/阴极净计数 `current_counts`/
  `current_counts_cathode`+R1），并核对全局电荷守恒 `(I_C−I_A)Δt=q·ΔN_inside`（`delta_n_inside`，
  逐事件记账恒等式）。单点带**验证状态** `validated`+`validation_notes`：硬门槛用可靠判据（正向收集、
  测量窗口充分、KMC 占据与条件稳态 p̂ 接近、机制闭合）；净计数交叉校验噪声大，只有阳极事件够多才纳入
  门槛（否则作诊断记录，不判失败）。系综层面给出成对差 `ΔJ=J_cond−(J_counts+R1)` 的跨种子均值/标准误
  与逐点验证计数。新增 `master_equation.py`：对小体系枚举 2^N 组态、按 KMC 事件表同口径组装生成矩阵、
  解稳态 π，算真实稳态阳极通量作条件通量估计量的**独立算法基准**（`test_master_equation.py` 复现 review
  的 25% 反例与单陷阱一致极限），仅验证用、不进生产逐点。

  **分层归因 by_layer（ADR-014）**：把逐缺陷通量 `defect_flux` 按 layer_id / interface_id 分组得到端口
  电流贡献（两界面各一栏、界面缺陷只计一次、R1 单列），闭合 `Σ_{ℓ,Γ} I_{ℓ,Γ}=I_TAT+I_PF`、
  `I_total=Σ I_{ℓ,Γ}+I_R1`（`by_layer_attribution`）。从单点贯通到 sweep / ensemble / CSV，另附三估计量、
  ΔN_inside、测量窗口、验证状态等诊断列。through_plane 截面通量本轮不做。

  **数值影响**：**单层路径逐位不变**——用 `git worktree` 拉改动前 HEAD 跑同一份单层配置（seed 固定、
  4000 事件），J、四项机制分解、by_defect、elapsed 全部逐位相同（20 位有效数字）：P_A≡1、F04 只加纯
  观测量、不触碰 RNG 流。ZAZ 端到端跑通并给出 by_layer（层0/层2/界面0/界面1 各一栏）与验证状态。

  **涉及编号**：R8（折进 P_A → R8_eff）；统计/估计量（三估计量、全局守恒、验证状态、分层归因）。文档侧
  订正 `docs/glossary.md`（P_c→P_A 带偏压势垒、条件通量估计量改为需独立验证、新增验证状态/全局守恒/主
  方程条目）、`README.md`（三估计量与验证、by_layer、测试计数 182→205、三份新测试文件行、`master_equation.py`
  目录项）。

  **pytest**：`pytest tests/ -v` 全绿，205 passed（182 + `test_pf_collection.py` 9 + `test_current_validation.py`
  9 + `test_master_equation.py` 5；`test_stack_rates.py` 的 R8 一例改为期望 R8_eff=P_A·R8）。Phase 5 不打 tag、
  不单独 commit（ZAZ 叠层全部修改完成后再整体提交）。

- **分段 WKB + 按材料取速率参数 + 分段 G_ab + 近邻截断重推（Phase 4 = ADR-012 + ADR-010 + ADR-013，并入 ADR-021 F02、review F05/F08）**
  （改 `src/kmc_leakage/rates/wkb.py`、`rates/trap_hopping.py`、`rates/__init__.py`、`config.py`、`kmc.py`；
  新增 `tests/test_stack_rates.py`、翻转 `tests/test_stack_schema.py` 的抛错断言、扩 `tests/test_richardson.py`；
  订正 `docs/rate_equations.md`、`docs/physics_model.md`、同步 `README.md`）

  **背景**：Phase 3 交付了变 κ 泊松、分段能带台阶、界面单侧场，但 KMC 运行仍在 `kmc._prepare`
  读 `stack.single_material`（声子/R1 参数）处抛 `NotImplementedError`。Phase 4 把速率侧从单一材料
  推广到叠层：分段 WKB、R2–R5 按材料/物种取参数、R7 新的分段几何因子、近邻截断按所有材料重推，
  从而让 ZrO₂/Al₂O₃/ZrO₂ 端到端跑通。"可信稳态电流"（F03 PF 收集势垒、F04 条件通量独立验证）属
  Phase 5，不在本轮。

  **分段 WKB（ADR-012）**：`wkb_transmission` 的 `m_di` 从标量放宽为可沿积分轴逐样点给出的数组
  （路径传播质量）。`kmc._band_on_path_steps` 用**界面双值采样**把节点导带底插到 WKB 路径上——在每个
  界面节点前补一个下侧值 `band_nodes[k]−ΔE_C`（同一 z 的双值），台阶不被插成斜坡。关键：界面双值只改
  **采样**、不改**积分网格**（路径样点数固定），所以同材料多层与单层的 WKB 积分逐位一致（review F06：
  把界面节点加进积分网格才会破坏同质退化）。无内部界面时逐位退回 `np.interp(path, z_nodes, band)`。
  电极–陷阱路径电势沿用横向平均电势 φ̄(z)，作 v1 显式标注的简化。R1（`r1_electrode_current`）同样改分段
  势垒 + 分段 m_di，势垒顶 `np.max` 叠层下自动落到 Al₂O₃ 内。

  **R2–R5/R8 按材料/物种取参数（review F08）**：`kmc._prepare`/`_rates` 不再读 `single_material`，改用
  `DefectEnsemble` 上已备好的逐缺陷数组。区分两类质量：R2/R3 前置因子、R4/R5 的 c_MP 与 r_D、R8 的
  ε_opt/ν 用**物种质量/物性**（缺陷自身的 `m_di`/`band_gap`/`eps_opt`/`attempt_frequency`）；WKB 透射用
  **路径传播质量** m_di(s)（沿路径按所在层取，`stack.m_di_at`）。声子权重/偏移由 `(_N_PHONON,)` 扩成逐
  缺陷 `(N, _N_PHONON)`。

  **R7 新几何因子 G_ab（ADR-010，并入 ADR-021 F02）**：`rates.segmented_overlap_factor` 沿 a→b 直线按
  所在层材料精确分段积分 `G_ab = exp[−∫(α_a+α_b)ds]`，α 用路径所在层的 m_di（不是物种质量）。
  `kmc._geometric_overlap` 按 `stack.n_materials` 选路径：单一材料用闭式 `r6_elastic_hop`（含单层与三层
  同材料，**逐位退回旧口径**），多材料用 `ν_ab·G_ab`、ν_ab=√(ν_a·ν_b)。**只换几何重叠因子、不动事件能量
  差**：Θ 里仍是 F02 的多体差 ΔH=(E_b−E_a)−U_ab（ADR-021 F02 覆盖 ADR-010 的能量处理），正反向细致
  平衡测试继续过。

  **近邻截断重推（ADR-013 / review F05）**：`kmc._hop_cutoff_radius` 的 α_min 从"1/max(r_D)（单材料）"改为
  **叠层所有材料里最慢的衰减**（每个"缺陷×层"组合 α=√(2·m_di(层)·[B_层−零场陷阱能级]₊)/ℏ 里最小的正
  值——Al₂O₃ 低隧穿质量压过高势垒、衰减更慢）；判据从"单对速率 < 固定容差 1e-24"改为"**被丢弃的总速率
  上界 < 机制表最小速率**"：N_out·ν_max·exp(−2·α_min·r_cut) < 零场 R8_min，据此反推容差再调
  `hop_cutoff_radius`。删掉写死的 `_HOP_CUTOFF_TOLERANCE`。存在零衰减区（α_min=0）时退回列全部有序对。

  **翻转抛错断言 + 端到端**：`test_zaz_run_raises_until_layered_physics` 从"断言抛 NotImplementedError"
  改为 `test_zaz_run_end_to_end_and_decomposes`（断言跑通、机制分解之和 = 总电流、按缺陷分解 + R1 =
  总电流、TAT/PF 都出现）。新增 `tests/test_stack_rates.py`（10 例）：分段 m_di 的 WKB、界面双值台阶不插
  斜坡、R1 势垒顶落在 Al₂O₃、分段 G_ab 同质退回闭式 + 跨层用 Al₂O₃ 低质量（F08，比"全 ZrO₂"闭式大一个
  量级以上）、ZAZ 里 R7 正反细致平衡、截断按最慢材料推、R8 逐缺陷取 ε_opt/ν、三层同材料+ΔE_C=0 的速率
  表（rtol ≤ 1e-12）与轨迹（rtol ≤ 1e-9）退回单层（ADR-018）。`test_richardson.py` 加叠层 E_max 参考的
  R1 供给函数导出式对照（ADR-012），单层 m_e/m_cb 比值测试保留为退化用例。

  **数值影响**：**单层路径逐位不变**——用 `git worktree` 拉出改动前 HEAD 跑同一份单层配置（50×50×9 nm、
  3×10¹⁸、高斯能级、界面富集、seed 固定、4000 事件），J、四项机制分解、elapsed 全部逐位相同（20 位有效
  数字）。分段 WKB 无界面时退回 `np.interp`、逐缺陷数组在单材料下全相等（与标量逐位一致）、G_ab 走闭式
  分支、截断的 α_min 在单材料下仍是 1/max(r_D)——四处都退回原路径。三层同材料+ΔE_C=0 同样逐位退回单层
  （退化用例 ADR-018）。ZAZ（4/1/4 nm）现能出机制分解：TAT 主导、PF 次之、R1 两项极小，结构自洽。

  **涉及编号**：R1（分段势垒 + 分段 m_di，对照测试改 E_max 参考）、R2–R5（逐缺陷物种参数 + 分段路径
  质量的 WKB）、R6/R7（分段 G_ab，保留 F02 事件能量差）、R8（逐缺陷 ε_opt/ν）。文档侧订正
  `docs/rate_equations.md`（R6 加"叠层推广—分段衰减"、重写截断"实现注意"、R7 加"叠层只换几何因子"、
  "已确认"两条）、`docs/physics_model.md`（KMC 第 2 步与"已确认—陷阱间跃迁的近邻表"）、`README.md`
  （测试计数 171→182、测试表两行 + 新增 `test_stack_rates.py` 行、近邻表已知局限一段）。

  **pytest**：`pytest tests/ -v` 全绿，182 passed（171 + `test_stack_rates.py` 10 条 + `test_richardson.py`
  叠层 E_max 对照 1 条；`test_stack_schema.py` 的抛错断言翻转为端到端）。Phase 4 不打 tag。

- **变 κ 泊松 + 分段能带 + 界面单侧场读取接口（Phase 3 = ADR-011 + 带阶，并入 review §4.2 / F08）**
  （改 `src/kmc_leakage/electrostatics.py`、`config.py`、`bands.py`、`kmc.py`、`sweep.py`；
  新增 `tests/test_stack_physics.py`；订正 `docs/physics_model.md` 一句、同步 `README.md`）

  **背景**：ADR-020 的 Phase 3 把静电与能带从单一介质推广到叠层——变 κ 泊松、界面对齐
  网格、分段能带台阶、V_bi 关系。这是让多材料叠层"真正跑得动"的静电/能带地基；速率侧
  （分段 WKB、R2–R5 按材料取参数、R7 新 G_ab）仍属 Phase 4。因此本期之后**多材料叠层的
  静电与耦合矩阵已能构造**，但 KMC 运行仍在 `_prepare` 读 `single_material`（声子/R1 参数）
  处抛 `NotImplementedError`——`test_zaz_run_raises_until_layered_physics` 仍绿，Phase 4 翻。

  **变 κ 泊松（ADR-011 / review §4.2）**：`PoissonSolver` 的介电常数从单一标量
  （`single_material.kappa`）改为**按单元取值**——单元中心落在哪层就取该层的静态 κ
  （`stack.kappa_at`）。三对角系数按变系数离散式重构：下/上对角改为 κ_{k∓½}，主对角含
  κ̄_k·K⊥²（κ̄ 为相邻两单元 κ 的算术平均），右端项不再除 κ、顶边界项乘末单元 κ_{nz-½}。
  原来"逆分母兼作上对角 c'"的优化依赖上对角恒为 1，变 κ 后上对角是 κ_{k+½}、失效——改为
  一般 Thomas，前向存分母倒数、回代另存 c'。系数仍只依赖网格与各层 κ、与电荷无关，
  预分解后每步复用。

  **界面对齐网格**：`PoissonSolver.__init__` 校验每个内部界面恰好落在网格节点上（层厚是
  hz 的整数倍），否则报错并提示调 grid.nz 或层厚。单层无内部界面，自动通过。

  **分段能带（review §4.1）**：`conduction_band_edge` 从 `E_B − eφ` 改为
  `B_ℓ(z) − eφ(z)`（新增 `device.band_reference_profile`、`stack.layer_indices_of`），E_C
  在界面处有真实台阶（阻挡层内抬高一个 ΔE_C）；`band_profile` 的 E_V 改用逐层 Eg
  （`device.band_gap_profile`）。V_bi 关系 q·V_bi = B_c + ΣΔE_C − B_a 与 `band_offsets`
  在 Phase 1 已就位，本期补齐 ZAZ 台阶 / 显式 ΔE_C 覆盖 / V_bi 的测试。

  **界面单侧场读取接口（ADR-017 §2 / review F08）**：界面上 E_z 两侧差 κ 比，`np.gradient`
  的中心差分给的是错误的两侧平均。新增 `solver.electric_field_z_sided`（逐节点单侧差分）与
  `solver.field_vectors_at(phi, positions, field_side)`——**所有吃场的机制（R4/R5 前置因子、
  R8）读同一份场**：体缺陷（field_side=0）用中心差分、界面缺陷（±1）的 E_z 换单侧差分。
  `ElectrostaticCoupling` 与 `_prepare` 的场读取一律经此接口，`field_side` 从
  `DefectEnsemble` 取。

  **自作用（ADR-011）**：仍按耦合矩阵对角元整项扣除。变 κ 后对角元里含真实的介电失配
  像势，随自作用一并不计，与"不考虑镜像力"约定一致。

  **消费方迁移**：静电与能带侧的消费方从 `stack.single_material` 换成按位置/按层查询
  （`kappa_at` / `band_reference_profile` / `band_gap_profile`）。速率侧的 `single_material`
  读取（`_prepare` 的声子/R1、`_rates` 的 R4/R5/R8 材料参数）留到 Phase 4。

  **数值影响**：单层路径逐位不变——均匀 κ 下变系数 Thomas 退回常系数解、无内部界面、
  field_side 恒 0，静电/能带/场读取三处都退回原路径（170 条既有测试全绿）。多材料侧
  实测（ZAZ 4/1/4 nm）：无电荷分压最大误差 8.9e-16 V、电位移跨层守恒（相对散布 1.6e-15）、
  C/A=0.0275285 F/m²（与 review §4.2 独立一维方案逐位吻合）、面电荷跳变 = σ、变系数离散
  残差 1.7e-9（舍入量级）、响应矩阵互易 2e-17；界面单侧场按 κ 比（38/9）跳变、中心差分落在
  两侧平均。

  **涉及编号**：不改任何速率公式本身（R1–R8 未改，`rates/` 未动）；影响泊松离散、导带底
  剖面、局域场读取接口。文档侧 `docs/physics_model.md` 订正"三对角系数仅依赖网格"为
  "依赖网格与各层 κ"，`README.md` 同步实现要点 2、当前范围、测试表、测试计数。

  **pytest**：`pytest tests/ -v` 全绿，171 passed（156 + `test_stack_physics.py` 15 条：
  变 κ 分压/电容/面电荷跳变/变系数残差/界面对齐校验/同材料多层退回单层/响应互易、
  能带台阶/显式 ΔE_C 覆盖/V_bi 关系、界面单侧场/field_vectors_at 选侧/体缺陷回归锚点）。
  Phase 3 不打 tag。

- **分层缺陷生成 + 界面缺陷物种 + 全器件 R1-only 空集合路径（Phase 2 = ADR-005/008，
  并入 review F06/F07/F08）**
  （改 `src/kmc_leakage/defects.py`、`config.py`、`bands.py`、`kmc.py`、`sweep.py`、
  `__main__.py`；改多份测试，`tests/test_defects.py` 重写并扩充）

  **背景**：ADR-020 的 Phase 2 把 Phase 1 里"多层体缺陷抛 NotImplementedError、界面缺陷只
  解析不生成"的临时限制换成真正的分层生成。缺陷生成从"整器件一份配置"改为"逐层体缺陷 +
  逐界面面缺陷"，每个缺陷携带自己的材料/物种参数、导带基准与层/界面标签。多材料叠层此时
  仍不能整段跑（静电/速率还读 `single_material`，属 Phase 3/4），但缺陷侧已按叠层生成。

  **分层体缺陷（ADR-005）**：`generate_defects(device, rng)` 不再读 `config.defects`，改为遍历
  `stack.layers`：每层体缺陷数 = `round(浓度 × 层体积)`（按层一致取整）。空层（如 Al₂O₃ 体相，
  `layer.defects is None`）不生成体缺陷。**F06**：单层退化仍是"一个 layer 覆盖整器件"、不拆成
  子层，所以"分层表达"不会改变计数——退化配置的缺陷数与旧整器件生成**逐位相同**（已在
  worktree 上端到端复核：`examples/single_run.yaml`、seed 12345 下缺陷坐标/能级、单点 J、机制
  分解、统计窗口全部逐位一致）。

  **界面缺陷物种（ADR-008）**：每个内部界面按面密度 Nₛ 生成 `round(Nₛ × A)` 个界面缺陷，放在
  界面平面上（z 固定、xy 随机），与体缺陷预算分开计。界面缺陷是**独立物种**：能级相对势阱
  （导带较低的一侧，ZAZ 即 ZrO₂ 侧）的局部导带底、物性（m_di / ε_opt / Eg / ℏω / S / ν0 / Z）
  默认取该侧材料。读局域场用**单侧差分**（`field_side` 标记 ±1，ADR-017 §2 / F08），供后续
  R4/R5/R8 共用。

  **每缺陷物性数组**：`DefectEnsemble` 由"一份标量物性 + 位置/能级数组"扩成"逐缺陷携带
  positions/ED/r_D/charge_state/m_di/eps_opt/band_gap/phonon_energy/huang_rhys/attempt_frequency/
  band_reference/layer_id/interface_id/field_side"。`r_D` 按各自 m_di 算；`charge_state` 从标量
  改数组（体缺陷取层配置 Z，界面缺陷取物种 Z）。新增 `DefectEnsemble.uniform` 便捷构造与
  `empty_ensemble()`。

  **导带基准与内建电势（review §4.1）**：新增 `stack.band_offsets`（各层累积带阶 ΣΔE_C，
  ΔE_C 缺省按电子亲和能对齐 χ_j−χ_{j+1}、可显式覆盖）与 `device.layer_band_references()`
  （B_ℓ = 接触势垒 + 累积带阶）。`trap_levels` 改用逐缺陷 B_ℓ：E_i = B_ℓ − eφ − E_D。
  `barrier()` 按接触侧那一层材料取 χ；`built_in_potential()` 从 (Φ_c−Φ_a)/q 推广为
  q·V_bi = B_c + ΣΔE_C − B_a——电子亲和能对齐时数值不变，显式带阶覆盖也进这个累积关系。
  单层退化（无界面、ΣΔE_C=0）与旧行为逐位一致。

  **F07 空集合 / R1-only 完整路径**：全器件无缺陷（所有层空、无界面缺陷）返回空实现，整条链
  容空集合——`generate_defects`（返回空）、`hop_cutoff_radius`（改按 α_min 收参数，空时不调用）、
  `neighbor_pairs`（N<2 返回空表）、`ElectrostaticCoupling`（空矩阵）、KMC（总速率为零即停、
  只加 R1）、`SimulationResult`（`by_defect` 空数组）、`sweep`、可视化都跑通。`SimulationResult.
  n_events` 改为**实际执行事件数**（不再返回配置值，review F07）：R1-only 时为 0。

  **F08 界面缺陷电荷进泊松源项**：界面缺陷电荷 q(Z−n) 与体缺陷同口径进静电叠加——静电耦合
  矩阵按全部缺陷位置（体 + 界面）预计算，组态叠加权重取 n_i−Z_i（承接 F01），界面缺陷自动
  纳入同一泊松源项。固定界面面电荷 σ 与陷阱电荷分别计（σ 默认 0，其源项接入属 Phase 3）。

  **全局最小间距**：`_global_standoff` 取各配置层 min_separation 的最大值，同时管缺陷两两之间、
  缺陷到电极、体缺陷到内部界面（界面缺陷豁免"内部界面避让"、仍守两两间距）。跨层缺陷对也吃
  这一个约束。

  **hop 截断半径接口（承接 F05 前置）**：`hop_cutoff_radius(r_D, tol)` 改为 `hop_cutoff_radius(
  alpha_min, tol)`，r_cut = −ln(tol)/(2·α_min)。单一材料传 α_min = 1/max(r_D) 与旧口径逐位相同
  （参考预设仍 4.6 nm）；按所有材料最慢衰减推 α_min 的叠层版本属 Phase 4。

  **涉及编号**：不改任何速率公式本身（R1–R8 未改，`rates/` 未动）；影响缺陷生成、静电叠加
  权重、能级参考、内建电势。多材料叠层的分层物理（变 κ 泊松、分段能带/WKB、按材料取速率
  参数）仍属 Phase 3/4，ZAZ 配置能生成缺陷但运行时仍在静电阶段抛 NotImplementedError。

  **pytest**：`pytest tests/ -v` 全绿，156 passed（148 + `test_defects.py` 扩充的分层/界面/空集合
  用例 + `test_kmc_consistency.py` 的 R1-only 用例）。Phase 2 不打 tag。

- **配置 schema 改造：materials + layers + interfaces 的有序介质层表示，统一材料查询接口
  （Phase 1 = ADR-003 + ADR-015，并入 review F11）**
  （改 `src/kmc_leakage/config.py`、`io.py`、`electrostatics.py`、`bands.py`、`defects.py`、
  `kmc.py`、`examples/*.yaml`、多份测试；新增 `tests/test_stack_schema.py`）

  **背景**：ADR-020 的 Phase 1 把单一 `dielectric` 表示换成"有序介质层 + 材料库 + 显式界面"
  （ADR-003/015），为后续 ZAZ 叠层铺路。review F11 指出：若只改 schema 而把静电/速率消费方
  留到 Phase 3/4，Phase 1 满足不了"每期全测试通过"——因此本期把**所有消费方一并迁到统一查询
  接口、先让单层路径跑通**。Phase 2（分层缺陷生成、界面缺陷物种）不在本期。

  **schema（ADR-015）**：YAML 顶层由 `dielectric` + `defects` 换成
  - `materials`：按名字典，每个材料一组物性（κ、ε_opt、Eg、χ、m_cb/m_vb/m_di、ℏω、S、ν0）；
  - `layers`：有序层，各 `material`（按名引用）、`thickness_nm`、可选 `defects`（该层体缺陷）；
  - `interfaces`：可选，数量 = 层数-1，携带 `delta_Ec_eV` 覆盖、`sigma_C_per_cm2`、界面缺陷
    （面密度 `surface_density_cm2` 的独立物种）；
  - `Lz` 不再是配置项，由各层 `thickness_nm` 求和。

  单层退化 = `layers` 只含一个元素（`examples/*.yaml` 即此形态）。

  **统一查询接口**：新增 `DielectricStack`（`materials`/`layers`/`interfaces` + `Lz`/
  `single_material`/`bulk_defects`/`material_of`/`single()`），挂在 `DeviceConfig.stack` 上。
  `DeviceConfig.Lz`、`Config.defects` 改为只读属性（分别委托 `stack.Lz`、`stack.bulk_defects`），
  所以 `kmc.py`/`sweep.py` 里 `config.defects` 的读法不动。消费方（electrostatics 的 κ、bands 的
  Eg、defects 的 m_di、kmc 的 m_di/ℏω/S/ν0/ε_opt/Eg）一律改经 `stack.single_material` 取参数，
  不再读 `device.dielectric.*`。多材料叠层的分层物理属 Phase 3/4：`single_material` 在层引用多于
  一种材料时抛 `NotImplementedError`——ZAZ 配置能被 schema 完整解析，但运行时明确抛错。

  **io 校验（ADR-015 / review F11）**：不做运行时兼容层，旧 schema 直接报错并提示新写法——
  顶层 `dielectric`、顶层 `defects`、`device.Lz_nm` 三个旧键各自给出指向 examples 的提示
  （注意拒绝的是 YAML 顶层键，不是 `config.py` 的 `device.dielectric` 属性，见 review F11）。
  另校验材料引用是否存在、层厚为正、interfaces 数量 = 层数-1；面密度/面电荷按 cm⁻² → m⁻²
  （centi⁻²）换算，体浓度仍 cm⁻³。一次性迁移 `examples/single_run.yaml`、`iv_sweep.yaml`
  与全部测试，不留旧写法。

  **单层退化验收（ADR-018）**：`tests/test_stack_schema.py` 新增单元素 layers 退化用例——
  同一份材料/缺陷/器件/网格/种子，经 YAML 解析与直接构造两条路给出逐位相同的确定性量
  （Lz、材料参数、R1 两分量 rtol ≤ 1e-12）与同种子完整轨迹（J、机制分解 rtol ≤ 1e-9）。
  另有 `DielectricStack` 查询接口的单测、ZAZ 三层 schema 解析（材料/层/界面/界面缺陷/Lz 求和）
  与"多材料运行时抛错"的边界用例。

  **数值影响**：单层路径逐位不变——消费方经 `stack.single_material` 取到的是同一份材料参数，
  缺陷生成、静电、速率、J 都与改造前相同（端到端复核 `examples/single_run.yaml` 单点：
  N=68、TAT 99.72%/PF 0.28%，与 Phase 0 基准一致）。schema 是结构层改造，不动任何 R 编号的
  速率公式，`rates/` 未改。

  **涉及编号**：不涉及速率公式本身（R1–R8 未改）；影响 `config.py`/`io.py` 与 electrostatics/
  bands/defects/kmc 里取材料参数的入口。文档侧改 `README.md`（配置格式、库用法、目录结构、
  测试表）。

  **pytest**：`pytest tests/ -v` 全绿，148 passed（原 134 + `test_stack_schema.py` 7 条 +
  `test_config_io.py` 7 条 schema 校验 / 旧键拒绝）。Phase 1 不打 tag。

- **单层模型：空陷阱固定电荷 +Z·q 进泊松源项（F01），R7 改用多体事件能量差（F02）**
  （改 `src/kmc_leakage/kmc.py`；新增 `tests/test_defect_charge_and_hop_energy.py`、
  改 `tests/test_kmc_consistency.py`；订正 `docs/rate_equations.md`；
  `docs/adr/zaz-stack-decisions.md` 的 ADR-009 补背景、`ZAZ_STACK_MODIFICATION_PLAN.md`
  顶部加 ADR 指针）

  **背景**：ADR-021（整合 `docs/reviews/zaz-stack-review-2026-09-15.md` 的 F01–F04）
  把 ADR-020 的 Phase 0 从"只修 F01"改成"F01 与 F02 一起修、之后再冻结单层基准"。
  本次落地 F01+F02 与相关文档订正，并重建单层基准。F03/F04 属"首次叠层电流交付前"，
  不在本阶段。

  **F01（缺陷固定电荷）**：`kmc.py::_electrostatics` 原来只把占据电子电荷 −q·n_i 投进
  静电叠加，空施主陷阱的固定正电荷 +Z·q 没进源项（Z 只进 R8）。改为按净电荷
  q·(Z_i−n_i) 叠加：耦合矩阵按一个 −q 电子预计算，组态叠加权重取 n_i−Z_i，电势、
  电场、横向平均剖面与自作用扣除四处用同一权重（ADR-021 F01）。R1 仍按零空间电荷
  势垒求值，不受影响。

  **F02（R7 多体事件能量）**：陷阱间跃迁的 MA 能量因子原来用跳跃前组态的单点能级差
  E_b−E_a，漏掉了两处电子的库仑互作用能。改用完整事件能量差 ΔH(a→b)=(E_b−E_a)−U_ab
  （U_ab=q²·G_ab，由静电耦合矩阵非对角元对称化得到）：终态 b 的能级去掉出发陷阱 a
  的贡献。正反向速率在各自出发组态求值后满足细致平衡。`_prepare` 一次算好 `_hop_U_ab`
  （与组态无关），`_rates` 里 R7 用 energy[b]−U_ab。

  **数值影响**：F01 改变能带弯曲、F02 改变陷阱间跃迁速率，单层 J 与旧值不同——按
  ADR-002/ADR-021 接受，旧含 bug 数值只作历史记录、不作回归钉子。复核报告的独立锚点
  逐位复现：U_ab=0.03476324 eV、两个空施主陷阱各 +0.03476324 V、占据（中性）0 V；
  F02 修复后正反比 0.5105069 = exp(−ΔH/kT)（旧口径 0.2606）。

  **修复后单层基准**（`examples/single_run.yaml`：50×50×9 nm、3×10¹⁸ cm⁻³、高斯能级
  ED=1.15 eV/σ=0.15 eV、界面富集 g=5/λ=1 nm、3 V、25 °C、M=5、n_events=20000）：
  J = 5.7523e-04 A/cm²，相对标准误 34.6%，σ(log₁₀J)=0.538。机制分解：TAT 5.7367e-04
  （99.73%）、PF 1.5602e-06（0.27%）、直接/FN 3.29e-14、热电子 4.16e-23 A/cm²。抽到的
  种子 [1395795651, 1125651586, 1663284447, 1221919100, 617344681]——照抄回
  `simulation.seed` 即可复现（`ensemble.count` 每次随机抽种子）。TAT/PF 占比已补进
  ADR-009 作背景。

  **文档**：`docs/rate_equations.md` 订正 R1 热发射对照式（原
  `A*·T²·exp[−qφ_B/kT]·[exp(qU/kT)−1]` 随偏压发散、与代码的有界双向形式矛盾，错在
  文档而非实现，见 ADR-002），改为 `A*·T²·[exp(−qφ_B/kT)−exp(−q(φ_B+U)/kT)]`；R7 一节
  补"能量差取完整事件能量差、非单点能级差"的说明（ADR-021 F02）。
  `ZAZ_STACK_MODIFICATION_PLAN.md` 顶部加指向 ADR 的说明（§6.4→ADR-002、
  §4.2→ADR-011 已订正）。

  **涉及编号**：R7（陷阱间跃迁，直接改）、R2–R5/R8（经静电场与能带弯曲间接受 F01
  影响）、R1（对照式文档订正）。

  **pytest**：`pytest tests/ -v` 全绿，134 passed（原 129 + 新增 5 条 F01/F02 检验；
  另有 2 条陷阱间跃迁一致性测试跟随 U_ab 订正）。

- **系综种子改为每次运行随机抽取**（`io.py`、`config.py`、两个示例 YAML、
  `tests/test_config_io.py`；不涉及具体 R 编号）

  **改了什么**：`ensemble` 段的配置项从写死的 `seeds: [...]` 列表改成
  `count: <M>`。加载配置时由 `io._draw_seeds` 用不带种子的 `np.random.default_rng()`
  （取 OS 熵）随机抽取 count 个互异的种子，装进 `EnsembleConfig.seeds`。抽取在
  `load_config` 里发生一次，运行内保持稳定，下游 `config.seeds`、CSV 的
  `J_seed<seed>` 列、CLI 进度里逐个打印种子的口径全部不变。

  **为什么**：原先五个固定种子每次跑出的是同一批缺陷实现，掩盖了"换一批实现
  重估同一工作点"这层随机性。改成每次运行抽新种子后，重跑即是独立的系综抽样；
  抽到的种子随 CLI 进度与 CSV 落盘，需要复现某次结果时照抄回去即可。

  **pytest**：`pytest tests/` 全绿（129 passed）。新增
  `test_ensemble_count_draws_distinct_seeds`（count 抽出互异整数种子）、
  `test_ensemble_seeds_are_random_across_loads`（两次加载种子不同），
  `test_bad_ensemble_count_is_rejected` 取代原 seeds 列表的校验用例。

- **新增 `docs/electrode_trap_rates.md`：R2–R5 各参数的计算方法**
  （新增 `docs/electrode_trap_rates.md`，只写文档，不改代码）

  **改了什么**：把 R2/R3（弹性俘获/逃逸）与 R4/R5（多声子俘获/逃逸）公式里每一个
  参数的来源、计算式、单位、代码位置整理成一份文档。`docs/rate_equations.md` 给的是
  公式本身，这份补的是"公式里的 E1、f_C、P、N_C、L_p、c_MP 分别怎么算出来"——参数
  装配全在 `kmc.py` 的 `_rates()` 里，原先只能读代码。

  内容分七节：共同前置量（E_D 采样、r_D、缺陷处电势与局域场的线性叠加与自作用扣除、
  导带底剖面、电极导带底与 E1、WKB 路径与 P(E)、f_C）；R4/R5 特有的 c_MP 常数因子、
  δ 函数塌缩成对 p 求和、N_C、Huang-Rhys 线型 L_p（含 `ive` 防溢出改写与
  `[(n̄+1)/n̄]^{p/2} = exp(p·ℏω/2k_BT)` 这一步等价变形）、R5 的玻尔兹曼因子；数值参数
  一览；一组实测中间量；验证测试对照表；容易出错的七点。

  **为什么**：E_D（深度，恒正）与 E（总能量，可负）、E1（以电极导带底为参考）三个
  能量各有各的参考点，m_e/m_di/m_cb 三个质量各归各位，走错一个就是几个数量级的偏差，
  但这些约定散在 `bands.py`、`defects.py`、`kmc.py` 的注释里。

  **实测中间量哪来的**：参考预设、2 V、25 °C、seed 12345、全空占据组态下取 z ≈ 4.46 nm
  的缺陷，直接从 `KMCSimulation._rates()` 的中间量导出，一起给出三条可手算复核的关系
  （R2 = 前置因子·f·P、R3/R2 = (1−f)/f、R5/R4 = exp((E−E_F)/k_BT)，实测 1.065e27 对
  解析 1.07e27）。

  **涉及编号**：R2、R3、R4、R5（`docs/rate_equations.md` 对应各节）。

  **pytest**：`pytest tests/ -q` 128 passed（纯文档改动，作为回归基线记录）。

- **扫描结果画成 I-V / I-T 曲线图：同一张图上画出总电流、TAT 与 PF 三族曲线，
  颜色分机制、深浅分温度/电压，不画误差棒**
  （改 `src/kmc_leakage/visualization.py`、`src/kmc_leakage/__main__.py`、
  `tests/test_visualization.py`、`tests/test_sweep.py`、
  `docs/physics_model.md`、`README.md`）

  **改了什么**：`visualization.py` 新增 `plot_iv_curves` 与 `plot_it_curves`，共用
  私有的 `_current_curves`。I-V 图横轴电压、一条曲线一个温度；I-T 图横轴温度（°C）、
  一条曲线一个电压——同一份数据换个切法（内部就是把 J 矩阵转置）；两张图纵轴都是
  J 的对数轴、单位 A/cm²。输入是 `SweepResult` / `EnsembleSweepResult` 的
  `voltages`、`temperatures`、`current_density` 三个数组（J 矩阵形状
  (温度点数, 电压点数)）。CLI 的 `sweep` 加 `--plot-prefix`，扫完落
  `<前缀>_iv.png` 与 `<前缀>_it.png`。

  两个函数再各加一个可选参数 `by_mechanism`（就是 `SweepResult.by_mechanism` 那样的
  矩阵字典），传入时把 **TAT 与 PF 两族曲线画进同一张图**，两个维度分到两个通道：
  **颜色分机制**（总电流蓝、TAT 橙、PF 绿），**深浅分行**（同一色系里浅 = 低温/低压、
  深 = 高温/高压）。线型与标记同样跟机制走（总电流实线圆点、TAT 虚线方点、PF 点线
  三角），是冗余的第二通道——识别机制不必只靠颜色，色觉障碍与黑白打印下仍可读。
  图例分两段：先各行的温度/电压（画的就是总电流那条线，深浅即行序），再拿三条
  `Line2D` 代理标出机制的颜色与线型，否则 行数 × 3 条曲线会把图例胀成一整块。
  `_current_curves` 相应多一个 `components` 参数（`_Component` NamedTuple 的序列），
  取色集中在 `_shades` 一个函数里。CLI 的 `_write_sweep_plots` 一律把机制分解传下去。

  **配色怎么定的**：三条单色系 `Blues` / `Oranges` / `Greens`，按行数在 0.50–0.95
  区间上等分取样。下限 0.50 是为了最浅那档在白底上仍有 2.1:1 以上的对比度（再浅就
  快看不见了），上限 0.95 是为了最深那档不糊成黑色；只画一行时取区间中点，也就是
  图例代理用的那一档。三个中点（`#2777b8` / `#de4e05` / `#29914a`）作为分类色实测
  通过色觉障碍分离度检验——全配对最差 ΔE 8.4（deuteranopia，阈值 8）、常视 20.4
  （阈值 15）、白底对比度均 ≥ 3:1；每条色系的 5 档深浅也通过单色渐变检验（亮度单调、
  相邻 ΔL ≥ 0.06）。**行数超过 5 档时深浅分辨会变紧**：默认 I-T 图是 11 个电压点，
  相邻两档本身分不开，只能靠图例与"越深越高"的单调方向读——这是把 33 条曲线放进
  一张图的固有代价，与实现方式无关。

  **不画误差棒**。系综的标准误 σ/√M 照算不误，只是走数字出口：CLI 摘要逐点打印、
  CSV 有 `J_sem_A_cm2` 一列、`EnsembleSweepResult.current_density_sem` 也还在。绘图
  侧因此把 `current_density_sem` / `by_mechanism_sem` 两个入参、`errorbar` 调用以及
  "误差棒下端压在 0.999·J 以内"那段对数轴裁剪一并删掉，曲线改用 `axes.plot`——留一个
  被静默忽略的参数比删掉更糟。

  **为什么这么接**：出图跟着扫描一次做完，不引入"从 CSV 反读"的第二条链路。绘图
  函数仍只接受已经算好的数组、返回 Figure——不读配置、不落盘、不触发仿真，
  `savefig` 发生在 `__main__._write_sweep_plots`，因此 `visualization.py` 对
  `sweep.py` / `ensemble.py` 两个结果类型都不产生依赖，单实现扫描与系综扫描走同一个
  函数。单位换算放在绘图这个输出边界，与 `write_csv` 同口径：J → A/cm²、T → °C。
  机制分解做成可选参数而不是新开一对函数，是因为要的正是"画在**同一张**图上"——
  总电流与分量共用一套坐标轴，拆成两个函数反而要把 axes 传进去，破坏"函数返回
  Figure"的约定。

  **只画 TAT 与 PF，不画 R1 的两项**：`direct_fn` 比总电流小 7.5–10.7 个数量级、
  `thermionic` 更小（8–15 个），放进同一张对数图会把纵轴拉开十几个十倍程，TAT 与
  PF 的交叉点反而看不清。四项完整分解仍在 CSV 里。要画的两项写在模块级
  `_COMPONENT_STYLES` 常量里。

  **一个数值细节**：分量在某个工作点上可以恰好是 0——统计窗口内没有该类事件，不是
  "很小"而是没有。`_log_safe` 把非正值换成 nan，曲线上留一个断点；让 matplotlib
  静默丢点也画得出来，但看图的人分不清"这里是 0"和"这里没采到"。

  **实测形状**（示例配置减到 5 电压 × 3 温度、3000 事件，单实现，A/cm²）：2 V 处
  PF 主导（25 °C：PF 1.68e-08 对 TAT 3.09e-10），TAT 在 2–3 V 之间越过 PF 并在高压端
  主导总电流（4 V / 25 °C：TAT 4.46e-01 对 PF 2.50e-05），温度越高交叉点越靠后
  （25 °C 在 2–2.5 V 之间，125 °C 推到 2.5–3 V）。I-T 图上 4 V 的 TAT 从 25 到 125 °C
  只涨 11%（近乎水平），同一电压的 PF 涨 151 倍，与 R8 的 exp(−E_D/k_BT) 一致。
  同一张图上两族曲线的交叉与分工因此一眼看得出来，这正是画在一起的目的。

  **测试**：新增 21 条。`tests/test_visualization.py` 盯住数据换算（曲线上的 y 就是
  `current_density × centi²`、I-T 的横轴是 `T − 273.15`）、切法（I-V 一条曲线一个
  温度、I-T 一条曲线一个电压，转置写错则曲线条数就不对）、图例标注的数值、单工作点
  （1 温度 × 1 电压）也能出图、**图上没有误差棒**（`containers` 与 `collections` 都
  为空——`errorbar` 会留下 `ErrorbarContainer` 和画竖线的 `LineCollection`，改回去就
  翻）；机制分量这边另加七组：分量数值逐点核对、**I-T 侧分量矩阵跟着转置**（漏掉
  `.T` 就翻）、**每个机制一个色系**（同一行三条线颜色互不相同，且 RGB 主通道分别落在
  B/R/G 上，换错色系就翻）、**行序逐行变深**且最浅一档的相对亮度仍在 2:1 对比度以内、
  **色系不随行数漂**（少画几个温度点，机制颜色不变，且单行取的正是图例代理那一档）、
  两段式图例的完整标签列表、R1 两项确实不上图、分量为 0 时只多出一个 nan 且总电流
  曲线不受影响。`tests/test_sweep.py` 补一条 CLI 侧的 `--plot-prefix` 落盘检查（按
  CLI 的实际调用带上 `by_mechanism`）。原有的"绘图函数不碰文件系统"那条把两个新
  函数也纳进去。

  **涉及**：`docs/physics_model.md` 的"可视化接口"一节——I-V / I-T 曲线从"后续可能
  添加"移到"本阶段实现"，并写明同图叠机制分量的通道分工与"图上不画误差棒"；"预期
  输出"与"已确认"里描述 σ/√M 的措辞由"误差棒"改为"不确定度"（口径一个字没改，只是
  它现在不出现在图上）。"后续可能添加"里"各机制…的电流占比分解图"相应收窄为按缺陷
  位置的那张。统计口径沿用 `docs/rate_equations.md` 的机制对照表（TAT = R2–R7，
  PF = R8），物理模型与速率公式（R1–R8）未改。

  **pytest**：`pytest tests/ -v` 128 passed（原 107 条 + 新增 21 条）。

- **I-V/I-T 扫描的示例配置补齐缺陷分布旋钮，并补测试盯住扫描确实吃这些参数**
  （改 `examples/iv_sweep.yaml`、`tests/test_sweep.py`、`tests/test_config_io.py`、
  `README.md`）

  **改了什么**：`examples/iv_sweep.yaml` 的 `defects` 段原来是简版——只有
  `level.mode: fixed` 与 `spatial.mode: uniform`，没写 `sigma_eV` / `enrichment` /
  `decay_length_nm`，看配置的人不知道扫描也能开高斯能级和界面富集。现在与
  `single_run.yaml` 同构，参数都预置好、注释标出可选模式，默认值仍是 fixed/uniform
  ——示例跑出来的结果没有变化。

  **代码一行没改**：`run` 与 `sweep` 本来就共用同一条链路。`io._defects` 解析两种
  模式与后面用 run 还是 sweep 无关；`sweep.run_sweep` 调的就是同一个
  `generate_defects(device, config.defects, rng)`；`KMCSimulation` 的近邻表按**逐个
  缺陷**的 r_D 算截断半径，高斯能级下 r_D 不同也不必特殊处理；
  `ElectrostaticCoupling` 只依赖缺陷位置，界面富集同样不受影响。缺的只是示例配置
  的可发现性与一道回归防线。

  **为什么补测试**：这条链路是隐式共用的——`run_sweep` 自己调 `generate_defects`，
  一旦有人在扫描侧另传一份缺陷配置、或漏掉 `config.defects` 的字段，分布模式会
  悄悄退化成 fixed/uniform 而扫描照跑、不报错。新增
  `test_sweep_honours_gaussian_level_and_interface_enrichment`：断言扫描用的缺陷与
  按同一配置、同一种子直接生成的**逐位相同**，且 ED 分散、r_D 逐缺陷不同、z 采样
  与 uniform 不同，两种模式下机制分解仍可加。`test_example_configs_load` 加三条
  断言，防止示例配置退回简版。变异检验（强行把扫描侧的模式改回 fixed/uniform）下
  四条断言全部翻转，测试确实能咬住回归。

  **涉及**：R6/R7 的逐缺陷 r_D 与 `docs/physics_model.md` 的"缺陷（陷阱）模型"
  一节，速率公式本身未改。

  **顺带发现（未处理）**：`np.allclose` 的默认 `atol=1e-8` 在这个项目里是个坑——
  缺陷坐标是 1e-9 m、能级是 1e-19 J 量级，都比 atol 小，`np.allclose(位置, 位置)`
  会恒真（新测试第一版就栽在这里）。新测试改用 `np.array_equal` 与 `atol=0.0`。
  已有的 `test_sweep_mechanism_decomposition_sums_to_total` 写的是 `rtol=1e-12`，
  但测试配置下 J ≈ 3e-3 A/m²，默认 atol 折算成 3.5e-6 的相对容差、实际把 rtol
  盖掉了；量级上仍能抓住真的分解错误，本次未改动。

  **pytest**：`pytest tests/ -v` 全绿，107 passed。

- **陷阱间跃迁（R6/R7）改用近邻表，不再建 (N,N) 稠密矩阵；默认器件尺寸改回
  50 × 50 × 9 nm**（改 `src/kmc_leakage/defects.py`、`src/kmc_leakage/kmc.py`、
  `examples/*.yaml`）

  **改了什么**：
  1. `defects.py` 新增 `NeighborPairs`（有序缺陷对 a/b/r）、`hop_cutoff_radius`
     与 `neighbor_pairs`。截断半径按"R6 的几何因子衰减到 `tolerance`"定义，取实现
     里**最大**的 r_D 来算（那是衰减最慢的一对，对其余组合都保守），不写死长度。
     距离分块计算，峰值内存由 `_PAIR_CHUNK_ELEMENTS` 定住，不随 N² 增长。
     `pair_distances` 保留不动，仍供小 N 下的核对与测试使用。
  2. `kmc.py` 的 `_hop_geometric` 从 (N,N) 变成近邻表上的 (P,)；`_rates` 返回的
     跃迁速率同样是 (P,)，第 k 项对应 `_pairs.a[k] → _pairs.b[k]`。新增
     `_hop_sums`，用 `np.bincount` 把稀疏速率聚合成流入/流出总速率，替代原来的
     `occf @ hop` 与 `hop @ empty`。事件表的跃迁段与事件解码跟着换成按对索引。
     截断容差 `_HOP_CUTOFF_TOLERANCE = 1e-24` 与 `_N_PHONON` 等数值参数放在一起。
  3. 两个示例配置的器件尺寸 100×100×8 → **50×50×9 nm**（缺陷数 240 → 68），
     `grid` 64/64/32 → **32/32/36**，保持 hx = hy = 1.5625 nm、hz = 0.25 nm——
     固定格点数改尺寸会改变网格步长与 CIC 自作用势（∝ 1/h）。

  **为什么**：把浓度提到 3×10¹⁹ cm⁻³ 时 `MemoryError: Unable to allocate 55.6 MiB
  for an array with shape (2700, 2700)`。N 从 240 涨到 2700 让所有 N² 数组涨 126 倍，
  而 KMC 主循环每步要新分配 5~6 个 (N,N)（MA 因子、hop、event_hop、flat、cumsum），
  加上 `pair_distances` 为返回 55.6 MiB 的结果瞬时吃掉约 600 MiB，进程峰值贴到了
  可用内存上限。R6 的几何因子 exp[-r_ab(1/r_Da+1/r_Db)] 的 e 折长度只有 r_D/2
  （参考预设下 0.083 nm），远小于缺陷平均间距，稠密矩阵 99% 以上的元素在双精度下
  就是零——存它们既费内存又费时间。

  **涉及**：R6/R7（`docs/rate_equations.md` 的 R6 一节 + "已确认"清单）。速率公式
  本身一个字没改，`rates/trap_hopping.py` 未改动，改的是事件表存哪些缺陷对。
  文档侧改了 `docs/physics_model.md`（默认尺寸、预设表、KMC 算法第 2 步、"已确认"
  三条、待确认清单）、`docs/rate_equations.md`、`README.md`（示例输出、耗时参考、
  YAML 示例、系综一节、已知局限、测试表、目录结构）。

  **数值影响**：与稠密实现**逐位相同**。用 `git worktree` 拉出改动前的 HEAD 跑同一
  份配置，3×10¹⁸（N=68）与 3×10¹⁹（N=675）两个浓度下每个种子的 J、系综平均、标准误、
  四项机制分解全部逐位一致——被丢掉的对速率小到加进 `cumsum` 都不改变双精度结果，
  连 KMC 选中的事件序列都没变。器件尺寸改小是独立的口径变更，J 的绝对值当然会变
  （50×50×9 nm、2 V、M=5 实测 J = 1.6555e-08 A/cm²，相对标准误 11.7%）。

  **性能**（本机实测）：

  | 工况 | 改动前 | 改动后 |
  |---|---|---|
  | 100×100×9 nm @3e19（N=2700）峰值 RSS | 911 MiB | 414 MiB |
  | 同上，单步耗时 | 0.300 s | 0.089 s |
  | 同上，20000 事件 | 1.67 h | 0.49 h |
  | 同上，跃迁表内存 | 55.6 MiB（N²） | 0.22 MiB（28762 对 = N² 的 0.4%） |
  | 新默认 50×50×9 nm @3e18（N=68）单步 | — | 0.002 s，峰值 80 MiB |

  剩下的 222 MiB 常驻是静电耦合矩阵 `phi_matrix` (N,N) + `field_matrix` (3,N,N)，
  仍是 O(N²)，本次没动（它们只以 matvec 和对角元两种方式使用，实测改成每步重解
  一次泊松是 12.3 ms vs matvec 9.5 ms、常驻 4 MiB vs 222 MiB，留待后续决定）。

  **pytest**：`pytest tests/ -v` 106 passed。新增 4 个测试：`test_defects.py` 里
  近邻表与"稠密矩阵+掩码"逐位一致（monkeypatch 把分块阈值压到强制走多块路径，
  专门覆盖块边界上的自身对排除）、截断半径命中给定容差、截断半径覆盖整盒时退化为
  全部有序对；`test_kmc_consistency.py` 里近邻表与稠密 R7 矩阵在保留对上逐位相同、
  截掉的总速率 ≤ 1e-15 相对量、`_hop_sums` 与稠密 matvec 等价。原
  `test_hopping_channel_is_r7_not_r6_plus_r7` 改为按近邻表索引核对。

- **加入系综平均：M 个缺陷实现取算术平均；侧向尺寸放大到 100 × 100 nm**
  （新增 `src/kmc_leakage/ensemble.py`，改 `config.py`、`io.py`、`__main__.py`、
  `examples/*.yaml`）

  **改了什么**：
  1. 新增配置段 `ensemble.seeds`（`EnsembleConfig`）。`Config.seeds` 给出回落规则：
     整段省略就退化为 `simulation.seed` 一个实现，旧配置行为逐位不变。读入时拒绝
     空列表和重复种子——重复实现不带新信息，却会让标准误偏小。
  2. 新增 `ensemble.py`：`run_ensemble`（单点）与 `run_ensemble_sweep`（扫描）逐个
     种子跑完取**算术平均**，误差棒取 σ/√M（M=1 时按 0 处理）；`write_ensemble_csv`
     写平均、标准误、各机制平均，外加每个实现的原始值 `J_seed<种子>_A_cm2`，换平均
     口径不必重跑。一个种子对应一次独立实现——种子同时驱动缺陷生成与 KMC 轨迹，
     每个实现自己重做一遍静电预计算，实现之间没有共享状态。
  3. CLI 的 `run` / `sweep` 在有 `ensemble` 段时走系综路径（逐实现打进度、最后给
     平均 ± 标准误）；`plot` 改画 `cfg.seeds[0]` 的实现，与 run/sweep 用的第一个
     种子对上。
  4. 两个示例配置侧向尺寸 50 → 100 nm，缺陷数 60 → 240；`grid.nx/ny` 32 → 64 同步
     放大，保持 hx = hy = 1.5625 nm。各配 5 个种子。

  **为什么**：单次 KMC 只看到一套具体的缺陷坐标，而漏电流对这套坐标敏感，之前只能
  给出"某一个器件"的 J。要给出可与实验比绝对值的 I-V，得对多个实现取算术平均——
  电流是可加量，实验测到的也是整片电极上大量局部通道之和。扩大面积与加种子是同一
  种方差缩减（都在增加独立缺陷样本数），方案里两条一起用。

  **涉及**：不涉及任何 R 编号的速率公式，KMC 内部与 `rates/` 均未改动。
  文档侧改了 `docs/physics_model.md`（器件默认尺寸、默认参考预设表、预期输出、
  待确认清单与"已确认"三条）、`README.md`（快速开始与耗时、库用法、`ensemble` 配置
  一节、实现要点 5、已知局限、测试表、目录结构）、`PROJECT_CONTEXT.md`。

  **数值影响**：没有 `ensemble` 段的配置逐位不变（`Config.seeds` 回落到
  `simulation.seed`，`run_ensemble` 在 M=1 时与 `KMCSimulation(config).run()` 逐位
  相同，已写成测试）。示例配置换了器件尺寸，J 本身随之改变。

  实测（16 个实现，n_events=2000，参考预设）：

  | 侧向尺寸 | 缺陷数 N | σ/mean @2 V | σ(log₁₀J) @2 V | σ/mean @4 V | σ(log₁₀J) @4 V |
  |---|---|---|---|---|---|
  | 50 × 50 nm | 60 | 38.8% | 0.190 | 52.0% | 0.329 |
  | 100 × 100 nm | 240 | 16.9% | 0.080 | 27.9% | 0.138 |

  缺陷数翻 4 倍把 σ 砍掉约一半（σ ∝ 1/√N），两种尺寸的均值在各自标准误内一致
  （2 V 差 8%，4 V 差 0.8%），没有系统性偏移。100 × 100 nm 下 M=5 对应 8%（2 V）
  到 13%（4 V）的相对标准误。

  **机时**：每事件 1.9 → 7.4 ms（事件表里 N² 项的陷阱间跃迁涨了 16 倍），每个实现
  另加 2.6 s 静电预计算。`examples/single_run.yaml` 的 5 个实现 × 20000 事件实测
  753 s。顺带测到的一条：把 4 V 相对标准误压到 10%，50 nm 需 M≈27、100 nm 需 M≈8，
  单实现机时正好差 3.8 倍——两条方差缩减路线**总机时几乎相同**，差别只在内存
  （扩面积 O(N²)，加种子 O(1)）。

  **测试**：新增 `tests/test_ensemble.py`（10 例）——平均确实是算术平均、每个种子确实
  换了一整套缺陷（逐个对照 `generate_defects` 的独立复现，并两两确认不重合）、M=1
  退化为普通单点仿真且标准误为 0、标准误 = σ/√M、机制分解之和 = 平均总电流、
  **R1 两项在系综里零散布**（按构造与缺陷无关，同时是"实现之间没串味"的自检）、
  进度回调、扫描系综的形状与实现内缺陷复用、CSV 保留各实现原始值。
  `tests/test_config_io.py` 新增 3 例：`ensemble.seeds` 解析、无 `ensemble` 段时回落到
  `simulation.seed`、空列表/重复种子/非列表被拒；`test_example_configs_load` 加断言
  钉住两个示例是 100 nm、5 个种子、且 `Lx/nx == Ly/ny == 1.5625 nm`。
  `tests/conftest.py` 的 `make_config` 增加 `ensemble` 参数（缺省 None，既有测试不受
  影响）。另外端到端核对了 `run` / `sweep` 的系综输出与 CSV（逐行核对每实现均值
  = `J_mean_A_cm2`）、`plot` 的种子提示。

  **pytest 结果**：`pytest tests/ -q` → **102 passed in 18.05s**（改动前 88 passed）。

- **R1 改为按公式一次性求值，电流在统计末尾加到结果上**（`src/kmc_leakage/kmc.py`）

  **改了什么**：R1（电极↔电极，直接/FN 隧穿 + 热电子发射）此前虽然不进 KMC 事件表，
  但仍在主循环内**每步重算一次**，用的是含缺陷空间电荷的能带剖面，结果按
  `Σ R1·dt / T` 做轨迹时间平均。现在改为：势垒剖面取零空间电荷的外加偏压解
  （`E_C(z) = E_B − e·(U+V_bi)·z/Lz`，无电荷时泊松解严格线性），在 `_prepare()` 里
  按公式求值一次，`run()` 末尾把 `I_R1 = q·R1` 直接加进 `by_mechanism`。
  新增模块级函数 `kmc.r1_electrode_current`；`_rates()` 的返回值从
  `(blocks, hop, r1_tunnel, r1_thermionic)` 收窄为 `(blocks, hop)`。
  速率公式本身（`rates/electrode_tunneling.py` 的 WKB 数值积分）未改动。

  **为什么**：
  1. R1 走的是理想平面势垒的物理图像，Richardson-Schottky 对照式也是
     （`docs/rate_equations.md` R1 一节的"适用范围"），让它跟随随机轨迹上的空间电荷
     没有物理收益，反而让"偏差随缺陷浓度趋零而收敛"这条判据依赖一个绕远的论证。
     求值口径统一后，两边是同一个势垒模型且都与缺陷无关，判据按构造严格满足。
  2. 每步两段 WKB 积分（129 路径点 × 128 能量点 × 2）换来的是一个对总电流贡献
     10⁻¹¹–10⁻⁸ 的量。移出主循环后单点实测 **2.04 → 1.80 ms/事件（−12%）**，
     20000 事件从 40.9 s 降到 36.0 s；测试套件 14.96 s → 11.32 s。

  **涉及**：R1；`docs/rate_equations.md`（R1 一节的"实现注意"与"适用范围"、
  "已确认"清单）、`docs/current_statistics.md` §1.2、`README.md`（实现要点 3、
  耗时参考、示例输出、测试表）。

  **数值影响**：`current` / `tat` / `pf` / `elapsed_time` 逐位不变（KMC 随机数流未被
  触碰）。`thermionic` 也逐位不变——势垒顶被 Dirichlet 边界钉在 z=0、阈值以上
  P(E)=1，空间电荷本来就进不来。唯一变化的是 `direct_fn`：从轨迹平均值变为理想
  势垒值，2 V 实测 `1.3407e-17 → 1.3561e-17 A/cm²`（+1.15%），总电流不变。

  **行为变化**：`elapsed_time == 0` 时（`n_events` 极小）以前四个机制分量全为 0，
  现在 `current` 等于 `I_R1`。R1 不需要 KMC 统计窗口，这样更正确，未加特殊分支。

### 测试

- 新增 `test_kmc_consistency.py::test_r1_contribution_is_the_ideal_barrier_formula`：
  手写理想梯形势垒直接调 `r1_electrode_electrode`，结果必须与 `by_mechanism` 的两项
  吻合（rel=1e-9），把"R1 = q × 理想势垒公式值"这个新口径钉死
- 新增 `test_kmc_consistency.py::test_r1_contribution_is_independent_of_kmc_sampling`：
  换 seed、换 `n_events`，R1 两项必须逐位相同（同一批运行里 `tat` 确实在动）
- `test_richardson.py::test_deviation_does_not_grow_as_defect_concentration_drops`：
  断言从"低浓度端不更差"收紧为三个浓度下热发射电流密度**严格相等**，docstring 里的
  理由从"势垒顶被 Dirichlet 边界钉住"改为"R1 在外加偏压剖面上求值，与缺陷无关"
- `test_kmc_consistency.py::test_hopping_channel_is_r7_not_r6_plus_r7`：跟随
  `_rates()` 返回值收窄，解包由 4 元改 2 元

**pytest 结果**：`pytest tests/ -q` → **88 passed in 11.32s**（改动前 86 passed）。

另外核对了两条端到端路径：`examples/single_run.yaml` 单点输出与上面的数值影响一致；
`examples/iv_sweep.yaml` 的 15 点扫描里 `J_direct_fn` 随 U、T 单调上升，`J_thermionic`
在同一温度下与偏压无关（`eU ≫ k_BT` 时 `exp(−(E_B+eU)/kT)` 可忽略，且本项目不含镜像力
势垒降低、没有场致 Schottky 增强），25→125 °C 变化 4.3×10⁷ 倍，与
`A*T²exp(−E_B/kT)` 的解析比 4.35×10⁷ 吻合。

### 已知问题（本次未处理）

- CLI 的进度/摘要输出含 `A/cm²`，在 GBK 代码页的 Windows 控制台上抛
  `UnicodeEncodeError`（`__main__.py` 的 `typer.echo`）。与本次改动无关，
  临时绕法是 `PYTHONIOENCODING=utf-8`。
