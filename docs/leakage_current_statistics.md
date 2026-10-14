# 本项目漏电流统计方法说明

> Codex 编写日期：2026-09-08。本文为新增说明文档。
>
> 依据：[PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)、[物理模型规格](physics_model.md)、[速率方程规格](rate_equations.md)，并对照当前统计代码与已有测试编写。本文说明实际计算方法及其适用条件；本次进行了静态核对，未运行新的仿真或测试。

## 1. 计算结果的整体结构

本项目先在一个固定缺陷实现上，用 KMC 采样陷阱占据状态及其持续时间，计算陷阱相关电流；再加入直接在电极之间输运的 R1 电流。配置多个随机种子时，对各实现的电流密度取算术平均。

单个实现的主输出为：

$$
I_{\mathrm{total}}
=
I_{\mathrm{direct/FN}}
+
I_{\mathrm{thermionic}}
+
I_{\mathrm{TAT}}
+
I_{\mathrm{PF}},
\qquad
J=\frac{I_{\mathrm{total}}}{A},
\qquad
A=L_xL_y.
$$

这里有两层不同的平均：

| 层次 | 平均对象 | 权重 |
|---|---|---|
| 单个实现内部 | KMC 访问的各个占据组态 | 各组态实际持续的模拟时间 |
| 多个缺陷实现之间 | 每个实现已经算好的电流密度 | 每个实现等权 |

主输出 `current` 使用条件稳态通量估计量；程序同时保留瞬时通量平均和净事件计数，用于交叉检查。

## 2. 电流的物理定义、符号和单位

设测量窗口内，正极净接收的电子数为：

$$
\Delta N_a=N_{\mathrm{到达正极}}-N_{\mathrm{离开正极}}.
$$

对应的电子输运电流按以下口径报告：

$$
I=q\frac{\Delta N_a}{T_{\mathrm m}},
\qquad q=e>0.
$$

项目规定：**正号表示电子从负极流向正极**。这是结果字段采用的方向约定；若用同一空间坐标表示传统电流矢量，其方向与电子运动方向相反。

`T_m` 是物理模拟时间，单位 s。程序运行所用的电脑时间不参与电流计算。这里统计的是输运电流，没有计算电压改变时的电容位移电流。

内部使用 SI 单位：

| 物理量 | 内部单位 | 常见输出单位 |
|---|---|---|
| 电流 `current` | A | A |
| 电极面积 `area` | m² | m² |
| 电流密度 `current_density` | A/m² | CLI、CSV 中为 A/cm² |
| 测量时间 `elapsed_time` | s | s |

换算关系为：

$$
J\,[\mathrm{A/cm^2}]
=
10^{-4}J\,[\mathrm{A/m^2}].
$$

电流密度除以的是电极面积 `Lx × Ly`。膜厚通过电场、势垒和跃迁速率影响结果。

## 3. KMC 如何产生统计时间

实现位置：[kmc.py](../src/kmc_leakage/kmc.py) 中的 `KMCSimulation.run()`。

### 3.1 事件速率与随机时间步

每个陷阱最多占据一个电子，记其占据状态为：

$$
n_i\in\{0,1\}.
$$

程序由当前组态计算电势、电场和跃迁速率，然后构造允许发生的事件：

| 事件 | 陷阱占据条件 |
|---|---|
| 电极 → 陷阱俘获 | 目标陷阱为空，乘以 \(1-n_i\) |
| 陷阱 → 电极逃逸或 PF 发射 | 源陷阱已占据，乘以 \(n_i\) |
| 陷阱 \(i\) → 陷阱 \(j\) | 源已占据且目标为空，乘以 \(n_i(1-n_j)\) |

设允许事件的总速率为：

$$
R_{\mathrm{tot}}=\sum_j R_j.
$$

Gillespie 方法抽取均匀随机数 \(u_1,u_2\)，用下式推进时间，并按各事件的速率占比选择下一事件：

$$
\Delta t=-\frac{\ln u_1}{R_{\mathrm{tot}}},
\qquad
P(j)=\frac{R_j}{R_{\mathrm{tot}}}.
$$

这里的事件表包含陷阱相关过程。R1 在初始化时单独求值，不参与上述总速率。

程序先用**事件发生前的组态**累计该段 \(\Delta t\) 内的通量和占据，再执行事件。这样统计的状态与其持续时间相对应。

### 3.2 预热期与测量窗口

初始陷阱全部为空。前一部分事件用作预热，预热事件数为：

$$
N_{\mathrm{burn}}
=
\operatorname{int}
\left(N_{\mathrm{events}}f_{\mathrm{burn}}\right).
$$

对于正常的非负配置，这相当于向下取整。预热期间仍然更新占据状态，但不累计测量结果。

例如当前 [single_run.yaml](../examples/single_run.yaml) 使用：

~~~yaml
simulation:
  n_events: 20000
  burn_in_fraction: 0.2
~~~

完整执行时，前 4000 个事件为预热，后 16000 个事件用于测量。对应的测量时长为：

$$
T_{\mathrm m}=\sum_{k\in\mathrm{测量窗口}}\Delta t_k.
$$

`elapsed_time` 保存的就是这个时长，预热时间不包含在内。由于每步时间随机且随速率变化，20% 的预热事件并不对应固定的 20% 物理时间。

任何状态量 \(g\) 的时间平均都采用：

$$
\langle g\rangle_t
=
\frac{\sum_k g(s_k)\Delta t_k}{T_{\mathrm m}}.
$$

按事件次数对各组态等权平均会过度强调停留时间短、跳转频繁的状态。

当前程序按指定事件数结束，也会在总速率非正时提前结束；没有自动稳态判定或按目标误差自动延长采样的逻辑。因此，完成指定事件数后仍需检查预热与采样是否充分。

### 3.3 快陷阱移出事件预算（可选，ADR-023）

贴电极的陷阱与电极交换电子的速率可达 1e9–1e10 s⁻¹，\(R_{\mathrm{tot}}\) 几乎全由它们贡献，固定事件数下 \(T_{\mathrm m}\) 只有几十纳秒。配置 `simulation.fast_trap_rate_s` \(=R_{\mathrm{cut}}\) 后，全空组态下电极交换总速率 \(\Gamma_i\ge R_{\mathrm{cut}}\) 的陷阱（集合 \(F\)）退出事件表：

- 它们的占据取与当前慢组态自洽的条件稳态 \(\hat p_i\)（6.2 的公式，每步开头迭代到变化 \(\le 10^{-3}\)），进入静电叠加与跃迁的 Pauli 阻塞；
- 快→慢、慢→快的跃迁仍是事件，速率分别乘 \(\hat p_a\) 与 \(1-\hat p_b\)，执行时只改慢陷阱一端；快↔快跃迁不进事件表；
- \(R_{\mathrm{tot}}\) 只剩慢过程，同样的事件数覆盖的物理时间长得多。

要求快陷阱在两次事件之间来得及弛豫：\(\min_{i\in F}\Gamma_i\big/(N_{\mathrm{events}}/T_{\mathrm m})<10\) 时判未通过验证。

## 4. 第一种估计量：正极净事件计数

对应字段：`current_counts`。

每次真正执行一个与正极有关的事件，净电子计数按下表更新：

| 实际事件 | 对正极净电子计数的贡献 |
|---|---:|
| 正极 → 陷阱，R2 / R4 俘获 | −1 |
| 陷阱 → 正极，R3 / R5 逃逸 | +1 |
| 陷阱 → 导带，R8 PF 发射 | +1 |
| 负极 ↔ 陷阱 | 0 |
| 陷阱 ↔ 陷阱 | 0 |

R8 计为 +1，是因为当前模型把发射到导带的电子按立即被正极收集处理。这一收集假设对应项目主要研究的正偏压工况。

计数电流为：

$$
I_{\mathrm{counts}}
=
q\frac{N_+-N_-}{T_{\mathrm m}}.
$$

`current_counts` 只包含实际执行了的陷阱相关事件。启用快陷阱（3.3）时，快陷阱的事件不再执行，它们的正极净电流 `current_fast` 按主估计量的条件稳态通量补上。与总电流比较时应使用（字段 `current_counts_total`）：

$$
I_{\mathrm{counts,total}}
=
I_{\mathrm{counts}}+I_{\mathrm{fast}}+I_{\mathrm{R1}},
\qquad
I_{\mathrm{R1}}
=
I_{\mathrm{direct/FN}}+I_{\mathrm{thermionic}}.
$$

全局守恒相应写成 \((I_C-I_A)\Delta t=q(\Delta N_{\mathrm{inside}}-N_{\mathrm{fast\to slow}})\)，\(N_{\mathrm{fast\to slow}}\)（`fast_exchange`）是快→慢减慢→快跃迁的净次数，未启用快陷阱时为 0。

`n_positive_events` 记录的是：

$$
N_{\mathrm{positive\ events}}=N_++N_-.
$$

它表示测量窗口内正极相关事件的总次数。一个电子在正极与陷阱之间反复进出，会增加这个次数，而净输运可能很小。

有限窗口内，计数电流以 \(q/T_{\mathrm m}\) 为一个电子对应的变化量。漏电流很小时，窗口内可能没有足够的净输运事件；大量正反事件的差也可能出现很强的统计涨落。这正是项目保留其他估计量的原因。

## 5. 第二种估计量：瞬时正极通量的时间平均

对应字段：`current_direct`。

定义第 \(i\) 个陷阱的正极侧速率：

$$
C_{a,i}=R_{2a,i}+R_{4a,i},
$$

$$
E_{a,i}^{\mathrm{TAT}}=R_{3a,i}+R_{5a,i},
\qquad
P_i=R_{8,i}.
$$

这些速率的单位均为 s⁻¹。下标 \(a\) 表示正极；\(P_i\) 是 PF 发射速率。

当前组态下，陷阱 \(i\) 对正极的瞬时净电子通量为：

$$
j_i(s)
=
n_i\left(E_{a,i}^{\mathrm{TAT}}+P_i\right)
-
(1-n_i)C_{a,i}.
$$

于是：

$$
I_{\mathrm{direct}}
=
I_{\mathrm{R1}}
+
\frac{q}{T_{\mathrm m}}
\sum_k \Delta t_k\sum_i j_i(s_k).
$$

它利用每个停留区间内的事件速率累计期望通量。对于很少实际发生的事件，只要该组态下存在非零速率，也会贡献统计值。

但这里仍需对两个方向的大通量相减，采样方差仍可能很大。`current_direct` 中的 “direct” 表示直接对瞬时通量求平均；直接/FN 隧穿这一物理机制对应的是 `by_mechanism["direct_fn"]`。

## 6. 主估计量：条件稳态通量的时间平均

实现位置：`KMCSimulation._hop_sums()` 和 `KMCSimulation._conditional_flux()`。

### 6.1 合并单陷阱的流入和流出速率

除正极侧速率外，定义负极侧速率：

$$
C_{c,i}=R_{2c,i}+R_{4c,i},
\qquad
E_{c,i}=R_{3c,i}+R_{5c,i}.
$$

令 \(h_{ij}\) 表示陷阱 \(i\to j\) 的未施加端点占据条件的跃迁速率。其他陷阱提供的流入和流出速率为：

$$
H_{\mathrm{in},i}
=
\sum_{j\ne i}n_jh_{ji},
\qquad
H_{\mathrm{out},i}
=
\sum_{j\ne i}(1-n_j)h_{ij}.
$$

代码只在近邻表保留的陷阱对上求和。当前陷阱间通道采用 R7，即 R6 的几何因子乘以 Miller-Abrahams 能量因子。

陷阱 \(i\) 的总俘获、总逃逸速率及两者之和为：

$$
C_i=C_{c,i}+C_{a,i}+H_{\mathrm{in},i},
$$

$$
E_i=E_{c,i}+E_{a,i}^{\mathrm{TAT}}+P_i+H_{\mathrm{out},i},
\qquad
D_i=C_i+E_i.
$$

这里 \(E_i\) 表示逃逸速率，单位 s⁻¹；陷阱深度在项目中记为 \(E_D\)，两者应区分。

### 6.2 条件稳态占据概率

把周围环境及上述速率暂时固定，单陷阱占据概率满足：

$$
\frac{\mathrm dp_i}{\mathrm dt}
=
(1-p_i)C_i-p_iE_i.
$$

令其达到局部稳态，得到：

$$
\hat p_i=\frac{C_i}{C_i+E_i}=\frac{C_i}{D_i}.
$$

将 \(\hat p_i\) 代入正极净通量：

$$
\hat j_i
=
\hat p_i(E_{a,i}^{\mathrm{TAT}}+P_i)
-
(1-\hat p_i)C_{a,i}.
$$

展开分子并消去正极俘获与逃逸之间的共同项，得到代码采用的总通量形式：

$$
\boxed{
\hat j_i
=
\frac{
(C_{c,i}+H_{\mathrm{in},i})(E_{a,i}^{\mathrm{TAT}}+P_i)
-
(E_{c,i}+H_{\mathrm{out},i})C_{a,i}
}{D_i}
}.
$$

这样可以在代数层面消去同一正极与陷阱之间快速往返的共同项，减少估计净漏电流时的涨落。若 \(D_i=0\)，代码将该陷阱的条件占据和通量贡献设为 0。

KMC 仍负责采样组态：其他陷阱的占据会改变静电环境、跃迁速率和可用目标空位，所以每一步的 \(\hat j_i\) 都可能不同。

### 6.3 TAT 与 PF 如何分别统计

代码将上述总通量分为两项：

$$
\hat j_i^{\mathrm{TAT}}
=
\frac{
(C_{c,i}+H_{\mathrm{in},i})E_{a,i}^{\mathrm{TAT}}
-
(E_{c,i}+H_{\mathrm{out},i}+P_i)C_{a,i}
}{D_i},
$$

$$
\hat j_i^{\mathrm{PF}}
=
\frac{
(C_{c,i}+C_{a,i}+H_{\mathrm{in},i})P_i
}{D_i}
=
\hat p_iP_i.
$$

两式相加严格等于 \(\hat j_i\)。分别进行时间积分后：

$$
I_{\mathrm{TAT}}
=
\frac{q}{T_{\mathrm m}}
\sum_k\Delta t_k\sum_i\hat j_i^{\mathrm{TAT}}(s_k),
$$

$$
I_{\mathrm{PF}}
=
\frac{q}{T_{\mathrm m}}
\sum_k\Delta t_k\sum_i\hat j_i^{\mathrm{PF}}(s_k).
$$

机制归类按与收集电极交换电子的末端通道进行：R3/R5 向正极逃逸归入 TAT，R8 发射后收集归入 PF；正极通过 R2/R4 俘获向陷阱提供电子，对 TAT 记负贡献。

因此，分量是带方向的净贡献。TAT 分量可以为负，机制占比也不必是 0 到 1 之间的概率。例如电子从正极被俘获后通过 PF 返回正极，分别对 TAT 和 PF 记负、正贡献，两者在总电流中抵消。

### 6.4 这一估计量的适用条件

源码将其称为条件期望或 Rao-Blackwell 化估计量。这里需要区分两个数学步骤：

| 步骤 | 性质 |
|---|---|
| 对固定 \(C_i,E_i\) 求两态系统的稳态占据，并代数消去共同项 | 可直接由上述方程推导 |
| 在相互耦合、环境随时间变化的 KMC 系统中，用 \(\hat p_i\) 替换实际占据来估计平均通量 | 需要验证局部稳态近似 |

尤其需要陷阱自身占据的弛豫时间相对环境变化足够短，并且冻结速率的处理足够准确。对于一般的耦合跃迁网络，不能仅凭上述代数推导就宣称主估计量严格无偏。

增大采样长度可以减小随机误差；近似本身若存在系统偏差，需要通过与净计数、瞬时通量等结果比较来识别。

### 6.5 强耦合簇：把条件稳态推广到几个陷阱（ADR-022）

实现位置：`clusters.py`、`KMCSimulation._trap_flux()`、`KMCSimulation._cluster_flux()`。

6.4 的前提在强耦合的陷阱对上不成立：\(H_{\mathrm{in},i}\) 里的邻居占据 \(n_j\) 是采样值，而邻居 \(j\) 与 \(i\) 之间的跃迁和它们自身的电极交换一样快。结果是两类误差：测量窗口很短时，\(j\) 的稀有填充落进窗口就高估（I-V 尖峰）、没落进来就漏算；窗口再长，\(\hat p_i\) 也丢掉了 \(n_i\) 与 \(n_j\) 之间的关联（强耦合对上实测只剩精确值的 1–7%）。

因此每个工作点初始化时，把满足

$$
\frac{h_{ab}}{\min(\Gamma_a,\Gamma_b)}\ge 10^{-3}
$$

的陷阱对连成簇（\(\Gamma_i\) 为陷阱 \(i\) 与电极交换电子的总速率，上限 8 个陷阱）。测量期每步对簇 \(K\) 枚举簇内全部 \(2^k\) 个组态 \(s\)，按各组态的真实电荷分布重算簇成员的速率，簇外占据取当前组态作为环境，解簇内主方程稳态 \(\pi_K\)，成员 \(i\) 的正极净通量为

$$
\phi_i=\sum_s \pi_K(s)\left[n_i(s)\left(E_{a,i}^{\mathrm{TAT}}+P_i\right)(s)-\left(1-n_i(s)\right)C_{a,i}(s)\right].
$$

计算时这一式与"其余通道的净流入"两种等价写法取总量较小的一种，避免贴电极的成员放大舍入误差。单陷阱就是 \(k=1\) 的特例，仍走 6.2–6.3 的公式。孤立簇上结果与 `master_equation.py` 的精确稳态一致，且与轨迹无关。

## 7. R1 电流如何加入总结果

实现位置：`r1_electrode_current()`、`KMCSimulation._prepare()`，以及 [electrode_tunneling.py](../src/kmc_leakage/rates/electrode_tunneling.py)。

R1 描述电极到电极的输运。当前项目约定使用零空间电荷的外加偏压势垒，在每个工作点初始化时通过能量积分求值：

$$
I_{\mathrm{R1}}=qR_1.
$$

R1 不改变陷阱占据状态。它采用的势垒也不随 KMC 组态更新，因此可以单独计算后加入统计结果。

以路径上的最高势垒能量 \(E_{\max}\) 为阈值，积分分为：

| 能量区间 | 保存字段 |
|---|---|
| \(E_t<E_{\max}\)：直接/FN 隧穿 | `by_mechanism["direct_fn"]` |
| \(E_t\ge E_{\max}\)：热电子发射 | `by_mechanism["thermionic"]` |

这两项在固定工作点和材料参数下，与随机种子、缺陷实现及 KMC 事件数无关。它们没有 KMC 采样噪声，但仍受能量积分精度和模型参数影响。

三种估计量的比较口径如下：

| 字段或组合 | 陷阱相关部分 | 是否已含 R1 |
|---|---|---|
| `current` | 条件稳态通量的时间平均 | 是 |
| `current_direct` | 实际占据下的瞬时通量时间平均 | 是 |
| `current_counts` | 正极净事件数除以测量时间 | 否 |
| `current_counts + I_R1` | 同上 | 是 |

## 8. 按缺陷分解与占据概率

### 8.1 每个缺陷的电流贡献

`by_defect[i]` 保存：

$$
I_i
=
\frac{q}{T_{\mathrm m}}
\sum_k\Delta t_k
\left[
\hat j_i^{\mathrm{TAT}}(s_k)+
\hat j_i^{\mathrm{PF}}(s_k)
\right].
$$

它将末端净通量归到相应陷阱上，因此满足：

$$
\sum_i I_i=I_{\mathrm{TAT}}+I_{\mathrm{PF}},
$$

$$
\sum_i I_i+I_{\mathrm{R1}}=I_{\mathrm{total}}.
$$

这一分解可用来寻找贡献较大的陷阱，也可按位置求和形成空间分箱。沿一条路径经过的内部陷阱跃迁会影响输运，但不会每经过一个陷阱就额外计一次到达正极的电子。

### 8.2 两种平均占据

程序同时保存：

$$
\overline n_i
=
\frac{\sum_k n_i(s_k)\Delta t_k}{T_{\mathrm m}}
\quad\longrightarrow\quad
\texttt{occupancy[i]},
$$

$$
\overline{\hat p}_i
=
\frac{\sum_k\hat p_i(s_k)\Delta t_k}{T_{\mathrm m}}
\quad\longrightarrow\quad
\texttt{occupancy\_mean\_field[i]}.
$$

两者接近，是局部稳态近似的诊断信号。单凭平均占据接近仍不能证明电流精确，因为电流还涉及占据与状态相关速率之间的相关性。

## 9. 多个随机种子的系综平均

实现位置：[ensemble.py](../src/kmc_leakage/ensemble.py)。

`ensemble.count` 指定缺陷实现的个数 M，每次运行随机抽取 M 个互异种子。一个种子决定缺陷位置、随机能级以及 KMC 轨迹；不同实现分别完成统计后，在同一个工作点平均。

设第 \(m\) 个实现的电流密度为 \(J_m\)，共 \(M\) 个实现，则：

$$
\overline J=\frac{1}{M}\sum_{m=1}^{M}J_m.
$$

对于 \(M\ge2\)，实现间的样本标准差与平均值标准误分别为：

$$
s_J
=
\sqrt{
\frac{1}{M-1}
\sum_{m=1}^{M}(J_m-\overline J)^2
},
\qquad
\mathrm{SEM}=\frac{s_J}{\sqrt M}.
$$

代码使用 `std(ddof=1)`。输出字段为 `current_density` 和 `current_density_sem`。

这一平均有以下含义：

| 项目 | 解释 |
|---|---|
| 算术平均 | 对同一面积的不同缺陷实现等权平均，符合项目规定的电流可加口径 |
| 各实现测量时间不同 | 先分别求 \(J_m\)，再等权平均；将计数与时长跨实现合并会改变权重 |
| SEM 的来源 | 当前采样方案下，包含缺陷实现差异与有限轨迹采样造成的散布 |
| \(M=1\) | 代码返回 SEM=0，表示单样本无法估计实现间离散度 |
| 机制分量 | 各机制分别计算平均电流密度和 SEM |
| 总电流的 SEM | 直接由总电流样本计算；各机制之间可能相关，不能直接相加各机制 SEM |
| R1 的 SEM | 固定工作点下各实现的 R1 相同，其采样标准误理论上为 0 |

SEM 反映的是平均值的采样不确定度，不能覆盖模型近似、参数误差和未充分预热造成的偏差。

注意单位区别：`SimulationResult.by_mechanism` 是 **A**；`SweepResult`、`EnsembleResult` 和 `EnsembleSweepResult` 的同名字段是 **A/m²**。

## 10. I-V / I-T 扫描及结果读取

### 10.1 扫描过程

[sweep.py](../src/kmc_leakage/sweep.py) 在一个实现内复用同一套缺陷位置和能级，在各电压、温度点分别启动 KMC 统计。每个工作点都从全空占据开始并单独预热。

同一实现各工作点使用相同随机种子，因此曲线上不同点的随机误差可能相关。这种扫描描述各工作点的统计输运特性，没有继承上一个电压点的末态来模拟动态迟滞。

系综扫描最外层遍历种子，然后在每个工作点对多个实现取平均。结果矩阵形状为：

~~~text
(温度点数, 电压点数)
~~~

### 10.2 CSV 字段

| 单实现扫描 | 系综扫描 | 含义 |
|---|---|---|
| `J_total_A_cm2` | `J_mean_A_cm2` | 总电流密度或其系综平均 |
| — | `J_sem_A_cm2` | 总电流密度平均值的标准误 |
| `J_direct_fn_A_cm2` | 同名字段 | 直接/FN 隧穿分量 |
| `J_thermionic_A_cm2` | 同名字段 | 热电子发射分量 |
| `J_tat_A_cm2` | 同名字段 | TAT 分量 |
| `J_pf_A_cm2` | 同名字段 | PF 分量 |
| — | `J_seed<种子>_A_cm2` | 各实现的原始总电流密度 |

当前系综 CSV 保存各机制的平均值；各机制 SEM 可从 Python 结果对象读取。CSV 没有保存三种估计量的完整交叉检查数据。

### 10.3 从 Python 读取交叉检查量

以下是使用示例，本次整理文档未执行它：

~~~python
from kmc_leakage.io import load_config
from kmc_leakage.ensemble import run_ensemble
from scipy import constants

config = load_config("examples/single_run.yaml")
point = run_ensemble(config)
per_cm2 = constants.centi ** 2

print("J_mean [A/cm²]:", point.current_density * per_cm2)
print("SEM [A/cm²]:", point.current_density_sem * per_cm2)

result = point.results[0]
r1 = (
    result.by_mechanism["direct_fn"]
    + result.by_mechanism["thermionic"]
)

print("条件稳态估计 [A]:", result.current)
print("瞬时通量估计 [A]:", result.current_direct)
print("净事件估计，含 R1 [A]:", result.current_counts + r1)
print("测量窗口 [s]:", result.elapsed_time)
print("正极相关事件数:", result.n_positive_events)
~~~

## 11. 如何判断统计结果是否可信

以下检查针对电流统计方法。它们是验证建议，不表示本次已执行相应数值验证。

| 检查 | 具体做法与含义 |
|---|---|
| 有效测量窗口 | 确认 `elapsed_time > 0`。没有有效测量时间时，不能把返回的陷阱电流零值当作物理结论 |
| 预热充分性 | 增加预热长度并保留足够测量事件，比较电流和占据是否稳定；当前程序不会自动判断 |
| 轨迹长度 | 增大 `n_events`，检查同一缺陷实现的结果随采样长度是否稳定 |
| 三种估计量 | 比较 `current`、`current_direct` 与 `current_counts + I_R1`，先统一 R1 口径 |
| 局部稳态近似 | 对照两种平均占据，并结合电流交叉检查；平均占据接近只提供部分证据 |
| 分解自洽 | 四类机制之和等于总电流；缺陷贡献之和等于 TAT + PF，允许浮点舍入误差 |
| 实现间散布 | 增加独立种子，观察均值和 SEM 的稳定性，同时保留各实现原始结果 |
| 参数适用范围 | 改变材料、浓度、温度或偏压后，重新检查局部稳态近似及导带电子收集假设 |

有两项尤其影响结果解释：

1. **正极事件很多，只说明两方向交换频繁。** 当正反通量接近时，净电流仍可能难以统计；`n_positive_events` 本身不是有效独立输运样本数。
2. **PF 收集有方向假设。** 当前 R8 始终按正极收集处理，且未建模导带到陷阱的反向俘获。零偏压残余电流已列入项目模型局限；反偏和近热平衡工况需要额外验证。

当前已有测试的覆盖范围也应准确理解：

- [test_kmc_consistency.py](../tests/test_kmc_consistency.py) 检查机制与缺陷分解、占据概率、R1 独立性以及估计量之间的一致性。
- 其中估计量一致性用例采用固定的 4 V、2000 个事件、seed=13，使用 `rel=0.3` 的比较容差。现有 README 提到的“5% 以内”与该断言不一致；不能将它作为当前测试保证。
- 该用例直接比较 `current_counts` 与总电流，依赖该工况下 R1 很小。一般工况应按本文第 7 节补上 R1 后比较。
- [test_ensemble.py](../tests/test_ensemble.py) 检查算术平均、样本标准误及系综结果结构。有限用例的存在不构成所有工作点收敛的证明。

## 12. 代码对应关系

| 内容 | 入口 |
|---|---|
| 单点参数、种子选择、快陷阱判据 | [config.py](../src/kmc_leakage/config.py)：`SimulationConfig`（含 `fast_trap_rate`）、`Config.seeds` |
| 事件推进、预热、时间加权、计数及结果组合 | [kmc.py](../src/kmc_leakage/kmc.py)：`KMCSimulation.run` |
| 陷阱间流入与流出 | [kmc.py](../src/kmc_leakage/kmc.py)：`_hop_sums` |
| 条件稳态通量与 TAT/PF 分解 | [kmc.py](../src/kmc_leakage/kmc.py)：`_conditional_flux`、`_trap_flux` |
| 强耦合簇的识别与簇内条件稳态 | [clusters.py](../src/kmc_leakage/clusters.py)；[kmc.py](../src/kmc_leakage/kmc.py)：`_build_clusters`、`_cluster_flux` |
| R1 积分与两类分量 | [electrode_tunneling.py](../src/kmc_leakage/rates/electrode_tunneling.py)：`r1_electrode_electrode` |
| 单实现扫描与 CSV | [sweep.py](../src/kmc_leakage/sweep.py)：`run_sweep`、`write_csv` |
| 系综平均、SEM 与 CSV | [ensemble.py](../src/kmc_leakage/ensemble.py)：`_mean_sem`、`run_ensemble`、`run_ensemble_sweep`、`write_ensemble_csv` |

## Codex 文档记录

- 新增 `docs/leakage_current_statistics.md`。
- 整理电流定义、三种估计量的公式、时间加权、机制及缺陷分解、系综平均、单位和结果字段。
- 明确条件稳态估计的适用条件、统计收敛检查方法及 README 与当前测试容差之间的差异。
- 本次操作仅新增本文档，未修改已有文档或项目源代码。