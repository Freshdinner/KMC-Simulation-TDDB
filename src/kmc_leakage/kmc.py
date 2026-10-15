"""KMC 主循环（Gillespie direct method）与电流统计。

算法流程见 docs/physics_model.md 的"KMC 算法"一节。

三点实现说明：

1. **静电求解的线性叠加。** 泊松方程是线性的，缺陷位置在一次实现里固定
   不变，因此初始化时解 N+1 次泊松方程（外加偏压 1 次 + 每个缺陷单位电荷
   各 1 次；有固定电荷 q_fix 时再加 1 次，它们与组态无关），之后任一占据组态的
   电势/电场都由这些解按占据数线性叠加得到，主循环里不再需要解泊松方程。
   这既精确又把每步开销降到矩阵-向量乘。
   缺陷 i 自身电荷在 i 处的贡献（CIC 自作用）就是这些矩阵的对角元，按
   项目"不考虑镜像力"的约定整项扣除。

2. **R1 一次性求值，不进主循环。** 电极↔电极隧穿不改变占据组态，把它放进
   事件表只会产生大量"空转"事件。它走的是理想平面势垒的物理图像，与
   Richardson-Schottky 对照式同口径，所以势垒剖面取零空间电荷的外加偏压解
   （E_C(z) = E_B - e·(U+V_bi)·z/Lz，严格线性）：初始化时按公式求值一次，
   run() 末尾把 I_R1 = q·R1 直接加进 by_mechanism。结果与 KMC 轨迹无关，
   不随 seed、n_events 变化，也不含采样噪声。

3. **电流估计量。** 稳态净电流是正极侧两个大通量之差：正极→陷阱的俘获
   与陷阱→正极的逃逸，二者在稳态下几乎精确抵消，直接数事件（或直接对
   瞬时通量做时间平均）会被这个相消淹没——本项目参数下噪声比信号大 2–3
   个数量级。把单个陷阱的条件稳态占据 p̂_i = C_i/(C_i+E_i) 代入净通量表
   达式后，相消项 C_a·E_a 解析地约掉：

       flux_i = [(C_c+H_in)·E_a - (E_c+H_out)·C_a] / (C_i+E_i)

   C_c/C_a 为负极/正极侧俘获总速率，E_c/E_a 为向负极/正极的逃逸总速率
   （E_a 含 R8），H_in/H_out 为陷阱间跃迁的进出速率。这是把 KMC 采到的
   组态做 Rao-Blackwell 化：陷阱占据在环境变化的时间尺度上快速弛豫时
   精确，本项目参数下这一条充分满足。KMC 仍然在做实事——组态决定静电
   环境、Θ 因子与跃迁的 Pauli 阻塞。

   直接估计量（瞬时通量时间平均）与数事件的结果同时给出，作为交叉检验：
   统计量足够时三者一致。KMC 采样得到的占据概率与条件稳态 p̂ 的对比
   （`occupancy` vs `occupancy_mean_field`）则直接检验上述假设是否成立。

4. **陷阱间跃迁按近邻表存储。** R6/R7 的几何因子沿路径按材料分段衰减（同质时
   e 折长度只有 r_D/2，参考预设下 0.083 nm），远小于缺陷平均间距，所以稠密
   (N,N) 跃迁矩阵在双精度下几乎全是零。事件表只列截断半径内的有序对
   （`defects.neighbor_pairs`），截断半径按叠层所有材料里最慢的衰减 α_min 与
   "被丢弃的总速率上界 < 机制表最小速率"推出（`_hop_cutoff_radius`，ADR-013/F05），
   不是写死的长度。内存与每步开销因此从 O(N²) 降到 O(近邻对数) ∝ O(N)。

5. **强耦合簇的条件稳态（ADR-022）。** 说明 3 的估计量只把陷阱**自身**的占据解析
   平均掉，H_in/H_out 里的邻居占据仍是 KMC 采样值。两个陷阱间的跃迁与它们自身的
   电极交换一样快时（如负极侧陷阱 a 以 3e6 s⁻¹ 慢慢填充、再以 3e7 s⁻¹ 跳给紧挨着
   的 b，b 以 1e8 s⁻¹ 放给正极），b 的电流取决于 a 的稀有填充是否落在测量窗口里：
   落进来就高估、形成 I-V 尖峰，没落进来就整条通路漏算。所以工作点初始化时，把
   跃迁速率 ≥ `_CLUSTER_COUPLING` × 两端电极交换速率较小者的陷阱对连成强耦合簇
   （`clusters.coupled_clusters`，全空组态下的速率），测量时对每个簇枚举簇内全部
   2^k 个占据组态、按各组态的真实电荷分布重算簇成员速率，解簇内主方程稳态
   （`clusters.cluster_steady_state`），簇外占据作为给定环境。簇成员的通量与 p̂
   由此给出，不再依赖簇内的采样；单陷阱仍走说明 3 的公式，逐位不变。孤立簇的
   结果与 `master_equation.py` 的精确稳态一致。

6. **快陷阱移出事件预算（ADR-023，可选）。** 贴着电极的陷阱与电极交换电子可达
   1e9–1e10 s⁻¹，Gillespie 的事件几乎全花在它们的来回翻转上，n_events 个事件只推进
   几十纳秒，慢变量（库仑环境、慢陷阱的填充）采不到。配置 `simulation.fast_trap_rate_s`
   = R_cut 后，全空组态下电极交换总速率 Γ_i ≥ R_cut 的陷阱记为快陷阱：它们的俘获/
   逃逸/PF 事件移出事件表，占据不再取 0/1，改用与当前慢组态自洽的条件稳态 p̂（每步
   迭代到变化 ≤ `_FAST_P_TOL`），进入静电叠加与跃迁的 Pauli 阻塞；快→慢、慢→快的
   跃迁仍是事件，速率乘 p̂ 或 1−p̂，执行时只改慢陷阱一端；快↔快跃迁移出。这是
   Jegert 等人 R_cutoff 方法的变体：原文用此前迭代的时间平均占据，这里用随组态更新
   的 p̂。估计量不受影响：快陷阱的电流由说明 3 的公式给出（本来就不依赖采样），同时
   补进数事件与瞬时通量两个交叉估计量；快陷阱照样可以是强耦合簇的成员，簇内按组态
   精确枚举。所以近似只进入**采样的动力学**——慢陷阱感受到的是快陷阱的平均电荷——要求
   快陷阱比剩余事件快得多：测量窗口内平均事件速率与 min Γ_F 之比低于
   `_FAST_SEPARATION_MIN` 时判未通过验证。不配置时逐位退回原算法。

统计口径按"电子抵达收集电极（或进入导带后被收集）的最后一步事件类型"
归类，见 docs/rate_equations.md 的机制对照表。R8 发射到导带的电子按
"立即被正极收集"处理（U>0 时导带电子向正极漂移）。

TDDB 用到的两个可选输入（ADR-060），都不给时与漏电流侧逐位相同：

- **逐缺陷温度** `temperature`：缺陷 i 的电极速率（声子线型、费米分布、R5 的玻尔兹曼因子、R8）取 T_i，
  陷阱间跃迁的 MA 因子取两端温度的平均；R1 与 PF 收集概率仍取工作点温度（电极温度）。
- **接电极的导体** `conductors`（`Conductors`）：成员是电极的延伸，不在缺陷集合里；近旁陷阱与成员交换电子的速率
  并进对应电极侧的 R2/R3 块。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import constants
from scipy.spatial import cKDTree

from .bands import (
    positive_fermi_level,
    conduction_band_edge,
    electrode_band_bottom,
    fermi_dirac,
    free_electron_dos,
    r1_band_bottom,
    top_boundary_potential,
    trap_levels,
)
from .clusters import (
    cluster_steady_state,
    coupled_clusters,
    coupling_strength,
    enumerate_occupations,
)
from .config import Config, DeviceConfig
from .defects import DefectEnsemble, generate_defects, hop_cutoff_radius, neighbor_pairs
from .electrostatics import PoissonSolver
from .rates import (
    miller_abrahams_factor,
    multiphonon_coupling,
    phonon_transition_probability,
    r1_electrode_electrode,
    r2_elastic_capture,
    r3_elastic_emission,
    r4_multiphonon_capture,
    r5_multiphonon_emission,
    r6_elastic_hop,
    r8_pf_emission,
    segmented_overlap_factor,
    wkb_transmission,
)

# 四类统计口径（见 docs/rate_equations.md 的机制对照表）
MECHANISMS = ("direct_fn", "thermionic", "tat", "pf")
MECHANISM_LABELS = {
    "direct_fn": "直接/FN隧穿",
    "thermionic": "热电子发射",
    "tat": "TAT",
    "pf": "PF发射",
}

# 数值参数（取值依据见 tests/test_numerics_convergence.py 的收敛性检验）
_N_PHONON = 12  # R4/R5 求和保留的声子数 p = 0..11（p≥7 起贡献 <1e-4）
_TRAP_PATH_POINTS = 65  # 电极↔陷阱 WKB 路径采样点
_R1_PATH_POINTS = 129  # R1 全膜厚 WKB 路径采样点
_R1_ENERGY_POINTS = 128  # R1 能量积分点数（[0, 阈值] 与阈值以上各自；费米能级以下同间距）
_PF_BARRIER_POINTS = 257  # F03 阻挡层 PF 收集势垒的 WKB 采样点
# 强耦合簇（ADR-022，模块文档说明 5）：跃迁速率 ≥ 此比例 × 两端电极交换速率较小者即连边。
# t2_2nm 叠层三个实现 × 三个工况实测：取 1e-3 时每个实现 10–14 个多陷阱簇、最大 6 个陷阱
_CLUSTER_COUPLING = 1e-3
_CLUSTER_MAX_SIZE = 8  # 簇内组态数 2^k 的上限（256）；超限的分量在内部提高阈值再切分

# F04 每个工作点的"未通过验证"判据阈值（ADR-021 F04，见 _validate_result）
_MIN_MEASURED_EVENTS = 100  # 测量窗口至少要有这么多事件才算采到
_MEAN_FIELD_TOL = 0.1  # KMC 占据与条件稳态 p̂ 的平均偏差上限（局部稳态/条件通量成立的前提）
_COUNTS_SUFFICIENT = 50  # 正极事件数达到此值，净计数交叉校验才纳入硬门槛（否则统计为空、跳过）
_COUNTS_REL_TOL = 0.5  # 计数够用时 |J_counts+R1 − J_cond| 相对 |J_cond| 的上限（计数本身仍噪声大）
_N_BATCHES = 10  # 测量窗口按事件数等分的批次数，批次均值法估计电流的统计误差
# 批次均值法给出的电流相对标准误上限。实测正常点 ≤0.001（条件稳态估计量很平滑），
# 跃迁供给通路偶尔被填充造成的尖峰为 0.19–0.26（t2_2nm 叠层、seed 243366264）
_BATCH_REL_TOL = 0.05
# 快陷阱（ADR-023）：min Γ_F 至少是测量窗口内平均事件速率的这么多倍，快陷阱才来得及
# 在两次事件之间弛豫到 p̂（绝热消去成立）
_FAST_SEPARATION_MIN = 10.0
# 快陷阱 p̂ 与当前慢组态自洽迭代的收敛容差（占据概率的绝对变化）。t2_2nm 叠层 125 °C/4.3 V、
# R_cut=1e9 实测：1e-3 时平均每步 1.10 次 _rates、1e-6 时 1.93 次，两者 J 前 6 位相同
_FAST_P_TOL = 1e-3
_FAST_MAX_ITER = 20

# 事件块：每块长度 N，块内索引即缺陷编号。前四块是俘获，后五块是逃逸/发射。
_B_R2C, _B_R2A, _B_R4C, _B_R4A, _B_R3C, _B_R3A, _B_R5C, _B_R5A, _B_R8 = range(9)
_N_BLOCKS = 9
_N_CAPTURE_BLOCKS = 4
# 各块对"正极净流入电子数"的贡献（+1 到达正极，-1 由正极取走）
_POSITIVE_SIGN = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
# 各块对"负极净送入器件电子数"的贡献（+1 负极注入陷阱，-1 陷阱回到负极）；
# R8 只到正极，对负极记 0。negative_net − positive_net = ΔN_inside，全局守恒的基础（F04）
_NEGATIVE_SIGN = np.array([1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
_BLOCK_MECHANISM = ("tat",) * 8 + ("pf",)


@dataclass
class SimulationResult:
    """单点仿真结果。电流均为净值，正号 = 电子由负极流向正极。"""

    voltage: float  # [V]
    temperature: float  # [K]
    elapsed_time: float  # 测量窗口时长 [s]
    current: float  # [A]
    current_density: float  # [A/m^2]
    area: float  # 电极面积 [m^2]，把上面各项电流换成电流密度用
    by_mechanism: Dict[str, float]  # 各机制电流 [A]，四项之和 = current
    by_defect: np.ndarray  # 各缺陷电流贡献 [A]，之和 + R1 两项 = current
    by_layer: Dict[str, float]  # 分层归因 [A]（layer_<id> / interface_<id>），之和 = TAT+PF
    occupancy: np.ndarray  # KMC 采样得到的时间平均占据概率
    occupancy_mean_field: np.ndarray  # 条件稳态 p̂ 的时间平均，与上一项对照
    n_events: int  # 实际执行的事件数（预热 + 测量）
    measured_events: int  # 测量窗口内执行的事件数
    # --- 三个估计量与全局守恒（F04，ADR-021）---
    current_direct: float  # 瞬时正极通量的时间平均 [A]（相消未消去，方差大）
    current_counts: float  # 正极净事件电流 [A]（只数执行了的事件，不含 R1 与快陷阱）
    current_counts_negative: float  # 负极净事件电流 [A]（同上，即 I_C − I_R1）
    n_positive_events: int
    # 测量窗口内慢陷阱电子数变化 ΔN_inside。守恒：(I_C−I_A)·Δt = q·(ΔN_inside − fast_exchange)
    delta_n_inside: int
    # --- 收敛/验证状态（F04，ADR-021）---
    validated: bool  # 是否通过验证；未通过时不得作为可信稳态电流交付
    validation_notes: Tuple[str, ...]  # 未通过/存疑的具体原因
    # --- 快陷阱（ADR-023），未启用时全为 0 ---
    n_fast_traps: int = 0
    current_fast: float = 0.0  # 快陷阱的正极净电流 [A]（条件稳态通量，它们的事件不再执行）
    fast_exchange: int = 0  # 测量窗口内由快陷阱跳进慢陷阱的净电子数（跃迁事件记账）
    defects: Optional[DefectEnsemble] = field(repr=False, default=None)

    @property
    def current_counts_total(self) -> float:
        """数事件估计量 [A]：正极净事件电流 + 快陷阱电流 + R1（F04 交叉校验的口径）。"""
        r1 = self.by_mechanism["direct_fn"] + self.by_mechanism["thermionic"]
        return self.current_counts + self.current_fast + r1

    @property
    def mechanism_fractions(self) -> Dict[str, float]:
        """各机制占总电流的比例。"""
        if self.current == 0.0:
            return {name: 0.0 for name in MECHANISMS}
        return {name: value / self.current for name, value in self.by_mechanism.items()}


def by_layer_attribution(defects: DefectEnsemble, defect_flux: np.ndarray) -> Dict[str, float]:
    """把逐缺陷通量按 layer_id / interface_id 分组的端口电流贡献（ADR-014）。

    体缺陷归到 `layer_<id>`，界面缺陷归到 `interface_<id>`（两个界面各一栏、只计
    一次）。键按实际出现的层/界面动态生成；分组之和严格等于 `defect_flux.sum()`
    = I_TAT + I_PF，故 I_total = Σ_{ℓ,Γ} I_{ℓ,Γ} + I_R1（R1 单列，不进分层）。
    区别于 through_plane 截面通量（v1 不做）。
    """
    out: Dict[str, float] = {}
    layer_id = np.asarray(defects.layer_id)
    interface_id = np.asarray(defects.interface_id)
    for lid in np.unique(layer_id[layer_id >= 0]):
        out["layer_{}".format(int(lid))] = float(defect_flux[layer_id == lid].sum())
    for iid in np.unique(interface_id[interface_id >= 0]):
        out["interface_{}".format(int(iid))] = float(defect_flux[interface_id == iid].sum())
    return out


def batch_relative_error(charge: np.ndarray, duration: np.ndarray, current: float) -> float:
    """批次均值法估计的电流相对标准误。

    `charge[k]`、`duration[k]` 是第 k 个批次内缺陷通量的时间积分与批次时长，
    电流取比值估计 ΣQ/ΣT，标准误按比值估计量的方差公式
    SE = √[K/(K−1)·Σ(Q_k − Ī·T_k)²] / ΣT，Ī = ΣQ/ΣT，再除以总电流 `current`
    （含 R1，确定性部分不贡献误差）。批次间一致时误差小；某个低频事件（如一条
    跃迁供给通路偶尔被填充）只落在一两个批次里并主导电流时误差大。有效批次
    不足 2 个或总电流为 0 时返回 0（无从估计）。
    """
    used = duration > 0.0
    q, t = charge[used], duration[used]
    k = q.size
    if k < 2 or current == 0.0:
        return 0.0
    total_t = t.sum()
    mean_current = q.sum() / total_t
    se = np.sqrt(k / (k - 1) * np.sum((q - mean_current * t) ** 2)) / total_t
    return float(se / abs(current))


def _validate_result(
    n_defects: int,
    forward: bool,
    elapsed: float,
    measured_events: int,
    occ_mean_abs_dev: float,
    n_positive_events: int,
    current: float,
    current_counts_total: float,
    batch_rel_error: float,
    timescale_separation: float = np.inf,
) -> Tuple[bool, Tuple[str, ...]]:
    """按 ADR-021 F04 判定单点是否通过验证，返回 (validated, 原因列表)。

    硬门槛用可靠可算的判据：正向收集（U+V_bi>0）、测量窗口充分、KMC 占据与条件
    稳态 p̂ 接近（局部稳态/条件通量成立的前提）、批次均值法给出的电流相对标准误
    不超限（测量窗口内电流统计收敛）、启用快陷阱时时间尺度分离足够（min Γ_F 与
    平均事件速率之比，ADR-023）、机制分量闭合（构造保证）。净事件计数交叉
    校验 ΔJ=J_cond−(J_counts+R1) 噪声大，只有正极事件数足够（统计非空）时才纳入
    门槛，否则作诊断记录、不判失败（Q1 口径）。全器件无缺陷（R1-only）没有统计
    估计量要验证、R1 是确定性解，直接通过。
    """
    if n_defects == 0:
        return True, ()
    notes = []
    if not forward:
        notes.append("非正向收集（U+V_bi≤0），PF 收集方向未定义")
    if elapsed <= 0.0 or measured_events < _MIN_MEASURED_EVENTS:
        notes.append("测量窗口不足（events={}）".format(measured_events))
    if occ_mean_abs_dev >= _MEAN_FIELD_TOL:
        notes.append("占据与条件稳态 p̂ 偏差过大（{:.3f}≥{}）".format(occ_mean_abs_dev, _MEAN_FIELD_TOL))
    if batch_rel_error > _BATCH_REL_TOL:
        notes.append("批次间电流不一致，相对标准误 {:.2f}>{}".format(batch_rel_error, _BATCH_REL_TOL))
    if timescale_separation < _FAST_SEPARATION_MIN:
        notes.append(
            "快陷阱与剩余事件的时间尺度分离不足（{:.1f}<{}）".format(timescale_separation, _FAST_SEPARATION_MIN)
        )
    if n_positive_events >= _COUNTS_SUFFICIENT and current != 0.0:
        rel = abs(current_counts_total - current) / abs(current)
        if rel > _COUNTS_REL_TOL:
            notes.append("计数交叉校验超差（ΔJ/J={:.2f}>{}）".format(rel, _COUNTS_REL_TOL))
    return len(notes) == 0, tuple(notes)


def _band_on_path_steps(
    path: np.ndarray,
    z_nodes: np.ndarray,
    band_nodes: np.ndarray,
    iface_idx: np.ndarray,
    delta_Ec: np.ndarray,
) -> np.ndarray:
    """把节点导带底 `band_nodes` 插到路径 `path` 上，界面处保留真实台阶（ADR-012）。

    `band_nodes` 是 `z_nodes` 上的 E_C(z)，界面节点取的是**上侧**层的值（导带
    基准 `layer_indices_of` 用右开区间）。这里在每个界面节点前补一个下侧值
    `band_nodes[k]−ΔE_C`（同一 z 的双值），于是界面两侧各存一个 E_C、台阶不被
    插成斜坡。无内部界面（单层）时 `iface_idx` 为空，逐位退回
    `np.interp(path, z_nodes, band_nodes)`——单层路径不变。

    界面双值只改**采样**、不改积分网格（路径样点数固定），因此同材料多层与单层
    的 WKB 积分逐位一致（review F06：加界面节点到积分网格才会破坏同质退化）。
    """
    flat = np.asarray(path, dtype=float).ravel()
    z_aug, band_aug = _band_nodes_with_steps(z_nodes, band_nodes, iface_idx, delta_Ec)
    return np.interp(flat, z_aug, band_aug).reshape(np.shape(path))


def _band_nodes_with_steps(
    z_nodes: np.ndarray, band_nodes: np.ndarray, iface_idx: np.ndarray, delta_Ec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """在每个界面节点前补下侧值 `band−ΔE_C`（同一 z 的双值），返回 (z_aug, band_aug)。

    `band_nodes` 的最后一维对应 `z_nodes`，前面可以带批量维（每行一个组态），一次
    补完。无内部界面时原样返回。`_band_on_path_steps` 与簇内条件稳态共用。
    """
    if iface_idx.size == 0:
        return z_nodes, band_nodes
    z_aug = np.insert(z_nodes, iface_idx, z_nodes[iface_idx])
    band_aug = np.insert(band_nodes, iface_idx, band_nodes[..., iface_idx] - delta_Ec, axis=-1)
    return z_aug, band_aug


def _interface_step_nodes(device: DeviceConfig, z_nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """界面节点索引与各界面的 ΔE_C（分段能带台阶的插值用，ADR-012）。"""
    interfaces = device.stack.interface_positions
    if interfaces.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    hz = z_nodes[1] - z_nodes[0]
    idx = np.rint(interfaces / hz).astype(int)
    delta_Ec = np.array([device.stack.delta_Ec(j) for j in range(len(interfaces))])
    return idx, delta_Ec


def r1_electrode_current(
    device: DeviceConfig,
    voltage: float,
    temperature: float,
    z_nodes: np.ndarray,
    phi_bar: np.ndarray,
) -> Tuple[float, float]:
    """R1 的两个电流分量 (直接/FN隧穿, 热电子发射)，单位 A。
    """
    z = np.linspace(0.0, device.Lz, _R1_PATH_POINTS)
    band_nodes = conduction_band_edge(device, z_nodes, phi_bar)
    iface_idx, delta_Ec = _interface_step_nodes(device, z_nodes)
    band = _band_on_path_steps(z, z_nodes, band_nodes, iface_idx, delta_Ec)
    tunnel, thermionic = r1_electrode_electrode(
        z,
        band,
        device.area,
        voltage,
        temperature,
        device.negative.effective_mass,
        device.stack.m_di_at(z),
        r1_band_bottom(device, voltage),
        n_points=_R1_ENERGY_POINTS,
    )
    return constants.elementary_charge * tunnel, constants.elementary_charge * thermionic


def _blocking_layer_indices(band_refs: np.ndarray) -> np.ndarray:
    """叠层里的阻挡层：导带基准高于两侧相邻层的内部层
    """
    n = band_refs.shape[0]
    if n < 3:
        return np.zeros(0, dtype=int)
    interior = np.arange(1, n - 1)
    higher = (band_refs[interior] > band_refs[interior - 1]) & (
        band_refs[interior] > band_refs[interior + 1]
    )
    return interior[higher]


def pf_collection_transmission(
    delta_Ec: float,
    voltage_drop: float,
    thickness: float,
    m_di: float,
    temperature: float,
    n_points: int = _PF_BARRIER_POINTS,
) -> float:
    """阻挡层 PF 收集概率 P_A = min(1, T_WKB + exp(−ΔE_C/k_BT))（ADR-021 F03）。

    带偏压势垒剖面（正向、从阻挡层入口起算 x）::

        V(x) = ΔE_C − q·F·x,   F = Δφ/t_Al,   x ∈ [0, t_Al]

    ΔE_C 是入口处（发射侧 ZrO₂ → Al₂O₃）的导带台阶，入射能量取已热化到 Al₂O₃
    入口处 ZrO₂ 局部导带底（ADR-017 §1）。`voltage_drop` = 阻挡层上的无电荷分压
    |Δφ_Al| [V]。势垒随场倾斜成梯形/三角形，WKB 透射在这个未圆化的直线势垒上做
    （与 R1 同口径，不含镜像力）。热跃因子 exp(−ΔE_C/k_BT) 是越过入口台阶顶的热
    激发。零偏压（voltage_drop=0）退回矩形势垒。封顶 1 保证 0 ≤ P_A ≤ 1。
    """
    kT = constants.Boltzmann * temperature
    q = constants.elementary_charge
    x = np.linspace(0.0, thickness, n_points)
    barrier = delta_Ec - (q * voltage_drop / thickness) * x
    t_wkb = float(wkb_transmission(x, barrier, np.array(0.0), m_di))
    thermal = float(np.exp(-delta_Ec / kT))
    return min(1.0, t_wkb + thermal)


class ElectrostaticCoupling:
    """与偏压无关的静电预计算：缺陷之间的电势/电场耦合矩阵。

    `phi_matrix[i, j]` = 缺陷 i 处一个电子的电荷在缺陷 j 处产生的电势 [V]。
    对角元即 CIC 自作用项，使用时整项扣除。缺陷位置不变，故这些矩阵在整个
    扫描过程中都可复用。

    `field_matrix[:, i, j]` = 缺陷 i 的电荷在缺陷 j 处产生的电场矢量，按缺陷 j 的
    读场侧取值：体缺陷用常规中心差分、界面缺陷用单侧差分（`field_side`，
    ADR-017 §2 / review F08）。所有吃场的机制共用这一份场。变 κ 后的介电失配像势
    含在对角元里，与金属像势一同不计（ADR-011，不考虑镜像力）。
    """

    def __init__(
        self, solver: PoissonSolver, positions: np.ndarray, field_side: Optional[np.ndarray] = None
    ) -> None:
        n = positions.shape[0]
        if field_side is None:
            field_side = np.zeros(n, dtype=int)
        self.phi_matrix = np.empty((n, n))
        self.field_matrix = np.empty((3, n, n))
        self.phi_bar = np.empty((n, solver.nz + 1))
        for i in range(n):
            rho = solver.deposit(positions[i : i + 1], np.array([-constants.elementary_charge]))
            phi = solver.solve(rho, 0.0)
            self.phi_matrix[i] = solver.interpolate(phi, positions)
            self.field_matrix[:, i, :] = solver.field_vectors_at(phi, positions, field_side)
            self.phi_bar[i] = solver.lateral_average(phi)
        self.phi_self = np.diag(self.phi_matrix).copy()
        idx = np.arange(n)
        self.field_self = self.field_matrix[:, idx, idx].copy()


@dataclass(frozen=True)
class _Cluster:
    """一个强耦合簇在近邻表上的索引（工作点初始化时算好，每步复用）。

    `pairs` 是与簇有关的有序对在近邻表里的下标（簇内对 + 簇外→簇 + 簇→簇外），
    每步只对这些对重算跃迁速率；`*_cols` 是各类对在 `pairs` 里的位置，成员一律用
    簇内局部编号（0..k−1），簇外端点用全局缺陷编号。
    """

    members: np.ndarray  # (k,) 全局缺陷编号，升序
    states: np.ndarray  # (2^k, k) 簇内全部占据组态
    pairs: np.ndarray
    intra_cols: np.ndarray
    intra_a: np.ndarray
    intra_b: np.ndarray
    in_cols: np.ndarray
    in_member: np.ndarray
    in_source: np.ndarray
    out_cols: np.ndarray
    out_member: np.ndarray
    out_target: np.ndarray


@dataclass(frozen=True)
class _ClusterTable:
    """一个强耦合簇在给定环境占据下、簇内每个组态 s 的能级与速率（`_cluster_tables`）。

    成员按 `cluster.members` 的局部编号排列；跃迁速率按 `cluster.pairs` 的列排列。
    """

    cluster: _Cluster
    energy: np.ndarray  # (S, N) 组态 s 下全部陷阱的能级（簇外占据取环境值）
    block: np.ndarray  # (9, S, k) 成员的电极速率块
    weighted: Optional[np.ndarray]  # (9, S, k) 按替换的声子线型再算的电极速率块，未请求时为 None
    hop: np.ndarray  # (S, P) 与簇有关的有序对的跃迁速率
    hop_in: np.ndarray  # (S, k) 簇外陷阱跳入成员的总速率（已乘源的环境占据）
    hop_out: np.ndarray  # (S, k) 成员跳往簇外的总速率（已乘目标空的环境概率）


@dataclass(frozen=True)
class Conductors:
    """接电极的导体（TDDB 的导电簇，ADR-060）：成员是电极的延伸，不在输运的缺陷集合里。

    - **与陷阱交换电子**：成员是它费米能级 μ 处的电子库。近旁陷阱 i 与成员 m 之间走陷阱间通道的形式
      （docs/rate_equations.md 的 R6/R7）：几何因子 ν_i·exp[−r·(1/r_D,i + 1/r_D,m)]，成员 → 陷阱乘 Θ(μ, E_i)，
      陷阱 → 成员乘 Θ(E_i, μ)，成员送出电子时按占满、接收时按全空计。两者之比是 exp[−(E_i − μ)/k_BT]，只与这个
      电子库交换的陷阱平衡在费米分布 f(E_i − μ)。MA 因子的温度取两端温度的平均。
    - **记账**：交换速率按 `positive_share` 记进正极侧的 R2A/R3A 块，其余记进负极侧的 R2C/R3C 块，电流沿用电极事件
      的记账；放热按电极事件算出后由 `thermal.heat_release` 移到成员处（`KMCSimulation._conductor_heat`）。
    - **静电**：成员电势固定，成员电荷随环境变。调用方把这一点折进传给 `KMCSimulation` 的耦合矩阵（消去成员后的
      等效耦合）与这里的背景项：外加偏压与固定电荷在成员上感应的电荷、成员上离开成员位置的固定电荷，在各陷阱处的
      电势、电场与横向平均剖面。

    单一材料的闭式几何因子（TDDB 本阶段只做单层）。
    """

    positions: np.ndarray  # (M, 3) 成员坐标 [m]
    fermi_level: np.ndarray  # (M,) 成员处的费米能级 [J]（全局零点 = 负极费米能级）
    positive_share: np.ndarray  # (M,) 与陷阱交换的电子记到正极侧的份额：接负极 0、接正极 1
    r_D: np.ndarray  # (M,) 成员的局域化半径 [m]
    temperature: np.ndarray  # (M,) 成员处的温度 [K]
    phi_at: np.ndarray  # (N,) 背景在各陷阱处的电势 [V]
    field_at: np.ndarray  # (3, N) 背景在各陷阱处的电场矢量 [V/m]（读场侧同耦合矩阵）
    phi_bar: np.ndarray  # (nz+1,) 背景的横向平均电势剖面 [V]


class KMCSimulation:
    """单个偏压/温度点的 KMC 仿真。

    `temperature` 是逐缺陷温度 (N,) [K]，缺省全部取工作点温度；`conductors` 是接电极的导体（ADR-060），缺省没有。
    """

    def __init__(
        self,
        config: Config,
        defects: Optional[DefectEnsemble] = None,
        coupling: Optional[ElectrostaticCoupling] = None,
        solver: Optional[PoissonSolver] = None,
        temperature: Optional[np.ndarray] = None,
        conductors: Optional[Conductors] = None,
    ) -> None:
        self.config = config
        self.device = config.device
        self.sim = config.simulation
        self.rng = np.random.default_rng(self.sim.seed)
        self.defects = defects if defects is not None else generate_defects(
            self.device, self.rng
        )
        self.solver = solver if solver is not None else PoissonSolver(self.device, self.sim.grid)
        self.coupling = coupling if coupling is not None else ElectrostaticCoupling(
            self.solver, self.defects.positions, self.defects.field_side
        )
        n = len(self.defects)
        self._temperature = (
            np.full(n, self.sim.temperature) if temperature is None else np.asarray(temperature, dtype=float)
        )
        self.conductors = conductors
        self._prepare()

    # ------------------------------------------------------------------
    # 初始化：与占据组态无关的量
    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        device, sim = self.device, self.sim
        stack = device.stack
        self.n_defects = len(self.defects)
        defects = self.defects
        positions = defects.positions

        # 外加偏压对应的电势（零电荷解），及其在缺陷处的取值
        phi_applied = self.solver.solve(
            np.zeros(sim.grid.shape), top_boundary_potential(device, sim.voltage)
        )
        self._phi_applied_at = self.solver.interpolate(phi_applied, positions)
        self._field_applied_at = self.solver.field_vectors_at(
            phi_applied, positions, self.defects.field_side
        )
        self._phi_bar_applied = self.solver.lateral_average(phi_applied)
        # 固定电荷（TDDB 生成空位带的 O_i⁻，ADR-048）不随占据变，与外加偏压一起作为每个组态共用的静电背景；
        # 接电极的导体（ADR-060）带来的背景并在这里
        self._phi_fixed_at, self._field_fixed_at, self._phi_bar_fixed = self._fixed_charge_background()
        if self.conductors is not None:
            self._phi_fixed_at = self._phi_fixed_at + self.conductors.phi_at
            self._field_fixed_at = self._field_fixed_at + self.conductors.field_at
            self._phi_bar_fixed = self._phi_bar_fixed + self.conductors.phi_bar
        # R8 的 β 取电子发射后缺陷所在位置留下的电荷（以 e 为单位）：Z 加上与缺陷重合的固定电荷
        self._emission_charge = defects.charge_state + defects.colocated_fixed_charge / constants.elementary_charge

        # 电极侧能量参考（全局零点 = 负极费米能级）
        self._fermi_negative = 0.0
        self._fermi_positive = positive_fermi_level(sim.voltage)
        self._band_bottom = (
            electrode_band_bottom(self._fermi_negative, device.negative.fermi_energy),
            electrode_band_bottom(self._fermi_positive, device.positive.fermi_energy),
        )

        # WKB 路径：电极→缺陷的采样点在几何上固定，只有势垒值每步变化
        t = np.linspace(0.0, 1.0, _TRAP_PATH_POINTS)
        z_i = positions[:, 2][:, None]
        self._path_negative = z_i * t[None, :]
        self._path_positive = z_i + (device.Lz - z_i) * t[None, :]
        self._z_nodes = self.solver.z

        # 分段能带：界面台阶插值用的节点索引与 ΔE_C（一次算好，供 _rates 每步复用）；
        # 沿电极→缺陷路径的隧穿质量 m_di(s) 按所在层取（路径传播质量，ADR-012/F08），
        # 几何上固定、与组态无关，一次算好
        self._iface_node_idx, self._delta_Ec = _interface_step_nodes(device, self._z_nodes)
        self._m_di_path_negative = stack.m_di_at(self._path_negative)
        self._m_di_path_positive = stack.m_di_at(self._path_positive)

        # R1：不经过任何缺陷，势垒剖面就是零空间电荷的理想（分段）梯形，与占据组态
        # 无关——按公式求值一次，run() 末尾直接加到结果上
        self._r1_current = r1_electrode_current(
            device, sim.voltage, sim.temperature, self._z_nodes, self._phi_bar_applied
        )

        # 声子相关量（逐缺陷：ℏω、S 随材料/物种变，温度取缺陷处的值）。形状 (N, _N_PHONON)。
        self._phonon_index = np.arange(_N_PHONON)
        self._phonon_weights = phonon_transition_probability(
            self._phonon_index[None, :],
            defects.huang_rhys[:, None],
            defects.phonon_energy[:, None],
            self._temperature[:, None],
        )
        self._phonon_offsets = self._phonon_index[None, :] * defects.phonon_energy[:, None]

        # R6 的几何因子只依赖缺陷位置/材料/能级，一次算好；R7 每步只补 Θ 因子。
        # 只列截断半径内的有序对——理由见模块文档说明 4。截断半径按叠层所有材料里
        # 最慢的衰减推、判据用被丢弃的总速率上界（ADR-013/F05）；空实现没有任何对。
        r_cut = self._hop_cutoff_radius() if self.n_defects >= 2 else 1.0
        self._pairs = neighbor_pairs(positions, device.Lx, device.Ly, r_cut)
        self._hop_geometric = self._geometric_overlap(self._pairs)
        # MA 因子的温度：两端缺陷温度的平均，a→b 与 b→a 相同，细致平衡不受影响
        self._pair_temperature = 0.5 * (self._temperature[self._pairs.a] + self._temperature[self._pairs.b])
        self._prepare_conductor_pairs()

        # R7 的事件能量差要减去 a、b 两处电子的库仑互作用能 U_ab = q²·G_ab：
        # 耦合矩阵存的是一个 −q 电子在对方处的电势 (−q)·G_ab，故 U_ab = −q·phi_ab。
        # 对 a↔b 两个方向取对称化的 phi，保证正反向 U 相等、细致平衡不被离散噪声
        # 破坏（见 ADR-021 F02）。近邻表与耦合矩阵都与组态无关，一次算好。
        phi_ab = self.coupling.phi_matrix[self._pairs.a, self._pairs.b]
        phi_ba = self.coupling.phi_matrix[self._pairs.b, self._pairs.a]
        self._hop_U_ab = -0.5 * constants.elementary_charge * (phi_ab + phi_ba)

        # F03：PF 收集概率 P_A（位置感知，与 KMC 组态无关，一次算好）。收集电极随
        # 驱动电势 U+V_bi 定：>0 正极收集（正向工况）。R8_eff = P_A·R8 在 _rates 里
        # 折进事件表。单层无阻挡层 → P_A≡1，逐位退回原路径。
        self._drive_potential = sim.voltage + device.built_in_potential()
        self._pf_forward = self._drive_potential > 0.0
        self._pf_collection = self._compute_pf_collection()

        # 快陷阱（说明 6）与强耦合簇（说明 5）都按全空组态下的速率划分、工作点内固定；
        # 要用上面全部速率量，放在初始化最后。快陷阱只退出事件表，照样可以进簇
        self._fast = np.zeros(self.n_defects, dtype=bool)
        self._fast_gamma_min = np.inf
        self._clusters: List[_Cluster] = []
        if self.n_defects > 0:
            blocks, hop = self._rates(np.zeros(self.n_defects, dtype=bool))
            gamma = blocks.sum(axis=0)  # 与电极交换电子的总速率（俘获 + 逃逸 + PF）
            if sim.fast_trap_rate is not None:
                self._fast = gamma >= sim.fast_trap_rate
                if np.any(self._fast):
                    self._fast_gamma_min = float(gamma[self._fast].min())
            self._clusters = self._build_clusters(blocks, hop)
        # 快↔快跃迁不进事件表（两端都用 p̂，交换不改变任何慢变量）
        self._hop_event_mask = (~(self._fast[self._pairs.a] & self._fast[self._pairs.b])).astype(float)

    def _fixed_charge_background(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """全部固定电荷在各缺陷处的电势 (N,) 与电场矢量 (3, N)，以及横向平均电势剖面，一次泊松求解。

        与缺陷重合的固定电荷是该缺陷净电荷 q·(Z − n) + q_fix 的一部分，它在本缺陷处的贡献（耦合矩阵对角元
        那一项）与缺陷自身的电荷一样整项扣除；离开缺陷位置的固定电荷是另一个点电荷，对本缺陷照常计入。
        横向平均剖面不扣自身项，与 `_electrostatics` 的口径相同。没有固定电荷时全为 0。
        """
        defects, solver = self.defects, self.solver
        n = self.n_defects
        if not np.any(defects.fixed_charge != 0.0):
            return np.zeros(n), np.zeros((3, n)), np.zeros(solver.nz + 1)
        phi = solver.solve(solver.deposit(defects.fixed_charge_positions, defects.fixed_charge), 0.0)
        # 耦合矩阵按一个电子（−q）预计算：重合的 q_fix 折合成 −q_fix/q 个电子
        own = -defects.colocated_fixed_charge / constants.elementary_charge
        phi_at = solver.interpolate(phi, defects.positions) - own * self.coupling.phi_self
        field_at = solver.field_vectors_at(phi, defects.positions, defects.field_side) - own * self.coupling.field_self
        return phi_at, field_at, solver.lateral_average(phi)

    @property
    def cluster_members(self) -> List[np.ndarray]:
        """本工作点的强耦合簇（ADR-022），每个簇是升序的全局缺陷编号数组。"""
        return [cluster.members for cluster in self._clusters]

    def _build_clusters(self, blocks: np.ndarray, hop: np.ndarray) -> List[_Cluster]:
        """按全空组态下的速率 `blocks`/`hop` 找强耦合簇（ADR-022）。"""
        n = self.n_defects
        if n < 2:
            return []
        pairs = self._pairs
        strength = coupling_strength(pairs.a, pairs.b, hop, blocks.sum(axis=0))
        groups = coupled_clusters(
            n, pairs.a, pairs.b, strength, _CLUSTER_COUPLING, _CLUSTER_MAX_SIZE
        )
        return [self._cluster_plan(members) for members in groups]

    def _cluster_plan(self, members: np.ndarray) -> _Cluster:
        """把一个簇的成员整理成近邻表上的索引（簇内对、流入对、流出对）。"""
        pairs = self._pairs
        local = np.full(self.n_defects, -1)
        local[members] = np.arange(members.size)
        la, lb = local[pairs.a], local[pairs.b]
        intra = (la >= 0) & (lb >= 0)
        inward = (la < 0) & (lb >= 0)
        outward = (la >= 0) & (lb < 0)
        selected = np.flatnonzero(intra | inward | outward)
        intra_cols = np.flatnonzero(intra[selected])
        in_cols = np.flatnonzero(inward[selected])
        out_cols = np.flatnonzero(outward[selected])
        a_sel, b_sel = pairs.a[selected], pairs.b[selected]
        return _Cluster(
            members=members,
            states=enumerate_occupations(members.size),
            pairs=selected,
            intra_cols=intra_cols,
            intra_a=local[a_sel[intra_cols]],
            intra_b=local[b_sel[intra_cols]],
            in_cols=in_cols,
            in_member=local[b_sel[in_cols]],
            in_source=a_sel[in_cols],
            out_cols=out_cols,
            out_member=local[a_sel[out_cols]],
            out_target=b_sel[out_cols],
        )

    def _defect_host_layers(self) -> np.ndarray:
        """每缺陷的宿主层号：体缺陷取 layer_id，界面缺陷取势阱侧层（field_side 决定上/下）。"""
        d = self.defects
        interface_host = d.interface_id + (d.field_side > 0).astype(int)
        return np.where(d.layer_id >= 0, d.layer_id, interface_host)

    def _compute_pf_collection(self) -> np.ndarray:
        """每缺陷的 PF 收集概率 P_A（ADR-021 F03 / ADR-009 位置感知）。

        R8_eff = P_A·R8。只有与收集电极之间隔着阻挡层（Al₂O₃）的陷阱用 P_A，同侧的
        立即收集（P_A=1）。势垒/透射在零空间电荷的带偏压叠层能带上算，与 KMC 组态
        无关（与 R1 同口径）。单层（无阻挡层）恒为 1，R8_eff = R8 逐位退回原路径。
        零偏压/反偏工况不满足正向收集假设，此处仍给出定义好的 P_A，但结果由
        `_validate_result` 标"未通过验证"。
        """
        device = self.device
        n = self.n_defects
        p = np.ones(n)
        if n == 0:
            return p
        band_refs = device.layer_band_references()
        blocking = _blocking_layer_indices(band_refs)
        if blocking.size == 0:
            return p
        # 收集电极方向：驱动电势 >0 正极（z=Lz）收集；<0 负极（=0 时退化取正极，工况标未验证）
        collect_positive = self._drive_potential >= 0.0
        host = self._defect_host_layers()
        stack = device.stack
        thickness = stack.layer_thickness
        kappa = stack.layer_kappa
        m_di_layer = stack.layer_m_di
        s_kappa = float(np.sum(thickness / kappa))
        for b in blocking:
            if collect_positive:
                crossing = host < b  # 收集在正极：宿主层在阻挡层下方的陷阱要穿过它
                entrance = b - 1
            else:
                crossing = host > b  # 收集在负极：宿主层在阻挡层上方的陷阱要穿过它
                entrance = b + 1
            delta_Ec = float(band_refs[b] - band_refs[entrance])
            if delta_Ec <= 0.0 or not np.any(crossing):
                continue
            v_drop = abs(self._drive_potential) * (thickness[b] / kappa[b]) / s_kappa
            p_b = pf_collection_transmission(
                delta_Ec, v_drop, float(thickness[b]), float(m_di_layer[b]), self.sim.temperature
            )
            p[crossing] *= p_b
        return p

    def _geometric_overlap(self, pairs) -> np.ndarray:
        """近邻表上各有序对的 R6 几何因子 ν·G_ab（组态无关，_prepare 一次算好）。

        单一材料叠层用闭式 `ν·exp[−r_ab(1/r_D,a+1/r_D,b)]`（含单层与三层同材料，
        逐位退回旧口径）；多材料叠层用分段衰减 ν_ab·G_ab（ADR-010），α 沿路径按
        所在层的 m_di 取（路径传播质量，review F08），ν_ab = √(ν_a·ν_b)。
        """
        defects = self.defects
        if self.device.stack.n_materials == 1:
            # 单一材料：闭式几何因子。ν 逐对取（单材料下各缺陷相等），r_D 用物种质量
            return r6_elastic_hop(
                defects.attempt_frequency[pairs.a],
                pairs.r,
                defects.r_D[pairs.a],
                defects.r_D[pairs.b],
            )
        # 多材料：分段几何因子。trap_energy_i = B_ℓ(i) − E_D,i（零场陷阱能级，绝对能标）
        trap_energy = defects.band_reference - defects.ED
        nu_ab = np.sqrt(defects.attempt_frequency[pairs.a] * defects.attempt_frequency[pairs.b])
        overlap = segmented_overlap_factor(
            defects.positions[pairs.a, 2],
            defects.positions[pairs.b, 2],
            pairs.r,
            trap_energy[pairs.a],
            trap_energy[pairs.b],
            self.device.stack.layer_edges,
            self.device.stack.layer_m_di,
            self.device.layer_band_references(),
        )
        return nu_ab * overlap

    def _hop_cutoff_radius(self) -> float:
        """R6/R7 近邻表的截断半径（ADR-013/F05）。

        α_min 取叠层**所有材料**里最慢的衰减：对每个 (缺陷, 层) 组合算
        α = √(2·m_di(层)·[B_层 − 零场陷阱能级]₊)/ℏ，取最小的正值（低隧穿质量能
        压过高势垒，Al₂O₃ 常比 ZrO₂ 衰减更慢）。判据用**被丢弃的总速率上界** <
        机制表最小速率（零场 R8）：N_out·ν_max·exp(−2·α_min·r_cut) < R8_min，
        据此反推容差再求半径。存在零衰减区（α_min=0）时退回列全部有序对。
        """
        defects = self.defects
        device = self.device

        trap_energy = defects.band_reference - defects.ED  # (N,) 零场陷阱能级
        band_refs = device.layer_band_references()  # (n_layers,)
        layer_m_di = device.stack.layer_m_di  # (n_layers,)
        barrier = band_refs[None, :] - trap_energy[:, None]  # (N, n_layers)
        alpha = np.sqrt(2.0 * layer_m_di[None, :] * np.clip(barrier, 0.0, None)) / constants.hbar
        positive = alpha > 0.0
        if not np.any(positive):
            # 零衰减区：几何因子不衰减，退回全表（对角线覆盖整盒的半径）
            return float(np.hypot(np.hypot(device.Lx, device.Ly), device.Lz))
        alpha_min = float(alpha[positive].min())

        nu_max = float(np.max(defects.attempt_frequency))
        n_out = self.n_defects * (self.n_defects - 1)  # 截断外有序对数的保守上界
        tolerance = self._slowest_pf_rate() / (n_out * nu_max)  # N_out·ν_max·tol = R8_min，恒落在 (0,1)
        return hop_cutoff_radius(alpha_min, tolerance)

    def _slowest_pf_rate(self) -> float:
        """机制表最小速率：零场 R8 = ν·exp(−E_D/k_BT_i) 里最小的那个（最深、最冷的陷阱）[1/s]。"""
        defects = self.defects
        kT = constants.Boltzmann * self._temperature
        return float(np.min(defects.attempt_frequency * np.exp(-defects.ED / kT)))

    def _prepare_conductor_pairs(self) -> None:
        """近旁陷阱与接电极导体成员的交换对（`Conductors`，ADR-060），按陷阱编号排好，第 i 个陷阱的对是
        `_conductor_ptr[i]:_conductor_ptr[i+1]`。每对存成员编号、几何因子 ν_i·exp[−r·(1/r_D,i + 1/r_D,m)] 与两端温度的平均。

        截断判据同陷阱间通道（ADR-013）：截断外的对数上界 N·M 乘 ν_max·exp(−α_min·r_cut) 不超过零场 R8 的最小值，
        α_min = 1/max r_D,陷阱 + 1/max r_D,成员；距离取 xy 最小镜像。
        """
        n = self.n_defects
        conductors = self.conductors
        m = 0 if conductors is None else conductors.positions.shape[0]
        self._conductor_ptr = np.zeros(n + 1, dtype=int)
        self._conductor_site = np.zeros(0, dtype=int)
        self._conductor_geometric = np.zeros(0)
        self._conductor_pair_temperature = np.zeros(0)
        if n == 0 or m == 0:
            return
        defects, device = self.defects, self.device
        nu_max = float(np.max(defects.attempt_frequency))
        alpha_min = 1.0 / float(defects.r_D.max()) + 1.0 / float(conductors.r_D.max())
        reach = np.log(n * m * nu_max / self._slowest_pf_rate()) / alpha_min
        box = np.array([device.Lx, device.Ly, 4.0 * device.Lz])  # xy 周期；z 盒长取 4·Lz，不会回绕
        pairs = cKDTree(np.mod(self.defects.positions, box), boxsize=box).sparse_distance_matrix(
            cKDTree(np.mod(conductors.positions, box), boxsize=box), reach, output_type="ndarray"
        )
        order = np.lexsort((pairs["j"], pairs["i"]))
        trap, site, r = pairs["i"][order], pairs["j"][order], pairs["v"][order]
        self._conductor_ptr = np.searchsorted(trap, np.arange(n + 1))
        self._conductor_site = site
        self._conductor_geometric = r6_elastic_hop(defects.attempt_frequency[trap], r, defects.r_D[trap], conductors.r_D[site])
        self._conductor_pair_temperature = 0.5 * (self._temperature[trap] + conductors.temperature[site])

    def _conductor_terms(
        self, energy: np.ndarray, idx: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """各行与导体成员的交换（`Conductors`）：第 r 行是缺陷 idx[r]（缺省为全部缺陷、按编号）能级为 energy[r] 的一个组态。

        返回 (行号, 交换对编号, 成员 → 陷阱的速率, 陷阱 → 成员的速率)，每个 (行, 交换对) 一项；速率不含占据屏蔽。
        """
        defect = np.arange(self.n_defects) if idx is None else np.asarray(idx)
        start = self._conductor_ptr[defect]
        count = self._conductor_ptr[defect + 1] - start
        row = np.repeat(np.arange(defect.size), count)
        pair = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count) + np.repeat(start, count)
        fermi = self.conductors.fermi_level[self._conductor_site[pair]]
        geometric, temperature = self._conductor_geometric[pair], self._conductor_pair_temperature[pair]
        capture = geometric * miller_abrahams_factor(fermi, energy[row], temperature)
        escape = geometric * miller_abrahams_factor(energy[row], fermi, temperature)
        return row, pair, capture, escape

    def _conductor_heat(
        self, energy: np.ndarray, occupied: np.ndarray, idx: np.ndarray, weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """导体交换通道的放热按成员分开（`thermal.heat_release`，ADR-060）。

        第 r 行是缺陷 idx[r] 的一个组态：能级 energy[r]、占据 occupied[r]、权重 weight[r]（孤立陷阱取 1，簇成员取
        簇内稳态分布）。记进电极侧 s（费米能级 μ_s）的那部分交换放出 (μ_s − E)·俘获 + (E − μ_s)·逃逸，与电极事件的
        记账相同。返回 (各成员处的放热 (M,), 负极侧与正极侧移走的放热 (2,)) [W]。
        """
        row, pair, capture, escape = self._conductor_terms(energy, idx)
        site = self._conductor_site[pair]
        share = self.conductors.positive_share[site]
        e, o, w = energy[row], occupied[row], weight[row]
        sides = [
            w * part * ((1.0 - o) * (fermi - e) * capture + o * (e - fermi) * escape)
            for fermi, part in ((self._fermi_negative, 1.0 - share), (self._fermi_positive, share))
        ]
        sites = np.bincount(site, sides[0] + sides[1], self.conductors.positions.shape[0])
        return sites, np.array([sides[0].sum(), sides[1].sum()])

    # ------------------------------------------------------------------
    # 每步：由占据组态构造全部速率
    # ------------------------------------------------------------------
    def _electrostatics(self, occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (缺陷处电势, 缺陷处电场大小, 横向平均电势剖面)。

        每个缺陷的净电荷是 q·(Z_i − n_i) + q_fix,i：空施主带 +Z·q，俘获一个电子后少一个 q，
        固定电荷 q_fix 不随占据变。随占据变的部分：耦合矩阵按一个电子的负电荷 −q 预计算，
        所以组态叠加权重取 n_i − Z_i（见 ADR-021 F01）；局部电势、电场与自作用扣除三处用
        同一权重，缺陷自身电荷的贡献（对角元）按此权重整项扣除。固定电荷的贡献是与组态
        无关的背景（`_fixed_charge_background`），与外加偏压的解一起加上。
        """
        weight = occ.astype(float) - self.defects.charge_state
        phi = (
            self._phi_applied_at
            + self._phi_fixed_at
            + weight @ self.coupling.phi_matrix
            - weight * self.coupling.phi_self
        )
        field = (
            self._field_applied_at
            + self._field_fixed_at
            + weight @ self.coupling.field_matrix
            - weight * self.coupling.field_self
        )
        phi_bar = self._phi_bar_applied + self._phi_bar_fixed + weight @ self.coupling.phi_bar
        return phi, np.linalg.norm(field, axis=0), phi_bar

    def _electrostatics_batch(
        self, occ_states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`_electrostatics` 的批量形式：`occ_states` 形状 (S, N)，每行一个组态。

        返回 (S, N) 的电势与电场大小、(S, nz+1) 的横向平均电势剖面，叠加方式与
        权重口径同 `_electrostatics`。簇内条件稳态一次算完簇内全部组态用。
        """
        coupling = self.coupling
        weight = occ_states.astype(float) - self.defects.charge_state
        phi = self._phi_applied_at + self._phi_fixed_at + weight @ coupling.phi_matrix - weight * coupling.phi_self
        field = (
            self._field_applied_at[:, None, :]
            + self._field_fixed_at[:, None, :]
            + np.matmul(weight, coupling.field_matrix)
            - weight[None, :, :] * coupling.field_self[:, None, :]
        )
        phi_bar = self._phi_bar_applied + self._phi_bar_fixed + weight @ coupling.phi_bar
        return phi, np.linalg.norm(field, axis=0), phi_bar

    def _path_barriers(self, band: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """全部电极→缺陷路径上的势垒剖面 (负极侧, 正极侧)，形状 (N, 路径点)。

        `band` 是一个组态在 z 节点上的一维导带底剖面（含台阶）。界面双值保留真实
        台阶，不插成斜坡（ADR-012）。
        """
        barrier_c = _band_on_path_steps(
            self._path_negative, self._z_nodes, band, self._iface_node_idx, self._delta_Ec
        )
        barrier_a = _band_on_path_steps(
            self._path_positive, self._z_nodes, band, self._iface_node_idx, self._delta_Ec
        )
        return barrier_c, barrier_a

    def _electrode_blocks(
        self,
        energy: np.ndarray,
        field: np.ndarray,
        barrier_c: np.ndarray,
        barrier_a: np.ndarray,
        idx: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """电极↔陷阱的 9 块速率（R2–R5 两侧 + R8_eff），形状 (9, 行数)。

        第 r 行对应缺陷 `idx[r]`（缺省为全部缺陷、按编号排列）；`energy`、`field`、
        `barrier_*` 是这一行所处组态下的陷阱能级、局域场大小与电极→陷阱路径上的
        势垒。同一缺陷可以在不同组态下出现多行（簇内条件稳态），`_rates` 与簇内
        条件稳态共用这一份速率公式。

        `weights` 替换 R4/R5 求和里的声子线型 L_p（形状同 `_phonon_weights`，缺省就用
        它）。传 L_p·p·ℏω 时 R4/R5 四块给出的是单位时间随俘获放给晶格、随逃逸从晶格
        吸收的声子能量 [J/s]，TDDB 的功率 ΔE·R（式 T13，`thermal.heat_release`）由此与
        速率走同一套公式；其余五块不受影响。

        温度逐行取缺陷处的值。有接电极的导体时（`Conductors`），陷阱与导体成员的交换按成员所在电极侧并进 R2/R3 块。
        """
        device = self.device
        defects = self.defects

        def pick(values: np.ndarray) -> np.ndarray:
            return values if idx is None else values[idx]

        temperature = pick(self._temperature)
        ED = pick(defects.ED)
        m_di = pick(defects.m_di)
        phonon_weights = pick(self._phonon_weights if weights is None else weights)
        phonon_offsets = pick(self._phonon_offsets)
        blocks = np.zeros((_N_BLOCKS, energy.shape[0]))
        # R4/R5 的 c_MP 常数因子（含局域场 F²）：用物种质量与物种带隙，两侧电极共用
        coupling = multiphonon_coupling(pick(defects.r_D), field, pick(defects.band_gap), m_di)

        for side, (path, barrier, m_di_path, electrode, fermi, bottom) in enumerate(
            (
                (pick(self._path_negative), barrier_c, pick(self._m_di_path_negative), device.negative, self._fermi_negative, self._band_bottom[0]),
                (pick(self._path_positive), barrier_a, pick(self._m_di_path_positive), device.positive, self._fermi_positive, self._band_bottom[1]),
            )
        ):
            # --- R2 / R3：弹性隧穿（陷阱能级处的单一能量） ---
            transmission = wkb_transmission(path, barrier, energy, m_di_path)
            f_electrode = fermi_dirac(energy, fermi, temperature)
            E1 = energy - bottom
            blocks[_B_R2C + side] = r2_elastic_capture(
                E1, ED, electrode.effective_mass, m_di, f_electrode, transmission
            )
            blocks[_B_R3C + side] = r3_elastic_emission(
                E1, ED, electrode.effective_mass, m_di, f_electrode, transmission
            )

            # --- R4 / R5：多声子跃迁（对 p 求和，E_p = E + p·ℏω） ---
            energies_p = energy[:, None] + phonon_offsets
            transmission_p = wkb_transmission(
                path[:, None, :], barrier[:, None, :], energies_p, m_di_path[:, None, :]
            )
            f_p = fermi_dirac(energies_p, fermi, temperature[:, None])
            dos_p = free_electron_dos(energies_p - bottom, electrode.effective_mass)
            blocks[_B_R4C + side] = r4_multiphonon_capture(
                coupling, dos_p, f_p, transmission_p, phonon_weights
            )
            blocks[_B_R5C + side] = r5_multiphonon_emission(
                coupling,
                dos_p,
                f_p,
                transmission_p,
                phonon_weights,
                self._phonon_index,
                pick(defects.phonon_energy)[:, None],
                temperature[:, None],
            )

        # --- R8：PF 热发射（折进 F03 收集概率 → R8_eff = P_A·R8） ---
        # 被阻挡层挡回原陷阱的分支是 KMC 空事件（状态不变），不单列——直接把 R8 换成
        # R8_eff 进事件表与 _conditional_flux（ADR-009/021 F03）。field 沿用 Phase 3
        # 的界面单侧场（ADR-017 §2），R8_eff 不另取场。β 的电荷取发射后缺陷处留下的电荷
        # （Z 加上重合的固定电荷，`_emission_charge`）。
        blocks[_B_R8] = pick(self._pf_collection) * r8_pf_emission(
            pick(defects.attempt_frequency),
            ED,
            field,
            pick(self._emission_charge),
            pick(defects.eps_opt),
            temperature,
        )
        if self._conductor_site.size:
            rows = energy.shape[0]
            row, pair, capture, escape = self._conductor_terms(energy, idx)
            share = self.conductors.positive_share[self._conductor_site[pair]]
            blocks[_B_R2C] += np.bincount(row, capture * (1.0 - share), rows)
            blocks[_B_R2A] += np.bincount(row, capture * share, rows)
            blocks[_B_R3C] += np.bincount(row, escape * (1.0 - share), rows)
            blocks[_B_R3A] += np.bincount(row, escape * share, rows)
        return blocks

    def _rates(self, occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """构造当前组态下的全部速率。

        返回 (块速率 (9, N), 近邻表上各有序对的跃迁速率 (P,))。R1 与占据组态
        无关，已在 _prepare 里求值一次，不在这里重算。

        **块速率不按占据状态屏蔽**：第 b 块第 i 个元素是"缺陷 i 处在该跃迁
        所需状态时"的速率。屏蔽只在组装事件表时施加，电流估计量需要未屏蔽
        的值。跃迁速率同样不含端点占据屏蔽，第 k 个元素对应缺陷对
        `self._pairs.a[k] → self._pairs.b[k]`。

        材料参数逐缺陷取（叠层按层/物种各异，ADR-012/F08）：R2/R3 前置因子、
        R4/R5 的 c_MP 与 r_D、R8 的 ε_opt/ν 用**物种质量/物性**（缺陷自身的
        `m_di`/`band_gap`/`eps_opt`/`attempt_frequency`）；WKB 透射用**路径传播
        质量** m_di(s)（沿路径按所在层取，`self._m_di_path_*`）。单层退化时两者
        相等、逐点恒定，逐位退回旧标量口径。
        """
        energy, field_at, barrier_c, barrier_a = self._trap_state(occ)
        blocks = self._electrode_blocks(energy, field_at, barrier_c, barrier_a)
        return blocks, self._hop_rates(energy)

    def _trap_state(self, occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """组态 `occ` 下的 (陷阱能级, 局域场大小, 负极侧路径势垒, 正极侧路径势垒)。"""
        phi_at, field_at, phi_bar = self._electrostatics(occ)
        energy = trap_levels(self.defects.band_reference, phi_at, self.defects.ED)  # 陷阱上电子的总能量
        band = conduction_band_edge(self.device, self._z_nodes, phi_bar)  # 一维导带底剖面（含台阶）
        barrier_c, barrier_a = self._path_barriers(band)
        return energy, field_at, barrier_c, barrier_a

    def _hop_rates(self, energy: np.ndarray) -> np.ndarray:
        """R7：近邻表上各有序对的陷阱→陷阱跃迁速率（R6 的几何因子 × MA 能量因子）。

        终态 b 的能级取"电子已离开 a"之后的值 energy[b] − U_ab：当前组态里
        energy[b] 含出发陷阱 a 上电子的贡献，跳跃发生后这部分消失。这样 MA 因子
        用的是完整事件能量差 ΔH_{a→b} = (E_b − E_a) − U_ab，正反向满足细致平衡
        （见 docs/rate_equations.md R7 一节与 ADR-021 F02）。
        """
        pairs = self._pairs
        return self._hop_geometric * miller_abrahams_factor(
            energy[pairs.a], energy[pairs.b] - self._hop_U_ab, self._pair_temperature
        )

    def _hop_sums(
        self, hop: np.ndarray, occf: np.ndarray, empty: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """把近邻表上的跃迁速率按缺陷聚合成流入/流出总速率。

        与稠密矩阵下的 `occf @ hop`（流入 b 的 Σ_a occ_a·R(a→b)）与
        `hop @ empty`（自 a 流出的 Σ_b R(a→b)·(1-occ_b)）逐项等价。
        """
        pairs = self._pairs
        n = self.n_defects
        hop_in = np.bincount(pairs.b, weights=hop * occf[pairs.a], minlength=n)
        hop_out = np.bincount(pairs.a, weights=hop * empty[pairs.b], minlength=n)
        return hop_in, hop_out

    @staticmethod
    def _conditional_flux(
        blocks: np.ndarray, hop_in: np.ndarray, hop_out: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """条件稳态下每个陷阱的正极净通量分解。

        返回 (TAT 分量, PF 分量, 条件稳态占据 p̂)，单位 1/s。
        相消项 C_a·E_a 已解析约掉，见模块文档说明 3。
        """
        capture_c = blocks[_B_R2C] + blocks[_B_R4C] + hop_in
        capture_a = blocks[_B_R2A] + blocks[_B_R4A]
        escape_c = blocks[_B_R3C] + blocks[_B_R5C] + hop_out
        escape_a_tat = blocks[_B_R3A] + blocks[_B_R5A]
        pf = blocks[_B_R8]

        total = capture_c + capture_a + escape_c + escape_a_tat + pf
        with np.errstate(invalid="ignore", divide="ignore"):
            inv = np.where(total > 0.0, 1.0 / total, 0.0)
        tat = (capture_c * escape_a_tat - (escape_c + pf) * capture_a) * inv
        pf_flux = (capture_c + capture_a) * pf * inv
        return tat, pf_flux, (capture_c + capture_a) * inv

    def _trap_flux(
        self, occ: np.ndarray, blocks: np.ndarray, hop: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """当前组态下每个陷阱的正极净通量 (TAT, PF) 与条件稳态占据 p̂，单位 1/s。

        单陷阱用 `_conditional_flux`（模块文档说明 3）；强耦合簇的成员换成簇内条件
        稳态（说明 5），簇外占据取 `occ` 作为环境。`blocks`/`hop` 是 `_rates(occ)`
        的结果。
        """
        occf = occ.astype(float)
        tat, pf, p_hat = self._conditional_flux(blocks, *self._hop_sums(hop, occf, 1.0 - occf))
        if self._clusters:
            self._cluster_flux(occf, tat, pf, p_hat)
        return tat, pf, p_hat

    def _cluster_flux(
        self, occf: np.ndarray, tat: np.ndarray, pf: np.ndarray, p_hat: np.ndarray
    ) -> None:
        """全部强耦合簇在环境占据 `occf` 下的条件稳态，结果写回簇成员（ADR-022）。

        每个簇枚举簇内 2^k 个组态：按组态的真实电荷分布重算簇成员的电极速率和与
        簇有关的跃迁速率（库仑作用按组态精确计入，`_cluster_tables`），簇外占据固定为
        `occf`，解簇内主方程稳态，把成员的 (TAT, PF, 占据期望) 写进 `tat`/`pf`/`p_hat`。
        """
        for table in self._cluster_tables(occf):
            c, block = table.cluster, table.block
            members = c.members
            tat[members], pf[members], p_hat[members] = cluster_steady_state(
                c.states,
                capture_positive=block[_B_R2A] + block[_B_R4A],
                capture_other=block[_B_R2C] + block[_B_R4C] + table.hop_in,
                escape_positive=block[_B_R3A] + block[_B_R5A],
                pf=block[_B_R8],
                escape_other=block[_B_R3C] + block[_B_R5C] + table.hop_out,
                intra_a=c.intra_a,
                intra_b=c.intra_b,
                intra_rate=table.hop[:, c.intra_cols],
            )

    def _cluster_tables(
        self, occf: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> List["_ClusterTable"]:
        """全部强耦合簇在环境占据 `occf` 下、簇内每个组态的能级与速率表（ADR-022）。

        每个簇枚举簇内 2^k 个组态，按组态的真实电荷分布重算簇成员的电极速率和与簇有关的
        跃迁速率，簇外占据固定为 `occf`。所有簇的全部组态拼在一起做一次静电叠加、一次
        电极速率计算，省掉逐簇调用的开销。`weights` 给出时另按它再算一遍电极速率块
        （`_electrode_blocks` 的同名参数），放进 `weighted`。
        """
        defects = self.defects
        clusters = self._clusters
        if not clusters:
            return []

        # 全部簇的全部组态按簇拼成 (R, N)；簇 c 占第 row_start[c] 起的 2^k 行
        occ_states = np.concatenate(
            [np.repeat(occf[None, :], c.states.shape[0], axis=0) for c in clusters]
        )
        row_start = np.cumsum([0] + [c.states.shape[0] for c in clusters])
        for c, start, stop in zip(clusters, row_start[:-1], row_start[1:]):
            occ_states[start:stop, c.members] = c.states
        phi_at, field_at, phi_bar = self._electrostatics_batch(occ_states)
        energy = trap_levels(defects.band_reference, phi_at, defects.ED)  # (R, N)
        band = conduction_band_edge(self.device, self._z_nodes, phi_bar)  # (R, nz+1)
        z_aug, band_aug = _band_nodes_with_steps(
            self._z_nodes, band, self._iface_node_idx, self._delta_Ec
        )

        # 簇成员的电极速率：每个 (组态行, 成员) 一行，按簇、组态、成员的顺序排列。
        # 电极→成员的路径势垒同 _path_barriers（界面双值已在上面一次补好）
        rows, idx, barrier_c, barrier_a = [], [], [], []
        for c, start, stop in zip(clusters, row_start[:-1], row_start[1:]):
            path_c, path_a = self._path_negative[c.members], self._path_positive[c.members]
            for r in range(start, stop):
                barrier_c.append(np.interp(path_c.ravel(), z_aug, band_aug[r]).reshape(path_c.shape))
                barrier_a.append(np.interp(path_a.ravel(), z_aug, band_aug[r]).reshape(path_a.shape))
            rows.append(np.repeat(np.arange(start, stop), c.members.size))
            idx.append(np.tile(c.members, stop - start))
        rows, idx = np.concatenate(rows), np.concatenate(idx)
        rate_inputs = (
            energy[rows, idx],
            field_at[rows, idx],
            np.concatenate(barrier_c),
            np.concatenate(barrier_a),
            idx,
        )
        blocks = self._electrode_blocks(*rate_inputs)
        weighted = None if weights is None else self._electrode_blocks(*rate_inputs, weights)

        tables = []
        offset = 0
        for c, start, stop in zip(clusters, row_start[:-1], row_start[1:]):
            n_states, k = c.states.shape
            columns = slice(offset, offset + n_states * k)
            offset += n_states * k

            # 与簇有关的跃迁速率，终态能级扣掉出发陷阱电子的库仑贡献（同 _hop_rates）
            sel = c.pairs
            e = energy[start:stop]
            hop = self._hop_geometric[sel] * miller_abrahams_factor(
                e[:, self._pairs.a[sel]], e[:, self._pairs.b[sel]] - self._hop_U_ab[sel], self._pair_temperature[sel]
            )  # (S, P)
            hop_in = np.zeros((n_states, k))
            hop_out = np.zeros((n_states, k))
            np.add.at(hop_in.T, c.in_member, (occf[c.in_source] * hop[:, c.in_cols]).T)
            np.add.at(hop_out.T, c.out_member, ((1.0 - occf[c.out_target]) * hop[:, c.out_cols]).T)
            tables.append(
                _ClusterTable(
                    cluster=c,
                    energy=e,
                    block=blocks[:, columns].reshape(_N_BLOCKS, n_states, k),
                    weighted=None if weighted is None else weighted[:, columns].reshape(_N_BLOCKS, n_states, k),
                    hop=hop,
                    hop_in=hop_in,
                    hop_out=hop_out,
                )
            )
        return tables

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        """跑完 n_events 步并返回稳态电流与分解统计。"""
        n = self.n_defects
        sim = self.sim
        pairs = self._pairs
        occ = np.zeros(n, dtype=bool)  # 初始占据状态：全空（快陷阱处恒为 False，占据看 p_fast）
        burn_in = int(sim.n_events * sim.burn_in_fraction)
        # 快陷阱（说明 6）：占据取与当前慢组态自洽的条件稳态 p̂，每步开头迭代求出
        fast = self._fast
        has_fast = bool(fast.any())
        p_fast = np.zeros(n)
        fast_flux = 0.0  # 快陷阱正极净通量的时间积分
        fast_exchange = 0  # 快→慢跃迁减慢→快跃迁的净次数（测量窗口内）

        elapsed = 0.0
        mech_flux = {"tat": 0.0, "pf": 0.0}
        defect_flux = np.zeros(n)
        occupancy_time = np.zeros(n)
        occupancy_mf = np.zeros(n)
        direct_flux = 0.0
        counts = 0.0  # 正极净电子计数
        negative_counts = 0.0  # 负极净电子计数（F04 全局守恒）
        n_positive_events = 0
        n_negative_events = 0
        n_executed = 0  # 实际执行的事件数（可能因总速率归零早停，如全器件无缺陷）
        measured_events = 0  # 测量窗口内执行的事件数
        n_inside_start = 0  # 测量开始时器件内电子数（初始全空 = 0）
        measuring = burn_in <= 0
        mask = np.empty((_N_BLOCKS, n))
        # 批次均值法：测量窗口按事件数等分成 _N_BATCHES 段，分段累计通量与时长
        batch_events = max(1, -(-(sim.n_events - burn_in) // _N_BATCHES))
        batch_charge = np.zeros(_N_BATCHES)
        batch_time = np.zeros(_N_BATCHES)

        for step in range(sim.n_events):
            occ_now = np.where(fast, p_fast, occ) if has_fast else occ
            blocks, hop = self._rates(occ_now)
            if has_fast:
                # 快陷阱的 p̂ 与当前慢组态自洽：按本步速率更新 p̂，变化超过容差就重算速率
                for _ in range(_FAST_MAX_ITER):
                    occf = occ_now.astype(float)
                    p_new = self._conditional_flux(blocks, *self._hop_sums(hop, occf, 1.0 - occf))[2][fast]
                    converged = np.max(np.abs(p_new - p_fast[fast])) <= _FAST_P_TOL
                    p_fast[fast] = p_new
                    occ_now = np.where(fast, p_fast, occ)
                    if converged:
                        break
                    blocks, hop = self._rates(occ_now)
            occf = occ_now.astype(float)
            empty = 1.0 - occf

            # 事件表：俘获类要求空位，逃逸/发射类要求已占据；快陷阱的块整列移出，
            # 跃迁按端点的 p̂ / 1−p̂ 加权，快↔快跃迁移出
            mask[:_N_CAPTURE_BLOCKS] = empty
            mask[_N_CAPTURE_BLOCKS:] = occf
            event_hop = hop * occf[pairs.a] * empty[pairs.b]
            if has_fast:
                mask[:, fast] = 0.0
                event_hop = event_hop * self._hop_event_mask
            event_blocks = blocks * mask

            total_rate = event_blocks.sum() + event_hop.sum()
            if total_rate <= 0.0:
                break
            n_executed += 1
            dt = -np.log(self.rng.random()) / total_rate

            if measuring:
                tat, pf, p_hat = self._trap_flux(occ_now, blocks, hop)
                elapsed += dt
                occupancy_time += occf * dt
                occupancy_mf += p_hat * dt
                defect_flux += (tat + pf) * dt
                mech_flux["tat"] += tat.sum() * dt
                mech_flux["pf"] += pf.sum() * dt
                direct_flux += (_POSITIVE_SIGN[:, None] * event_blocks).sum() * dt
                if has_fast:
                    # 快陷阱的事件不再执行，它们的正极通量按条件稳态补进两个交叉估计量
                    flux_fast = (tat[fast] + pf[fast]).sum() * dt
                    direct_flux += flux_fast
                    fast_flux += flux_fast
                batch = min(measured_events // batch_events, _N_BATCHES - 1)
                batch_charge[batch] += (tat.sum() + pf.sum()) * dt
                batch_time[batch] += dt

            # 选择并执行事件
            flat = np.concatenate([event_blocks.ravel(), event_hop])
            cumulative = np.cumsum(flat)
            index = int(
                np.searchsorted(cumulative, self.rng.random() * cumulative[-1], side="right")
            )
            index = min(index, flat.size - 1)

            if index < _N_BLOCKS * n:
                block, i = divmod(index, n)
                occ[i] = block < _N_CAPTURE_BLOCKS
                if measuring:
                    a_sign = _POSITIVE_SIGN[block]
                    if a_sign != 0.0:
                        n_positive_events += 1
                        counts += a_sign
                    c_sign = _NEGATIVE_SIGN[block]
                    if c_sign != 0.0:
                        n_negative_events += 1
                        negative_counts += c_sign
            else:
                k = index - _N_BLOCKS * n
                a, b = pairs.a[k], pairs.b[k]
                occ[a] = False  # 快陷阱一端的 p̂ 在下一步开头重新自洽，布尔占据保持 False
                occ[b] = not fast[b]
                if measuring:
                    fast_exchange += int(fast[a]) - int(fast[b])

            if measuring:
                measured_events += 1
            elif step + 1 >= burn_in:
                measuring = True
                n_inside_start = int(occ.sum())  # 测量窗口起点的器件内（慢陷阱）电子数

        scale = constants.elementary_charge / elapsed if elapsed > 0.0 else 0.0
        r1_tunnel, r1_thermionic = self._r1_current
        by_mechanism = {
            "direct_fn": r1_tunnel,
            "thermionic": r1_thermionic,
            "tat": mech_flux["tat"] * scale,
            "pf": mech_flux["pf"] * scale,
        }
        current = float(sum(by_mechanism.values()))
        inv_time = 1.0 / elapsed if elapsed > 0.0 else 0.0
        by_defect = defect_flux * scale
        occupancy = occupancy_time * inv_time
        occupancy_mf_avg = occupancy_mf * inv_time
        occ_dev = float(np.mean(np.abs(occupancy - occupancy_mf_avg))) if n > 0 else 0.0
        current_counts = float(counts * scale)
        current_fast = float(fast_flux * scale)
        current_counts_total = current_counts + current_fast + (r1_tunnel + r1_thermionic)
        delta_n_inside = int(occ.sum()) - n_inside_start
        batch_rel_error = batch_relative_error(
            constants.elementary_charge * batch_charge, batch_time, current
        )
        # 时间尺度分离：min Γ_F 与测量窗口内平均事件速率之比（没有快陷阱时为 ∞）
        separation = np.inf
        if has_fast:
            separation = self._fast_gamma_min * elapsed / measured_events if measured_events else 0.0
        validated, validation_notes = _validate_result(
            n, self._pf_forward, elapsed, measured_events, occ_dev,
            n_positive_events, current, current_counts_total, batch_rel_error, separation,
        )
        return SimulationResult(
            voltage=sim.voltage,
            temperature=sim.temperature,
            elapsed_time=elapsed,
            current=current,
            current_density=current / self.device.area,
            area=self.device.area,
            by_mechanism=by_mechanism,
            by_defect=by_defect,
            by_layer=by_layer_attribution(self.defects, by_defect),
            occupancy=occupancy,
            occupancy_mean_field=occupancy_mf_avg,
            n_events=n_executed,
            measured_events=measured_events,
            current_direct=float(direct_flux * scale + r1_tunnel + r1_thermionic),
            current_counts=current_counts,
            current_counts_negative=float(negative_counts * scale),
            n_positive_events=n_positive_events,
            delta_n_inside=delta_n_inside,
            validated=validated,
            validation_notes=validation_notes,
            n_fast_traps=int(fast.sum()),
            current_fast=current_fast,
            fast_exchange=fast_exchange,
            defects=self.defects,
        )


def run_single(config: Config, defects: Optional[DefectEnsemble] = None) -> SimulationResult:
    """跑一个单点仿真。"""
    return KMCSimulation(config, defects=defects).run()
