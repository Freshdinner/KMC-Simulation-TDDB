"""KMC 主循环（Gillespie direct method）与电流统计。

算法流程见 docs/physics_model.md 的"KMC 算法"一节。

三点实现说明：

1. **静电求解的线性叠加。** 泊松方程是线性的，缺陷位置在一次实现里固定
   不变，因此初始化时解 N+1 次泊松方程（外加偏压 1 次 + 每个缺陷单位电荷
   各 1 次），之后任一占据组态的电势/电场都由这些解按占据数线性叠加得到，
   主循环里不再需要解泊松方程。这既精确又把每步开销降到矩阵-向量乘。
   缺陷 i 自身电荷在 i 处的贡献（CIC 自作用）就是这些矩阵的对角元，按
   项目"不考虑镜像力"的约定整项扣除。

2. **R1 一次性求值，不进主循环。** 电极↔电极隧穿不改变占据组态，把它放进
   事件表只会产生大量"空转"事件。它走的是理想平面势垒的物理图像，与
   Richardson-Schottky 对照式同口径，所以势垒剖面取零空间电荷的外加偏压解
   （E_C(z) = E_B - e·(U+V_bi)·z/Lz，严格线性）：初始化时按公式求值一次，
   run() 末尾把 I_R1 = q·R1 直接加进 by_mechanism。结果与 KMC 轨迹无关，
   不随 seed、n_events 变化，也不含采样噪声。

3. **电流估计量。** 稳态净电流是阳极侧两个大通量之差：阳极→陷阱的俘获
   与陷阱→阳极的逃逸，二者在稳态下几乎精确抵消，直接数事件（或直接对
   瞬时通量做时间平均）会被这个相消淹没——本项目参数下噪声比信号大 2–3
   个数量级。把单个陷阱的条件稳态占据 p̂_i = C_i/(C_i+E_i) 代入净通量表
   达式后，相消项 C_a·E_a 解析地约掉：

       flux_i = [(C_c+H_in)·E_a - (E_c+H_out)·C_a] / (C_i+E_i)

   C_c/C_a 为阴极/阳极侧俘获总速率，E_c/E_a 为向阴极/阳极的逃逸总速率
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

统计口径按"电子抵达收集电极（或进入导带后被收集）的最后一步事件类型"
归类，见 docs/rate_equations.md 的机制对照表。R8 发射到导带的电子按
"立即被阳极收集"处理（U>0 时导带电子向阳极漂移）。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import constants

from .bands import (
    anode_fermi_level,
    conduction_band_edge,
    electrode_band_bottom,
    fermi_dirac,
    free_electron_dos,
    top_boundary_potential,
    trap_levels,
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
_R1_ENERGY_POINTS = 128  # R1 能量积分点数（阈值两侧各自）
_PF_BARRIER_POINTS = 257  # F03 阻挡层 PF 收集势垒的 WKB 采样点

# F04 每个工作点的"未通过验证"判据阈值（ADR-021 F04，见 _validate_result）
_MIN_MEASURED_EVENTS = 100  # 测量窗口至少要有这么多事件才算采到
_MEAN_FIELD_TOL = 0.1  # KMC 占据与条件稳态 p̂ 的平均偏差上限（局部稳态/条件通量成立的前提）
_COUNTS_SUFFICIENT = 50  # 阳极事件数达到此值，净计数交叉校验才纳入硬门槛（否则统计为空、跳过）
_COUNTS_REL_TOL = 0.5  # 计数够用时 |J_counts+R1 − J_cond| 相对 |J_cond| 的上限（计数本身仍噪声大）

# 事件块：每块长度 N，块内索引即缺陷编号。前四块是俘获，后五块是逃逸/发射。
_B_R2C, _B_R2A, _B_R4C, _B_R4A, _B_R3C, _B_R3A, _B_R5C, _B_R5A, _B_R8 = range(9)
_N_BLOCKS = 9
_N_CAPTURE_BLOCKS = 4
# 各块对"阳极净流入电子数"的贡献（+1 到达阳极，-1 由阳极取走）
_ANODE_SIGN = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
# 各块对"阴极净送入器件电子数"的贡献（+1 阴极注入陷阱，-1 陷阱回到阴极）；
# R8 只到阳极，对阴极记 0。cathode_net − anode_net = ΔN_inside，全局守恒的基础（F04）
_CATHODE_SIGN = np.array([1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
_BLOCK_MECHANISM = ("tat",) * 8 + ("pf",)


@dataclass
class SimulationResult:
    """单点仿真结果。电流均为净值，正号 = 电子由阴极流向阳极。"""

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
    current_direct: float  # 瞬时阳极通量的时间平均 [A]（相消未消去，方差大）
    current_counts: float  # 阳极净事件电流 [A]（不含 R1，即 I_A − I_R1）
    current_counts_cathode: float  # 阴极净事件电流 [A]（不含 R1，即 I_C − I_R1）
    n_anode_events: int
    delta_n_inside: int  # 测量窗口内器件内电子数变化 ΔN_inside（守恒：ΔI·Δt = q·ΔN_inside）
    # --- 收敛/验证状态（F04，ADR-021）---
    validated: bool  # 是否通过验证；未通过时不得作为可信稳态电流交付
    validation_notes: Tuple[str, ...]  # 未通过/存疑的具体原因
    defects: Optional[DefectEnsemble] = field(repr=False, default=None)

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


def _validate_result(
    n_defects: int,
    forward: bool,
    elapsed: float,
    measured_events: int,
    occ_mean_abs_dev: float,
    n_anode_events: int,
    current: float,
    current_counts_total: float,
) -> Tuple[bool, Tuple[str, ...]]:
    """按 ADR-021 F04 判定单点是否通过验证，返回 (validated, 原因列表)。

    硬门槛用可靠可算的判据：正向收集（U+V_bi>0）、测量窗口充分、KMC 占据与条件
    稳态 p̂ 接近（局部稳态/条件通量成立的前提）、机制分量闭合（构造保证）。净事件
    计数交叉校验 ΔJ=J_cond−(J_counts+R1) 噪声大，只有阳极事件数足够（统计非空）时
    才纳入门槛，否则作诊断记录、不判失败（Q1 口径）。全器件无缺陷（R1-only）没有
    统计估计量要验证、R1 是确定性解，直接通过。
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
    if n_anode_events >= _COUNTS_SUFFICIENT and current != 0.0:
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
    shape = np.shape(path)
    if iface_idx.size == 0:
        return np.interp(flat, z_nodes, band_nodes).reshape(shape)
    z_aug = np.insert(z_nodes, iface_idx, z_nodes[iface_idx])
    band_aug = np.insert(band_nodes, iface_idx, band_nodes[iface_idx] - delta_Ec)
    return np.interp(flat, z_aug, band_aug).reshape(shape)


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

    I = q·R1，速率 R1 见 docs/rate_equations.md 的 R1 一节。`phi_bar` 是
    `z_nodes` 上的横向平均电势剖面 [V]；取零空间电荷（外加偏压）解时势垒是
    理想（分段）梯形，势垒顶恰在最高的那段——单层里是 E_B、叠层里可落在阻挡层
    内（ADR-012）。分段 WKB：势垒用界面双值保留真实台阶，隧穿质量沿路径按层
    取（`stack.m_di_at`）。单层退回原式（无界面、m_di 恒定）。
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
        device.cathode.effective_mass,
        device.stack.m_di_at(z),
        n_points=_R1_ENERGY_POINTS,
    )
    return constants.elementary_charge * tunnel, constants.elementary_charge * thermionic


def _blocking_layer_indices(band_refs: np.ndarray) -> np.ndarray:
    """叠层里的阻挡层：导带基准高于两侧相邻层的内部层（glossary 的 Blocking layer）。

    ZAZ 里正是中间 Al₂O₃。单层或无内部凸起时返回空——PF 收集退回"立即收集"。
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


class KMCSimulation:
    """单个偏压/温度点的 KMC 仿真。"""

    def __init__(
        self,
        config: Config,
        defects: Optional[DefectEnsemble] = None,
        coupling: Optional[ElectrostaticCoupling] = None,
        solver: Optional[PoissonSolver] = None,
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

        # 电极侧能量参考（全局零点 = 阴极费米能级）
        self._fermi_cathode = 0.0
        self._fermi_anode = anode_fermi_level(sim.voltage)
        self._band_bottom = (
            electrode_band_bottom(self._fermi_cathode, device.cathode.fermi_energy),
            electrode_band_bottom(self._fermi_anode, device.anode.fermi_energy),
        )

        # WKB 路径：电极→缺陷的采样点在几何上固定，只有势垒值每步变化
        t = np.linspace(0.0, 1.0, _TRAP_PATH_POINTS)
        z_i = positions[:, 2][:, None]
        self._path_cathode = z_i * t[None, :]
        self._path_anode = z_i + (device.Lz - z_i) * t[None, :]
        self._z_nodes = self.solver.z

        # 分段能带：界面台阶插值用的节点索引与 ΔE_C（一次算好，供 _rates 每步复用）；
        # 沿电极→缺陷路径的隧穿质量 m_di(s) 按所在层取（路径传播质量，ADR-012/F08），
        # 几何上固定、与组态无关，一次算好
        self._iface_node_idx, self._delta_Ec = _interface_step_nodes(device, self._z_nodes)
        self._m_di_path_cathode = stack.m_di_at(self._path_cathode)
        self._m_di_path_anode = stack.m_di_at(self._path_anode)

        # R1：不经过任何缺陷，势垒剖面就是零空间电荷的理想（分段）梯形，与占据组态
        # 无关——按公式求值一次，run() 末尾直接加到结果上
        self._r1_current = r1_electrode_current(
            device, sim.voltage, sim.temperature, self._z_nodes, self._phi_bar_applied
        )

        # 声子相关量（逐缺陷：ℏω、S 随材料/物种变）。形状 (N, _N_PHONON)。
        self._phonon_index = np.arange(_N_PHONON)
        self._phonon_weights = phonon_transition_probability(
            self._phonon_index[None, :],
            defects.huang_rhys[:, None],
            defects.phonon_energy[:, None],
            sim.temperature,
        )
        self._phonon_offsets = self._phonon_index[None, :] * defects.phonon_energy[:, None]

        # R6 的几何因子只依赖缺陷位置/材料/能级，一次算好；R7 每步只补 Θ 因子。
        # 只列截断半径内的有序对——理由见模块文档说明 4。截断半径按叠层所有材料里
        # 最慢的衰减推、判据用被丢弃的总速率上界（ADR-013/F05）；空实现没有任何对。
        r_cut = self._hop_cutoff_radius() if self.n_defects >= 2 else 1.0
        self._pairs = neighbor_pairs(positions, device.Lx, device.Ly, r_cut)
        self._hop_geometric = self._geometric_overlap(self._pairs)

        # R7 的事件能量差要减去 a、b 两处电子的库仑互作用能 U_ab = q²·G_ab：
        # 耦合矩阵存的是一个 −q 电子在对方处的电势 (−q)·G_ab，故 U_ab = −q·phi_ab。
        # 对 a↔b 两个方向取对称化的 phi，保证正反向 U 相等、细致平衡不被离散噪声
        # 破坏（见 ADR-021 F02）。近邻表与耦合矩阵都与组态无关，一次算好。
        phi_ab = self.coupling.phi_matrix[self._pairs.a, self._pairs.b]
        phi_ba = self.coupling.phi_matrix[self._pairs.b, self._pairs.a]
        self._hop_U_ab = -0.5 * constants.elementary_charge * (phi_ab + phi_ba)

        # F03：PF 收集概率 P_A（位置感知，与 KMC 组态无关，一次算好）。收集电极随
        # 驱动电势 U+V_bi 定：>0 阳极收集（正向工况）。R8_eff = P_A·R8 在 _rates 里
        # 折进事件表。单层无阻挡层 → P_A≡1，逐位退回原路径。
        self._drive_potential = sim.voltage + device.built_in_potential()
        self._pf_forward = self._drive_potential > 0.0
        self._pf_collection = self._compute_pf_collection()

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
        # 收集电极方向：驱动电势 >0 阳极（z=Lz）收集；<0 阴极（=0 时退化取阳极，工况标未验证）
        collect_anode = self._drive_potential >= 0.0
        host = self._defect_host_layers()
        stack = device.stack
        thickness = stack.layer_thickness
        kappa = stack.layer_kappa
        m_di_layer = stack.layer_m_di
        s_kappa = float(np.sum(thickness / kappa))
        for b in blocking:
            if collect_anode:
                crossing = host < b  # 收集在阳极：宿主层在阻挡层下方的陷阱要穿过它
                entrance = b - 1
            else:
                crossing = host > b  # 收集在阴极：宿主层在阻挡层上方的陷阱要穿过它
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
        kT = constants.Boltzmann * self.sim.temperature

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

        # 机制表最小速率取零场 R8 = ν·exp(−E_D/k_BT) 里最小的那个（最深陷阱）
        r8_min = float(np.min(defects.attempt_frequency * np.exp(-defects.ED / kT)))
        nu_max = float(np.max(defects.attempt_frequency))
        n_out = self.n_defects * (self.n_defects - 1)  # 截断外有序对数的保守上界
        tolerance = r8_min / (n_out * nu_max)  # N_out·ν_max·tol = R8_min，恒落在 (0,1)
        return hop_cutoff_radius(alpha_min, tolerance)

    # ------------------------------------------------------------------
    # 每步：由占据组态构造全部速率
    # ------------------------------------------------------------------
    def _electrostatics(self, occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (缺陷处电势, 缺陷处电场大小, 横向平均电势剖面)。

        每个缺陷的净电荷是 q·(Z_i − n_i)：空施主带 +Z·q，俘获一个电子后中性。
        耦合矩阵按一个电子的负电荷 −q 预计算，所以组态叠加权重取 n_i − Z_i
        （见 ADR-021 F01）；局部电势、电场与自作用扣除三处用同一权重。缺陷自身
        电荷的贡献（对角元）按此权重整项扣除。
        """
        weight = occ.astype(float) - self.defects.charge_state
        phi = (
            self._phi_applied_at
            + weight @ self.coupling.phi_matrix
            - weight * self.coupling.phi_self
        )
        field = (
            self._field_applied_at
            + weight @ self.coupling.field_matrix
            - weight * self.coupling.field_self
        )
        phi_bar = self._phi_bar_applied + weight @ self.coupling.phi_bar
        return phi, np.linalg.norm(field, axis=0), phi_bar

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
        device, sim = self.device, self.sim
        defects = self.defects
        ED = defects.ED

        phi_at, field_at, phi_bar = self._electrostatics(occ)
        energy = trap_levels(defects.band_reference, phi_at, ED)  # 陷阱上电子的总能量
        band = conduction_band_edge(device, self._z_nodes, phi_bar)  # 一维导带底剖面（含台阶）

        # 电极↔缺陷路径上的势垒剖面：界面双值保留真实台阶，不插成斜坡（ADR-012）
        barrier_c = _band_on_path_steps(
            self._path_cathode, self._z_nodes, band, self._iface_node_idx, self._delta_Ec
        )
        barrier_a = _band_on_path_steps(
            self._path_anode, self._z_nodes, band, self._iface_node_idx, self._delta_Ec
        )

        blocks = np.zeros((_N_BLOCKS, self.n_defects))
        # R4/R5 的 c_MP 常数因子（含局域场 F²）：用物种质量与物种带隙，两侧电极共用
        coupling = multiphonon_coupling(defects.r_D, field_at, defects.band_gap, defects.m_di)

        for side, (path, barrier, m_di_path, electrode, fermi, bottom) in enumerate(
            (
                (self._path_cathode, barrier_c, self._m_di_path_cathode, device.cathode, self._fermi_cathode, self._band_bottom[0]),
                (self._path_anode, barrier_a, self._m_di_path_anode, device.anode, self._fermi_anode, self._band_bottom[1]),
            )
        ):
            # --- R2 / R3：弹性隧穿（陷阱能级处的单一能量） ---
            transmission = wkb_transmission(path, barrier, energy, m_di_path)
            f_electrode = fermi_dirac(energy, fermi, sim.temperature)
            E1 = energy - bottom
            blocks[_B_R2C + side] = r2_elastic_capture(
                E1, ED, electrode.effective_mass, defects.m_di, f_electrode, transmission
            )
            blocks[_B_R3C + side] = r3_elastic_emission(
                E1, ED, electrode.effective_mass, defects.m_di, f_electrode, transmission
            )

            # --- R4 / R5：多声子跃迁（对 p 求和，E_p = E + p·ℏω） ---
            energies_p = energy[:, None] + self._phonon_offsets
            transmission_p = wkb_transmission(
                path[:, None, :], barrier[:, None, :], energies_p, m_di_path[:, None, :]
            )
            f_p = fermi_dirac(energies_p, fermi, sim.temperature)
            dos_p = free_electron_dos(energies_p - bottom, electrode.effective_mass)
            blocks[_B_R4C + side] = r4_multiphonon_capture(
                coupling, dos_p, f_p, transmission_p, self._phonon_weights
            )
            blocks[_B_R5C + side] = r5_multiphonon_emission(
                coupling,
                dos_p,
                f_p,
                transmission_p,
                self._phonon_weights,
                self._phonon_index,
                defects.phonon_energy[:, None],
                sim.temperature,
            )

        # --- R8：PF 热发射（折进 F03 收集概率 → R8_eff = P_A·R8） ---
        # 被阻挡层挡回原陷阱的分支是 KMC 空事件（状态不变），不单列——直接把 R8 换成
        # R8_eff 进事件表与 _conditional_flux（ADR-009/021 F03）。field_at 沿用 Phase 3
        # 的界面单侧场（ADR-017 §2），R8_eff 不另取场。
        blocks[_B_R8] = self._pf_collection * r8_pf_emission(
            defects.attempt_frequency,
            ED,
            field_at,
            defects.charge_state,
            defects.eps_opt,
            sim.temperature,
        )

        # --- R7：陷阱→陷阱（R6 的几何因子 × MA 能量因子），只在近邻表上算 ---
        # 终态 b 的能级取"电子已离开 a"之后的值 energy[b] − U_ab：当前组态里
        # energy[b] 含出发陷阱 a 上电子的贡献，跳跃发生后这部分消失。这样 MA 因子
        # 用的是完整事件能量差 ΔH_{a→b} = (E_b − E_a) − U_ab，正反向满足细致平衡
        # （见 docs/rate_equations.md R7 一节与 ADR-021 F02）。
        pairs = self._pairs
        hop = self._hop_geometric * miller_abrahams_factor(
            energy[pairs.a], energy[pairs.b] - self._hop_U_ab, sim.temperature
        )
        return blocks, hop

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
        """条件稳态下每个陷阱的阳极净通量分解。

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

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        """跑完 n_events 步并返回稳态电流与分解统计。"""
        n = self.n_defects
        sim = self.sim
        pairs = self._pairs
        occ = np.zeros(n, dtype=bool)  # 初始占据状态：全空
        burn_in = int(sim.n_events * sim.burn_in_fraction)

        elapsed = 0.0
        mech_flux = {"tat": 0.0, "pf": 0.0}
        defect_flux = np.zeros(n)
        occupancy_time = np.zeros(n)
        occupancy_mf = np.zeros(n)
        direct_flux = 0.0
        counts = 0.0  # 阳极净电子计数
        cathode_counts = 0.0  # 阴极净电子计数（F04 全局守恒）
        n_anode_events = 0
        n_cathode_events = 0
        n_executed = 0  # 实际执行的事件数（可能因总速率归零早停，如全器件无缺陷）
        measured_events = 0  # 测量窗口内执行的事件数
        n_inside_start = 0  # 测量开始时器件内电子数（初始全空 = 0）
        measuring = burn_in <= 0
        mask = np.empty((_N_BLOCKS, n))

        for step in range(sim.n_events):
            blocks, hop = self._rates(occ)
            empty = ~occ
            occf = occ.astype(float)

            # 事件表：俘获类要求空位，逃逸/发射类要求已占据
            mask[:_N_CAPTURE_BLOCKS] = empty
            mask[_N_CAPTURE_BLOCKS:] = occ
            event_blocks = blocks * mask
            event_hop = hop * occf[pairs.a] * empty[pairs.b]

            total_rate = event_blocks.sum() + event_hop.sum()
            if total_rate <= 0.0:
                break
            n_executed += 1
            dt = -np.log(self.rng.random()) / total_rate

            if measuring:
                tat, pf, p_hat = self._conditional_flux(
                    blocks, *self._hop_sums(hop, occf, empty)
                )
                elapsed += dt
                occupancy_time += occ * dt
                occupancy_mf += p_hat * dt
                defect_flux += (tat + pf) * dt
                mech_flux["tat"] += tat.sum() * dt
                mech_flux["pf"] += pf.sum() * dt
                direct_flux += (_ANODE_SIGN[:, None] * event_blocks).sum() * dt

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
                    a_sign = _ANODE_SIGN[block]
                    if a_sign != 0.0:
                        n_anode_events += 1
                        counts += a_sign
                    c_sign = _CATHODE_SIGN[block]
                    if c_sign != 0.0:
                        n_cathode_events += 1
                        cathode_counts += c_sign
            else:
                k = index - _N_BLOCKS * n
                occ[pairs.a[k]] = False
                occ[pairs.b[k]] = True

            if measuring:
                measured_events += 1
            elif step + 1 >= burn_in:
                measuring = True
                n_inside_start = int(occ.sum())  # 测量窗口起点的器件内电子数

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
        current_counts_total = current_counts + r1_tunnel + r1_thermionic
        delta_n_inside = int(occ.sum()) - n_inside_start
        validated, validation_notes = _validate_result(
            n, self._pf_forward, elapsed, measured_events, occ_dev,
            n_anode_events, current, current_counts_total,
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
            current_counts_cathode=float(cathode_counts * scale),
            n_anode_events=n_anode_events,
            delta_n_inside=delta_n_inside,
            validated=validated,
            validation_notes=validation_notes,
            defects=self.defects,
        )


def run_single(config: Config, defects: Optional[DefectEnsemble] = None) -> SimulationResult:
    """跑一个单点仿真。"""
    return KMCSimulation(config, defects=defects).run()
