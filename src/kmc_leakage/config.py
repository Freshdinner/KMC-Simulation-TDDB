"""配置数据结构。

**所有字段一律 SI 单位**（J, m, K, C, s, 1/m^3）。YAML 里的 eV / nm /
cm^-3 / degC 在 `io.py` 的读入边界换算，换算系数取自 scipy.constants。

介质结构采用"有序介质层 + 材料库 + 显式界面"表示（ADR-003）：`DielectricStack`
持有按名索引的材料库、有序 `layers`（各引用材料、给厚度、配缺陷）与内部
`interfaces`。单层退化为只含一个元素的 `layers`。全项目经 `DielectricStack`
的统一查询接口取材料参数，不再直接读单一 `dielectric` 字段。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ElectrodeMaterial:
    """电极材料参数。"""

    name: str
    work_function: float  # 功函数 Φ_m [J]
    effective_mass: float  # 有效质量 [kg]（默认自由电子质量）
    fermi_energy: float  # 费米能相对电极导带底的高度 [J]

    # fermi_energy 是 R2/R3 的 E1（"电极中电子总能量"）和 R4/R5 的 N_C
    # （自由电子态密度 ∝ √E）所需的电极侧能量零点：全局能量零点取阴极
    # 费米能级，电极导带底则位于 -fermi_energy 处。


@dataclass(frozen=True)
class DielectricMaterial:
    """介质材料参数。

    两个介电常数不可互换：`kappa` 只进泊松方程，`eps_opt` 只进 R8。
    两种有效质量也不可互换：`m_cb`/`m_vb` 是裸带边质量（Weinberg 关系的
    输入），`m_di` 是速率公式 R1–R7 里实际使用的隧穿质量。
    """

    name: str
    kappa: float  # 静态介电常数（泊松方程用）
    eps_opt: float  # 光学介电常数（**只**用于 R8）
    band_gap: float  # Eg [J]
    electron_affinity: float  # 电子亲和能 χ [J]
    m_cb: float  # 导带边有效质量 [kg]
    m_vb: float  # 价带边有效质量 [kg]
    m_di: float  # 隧穿有效质量 [kg]（当前数值上 = m_cb，但独立配置）
    phonon_energy: float  # ℏω [J]
    huang_rhys: float  # Huang 因子 S
    attempt_frequency: float  # ν0 [1/s]，R6/R7/R8 共用


@dataclass(frozen=True)
class DefectLevelConfig:
    """缺陷能级分布：固定值或高斯。"""

    mode: str  # "fixed" | "gaussian"
    ED: float  # 能级深度（相对导带底）均值 [J]
    sigma: float = 0.0  # 高斯标准差 [J]


@dataclass(frozen=True)
class DefectSpatialConfig:
    """缺陷空间分布：体内均匀或界面富集。

    界面富集轮廓（本项目选定的形式）::

        n(z) ∝ 1 + (g-1)·[exp(-z/λ) + exp(-(Lz-z)/λ)]

    g = enrichment（界面处相对体内的富集因子），λ = decay_length。
    轮廓做归一化，缺陷总数始终 = 浓度 × 体积，因此浓度控制数量、轮廓只
    控制形状——这样不同轮廓下的 I-V 才是在同一缺陷数下比较。采样时轮廓
    锚定到可采样窗口（层厚减两端 min_separation），峰落在距界面 standoff
    处，见 defects._sample_z。
    """

    mode: str  # "uniform" | "interface"
    enrichment: float = 1.0
    decay_length: float = 0.0  # [m]


@dataclass(frozen=True)
class DefectConfig:
    """一层体缺陷（陷阱）参数。"""

    concentration: float  # nD [1/m^3]
    level: DefectLevelConfig
    spatial: DefectSpatialConfig
    charge_state: float = 1.0  # 未占据时的电荷 Z（施主型 Z=+1）
    min_separation: float = 0.0  # 最近邻距离约束 [m]


@dataclass(frozen=True)
class InterfaceDefectConfig:
    """界面缺陷参数：以面密度 Nₛ 配置的独立物种（ADR-008）。

    Phase 1 只解析、校验单位并存储；界面缺陷的生成（数量 = Nₛ·A、放在界面
    平面上）属 Phase 2，届时再补物种物性（m_di / ε_opt / ℏω / S 默认取 ZrO₂
    侧材料值）。能级 ED 相对界面所属材料侧的局部导带底。
    """

    surface_density: float  # Nₛ [1/m^2]
    level: DefectLevelConfig
    charge_state: float = 1.0  # Z（未占据时的电荷）


@dataclass(frozen=True)
class Layer:
    """叠层中一段连续同材料区域（见 glossary 的 Layer）。

    有独立 `layer_id`、材料引用、厚度与体缺陷配置。两层同材料可共享材料参数
    但各有独立 `layer_id`，以便分别设缺陷。
    """

    layer_id: str
    material: str  # 引用 DielectricStack.materials 里的材料名
    thickness: float  # t_ℓ [m]
    defects: Optional[DefectConfig] = None  # 该层体缺陷配置；None = 空层（如阻挡层体相）


@dataclass(frozen=True)
class Interface:
    """相邻两层的内部界面（见 glossary 的 Interface）"""

    delta_Ec: Optional[float] = None  # 导带带阶 ΔE_C [J]；None = 电子亲和能对齐
    sigma: float = 0.0  # 固定界面面电荷 [C/m^2]
    defects: Optional[InterfaceDefectConfig] = None


@dataclass(frozen=True)
class DielectricStack:
    """有序介质层 + 材料库 + 内部界面"""

    materials: Dict[str, DielectricMaterial]
    layers: Tuple[Layer, ...]
    interfaces: Tuple[Interface, ...]

    @property
    def Lz(self) -> float:
        """介质总厚度 = 各层厚度之和 [m]。"""
        return float(sum(layer.thickness for layer in self.layers))

    def material_of(self, layer: Layer) -> DielectricMaterial:
        """给定层的材料参数。"""
        return self.materials[layer.material]

    @property
    def single_material(self) -> DielectricMaterial:
        """单层路径下唯一的介质材料。

        层引用多于一种材料时抛 NotImplementedError——多材料叠层的分层物理
        （变 κ 泊松、分段 WKB、按材料取速率参数）在 Phase 3/4 实现。
        """
        names = {layer.material for layer in self.layers}
        if len(names) != 1:
            raise NotImplementedError(
                "多材料叠层的分层物理在 Phase 3/4 实现；Phase 1 只支持单一介质"
                "材料的单层路径（当前层引用了 {} 种材料）".format(len(names))
            )
        return self.materials[next(iter(names))]

    @property
    def bulk_defects(self) -> DefectConfig:
        """单层路径下的体缺陷配置（唯一一层带 defects 的那层）。

        多于一层带体缺陷时抛 NotImplementedError——分层缺陷生成属 Phase 2。
        """
        configured = [layer.defects for layer in self.layers if layer.defects is not None]
        if len(configured) == 1:
            return configured[0]
        if not configured:
            raise ValueError("没有任何层配置体缺陷（layers[i].defects）")
        raise NotImplementedError(
            "分层体缺陷生成在 Phase 2 实现；Phase 1 只支持单层的体缺陷配置"
            "（当前 {} 层带体缺陷）".format(len(configured))
        )

    @classmethod
    def single(
        cls,
        material: DielectricMaterial,
        thickness: float,
        defects: Optional[DefectConfig] = None,
        layer_id: str = "L0",
    ) -> "DielectricStack":
        """单层叠层：一种材料、一层、无内部界面（单层退化的构造入口）。"""
        return cls(
            materials={material.name: material},
            layers=(Layer(layer_id=layer_id, material=material.name, thickness=thickness, defects=defects),),
            interfaces=(),
        )

    # ------------------------------------------------------------------
    # 叠层几何（z 方向）——分层缺陷生成、变 κ 泊松、分段能带/WKB 共用
    # ------------------------------------------------------------------
    @property
    def layer_edges(self) -> np.ndarray:
        """各层在 z 方向的累积边界 [m]，长度 = 层数+1：[0, t0, t0+t1, …, Lz]。"""
        return np.concatenate(([0.0], np.cumsum([layer.thickness for layer in self.layers])))

    @property
    def interface_positions(self) -> np.ndarray:
        """各内部界面的 z 坐标 [m]，长度 = 层数-1（= layer_edges 去掉两端）。"""
        return self.layer_edges[1:-1]

    def layer_index_of(self, z: float) -> int:
        """z 落在第几层（半开区间 [edge_i, edge_{i+1})，两端夹住）。"""
        return int(self.layer_indices_of(np.asarray([z]))[0])

    def layer_indices_of(self, z: np.ndarray) -> np.ndarray:
        """向量化的 `layer_index_of`：逐个 z 落在第几层（半开区间，两端夹住）。

        变 κ 泊松（按单元取 κ）、分段能带（按位置取 B_ℓ / Eg）共用同一套按位置
        查询，不再各写一份。界面对齐到网格节点后，单元中心不会落在界面上，归属
        唯一。界面节点按约定归上侧层；节点坐标 k·hz 与层厚累加值可能差 1 ulp
        （如 8 nm/nz=40 下 z₁₅ = 3e-9 < 3.0000000000000004e-9），所以比较前加
        Lz·1e-9 的容差，避免界面节点因舍入落回下侧层。
        """
        edges = self.layer_edges
        z = np.asarray(z, dtype=float) + 1e-9 * edges[-1]
        idx = np.searchsorted(edges, z, side="right") - 1
        return np.clip(idx, 0, len(self.layers) - 1)

    def kappa_at(self, z: np.ndarray) -> np.ndarray:
        """各 z 处所属层的静态介电常数 κ（变 κ 泊松按单元取值，ADR-011）。"""
        kappas = np.array([self.materials[layer.material].kappa for layer in self.layers])
        return kappas[self.layer_indices_of(z)]

    @property
    def layer_m_di(self) -> np.ndarray:
        """各层材料的隧穿有效质量 m_di [kg]，长度 = 层数。"""
        return np.array([self.materials[layer.material].m_di for layer in self.layers])

    @property
    def layer_kappa(self) -> np.ndarray:
        """各层材料的静态介电常数 κ，长度 = 层数（F03 的无电荷分压 Σ t_ℓ/κ_ℓ 用）。"""
        return np.array([self.materials[layer.material].kappa for layer in self.layers])

    @property
    def layer_band_gap(self) -> np.ndarray:
        """各层材料的带隙 Eg [J]，长度 = 层数（价带顶剖面与界面双值台阶用）。"""
        return np.array([self.materials[layer.material].band_gap for layer in self.layers])

    @property
    def layer_thickness(self) -> np.ndarray:
        """各层厚度 t_ℓ [m]，长度 = 层数（= layer_edges 的相邻差）。"""
        return np.diff(self.layer_edges)

    def m_di_at(self, z: np.ndarray) -> np.ndarray:
        """各 z 处所属层的隧穿有效质量 m_di [kg]（分段 WKB 的路径传播质量，ADR-012）。

        按位置取所在层材料的 m_di——这是速率公式里 WKB 积分沿路径实际使用的
        质量。与"物种质量"（缺陷自身的 m_di，进 r_D 与 R2/R3 前置因子）区分开
        （review F08）。单层退化时逐点恒为该单一材料的 m_di。
        """
        return self.layer_m_di[self.layer_indices_of(z)]

    @property
    def n_materials(self) -> int:
        """叠层引用的不同材料种数（=1 时是单一材料叠层，退回闭式几何因子/势垒）。"""
        return len({layer.material for layer in self.layers})

    def delta_Ec(self, interface_index: int) -> float:
        """界面 j（层 j 与 j+1 之间）的导带带阶 ΔE_C [J]。

        显式 `delta_Ec` 覆盖优先；否则按电子亲和能对齐 χ_j − χ_{j+1}（ADR-004）。
        """
        iface = self.interfaces[interface_index]
        if iface.delta_Ec is not None:
            return iface.delta_Ec
        chi_low = self.material_of(self.layers[interface_index]).electron_affinity
        chi_high = self.material_of(self.layers[interface_index + 1]).electron_affinity
        return chi_low - chi_high

    @property
    def band_offsets(self) -> np.ndarray:
        """各层导带基准相对第一层的累积带阶 [J]，长度 = 层数。

        offsets[ℓ] = Σ_{j<ℓ} ΔE_C,j；offsets[-1] = ΣΔE_C（V_bi 关系里的那一项）。
        """
        offsets = np.zeros(len(self.layers))
        for j in range(len(self.interfaces)):
            offsets[j + 1] = offsets[j] + self.delta_Ec(j)
        return offsets


@dataclass(frozen=True)
class DeviceConfig:
    """器件几何 + 介质叠层 + 电极/接触。"""

    Lx: float  # [m]
    Ly: float  # [m]
    cathode: ElectrodeMaterial
    anode: ElectrodeMaterial
    stack: DielectricStack
    barrier_height: Optional[float] = None  # 显式指定的接触势垒 E_B [J]，None 表示推算

    @property
    def Lz(self) -> float:
        """介质总厚度 = 叠层各层厚度之和 [m]。"""
        return self.stack.Lz

    @property
    def area(self) -> float:
        """电极面积 A = Lx·Ly [m^2]。"""
        return self.Lx * self.Ly

    @property
    def volume(self) -> float:
        """介质体积 [m^3]。"""
        return self.Lx * self.Ly * self.Lz

    def barrier(self, electrode: ElectrodeMaterial) -> float:
        """给定电极处的零场接触势垒 E_B [J]。

        显式配置的 barrier_height 优先；否则按能带对齐 E_B = Φ_m - χ 推算，χ
        取该电极所接触那一层的材料（阴极→第一层，阳极→最后一层）。
        """
        if self.barrier_height is not None:
            return self.barrier_height
        contact_layer = self.stack.layers[0] if electrode is self.cathode else self.stack.layers[-1]
        return electrode.work_function - self.stack.material_of(contact_layer).electron_affinity

    def layer_band_references(self) -> np.ndarray:
        """各层导带基准 B_ℓ 在全局能标（阴极费米能级为零点）上的位置 [J]。

        B_0 = 阴极接触势垒；B_ℓ = B_0 + 累积带阶（见 stack.band_offsets）。
        导带底剖面 E_C(z) = B_{ℓ(z)} - e·φ(z)（见 bands.py，review §4.1）。
        """
        return self.barrier(self.cathode) + self.stack.band_offsets

    def band_reference_profile(self, z: np.ndarray) -> np.ndarray:
        """各 z 处所属层的导带基准 B_ℓ(z) [J]。

        分段能带的台阶：E_C(z) = B_ℓ(z) - e·φ(z)，界面处 B_ℓ 跳变（真实台阶，不
        插斜坡）。单层退化时逐点恒为接触势垒 E_B（`band_offsets` 全零）。
        """
        return self.layer_band_references()[self.stack.layer_indices_of(z)]

    def band_gap_profile(self, z: np.ndarray) -> np.ndarray:
        """各 z 处所属层材料的带隙 Eg(z) [J]（价带顶 E_V = E_C - Eg 的分段台阶）。"""
        return self.stack.layer_band_gap[self.stack.layer_indices_of(z)]

    def built_in_potential(self) -> float:
        """内建电势 [V]（加在 z=Lz 边界上），满足 q·V_bi = B_c + ΣΔE_C - B_a。

        B_c/B_a 为阴/阳极接触势垒，ΣΔE_C 为累积带阶（review §4.1）。电子亲和能
        对齐时退化为 (Φ_c - Φ_a)/q；显式带阶覆盖会进入这个累积关系。单层且
        显式指定同一 E_B 时两界面势垒相同、无带阶，内建电势为 0。
        """
        from scipy import constants

        total_offset = float(self.stack.band_offsets[-1])
        return (
            self.barrier(self.cathode) + total_offset - self.barrier(self.anode)
        ) / constants.elementary_charge


@dataclass(frozen=True)
class GridConfig:
    """泊松方程网格。nx/ny 为周期方向格点数，nz 为 z 方向单元数。"""

    nx: int = 32
    ny: int = 32
    nz: int = 32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """节点数组形状（z 方向 nz+1 个节点，两端为 Dirichlet 边界）。"""
        return (self.nx, self.ny, self.nz + 1)


@dataclass(frozen=True)
class SimulationConfig:
    """单点仿真参数。"""

    voltage: float  # [V]（加在阳极 z=Lz 上）
    temperature: float  # [K]
    n_events: int = 20000
    burn_in_fraction: float = 0.2
    seed: int = 12345
    grid: GridConfig = GridConfig()
    # 快陷阱判据 R_cut [1/s]：与电极交换电子的总速率 ≥ 此值的陷阱移出事件表、占据取
    # 条件稳态 p̂（ADR-023）。None = 不移出，逐位退回原算法
    fast_trap_rate: Optional[float] = None


@dataclass(frozen=True)
class SweepConfig:
    """I-V / I-T 扫描范围。"""

    voltages: np.ndarray  # [V]
    temperatures: np.ndarray  # [K]


@dataclass(frozen=True)
class EnsembleConfig:
    """系综平均：每个种子对应一次独立的缺陷实现，结果取算术平均。"""

    seeds: Tuple[int, ...]


@dataclass(frozen=True)
class TDDBConfig:
    """TDDB 缺陷生成参数（PROJECT_CONTEXT_TDDB.md 第 3 节参数表，ADR-024/028/035/036/047）。

    只收已定取值的参数，每个字段在用到它的 Phase 加入。G 场加速项里的相对介电常数 k 取
    介质材料的静态 κ，前驱俘获线型用的声子能量 ℏω₀ 取宿主介质的值，应力电压与环境温度
    取 simulation 段的工作点，都不在这里另设。
    """

    activation_energy: float  # E_A：无场下形成 Frenkel 对的激活能 [J]（式 T2；双电子机制下即 E_A,2e）
    precursor_ionization_energy: float  # E_t1：前驱能级相对局部导带底的深度 [J]（式 T4/T9）
    # E_rel1：第一次电子俘获导致 O-Zr-O 键断裂、O 空位形成的弛豫能 [J]。只进 R_1e 的多声子
    # 线型（E_rel1/ℏω₀ 作为式 T10 的参数），与输运用的材料 Huang-Rhys 因子 huang_rhys 无关（ADR-030）
    precursor_relaxation_energy: float
    dipole_moment: float  # p₀：前驱有效偶极矩均值 [C·m]（式 T2/T3）
    # P_bb 用的宏观场：泊松解的场在标准差为此值的 Gaussian 窗口上取空间平均 [m]（式 T2，ADR-043）。
    # Lorentz 局域场因子 (2+k)/3 针对宏观场，窗口取 Lorentz 球的尺度；0 = 直接用泊松网格上的局域场
    field_averaging_width: float
    # --- 瞬态驱动（Phase 3，ADR-033）---
    # N_T：前驱位点浓度 [1/m^3]，体积生成率 N_T·R_1e·P_bb 的前置因子（§6.1 模型 C）。0 = 关掉生成
    precursor_concentration: float
    # 生成单元网格 (nx, ny, nz)：单元铺满 [0,Lx)×[0,Ly)×[standoff, Lz−standoff]，外层 Gillespie
    # 在单元表上抽生成事件（§6.1）。与泊松网格分开配置
    generation_grid: Tuple[int, int, int]
    max_generated_vacancies: int  # 生成的空位数达到此值即停（机时上限）
    stop_time: float  # 应力时间上限 [s]
    # --- 击穿判据与热反馈（Phase 4，ADR-036/047）---
    # 击穿电流密度 [A/m^2]：电流密度超过它即判击穿，这一刻就是 t_BD（§6.3，ADR-047）
    breakdown_current_density: float
    # k_TH：介质热导率 [W/(m·K)]（式 T14）。None = 等温，温度固定为环境值（Phase 3 的口径）
    thermal_conductivity: Optional[float] = None


@dataclass(frozen=True)
class Config:
    """完整配置。"""

    device: DeviceConfig
    simulation: SimulationConfig
    sweep: Optional[SweepConfig] = None
    ensemble: Optional[EnsembleConfig] = None
    tddb: Optional[TDDBConfig] = None  # 省略 = 纯漏电流仿真（稳态 run/sweep/plot 不读它）

    @property
    def seeds(self) -> Tuple[int, ...]:
        """系综里各实现的种子；没有 ensemble 段时退化为单个 simulation.seed。"""
        if self.ensemble is None:
            return (self.simulation.seed,)
        return self.ensemble.seeds

    @property
    def defects(self) -> DefectConfig:
        """单层路径下的体缺陷配置（= device.stack.bulk_defects）。

        Phase 1 保留 `config.defects` 这个读法给缺陷生成用；Phase 2 起改为逐层
        生成，届时消费方改读 stack.layers[i].defects。
        """
        return self.device.stack.bulk_defects
