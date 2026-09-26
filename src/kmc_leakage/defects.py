"""缺陷（陷阱）生成与几何工具。

缺陷参数见 docs/physics_model.md 的"缺陷（陷阱）模型"一节。

**分层生成（ADR-005/008）**：体缺陷按层生成（各层 `浓度 × 层体积`、一致取整），
界面缺陷按面密度 Nₛ·A 放在内部界面平面上、作为独立物种、物性默认取界面所属
（导带较低的一侧，即势阱侧）材料。每个缺陷携带自己的材料/物种参数、导带基准
B_ℓ、层/界面标签，以及一个"读场侧"标记（界面缺陷用单侧差分场，见 ADR-017 §2）。
全器件可以无缺陷（空集合），此时走 R1-only 路径（ADR-005 / review F07）。
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import constants
from scipy.integrate import cumulative_trapezoid

from .config import DefectConfig, DielectricMaterial, DeviceConfig

# 分块距离计算里每块的缺陷对数上限：块形状 (chunk, N)，chunk = 上限/N，
# 于是峰值内存与 N 无关，不随缺陷数平方增长
_PAIR_CHUNK_ELEMENTS = 1_000_000


def localization_radius(ED: np.ndarray, m_di: np.ndarray) -> np.ndarray:
    """缺陷局域化半径 r_D = ℏ/√(2·m_di·E_D)。

    见 docs/rate_equations.md R6 一节。能级来自高斯分布时不同缺陷的 r_D 不同，
    m_di 也可逐缺陷不同（不同层/物种），所以按元素计算。用的是隧穿质量 m_di，
    不是 m_cb。
    """
    ED = np.asarray(ED, dtype=float)
    if np.any(ED <= 0.0):
        raise ValueError("缺陷能级深度 E_D 必须为正")
    return constants.hbar / np.sqrt(2.0 * np.asarray(m_di, dtype=float) * ED)


@dataclass(frozen=True)
class DefectEnsemble:
    """一次实现（realization）里的全部缺陷，每个缺陷携带自己的物性与标签。

    体缺陷与界面缺陷合并存储；`layer_id` / `interface_id` 区分归属（另一个取 -1），
    `field_side` 标记读局域场的方式（0=体缺陷用常规插值，±1=界面缺陷用单侧差分）。
    """

    positions: np.ndarray  # (N, 3) [m]
    ED: np.ndarray  # (N,) 能级深度（相对该缺陷所属材料侧的局部导带底）[J]
    r_D: np.ndarray  # (N,) 局域化半径 [m]
    charge_state: np.ndarray  # (N,) Z（未占据时的电荷）
    m_di: np.ndarray  # (N,) 隧穿有效质量 [kg]
    eps_opt: np.ndarray  # (N,) 光学介电常数（R8 用）
    band_gap: np.ndarray  # (N,) Eg [J]（R4/R5 的 c_MP 用）
    phonon_energy: np.ndarray  # (N,) ℏω [J]
    huang_rhys: np.ndarray  # (N,) Huang 因子 S
    attempt_frequency: np.ndarray  # (N,) ν0 [1/s]
    band_reference: np.ndarray  # (N,) 所属材料导带基准 B_ℓ [J]（E_C = B_ℓ - eφ）
    layer_id: np.ndarray  # (N,) 体缺陷所属层号；界面缺陷取 -1
    interface_id: np.ndarray  # (N,) 界面缺陷所属界面号；体缺陷取 -1
    field_side: np.ndarray  # (N,) 0=体，−1=读界面下侧场，+1=读界面上侧场

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @classmethod
    def uniform(
        cls,
        positions: np.ndarray,
        ED: np.ndarray,
        material: DielectricMaterial,
        band_reference: float,
        charge_state: float = 1.0,
        layer_id: int = 0,
    ) -> "DefectEnsemble":
        """所有缺陷同材料、同导带基准的便捷构造（测试与单层退化用）。"""
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        ED = np.asarray(ED, dtype=float)
        n = positions.shape[0]

        def full(value, dtype=float):
            return np.full(n, value, dtype=dtype)

        return cls(
            positions=positions,
            ED=ED,
            r_D=localization_radius(ED, material.m_di),
            charge_state=full(charge_state),
            m_di=full(material.m_di),
            eps_opt=full(material.eps_opt),
            band_gap=full(material.band_gap),
            phonon_energy=full(material.phonon_energy),
            huang_rhys=full(material.huang_rhys),
            attempt_frequency=full(material.attempt_frequency),
            band_reference=full(band_reference),
            layer_id=full(layer_id, dtype=int),
            interface_id=full(-1, dtype=int),
            field_side=full(0, dtype=int),
        )


# 空集合（全器件无缺陷）时用的空 ensemble；R1-only 路径的地基（review F07）
def empty_ensemble() -> DefectEnsemble:
    """零缺陷的空实现（全器件无缺陷 / 空层组合）。"""
    z = np.zeros(0)
    zi = np.zeros(0, dtype=int)
    return DefectEnsemble(
        positions=np.zeros((0, 3)),
        ED=z, r_D=z, charge_state=z, m_di=z, eps_opt=z, band_gap=z,
        phonon_energy=z, huang_rhys=z, attempt_frequency=z, band_reference=z,
        layer_id=zi, interface_id=zi, field_side=zi,
    )


def defect_count(device: DeviceConfig, defects: DefectConfig) -> int:
    """按浓度 × 体积给出缺陷个数（整器件口径，供单层核对用）。"""
    return int(round(defects.concentration * device.volume))


def interface_profile(z: np.ndarray, Lz: float, enrichment: float, decay_length: float) -> np.ndarray:
    """界面富集轮廓（未归一化）。

    n(z) ∝ 1 + (g-1)·[exp(-z/λ) + exp(-(Lz-z)/λ)]

    g = enrichment 是界面处相对体内平台的密度倍数，λ = decay_length 是
    衰减长度。g=1 时退化为均匀分布。轮廓的归一化在采样时完成，缺陷总数
    始终等于 浓度 × 体积。采样时 z 相对**可采样窗口起点**、Lz 取可采样窗口
    厚度（层厚减两端 standoff）——见 _sample_z，峰因此锚在距界面 standoff 处。
    """
    z = np.asarray(z, dtype=float)
    return 1.0 + (enrichment - 1.0) * (
        np.exp(-z / decay_length) + np.exp(-(Lz - z) / decay_length)
    )


def _sample_z(
    n: int, z_min: float, z_max: float, defects: DefectConfig, rng: np.random.Generator,
) -> np.ndarray:
    """按空间分布模式采样 [z_min, z_max] 内的 z 坐标。

    界面富集轮廓锚定到**可采样窗口** [z_min, z_max]：这两端是层边界各让出
    standoff（min_separation，见 generate_defects）后的边界，富集峰因此落在
    距界面 standoff 处、而不是界面上。界面 standoff 内按数值稳定性要求本就
    不放缺陷，把峰锚在最靠近界面的可放置平面上，λ ≲ standoff 时富集才采样
    得到、直方图里才看得出来（否则峰整个落在被禁区里，采样窗口上近乎平坦）。
    """
    if defects.spatial.mode == "uniform":
        return rng.uniform(z_min, z_max, size=n)
    # 界面富集：数值逆变换采样，轮廓相对可采样窗口 [z_min, z_max] 归一
    grid = np.linspace(z_min, z_max, 2001)
    pdf = interface_profile(
        grid - z_min, z_max - z_min, defects.spatial.enrichment, defects.spatial.decay_length
    )
    cdf = cumulative_trapezoid(pdf, grid, initial=0.0)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)


def _sample_ED(n: int, level, rng: np.random.Generator) -> np.ndarray:
    """按能级分布模式采样 E_D（高斯模式下截断到正值）。体缺陷与界面缺陷共用。"""
    if level.mode == "fixed":
        return np.full(n, level.ED)
    ED = rng.normal(level.ED, level.sigma, size=n)
    # 截断高斯：E_D ≤ 0 没有物理意义（r_D 会发散），重采样
    for _ in range(1000):
        bad = ED <= 0.0
        if not np.any(bad):
            return ED
        ED[bad] = rng.normal(level.ED, level.sigma, size=int(bad.sum()))
    raise RuntimeError("高斯能级采样无法得到全部为正的 E_D，请检查 ED_eV / sigma_eV")


def minimum_image_delta(delta: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """xy 周期边界下的最小镜像位移（z 方向不周期）。"""
    delta = np.array(delta, dtype=float, copy=True)
    delta[..., 0] -= Lx * np.round(delta[..., 0] / Lx)
    delta[..., 1] -= Ly * np.round(delta[..., 1] / Ly)
    return delta


def pair_distances(positions: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """缺陷两两距离矩阵 r_ab（xy 取最小镜像）。

    稠密 (N,N) 结果，N 上千时不适合放进 KMC 主流程——那里用
    `neighbor_pairs`。这里保留完整矩阵，供小 N 下的核对与测试使用。
    """
    delta = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(minimum_image_delta(delta, Lx, Ly), axis=-1)


@dataclass(frozen=True)
class NeighborPairs:
    """截断半径内的有序缺陷对，作为 R6/R7 事件表的索引。

    R7 的 Θ 因子不对称（上坡付热激活代价、下坡不付），a→b 与 b→a 是两条
    不同的跃迁，所以两个方向各占一条；自身对 (i,i) 不进表。
    """

    a: np.ndarray  # (P,) 起点缺陷编号
    b: np.ndarray  # (P,) 终点缺陷编号
    r: np.ndarray  # (P,) 两缺陷距离 r_ab [m]

    def __len__(self) -> int:
        return int(self.a.shape[0])


def hop_cutoff_radius(alpha_min: float, tolerance: float) -> float:
    """R6/R7 几何重叠因子衰减到 `tolerance` 处的距离，作为截断半径。

    重叠因子沿路径衰减 exp[-∫(α_a+α_b)ds]，最慢衰减对应两端都取整个实现里
    **最小**的局域衰减常数 α（最大 r_D、最慢材料）。要求 exp(-2·α_min·r_cut)
    = tolerance，即 r_cut = -ln(tolerance)/(2·α_min)。异质叠层里 α_min 要覆盖
    所有材料与能级（ADR-013 / review F05）。传入 α_min = 1/max(r_D) 即退化为
    单一材料下按最大 r_D 定半径的旧口径。
    """
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance 必须落在 (0, 1) 内")
    if alpha_min <= 0.0:
        raise ValueError("alpha_min 必须为正")
    return -0.5 * float(np.log(tolerance)) / float(alpha_min)


def neighbor_pairs(positions: np.ndarray, Lx: float, Ly: float, r_cut: float) -> NeighborPairs:
    """列出距离小于 `r_cut` 的有序缺陷对（xy 取最小镜像）。

    R6 的 e 折长度只有 r_D/2（参考预设下 0.083 nm），远小于缺陷平均间距，
    所以稠密距离矩阵里绝大多数元素在双精度下就是零。这里分块求距离、只留
    截断内的对：峰值内存 O(`_PAIR_CHUNK_ELEMENTS`)，与 N² 无关。零缺陷 / 单个
    缺陷时没有任何有序对，返回空表（review F07）。
    """
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    empty = NeighborPairs(a=np.zeros(0, dtype=int), b=np.zeros(0, dtype=int), r=np.zeros(0))
    if n < 2:
        return empty
    if r_cut <= 0.0:
        raise ValueError("r_cut 必须为正")
    chunk = max(1, _PAIR_CHUNK_ELEMENTS // n)
    a_parts, b_parts, r_parts = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        delta = positions[start:stop, None, :] - positions[None, :, :]
        r = np.linalg.norm(minimum_image_delta(delta, Lx, Ly), axis=-1)
        r[np.arange(stop - start), np.arange(start, stop)] = np.inf  # 排除自身对
        rows, cols = np.nonzero(r < r_cut)
        a_parts.append(rows + start)
        b_parts.append(cols)
        r_parts.append(r[rows, cols])
    return NeighborPairs(
        a=np.concatenate(a_parts),
        b=np.concatenate(b_parts),
        r=np.concatenate(r_parts),
    )


def _global_standoff(device: DeviceConfig) -> float:
    """全局最小间距：缺陷两两之间 + 到电极 + 到内部界面（体缺陷避让）的距离。

    取各配置层里 min_separation 的最大值——同一实现里用统一的间距约束，跨层
    缺陷对、内部界面避让、电极避让都吃这一个值。
    """
    seps = [layer.defects.min_separation for layer in device.stack.layers if layer.defects is not None]
    return max(seps) if seps else 0.0


def _accept_candidate(
    candidate: np.ndarray, placed: List[np.ndarray], standoff: float, Lx: float, Ly: float
) -> bool:
    """候选缺陷是否满足与已放置缺陷的最小间距（xy 取最小镜像）。"""
    if standoff <= 0.0 or not placed:
        return True
    delta = minimum_image_delta(candidate[None, :] - np.array(placed), Lx, Ly)
    return bool(np.min(np.linalg.norm(delta, axis=-1)) >= standoff)


def _interface_host(device: DeviceConfig, band_refs: np.ndarray, interface_index: int) -> Tuple[int, int]:
    """界面缺陷的宿主层（导带较低的势阱侧）与读场侧（−1 下侧 / +1 上侧）。

    ADR-008 的"物性默认取 ZrO₂ 侧"在一般叠层里落为"取导带基准较低的一侧"
    （ZrO₂ 侧正是势阱侧）；相等时取下侧。读场侧同为该侧（ADR-017 §2）。
    """
    lower, upper = interface_index, interface_index + 1
    if band_refs[upper] < band_refs[lower]:
        return upper, +1
    return lower, -1


def _extend(store: dict, material: DielectricMaterial, ED: np.ndarray, positions: np.ndarray,
            band_reference: float, charge_state: float, layer_id: int, interface_id: int,
            field_side: int) -> None:
    """把一批同材料、同标签的缺陷追加进逐字段的收集器。"""
    n = len(ED)
    store["positions"].append(np.asarray(positions, dtype=float).reshape(n, 3))
    store["ED"].append(ED)
    store["r_D"].append(localization_radius(ED, material.m_di))
    store["charge_state"].append(np.full(n, charge_state))
    store["m_di"].append(np.full(n, material.m_di))
    store["eps_opt"].append(np.full(n, material.eps_opt))
    store["band_gap"].append(np.full(n, material.band_gap))
    store["phonon_energy"].append(np.full(n, material.phonon_energy))
    store["huang_rhys"].append(np.full(n, material.huang_rhys))
    store["attempt_frequency"].append(np.full(n, material.attempt_frequency))
    store["band_reference"].append(np.full(n, band_reference))
    store["layer_id"].append(np.full(n, layer_id, dtype=int))
    store["interface_id"].append(np.full(n, interface_id, dtype=int))
    store["field_side"].append(np.full(n, field_side, dtype=int))


def generate_defects(device: DeviceConfig, rng: np.random.Generator) -> DefectEnsemble:
    """生成一次缺陷实现（分层体缺陷 + 界面缺陷）。

    生成顺序：先各界面缺陷（放在界面平面、xy 随机），再各层体缺陷。全局最小间距
    （`_global_standoff`）同时作用于缺陷两两之间、缺陷到电极、体缺陷到内部界面；
    界面缺陷豁免"内部界面避让"（它们本就在界面上），但仍守缺陷两两间距。单层退化
    时（无界面、单层覆盖整器件）与旧的整器件生成逐位相同。全器件无缺陷返回空实现。
    """
    stack = device.stack
    band_refs = device.layer_band_references()
    standoff = _global_standoff(device)
    edges = stack.layer_edges

    store = {key: [] for key in (
        "positions", "ED", "r_D", "charge_state", "m_di", "eps_opt", "band_gap",
        "phonon_energy", "huang_rhys", "attempt_frequency", "band_reference",
        "layer_id", "interface_id", "field_side",
    )}
    placed: List[np.ndarray] = []

    def place(n: int, z_sampler, standoff_local: float) -> np.ndarray:
        """拒绝采样放 n 个缺陷，返回 (n,3) 坐标（xy 随机、z 由 z_sampler 给）。"""
        out = np.empty((n, 3))
        accepted = 0
        attempts = 0
        max_attempts = 1000 * max(n, 1)
        while accepted < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "在 min_separation 约束下无法放下 {} 个缺陷，请调小 min_separation_nm "
                    "或降低浓度".format(n)
                )
            candidate = np.array([rng.uniform(0.0, device.Lx), rng.uniform(0.0, device.Ly), z_sampler()])
            if _accept_candidate(candidate, placed, standoff_local, device.Lx, device.Ly):
                out[accepted] = candidate
                placed.append(candidate)
                accepted += 1
        return out

    # --- 界面缺陷：独立物种，放在界面平面（z 固定），物性取势阱侧材料 ---
    for j, interface in enumerate(stack.interfaces):
        cfg = interface.defects
        if cfg is None:
            continue
        n_s = int(round(cfg.surface_density * device.area))
        if n_s < 1:
            continue
        host_layer, field_side = _interface_host(device, band_refs, j)
        material = stack.material_of(stack.layers[host_layer])
        z_plane = float(stack.interface_positions[j])
        positions = place(n_s, lambda: z_plane, standoff)
        ED = _sample_ED(n_s, cfg.level, rng)
        _extend(store, material, ED, positions, float(band_refs[host_layer]),
                cfg.charge_state, -1, j, field_side)

    # --- 体缺陷：逐层生成，z 在 [层起点+standoff, 层终点-standoff] 内按分布采样 ---
    for ell, layer in enumerate(stack.layers):
        cfg = layer.defects
        if cfg is None:
            continue
        thickness = layer.thickness
        n_ell = int(round(cfg.concentration * device.area * thickness))
        if n_ell < 1:
            continue
        z_lo, z_hi = edges[ell] + standoff, edges[ell + 1] - standoff
        if z_hi <= z_lo:
            raise ValueError(
                "层 {} 太薄，min_separation 约束下 z 方向没有可放置体缺陷的区间".format(layer.layer_id)
            )
        material = stack.material_of(layer)
        positions = place(
            n_ell,
            lambda cfg=cfg, z_lo=z_lo, z_hi=z_hi: float(_sample_z(1, z_lo, z_hi, cfg, rng)[0]),
            standoff,
        )
        ED = _sample_ED(n_ell, cfg.level, rng)
        _extend(store, material, ED, positions, float(band_refs[ell]),
                cfg.charge_state, ell, -1, 0)

    if not store["ED"]:
        return empty_ensemble()

    return DefectEnsemble(
        positions=np.concatenate(store["positions"]),
        ED=np.concatenate(store["ED"]),
        r_D=np.concatenate(store["r_D"]),
        charge_state=np.concatenate(store["charge_state"]),
        m_di=np.concatenate(store["m_di"]),
        eps_opt=np.concatenate(store["eps_opt"]),
        band_gap=np.concatenate(store["band_gap"]),
        phonon_energy=np.concatenate(store["phonon_energy"]),
        huang_rhys=np.concatenate(store["huang_rhys"]),
        attempt_frequency=np.concatenate(store["attempt_frequency"]),
        band_reference=np.concatenate(store["band_reference"]),
        layer_id=np.concatenate(store["layer_id"]),
        interface_id=np.concatenate(store["interface_id"]),
        field_side=np.concatenate(store["field_side"]),
    )


def z_histogram(ensemble: DefectEnsemble, Lz: float, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """沿 z 的缺陷数密度直方图，返回 (bin 中心, 计数)。"""
    counts, edges = np.histogram(ensemble.positions[:, 2], bins=n_bins, range=(0.0, Lz))
    return 0.5 * (edges[:-1] + edges[1:]), counts


def expected_depth_density(
    device: DeviceConfig, points_per_layer: int = 400
) -> Tuple[np.ndarray, np.ndarray]:
    """体缺陷期望深度数密度 ρ(z)，与 `_sample_z` 的采样口径同源。

    轮廓锚定到可采样窗口 [层边界+standoff, 层边界−standoff]：standoff 死区内
    ρ=0（数值稳定性约束本就不放缺陷），窗口内按 uniform / interface 轮廓归一。
    数量守恒 N_ℓ = 浓度 × A × 层厚 → ρ(z) = 浓度 · 层厚 · pdf_z(z)（pdf_z 在窗口上
    积分为 1，standoff 把缺陷挤进更窄的窗口，故平台略高于标称浓度）。

    返回 (z, ρ)：z [m] 覆盖各层可采样窗口、层与层之间用 nan 断开，ρ [m^-3]。
    给可视化叠画解析轮廓用——汇总 M 个实现除以 M 后的深度直方图应当贴合这条线，
    是"采样器是否照轮廓在放缺陷"的独立对照，与直方图本身的泊松噪声无关。
    只含体缺陷（界面缺陷是面密度、量纲不同，不进这条曲线）。
    """
    standoff = _global_standoff(device)
    edges = device.stack.layer_edges
    z_parts: List[np.ndarray] = []
    rho_parts: List[np.ndarray] = []
    for ell, layer in enumerate(device.stack.layers):
        cfg = layer.defects
        z_lo, z_hi = edges[ell] + standoff, edges[ell + 1] - standoff
        if cfg is None or z_hi <= z_lo:
            continue
        z = np.linspace(z_lo, z_hi, points_per_layer)
        if cfg.spatial.mode == "uniform":
            shape = np.ones_like(z)
        else:
            shape = interface_profile(
                z - z_lo, z_hi - z_lo, cfg.spatial.enrichment, cfg.spatial.decay_length
            )
        pdf = shape / np.trapz(shape, z)  # ∫ pdf dz = 1 over [z_lo, z_hi]
        rho = cfg.concentration * layer.thickness * pdf
        if z_parts:  # 层间用 nan 断开，画出来是分段的
            z_parts.append(np.array([np.nan]))
            rho_parts.append(np.array([np.nan]))
        z_parts.append(z)
        rho_parts.append(rho)
    if not z_parts:
        return np.array([]), np.array([])
    return np.concatenate(z_parts), np.concatenate(rho_parts)
