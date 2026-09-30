"""配置文件（YAML）读写与单位换算。

**这里是唯一的单位换算边界**：YAML 用器件物理惯用单位（eV, nm, cm^-3,
cm^-2, degC），换算系数一律取自 `scipy.constants`，不写字面量；换算之后
所有下游模块只见 SI。

结构 schema（ADR-003/015）：顶层 `materials`（按名字典）+ `layers`（有序层，
各引用材料、给 `thickness_nm`、配 `defects`）+ `interfaces`（可选，数量 =
层数-1，携带 ΔE_C 覆盖、σ、界面缺陷）+ `device`（几何与接触势垒）+
`electrodes`。`Lz` 由层厚求和。旧 schema 的顶层 `dielectric` / `defects` 与
`device.Lz_nm` 一律直接报错并提示新写法（不做运行时兼容层）。可选的顶层 `tddb`
段给 TDDB 缺陷生成参数（ADR-024），省略即纯漏电流仿真。
"""

from typing import Any, Dict, Optional

import numpy as np
import yaml
from scipy import constants

from .config import (
    Config,
    DefectConfig,
    DefectLevelConfig,
    DefectSpatialConfig,
    DeviceConfig,
    DielectricMaterial,
    DielectricStack,
    ElectrodeMaterial,
    EnsembleConfig,
    GridConfig,
    Interface,
    InterfaceDefectConfig,
    Layer,
    SimulationConfig,
    SweepConfig,
    TDDBConfig,
)

# --- 单位换算系数（全部来自 scipy.constants） ---
EV = constants.electron_volt  # eV -> J
NM = constants.nano  # nm -> m
PER_CM3 = constants.centi ** -3  # cm^-3 -> m^-3（体浓度）
PER_CM2 = constants.centi ** -2  # cm^-2 -> m^-2（界面面密度、面电荷）
M_E = constants.electron_mass  # m_e -> kg
E_ANGSTROM = constants.elementary_charge * constants.angstrom  # e·Å -> C·m（偶极矩）


def celsius_to_kelvin(t_c: float) -> float:
    """摄氏度 -> 开尔文。"""
    return float(t_c) + constants.zero_Celsius


def kelvin_to_celsius(t_k: float) -> float:
    """开尔文 -> 摄氏度。"""
    return float(t_k) - constants.zero_Celsius


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise KeyError("配置缺少必需项 {}.{}".format(where, key))
    return mapping[key]


def _electrode(raw: Dict[str, Any], where: str) -> ElectrodeMaterial:
    return ElectrodeMaterial(
        name=str(raw.get("name", "electrode")),
        work_function=float(_require(raw, "work_function_eV", where)) * EV,
        effective_mass=float(raw.get("effective_mass_me", 1.0)) * M_E,
        fermi_energy=float(raw.get("fermi_energy_eV", 5.0)) * EV,
    )


def _material(name: str, raw: Dict[str, Any]) -> DielectricMaterial:
    """从 materials 字典的一项构造介质材料（名字取自字典的键）。"""
    where = "materials.{}".format(name)
    m_cb = float(_require(raw, "m_cb_me", where))
    return DielectricMaterial(
        name=name,
        kappa=float(_require(raw, "kappa", where)),
        eps_opt=float(_require(raw, "eps_opt", where)),
        band_gap=float(_require(raw, "band_gap_eV", where)) * EV,
        electron_affinity=float(raw.get("electron_affinity_eV", 0.0)) * EV,
        m_cb=m_cb * M_E,
        m_vb=float(_require(raw, "m_vb_me", where)) * M_E,
        # m_di 当前数值上等于 m_cb，但保持独立配置项：缺省才回落到 m_cb
        m_di=float(raw.get("m_di_me", m_cb)) * M_E,
        phonon_energy=float(_require(raw, "phonon_energy_eV", where)) * EV,
        huang_rhys=float(_require(raw, "huang_rhys_S", where)),
        attempt_frequency=float(raw.get("attempt_frequency_s", 1.0e13)),
    )


def _materials(raw: Any) -> Dict[str, DielectricMaterial]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("materials 必须是非空的「材料名 -> 参数」映射")
    return {str(name): _material(str(name), block) for name, block in raw.items()}


def _level(raw: Dict[str, Any], where: str) -> DefectLevelConfig:
    """缺陷能级分布（体缺陷与界面缺陷共用）。"""
    mode = str(raw.get("mode", "fixed"))
    if mode not in ("fixed", "gaussian"):
        raise ValueError("{}.mode 只能是 fixed 或 gaussian，收到 {!r}".format(where, mode))
    level = DefectLevelConfig(
        mode=mode,
        ED=float(_require(raw, "ED_eV", where)) * EV,
        sigma=float(raw.get("sigma_eV", 0.0)) * EV,
    )
    if mode == "gaussian" and level.sigma <= 0.0:
        raise ValueError("{}.mode=gaussian 时必须给出正的 sigma_eV".format(where))
    return level


def _defects(raw: Dict[str, Any], where: str) -> DefectConfig:
    """一层的体缺陷配置。"""
    level = _level(_require(raw, "level", where), where + ".level")
    spatial_raw = raw.get("spatial", {"mode": "uniform"})
    smode = str(spatial_raw.get("mode", "uniform"))
    if smode not in ("uniform", "interface"):
        raise ValueError("{}.spatial.mode 只能是 uniform 或 interface，收到 {!r}".format(where, smode))
    spatial = DefectSpatialConfig(
        mode=smode,
        enrichment=float(spatial_raw.get("enrichment", 1.0)),
        decay_length=float(spatial_raw.get("decay_length_nm", 0.0)) * NM,
    )
    if smode == "interface" and (spatial.enrichment <= 0.0 or spatial.decay_length <= 0.0):
        raise ValueError("{}.spatial.mode=interface 时 enrichment 与 decay_length_nm 必须为正".format(where))

    return DefectConfig(
        concentration=float(_require(raw, "concentration_cm3", where)) * PER_CM3,
        level=level,
        spatial=spatial,
        charge_state=float(raw.get("charge_state_Z", 1.0)),
        min_separation=float(raw.get("min_separation_nm", 0.0)) * NM,
    )


def _interface_defects(raw: Dict[str, Any], where: str) -> InterfaceDefectConfig:
    """界面缺陷配置（面密度 Nₛ 的独立物种）。Phase 1 只解析存储，不生成。"""
    return InterfaceDefectConfig(
        surface_density=float(_require(raw, "surface_density_cm2", where)) * PER_CM2,
        level=_level(_require(raw, "level", where), where + ".level"),
        charge_state=float(raw.get("charge_state_Z", 1.0)),
    )


def _layer(raw: Dict[str, Any], index: int, materials: Dict[str, DielectricMaterial]) -> Layer:
    where = "layers[{}]".format(index)
    material = str(_require(raw, "material", where))
    if material not in materials:
        raise ValueError(
            "{}.material={!r} 不在 materials 里（可用材料：{}）".format(where, material, sorted(materials))
        )
    thickness = float(_require(raw, "thickness_nm", where)) * NM
    if thickness <= 0.0:
        raise ValueError(
            "{}.thickness_nm 必须为正（要表达零厚层就删除该层，t=0 的单层退化即"
            "只保留一个 layer，见 examples/single_run.yaml）".format(where)
        )
    defects_raw = raw.get("defects")
    defects = _defects(defects_raw, where + ".defects") if defects_raw is not None else None
    return Layer(
        layer_id=str(raw.get("id", "L{}".format(index))),
        material=material,
        thickness=thickness,
        defects=defects,
    )


def _interface(raw: Dict[str, Any], index: int) -> Interface:
    where = "interfaces[{}]".format(index)
    delta = raw.get("delta_Ec_eV")
    defects_raw = raw.get("defects")
    return Interface(
        delta_Ec=None if delta is None else float(delta) * EV,
        sigma=float(raw.get("sigma_C_per_cm2", 0.0)) * PER_CM2,
        defects=_interface_defects(defects_raw, where + ".defects") if defects_raw is not None else None,
    )


def _stack(materials_raw: Any, layers_raw: Any, interfaces_raw: Optional[Any]) -> DielectricStack:
    materials = _materials(materials_raw)
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ValueError("layers 必须是非空的有序层列表")
    layers = tuple(_layer(layer, i, materials) for i, layer in enumerate(layers_raw))

    expected = max(0, len(layers) - 1)
    interfaces_raw = interfaces_raw or []
    if not isinstance(interfaces_raw, list):
        raise ValueError("interfaces 必须是列表")
    if len(interfaces_raw) != expected:
        raise ValueError(
            "interfaces 数量必须等于 层数-1 = {}（收到 {}）；单层不配 interfaces".format(
                expected, len(interfaces_raw)
            )
        )
    interfaces = tuple(_interface(item, i) for i, item in enumerate(interfaces_raw))
    return DielectricStack(materials=materials, layers=layers, interfaces=interfaces)


def _device(raw: Dict[str, Any], electrodes: Dict[str, Any], stack: DielectricStack) -> DeviceConfig:
    if "Lz_nm" in raw:
        raise ValueError(
            "device.Lz_nm 属于旧 schema：Lz 现在由各层 thickness_nm 求和得到，请删除 "
            "device.Lz_nm，改在 layers 里给各层 thickness_nm（见 examples/single_run.yaml）"
        )
    barrier = raw.get("barrier_height_eV")
    return DeviceConfig(
        Lx=float(_require(raw, "Lx_nm", "device")) * NM,
        Ly=float(_require(raw, "Ly_nm", "device")) * NM,
        cathode=_electrode(_require(electrodes, "cathode", "electrodes"), "electrodes.cathode"),
        anode=_electrode(_require(electrodes, "anode", "electrodes"), "electrodes.anode"),
        stack=stack,
        barrier_height=None if barrier is None else float(barrier) * EV,
    )


def _simulation(raw: Dict[str, Any]) -> SimulationConfig:
    grid_raw = raw.get("grid", {})
    fast_rate = raw.get("fast_trap_rate_s")
    if fast_rate is not None and not float(fast_rate) > 0.0:
        raise ValueError("simulation.fast_trap_rate_s 必须为正（单位 s⁻¹），收到 {!r}".format(fast_rate))
    return SimulationConfig(
        voltage=float(_require(raw, "voltage_V", "simulation")),
        temperature=celsius_to_kelvin(_require(raw, "temperature_C", "simulation")),
        n_events=int(raw.get("n_events", 20000)),
        burn_in_fraction=float(raw.get("burn_in_fraction", 0.2)),
        seed=int(raw.get("seed", 12345)),
        grid=GridConfig(
            nx=int(grid_raw.get("nx", 32)),
            ny=int(grid_raw.get("ny", 32)),
            nz=int(grid_raw.get("nz", 32)),
        ),
        fast_trap_rate=None if fast_rate is None else float(fast_rate),
    )


def _range(raw: Optional[Dict[str, Any]], default: Dict[str, float]) -> np.ndarray:
    """把 {start, stop, step} 展开成含端点的数组。"""
    spec = default if raw is None else raw
    start = float(spec["start"])
    stop = float(spec["stop"])
    step = float(spec["step"])
    if step <= 0.0:
        raise ValueError("扫描 step 必须为正")
    n = int(round((stop - start) / step)) + 1
    return start + step * np.arange(n)


def _sweep(raw: Optional[Dict[str, Any]]) -> SweepConfig:
    """扫描范围；缺省用 docs/physics_model.md 的默认值。"""
    raw = raw or {}
    voltages = _range(raw.get("voltage_V"), {"start": 1.0, "stop": 4.0, "step": 0.2})
    temps_c = _range(raw.get("temperature_C"), {"start": 0.0, "stop": 125.0, "step": 25.0})
    return SweepConfig(
        voltages=voltages,
        temperatures=np.array([celsius_to_kelvin(t) for t in temps_c]),
    )


def _draw_seeds(count: int) -> tuple:
    """从系统熵源随机抽取 count 个互异的种子。

    不带种子的 `default_rng()` 直接取 OS 熵，所以每次运行都是新的一批。
    2^31 的取值空间里抽 5 个碰撞概率可忽略，仍去重兜底以满足"互异"约束。
    """
    rng = np.random.default_rng()
    seeds: list = []
    while len(seeds) < count:
        seed = int(rng.integers(0, 2 ** 31 - 1))
        if seed not in seeds:
            seeds.append(seed)
    return tuple(seeds)


def _ensemble(raw: Optional[Dict[str, Any]]) -> Optional[EnsembleConfig]:
    """系综实现个数 M；整段省略返回 None，即退化为 simulation.seed 一个实现。

    配置只给 `count`（实现个数），每次运行从系统熵源随机抽取 count 个互异
    的种子——不再用写死的种子列表。抽取在 `load_config` 里发生一次，运行内
    稳定，需要复现某次结果时照抄回 simulation.seed / 单独指定即可。
    """
    if raw is None:
        return None
    count = _require(raw, "count", "ensemble")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("ensemble.count 必须是 >= 1 的整数")
    return EnsembleConfig(seeds=_draw_seeds(count))


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} 必须是 >= 1 的整数，收到 {!r}".format(where, value))
    return value


def _tddb(raw: Optional[Dict[str, Any]]) -> Optional[TDDBConfig]:
    """TDDB 缺陷生成参数（ADR-024/028/033/036）；整段省略返回 None，即纯漏电流仿真。

    三个能量（E_A、E_t1、E_rel1）必须为正；p₀ 必须 ≥ 0，取 0 即关掉场加速；N_T 必须 ≥ 0，
    取 0 即关掉生成（G = 0）。生成网格三个方向的单元数、空位数上限都是正整数，应力时间
    上限、渗流截断 d_c、限流值必须为正。热导率 k_TH 可省略（等温），给出时必须为正。
    """
    if raw is None:
        return None

    def positive(key: str, unit: str) -> float:
        value = float(_require(raw, key, "tddb"))
        if not value > 0.0:
            raise ValueError("tddb.{} 必须为正（单位 {}），收到 {!r}".format(key, unit, value))
        return value

    def energy(key: str) -> float:
        return positive(key, "eV") * EV

    dipole = float(_require(raw, "dipole_moment_eA", "tddb"))
    if not dipole >= 0.0:
        raise ValueError("tddb.dipole_moment_eA 必须 ≥ 0（单位 e·Å），收到 {!r}".format(dipole))
    concentration = float(_require(raw, "precursor_concentration_cm3", "tddb"))
    if not concentration >= 0.0:
        raise ValueError(
            "tddb.precursor_concentration_cm3 必须 ≥ 0（单位 cm⁻³），收到 {!r}".format(concentration)
        )
    grid = _require(raw, "generation_grid", "tddb")
    thermal_conductivity = None
    if raw.get("thermal_conductivity_W_mK") is not None:
        thermal_conductivity = positive("thermal_conductivity_W_mK", "W/(m·K)")
    return TDDBConfig(
        activation_energy=energy("activation_energy_eV"),
        precursor_ionization_energy=energy("precursor_ionization_energy_eV"),
        precursor_relaxation_energy=energy("precursor_relaxation_energy_eV"),
        dipole_moment=dipole * E_ANGSTROM,
        precursor_concentration=concentration * PER_CM3,
        generation_grid=tuple(
            _positive_int(_require(grid, axis, "tddb.generation_grid"), "tddb.generation_grid." + axis)
            for axis in ("nx", "ny", "nz")
        ),
        max_generated_vacancies=_positive_int(
            _require(raw, "max_generated_vacancies", "tddb"), "tddb.max_generated_vacancies"
        ),
        stop_time=positive("stop_time_s", "s"),
        percolation_distance=positive("percolation_distance_nm", "nm") * NM,
        compliance_current_density=positive("compliance_current_density_A_cm2", "A/cm²") * PER_CM2,
        thermal_conductivity=thermal_conductivity,
    )


_LEGACY_TOP_KEYS = {
    "dielectric": (
        "顶层 dielectric 属于旧 schema：介质材料改配在 materials（按名字典）里，各层在 "
        "layers 里按名引用（layers[i].material），见 examples/single_run.yaml"
    ),
    "defects": (
        "顶层 defects 属于旧 schema：体缺陷改配在各层下（layers[i].defects），界面缺陷配在 "
        "interfaces[i].defects，见 examples/single_run.yaml"
    ),
}


def _reject_legacy_keys(raw: Dict[str, Any]) -> None:
    """旧 schema 的顶层键直接报错并提示新写法（不做运行时兼容层，ADR-015）。"""
    for key, hint in _LEGACY_TOP_KEYS.items():
        if key in raw:
            raise ValueError("检测到旧 schema 的顶层键 {!r}。{}".format(key, hint))


def config_from_dict(raw: Dict[str, Any]) -> Config:
    """从已解析的字典构造 Config（单位换算在此完成）。"""
    _reject_legacy_keys(raw)
    stack = _stack(
        _require(raw, "materials", "<root>"),
        _require(raw, "layers", "<root>"),
        raw.get("interfaces"),
    )
    device = _device(
        _require(raw, "device", "<root>"),
        _require(raw, "electrodes", "<root>"),
        stack,
    )
    return Config(
        device=device,
        simulation=_simulation(_require(raw, "simulation", "<root>")),
        sweep=_sweep(raw.get("sweep")),
        ensemble=_ensemble(raw.get("ensemble")),
        tddb=_tddb(raw.get("tddb")),
    )


def load_config(path: str) -> Config:
    """读取 YAML 配置文件并换算为 SI 单位的 Config。"""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("配置文件 {} 的顶层必须是映射".format(path))
    return config_from_dict(raw)
