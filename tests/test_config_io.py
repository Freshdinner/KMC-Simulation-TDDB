"""配置读入与单位换算的检验。

单位换算只发生在 io.py 的读入边界，这里逐项核对换算结果，并覆盖新 schema
（materials + layers + interfaces）的结构校验与旧 schema 顶层键的拒绝。
"""

import dataclasses
import pathlib
import textwrap

import numpy as np
import pytest
from scipy import constants

from kmc_leakage.defects import defect_count
from kmc_leakage.io import celsius_to_kelvin, config_from_dict, load_config

EV = constants.electron_volt
M_E = constants.electron_mass

_YAML = textwrap.dedent(
    """
    materials:
      ref:
        kappa: 38.0
        eps_opt: 5.6
        band_gap_eV: 5.4
        electron_affinity_eV: 2.86
        m_cb_me: 1.2
        m_vb_me: 2.5
        m_di_me: 1.2
        phonon_energy_eV: 0.04
        huang_rhys_S: 1.0
        attempt_frequency_s: 1.0e13
    layers:
      - material: ref
        thickness_nm: 8.0
        defects:
          concentration_cm3: 3.0e18
          charge_state_Z: 1.0
          min_separation_nm: 0.5
          level:
            mode: fixed
            ED_eV: 1.15
          spatial:
            mode: uniform
    device:
      Lx_nm: 50.0
      Ly_nm: 50.0
      barrier_height_eV: 1.74
    electrodes:
      cathode: &e
        name: TiN
        work_function_eV: 4.6
        effective_mass_me: 1.0
        fermi_energy_eV: 5.0
      anode: *e
    simulation:
      voltage_V: 2.0
      temperature_C: 25.0
      n_events: 100
      seed: 1
      grid: {nx: 8, ny: 8, nz: 8}
    """
)


def _base_dict():
    import yaml

    return yaml.safe_load(_YAML)


def test_units_are_converted_to_si():
    cfg = config_from_dict(_base_dict())
    material = cfg.device.stack.single_material
    assert cfg.device.Lx == pytest.approx(50e-9)
    assert cfg.device.Lz == pytest.approx(8e-9)  # Lz 由层厚求和
    assert cfg.device.barrier_height == pytest.approx(1.74 * EV)
    assert material.band_gap == pytest.approx(5.4 * EV)
    assert material.phonon_energy == pytest.approx(0.04 * EV)
    assert cfg.device.cathode.work_function == pytest.approx(4.6 * EV)
    assert cfg.device.cathode.fermi_energy == pytest.approx(5.0 * EV)
    assert material.m_cb == pytest.approx(1.2 * M_E)
    assert material.m_di == pytest.approx(1.2 * M_E)
    assert cfg.defects.concentration == pytest.approx(3.0e24)  # cm^-3 -> m^-3
    assert cfg.defects.level.ED == pytest.approx(1.15 * EV)
    assert cfg.defects.min_separation == pytest.approx(0.5e-9)
    assert cfg.simulation.temperature == pytest.approx(298.15)


def test_fast_trap_rate_is_optional_and_positive():
    """simulation.fast_trap_rate_s（s⁻¹）缺省为 None（不移出快陷阱），给出时须为正（ADR-023）。"""
    raw = _base_dict()
    assert config_from_dict(raw).simulation.fast_trap_rate is None
    raw["simulation"]["fast_trap_rate_s"] = 1.0e9
    assert config_from_dict(raw).simulation.fast_trap_rate == pytest.approx(1.0e9)
    raw["simulation"]["fast_trap_rate_s"] = 0.0
    with pytest.raises(ValueError, match="fast_trap_rate_s"):
        config_from_dict(raw)


def test_tddb_section_is_optional_and_converted():
    """tddb 段缺省为 None（纯漏电流仿真）；三个能量从 eV 换算成 J、必须为正，p₀ 从 e·Å
    换算成 C·m、必须 ≥ 0，P_bb 宏观场的平均窗口从 nm 换算成 m、必须 ≥ 0（ADR-043），
    N_T 从 cm⁻³ 换算成 m⁻³、必须 ≥ 0，空位数上限是正整数，
    应力时间上限必须为正，导电通路的连通距离 r_c 从 nm 换算成 m、必须为正，以上各项都必须给出；
    k_TH 可省略（等温），给出时必须为正（ADR-024/028/033/036/054）。"""
    raw = _base_dict()
    assert config_from_dict(raw).tddb is None
    section = {
        "activation_energy_eV": 4.0,
        "precursor_ionization_energy_eV": 1.15,
        "precursor_relaxation_energy_eV": 0.04,
        "dipole_moment_eA": 7.1,
        "field_averaging_width_nm": 0.5,
        "precursor_concentration_cm3": 4.0e19,
        "max_generated_vacancies": 300,
        "stop_time_s": 1.0e4,
        "percolation_distance_nm": 1.0,
    }
    raw["tddb"] = dict(section)
    tddb = config_from_dict(raw).tddb
    assert tddb.activation_energy == pytest.approx(4.0 * EV)
    assert tddb.precursor_ionization_energy == pytest.approx(1.15 * EV)
    assert tddb.precursor_relaxation_energy == pytest.approx(0.04 * EV)
    assert tddb.dipole_moment == pytest.approx(7.1 * constants.elementary_charge * 1e-10)
    assert tddb.field_averaging_width == pytest.approx(0.5e-9)
    assert tddb.precursor_concentration == pytest.approx(4.0e25)
    assert tddb.max_generated_vacancies == 300
    assert tddb.stop_time == pytest.approx(1.0e4)
    assert tddb.percolation_distance == pytest.approx(1.0e-9)
    assert tddb.thermal_conductivity is None  # 省略 k_TH：等温
    raw["tddb"] = dict(section, thermal_conductivity_W_mK=1.5)
    assert config_from_dict(raw).tddb.thermal_conductivity == 1.5
    for bad in (0.0, -1.0):
        raw["tddb"] = dict(section, thermal_conductivity_W_mK=bad)
        with pytest.raises(ValueError, match="thermal_conductivity_W_mK"):
            config_from_dict(raw)
    for key in ("activation_energy_eV", "precursor_ionization_energy_eV", "precursor_relaxation_energy_eV"):
        raw["tddb"] = dict(section, **{key: 0.0})
        with pytest.raises(ValueError, match=key):
            config_from_dict(raw)
    raw["tddb"] = dict(section, dipole_moment_eA=0.0)
    assert config_from_dict(raw).tddb.dipole_moment == 0.0  # p₀ = 0：关掉场加速
    raw["tddb"] = dict(section, field_averaging_width_nm=0.0)
    assert config_from_dict(raw).tddb.field_averaging_width == 0.0  # 0：P_bb 用局域场
    raw["tddb"] = dict(section, precursor_concentration_cm3=0.0)
    assert config_from_dict(raw).tddb.precursor_concentration == 0.0  # N_T = 0：关掉生成
    for key, bad in (
        ("dipole_moment_eA", -1.0),
        ("field_averaging_width_nm", -0.1),
        ("precursor_concentration_cm3", -1.0),
        ("max_generated_vacancies", 0),
        ("max_generated_vacancies", 2.5),
        ("stop_time_s", 0.0),
        ("percolation_distance_nm", 0.0),
        ("percolation_distance_nm", -1.0),
    ):
        raw["tddb"] = dict(section, **{key: bad})
        with pytest.raises(ValueError, match=key):
            config_from_dict(raw)
    for key in section:
        raw["tddb"] = {k: v for k, v in section.items() if k != key}
        with pytest.raises(KeyError, match=key):
            config_from_dict(raw)


def test_lz_is_the_sum_of_layer_thicknesses():
    """Lz 不是配置项，由各层 thickness_nm 求和得到。"""
    raw = _base_dict()
    raw["layers"][0]["thickness_nm"] = 6.0
    cfg = config_from_dict(raw)
    assert cfg.device.Lz == pytest.approx(6e-9)
    assert cfg.device.stack.Lz == pytest.approx(6e-9)


def test_effective_masses_stay_independent_fields():
    """m_di 缺省时回落到 m_cb，但显式给出时必须独立生效。"""
    raw = _base_dict()
    raw["materials"]["ref"]["m_di_me"] = 0.5
    cfg = config_from_dict(raw)
    assert cfg.device.stack.single_material.m_di == pytest.approx(0.5 * M_E)
    assert cfg.device.stack.single_material.m_cb == pytest.approx(1.2 * M_E)

    del raw["materials"]["ref"]["m_di_me"]
    cfg = config_from_dict(raw)
    material = cfg.device.stack.single_material
    assert material.m_di == pytest.approx(material.m_cb)


def test_barrier_height_explicit_takes_priority_over_alignment():
    raw = _base_dict()
    cfg = config_from_dict(raw)
    assert cfg.device.barrier(cfg.device.cathode) == pytest.approx(1.74 * EV)
    assert cfg.device.built_in_potential() == 0.0

    del raw["device"]["barrier_height_eV"]
    cfg = config_from_dict(raw)
    # 推算值 E_B = Φ_m - χ = 4.6 - 2.86 = 1.74 eV（χ 取自单层材料）
    assert cfg.device.barrier(cfg.device.cathode) == pytest.approx(1.74 * EV)


def test_asymmetric_electrodes_give_built_in_potential():
    raw = _base_dict()
    del raw["device"]["barrier_height_eV"]
    raw["electrodes"] = {
        "cathode": {"name": "TiN", "work_function_eV": 4.6, "effective_mass_me": 1.0, "fermi_energy_eV": 5.0},
        "anode": {"name": "Pt", "work_function_eV": 5.3, "effective_mass_me": 1.0, "fermi_energy_eV": 5.0},
    }
    cfg = config_from_dict(raw)
    assert cfg.device.barrier(cfg.device.cathode) == pytest.approx(1.74 * EV)
    assert cfg.device.barrier(cfg.device.anode) == pytest.approx(2.44 * EV)
    assert cfg.device.built_in_potential() == pytest.approx(-0.7)


def test_sweep_defaults_match_spec():
    """默认扫描范围：I-V 1–4 V 步长 0.2 V；I-T 0–125 °C 步长 25 °C。"""
    cfg = config_from_dict(_base_dict())
    assert cfg.sweep.voltages.size == 16
    assert cfg.sweep.voltages[0] == pytest.approx(1.0)
    assert cfg.sweep.voltages[-1] == pytest.approx(4.0)
    assert cfg.sweep.temperatures.size == 6
    expected = [celsius_to_kelvin(t) for t in (0, 25, 50, 75, 100, 125)]
    assert np.allclose(cfg.sweep.temperatures, expected)


def test_sweep_range_can_be_overridden():
    raw = _base_dict()
    raw["sweep"] = {
        "voltage_V": {"start": 0.5, "stop": 1.5, "step": 0.5},
        "temperature_C": {"start": 25.0, "stop": 25.0, "step": 25.0},
    }
    cfg = config_from_dict(raw)
    assert np.allclose(cfg.sweep.voltages, [0.5, 1.0, 1.5])
    assert cfg.sweep.temperatures.size == 1


def test_invalid_modes_are_rejected():
    raw = _base_dict()
    raw["layers"][0]["defects"]["level"]["mode"] = "lorentzian"
    with pytest.raises(ValueError):
        config_from_dict(raw)

    raw = _base_dict()
    raw["layers"][0]["defects"]["level"] = {"mode": "gaussian", "ED_eV": 1.15}
    with pytest.raises(ValueError):
        config_from_dict(raw)

    raw = _base_dict()
    raw["layers"][0]["defects"]["spatial"] = {"mode": "interface", "enrichment": 5.0}
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_ensemble_count_draws_distinct_seeds():
    raw = _base_dict()
    raw["ensemble"] = {"count": 3}
    cfg = config_from_dict(raw)
    assert len(cfg.ensemble.seeds) == 3
    assert len(set(cfg.ensemble.seeds)) == 3  # 互异
    assert all(isinstance(s, int) for s in cfg.ensemble.seeds)
    assert cfg.seeds == cfg.ensemble.seeds


def test_ensemble_seeds_are_random_across_loads():
    """同一份配置两次加载应抽到不同的种子——每次运行都是新的一批。"""
    raw = _base_dict()
    raw["ensemble"] = {"count": 5}
    assert config_from_dict(raw).seeds != config_from_dict(raw).seeds


def test_without_ensemble_section_seeds_fall_back_to_simulation_seed():
    """没有 ensemble 段就是单实现，与加系综之前的行为一致。"""
    cfg = config_from_dict(_base_dict())
    assert cfg.ensemble is None
    assert cfg.seeds == (cfg.simulation.seed,)


def test_bad_ensemble_count_is_rejected():
    for count in (0, -1, "5", 2.5, [5], True):
        raw = _base_dict()
        raw["ensemble"] = {"count": count}
        with pytest.raises(ValueError):
            config_from_dict(raw)


def test_missing_required_key_is_reported():
    raw = _base_dict()
    del raw["materials"]["ref"]["eps_opt"]
    with pytest.raises(KeyError):
        config_from_dict(raw)


# --------------------------------------------------------------------------
# 新 schema 的结构校验（ADR-015 / review F11）
# --------------------------------------------------------------------------
def test_legacy_top_level_dielectric_is_rejected():
    """旧 schema 的顶层 dielectric 键直接报错并提示新写法（不做运行时兼容层）。"""
    raw = _base_dict()
    raw["dielectric"] = {"kappa": 38.0}
    with pytest.raises(ValueError) as exc:
        config_from_dict(raw)
    message = str(exc.value)
    assert "dielectric" in message
    assert "materials" in message and "layers" in message


def test_legacy_top_level_defects_is_rejected():
    """旧 schema 的顶层 defects 键（体缺陷现配在 layers 下）直接报错。"""
    raw = _base_dict()
    raw["defects"] = {"concentration_cm3": 3.0e18}
    with pytest.raises(ValueError) as exc:
        config_from_dict(raw)
    assert "layers" in str(exc.value)


def test_legacy_device_lz_is_rejected():
    """device.Lz_nm 属于旧 schema：Lz 现由层厚求和，给了就报错提示新写法。"""
    raw = _base_dict()
    raw["device"]["Lz_nm"] = 8.0
    with pytest.raises(ValueError) as exc:
        config_from_dict(raw)
    assert "Lz" in str(exc.value) and "thickness_nm" in str(exc.value)


def test_unknown_material_reference_is_rejected():
    raw = _base_dict()
    raw["layers"][0]["material"] = "does-not-exist"
    with pytest.raises(ValueError) as exc:
        config_from_dict(raw)
    assert "does-not-exist" in str(exc.value)


def test_nonpositive_layer_thickness_is_rejected():
    raw = _base_dict()
    raw["layers"][0]["thickness_nm"] = 0.0
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_single_layer_forbids_interfaces():
    """单层没有内部界面：interfaces 数量必须等于 层数-1 = 0。"""
    raw = _base_dict()
    raw["interfaces"] = [{"delta_Ec_eV": 1.0}]
    with pytest.raises(ValueError) as exc:
        config_from_dict(raw)
    assert "interfaces" in str(exc.value)


def test_example_configs_load():
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("single_run.yaml", "iv_sweep.yaml"):
        cfg = load_config(str(root / "examples" / name))
        assert cfg.device.Lz > 0.0
        assert cfg.defects.concentration > 0.0
        assert cfg.sweep is not None
        # 两个示例都按系综平均配置：5 个实现，50×50×8 nm 且格点与尺寸配套
        assert len(cfg.seeds) == 5
        assert cfg.device.Lx == pytest.approx(50e-9)
        assert cfg.device.Ly == pytest.approx(50e-9)
        assert cfg.device.Lz == pytest.approx(8e-9)
        # 单层退化：只含一个 layer、无内部界面
        assert len(cfg.device.stack.layers) == 1
        assert len(cfg.device.stack.interfaces) == 0
        assert cfg.device.Lx / cfg.simulation.grid.nx == pytest.approx(1.5625e-9)
        assert cfg.device.Ly / cfg.simulation.grid.ny == pytest.approx(1.5625e-9)
        assert cfg.device.Lz / cfg.simulation.grid.nz == pytest.approx(0.1e-9)
        # 缺陷段两个示例同构：高斯能级与界面富集的参数都预置好，改 mode 即可启用
        assert cfg.defects.level.sigma > 0.0
        assert cfg.defects.spatial.enrichment > 1.0
        assert cfg.defects.spatial.decay_length > 0.0


def test_tddb_example_matches_leakage_reference_preset():
    """zro2_tddb.yaml 的材料、缺陷、电极与默认盒子 single_run.yaml 逐项相同（材料名之外），
    盒子同为 50×50 nm（ADR-051）；横向网格按团簇库仑能收敛取 hx = hy = 0.25 nm、hz 不变
    （ADR-032）；前驱的 E_t1 与 O 空位预设相同（ADR-028）；E_rel1 与材料的 huang_rhys 各自取值
    （ADR-030）；p₀ = 1 eÅ（ADR-035），E_A = 1.30 eV 全部前驱相同（ADR-053）、N_T = 10²⁰ cm⁻³（ADR-052），
    k_TH = 2 W/(m·K)（ADR-040），击穿电流密度 10 A/cm²（ADR-047）。"""
    root = pathlib.Path(__file__).resolve().parents[1]
    tddb = load_config(str(root / "examples" / "zro2_tddb.yaml"))
    ref = load_config(str(root / "examples" / "single_run.yaml"))

    material = tddb.device.stack.single_material
    ref_material = ref.device.stack.single_material
    assert material.name == "ZrO2"
    assert dataclasses.replace(material, name=ref_material.name) == ref_material
    assert tddb.defects == ref.defects
    assert tddb.device.cathode == ref.device.cathode
    assert tddb.device.anode == ref.device.anode
    assert tddb.device.barrier_height == ref.device.barrier_height
    assert tddb.device.Lz == ref.device.Lz
    assert len(tddb.device.stack.layers) == 1

    assert tddb.device.Lx == pytest.approx(50e-9)
    assert tddb.device.Ly == pytest.approx(50e-9)
    assert defect_count(tddb.device, tddb.defects) == 60
    grid, ref_grid = tddb.simulation.grid, ref.simulation.grid
    assert tddb.device.Lx / grid.nx == pytest.approx(0.25e-9)
    assert tddb.device.Ly / grid.ny == pytest.approx(0.25e-9)
    assert grid.nz == ref_grid.nz
    assert tddb.simulation.n_events == ref.simulation.n_events

    assert len(tddb.seeds) == 30  # 系综个数（ADR-026、ADR-051）
    assert tddb.tddb.activation_energy == pytest.approx(1.30 * EV)
    assert tddb.tddb.precursor_ionization_energy == pytest.approx(tddb.defects.level.ED)
    assert tddb.tddb.precursor_relaxation_energy == pytest.approx(1.0 * EV)
    assert tddb.tddb.dipole_moment == pytest.approx(1.0 * constants.elementary_charge * 1e-10)
    assert tddb.tddb.field_averaging_width == pytest.approx(0.5e-9)  # ADR-043
    assert tddb.tddb.precursor_concentration == pytest.approx(1.0e20 * constants.centi ** -3)
    assert tddb.tddb.thermal_conductivity == pytest.approx(2.0)
    assert tddb.tddb.percolation_distance == pytest.approx(1.0e-9)  # r_c（ADR-054）
    assert tddb.tddb.max_generated_vacancies >= 1 and tddb.tddb.stop_time > 0.0
    assert ref.tddb is None
