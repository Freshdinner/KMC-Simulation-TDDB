"""DielectricStack 统一查询接口与单层退化用例（ADR-003/015/018、review F11）。

Phase 1 的验收核心：
- 单元素 layers（走 YAML 解析）与直接构造的单层叠层给出逐位相同的确定性量
  与同种子轨迹（退化用例，ADR-018）；
- 多材料叠层能被 schema 完整解析，但运行时明确抛错（分层物理属 Phase 3/4）；
- 统一查询接口 single_material / bulk_defects / Lz 的行为。
"""

import numpy as np
import pytest
from scipy import constants

from conftest import make_defect_config, make_material
from kmc_leakage.config import (
    Config,
    DefectLevelConfig,
    DeviceConfig,
    DielectricStack,
    ElectrodeMaterial,
    GridConfig,
    Interface,
    Layer,
    SimulationConfig,
)
from kmc_leakage.io import config_from_dict
from kmc_leakage.kmc import MECHANISMS, KMCSimulation

EV = constants.electron_volt
M_E = constants.electron_mass


# --------------------------------------------------------------------------
# DielectricStack 统一查询接口
# --------------------------------------------------------------------------
def test_lz_is_sum_of_layer_thickness():
    a, b = make_material("A"), make_material("B", kappa=9.0)
    stack = DielectricStack(
        materials={"A": a, "B": b},
        layers=(Layer("L0", "A", 4e-9), Layer("L1", "B", 1e-9), Layer("L2", "A", 4e-9)),
        interfaces=(Interface(), Interface()),
    )
    assert stack.Lz == pytest.approx(9e-9)


def test_single_material_resolves_when_homogeneous():
    material = make_material()
    stack = DielectricStack.single(material, 9e-9, make_defect_config())
    assert stack.single_material is material
    assert stack.material_of(stack.layers[0]) is material


def test_single_material_raises_on_multiple_materials():
    """多材料叠层的分层物理属 Phase 3/4：single_material 抛 NotImplementedError。"""
    a, b = make_material("A"), make_material("B", kappa=9.0)
    stack = DielectricStack(
        materials={"A": a, "B": b},
        layers=(Layer("L0", "A", 4e-9), Layer("L1", "B", 1e-9)),
        interfaces=(Interface(),),
    )
    with pytest.raises(NotImplementedError):
        stack.single_material


def test_bulk_defects_resolves_single_layer_and_flags_multilayer():
    material = make_material()
    defects = make_defect_config()
    stack = DielectricStack.single(material, 9e-9, defects)
    assert stack.bulk_defects is defects

    # 多层都带体缺陷 -> 分层生成属 Phase 2，抛 NotImplementedError
    two = DielectricStack(
        materials={"A": material},
        layers=(Layer("L0", "A", 4e-9, defects), Layer("L1", "A", 5e-9, defects)),
        interfaces=(Interface(),),
    )
    with pytest.raises(NotImplementedError):
        two.bulk_defects


# --------------------------------------------------------------------------
# 构造新 schema 字典的小工具
# --------------------------------------------------------------------------
_MATERIAL_YAML = {
    "kappa": 38.0, "eps_opt": 5.6, "band_gap_eV": 5.4, "electron_affinity_eV": 2.86,
    "m_cb_me": 1.2, "m_vb_me": 2.5, "m_di_me": 1.2, "phonon_energy_eV": 0.04,
    "huang_rhys_S": 1.0, "attempt_frequency_s": 1.0e13,
}
_DEFECTS_YAML = {
    "concentration_cm3": 3.0e18, "charge_state_Z": 1.0, "min_separation_nm": 0.5,
    "level": {"mode": "fixed", "ED_eV": 1.15}, "spatial": {"mode": "uniform"},
}
_ELECTRODE_YAML = {"name": "TiN", "work_function_eV": 4.6, "effective_mass_me": 1.0, "fermi_energy_eV": 5.0}


def _single_layer_dict():
    return {
        "materials": {"ref": dict(_MATERIAL_YAML)},
        "layers": [{"material": "ref", "thickness_nm": 8.0, "defects": dict(_DEFECTS_YAML)}],
        "device": {"Lx_nm": 20.0, "Ly_nm": 20.0, "barrier_height_eV": 1.74},
        "electrodes": {"cathode": dict(_ELECTRODE_YAML), "anode": dict(_ELECTRODE_YAML)},
        "simulation": {
            "voltage_V": 3.0, "temperature_C": 25.0, "n_events": 300, "seed": 7,
            "grid": {"nx": 8, "ny": 8, "nz": 16},
        },
    }


# --------------------------------------------------------------------------
# 单元素 layers 退化用例（ADR-018）
# --------------------------------------------------------------------------
def test_single_element_layers_matches_direct_single_layer():
    """单元素 layers（走 YAML 解析）与直接构造的单层叠层逐位相同。

    确定性量（Lz、材料参数、R1 两分量 rtol ≤ 1e-12）与同种子完整轨迹
    （J、机制分解 rtol ≤ 1e-9）都对上——这是 Phase 1 的单层退化验收。
    """
    cfg_yaml = config_from_dict(_single_layer_dict())

    # 直接构造：同一份材料 / 缺陷 / 器件 / 网格 / 种子
    material = make_material("ref")
    defects = make_defect_config(
        concentration_cm3=3.0e18,
        level=DefectLevelConfig(mode="fixed", ED=1.15 * EV),
        min_separation=0.5e-9,
    )
    electrode = ElectrodeMaterial("TiN", 4.6 * EV, M_E, 5.0 * EV)
    device = DeviceConfig(
        20e-9, 20e-9, electrode, electrode,
        DielectricStack.single(material, 8e-9, defects), barrier_height=1.74 * EV,
    )
    cfg_direct = Config(
        device=device,
        simulation=SimulationConfig(
            voltage=3.0, temperature=298.15, n_events=300, seed=7, grid=GridConfig(8, 8, 16)
        ),
    )

    # 确定性量：Lz 与材料参数逐位相同
    assert cfg_yaml.device.Lz == cfg_direct.device.Lz
    assert cfg_yaml.device.stack.single_material == cfg_direct.device.stack.single_material

    a = KMCSimulation(cfg_yaml).run()
    b = KMCSimulation(cfg_direct).run()
    # R1 两分量与轨迹无关，rtol ≤ 1e-12
    for name in ("direct_fn", "thermionic"):
        assert a.by_mechanism[name] == pytest.approx(b.by_mechanism[name], rel=1e-12)
    # 同种子完整轨迹：J 与机制分解逐位相同，rtol ≤ 1e-9
    assert a.current == pytest.approx(b.current, rel=1e-9)
    for name in MECHANISMS:
        assert a.by_mechanism[name] == pytest.approx(b.by_mechanism[name], rel=1e-9, abs=0.0)


# --------------------------------------------------------------------------
# 多材料叠层：schema 能解析，运行时明确抛错（Phase 1 边界）
# --------------------------------------------------------------------------
def _zaz_dict():
    return {
        "materials": {
            "ZrO2": dict(_MATERIAL_YAML),
            "Al2O3": dict(
                _MATERIAL_YAML, kappa=9.0, eps_opt=3.0, band_gap_eV=6.4, electron_affinity_eV=1.4,
                m_cb_me=0.3, m_vb_me=0.4, m_di_me=0.3, phonon_energy_eV=0.09,
            ),
        },
        "layers": [
            {"material": "ZrO2", "thickness_nm": 4.0, "defects": dict(_DEFECTS_YAML)},
            {"material": "Al2O3", "thickness_nm": 1.0},
            {"material": "ZrO2", "thickness_nm": 4.0},
        ],
        "interfaces": [
            {
                "sigma_C_per_cm2": 0.0,
                "defects": {
                    "surface_density_cm2": 1.0e12,
                    "level": {"mode": "fixed", "ED_eV": 1.15},
                    "charge_state_Z": 1.0,
                },
            },
            {},
        ],
        "device": {"Lx_nm": 50.0, "Ly_nm": 50.0, "barrier_height_eV": 1.74},
        "electrodes": {"cathode": dict(_ELECTRODE_YAML), "anode": dict(_ELECTRODE_YAML)},
        "simulation": {
            "voltage_V": 3.0, "temperature_C": 25.0, "n_events": 10, "seed": 7,
            "grid": {"nx": 8, "ny": 8, "nz": 36},
        },
    }


def test_zaz_schema_parses_structure():
    """ZAZ 三层配置能被 schema 完整解析：材料、层、界面、界面缺陷、Lz 求和。"""
    cfg = config_from_dict(_zaz_dict())
    stack = cfg.device.stack
    assert set(stack.materials) == {"ZrO2", "Al2O3"}
    assert [layer.material for layer in stack.layers] == ["ZrO2", "Al2O3", "ZrO2"]
    assert cfg.device.Lz == pytest.approx(9e-9)  # 4 + 1 + 4
    assert len(stack.interfaces) == 2
    # 界面缺陷面密度已换算到 m^-2；空层（Al2O3 体相）无体缺陷
    assert stack.interfaces[0].defects.surface_density == pytest.approx(1.0e12 * constants.centi ** -2)
    assert stack.layers[1].defects is None
    # Al2O3 的 κ / ε_opt / χ 独立解析
    assert stack.materials["Al2O3"].kappa == pytest.approx(9.0)
    assert stack.materials["Al2O3"].eps_opt == pytest.approx(3.0)


# --------------------------------------------------------------------------
# 多材料叠层：Phase 4 起端到端跑通（分段 WKB / 按材料取速率参数 / 分段 G_ab）
# --------------------------------------------------------------------------
def test_zaz_run_end_to_end_and_decomposes():
    """ZAZ 叠层端到端跑通并给出机制分解（Phase 4 打通，翻转 Phase 1 的抛错断言）。

    Phase 1–3 时多材料叠层在 kmc._prepare 读 single_material 处抛
    NotImplementedError；Phase 4 分段 WKB + 按材料取速率参数 + 分段 G_ab 打通后，
    ZAZ 能出稳态电流与分解。这里只钉住结构自洽（各分量之和 = 总电流、按缺陷分解
    + R1 = 总电流），不对绝对数值下判断——可信稳态电流的验证属 Phase 5（F03/F04）。
    """
    raw = _zaz_dict()
    raw["simulation"]["n_events"] = 400  # 够统计窗口出稳定的 TAT/PF 通量
    cfg = config_from_dict(raw)
    stack = cfg.device.stack
    assert stack.n_materials == 2  # 确实是多材料叠层，走分段路径

    result = KMCSimulation(cfg).run()
    assert np.isfinite(result.current) and result.current > 0.0
    # 机制分解之和 = 总电流
    assert sum(result.by_mechanism.values()) == pytest.approx(result.current, rel=1e-12)
    assert set(result.by_mechanism) == set(MECHANISMS)
    # 按缺陷分解 + R1 两项 = 总电流（R1 不经过任何缺陷）
    electrode_only = result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]
    assert result.by_defect.sum() + electrode_only == pytest.approx(result.current, rel=1e-10)
    assert result.by_defect.size == len(result.defects) > 0
    # TAT 与 PF 两条陷阱通道都真的出现了（叠层里 TAT 主导）
    assert result.by_mechanism["tat"] > 0.0
    assert result.by_mechanism["pf"] > 0.0

