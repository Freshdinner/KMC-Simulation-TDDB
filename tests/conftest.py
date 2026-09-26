"""测试共用的小配置：缺陷少、网格粗、事件少，保证测试跑得快。"""

import numpy as np
import pytest
from scipy import constants

from kmc_leakage.config import (
    Config,
    DefectConfig,
    DefectLevelConfig,
    DefectSpatialConfig,
    DeviceConfig,
    DielectricMaterial,
    DielectricStack,
    ElectrodeMaterial,
    GridConfig,
    SimulationConfig,
    SweepConfig,
)

EV = constants.electron_volt
M_E = constants.electron_mass


def make_material(name="reference-preset", **overrides):
    """默认参考预设的介质材料。"""
    params = dict(
        name=name,
        kappa=38.0,
        eps_opt=5.6,
        band_gap=5.4 * EV,
        electron_affinity=2.86 * EV,
        m_cb=1.2 * M_E,
        m_vb=2.5 * M_E,
        m_di=1.2 * M_E,
        phonon_energy=0.04 * EV,
        huang_rhys=1.0,
        attempt_frequency=1.0e13,
    )
    params.update(overrides)
    return DielectricMaterial(**params)


def make_defect_config(concentration_cm3=3.0e18, **kwargs):
    return DefectConfig(
        concentration=concentration_cm3 * constants.centi ** -3,
        level=kwargs.get("level", DefectLevelConfig(mode="fixed", ED=1.15 * EV)),
        spatial=kwargs.get("spatial", DefectSpatialConfig(mode="uniform")),
        charge_state=kwargs.get("charge_state", 1.0),
        min_separation=kwargs.get("min_separation", 0.5e-9),
    )


def make_device(barrier_height=1.74 * EV, Lx=20e-9, Ly=20e-9, Lz=8e-9, defects=None, **electrode_kwargs):
    """默认参考预设的小尺寸版本：单层退化叠层（一种材料、一层、无内部界面）。"""
    electrode = ElectrodeMaterial(
        name=electrode_kwargs.get("name", "TiN"),
        work_function=electrode_kwargs.get("work_function", 4.6 * EV),
        effective_mass=electrode_kwargs.get("effective_mass", M_E),
        fermi_energy=electrode_kwargs.get("fermi_energy", 5.0 * EV),
    )
    stack = DielectricStack.single(
        make_material(),
        thickness=Lz,
        defects=defects if defects is not None else make_defect_config(),
    )
    return DeviceConfig(Lx, Ly, electrode, electrode, stack, barrier_height=barrier_height)


def make_config(voltage=2.5, temperature=298.15, n_events=400, seed=7, **kwargs):
    device = kwargs.get("device")
    if device is None:
        device = make_device(defects=kwargs.get("defects", make_defect_config()))
    return Config(
        device=device,
        simulation=SimulationConfig(
            voltage=voltage,
            temperature=temperature,
            n_events=n_events,
            burn_in_fraction=0.2,
            seed=seed,
            grid=kwargs.get("grid", GridConfig(8, 8, 16)),
        ),
        sweep=kwargs.get(
            "sweep",
            SweepConfig(voltages=np.array([2.0, 3.0]), temperatures=np.array([298.15])),
        ),
        ensemble=kwargs.get("ensemble"),
    )


@pytest.fixture
def config():
    return make_config()
