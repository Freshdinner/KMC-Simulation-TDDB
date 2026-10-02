"""ADR-042 验证脚本的公共部分：重建状态（t = 0 初始缺陷或终态 npz），求内层平均场。"""
import time
from dataclasses import replace

import numpy as np

from kmc_leakage.defects import DefectEnsemble, generate_defects
from kmc_leakage.io import load_config
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.mean_field import solve_mean_field

STATES = {
    # 名字: (种子, npz 路径；None = t = 0 的初始缺陷)
    "t0_s1": (1, None), "t0_s2": (2, None), "t0_s3": (3, None), "t0_s4": (4, None),
    "p4_s1": (1, "results/9.30/phase4/p4_thermal_s1.npz"),  # ADR-037 终态，410 个缺陷
    "p4_s2": (2, "results/9.30/phase4/p4_thermal_s2.npz"),
    "p4_s3": (3, "results/9.30/phase4/p4_thermal_s3.npz"),
    "p4_s4": (4, "results/9.30/phase4/p4_thermal_s4.npz"),
    "rerun_s2": (2, "results/9.30/phase4_rerun/p4fix_thermal_s2.npz"),  # ADR-041 终态，1010 个缺陷
}


def load_state(name):
    seed, path = STATES[name]
    config = load_config("examples/zro2_tddb.yaml")
    config = replace(config, simulation=replace(config.simulation, seed=seed))
    if path is None:
        defects, start = generate_defects(config.device, np.random.default_rng(seed)), None
    else:
        data = np.load(path)
        material = config.device.stack.single_material
        defects = DefectEnsemble.uniform(data["positions"], data["ED"], material,
                                         band_reference=float(config.device.layer_band_references()[0]))
        start = data["occupancy"]
    inner = KMCSimulation(config, defects=defects)
    tick = time.time()
    mf = solve_mean_field(inner, start)
    return config, inner, mf.occupancy, time.time() - tick
