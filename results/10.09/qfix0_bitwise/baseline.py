"""q_fix = 0 时结果与改动前逐位相同的核对：漏电流侧 KMC、平均场、放热与 TDDB 短轨迹的全部输出存成 npz。

usage: baseline.py OUT.npz
改动前（0911d72 的 src）与改动后各跑一次，再用 compare.py 逐数组比较。改动后的代码里生成空位的固定电荷
（O_i⁻，`tddb._INTERSTITIAL_CHARGE`）置 0，就是改动前"生成反应丢掉 −e"的记账；其余场景全部缺陷 q_fix = 0。
"""
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from conftest import make_config, make_defect_config, make_device  # noqa: E402
from kmc_leakage import tddb as tddb_module  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402
from kmc_leakage.thermal import heat_release  # noqa: E402
from test_tddb import _config  # noqa: E402

if hasattr(tddb_module, "_INTERSTITIAL_CHARGE"):
    tddb_module._INTERSTITIAL_CHARGE = 0.0
    print("O_i charge set to 0 (pre-change bookkeeping)", flush=True)

out = {}


def seeded(config, seed, n_events):
    return replace(config, simulation=replace(config.simulation, seed=seed, n_events=n_events))


def record_kmc(tag, result):
    for name in ("current", "current_direct", "current_counts", "current_counts_cathode", "elapsed_time",
                 "n_events", "delta_n_inside", "current_fast"):
        out[tag + "/" + name] = np.asarray(getattr(result, name))
    for name, value in result.by_mechanism.items():
        out[tag + "/mech_" + name] = np.asarray(value)
    out[tag + "/by_defect"] = result.by_defect
    out[tag + "/occupancy"] = result.occupancy
    out[tag + "/occupancy_mf"] = result.occupancy_mean_field
    print(tag, result.current_density, flush=True)


def record_mf(tag, sim):
    result = solve_mean_field(sim)
    out[tag + "/current"] = np.asarray(result.current)
    out[tag + "/occupancy"] = result.occupancy
    out[tag + "/by_defect"] = result.by_defect
    out[tag + "/iterations"] = np.asarray(result.iterations)
    out[tag + "/residual"] = np.asarray(result.residual)
    heat = heat_release(sim, result.occupancy)
    out[tag + "/heat_defect"] = heat.defect
    out[tag + "/heat_electrodes"] = np.array([heat.cathode, heat.anode])
    print(tag, result.current_density, result.iterations, flush=True)


def record_tddb(tag, result):
    for name in ("time", "current", "generation_rate", "n_defects", "space_charge", "empty_generated", "iterations",
                 "residual", "converged", "n_clusters", "max_cluster_size", "largest_component", "n_split", "power",
                 "transport_temperature", "effective_temperature", "peak_temperature", "dense_cluster_size",
                 "birth_time", "occupancy"):
        out[tag + "/" + name] = np.asarray(getattr(result, name))
    for name, value in result.by_mechanism.items():
        out[tag + "/mech_" + name] = value
    out[tag + "/positions"] = result.defects.positions
    out[tag + "/ED"] = result.defects.ED
    out[tag + "/end_time"] = np.asarray(result.end_time)
    print(tag, result.stop_reason, result.end_time, result.current_density[-1], flush=True)


record_kmc("single_run", KMCSimulation(seeded(load_config(os.path.join(ROOT, "examples", "single_run.yaml")), 7, 3000)).run())
record_kmc("zro2_tddb", KMCSimulation(seeded(load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml")), 1, 2000)).run())
dense = make_device(defects=make_defect_config(concentration_cm3=3.0e19))
record_kmc("dense96", KMCSimulation(make_config(voltage=3.0, seed=1, device=dense)).run())
record_mf("mf_dense96", KMCSimulation(make_config(voltage=3.0, seed=1, device=dense)))
split = make_device(Lx=10e-9, Ly=10e-9, defects=make_defect_config(concentration_cm3=1.0e20))
record_mf("mf_split80", KMCSimulation(make_config(voltage=3.0, seed=1, device=split)))
record_tddb("tddb_dense96", tddb_module.run_tddb(_config(seed=1, concentration_cm3=3.0e19)))
record_tddb("tddb_thermal", tddb_module.run_tddb(_config(seed=2, concentration_cm3=3.0e19, thermal_conductivity=1.0,
                                                         max_generated_vacancies=3)))
np.savez(sys.argv[1], **out)
print("saved", len(out), "arrays", flush=True)
