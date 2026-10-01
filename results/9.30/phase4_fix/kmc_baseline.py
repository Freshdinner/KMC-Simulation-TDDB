"""漏电流侧 KMC / 扫描结果的逐位快照（ADR-038 改动前后对比用）。

usage: kmc_baseline.py OUT.npz
代码版本由 PYTHONPATH 决定：HEAD 归档或当前工作树的 src（tests 目录取同一份，conftest 用来造 96 缺陷体系）。
"""
import sys
from dataclasses import replace

import numpy as np

from kmc_leakage.io import load_config
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.sweep import run_sweep
from conftest import make_config, make_defect_config, make_device

out = {}


def record(tag, result):
    for name in ("current", "current_density", "current_direct", "current_counts", "current_counts_cathode",
                 "elapsed_time", "n_events", "measured_events", "n_anode_events", "delta_n_inside",
                 "current_fast", "fast_exchange", "n_fast_traps"):
        out[tag + "/" + name] = np.asarray(getattr(result, name))
    for name, value in result.by_mechanism.items():
        out[tag + "/mech_" + name] = np.asarray(value)
    out[tag + "/by_defect"] = result.by_defect
    out[tag + "/occupancy"] = result.occupancy
    out[tag + "/occupancy_mf"] = result.occupancy_mean_field
    out[tag + "/validated"] = np.asarray(result.validated)
    print(tag, result.current_density, result.validated, flush=True)


def seeded(config, seed, n_events, **sim):
    return replace(config, simulation=replace(config.simulation, seed=seed, n_events=n_events, **sim))


record("single_run", KMCSimulation(seeded(load_config("examples/single_run.yaml"), 7, 3000)).run())
record("stack_t2_2nm", KMCSimulation(seeded(load_config("examples/stack_t2_2nm.yaml"), 5, 2000)).run())
record("zro2_tddb", KMCSimulation(seeded(load_config("examples/zro2_tddb.yaml"), 1, 2000)).run())

dense = make_device(defects=make_defect_config(concentration_cm3=3.0e19))
record("dense96", KMCSimulation(make_config(voltage=3.0, seed=1, device=dense)).run())
record("dense96_fast", KMCSimulation(seeded(make_config(voltage=3.0, seed=1, device=dense), 1, 1500,
                                            fast_trap_rate=1.0e8)).run())

sweep = run_sweep(seeded(load_config("examples/iv_sweep.yaml"), 11, 800))
out["sweep/current_density"] = sweep.current_density
for name, value in sweep.by_mechanism.items():
    out["sweep/mech_" + name] = value
for it, row in enumerate(sweep.results):
    for iv, point in enumerate(row):
        out["sweep/occ_{}_{}".format(it, iv)] = point.occupancy
print("sweep", sweep.current_density, flush=True)

np.savez(sys.argv[1], **out)
print("saved", len(out), "arrays", flush=True)
