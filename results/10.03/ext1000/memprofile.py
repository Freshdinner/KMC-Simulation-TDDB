"""每步各阶段的内存峰值（tracemalloc，numpy 分配会计入）：从 ADR-045 的快照重建状态，逐阶段量峰值。

usage: memprofile.py SEED N
"""
import ctypes
import os
import sys
import time
import tracemalloc
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.defects import DefectEnsemble  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402
from kmc_leakage.tddb import (ElectrodeCaptureGeneration, TrapCaptureGeneration, generation_cells,  # noqa: E402
                              generation_rates, top_boundary_potential)
from scipy import constants  # noqa: E402
from kmc_leakage.thermal import heat_release, power_map, solve_temperature  # noqa: E402

seed, n = int(sys.argv[1]), int(sys.argv[2])
base = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
device, tddb, ambient = base.device, base.tddb, base.simulation.temperature
snap = np.load(os.path.join(ROOT, "results", "10.03", "rerun", "snap_s{}_n{:04d}.npz".format(seed, n)))
material = device.stack.single_material
defects = DefectEnsemble.uniform(snap["positions"], snap["ED"], material,
                                 band_reference=float(device.layer_band_references()[0]),
                                 charge_state=device.stack.layers[0].defects.charge_state)
config = replace(base, simulation=replace(base.simulation, seed=seed,
                                          temperature=float(snap["transport_temperature"])))
class _Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def working_set():
    """(当前工作集, 进程启动以来的峰值工作集) [MB]（Windows）。"""
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.WorkingSetSize / 2 ** 20, counters.PeakWorkingSetSize / 2 ** 20


def stage(name, func, *args):
    # Python 3.8 没有 reset_peak：每阶段重新开始追踪，峰值只计这一阶段新分配的内存
    tracemalloc.start()
    tick = time.time()
    out = func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("{:22s} peak +{:7.1f} MB  retained +{:7.1f} MB  {:6.1f} s | working set {:7.1f} MB, peak so far {:7.1f} MB"
          .format(name, peak / 2 ** 20, current / 2 ** 20, time.time() - tick, *working_set()), flush=True)
    return out


inner = stage("KMCSimulation", KMCSimulation, config, defects)
mf = stage("solve_mean_field", solve_mean_field, inner, snap["occupancy"])
heat = stage("heat_release", heat_release, inner, mf.occupancy)
T = stage("solve_temperature", lambda: solve_temperature(
    inner.solver, power_map(inner.solver, defects.positions, heat.defect), tddb.thermal_conductivity, ambient))
cells = generation_cells(device, tddb.generation_grid)
table = stage("generation_rates", generation_rates, inner, mf.occupancy, tddb, cells, T)
del table
# generation_rates 的两类事件分开量
solver, p = inner.solver, mf.occupancy
rho = solver.deposit(defects.positions, constants.elementary_charge * (defects.charge_state - p))
phi = solver.solve(rho, top_boundary_potential(device, config.simulation.voltage))
bond = stage("gaussian_average", lambda: solver.gaussian_average(
    np.stack(solver.electric_field(phi), axis=-1), tddb.field_averaging_width))
stage("ElectrodeCapture", lambda: ElectrodeCaptureGeneration(inner, tddb, cells, phi, bond, T))
stage("TrapCapture", lambda: TrapCaptureGeneration(inner, p, tddb, phi, bond, T))
print("N = {}".format(len(defects)))
