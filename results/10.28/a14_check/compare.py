"""A1–A4（平均场每次求 F 的提速）前后对拍：同一个续跑状态上，用给定的源码目录求一步与几步轨迹，结果存成 npz。

在 `../../10.27/weak_tip/state_s{SEED}.npz` 上按 weak_tip 的参数（drive.py）：
1. 一步：导体划分、输运–热稳态（`tddb._steady_state`）与生成率（含温度场），存全体缺陷占据、内层输运 / 间隙 / 细丝电流、
   各缺陷处放热与温度、温度场最高值、各前驱位点的生成率，以及求 F 的次数、每次耗时与本进程峰值工作集；
2. STEPS > 0 时：从这个状态起 `tddb.run_tddb` 跑 STEPS 步，存每步的时刻、电流密度与断键的前驱位点编号。
耦合矩阵与源码无关，第一次建好后存到 coupling_s{SEED}.npz，之后直接读（`run_tddb` 里的 `IncrementalCoupling.build`
也换成读这份）。

usage: compare.py SRC_DIR LABEL SEED STEPS
"""
import ctypes
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
src, label, seed, n_steps = os.path.abspath(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, src)

from scipy import constants  # noqa: E402

from kmc_leakage import mean_field as MF  # noqa: E402
from kmc_leakage import tddb as T  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402

assert os.path.dirname(os.path.abspath(T.__file__)) == os.path.join(src, "kmc_leakage")
nm, eV = constants.nano, constants.electron_volt
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
stack = config.device.stack
layers = tuple(replace(layer, defects=replace(layer.defects, min_separation=0.42 * nm)) for layer in stack.layers)
config = replace(config, device=replace(config.device, stack=replace(stack, layers=layers)),
                 simulation=replace(config.simulation, seed=seed),
                 tddb=replace(config.tddb, transfer_integral=2.4 * eV, activation_energy=1.60 * eV,
                              activation_energy_tail=0.0, field_averaging_width=0.6 * nm,
                              dipole_moment=0.9 * constants.elementary_charge * constants.angstrom))
device, sim_config, tddb = config.device, config.simulation, config.tddb


class _MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_working_set_mb():
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 2 ** 20


state = T.load_state(os.path.join(ROOT, "results", "10.27", "weak_tip", "state_s{}.npz".format(seed)))
defects = state.defects
solver = PoissonSolver(device, sim_config.grid)
cache = os.path.join(HERE, "coupling_s{}.npz".format(seed))
if not os.path.exists(cache):
    built = T.IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    np.savez(cache, phi_matrix=built.phi_matrix, field_matrix=built.field_matrix, phi_bar=built.phi_bar)
saved = np.load(cache)
coupling = T.IncrementalCoupling(solver, np.array(defects.positions, dtype=float), np.array(defects.field_side, dtype=int),
                                 saved["phi_matrix"], saved["field_matrix"], saved["phi_bar"])
base_peak = peak_working_set_mb()

flux_times = []
original_flux = MF.mean_field_flux


def flux(sim, occupancy):
    began = time.time()
    result = original_flux(sim, occupancy)
    flux_times.append(time.time() - began)
    return result


MF.mean_field_flux = flux
began = time.time()
layout = T.conductor_layout(defects, coupling, solver, sim_config.voltage, tddb.transfer_integral)
step = T._steady_state(config, layout, defects, coupling, solver, state.occupancy, state.temperature)
steady_wall = time.time() - began
began = time.time()
rates = T.generation_rates(step.transport, tddb, state.precursors, step.field).rates
rates_wall = time.time() - began
MF.mean_field_flux = original_flux
mf = step.transport.result
result = dict(
    occupancy=step.transport.occupancy, transport_current=mf.current, gap_current=step.gap_current,
    filament_current=step.filament_current, heat=step.heat, temperature=step.temperature,
    peak_temperature=float(step.field.max()), rates=rates, iterations=mf.iterations, residual=mf.residual,
    flux_times=np.array(flux_times), steady_wall=steady_wall, rates_wall=rates_wall,
    base_peak_MB=base_peak, step_peak_MB=peak_working_set_mb(),
)
print("{} seed {}: F {} 次（中位数 {:.2f} s），稳态 {:.1f} s，生成率 {:.1f} s，残差 {:.1e}，J = {:.12e} A，ΣG = {:.12e} 1/s，"
      "峰值工作集 {:.0f} -> {:.0f} MB".format(
          label, seed, mf.iterations, float(np.median(flux_times)), steady_wall, rates_wall, mf.residual, mf.current,
          rates.sum(), base_peak, result["step_peak_MB"]), flush=True)


class Stop(Exception):
    pass


if n_steps > 0:
    T.IncrementalCoupling.build = classmethod(lambda cls, *args: coupling)
    times, currents, events = [], [], []
    original_draw = T._draw_generation

    def draw(rates, rng):
        event = original_draw(rates, rng)
        events.append(-1 if event is None else event[1])
        return event

    def progress(n_generated, t, j):
        times.append(t)
        currents.append(j)
        print("  step n={} t={:.12e} J={:.12e}".format(n_generated, t, j), flush=True)

    def checkpoint(tddb_state):
        if len(times) >= n_steps:
            raise Stop()

    T._draw_generation = draw
    began = time.time()
    try:
        T.run_tddb(config, progress=progress, start=state, checkpoint=checkpoint)
    except Stop:
        pass
    result.update(step_times=np.array(times), step_currents=np.array(currents), events=np.array(events),
                  trajectory_wall=time.time() - began, trajectory_peak_MB=peak_working_set_mb())
    print("  {} 步 {:.0f} s，峰值工作集 {:.0f} MB".format(n_steps, result["trajectory_wall"], result["trajectory_peak_MB"]))
np.savez(os.path.join(HERE, "result_{}_s{}.npz".format(label, seed)), **result)
