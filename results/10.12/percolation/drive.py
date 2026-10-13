"""ADR-054 验收：贯穿通路判据、细丝电流、接电极的导电簇与电极等电位，examples/zro2_tddb.yaml 工况（50×50 nm、3.6 V / 25 °C、
开热反馈、E_A = 1.30 eV）下跑一条轨迹到击穿，边跑边写结果。

usage: drive.py SEED N_T_CM3 R_C_NM [STOP_TIME_S]

N_T_CM3 覆盖 `tddb.precursor_concentration_cm3`，R_C_NM 覆盖 `tddb.percolation_distance_nm`，STOP_TIME_S 覆盖 `tddb.stop_time_s`。
由 `../fixed_ea/drive.py` 改写：不改 `tddb.run_tddb`，在 tddb 模块的命名空间里包一层它每步调用的函数，在 `progress` 回调里追加一行 CSV：
- `steps_s{SEED}.csv`：第 k 行是第 k 个空位出生后的稳态（k = 0 为 t = 0），总电流、内层输运电流与细丝电流，ΣG 按电极项 / 陷阱项拆开
  （击穿记录不求生成率，记 nan），剩余前驱位点数（击穿记录记 −1），平均场收敛诊断，导电簇（阴极侧 / 阳极侧成员数、阴极侧最高点、阳极侧最低点、
  导体间隙），这个空位所用前驱的位置与它出生后所在导电簇接的电极（−1 不接、0 阴极、1 阳极、2 贯穿）；
- `snap_s{SEED}_n{NNNN}.npz`：每 25 个空位（含 0）与击穿记录存一次缺陷位置、能级、固定电荷、平均场占据、出生时刻、导电簇归属与剩余前驱位置；
- 结束时写 `final_s{SEED}.csv`（`write_tddb_csv`）与 `final_s{SEED}.npz`。
标准输出是逐步日志（第一行记代码版本）。
"""
import ctypes
import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import tddb as tddb_module  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import MECHANISMS  # noqa: E402
from kmc_leakage.tddb import conducting_clusters, conductor_gap, run_tddb, write_tddb_csv  # noqa: E402

SNAPSHOT_EVERY = 25
nm, eV = constants.nano, constants.electron_volt
per_cm2 = constants.centi ** 2  # A/m² -> A/cm²

seed = int(sys.argv[1])
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
tddb = replace(config.tddb, precursor_concentration=float(sys.argv[2]) / constants.centi ** 3,
               percolation_distance=float(sys.argv[3]) * nm)
if len(sys.argv) > 4:
    tddb = replace(tddb, stop_time=float(sys.argv[4]))
config = replace(config, simulation=replace(config.simulation, seed=seed), tddb=tddb)
device = config.device
ambient = config.simulation.temperature
area = device.area

commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
print("code {}{} | seed {} | V {} V | T {} K | E_A {:.3f} eV | N_T {:.2e} cm-3 | r_c {:.2f} nm | stop {:.3g} s".format(
    commit, " (src modified)" if dirty else "", seed, config.simulation.voltage, ambient, tddb.activation_energy / eV,
    tddb.precursor_concentration * constants.centi ** 3, tddb.percolation_distance / nm, tddb.stop_time), flush=True)


class _MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def working_set_mb():
    """(当前工作集, 启动以来的峰值工作集) [MB]（Windows）。"""
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.WorkingSetSize / 2 ** 20, counters.PeakWorkingSetSize / 2 ** 20


# ---- 包裹 run_tddb 每步调用的函数，抓这一步的状态 ----
state = {}
original = {name: getattr(tddb_module, name) for name in (
    "solve_mean_field", "solve_temperature", "transport_temperature", "generation_rates", "_draw_generation")}


def solve_mean_field(inner, start):
    state["inner"] = inner
    state["mf"] = original["solve_mean_field"](inner, start)
    state.pop("generation", None)  # 击穿记录不求生成率
    return state["mf"]


def transport_temperature(*args):
    state["T_effective"] = original["transport_temperature"](*args)
    return state["T_effective"]


def solve_temperature(*args):
    state["T_field"] = original["solve_temperature"](*args)
    return state["T_field"]


def generation_rates(inner, occupancy, tddb_config, precursors, temperature_field=None):
    state["precursors"] = precursors
    state["generation"] = original["generation_rates"](inner, occupancy, tddb_config, precursors, temperature_field)
    return state["generation"]


def draw_generation(rates, rng):
    event = original["_draw_generation"](rates, rng)
    if event is not None:
        state["birth"] = state["precursors"].positions[event[1]]
    return event


for name, func in (("solve_mean_field", solve_mean_field), ("solve_temperature", solve_temperature),
                   ("transport_temperature", transport_temperature), ("generation_rates", generation_rates),
                   ("_draw_generation", draw_generation)):
    setattr(tddb_module, name, func)

COLUMNS = (["n_generated", "t_s", "J_A_cm2", "J_transport_A_cm2", "J_filament_A_cm2"]
           + ["J_{}_A_cm2".format(m) for m in MECHANISMS] + [
    "G_total_s", "G_electrode_s", "G_trap_s", "n_precursors", "mf_iterations", "mf_newton_steps", "mf_residual",
    "mf_converged", "n_cathode_members", "n_anode_members", "cathode_front_nm", "anode_front_nm", "gap_nm",
    "T_effective_K", "T_peak_K", "birth_x_nm", "birth_y_nm", "birth_z_nm", "birth_side", "wall_s", "ws_MB", "ws_peak_MB"])
steps_path = os.path.join(HERE, "steps_s{}.csv".format(seed))
with open(steps_path, "w", encoding="utf-8") as handle:
    handle.write(",".join(COLUMNS) + "\n")
birth_time = []
start = time.time()


def progress(n_generated, t, j):
    inner, mf = state["inner"], state["mf"]
    generation = state.get("generation")
    if n_generated > 0:
        birth_time.append(t)
    defects, p = inner.defects, mf.occupancy
    n_initial = len(defects) - n_generated
    sides = conducting_clusters(defects.positions, device, tddb.percolation_distance)
    z = defects.positions[:, 2] / nm
    cathode, anode = (sides == 0) | (sides == 2), (sides == 1) | (sides == 2)
    point = state.pop("birth") if n_generated > 0 else np.full(3, np.nan)
    spanning = bool(np.any(sides == 2))
    filament = tddb_module._QUANTUM_CONDUCTANCE * config.simulation.voltage if spanning else 0.0
    left = -1 if generation is None else len(state["precursors"])  # 击穿记录不求生成率，剩余位点数记 -1
    if generation is None:
        G = G_electrode = G_trap = float("nan")
    else:
        G = float(generation.rates.sum())
        G_electrode = float(np.sum(generation.electrode * generation.bond_breaking))
        G_trap = float(np.sum(generation.trap * generation.bond_breaking))
    row = [n_generated, "{:.9e}".format(t), "{:.9e}".format(j * per_cm2), "{:.9e}".format(mf.current_density * per_cm2),
           "{:.9e}".format(filament / area * per_cm2)] + [
        "{:.9e}".format(mf.by_mechanism[m] / mf.area * per_cm2) for m in MECHANISMS] + [
        "{:.9e}".format(G), "{:.9e}".format(G_electrode), "{:.9e}".format(G_trap), left,
        mf.iterations, mf.newton_steps, "{:.3e}".format(mf.residual), int(mf.converged),
        int(cathode.sum()), int(anode.sum()), "{:.4f}".format(z[cathode].max() if cathode.any() else 0.0),
        "{:.4f}".format(z[anode].min() if anode.any() else device.Lz / nm),
        "{:.4f}".format(conductor_gap(defects.positions, sides, device) / nm),
        "{:.9f}".format(state.get("T_effective", ambient)),
        "{:.9f}".format(float(state["T_field"].max()) if "T_field" in state else ambient)] + [
        "{:.6f}".format(v / nm) for v in point] + [int(sides[-1]) if n_generated > 0 else -1] + [
        "{:.0f}".format(time.time() - start)] + ["{:.0f}".format(v) for v in working_set_mb()]
    with open(steps_path, "a", encoding="utf-8") as handle:
        handle.write(",".join(str(v) for v in row) + "\n")
    if n_generated % SNAPSHOT_EVERY == 0 or spanning:
        np.savez(os.path.join(HERE, "snap_s{}_n{:04d}.npz".format(seed, n_generated)),
                 positions=defects.positions, ED=defects.ED, fixed_charge=defects.fixed_charge, occupancy=p,
                 n_initial=n_initial, birth_time=np.concatenate([np.zeros(n_initial), birth_time]), time=t, sides=sides,
                 precursor_positions=state["precursors"].positions if "precursors" in state else np.zeros((0, 3)))
    print("{:4d} t={:.4e} J={:.4e} (TAT {:.4e}) G={:.3e} left={} it={} newton={} res={:.1e} cath={} front={:.2f} "
          "anode={} gap={:.2f} z={:.3f} side={} wall={:.0f} ws={}/{} MB".format(
              n_generated, t, j * per_cm2, mf.current_density * per_cm2, G, left, mf.iterations,
              mf.newton_steps, mf.residual, int(cathode.sum()), z[cathode].max() if cathode.any() else 0.0,
              int(anode.sum()), conductor_gap(defects.positions, sides, device) / nm, point[2] / nm,
              int(sides[-1]) if n_generated > 0 else -1, time.time() - start, row[-2], row[-1]),
          flush=True)


result = run_tddb(config, progress=progress)
write_tddb_csv(result, os.path.join(HERE, "final_s{}.csv".format(seed)))
np.savez(os.path.join(HERE, "final_s{}.npz".format(seed)), positions=result.defects.positions,
         fixed_charge=result.defects.fixed_charge, birth_time=result.birth_time, occupancy=result.occupancy,
         ED=result.defects.ED, n_initial=result.n_initial, end_time=result.end_time, stop_reason=result.stop_reason)
print("done {} end_time={:.6e} breakdown_time={} wall={:.0f}".format(
    result.stop_reason, result.end_time, result.breakdown_time, time.time() - start), flush=True)
