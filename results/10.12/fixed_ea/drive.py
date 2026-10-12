"""ADR-053 验收：E_A 全部前驱位点相同（1.30 eV、N_T = 10²⁰ cm⁻³）下 examples/zro2_tddb.yaml 默认工况（50×50 nm、3.6 V / 25 °C、
开热反馈、击穿判据 J > 10 A/cm²）跑一条轨迹，边跑边写结果。

usage: drive.py SEED [STOP_TIME_S]

STOP_TIME_S 覆盖 `tddb.stop_time_s`。P_bb 未饱和时 E_A 平移 ΔE 只把时刻乘 exp(ΔE/k_BT)，所以跑出来的击穿时刻可以直接换算成
t_BD = 1000 s 需要的 E_A。同一种子的前驱位置与 ADR-052 的轨迹（../../10.11/discrete/long）相同。

由 `../../10.11/discrete/drive.py` 改写：前驱位点不再带各自的 E_A。不改 `tddb.run_tddb`，在 tddb 模块的命名空间里包一层它每步
调用的函数，在 `progress` 回调里追加一行 CSV：
- `steps_s{SEED}.csv`：第 k 行是第 k 个空位出生后的稳态（k = 0 为 t = 0），ΣG 按电极项 / 陷阱项拆开，剩余前驱位点数，
  平均场收敛诊断，这个空位所用前驱的位置，生成空位的 z 最高点与中位数；
- `snap_s{SEED}_n{NNNN}.npz`：每 25 个空位（含 0）存一次缺陷位置、能级、固定电荷、平均场占据、出生时刻、输运温度与剩余前驱位置；
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
from kmc_leakage.tddb import run_tddb, write_tddb_csv  # noqa: E402

SNAPSHOT_EVERY = 25
nm, eV = constants.nano, constants.electron_volt
per_cm2 = constants.centi ** 2  # A/m² -> A/cm²

seed = int(sys.argv[1])
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, seed=seed))
if len(sys.argv) > 2:
    config = replace(config, tddb=replace(config.tddb, stop_time=float(sys.argv[2])))
tddb = config.tddb
ambient = config.simulation.temperature

commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
print("code {}{} | seed {} | V {} V | T {} K | E_A {:.3f} eV | N_T {:.2e} cm-3 | breakdown {:.1f} A/cm2 | stop {:.3g} s".format(
    commit, " (src modified)" if dirty else "", seed, config.simulation.voltage, ambient, tddb.activation_energy / eV,
    tddb.precursor_concentration * constants.centi ** 3, tddb.breakdown_current_density * per_cm2, tddb.stop_time), flush=True)


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
    "solve_mean_field", "heat_release", "solve_temperature", "transport_temperature", "generation_rates",
    "_cluster_diagnostics", "_draw_generation")}


def solve_mean_field(inner, start):
    state["inner"] = inner
    state["mf"] = original["solve_mean_field"](inner, start)
    return state["mf"]


def heat_release(inner, occupancy):
    state["heat"] = original["heat_release"](inner, occupancy)
    return state["heat"]


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


def cluster_diagnostics(inner):
    state["clusters"] = original["_cluster_diagnostics"](inner)
    return state["clusters"]


def draw_generation(rates, rng):
    event = original["_draw_generation"](rates, rng)
    if event is not None:
        state["birth"] = state["precursors"].positions[event[1]]
    return event


for name, func in (("solve_mean_field", solve_mean_field), ("heat_release", heat_release),
                   ("solve_temperature", solve_temperature), ("transport_temperature", transport_temperature),
                   ("generation_rates", generation_rates), ("_cluster_diagnostics", cluster_diagnostics),
                   ("_draw_generation", draw_generation)):
    setattr(tddb_module, name, func)

COLUMNS = (["n_generated", "t_s", "J_A_cm2"] + ["J_{}_A_cm2".format(m) for m in MECHANISMS] + [
    "G_total_s", "G_electrode_s", "G_trap_s", "n_precursors", "space_charge_C", "empty_generated",
    "mf_iterations", "mf_newton_steps", "mf_residual", "mf_converged", "n_clusters", "max_cluster_size",
    "largest_component", "n_split", "P_dielectric_W", "T_transport_K", "T_effective_K", "T_peak_K",
    "generated_top_nm", "generated_median_z_nm", "birth_x_nm", "birth_y_nm", "birth_z_nm",
    "wall_s", "ws_MB", "ws_peak_MB"])
steps_path = os.path.join(HERE, "steps_s{}.csv".format(seed))
with open(steps_path, "w", encoding="utf-8") as handle:
    handle.write(",".join(COLUMNS) + "\n")
birth_time = []
transport = [ambient]  # 下一步内层用的输运温度 = 这一步的 T_effective（run_tddb 的 `transport = effective`）
start = time.time()


def progress(n_generated, t, j):
    inner, mf, heat, generation = state["inner"], state["mf"], state["heat"], state["generation"]
    if n_generated > 0:
        birth_time.append(t)
    defects, p = inner.defects, mf.occupancy
    n_initial = len(defects) - n_generated
    z_generated = defects.positions[n_initial:, 2] / nm
    point = state.pop("birth") if n_generated > 0 else np.full(3, np.nan)
    T_effective = state.get("T_effective", ambient)
    T_peak = float(state["T_field"].max()) if "T_field" in state else ambient
    space_charge = constants.elementary_charge * float(np.sum(defects.charge_state - p)) + float(np.sum(defects.fixed_charge))
    G = generation.rates
    G_electrode = float(np.sum(generation.electrode * generation.bond_breaking))
    G_trap = float(np.sum(generation.trap * generation.bond_breaking))
    row = [n_generated, "{:.9e}".format(t), "{:.9e}".format(j * per_cm2)] + [
        "{:.9e}".format(mf.by_mechanism[m] / mf.area * per_cm2) for m in MECHANISMS] + [
        "{:.9e}".format(float(G.sum())), "{:.9e}".format(G_electrode), "{:.9e}".format(G_trap),
        len(state["precursors"]), "{:.9e}".format(space_charge), "{:.6g}".format(float(np.sum(1.0 - p[n_initial:]))),
        mf.iterations, mf.newton_steps, "{:.3e}".format(mf.residual), int(mf.converged), *state["clusters"],
        "{:.9e}".format(heat.dielectric), "{:.9f}".format(transport[0]), "{:.9f}".format(T_effective),
        "{:.9f}".format(T_peak),
        "{:.4f}".format(z_generated.max() if z_generated.size else 0.0),
        "{:.4f}".format(float(np.median(z_generated)) if z_generated.size else 0.0)] + [
        "{:.6f}".format(v / nm) for v in point] + ["{:.0f}".format(time.time() - start)] + [
        "{:.0f}".format(v) for v in working_set_mb()]
    with open(steps_path, "a", encoding="utf-8") as handle:
        handle.write(",".join(str(v) for v in row) + "\n")
    transport[0] = T_effective
    if n_generated % SNAPSHOT_EVERY == 0:
        np.savez(os.path.join(HERE, "snap_s{}_n{:04d}.npz".format(seed, n_generated)),
                 positions=defects.positions, ED=defects.ED, fixed_charge=defects.fixed_charge, occupancy=p,
                 n_initial=n_initial, birth_time=np.concatenate([np.zeros(n_initial), birth_time]), time=t,
                 transport_temperature=float(row[COLUMNS.index("T_transport_K")]),
                 precursor_positions=state["precursors"].positions)
    print("{:4d} t={:.4e} J={:.4e} G={:.3e} (trap {:.1e}) left={} empty={:.1f} it={} newton={} res={:.1e} split={} "
          "top={:.2f} dT={:.2e} z={:.3f} wall={:.0f} ws={}/{} MB".format(
              n_generated, t, j * per_cm2, G.sum(), G_trap, len(state["precursors"]),
              float(np.sum(1.0 - p[n_initial:])), mf.iterations, mf.newton_steps, mf.residual, state["clusters"][3],
              z_generated.max() if z_generated.size else 0.0, T_peak - ambient, point[2] / nm,
              time.time() - start, row[-2], row[-1]),
          flush=True)


result = run_tddb(config, progress=progress)
write_tddb_csv(result, os.path.join(HERE, "final_s{}.csv".format(seed)))
np.savez(os.path.join(HERE, "final_s{}.npz".format(seed)), positions=result.defects.positions,
         fixed_charge=result.defects.fixed_charge, birth_time=result.birth_time, occupancy=result.occupancy,
         ED=result.defects.ED, n_initial=result.n_initial, end_time=result.end_time, stop_reason=result.stop_reason)
print("done {} end_time={:.6e} breakdown_time={} wall={:.0f}".format(
    result.stop_reason, result.end_time, result.breakdown_time, time.time() - start), flush=True)
