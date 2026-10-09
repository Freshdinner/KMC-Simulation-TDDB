"""ADR-048 验收：生成按电荷守恒记账（O_i⁻ 与空位重合）+ 平均场牛顿法兜底之后，examples/zro2_tddb.yaml 默认工况
（3.6 V / 25 °C，开热反馈，击穿判据 J > 10 A/cm²）跑一条轨迹，边跑边写结果。

usage: drive.py SEED

由 `results/10.03/ext1000/drive.py` 改写：去掉 ADR-047 删掉的渗流判据与限流参数，加平均场的牛顿步数。不改
`tddb.run_tddb`，在 tddb 模块的命名空间里包一层它每步调用的函数，在 `progress` 回调里追加一行 CSV：
- `steps_s{SEED}.csv`：第 k 行是第 k 个空位出生后的稳态（k = 0 为 t = 0），ΣG 按单元事件 / 源陷阱事件拆开，含平均场
  收敛诊断、这个空位的出生位置与事件类型（cell = 电极项，trap = 陷阱项）、生成空位的 z 最高点；
- `snap_s{SEED}_n{NNNN}.npz`：每 25 个空位（含 0）存一次缺陷位置、能级、固定电荷、平均场占据、出生时刻、输运温度；
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
nm = constants.nano
per_cm2 = constants.centi ** 2  # A/m² -> A/cm²

seed = int(sys.argv[1])
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, seed=seed))
tddb = config.tddb
ambient = config.simulation.temperature

commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
print("code {}{} | seed {} | V {} V | T {} K | breakdown {:.1f} A/cm2 | k_TH {} W/(m K) | O_i charge {:.3e} C".format(
    commit, " (src modified)" if dirty else "", seed, config.simulation.voltage, ambient,
    tddb.breakdown_current_density * per_cm2, tddb.thermal_conductivity, tddb_module._INTERSTITIAL_CHARGE), flush=True)


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


def generation_rates(*args):
    # 上一步的包装 sample 引用着自己所在实例的绑定方法，构成引用环，拆开后整张事件表随即释放（ADR-046）
    if "table" in state:
        for events in state.pop("table"):
            del events.sample
    table = original["generation_rates"](*args)
    for kind, events in (("cell", table.cells), ("trap", table.traps)):
        def sample(index, rng, _sample=events.sample, _kind=kind):
            point = _sample(index, rng)
            if point is not None:
                state["birth"] = (_kind, int(index), point)
            return point
        events.sample = sample
    state["table"] = table
    return table


def cluster_diagnostics(inner):
    state["clusters"] = original["_cluster_diagnostics"](inner)
    return state["clusters"]


def draw_generation(rates, full, table, rng):
    event = original["_draw_generation"](rates, full, table, rng)
    state["full"] = full  # _draw_generation 原地改 full；run_tddb 之后 append 一个 False
    return event


for name, func in (("solve_mean_field", solve_mean_field), ("heat_release", heat_release),
                   ("solve_temperature", solve_temperature), ("transport_temperature", transport_temperature),
                   ("generation_rates", generation_rates), ("_cluster_diagnostics", cluster_diagnostics),
                   ("_draw_generation", draw_generation)):
    setattr(tddb_module, name, func)

COLUMNS = (["n_generated", "t_s", "J_A_cm2"] + ["J_{}_A_cm2".format(m) for m in MECHANISMS] + [
    "G_total_s", "G_cells_s", "G_traps_s", "n_full_events", "space_charge_C", "empty_generated",
    "mf_iterations", "mf_newton_steps", "mf_residual", "mf_converged", "n_clusters", "max_cluster_size",
    "largest_component", "n_split", "P_dielectric_W", "P_cathode_W", "P_anode_W", "UI_trap_W",
    "T_transport_K", "T_effective_K", "T_peak_K", "generated_top_nm", "generated_median_z_nm",
    "birth_event", "birth_index", "birth_x_nm", "birth_y_nm", "birth_z_nm", "wall_s", "ws_MB", "ws_peak_MB"])
steps_path = os.path.join(HERE, "steps_s{}.csv".format(seed))
with open(steps_path, "w", encoding="utf-8") as handle:
    handle.write(",".join(COLUMNS) + "\n")
birth_time = []
transport = [ambient]  # 下一步内层用的输运温度 = 这一步的 T_effective（run_tddb 的 `transport = effective`）
start = time.time()


def progress(n_generated, t, j):
    inner, mf, heat, table = state["inner"], state["mf"], state["heat"], state["table"]
    n_cells = table.cells.rates.size
    if "full" in state:
        mirror = np.append(state["full"], False)
    else:
        mirror = np.zeros(n_cells + table.traps.rates.size, dtype=bool)
    live = np.where(mirror, 0.0, np.concatenate([table.cells.rates, table.traps.rates]))
    if n_generated > 0:
        birth_time.append(t)
    defects, p = inner.defects, mf.occupancy
    n_initial = len(defects) - n_generated
    z_generated = defects.positions[n_initial:, 2] / nm
    i_r1 = mf.by_mechanism["direct_fn"] + mf.by_mechanism["thermionic"]
    kind, index, point = state.pop("birth") if n_generated > 0 else ("none", -1, np.full(3, np.nan))
    T_effective = state.get("T_effective", ambient)
    T_peak = float(state["T_field"].max()) if "T_field" in state else ambient
    space_charge = constants.elementary_charge * float(np.sum(defects.charge_state - p)) + float(np.sum(defects.fixed_charge))
    row = [n_generated, "{:.9e}".format(t), "{:.9e}".format(j * per_cm2)] + [
        "{:.9e}".format(mf.by_mechanism[m] / mf.area * per_cm2) for m in MECHANISMS] + [
        "{:.9e}".format(float(live.sum())), "{:.9e}".format(float(live[:n_cells].sum())),
        "{:.9e}".format(float(live[n_cells:].sum())), int(mirror.sum()), "{:.9e}".format(space_charge),
        "{:.6g}".format(float(np.sum(1.0 - p[n_initial:]))), mf.iterations, mf.newton_steps,
        "{:.3e}".format(mf.residual), int(mf.converged), *state["clusters"],
        "{:.9e}".format(heat.dielectric), "{:.9e}".format(heat.cathode), "{:.9e}".format(heat.anode),
        "{:.9e}".format(config.simulation.voltage * (mf.current - i_r1)),
        "{:.9f}".format(transport[0]), "{:.9f}".format(T_effective), "{:.9f}".format(T_peak),
        "{:.4f}".format(z_generated.max() if z_generated.size else 0.0),
        "{:.4f}".format(float(np.median(z_generated)) if z_generated.size else 0.0),
        kind, index] + ["{:.6f}".format(v / nm) for v in point] + ["{:.0f}".format(time.time() - start)] + [
        "{:.0f}".format(v) for v in working_set_mb()]
    with open(steps_path, "a", encoding="utf-8") as handle:
        handle.write(",".join(str(v) for v in row) + "\n")
    transport[0] = T_effective
    if n_generated % SNAPSHOT_EVERY == 0:
        np.savez(os.path.join(HERE, "snap_s{}_n{:04d}.npz".format(seed, n_generated)),
                 positions=defects.positions, ED=defects.ED, fixed_charge=defects.fixed_charge, occupancy=p,
                 n_initial=n_initial, birth_time=np.concatenate([np.zeros(n_initial), birth_time]), time=t,
                 transport_temperature=float(row[COLUMNS.index("T_transport_K")]))
    print("{:4d} t={:.4e} J={:.4e} G={:.3e} (cells {:.3e} traps {:.3e}) empty={:.1f} it={} newton={} res={:.1e} "
          "split={} top={:.2f} dT={:.2e} birth={} z={:.3f} wall={:.0f} ws={}/{} MB".format(
              n_generated, t, j * per_cm2, live.sum(), live[:n_cells].sum(), live[n_cells:].sum(),
              float(np.sum(1.0 - p[n_initial:])), mf.iterations, mf.newton_steps, mf.residual, state["clusters"][3],
              z_generated.max() if z_generated.size else 0.0, T_peak - ambient, kind, point[2] / nm,
              time.time() - start, row[-2], row[-1]),
          flush=True)


result = run_tddb(config, progress=progress)
write_tddb_csv(result, os.path.join(HERE, "final_s{}.csv".format(seed)))
np.savez(os.path.join(HERE, "final_s{}.npz".format(seed)), positions=result.defects.positions,
         fixed_charge=result.defects.fixed_charge, birth_time=result.birth_time, occupancy=result.occupancy,
         ED=result.defects.ED, n_initial=result.n_initial, end_time=result.end_time, stop_reason=result.stop_reason)
print("done {} end_time={:.6e} breakdown_time={} wall={:.0f}".format(
    result.stop_reason, result.end_time, result.breakdown_time, time.time() - start), flush=True)
