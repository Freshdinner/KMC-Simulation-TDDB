"""ADR-046：ADR-045 的轨迹延长到 1000 个空位（examples/zro2_tddb.yaml，开热反馈），边跑边写结果。

由 `results/10.03/rerun/drive.py` 复制而来，只多记一列进程工作集（`ws_MB`，Windows），其余逐字相同；
轨迹逐种子确定，前 400 步应与 `results/10.03/rerun/steps_s*.csv` 逐步相同。

usage: drive.py SEED [MAX_VAC] [COMPLIANCE_A_cm2]
（缺省：空位上限 1000、限流值 1e8 A/cm²——让轨迹只由渗流判据或空位上限停下）

不改 `tddb.run_tddb`：在 tddb 模块的命名空间里包一层它每步调用的函数（平均场、放热、温度、事件表、分簇诊断、
渗流团簇、抽生成事件），抓到这一步的完整状态，在 `progress` 回调里追加一行 CSV：
- `steps_s{SEED}.csv`：每求完一个状态写一行，第 k 行是第 k 个空位出生后的稳态（k = 0 为 t = 0），
  含这个空位的出生位置与事件类型（cell = 单元电极项，trap = 源陷阱项），ΣG 按单元事件 / 源陷阱事件拆开；
- `snap_s{SEED}_n{NNNN}.npz`：每 50 个空位（含 0）存一次缺陷位置、能级、平均场占据、出生时刻、输运温度与
  已满事件标记，供 `check_states.py` 在轨迹上重建状态做数值抽查；
- 结束时写完整的 `final_s{SEED}.csv`（`write_tddb_csv`）与 `final_s{SEED}.npz`。
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

SNAPSHOT_EVERY = 50
nm = constants.nano
per_cm2 = constants.centi ** 2  # A/m² -> A/cm²

seed = int(sys.argv[1])
max_vacancies = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
compliance = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0e8
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
tddb = replace(config.tddb, max_generated_vacancies=max_vacancies,
               compliance_current_density=compliance / per_cm2)
config = replace(config, tddb=tddb, simulation=replace(config.simulation, seed=seed))
device, ambient = config.device, config.simulation.temperature
if tddb.thermal_conductivity is None:
    raise SystemExit("zro2_tddb.yaml 里没有 k_TH，热反馈没开")

commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
print("code {}{} | seed {} | max_vacancies {} | compliance {:.1e} A/cm2 | k_TH {} W/(m K) | V {} V | T {} K | "
      "generation grid {} | sigma {:.2f} nm".format(
          commit, " (src modified)" if dirty else "", seed, max_vacancies, compliance, tddb.thermal_conductivity,
          config.simulation.voltage, ambient, tddb.generation_grid, tddb.field_averaging_width / nm), flush=True)

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
    "_cluster_diagnostics", "percolation_clusters", "_draw_generation")}


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
    # 上一步的包装 sample 引用着自己所在实例的绑定方法，构成引用环；不拆开的话整张事件表（致密层上约 140 MB）
    # 要等全代 GC 才释放，进程工作集每步涨一张表（ADR-045 的 1.9–2.4 GB 来自这里）
    if "table" in state:
        for events in state.pop("table"):
            del events.sample
    table = original["generation_rates"](*args)
    # 记下这一步抽中的是哪类事件：实例上的 sample 覆盖类方法，`_draw_generation` 调的就是它
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


def percolation_clusters(positions, device_, distance):
    labels, spanning = original["percolation_clusters"](positions, device_, distance)
    if distance == tddb.percolation_distance:
        state["perc"] = (positions, labels)
    else:
        state["dense"] = int(np.bincount(labels).max(initial=0))
    return labels, spanning


def draw_generation(rates, full, table, rng):
    # run_tddb 的 rates 已按已满标记置零；它应与 progress 里按镜像标记重算的那份逐位相同
    if not np.array_equal(rates, state["live"]):
        print("WARNING full-mask mirror differs from run_tddb", flush=True)
    event = original["_draw_generation"](rates, full, table, rng)
    state["full"] = full  # _draw_generation 原地改 full；run_tddb 之后 append 一个 False
    return event


for name, func in (("solve_mean_field", solve_mean_field), ("heat_release", heat_release),
                   ("solve_temperature", solve_temperature), ("transport_temperature", transport_temperature),
                   ("generation_rates", generation_rates), ("_cluster_diagnostics", cluster_diagnostics),
                   ("percolation_clusters", percolation_clusters), ("_draw_generation", draw_generation)):
    setattr(tddb_module, name, func)

COLUMNS = (["n_generated", "t_s", "J_A_cm2"] + ["J_{}_A_cm2".format(m) for m in MECHANISMS] + [
    "G_total_s", "G_cells_s", "G_traps_s", "n_full_events", "space_charge_C", "empty_generated", "empty_all",
    "mf_iterations", "mf_residual", "mf_converged", "n_clusters", "max_cluster_size", "largest_component", "n_split",
    "P_dielectric_W", "P_cathode_W", "P_anode_W", "UI_trap_W", "T_transport_K", "T_effective_K", "T_peak_K",
    "percolation_size", "percolation_top_nm", "cathode_cluster_top_nm", "dense_cluster_size",
    "birth_event", "birth_index", "birth_x_nm", "birth_y_nm", "birth_z_nm", "wall_s", "ws_MB", "ws_peak_MB"])
stem = os.path.join(os.environ.get("TDDB_OUT_DIR", HERE), "{}_s{}")  # what-if 包装脚本把结果写到自己的目录
steps_path = stem.format("steps", seed) + ".csv"
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
    state["live"] = live
    if n_generated > 0:
        birth_time.append(t)
    defects, p = inner.defects, mf.occupancy
    n_initial = len(defects) - n_generated
    positions, labels = state["perc"]
    sizes = np.bincount(labels)
    big = int(np.argmax(sizes))
    cathode = np.unique(labels[positions[:, 2] < tddb.percolation_distance])
    touching = np.isin(labels, cathode)
    i_r1 = mf.by_mechanism["direct_fn"] + mf.by_mechanism["thermionic"]
    kind, index, point = state.pop("birth") if n_generated > 0 else ("none", -1, np.full(3, np.nan))
    T_effective = state.get("T_effective", ambient)
    T_peak = float(state["T_field"].max()) if "T_field" in state else ambient
    row = [n_generated, "{:.9e}".format(t), "{:.9e}".format(j * per_cm2)] + [
        "{:.9e}".format(mf.by_mechanism[m] / mf.area * per_cm2) for m in MECHANISMS] + [
        "{:.9e}".format(float(live.sum())), "{:.9e}".format(float(live[:n_cells].sum())),
        "{:.9e}".format(float(live[n_cells:].sum())), int(mirror.sum()),
        "{:.9e}".format(constants.elementary_charge * float(np.sum(defects.charge_state - p))),
        "{:.6g}".format(float(np.sum(1.0 - p[n_initial:]))), "{:.6g}".format(float(np.sum(1.0 - p))),
        mf.iterations, "{:.3e}".format(mf.residual), int(mf.converged), *state["clusters"],
        "{:.9e}".format(heat.dielectric), "{:.9e}".format(heat.cathode), "{:.9e}".format(heat.anode),
        "{:.9e}".format(config.simulation.voltage * (mf.current - i_r1)),
        "{:.9f}".format(transport[0]), "{:.9f}".format(T_effective), "{:.9f}".format(T_peak),
        int(sizes[big]), "{:.4f}".format(positions[labels == big, 2].max() / nm),
        "{:.4f}".format(positions[touching, 2].max() / nm if touching.any() else 0.0), state["dense"],
        kind, index] + ["{:.6f}".format(v / nm) for v in point] + ["{:.0f}".format(time.time() - start)] + [
        "{:.0f}".format(v) for v in working_set_mb()]
    with open(steps_path, "a", encoding="utf-8") as handle:
        handle.write(",".join(str(v) for v in row) + "\n")
    transport[0] = T_effective
    if n_generated % SNAPSHOT_EVERY == 0:
        np.savez(stem.format("snap", seed) + "_n{:04d}.npz".format(n_generated),
                 positions=defects.positions, ED=defects.ED, occupancy=p, n_initial=n_initial,
                 birth_time=np.concatenate([np.zeros(n_initial), birth_time]), time=t,
                 transport_temperature=float(row[COLUMNS.index("T_transport_K")]), full=mirror, n_cells=n_cells)
    print("{:5d} t={:.4e} J={:.4e} G={:.3e} (cells {:.3e} traps {:.3e}) empty={:.1f} it={} res={:.1e} split={} "
          "perc={} top={:.2f} dT={:.2e} birth={} z={:.3f} wall={:.0f} ws={}/{} MB".format(
              n_generated, t, j * per_cm2, live.sum(), live[:n_cells].sum(), live[n_cells:].sum(),
              float(np.sum(1.0 - p[n_initial:])), mf.iterations, mf.residual, state["clusters"][3], sizes[big],
              positions[labels == big, 2].max() / nm, T_peak - ambient, kind, point[2] / nm, time.time() - start,
              row[-2], row[-1]),
          flush=True)


result = run_tddb(config, progress=progress)
write_tddb_csv(result, stem.format("final", seed) + ".csv")
np.savez(stem.format("final", seed) + ".npz", positions=result.defects.positions, birth_time=result.birth_time,
         occupancy=result.occupancy, ED=result.defects.ED, n_initial=result.n_initial, end_time=result.end_time,
         stop_reason=result.stop_reason)
print("done {} end_time={:.6e} wall={:.0f}".format(result.stop_reason, result.end_time, time.time() - start), flush=True)
