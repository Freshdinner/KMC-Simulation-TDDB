"""最小间隔 0.42 nm、E_A 高斯分布（1.60 ± 0.15 eV）的 3.6 V 测试：examples/zro2_tddb.yaml 上改四项，其余同 `../../10.19/accept/`（50×50 nm、3.6 V / 25 °C、
开热反馈、击穿后再跑 20 步、停在 3000 个空位或 10⁹ s）：
- `min_separation_nm` 0.26 → 0.42 nm：前驱位点的最大随机堆积间隔、初始缺陷两两之间与到电极界面的最小距离一起改；
- t₀ 0.5 → 2.4 eV：E_D = 1.15 eV 空位的离域距离 r_c = r_D·ln[t₀/(S·ℏω₀)] = 0.68 nm，与最小间隔之比 1.62（同 `../../10.21/sep040/`）；
- E_A：每个前驱位点的无场断键激活能取高斯分布 E_A,i = 1.60 eV + 0.15 eV·Z_i（Z_i 标准正态），替代指数尾（E₀ = 0）。
  在 tddb 命名空间里包 `place_precursors`：先按原函数铺位点，再用同一个随机数发生器给每个位点抽 E_A,i；续跑时位点与 E_A,i 从续跑状态读回。
跑一条轨迹，边跑边写结果。

usage: drive.py SEED MAX_WALL_H [STATE_NPZ]

MAX_WALL_H 是这条轨迹的机时上限 [h]：到点后在下一步开始时存续跑状态并退出（日志末行 "done wall"）。每步开始时还检查整机可用物理内存，
不够下一步用时同样存状态并退出（"done memory"），把内存让给另一条轨迹。给了 STATE_NPZ 就从这个续跑状态
接着跑（`tddb.load_state`，ADR-057），逐步数据写到 `steps_s{SEED}_from{NNNN}.csv`。由 `../../10.24/sigma15/drive.py` 改写（E_A 均值 1.45 → 1.60 eV）：不改 `tddb.run_tddb`，
在 tddb 模块的命名空间里包一层它每步调用的函数，在 `progress` 回调里追加一行 CSV：
- `steps_s{SEED}.csv`：第 k 行是第 k 个空位出生后的稳态（k = 0 为 t = 0），总电流与内层输运 / 间隙隧穿 / 细丝三部分，ΣG 按电极项 / 陷阱项
  拆开（击穿后最后一条不求生成率，记 nan），剩余前驱位点数，平均场与热迭代的收敛诊断，缺陷处最高温度与温度场最高值，导电簇（负极侧 / 正极侧
  成员数、负极侧最高点、正极侧最低点、导体间隙、最大悬空导电簇），这个空位的出生位置与它出生后所在导电簇接的电极（−1 不接、0 负极、1 正极、
  2 贯穿），耗时与工作集；
- `snap_s{SEED}_n{NNNN}.npz`：每 25 个空位（含 0）与有贯穿通路的记录存一次缺陷位置、能级、固定电荷、占据、缺陷处温度、出生时刻与导电簇归属；
- `state_s{SEED}.npz`：每 25 个空位与机时到点时存一次续跑状态（只留最新一份）；
- 正常结束时写 `final_s{SEED}.csv`（`write_tddb_csv`）与 `final_s{SEED}.npz`。
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
from kmc_leakage.tddb import Precursors, conductor_gap, load_state, run_tddb, save_state, write_tddb_csv  # noqa: E402

SNAPSHOT_EVERY = 25
nm = constants.nano
per_cm2 = constants.centi ** 2  # A/m² -> A/cm²
SEPARATION = 0.42 * nm  # min_separation
TRANSFER_INTEGRAL = 2.4 * constants.electron_volt  # t₀
ACTIVATION_ENERGY = 1.60 * constants.electron_volt  # E_A 高斯分布的均值
ACTIVATION_SPREAD = 0.15 * constants.electron_volt  # E_A 高斯分布的标准差
MEMORY_RESERVE_MB = 3072.0  # 内存保护：可用物理内存至少留这么多给系统与另一条轨迹（另一条一步内的增量约 2.6 GB）

seed = int(sys.argv[1])
max_wall = float(sys.argv[2]) * 3600.0
tag = "s{}".format(seed)
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
stack = config.device.stack
layers = tuple(replace(layer, defects=replace(layer.defects, min_separation=SEPARATION)) for layer in stack.layers)
config = replace(config, device=replace(config.device, stack=replace(stack, layers=layers)),
                 simulation=replace(config.simulation, seed=seed),
                 tddb=replace(config.tddb, transfer_integral=TRANSFER_INTEGRAL, activation_energy=ACTIVATION_ENERGY,
                              activation_energy_tail=0.0))
tddb = config.tddb
resume = load_state(sys.argv[3]) if len(sys.argv) > 3 else None
resumed_at = None if resume is None else len(resume.defects) - resume.n_initial
device = config.device
ambient = config.simulation.temperature
area = device.area

commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
print("code {}{} | seed {} | V {} V | T {} K | E_A {:.3f} eV | sigma_EA {:.3f} eV | min_sep {:.2f} nm | t0 {:.2f} eV "
      "| post_breakdown {} | stop {:.3g} s | wall {:.1f} h".format(
    commit, " (src modified)" if dirty else "", seed, config.simulation.voltage, ambient,
    tddb.activation_energy / constants.electron_volt, ACTIVATION_SPREAD / constants.electron_volt,
    config.defects.min_separation / nm, tddb.transfer_integral / constants.electron_volt, tddb.post_breakdown_steps,
    tddb.stop_time, max_wall / 3600.0), flush=True)
if resume is not None:
    print("resume from {} ({} generated, t = {:.6g} s, {} precursors)".format(
        sys.argv[3], resumed_at, resume.time, len(resume.precursors)), flush=True)


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


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_physical_mb():
    """整机可用物理内存 [MB]（Windows）。"""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 2 ** 20


# ---- 包裹 run_tddb 每步调用的函数，抓这一步的状态 ----
state = {}
original = {name: getattr(tddb_module, name) for name in (
    "_steady_state", "generation_rates", "_draw_generation", "place_precursors")}


def place_precursors(placement_device, defects, activation_energy, activation_tail, rng):
    began = time.time()
    positions = original["place_precursors"](placement_device, defects, activation_energy, activation_tail, rng).positions
    energies = activation_energy + ACTIVATION_SPREAD * rng.standard_normal(positions.shape[0])
    print("precursor sites {} ({:.3e} cm-3), placed in {:.0f} s; E_A mean {:.4f} eV, std {:.4f} eV, min {:.4f} eV".format(
        len(positions), len(positions) / (area * (device.Lz - 2.0 * config.defects.min_separation))
        * constants.centi ** 3, time.time() - began, energies.mean() / constants.electron_volt,
        energies.std() / constants.electron_volt, energies.min() / constants.electron_volt), flush=True)
    return Precursors(positions, energies)


def steady_state(*args):
    state["step"] = original["_steady_state"](*args)
    state.pop("generation", None)  # 击穿后最后一条不求生成率
    return state["step"]


def generation_rates(transport, tddb_config, precursors, temperature_field=None):
    began = time.time()
    state["precursors"] = precursors
    state["generation"] = original["generation_rates"](transport, tddb_config, precursors, temperature_field)
    state["rates_wall"] = time.time() - began
    return state["generation"]


def draw_generation(rates, rng):
    event = original["_draw_generation"](rates, rng)
    if event is not None:
        state["birth"] = state["precursors"].positions[event[1]]
    return event


for name, func in (("_steady_state", steady_state), ("generation_rates", generation_rates),
                   ("_draw_generation", draw_generation), ("place_precursors", place_precursors)):
    setattr(tddb_module, name, func)

COLUMNS = (["n_generated", "t_s", "J_A_cm2", "J_transport_A_cm2", "J_gap_A_cm2", "J_filament_A_cm2"]
           + ["J_{}_A_cm2".format(m) for m in MECHANISMS] + [
    "G_total_s", "G_electrode_s", "G_trap_s", "n_precursors", "mf_iterations", "mf_newton_steps", "mf_residual",
    "mf_converged", "thermal_iterations", "thermal_converged", "T_defect_max_K", "T_peak_K", "power_W",
    "n_negative_members", "n_positive_members", "negative_front_nm", "positive_front_nm", "gap_nm",
    "max_floating", "floating_bottom_nm", "floating_top_nm",
    "birth_x_nm", "birth_y_nm", "birth_z_nm", "birth_side", "rates_wall_s", "wall_s", "ws_MB", "ws_peak_MB"])
steps_path = os.path.join(
    HERE, "steps_{}.csv".format(tag) if resume is None else "steps_{}_from{:04d}.csv".format(tag, resumed_at))
with open(steps_path, "w", encoding="utf-8") as handle:
    handle.write(",".join(COLUMNS) + "\n")
birth_time = [] if resume is None else list(resume.birth_time[resume.n_initial:])
start = time.time()


class Stop(Exception):
    """停跑：(原因 wall / memory, 生成空位数, 应力时间)。"""


def save(tddb_state):
    path = os.path.join(HERE, "state_{}.npz".format(tag))
    save_state(tddb_state, path + ".tmp.npz")
    os.replace(path + ".tmp.npz", path)


def checkpoint(tddb_state):
    """每 25 个空位（续跑起点除外）存一次这一步开始时的状态；机时到点，或整机可用物理内存不够下一步用时存状态并停跑。
    下一步要用的内存按这个进程一步内的增量（峰值工作集 − 当前工作集）的 1.25 倍计，另留 MEMORY_RESERVE_MB 给系统与另一条轨迹。
    先写临时文件再替换。"""
    n_generated = len(tddb_state.defects) - tddb_state.n_initial
    if time.time() - start > max_wall:
        save(tddb_state)
        raise Stop("wall", n_generated, tddb_state.time)
    current, peak = working_set_mb()
    if available_physical_mb() < 1.25 * (peak - current) + MEMORY_RESERVE_MB:
        save(tddb_state)
        raise Stop("memory", n_generated, tddb_state.time)
    if n_generated % SNAPSHOT_EVERY or n_generated == resumed_at:
        return
    save(tddb_state)


def progress(n_generated, t, j):
    step = state["step"]
    transport, mf, layout = step.transport, step.transport.result, step.transport.layout
    generation = state.get("generation")
    born = "birth" in state  # 这条记录前刚出生一个空位（t = 0 与续跑起点的记录没有）
    if born:
        birth_time.append(t)
    defects = transport.defects
    n_initial = len(defects) - n_generated
    sides, labels = layout.sides, layout.labels
    z = defects.positions[:, 2] / nm
    negative, positive = (sides == 0) | (sides == 2), (sides == 1) | (sides == 2)
    floating_sizes = np.where(sides < 0, np.bincount(labels)[labels], 0)
    largest = labels[int(np.argmax(floating_sizes))] if floating_sizes.max(initial=0) >= 2 else -1
    floating_z = z[labels == largest] if largest >= 0 else np.zeros(1)
    point = state.pop("birth") if born else np.full(3, np.nan)
    spanning = bool(np.any(sides == 2))
    left = len(state["precursors"]) if generation is not None else -1
    if generation is None:
        G = G_electrode = G_trap = float("nan")
    else:
        G = float(generation.rates.sum())
        G_electrode = float(np.sum(generation.electrode * generation.bond_breaking))
        G_trap = float(np.sum(generation.trap * generation.bond_breaking))
    gap = conductor_gap(defects.positions, sides, device) / nm
    t_defect = float(np.max(step.temperature, initial=ambient))
    t_peak = ambient if step.field is None else float(step.field.max())
    density = lambda current: "{:.9e}".format(current / area * per_cm2)  # noqa: E731
    row = [n_generated, "{:.9e}".format(t), "{:.9e}".format(j * per_cm2), density(mf.current),
           density(step.gap_current), density(step.filament_current)] + [
        density(mf.by_mechanism[m]) for m in MECHANISMS] + [
        "{:.9e}".format(G), "{:.9e}".format(G_electrode), "{:.9e}".format(G_trap), left,
        mf.iterations, mf.newton_steps, "{:.3e}".format(mf.residual), int(mf.converged),
        step.iterations, int(step.converged), "{:.6f}".format(t_defect), "{:.6f}".format(t_peak),
        "{:.6e}".format(float(step.heat.sum())),
        int(negative.sum()), int(positive.sum()), "{:.4f}".format(z[negative].max() if negative.any() else 0.0),
        "{:.4f}".format(z[positive].min() if positive.any() else device.Lz / nm), "{:.4f}".format(gap),
        int(floating_sizes.max(initial=0)), "{:.4f}".format(floating_z.min()), "{:.4f}".format(floating_z.max())] + [
        "{:.6f}".format(v / nm) for v in point] + [int(sides[-1]) if born else -1] + [
        "{:.1f}".format(state.get("rates_wall", float("nan")) if generation is not None else float("nan")),
        "{:.0f}".format(time.time() - start)] + ["{:.0f}".format(v) for v in working_set_mb()]
    with open(steps_path, "a", encoding="utf-8") as handle:
        handle.write(",".join(str(v) for v in row) + "\n")
    if n_generated % SNAPSHOT_EVERY == 0 or spanning:
        np.savez(os.path.join(HERE, "snap_{}_n{:04d}.npz".format(tag, n_generated)),
                 positions=defects.positions, ED=defects.ED, fixed_charge=defects.fixed_charge,
                 occupancy=transport.occupancy, temperature=step.temperature, n_initial=n_initial,
                 birth_time=np.concatenate([np.zeros(n_initial), birth_time]), time=t, sides=sides)
    print("{:4d} t={:.4e} J={:.4e} (TAT {:.4e} gap {:.3e} fil {:.3e}) G={:.3e} left={} it={} newton={} res={:.1e} "
          "therm={}{} Tdef={:.3f} Tpk={:.3f} neg={} front={:.2f} pos={} gap={:.2f} float={} z={:.3f} side={} "
          "rates={}s wall={:.0f} ws={}/{} MB".format(
              n_generated, t, j * per_cm2, mf.current / area * per_cm2, step.gap_current / area * per_cm2,
              step.filament_current / area * per_cm2, G, left, mf.iterations, mf.newton_steps, mf.residual,
              step.iterations, "" if step.converged else "!", t_defect, t_peak, int(negative.sum()),
              z[negative].max() if negative.any() else 0.0, int(positive.sum()), gap, int(floating_sizes.max(initial=0)),
              point[2] / nm, int(sides[-1]) if born else -1, row[-4], time.time() - start, row[-2], row[-1]),
          flush=True)


try:
    result = run_tddb(config, progress=progress, start=resume, checkpoint=checkpoint)
except Stop as stop:
    print("done {} n_generated={} t={:.6e} wall={:.0f} (state saved)".format(*stop.args, time.time() - start), flush=True)
    sys.exit(0)
write_tddb_csv(result, os.path.join(HERE, "final_{}.csv".format(tag)))
np.savez(os.path.join(HERE, "final_{}.npz".format(tag)), positions=result.defects.positions,
         fixed_charge=result.defects.fixed_charge, birth_time=result.birth_time, occupancy=result.occupancy,
         ED=result.defects.ED, temperature=result.temperature_at_defects, n_initial=result.n_initial,
         end_time=result.end_time, stop_reason=result.stop_reason)
print("done {} end_time={:.6e} breakdown_time={} wall={:.0f}".format(
    result.stop_reason, result.end_time, result.breakdown_time, time.time() - start), flush=True)
