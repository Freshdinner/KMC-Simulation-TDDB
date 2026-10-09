"""TDDB 外层一步的内存去向：L×L×8 nm 盒子（box.box_config），seed 1，3.6 V / 25 °C，开热反馈，跑到第 K 个空位。

usage: profile_step.py L_NM K

不改 src：在 tddb / electrostatics 的命名空间里给每步调用的函数包一层，记录
- 每个函数（按调用嵌套，取最内层）执行期间工作集的最大值：采样线程每 1 ms 读一次；
- 每次调用前后进程峰值工作集的变化：峰值被哪一次调用抬高、抬到多少（精确，不靠采样）；
- 每个状态求完后事件表常驻数组的字节数（电极项、陷阱项分开），以及调用前的工作集（上一步留下的部分）。
"""
import sys
import threading
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np

from box import box_config
from kmc_leakage import electrostatics, tddb as tddb_module
from kmc_leakage.tddb import ElectrodeCaptureGeneration, TrapCaptureGeneration, run_tddb
from memory import working_set_mb

side, steps = float(sys.argv[1]), int(sys.argv[2])
config = box_config(side)
config = replace(config, simulation=replace(config.simulation, seed=1),
                 tddb=replace(config.tddb, max_generated_vacancies=steps))

stack = ["(外层)"]
state = {"step": 0}
sampled = defaultdict(float)  # (步, 最内层函数) -> 采样到的最大工作集 [MB]
raised = []  # (步, 函数调用路径, 调用前峰值, 调用后峰值)
calls = defaultdict(lambda: [0, 0.0])  # (步, 函数) -> [次数, 墙钟 s]
entry = {}  # (步, 函数) -> 第一次调用时的工作集 [MB]
running = [True]


def sampler():
    while running[0]:
        ws = working_set_mb()[0]
        key = (state["step"], stack[-1])
        if ws > sampled[key]:
            sampled[key] = ws
        time.sleep(0.001)


def wrap(owner, name, label):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        key = (state["step"], label)
        ws, peak_before = working_set_mb()
        entry.setdefault(key, ws)
        stack.append(label)
        start = time.time()
        try:
            return original(*args, **kwargs)
        finally:
            calls[key][0] += 1
            calls[key][1] += time.time() - start
            peak_after = working_set_mb()[1]
            if peak_after > peak_before + 1.0:
                raised.append((state["step"], " > ".join(stack[1:]), peak_before, peak_after))
            stack.pop()

    setattr(owner, name, wrapper)


for owner, name, label in (
    (tddb_module, "KMCSimulation", "KMCSimulation"),
    (tddb_module, "solve_mean_field", "solve_mean_field"),
    (tddb_module, "heat_release", "heat_release"),
    (tddb_module, "power_map", "power_map"),
    (tddb_module, "solve_temperature", "solve_temperature"),
    (tddb_module, "transport_temperature", "transport_temperature"),
    (tddb_module, "generation_rates", "generation_rates"),
    (electrostatics.PoissonSolver, "gaussian_average", "gaussian_average"),
    (ElectrodeCaptureGeneration, "__init__", "电极项"),
    (ElectrodeCaptureGeneration, "_nodes", "电极项._nodes"),
    (ElectrodeCaptureGeneration, "_blocked_intervals", "电极项._blocked_intervals"),
    (ElectrodeCaptureGeneration, "_excluded", "电极项._excluded"),
    (tddb_module, "_kinked_log_linear_integral", "_kinked_log_linear_integral"),
    (TrapCaptureGeneration, "__init__", "陷阱项"),
    (tddb_module, "_cluster_diagnostics", "_cluster_diagnostics"),
    (tddb_module, "distance_clusters", "distance_clusters"),
    (tddb_module, "_draw_generation", "_draw_generation"),
    (tddb_module.IncrementalCoupling, "extended", "coupling.extended"),
):
    wrap(owner, name, label)

tables = []
original_rates = tddb_module.generation_rates


def keep_table(*args, **kwargs):
    table = original_rates(*args, **kwargs)
    tables.append(table)
    return table


tddb_module.generation_rates = keep_table


def array_bytes(obj):
    """对象属性里 ndarray 的字节数（含 dict / tuple 里的数组），按属性名分开。"""
    out = {}
    for name, value in vars(obj).items():
        items = value.values() if isinstance(value, dict) else [value]
        total = 0
        for item in items:
            for v in (item if isinstance(item, tuple) else (item,)):
                if isinstance(v, np.ndarray):
                    total += v.nbytes
        if total:
            out[name] = total
    return out


def progress(n_generated, t, j):
    table = tables[-1]
    tables.clear()
    for label, part in (("电极项", table.cells), ("陷阱项", table.traps)):
        sizes = array_bytes(part)
        top = sorted(sizes.items(), key=lambda kv: -kv[1])[:6]
        print("  [{}] {} 常驻数组 {:.0f} MB：{}".format(n_generated, label, sum(sizes.values()) / 2 ** 20,
              ", ".join("{} {:.0f}".format(k, v / 2 ** 20) for k, v in top)), flush=True)
    state["step"] += 1


thread = threading.Thread(target=sampler, daemon=True)
thread.start()
start = time.time()
result = run_tddb(config, progress=progress)
running[0] = False
thread.join()

print("\n{:g}x{:g} nm，泊松网格 {}，生成网格 {}，总墙钟 {:.0f} s，峰值工作集 {:.0f} MB".format(
    side, side, config.simulation.grid, config.tddb.generation_grid, time.time() - start, working_set_mb()[1]))
print("\n各函数（最内层）执行期间采样到的最大工作集 / 第一次调用时的工作集 [MB]，次数，墙钟 [s]")
for step in range(state["step"]):
    print("-- 状态 {}".format(step))
    for (s, label), ws in sorted(sampled.items(), key=lambda kv: -kv[1]):
        if s == step:
            n, wall = calls.get((s, label), (0, 0.0))
            print("   {:32s} 最大 {:6.0f}  进入时 {:6.0f}  次数 {:3d}  墙钟 {:6.1f}".format(
                label, ws, entry.get((s, label), float("nan")), n, wall))
print("\n抬高进程峰值的调用（状态, 调用路径, 调用前峰值 → 调用后峰值 [MB]）")
for step, path, before, after in raised:
    print("   {}  {:70s} {:6.0f} → {:6.0f}".format(step, path, before, after))
