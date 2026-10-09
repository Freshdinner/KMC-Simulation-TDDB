"""L×L×8 nm 盒子里 TDDB 外层循环每步的耗时与内存：seed 1，3.6 V / 25 °C，开热反馈，跑到第 K 个空位为止
（tddb.run_tddb 原样调用，只把 max_generated_vacancies 设成 K）。

usage: step_cost.py L_NM K

每求完一个状态打印一行：已生成空位数、t、J、距上一行的墙钟时间、当前/峰值工作集。
"""
import sys
import time
from dataclasses import replace

from scipy import constants

from box import box_config
from kmc_leakage.tddb import run_tddb
from memory import working_set_mb

side, steps = float(sys.argv[1]), int(sys.argv[2])
config = box_config(side)
config = replace(config, simulation=replace(config.simulation, seed=1),
                 tddb=replace(config.tddb, max_generated_vacancies=steps))
print("{:g}x{:g} nm: grid {} | generation grid {}".format(
    side, side, config.simulation.grid, config.tddb.generation_grid), flush=True)
last = [time.time()]


def progress(n_generated, t, j):
    now = time.time()
    print("{:3d} t={:.4e} s J={:.4e} A/cm2 | step wall {:.1f} s | ws {:.0f}/{:.0f} MB".format(
        n_generated, t, j * constants.centi ** 2, now - last[0], *working_set_mb()), flush=True)
    last[0] = now


result = run_tddb(config, progress=progress)
print("done {} | peak ws {:.0f} MB".format(result.stop_reason, working_set_mb()[1]), flush=True)
