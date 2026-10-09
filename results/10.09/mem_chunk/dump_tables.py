"""ADR-050 逐位对比用：同一组状态下的生成事件表、落点与短轨迹，改动前后各跑一次。

usage: dump_tables.py TAG        （TAG = before / after）

状态（3.6 V，开热反馈，事件表的温度场取法同 run_tddb）：
- 20×20 nm（examples/zro2_tddb.yaml）seed 1–4 的 t = 0；
- results/10.09/accept/snap_s1_n0025.npz：35 个缺陷（25 个生成空位带 q_fix = −e），禁区球挡住的求积段多；
- 30×30 nm（hx = 0.25 nm）seed 1 的 t = 0。
每个状态存单元事件速率、逐段权重、源陷阱事件速率；按固定种子在速率最高的 30 个单元与随机 30 个单元上各抽一个落点，
再用 _draw_generation 抽 20 次 (Δt, 落点)。轨迹：20 nm seed 1、2 各跑到第 10 个空位，存 CSV 与最终缺陷集合。
"""
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(HERE, "..", "big_box"))

from box import box_config  # noqa: E402
from kmc_leakage.defects import layer_bulk_defects, localization_radius  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402
from kmc_leakage.tddb import _draw_generation, generation_cells, generation_rates, run_tddb, write_tddb_csv  # noqa: E402
from kmc_leakage.thermal import heat_release, power_map, solve_temperature  # noqa: E402

tag = sys.argv[1]
small = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))


def table_of(config, defects=None, start=None):
    inner = KMCSimulation(config, defects=defects)
    occupancy = solve_mean_field(inner, start).occupancy
    tddb = config.tddb
    heat = heat_release(inner, occupancy)
    field = solve_temperature(inner.solver, power_map(inner.solver, inner.defects.positions, heat.defect),
                              tddb.thermal_conductivity, config.simulation.temperature)
    cells = generation_cells(config.device, tddb.generation_grid)
    return generation_rates(inner, occupancy, tddb, cells, field)


states = {}
for seed in (1, 2, 3, 4):
    states["L20_s{}".format(seed)] = (replace(small, simulation=replace(small.simulation, seed=seed)), None, None)
snap = np.load(os.path.join(ROOT, "results", "10.09", "accept", "snap_s1_n0025.npz"))
snap_config = replace(small, simulation=replace(small.simulation, seed=1,
                                                temperature=float(snap["transport_temperature"])))
defects = layer_bulk_defects(snap_config.device, 0, snap["positions"], np.random.default_rng(1))
defects = replace(defects, ED=snap["ED"], r_D=localization_radius(snap["ED"], defects.m_di),
                  fixed_charge=snap["fixed_charge"])
states["snap_s1_n25"] = (snap_config, defects, snap["occupancy"])
big = box_config(30)
states["L30_s1"] = (replace(big, simulation=replace(big.simulation, seed=1)), None, None)

out = {}
for name, (config, defects, start) in states.items():
    table = table_of(config, defects, start)
    cells = table.cells
    out[name + "_cell_rates"] = cells.rates
    out[name + "_weights"] = cells._weights
    out[name + "_trap_rates"] = table.traps.rates
    rng = np.random.default_rng(7)
    positive = np.flatnonzero(cells.rates > 0.0)
    picks = np.concatenate([np.argsort(-cells.rates)[:30], rng.choice(positive, 30, replace=False)])
    points = [cells.sample(int(c), rng) for c in picks]
    out[name + "_cell_samples"] = np.array([np.full(3, np.nan) if p is None else p for p in points])
    rates = np.concatenate([cells.rates, table.traps.rates])
    draws = []
    for _ in range(20):
        dt, position = _draw_generation(rates, np.zeros(rates.size, dtype=bool), table, rng)
        draws.append(np.append(position, dt))
    out[name + "_draws"] = np.array(draws)
    print("{}: ΣG_cells {:.6e}  ΣG_traps {:.6e}  blocked columns {}".format(
        name, cells.rates.sum(), table.traps.rates.sum(), cells._blocked_column.size), flush=True)
np.savez(os.path.join(HERE, "tables_{}.npz".format(tag)), **out)

for seed in (1, 2):
    config = replace(small, simulation=replace(small.simulation, seed=seed),
                     tddb=replace(small.tddb, max_generated_vacancies=10))
    result = run_tddb(config)
    write_tddb_csv(result, os.path.join(HERE, "traj_s{}_{}.csv".format(seed, tag)))
    np.savez(os.path.join(HERE, "traj_s{}_{}.npz".format(seed, tag)), positions=result.defects.positions,
             ED=result.defects.ED, birth_time=result.birth_time, occupancy=result.occupancy, time=result.time,
             current=result.current, generation_rate=result.generation_rate)
    print("trajectory seed {}: {} records, stop {}".format(seed, result.time.size, result.stop_reason), flush=True)
