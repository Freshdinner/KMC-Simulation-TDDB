"""把 ADR-056 验收停跑前的快照转成续跑状态（`tddb.TDDBState`，ADR-057）。

usage: make_state.py SEED R_C_NM N

读 `snap_s{SEED}_rc{RC}_n{N:04d}.npz` 与 `steps_s{SEED}_rc{RC}.csv`，写 `state_s{SEED}_rc{RC}_n{N:04d}.npz`（`tddb.save_state`）：
- 初始缺陷与前驱位点：`tddb.initial_state` 按同一种子重新生成，核对初始缺陷位置与快照逐位相同；前驱位点去掉已出生空位的位置；
- 生成空位：按快照的位置、能级与 O_i⁻ 固定电荷放回（局域化半径按能级重算，其余物性同 `layer_bulk_defects`）；
- 出生时刻、时刻取快照；平均场热启动取快照里第 N 条记录已收敛的占据；输运温度取 steps 第 N − 1 行的 T_effective_K（第 N 条记录
  用的就是它）；
- 随机数流：这批快照没存随机数状态，换一个新流 `default_rng([SEED, N])`。续跑的轨迹因此与停跑前那条不同，但从同一个状态出发、
  统计上等价。
"""
import csv
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.defects import append_defects, layer_bulk_defects, localization_radius  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.tddb import Precursors, TDDBState, initial_state, save_state  # noqa: E402

seed, r_c_nm, n = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
tag = "s{}_rc{:03d}".format(seed, int(round(100 * r_c_nm)))
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, seed=seed),
                 tddb=replace(config.tddb, percolation_distance=r_c_nm * constants.nano))
device = config.device
snap = np.load(os.path.join(HERE, "snap_{}_n{:04d}.npz".format(tag, n)))
with open(os.path.join(HERE, "steps_{}.csv".format(tag)), encoding="utf-8") as handle:
    rows = {int(row["n_generated"]): row for row in csv.DictReader(handle)}

first = initial_state(config)
n0 = first.n_initial
assert int(snap["n_initial"]) == n0 and np.array_equal(first.defects.positions, snap["positions"][:n0])
born = snap["positions"][n0:]
assert born.shape[0] == n
index = {tuple(p): k for k, p in enumerate(first.precursors.positions)}
keep = np.ones(len(first.precursors), dtype=bool)
keep[[index[tuple(p)] for p in born]] = False  # 每个出生位置都必须是一个前驱位点

material = device.stack.single_material
vacancies = layer_bulk_defects(device, 0, born, np.random.default_rng(0))
vacancies = replace(vacancies, ED=snap["ED"][n0:], r_D=localization_radius(snap["ED"][n0:], material.m_di),
                    fixed_charge=snap["fixed_charge"][n0:])
defects = append_defects(first.defects, vacancies)
assert np.array_equal(defects.positions, snap["positions"]) and np.array_equal(defects.ED, snap["ED"])
assert np.array_equal(defects.fixed_charge, snap["fixed_charge"])

state = TDDBState(
    defects=defects,
    n_initial=n0,
    birth_time=snap["birth_time"],
    time=float(snap["time"]),
    occupancy=snap["occupancy"],
    precursors=Precursors(first.precursors.positions[keep]),
    transport_temperature=float(rows[n - 1]["T_effective_K"]),
    rng=np.random.default_rng([seed, n]),
)
assert state.time == float(rows[n]["t_s"]) or abs(state.time / float(rows[n]["t_s"]) - 1.0) < 1e-8
path = os.path.join(HERE, "state_{}_n{:04d}.npz".format(tag, n))
save_state(state, path)
print("{}: {} defects ({} generated), {} precursors, t = {:.6g} s, T_transport = {:.6f} K -> {}".format(
    tag, len(defects), n, len(state.precursors), state.time, state.transport_temperature, os.path.basename(path)))
