"""t = 0 的电极项生成被积函数（不含 N_T）按节点取出，存成"单个前驱的生成速率谱"：
每个求积节点 (z 层, 列) 的 ln R_1e 与 P_bb 指数 g = E_eff/k_BT（E_A 取配置均值），以及节点代表的体积。
供 depletion.py 估算"前驱消耗 + E_A 分布"下的 N(t)。默认 50×50 nm、3.6 V、seed 1。
读的是模型 C 的单元积分内部量（`ElectrodeCaptureGeneration`），要在 ADR-052 之前的代码（735736c）上运行；输出的
spectrum_s{SEED}.npz 约 150 MB，不进版本库，重跑约 1 min。
usage: rate_spectrum.py [SEED]
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kmc_leakage.defects import generate_defects  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402
from kmc_leakage.tddb import IncrementalCoupling, generation_cells, generation_rates  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, seed=seed))
device, tddb = config.device, config.tddb
t0 = time.time()
rng = np.random.default_rng(seed)
defects = generate_defects(device, rng)
solver = PoissonSolver(device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
inner = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
mf = solve_mean_field(inner, None)
table = generation_rates(inner, mf.occupancy, tddb, generation_cells(device, tddb.generation_grid))
cells = table.cells
print("J0 = {:.3g} A/cm2, sum G_cells = {:.4g} 1/s, sum G_traps = {:.4g} 1/s, {:.0f} s".format(
    mf.current_density * 1e-4, cells.rates.sum(), table.traps.rates.sum(), time.time() - t0))
np.savez_compressed(
    os.path.join(HERE, "spectrum_s{}.npz".format(seed)),
    levels=cells._levels, log_rate=cells._log_rate.astype(np.float32), barrier=cells._barrier.astype(np.float32),
    dz=cells._dz, column_area=np.prod(cells._column_size), n_columns=cells._columns.shape[0],
    weights_sum=cells._weights.sum(), N_T=tddb.precursor_concentration, area=device.Lx * device.Ly,
    E_A=tddb.activation_energy, T=config.simulation.temperature, J0=mf.current_density,
)
