"""单个种子的 t = 0 平均场在各盒子下的耗时与内存（决定 j0_box_size.py 的种子数）。"""
import time
from dataclasses import replace

from scipy import constants

from box import box_config
from kmc_leakage.defects import defect_count
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.mean_field import solve_mean_field
from memory import working_set_mb

for side in (20, 50):
    config = box_config(side)
    start = time.time()
    solver = PoissonSolver(config.device, config.simulation.grid)
    t_solver = time.time() - start
    for seed in (1, 2):
        start = time.time()
        sim = KMCSimulation(replace(config, simulation=replace(config.simulation, seed=seed)), solver=solver)
        t_sim = time.time() - start
        mf = solve_mean_field(sim)
        print("{} nm: N={} (expected {}) solver {:.1f} s, sim {:.1f} s, mf {:.1f} s, J={:.3g} A/cm2, ws {:.0f}/{:.0f} MB".format(
            side, len(sim.defects), defect_count(config.device, config.defects), t_solver, t_sim,
            time.time() - start - t_sim, mf.current_density * constants.centi ** 2, *working_set_mb()), flush=True)
