"""一次不动点映射 F 的耗时分解（快照状态）。"""
import cProfile, os, pstats, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import load_state  # noqa
from kmc_leakage import mean_field
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.tddb import IncrementalCoupling
from kmc_leakage.electrostatics import PoissonSolver
seed, n = int(sys.argv[1]), int(sys.argv[2])
config, defects, snap = load_state(seed, n)
solver = PoissonSolver(config.device, config.simulation.grid)
coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
t = time.time(); sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver); print("sim build %.2f s" % (time.time() - t))
p = snap["occupancy"]
mean_field.mean_field_flux(sim, p)
t = time.time()
for _ in range(3):
    mean_field.mean_field_flux(sim, p)
print("F eval %.3f s" % ((time.time() - t) / 3), "pairs", len(sim._pairs), "clusters", [m.size for m in sim.cluster_members])
cProfile.run("mean_field.mean_field_flux(sim, p)", os.path.join(HERE, "prof.out"))
pstats.Stats(os.path.join(HERE, "prof.out")).sort_stats("cumulative").print_stats(18)
