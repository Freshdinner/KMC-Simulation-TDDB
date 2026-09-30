"""Phase 4 validation driver: one TDDB trajectory from examples/zro2_tddb.yaml.

usage: drive.py SEED MODE OUTDIR [COMPLIANCE_A_cm2] [MAX_VAC]
MODE = thermal | iso
"""
import sys, time
sys.path.insert(0, "src")
from dataclasses import replace
import numpy as np
from scipy import constants
from kmc_leakage import tddb as tddb_module
from kmc_leakage.io import load_config
from kmc_leakage.tddb import run_tddb, write_tddb_csv

seed, mode, outdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = load_config(sys.argv[6] if len(sys.argv) > 6 else "examples/zro2_tddb.yaml")
label = sys.argv[7] if len(sys.argv) > 7 else "p4"
tddb = cfg.tddb
if mode == "iso":
    tddb = replace(tddb, thermal_conductivity=None)
if len(sys.argv) > 4:
    tddb = replace(tddb, compliance_current_density=float(sys.argv[4]) * constants.centi ** -2)
if len(sys.argv) > 5:
    tddb = replace(tddb, max_generated_vacancies=int(sys.argv[5]))
cfg = replace(cfg, tddb=tddb, simulation=replace(cfg.simulation, seed=seed))
state = {}
_perc, _temp = tddb_module.percolation_clusters, tddb_module.transport_temperature

def perc(positions, device, distance):
    labels, spanning = _perc(positions, device, distance)
    if distance == tddb.percolation_distance:
        state["perc"] = int(np.bincount(labels).max(initial=0))
        state["zmax"] = 0.0
        if labels.size:
            big = np.argmax(np.bincount(labels))
            state["zmax"] = float(positions[labels == big, 2].max()) * 1e9
    return labels, spanning

def temp(*args):
    value = _temp(*args)
    state["T"] = value
    return value

tddb_module.percolation_clusters = perc
tddb_module.transport_temperature = temp
start = time.time()

def progress(n, t, j):
    print("{:5d} t={:.4e} J={:.4e} perc={} zmax={:.2f} T={:.4f} wall={:.0f}".format(
        n, t, j * 1e-4, state.get("perc", -1), state.get("zmax", -1), state.get("T", float("nan")), time.time() - start), flush=True)

r = run_tddb(cfg, progress=progress)
tag = "{}/{}_{}_s{}".format(outdir, label, mode, seed)
write_tddb_csv(r, tag + ".csv")
np.savez(tag + ".npz", positions=r.defects.positions, birth_time=r.birth_time, occupancy=r.occupancy,
         ED=r.defects.ED, n_initial=r.n_initial, end_time=r.end_time, stop_reason=r.stop_reason)
print("done", r.stop_reason, r.end_time, "wall", time.time() - start, flush=True)
