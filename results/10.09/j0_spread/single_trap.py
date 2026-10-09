"""3.6 V / 25 °C、20×20×8 nm 盒子里只放一个缺陷：它承载的电流随 z 与 E_D 怎么变（簇级平均场）。"""
import os, sys
from dataclasses import replace
import numpy as np
from scipy import constants
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kmc_leakage.defects import DefectEnsemble
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.io import load_config
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.mean_field import solve_mean_field
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
config = replace(config, simulation=replace(config.simulation, voltage=3.6))
device = config.device
solver = PoissonSolver(device, config.simulation.grid)
material = device.stack.single_material
band_reference = float(device.layer_band_references()[0])
eV, nm = constants.electron_volt, constants.nano
zs = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
eds = [0.85, 1.0, 1.15, 1.3, 1.45]
print("J [A/cm²] of a single trap; rows z [nm], columns E_D [eV]:", eds)
for z in zs:
    row = []
    for ed in eds:
        d = DefectEnsemble.uniform(np.array([[10 * nm, 10 * nm, z * nm]]), np.array([ed * eV]), material, band_reference)
        mf = solve_mean_field(KMCSimulation(config, defects=d, solver=solver))
        row.append((mf.by_mechanism["tat"] + mf.by_mechanism["pf"]) / mf.area * 1e-4)
    print("z %.2f: " % z + "  ".join("%9.2e" % v for v in row))
