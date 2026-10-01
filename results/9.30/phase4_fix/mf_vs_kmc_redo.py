"""ADR-031 的平均场 / KMC 对拍在新平均场（ADR-038）下重做：KMC 逐位不变，直接取
results/9.29/tddb_phase2_mf_vs_kmc.csv 里的 KMC 值，只在同一批实现上重算簇级平均场。

usage: mf_vs_kmc_redo.py CASE OUT.csv [SEED_START SEED_STOP]
CASE = box20_3e18（N=10，六个工况）| box20_3e19（N=96，3 V / 25 °C）
重建口径同 ADR-031：examples/zro2_tddb.yaml，grid 13/13/80（ADR-025，ADR-032 之前），只改浓度、工况与种子。
代码版本由 PYTHONPATH 决定：HEAD 归档下应逐位复现 CSV 里的旧平均场值。
"""
import csv
import sys
import time
from dataclasses import replace

from scipy import constants

from kmc_leakage.config import GridConfig
from kmc_leakage.io import load_config
from kmc_leakage.kmc import KMCSimulation
from kmc_leakage.mean_field import solve_mean_field

case, out = sys.argv[1], sys.argv[2]
seed_range = range(int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else None
concentration = {"box20_3e18": 3.0e18, "box20_3e19": 3.0e19}[case]
rows = [r for r in csv.DictReader(open("results/9.29/tddb_phase2_mf_vs_kmc.csv", encoding="utf-8"))
        if r["case"] == case and r["solver"] == "mf"]
base = load_config("examples/zro2_tddb.yaml")
layer = base.device.stack.layers[0]
layer = replace(layer, defects=replace(layer.defects, concentration=concentration * constants.centi ** -3))
device = replace(base.device, stack=replace(base.device.stack, layers=(layer,)))

with open(out, "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["case", "seed", "voltage_V", "temperature_C", "n_defects", "J_mf_csv", "J_mf", "J_tat", "J_pf",
                     "iterations", "converged", "runtime_s"])
    for r in rows:
        seed = int(r["seed"])
        if seed_range is not None and seed not in seed_range:
            continue
        start = time.time()
        simulation = replace(base.simulation, voltage=float(r["voltage_V"]),
                             temperature=float(r["temperature_C"]) + constants.zero_Celsius,
                             seed=seed, grid=GridConfig(13, 13, 80))
        sim = KMCSimulation(replace(base, device=device, simulation=simulation))
        mf = solve_mean_field(sim)
        per_cm2 = constants.centi ** 2
        writer.writerow([case, seed, r["voltage_V"], r["temperature_C"], sim.n_defects, r["J_mf_csv"] if "J_mf_csv" in r else r["J_A_cm2"],
                         repr(mf.current_density * per_cm2), repr(mf.by_mechanism["tat"] / mf.area * per_cm2),
                         repr(mf.by_mechanism["pf"] / mf.area * per_cm2), mf.iterations, int(mf.converged),
                         round(time.time() - start, 2)])
        handle.flush()
print("done", case, flush=True)
