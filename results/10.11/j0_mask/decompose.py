"""J₀ 为什么一直盖到十几个空位以后（3.6 V / 25 °C，ADR-051 的 50×50 nm 轨迹，簇级平均场）：

A. 单缺陷电流随深度 z 与能级 E_D 的变化，对比初始缺陷（q_fix = 0，空时 +e）与生成空位（带重合的 O_i⁻，
   q_fix = −e，空时中性）。20×20 nm 盒子里只放一个缺陷，电流换算成 50×50 nm 盒子的电流密度。
B. 轨迹快照（第 0/25/50/75 个空位与击穿前最后一条）上按缺陷分解电流：初始缺陷与生成空位各承载多少，
   承载电流的缺陷在什么深度。
usage: decompose.py
"""
import os
import sys
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kmc_leakage.config import GridConfig  # noqa: E402
from kmc_leakage.defects import DefectEnsemble  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.mean_field import solve_mean_field  # noqa: E402

eV, nm, e = constants.electron_volt, constants.nano, constants.elementary_charge
base = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
material = base.device.stack.single_material
band_reference = float(base.device.layer_band_references()[0])
charge_state = base.device.stack.layers[0].defects.charge_state
BOX50 = (50 * nm) ** 2

print("== A. 单缺陷电流 [A/cm²，换算到 50×50 nm 盒子]，行 z [nm]，列 E_D [eV]")
small = replace(base, device=replace(base.device, Lx=20 * nm, Ly=20 * nm),
                simulation=replace(base.simulation, grid=replace(base.simulation.grid, nx=80, ny=80)))
solver = PoissonSolver(small.device, small.simulation.grid)
zs = (0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0)
eds = (1.0, 1.15, 1.3)
for kind, q_fix in (("初始缺陷 q_fix=0", 0.0), ("生成空位 q_fix=−e", -e)):
    print("  " + kind + "：        " + "  ".join("E_D={:.2f}".format(v) for v in eds) + "   | 占据 p（E_D=1.15）")
    for z in zs:
        row, occ = [], None
        for ed in eds:
            d = DefectEnsemble.uniform(np.array([[10 * nm, 10 * nm, z * nm]]), np.array([ed * eV]), material,
                                       band_reference, charge_state=charge_state)
            d = replace(d, fixed_charge=np.full(1, q_fix))
            mf = solve_mean_field(KMCSimulation(small, defects=d, solver=solver))
            row.append(mf.current / BOX50 * 1e-4)
            if ed == 1.15:
                occ = float(mf.occupancy[0])
        print("    z {:.2f}: ".format(z) + "  ".join("{:9.2e}".format(v) for v in row) + "   | {:.3g}".format(occ))

print("\n== B. 快照上按缺陷分解电流（50×50 nm）")
D = os.path.join(ROOT, "results", "10.09", "accept50")
solver50 = PoissonSolver(base.device, base.simulation.grid)
for seed in (1, 2, 3, 4):
    names = ["snap_s{}_n{:04d}.npz".format(seed, n) for n in (0, 25, 50, 75)] + ["final_s{}.npz".format(seed)]
    for name in names:
        path = os.path.join(D, name)
        if not os.path.exists(path):
            continue
        snap = np.load(path)
        n0 = int(snap["n_initial"])
        d = DefectEnsemble.uniform(snap["positions"], snap["ED"], material, band_reference, charge_state=charge_state)
        d = replace(d, fixed_charge=snap["fixed_charge"])
        config = base if "transport_temperature" not in snap.files else replace(
            base, simulation=replace(base.simulation, temperature=float(snap["transport_temperature"])))
        mf = solve_mean_field(KMCSimulation(config, defects=d, solver=solver50), snap["occupancy"])
        J = mf.current_density * 1e-4
        per = mf.by_defect / BOX50 * 1e-4
        z = snap["positions"][:, 2] / nm
        gen = np.arange(per.size) >= n0
        top = np.argsort(per)[::-1][:5]
        n_gen = int(gen.sum())
        zg = z[gen]
        # 生成空位按深度分组承载的电流
        bins = ((0.5, 1.1), (1.1, 1.3), (1.3, 1.5), (1.5, 8.0))
        by_bin = ", ".join("{:.1f}–{:.1f}:{}个/{:.0%}".format(lo, hi, int(((zg >= lo) & (zg < hi)).sum()),
                                                            per[gen][(zg >= lo) & (zg < hi)].sum() / J) for lo, hi in bins)
        print("s{} N={:3d}: J={:.3f} 收敛={} | 初始缺陷 {:.0%}，生成空位 {:.0%} | 生成空位按 z [nm] 分组 {}".format(
            seed, n_gen, J, mf.converged, per[~gen].sum() / J, per[gen].sum() / J, by_bin))
        print("        电流最大的 5 个缺陷：" + "; ".join(
            "{}{} z={:.2f} E_D={:.2f} {:.0%}".format("G" if gen[i] else "I", i, z[i], snap["ED"][i] / eV, per[i] / J)
            for i in top))
