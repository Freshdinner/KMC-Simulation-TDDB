"""在 ADR-046 what-if（O_i⁻ 与空位重合，生成空位 Z = 0）不收敛的快照上复现平均场迭代，逐次打印残差。

usage: repro.py SEED N [start]    start = snap（快照里的末次迭代占据，缺省）| zeros | ones_new
快照来自 results/10.03/oi_colocated/snap_s{SEED}_n{NNNN}.npz（3.0 V、该快照的输运温度）。
"""
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import mean_field  # noqa: E402
from kmc_leakage.defects import append_defects, generate_defects, layer_bulk_defects, localization_radius  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.kmc import KMCSimulation  # noqa: E402
from kmc_leakage.tddb import IncrementalCoupling  # noqa: E402
from kmc_leakage.electrostatics import PoissonSolver  # noqa: E402


def load_state(seed, n, voltage=3.0):
    snap = np.load(os.path.join(ROOT, "results", "10.03", "oi_colocated", "snap_s{}_n{:04d}.npz".format(seed, n)))
    config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
    sim = replace(config.simulation, seed=seed, voltage=voltage, temperature=float(snap["transport_temperature"]))
    config = replace(config, simulation=sim)
    device = config.device
    rng = np.random.default_rng(seed)
    initial = generate_defects(device, rng)
    n_initial = int(snap["n_initial"])
    assert len(initial) == n_initial
    assert np.array_equal(initial.positions, snap["positions"][:n_initial])
    generated = layer_bulk_defects(device, 0, snap["positions"][n_initial:], rng)
    ED = snap["ED"][n_initial:]
    generated = replace(generated, ED=ED, r_D=localization_radius(ED, generated.m_di),
                        charge_state=np.zeros(len(generated)))
    defects = append_defects(initial, generated)
    assert np.array_equal(defects.ED, snap["ED"])
    return config, defects, snap


def solve_logged(sim, p0, max_iter=200, log=True):
    """mean_field.solve_mean_field 的同一套迭代，逐次记 (残差, 残差最大的陷阱)。"""
    n = sim.n_defects
    p = np.clip(p0, 0.0, 1.0)
    xs, gs, history = [], [], []
    for it in range(1, max_iter + 1):
        _, _, p_hat = mean_field.mean_field_flux(sim, p)
        x_hat = mean_field._log_occupancy(p_hat)
        if it == 1:
            x = mean_field._log_occupancy(p)
            g = x_hat - x
            res = float(np.max(np.abs(g)))
            history.append((res, int(np.argmax(np.abs(g)))))
            x = x_hat
        else:
            g = x_hat - x
            res = float(np.max(np.abs(g)))
            history.append((res, int(np.argmax(np.abs(g)))))
            if res <= mean_field._TOL:
                break
            xs.append(x)
            gs.append(g)
            if len(xs) > mean_field._HISTORY + 1:
                xs.pop(0)
                gs.pop(0)
            step = x + g
            if len(xs) > 1:
                dx = np.diff(np.array(xs), axis=0).T
                dg = np.diff(np.array(gs), axis=0).T
                gamma = np.linalg.lstsq(dg, g, rcond=None)[0]
                step = step - (dx + dg) @ gamma
            x = np.clip(step, mean_field._LOG_FLOOR, 0.0)
        p = np.exp(x)
        if log:
            print("{:4d} res={:.3e} worst={}".format(it, *history[-1]), flush=True)
    return p, history


if __name__ == "__main__":
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    start = sys.argv[3] if len(sys.argv) > 3 else "snap"
    config, defects, snap = load_state(seed, n)
    solver = PoissonSolver(config.device, config.simulation.grid)
    coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    sim = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
    p0 = {"snap": snap["occupancy"], "zeros": np.zeros(len(defects))}[start]
    result = mean_field.solve_mean_field(sim, p0)
    print("solve_mean_field: converged={} it={} res={:.3e} J={:.4e} A/cm2".format(
        result.converged, result.iterations, result.residual, result.current_density * 1e-4))
    np.save(os.path.join(HERE, "p_s{}_n{}_{}.npy".format(seed, n, start)), result.occupancy)
