"""ADR-066（牛顿法 Jacobian 沿用到下一步、Anderson 提前改用牛顿法）在 weak_tip seed 1 后期上的效果：从续跑状态
`../../10.27/weak_tip/state_s1_n1225.npz` 起按 `../../10.27/weak_tip/drive.py` 的设置跑 STEPS 条记录，逐条记下求 F 的次数、
牛顿步数、残差、电流密度与耗时，与原轨迹 `steps_s1_from1000.csv`（改动前的代码）同一条记录对照。结果写到 check.txt；给了 TOL 时把平均场收敛判据 `_TOL` 换成它，
结果写到 check_tol{TOL}.txt。

usage: check.py STEPS [TOL]
"""
import os
import sys
import time
from dataclasses import replace

import numpy as np
from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WEAK_TIP = os.path.join(ROOT, "results", "10.27", "weak_tip")
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import mean_field  # noqa: E402
from kmc_leakage import tddb as tddb_module  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.tddb import load_state, run_tddb  # noqa: E402

nm = constants.nano
steps = int(sys.argv[1])
name = "check.txt"
if len(sys.argv) > 2:
    mean_field._TOL = float(sys.argv[2])
    name = "check_tol{}.txt".format(sys.argv[2])
config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
stack = config.device.stack
layers = tuple(replace(layer, defects=replace(layer.defects, min_separation=0.42 * nm)) for layer in stack.layers)
config = replace(config, device=replace(config.device, stack=replace(stack, layers=layers)),
                 simulation=replace(config.simulation, seed=1),
                 tddb=replace(config.tddb, transfer_integral=2.4 * constants.electron_volt,
                              activation_energy=1.60 * constants.electron_volt, activation_energy_tail=0.0,
                              field_averaging_width=0.6 * nm,
                              dipole_moment=0.9 * constants.elementary_charge * constants.angstrom))
start = load_state(os.path.join(WEAK_TIP, "state_s1_n1225.npz"))
reference = np.genfromtxt(os.path.join(WEAK_TIP, "steps_s1_from1000.csv"), delimiter=",", names=True)
reference = {int(r["n_generated"]): r for r in reference}

lines = ["n    求F  牛顿  残差      J [A/cm²]        原轨迹 求F 牛顿  J [A/cm²]        J 相对差   耗时 [s]"]
records = []
began = [time.time()]
original = tddb_module._steady_state


def steady_state(*args):
    step = original(*args)
    records.append((step.transport.result, step.iterations, time.time() - began[0]))
    return step


class Done(Exception):
    pass


def progress(n, t, j):
    mf, thermal, wall = records[-1]
    density = j * constants.centi ** 2
    ref = reference.get(n)
    if ref is None:
        tail = "  （原轨迹没有这条）"
    else:
        tail = "  {:4d} {:3d}  {:.9e}  {:.1e}".format(int(ref["mf_iterations"]), int(ref["mf_newton_steps"]),
                                                     ref["J_A_cm2"], abs(density / ref["J_A_cm2"] - 1.0))
    lines.append("{:4d} {:4d} {:4d}  {:.1e}  {:.9e}{}  {:.0f}".format(
        n, mf.iterations, mf.newton_steps, mf.residual, density, tail, wall))
    print(lines[-1], flush=True)
    began[0] = time.time()
    if len(records) >= steps:
        raise Done


tddb_module._steady_state = steady_state
try:
    run_tddb(config, progress=progress, start=start)
except Done:
    pass
text = "\n".join(lines)
open(os.path.join(HERE, name), "w", encoding="utf-8").write(text + "\n")
