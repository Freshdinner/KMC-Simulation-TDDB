"""本次验收（E_A = 1.6 eV，seed 1、2，各 2 × 10 h 机时）的结构图。

path_accept.pdf / .png（按 Nature 版式：双栏 183 mm、Arial 7 pt）：停跑时缺陷的 x–z、y–z 投影（最后一个快照加上之后各步的出生位置），
孤立空位、悬空导电簇、接负极导体分色。逐步数据 = `steps_s{SEED}.csv` 里续跑起点之前的行 + `steps_s{SEED}_from{N}.csv`。
usage: plot.py
"""
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.tddb import conducting_clusters, distance_clusters  # noqa: E402

# Nature 版式：Arial 5–7 pt，线宽 0.5–1 pt，刻度朝外，去掉上、右边框
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False, "pdf.fonttype": 42,
    "savefig.dpi": 600})
MM = 1.0 / 25.4  # [inch]
SEEDS = (1, 2)
NM = 1e-9
CONFIG = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
DEVICE = CONFIG.device
R_C = CONFIG.tddb.percolation_distance / NM
LZ = DEVICE.Lz / NM


def steps(seed):
    """拼接续跑前后的逐步数据（出生时刻与位置）。"""
    rows = list(csv.DictReader(open(os.path.join(HERE, "steps_s{}.csv".format(seed)), encoding="utf-8")))
    for resumed in sorted(glob.glob(os.path.join(HERE, "steps_s{}_from*.csv".format(seed)))):
        start = int(resumed.rsplit("from", 1)[1][:4])
        rows = [r for r in rows if int(r["n_generated"]) < start] + list(csv.DictReader(open(resumed, encoding="utf-8")))
    return {name: np.array([float(r[name]) for r in rows]) for name in ("t_s", "birth_x_nm", "birth_y_nm", "birth_z_nm")}


def final_positions(seed, d):
    """停跑时缺陷位置 [nm]：最后一个快照，加上它之后各步的出生位置。"""
    snap = np.load(sorted(glob.glob(os.path.join(HERE, "snap_s{}_n*.npz".format(seed))))[-1])
    n_snap = snap["positions"].shape[0] - int(snap["n_initial"])
    later = np.column_stack([d["birth_x_nm"], d["birth_y_nm"], d["birth_z_nm"]])[n_snap + 1:]
    return np.vstack([snap["positions"] / NM, later]), int(snap["n_initial"])


data = {seed: steps(seed) for seed in SEEDS}

fig, axes = plt.subplots(len(SEEDS), 2, figsize=(183 * MM, 72 * MM), constrained_layout=True, squeeze=False)
for row, seed in enumerate(SEEDS):
    pos, n_initial = final_positions(seed, data[seed])
    sides = conducting_clusters(pos * NM, DEVICE, R_C * NM)
    labels = distance_clusters(pos * NM, DEVICE, R_C * NM)
    floating = (sides < 0) & (np.bincount(labels)[labels] >= 2)
    for col, (axis, name) in enumerate(((0, "x"), (1, "y"))):
        ax = axes[row, col]
        for mask, color, size, label in (
                ((sides < 0) & ~floating, "#BBBBBB", 2.5, "Isolated vacancy"),
                (floating, "#CC79A7", 5, "Floating conductor (≥ 2)"),
                ((sides == 0) | (sides == 2), "#0072B2", 9, "Negative-side conductor"),
                ((sides == 1) | (sides == 2), "#D55E00", 9, "Positive-side conductor")):
            ax.scatter(pos[mask, axis], pos[mask, 2], s=size, color=color, label=label, lw=0)
        ax.scatter(pos[:n_initial, axis], pos[:n_initial, 2], s=9, facecolors="none", edgecolors="k", lw=0.4,
                   label="Initial defect")
        ax.set_xlim(0, DEVICE.Lx / NM)
        ax.set_ylim(0, LZ)
        ax.set_xlabel("{} (nm)".format(name))
        ax.set_ylabel("z (nm)")
        ax.text(1.0, 1.02, "seed {}: N = {}, t = {:.2g} s".format(seed, pos.shape[0] - n_initial, data[seed]["t_s"][-1]),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
for ax, letter in zip(axes.flat, "abcd"):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")
handles, names = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, names, frameon=False, loc="outside lower center", ncol=5, markerscale=1.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "path_accept.{}".format(ext)))
print("wrote path_accept.pdf, path_accept.png")
