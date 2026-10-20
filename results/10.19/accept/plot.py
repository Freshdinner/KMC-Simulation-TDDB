"""ADR-063/064 之后 3.6 V 验收的结构图：path_accept.pdf / .png（Nature 版式：双栏 183 mm、Arial 7 pt）。

每个种子取最后一个快照（`snap_s{SEED}_n*.npz`，含 drive.py 存下的导电簇归属 `sides`）的 x–z、y–z 投影：孤立空位、悬空导电簇、接负极 /
接正极导体分色，初始缺陷画空心圈。导电簇编号按同一套离域判据重算（`tddb.conducting_clusters`），用来区分悬空导电簇。
usage: plot.py
"""
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
from kmc_leakage.defects import DefectEnsemble  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.tddb import conducting_clusters  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False, "pdf.fonttype": 42,
    "savefig.dpi": 600})
MM = 1.0 / 25.4
NM = 1e-9
SEEDS = (1, 2)
CONFIG = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
DEVICE = CONFIG.device
MATERIAL = DEVICE.stack.single_material

fig, axes = plt.subplots(len(SEEDS), 2, figsize=(183 * MM, 72 * MM), constrained_layout=True, squeeze=False)
for row, seed in enumerate(SEEDS):
    path = sorted(glob.glob(os.path.join(HERE, "snap_s{}_n*.npz".format(seed))))[-1]
    snap = np.load(path)
    positions, n_initial = snap["positions"], int(snap["n_initial"])
    defects = DefectEnsemble.uniform(
        positions, snap["ED"], MATERIAL, band_reference=float(DEVICE.layer_band_references()[0]), charge_state=1.0)
    labels, sides = conducting_clusters(defects, DEVICE, CONFIG.tddb.transfer_integral)
    floating = (sides < 0) & (np.bincount(labels)[labels] >= 2)
    pos = positions / NM
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
        ax.set_ylim(0, DEVICE.Lz / NM)
        ax.set_xlabel("{} (nm)".format(name))
        ax.set_ylabel("z (nm)")
        ax.text(1.0, 1.02, "seed {}: N = {}, t = {:.2g} s".format(seed, len(pos) - n_initial, float(snap["time"])),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
for ax, letter in zip(axes.flat, "abcd"):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")
handles, names = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, names, frameon=False, loc="outside lower center", ncol=5, markerscale=1.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "path_accept.{}".format(ext)))
print("wrote path_accept.pdf, path_accept.png")
