"""最小间隔 0.5 nm、E_A 固定 1.45 eV、t₀ = 2.0 eV 的 3.6 V 测试的结构图：path_sep50.pdf / .png，版式同 `../../10.21/sep040/path_sep040`
（Nature 版式：双栏 183 mm、7 pt；Times New Roman + stix 公式、坐标轴框线与刻度线 1.5 pt）。

每个种子取击穿那一刻的快照（`snap_s{SEED}_n*.npz` 里第一个有贯穿通路的，含 drive.py 存下的导电簇归属 `sides`；没有击穿时取最后一个）的
x–z、y–z 投影：孤立空位、悬空导电簇、只接负极 / 只接正极的导体、两侧都接的贯穿导体分色，初始缺陷画空心圈。导电簇编号按同一套离域判据
（t₀ = 2.0 eV）重算（`tddb.conducting_clusters`），用来区分悬空导电簇。
usage: plot.py
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import constants  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from kmc_leakage.defects import DefectEnsemble  # noqa: E402
from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.tddb import conducting_clusters  # noqa: E402

plt.rcParams.update({
    "font.family": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 1.5, "xtick.major.width": 1.5,
    "ytick.major.width": 1.5, "xtick.major.size": 3, "ytick.major.size": 3, "xtick.direction": "out",
    "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "savefig.dpi": 600})
MM = 1.0 / 25.4
NM = 1e-9
SEEDS = (1, 2)
TRANSFER_INTEGRAL = 2.0 * constants.electron_volt  # t₀ = 2.0 eV（同 drive.py）
CONFIG = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
DEVICE = CONFIG.device
MATERIAL = DEVICE.stack.single_material

fig, axes = plt.subplots(len(SEEDS), 2, figsize=(183 * MM, 72 * MM), constrained_layout=True, squeeze=False)
for row, seed in enumerate(SEEDS):
    snaps = [np.load(path) for path in sorted(glob.glob(os.path.join(HERE, "snap_s{}_n*.npz".format(seed))))]
    spanning = [snap for snap in snaps if np.any(snap["sides"] == 2)]
    snap = spanning[0] if spanning else snaps[-1]
    positions, n_initial = snap["positions"], int(snap["n_initial"])
    defects = DefectEnsemble.uniform(
        positions, snap["ED"], MATERIAL, band_reference=float(DEVICE.layer_band_references()[0]), charge_state=1.0)
    labels, sides = conducting_clusters(defects, DEVICE, TRANSFER_INTEGRAL)
    floating = (sides < 0) & (np.bincount(labels)[labels] >= 2)
    pos = positions / NM
    for col, (axis, name) in enumerate(((0, "x"), (1, "y"))):
        ax = axes[row, col]
        for mask, color, size, label in (
                ((sides < 0) & ~floating, "#BBBBBB", 2.5, "Isolated vacancy"),
                (floating, "#CC79A7", 5, "Floating conductor (≥ 2)"),
                (sides == 0, "#0072B2", 9, "Negative-side conductor"),
                (sides == 1, "#D55E00", 9, "Positive-side conductor"),
                (sides == 2, "#009E73", 9, "Spanning conductor")):
            ax.scatter(pos[mask, axis], pos[mask, 2], s=size, color=color, label=label, lw=0)
        ax.scatter(pos[:n_initial, axis], pos[:n_initial, 2], s=9, facecolors="none", edgecolors="k", lw=0.4,
                   label="Initial defect")
        ax.set_xlim(0, DEVICE.Lx / NM)
        ax.set_ylim(0, DEVICE.Lz / NM)
        ax.set_xlabel("{} (nm)".format(name))
        ax.set_ylabel("z (nm)")
        ax.text(1.0, 1.02, "seed {}: N = {}, {} = {:.0f} s".format(
            seed, len(pos) - n_initial, r"$t_\mathrm{BD}$" if spanning else "t", float(snap["time"])),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
for ax, letter in zip(axes.flat, "abcd"):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")
handles, names = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, names, frameon=False, loc="outside lower center", ncol=6, markerscale=1.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "path_sep50.{}".format(ext)))
print("wrote path_sep50.pdf, path_sep50.png")
