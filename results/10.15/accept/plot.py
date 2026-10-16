"""本次验收（seed 1、2，各 10 h 机时）的结构图与生成率诊断图。

- path_accept.png：停跑时缺陷的 x–z、y–z 投影（停跑时的快照加上之后各步的出生位置），孤立空位、悬空导电簇、接负极导体分色；
- generation_z.png：(a) 新空位出生高度的直方图；(b) 停跑状态下按 z 分层的 ΣG（电极项 / 陷阱项）；(c) 位点平均 R_1e（电极项 / 陷阱项）
  与 P_bb 随 z 的变化（`diagnose.py` 的输出），阴影标出能直接接负极的薄层 z ∈ [min_separation, r_c)。
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

plt.rcParams["axes.unicode_minus"] = False
SEEDS = (1, 2)
COLORS = ("#2a78d6", "#eb6834")
NM = 1e-9
CONFIG = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
DEVICE = CONFIG.device
R_C = CONFIG.tddb.percolation_distance / NM
D_MIN = CONFIG.defects.min_separation / NM
LZ = DEVICE.Lz / NM


def steps(seed):
    rows = list(csv.DictReader(open(os.path.join(HERE, "steps_s{}.csv".format(seed)), encoding="utf-8")))
    return {name: np.array([float(r[name]) for r in rows]) for name in ("t_s", "birth_x_nm", "birth_y_nm", "birth_z_nm")}


def final_positions(seed, d):
    """停跑时缺陷位置 [nm]：最后一个快照，加上它之后各步的出生位置。"""
    snap = np.load(sorted(glob.glob(os.path.join(HERE, "snap_s{}_n*.npz".format(seed))))[-1])
    n_snap = snap["positions"].shape[0] - int(snap["n_initial"])
    later = np.column_stack([d["birth_x_nm"], d["birth_y_nm"], d["birth_z_nm"]])[n_snap + 1:]
    return np.vstack([snap["positions"] / NM, later]), int(snap["n_initial"])


def style(ax):
    ax.grid(True, which="major", color="#e4e4e2", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)


data = {seed: steps(seed) for seed in SEEDS}

fig, axes = plt.subplots(len(SEEDS), 2, figsize=(13, 3.2 * len(SEEDS)), constrained_layout=True, squeeze=False)
for row, seed in enumerate(SEEDS):
    pos, n_initial = final_positions(seed, data[seed])
    sides = conducting_clusters(pos * NM, DEVICE, R_C * NM)
    labels = distance_clusters(pos * NM, DEVICE, R_C * NM)
    floating = (sides < 0) & (np.bincount(labels)[labels] >= 2)
    for col, (axis, name) in enumerate(((0, "x"), (1, "y"))):
        ax = axes[row, col]
        for mask, color, size, label in (
                ((sides < 0) & ~floating, "#b9b9b6", 7, "isolated vacancy"),
                (floating, "#8a5cc2", 10, "floating conductor (≥ 2)"),
                ((sides == 0) | (sides == 2), "#2a78d6", 22, "negative-side conductor"),
                ((sides == 1) | (sides == 2), "#eb6834", 22, "positive-side conductor")):
            ax.scatter(pos[mask, axis], pos[mask, 2], s=size, color=color, label=label, lw=0)
        ax.scatter(pos[:n_initial, axis], pos[:n_initial, 2], s=22, facecolors="none", edgecolors="k", lw=0.6,
                   label="initial defect")
        ax.set_xlim(0, DEVICE.Lx / NM)
        ax.set_ylim(0, LZ)
        ax.set_xlabel("{} (nm)".format(name))
        ax.set_ylabel("z (nm)  [negative electrode at z = 0]")
        ax.set_title("seed {}: {} vacancies generated, t = {:.3g} s (10 h wall limit)".format(
            seed, pos.shape[0] - n_initial, data[seed]["t_s"][-1]), fontsize=10)
handles, names = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, names, frameon=False, fontsize=9, loc="outside lower center", ncol=5)
fig.savefig(os.path.join(HERE, "path_accept.png"), dpi=150)

fig, (ax_h, ax_g, ax_r) = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
bins = np.arange(0.2, LZ + 1e-9, 0.1)
for seed, color in zip(SEEDS, COLORS):
    z = data[seed]["birth_z_nm"][1:]
    ax_h.hist(z[np.isfinite(z)], bins=bins, histtype="step", lw=2, color=color, label="seed {}".format(seed))
    path = os.path.join(HERE, "diagnose_s{}.npz".format(seed))
    if not os.path.exists(path):
        continue
    d = np.load(path)
    mid = 0.5 * (d["edges"][1:] + d["edges"][:-1])
    ax_g.plot(mid, d["G_el"], color=color, lw=2, label="seed {} electrode term".format(seed))
    ax_g.plot(mid, d["G_tr"], color=color, lw=2, ls="--", label="seed {} trap term".format(seed))
    ax_r.plot(mid, d["R_el"], color=color, lw=2, label="seed {} ⟨R_1e⟩ electrode".format(seed))
    ax_r.plot(mid, d["R_tr"], color=color, lw=2, ls="--", label="seed {} ⟨R_1e⟩ trap".format(seed))
    ax_r.plot(mid, d["P_bb"], color=color, lw=1.2, ls=":", label="seed {} ⟨P_bb⟩".format(seed))
for ax in (ax_h, ax_g, ax_r):
    ax.axvspan(D_MIN, R_C, color="#f2c94c", alpha=0.6, lw=0)
    ax.set_xlim(0, LZ)
    ax.set_xlabel("z (nm)")
    style(ax)
ax_h.set_ylabel("births per 0.1 nm")
ax_h.set_title("(a) where new vacancies are born")
ax_g.set_yscale("log")
ax_g.set_ylabel("ΣG in 0.1 nm slab (1/s)")
ax_g.set_title("(b) generation rate by height, state at stop")
ax_r.set_yscale("log")
ax_r.set_ylabel("per-site mean")
ax_r.set_title("(c) capture rate R_1e (1/s) and P_bb per site")
ax_h.annotate("can touch negative\nelectrode: z ∈ [{:.2f}, {:.2f}) nm".format(D_MIN, R_C), (R_C, 0.92),
              xycoords=("data", "axes fraction"), xytext=(8, 0), textcoords="offset points", fontsize=9, color="#3a3a3a")
for ax in (ax_h, ax_g, ax_r):
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("Why the negative-side conductor does not grow (3.6 V / 25 °C, r_c = {:.2f} nm, sites packed at {:.2f} nm)".format(
    R_C, D_MIN))
fig.savefig(os.path.join(HERE, "generation_z.png"), dpi=150)
print("wrote path_accept.png, generation_z.png")
