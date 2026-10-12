"""生成前沿与连通性：比较 N_T = 10²⁰（../discrete/long）与 10²¹ cm⁻³（本目录）两组轨迹，seed 1–2（同一种子初始缺陷相同）。

usage: analyze_front.py

每 25 个空位一张快照（drive.py 写的 snap_s{SEED}_n{NNNN}.npz）。对每张快照：
- 生成空位的深度：中位数、90% 分位、最深；
- 按深度分层（每 1 nm）的前驱用掉比例（与 n = 0 快照的前驱位置比）；
- 连通性：全部空位（初始 + 生成）两两间距 ≤ r_c 算相连，到电极界面的距离 ≤ r_c 算接上电极（xy 周期）。记阴极连通簇伸到
  的最深 z、阳极连通簇伸到的最浅 z、两者之间的间隙，以及有没有一个簇同时接上两极；
- 电流密度 J 取 steps_s{SEED}.csv 里同一空位数的那一行。
写 front.txt（摘要表 + 逐张快照表）与 front.png（连通簇前沿随空位数、随时间；终态前驱用掉比例随深度；I-t）。
"""
import os
import sys

import numpy as np
from scipy import constants
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

nm = constants.nano
L, LZ = 50.0 * nm, 8.0 * nm
RADII = (0.7 * nm, 1.0 * nm, 1.3 * nm)
MAIN = 1  # 摘要表与图用 r_c = 1.0 nm
LAYERS = np.arange(0.0, 9.0, 1.0)  # nm
FINE = np.arange(0.5, 7.75, 0.25)  # nm，前驱撒在 [0.5, 7.5] nm
SEEDS = (1, 2)
RUNS = (("1e20", os.path.join(HERE, "..", "discrete", "long")), ("1e21", HERE))
COLORS = {"1e20": "#1f77b4", "1e21": "#ff7f0e"}
STYLES = {1: "-", 2: "--"}


def fronts(positions, radius):
    """(阴极连通簇最深 z, 阳极连通簇最浅 z, 是否贯穿) [m]。"""
    tree = cKDTree(positions % [L, L, 4 * LZ], boxsize=[L, L, 4 * LZ])
    pairs = tree.query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(positions),) * 2)
    _, label = connected_components(graph, directed=False)
    z = positions[:, 2]
    cathode, anode = set(label[z <= radius]), set(label[LZ - z <= radius])
    reach = max((z[label == c].max() for c in cathode), default=0.0)
    start = min((z[label == c].min() for c in anode), default=LZ)
    return reach, start, bool(cathode & anode)


def used_percent(left, initial):
    return 100.0 * (1.0 - left / np.maximum(initial, 1))


runs = {}
for label, directory in RUNS:
    for seed in SEEDS:
        steps = np.genfromtxt(os.path.join(directory, "steps_s{}.csv".format(seed)), delimiter=",", names=True)
        initial = np.load(os.path.join(directory, "snap_s{}_n0000.npz".format(seed)))["precursor_positions"][:, 2] / nm
        records = []
        for n in range(0, int(steps["n_generated"][-1]) + 1, 25):
            snap = np.load(os.path.join(directory, "snap_s{}_n{:04d}.npz".format(seed, n)))
            positions, n_initial = snap["positions"], int(snap["n_initial"])
            left = snap["precursor_positions"][:, 2] / nm
            records.append(dict(
                n=n, t=float(snap["time"]), J=float(steps["J_A_cm2"][n]), z=positions[n_initial:, 2] / nm,
                layers=np.histogram(left, bins=LAYERS)[0], fine=np.histogram(left, bins=FINE)[0],
                fronts=[fronts(positions, r) for r in RADII]))
        runs[(label, seed)] = dict(steps=steps, records=records, initial_layers=np.histogram(initial, bins=LAYERS)[0],
                                   initial_fine=np.histogram(initial, bins=FINE)[0], n_precursors=initial.size)

# ---- 表 ----
lines = ["摘要（r_c = 1.0 nm；间隙 = 阳极连通簇最浅处 − 阴极连通簇最深处）",
         "{:>6} {:>4} {:>5} {:>10} {:>7} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} {:>5} | {:>6} {:>6} {:>13}".format(
             "N_T", "seed", "n", "t [s]", "J", "z50", "z90", "zmax", "0-1", "1-2", "2-3", "3-8", "reach", "gap",
             "gap@r_c=1.3"),
         "（N_T [cm-3]；t [s]；J [A/cm2]；z50/z90/zmax：生成空位深度的中位数/90% 分位/最深 [nm]；0-1 … 3-8：该深度层 [nm] 的"
         "前驱用掉比例 [%]；reach：阴极连通簇最深处 [nm]）"]
for (label, seed), run in runs.items():
    for record in {r["n"]: r for r in (run["records"][8], run["records"][-1])}.values():  # n = 200 与最后一张快照
        layers, initial = record["layers"], run["initial_layers"]
        used = list(used_percent(layers[:3], initial[:3])) + [used_percent(layers[3:].sum(), initial[3:].sum())]
        reach, start, _ = record["fronts"][MAIN]
        lines.append("{:>6} {:>4} {:>5} {:>10.3e} {:>7.2f} | {:>5.2f} {:>5.2f} {:>5.2f} | {:>5.0f} {:>5.0f} {:>5.0f} {:>5.1f} | "
                     "{:>6.2f} {:>6.2f} {:>13.2f}".format(
                         label, seed, record["n"], record["t"], record["J"], *np.percentile(record["z"], [50, 90, 100]),
                         *used, reach / nm, (start - reach) / nm,
                         (record["fronts"][2][1] - record["fronts"][2][0]) / nm))
lines.append("")
for (label, seed), run in runs.items():
    lines.append("== N_T = {} cm-3, seed {}: {} precursors at t = 0, per nm layer {}".format(
        label, seed, run["n_precursors"], run["initial_layers"].tolist()))
    lines.append("   {:>5} {:>10} {:>7} | {:>6} {:>6} {:>6} | {:<34} | {}".format(
        "n", "t [s]", "J", "z50", "z90", "zmax", "used per layer 0-1,1-2,...,7-8 nm [%]",
        "cathode reach / anode start / gap [nm] (spanning) at r_c = 0.7, 1.0, 1.3 nm"))
    for record in run["records"]:
        if record["z"].size:
            depth = "{:6.2f} {:6.2f} {:6.2f}".format(*np.percentile(record["z"], [50, 90, 100]))
        else:
            depth = "{:>6} {:>6} {:>6}".format("-", "-", "-")
        lines.append("   {:5d} {:10.3e} {:7.2f} | {} | {:<34} | {}".format(
            record["n"], record["t"], record["J"], depth,
            ",".join("{:.0f}".format(u) for u in used_percent(record["layers"], run["initial_layers"])),
            "  ".join("{:.2f}/{:.2f}/{:.2f}{}".format(c / nm, a / nm, max(a - c, 0.0) / nm, " SPAN" if s else "")
                      for c, a, s in record["fronts"])))
with open(os.path.join(HERE, "front.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- 图 ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
by_n, by_t, depth, it = axes.ravel()
for (label, seed), run in runs.items():
    records, color, style = run["records"], COLORS[label], STYLES[seed]
    name = "N$_T$ = {} cm$^{{-3}}$, seed {}".format(label, seed)
    n = np.array([r["n"] for r in records])
    t = np.array([r["t"] for r in records])
    reach = np.array([r["fronts"][MAIN][0] for r in records]) / nm
    start = np.array([r["fronts"][MAIN][1] for r in records]) / nm
    for ax, x, keep in ((by_n, n, n >= 0), (by_t, t, t > 0)):
        ax.plot(x[keep], reach[keep], style, color=color, lw=1.8, label=name)
        ax.plot(x[keep], start[keep], style, color=color, lw=1.0, alpha=0.6)
    final = records[-1]
    used = used_percent(final["fine"], run["initial_fine"])
    depth.step(FINE[:-1], used, where="post", color=color, ls=style, lw=1.5,
               label="{}（n = {}）".format(name, final["n"]))
    steps = run["steps"]
    it.plot(steps["t_s"][1:], steps["J_A_cm2"][1:], style, color=color, lw=1.2, label=name)

for ax, xlabel in ((by_n, "生成空位数 n"), (by_t, "应力时间 t [s]")):
    ax.set_ylim(0.0, 8.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("z [nm]（阴极在 0，阳极在 8）")
    ax.text(0.98, 0.84, "上方细线：阳极侧连通簇最浅处", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.text(0.98, 0.04, "下方粗线：阴极侧连通簇最深处", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.grid(alpha=0.3)
by_n.set_title("(a) 连通簇前沿，r$_c$ = 1.0 nm，按生成空位数", fontsize=10)
by_t.set_title("(b) 连通簇前沿，r$_c$ = 1.0 nm，按时间", fontsize=10)
by_t.set_xscale("log")
by_n.legend(fontsize=8, loc="center left")
depth.set_xlim(0.0, 8.0)
depth.set_ylim(0.0, 100.0)
depth.set_xlabel("离阴极深度 z [nm]")
depth.set_ylabel("前驱用掉比例 [%]（每 0.25 nm 一层）")
depth.set_title("(c) 最后一张快照的前驱用掉比例", fontsize=10)
depth.legend(fontsize=8, loc="upper right")
depth.grid(alpha=0.3)
it.axhline(10.0, color="0.5", lw=0.8, ls=":")
it.text(1.2e-3, 11.0, "原击穿判据 10 A/cm²", fontsize=8, color="0.4", va="bottom")
it.set_xscale("log")
it.set_yscale("log")
it.set_xlim(1e-3, 1e7)
it.set_xlabel("应力时间 t [s]")
it.set_ylabel("J [A/cm²]")
it.set_title("(d) I-t（N$_T$ = 10$^{21}$ 的两条去掉了击穿判据）", fontsize=10)
it.legend(fontsize=8, loc="lower right")
it.grid(alpha=0.3, which="both")
fig.suptitle("前驱浓度 N$_T$ = 10$^{20}$ 与 10$^{21}$ cm$^{-3}$：3.6 V、25 °C、50×50×8 nm，E$_A$ = 1.30 ± 0.10 eV", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "front.png"), dpi=130)
print("wrote front.txt, front.png")
