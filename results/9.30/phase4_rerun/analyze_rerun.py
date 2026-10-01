"""Phase 4 重跑（ADR-038/039/040 之后）的汇总：表格与图。

usage: analyze_rerun.py
- 图（p4fix_summary.png）只用逐步日志，跑到一半也能出：上排 J(t) 双对数（开 / 关热反馈），下排最大渗流团簇的
  最高点随生成空位数的变化（阳极触发线 Lz − d_c）。日志取 logs/run_{mode}_s{seed}.log；没有就取被低内存回收停掉的
  那份——第二次（只跑开热反馈）的 *_killed_oom2.log 优先，关热反馈的对照只有第一次的 *_killed_oom.log（ADR-041）。
- 表只对已经结束、写出 CSV 与 npz 的轨迹给出（seed 2）。
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (1, 2, 3, 4)
# 两组轨迹常常逐步重合：关热反馈的画宽线垫在下面，开热反馈的细线画在上面，两条都看得见
MODES = (("iso", "#eb6834", "isothermal", 4.0), ("thermal", "#2a78d6", "heat feedback", 1.5))
LZ_NM, DC_NM, VOLTAGE, AREA_CM2, AMBIENT = 8.0, 0.9, 3.0, 20e-7 * 20e-7, 298.15
LINE = re.compile(r"^\s*(\d+) t=(\S+) J=(\S+) perc=(\d+) zmax=(\S+) T=(\S+) wall=(\d+)")


def read_log(mode, seed):
    """逐步日志；关热反馈的对照只在第一次（8 个进程并发、内存不足被停掉）的运行里有，取那份。"""
    rows = []
    path = os.path.join(HERE, "logs", "run_{}_s{}.log".format(mode, seed))
    if not os.path.exists(path):
        for suffix in ("_killed_oom2", "_killed_oom"):
            path = os.path.join(HERE, "logs", "run_{}_s{}{}.log".format(mode, seed, suffix))
            if os.path.exists(path):
                break
    if not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8"):
        m = LINE.match(line)
        if m:
            rows.append([float(v) for v in m.groups()])
    return np.array(rows) if rows else None


def figure():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for col, seed in enumerate(SEEDS):
        top, bottom = axes[0, col], axes[1, col]
        for mode, color, label, width in MODES:
            data = read_log(mode, seed)
            if data is None:
                continue
            n, t, j, perc, zmax = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
            keep = t > 0
            top.step(t[keep], j[keep], where="post", color=color, lw=width, label=label)
            # 最大团簇只有一两个缺陷时它的最高点就是随便哪个缺陷的位置，从 ≥ 3 个缺陷起画
            grown = perc >= 3
            bottom.plot(n[grown], zmax[grown], color=color, lw=width, label=label)
        top.set_xscale("log")
        top.set_yscale("log")
        top.set_title("seed {}".format(seed))
        top.set_xlabel("stress time t [s]")
        bottom.axhline(LZ_NM - DC_NM, color="#52514e", lw=1, ls="--")
        bottom.text(0.03, LZ_NM - DC_NM - 0.15, "anode contact (Lz - d_c)", color="#52514e", fontsize=8, va="top",
                    transform=bottom.get_yaxis_transform())
        bottom.set_ylim(0, LZ_NM)
        bottom.set_xlabel("generated vacancies")
        for ax in (top, bottom):
            ax.grid(color="#e4e3df", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("J [A/cm$^2$]")
    axes[1, 0].set_ylabel("top of largest percolation cluster z [nm]")
    axes[0, 0].legend(frameon=False)
    fig.savefig(os.path.join(HERE, "p4fix_summary.png"), dpi=130)


def table():
    from kmc_leakage.io import load_config
    from kmc_leakage.tddb import fit_power_law, percolation_clusters

    device = load_config(os.path.join(HERE, "..", "..", "..", "examples", "zro2_tddb.yaml")).device
    print("| run | stop | t_end [s] | N_gen | J(0) -> J(end) [A/cm2] | n (R2) | perc size | perc top [nm] | "
          "gen z median / 90% / max [nm] | empty gen | dT_eff max [K] | dT_peak max [K] | P_diel/UI (last) | t(J>=1e4) |")
    for seed in SEEDS:
        for mode, _, _, _ in MODES:
            stem = os.path.join(HERE, "p4fix_{}_s{}".format(mode, seed))
            if not (os.path.exists(stem + ".csv") and os.path.exists(stem + ".npz")):
                continue
            csv = np.genfromtxt(stem + ".csv", delimiter=",", names=True)
            npz = np.load(stem + ".npz")
            t0, t1, j = csv["t_start_s"], csv["t_end_s"], csv["J_total_A_cm2"]
            end = float(npz["end_time"])
            fit = "-"
            if t0.size > 1:
                times = np.logspace(np.log10(t0[1]), np.log10(end), int(np.ceil(np.log10(end / t0[1]) * 20)) + 1)
                index = np.clip(np.searchsorted(t0, times, side="right") - 1, 0, None)
                f = fit_power_law(times, j[index])
                fit = "{:.2f} ({:.2f})".format(f.exponent, f.r_squared)
            positions = npz["positions"]
            labels, _ = percolation_clusters(positions, device, DC_NM * 1e-9)
            big = np.argmax(np.bincount(labels))
            z_gen = positions[int(npz["n_initial"]):, 2] * 1e9
            ui = VOLTAGE * j[-1] * AREA_CM2
            over = np.flatnonzero(j >= 1e4)
            print("| {} s{} | {} | {:.3g} | {} | {:.2e} -> {:.2e} | {} | {} | {:.2f} | {:.2f} / {:.2f} / {:.2f} | {:.0f} | "
                  "{:.2e} | {:.2e} | {:.3f} | {} |".format(
                      mode, seed, str(npz["stop_reason"]), end, int(csv["n_generated"][-1]), j[0], j[-1], fit,
                      int(csv["percolation_size"][-1]), positions[labels == big, 2].max() * 1e9,
                      np.median(z_gen), np.percentile(z_gen, 90), z_gen.max(), csv["empty_generated"][-1],
                      np.max(csv["T_effective_K"] - AMBIENT), np.max(csv["T_peak_K"] - AMBIENT),
                      csv["power_W"][-1] / ui, "{:.3g} s".format(t0[over[0]]) if over.size else "-"))


if __name__ == "__main__":
    figure()
    table()
