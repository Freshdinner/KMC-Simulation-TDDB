"""ADR-046 汇总（由 ../rerun/analyze.py 改写，轨迹延长到 1000 个空位）：逐种子的数值验证与物理验收表，以及两张图。
只读 steps_s*.csv，跑到一半也能出。

usage: analyze.py
- 复现：前 400 条记录与 ADR-045（../rerun/steps_s*.csv）逐列比较（墙钟列除外）；
- 数值验证：平均场收敛、强耦合分量切开数、ΣG = 单元事件 + 源陷阱事件（与 run_tddb 自己写的 final CSV 比）、
  新空位的 min_separation 与可采样窗口（取 final npz，没有就取最新的快照）、能量守恒、t = 0 电流与 ADR-041 比、
  进程工作集；
- 物理验收：I-t 双对数拟合（`fit_transient`，全程与 400 个空位以后两段）、渗流团簇最高点、新空位 z 分布、温升、
  空态空位与净空间电荷、ΣG 的演化与两类事件的份额；ADR-041 的对照取 results/9.30/phase4_rerun/logs 的逐步日志；
- 图：ext1000_summary.png（J(t)、ΣG(n)、渗流团簇最高点与新空位 z）、ext1000_charge_thermal.png（J 对空位数、
  空态空位数、温升）。
"""
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.io import load_config  # noqa: E402
from kmc_leakage.tddb import fit_power_law, fit_transient  # noqa: E402

SEEDS = (1, 2, 3, 4)
MARKS = (100, 400, 700, 1000)
ADR045 = os.path.join(HERE, "..", "rerun")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ADR041_LOGS = os.path.join(ROOT, "results", "9.30", "phase4_rerun", "logs")
ADR041_LINE = re.compile(r"^\s*(\d+) t=(\S+) J=(\S+) perc=(\d+) zmax=(\S+) T=(\S+) wall=(\d+)")

config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
device, tddb = config.device, config.tddb
nm = 1e-9
LZ = device.Lz / nm
DC = tddb.percolation_distance / nm
STANDOFF = device.stack.layers[0].defects.min_separation / nm
AMBIENT = config.simulation.temperature
E = 1.602176634e-19


def load_steps(seed):
    path = os.path.join(HERE, "steps_s{}.csv".format(seed))
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(data) if data.size else None


def reproduction_table(runs):
    """前 400 条记录与 ADR-045 的 steps CSV 逐字比较（两份驱动写同样的格式；墙钟与工作集列除外）。"""
    print("\n## 复现（前 400 条记录 vs ADR-045）")
    print("| seed | 比较的记录数 | 逐字相同的记录数 | 第一条不同的记录 |")
    print("|---|---|---|---|")
    for seed in runs:
        with open(os.path.join(ADR045, "steps_s{}.csv".format(seed)), encoding="utf-8") as handle:
            old = handle.read().splitlines()
        with open(os.path.join(HERE, "steps_s{}.csv".format(seed)), encoding="utf-8") as handle:
            new = handle.read().splitlines()
        width = old[0].split(",").index("wall_s")
        if new[0].split(",")[:width] != old[0].split(",")[:width]:
            raise SystemExit("列不一致")
        pairs = list(zip(old[1:], new[1:]))
        same = [a.split(",")[:width] == b.split(",")[:width] for a, b in pairs]
        bad = [k for k, ok in enumerate(same) if not ok]
        print("| {} | {} | {} | {} |".format(seed, len(pairs), sum(same), bad[0] if bad else "-"))


def load_adr041(seed):
    """ADR-041 开热反馈的逐步日志：seed 2 跑完（run_thermal_s2.log），其余取第二次被回收停掉的那份。"""
    for name in ("run_thermal_s{}.log", "run_thermal_s{}_killed_oom2.log"):
        path = os.path.join(ADR041_LOGS, name.format(seed))
        if os.path.exists(path):
            rows = [[float(v) for v in m.groups()] for m in map(ADR041_LINE.match, open(path, encoding="utf-8")) if m]
            return np.array(rows)
    return None


class Trajectory:
    """`fit_transient` 用到的那几项：分段常数的 J(t)，面积记 1，电流单位就是 A/cm²。"""

    area = 1.0

    def __init__(self, time, density):
        self.time, self.current, self.end_time = time, density, float(time[-1])

    def current_at(self, times):
        index = np.searchsorted(self.time, np.asarray(times, dtype=float), side="right") - 1
        return self.current[np.clip(index, 0, None)]


def at(data, n, key):
    rows = data[data["n_generated"] == n]
    return rows[key][0] if rows.size else np.nan


def latest_positions(seed):
    """(位置 [nm], 初始缺陷数)：结束了取 final npz，否则取最新的快照。"""
    final = os.path.join(HERE, "final_s{}.npz".format(seed))
    snaps = sorted(glob.glob(os.path.join(HERE, "snap_s{}_n*.npz".format(seed))))
    path = final if os.path.exists(final) else (snaps[-1] if snaps else None)
    if path is None:
        return None, None
    data = np.load(path)
    return data["positions"] / nm, int(data["n_initial"])


def birth_checks(positions, n_initial):
    """每个生成空位出生时到之前全部缺陷的最小距离（xy 最小镜像）与 z 的范围 [nm]。"""
    box = np.array([device.Lx, device.Ly]) / nm
    nearest = []
    for k in range(n_initial, positions.shape[0]):
        d = positions[:k] - positions[k]
        d[:, :2] -= box * np.round(d[:, :2] / box)
        nearest.append(np.sqrt((d ** 2).sum(axis=1)).min())
    z = positions[n_initial:, 2]
    return (min(nearest) if nearest else np.nan), (z.min() if z.size else np.nan), (z.max() if z.size else np.nan)


def numerical_table(runs):
    print("\n## 数值验证")
    print("| seed | 记录数 | 平均场全收敛 | 迭代数最大 | 残差最大 | 最大强耦合分量 | 切开分量数最大 | "
          "ΣG vs 单元+陷阱（final CSV）| 已满事件最多 | 出生时最小间距 [nm] | 新空位 z 范围 [nm] | "
          "(介质+电极放热)/U·I_陷阱 − 1 | 介质放热/U·I_陷阱 | J(0) 本次 / ADR-041 [A/cm²] | 峰值工作集 [MB] |")
    print("|" + "---|" * 15)
    for seed, data in runs.items():
        final_csv = os.path.join(HERE, "final_s{}.csv".format(seed))
        g_check = "未结束"
        if os.path.exists(final_csv):
            final = np.genfromtxt(final_csv, delimiter=",", names=True)
            parts = data["G_cells_s"] + data["G_traps_s"]
            g_check = "{:.1e}".format(np.max(np.abs(final["generation_rate_s"] / parts - 1.0)))
        positions, n_initial = latest_positions(seed)
        d_min, z_lo, z_hi = birth_checks(positions, n_initial) if positions is not None else (np.nan,) * 3
        ui = data["UI_trap_W"]
        balance = (data["P_dielectric_W"] + data["P_cathode_W"] + data["P_anode_W"]) / ui - 1.0
        ratio = data["P_dielectric_W"] / ui
        old = load_adr041(seed)
        print("| {} | {} | {} | {} | {:.1e} | {} | {} | {} | {} | {:.4f} | {:.3f}–{:.3f} | {:.1e} | {:+.3f} ～ {:+.3f} | "
              "{:.4e} / {} | {:.0f} |".format(
                  seed, data.size, bool(np.all(data["mf_converged"] == 1)), int(data["mf_iterations"].max()),
                  data["mf_residual"].max(), int(data["largest_component"].max()), int(data["n_split"].max()),
                  g_check, int(data["n_full_events"].max()), d_min, z_lo, z_hi, np.max(np.abs(balance)),
                  ratio.min(), ratio.max(), data["J_A_cm2"][0], "{:.4e}".format(old[0, 2]) if old is not None else "-",
                  data["ws_peak_MB"].max()))
    print("（判据：min_separation = {:.2f} nm，可采样窗口 z ∈ [{:.2f}, {:.2f}] nm）".format(STANDOFF, STANDOFF, LZ - STANDOFF))


def first(data, key, level):
    """`key` 第一次达到 `level` 时的 (空位数, 时刻 [s])。"""
    hit = np.flatnonzero(data[key] >= level)
    return (int(data["n_generated"][hit[0]]), data["t_s"][hit[0]]) if hit.size else (None, None)


def physics_table(runs):
    marks = " / ".join(str(m) for m in MARKS)
    print("\n## I-t 与时间标尺")
    print("| seed | 空位数 | J(0) → J(终点) [A/cm²] | n（R²），全程 | t @{} [s] | J @{} [A/cm²] | "
          "J 对空位数的斜率 d log J / d log N（R²），400 个空位以后 | J 达到 1 / 10 / 100 A/cm² 时（空位数，t [s]） | "
          "ADR-041 t @{} [s] | ADR-041 J @{} [A/cm²] |".format(marks, marks, marks, marks))
    print("|" + "---|" * 10)
    for seed, data in runs.items():
        n = data["n_generated"]
        fit = "-"
        if data.size > 2:
            f, _ = fit_transient(Trajectory(data["t_s"], data["J_A_cm2"]))
            fit = "{:.2f}（{:.2f}）".format(f.exponent, f.r_squared)
        late = n >= 400
        slope = "-"
        if np.count_nonzero(late) > 2:
            # 生成时刻在 400 个空位以后挤在一起，对 t 拟合没有意义；按空位数拟合，看电流对缺陷累积的响应
            s = fit_power_law(n[late], data["J_A_cm2"][late])
            slope = "{:.2f}（{:.2f}）".format(s.exponent, s.r_squared)
        old = load_adr041(seed)
        old_t = [old[old[:, 0] == m, 1] for m in MARKS] if old is not None else []
        old_j = [old[old[:, 0] == m, 2] for m in MARKS] if old is not None else []
        print("| {} | {} | {:.2e} → {:.2e} | {} | {} | {} | {} | {} | {} | {} |".format(
            seed, int(n[-1]), data["J_A_cm2"][0], data["J_A_cm2"][-1], fit,
            " / ".join("{:.4g}".format(at(data, m, "t_s")) for m in MARKS),
            " / ".join("{:.3g}".format(at(data, m, "J_A_cm2")) for m in MARKS), slope,
            " / ".join("({}, {})".format(*((k, "{:.4g}".format(t)) if k is not None else ("-", "-")))
                       for k, t in (first(data, "J_A_cm2", level) for level in (1.0, 10.0, 100.0))),
            " / ".join("{:.3g}".format(t[0]) if len(t) else "-" for t in old_t),
            " / ".join("{:.3g}".format(j[0]) if len(j) else "-" for j in old_j)))

    print("\n## 渗流与生成位置")
    print("| seed | 最大渗流团簇（个）@{} | 最高点 z @{} [nm] | 触阴极团簇最高点 @终点 [nm] | "
          "ADR-041 最高点 @{} [nm] | 新空位 z 中位数 / 90% / 最大 [nm] | z > 2 nm | 后 100 个 z 中位数 / 90% | "
          "源陷阱事件生成的比例 |".format(marks, marks, marks))
    print("|" + "---|" * 9)
    for seed, data in runs.items():
        born = data[data["n_generated"] > 0]
        z = born["birth_z_nm"]
        late = z[-100:]
        old = load_adr041(seed)
        old_top = [old[old[:, 0] == m, 4] for m in MARKS] if old is not None else []
        print("| {} | {} | {} | {:.2f} | {} | {:.2f} / {:.2f} / {:.2f} | {:.1%} | {:.2f} / {:.2f} | {:.1%} |".format(
            seed, " / ".join("{:.0f}".format(at(data, m, "percolation_size")) for m in MARKS),
            " / ".join("{:.2f}".format(at(data, m, "percolation_top_nm")) for m in MARKS),
            data["cathode_cluster_top_nm"][-1],
            " / ".join("{:.2f}".format(t[0]) if len(t) else "-" for t in old_top),
            np.median(z), np.percentile(z, 90), z.max(), np.mean(z > 2.0), np.median(late), np.percentile(late, 90),
            np.mean(born["birth_event"] == "trap")))

    print("\n## 热反馈与电流机制")
    print("| seed | 温度场最高升幅 max [K] | 缺陷处加权温升 max [K] | 输运温度 − T_amb 范围 [K] | "
          "PF 发射占电流 @{} | 介质放热 / U·I_陷阱 @{} | 阳极放热 / U·I_陷阱 @终点 | U·I @终点 [W] |".format(marks, marks))
    print("|" + "---|" * 8)
    for seed, data in runs.items():
        transport = data["T_transport_K"] - AMBIENT
        print("| {} | {:.2e} | {:.2e} | {:+.1e} ～ {:+.1e} | {} | {} | {:.2f} | {:.2e} |".format(
            seed, np.max(data["T_peak_K"] - AMBIENT), np.max(data["T_effective_K"] - AMBIENT),
            transport.min(), transport.max(),
            " / ".join("{:.0%}".format(at(data, m, "J_pf_A_cm2") / at(data, m, "J_A_cm2")) for m in MARKS),
            " / ".join("{:+.2f}".format(at(data, m, "P_dielectric_W") / at(data, m, "UI_trap_W")) for m in MARKS),
            data["P_anode_W"][-1] / data["UI_trap_W"][-1], data["UI_trap_W"][-1]))

    print("\n## 正空间电荷与 ΣG")
    print("| seed | 空态生成空位 @{}（占生成数） | 净空间电荷 @0 / 终点 [e] | ΣG @0 / {} [s⁻¹] | "
          "ΣG 最大值（空位数） | 每 100 个空位一段的 ΣG 中位数 [s⁻¹] | 陷阱项份额 @0 / {} |".format(marks, marks, marks))
    print("|" + "---|" * 7)
    for seed, data in runs.items():
        g, n = data["G_total_s"], data["n_generated"]
        empty = " / ".join("{:.0f}（{:.0%}）".format(at(data, m, "empty_generated"), at(data, m, "empty_generated") / m)
                           for m in MARKS)
        windows = " / ".join("{:.2g}".format(np.median(g[(n >= lo) & (n < lo + 100)]))
                             for lo in range(0, int(n[-1]), 100))
        print("| {} | {} | {:.1f} / {:.1f} | {} | {:.2e}（{}） | {} | {} |".format(
            seed, empty, data["space_charge_C"][0] / E, data["space_charge_C"][-1] / E,
            " / ".join("{:.2e}".format(at(data, m, "G_total_s")) for m in (0,) + MARKS), g.max(),
            int(n[np.argmax(g)]), windows,
            " / ".join("{:.1%}".format(at(data, m, "G_traps_s") / at(data, m, "G_total_s")) for m in (0,) + MARKS)))


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK, labelsize=8)


def summary_figure(runs):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), constrained_layout=True)
    for col, seed in enumerate(SEEDS):
        top, middle, bottom = axes[:, col]
        data, old = runs.get(seed), load_adr041(seed)
        top.set_title("seed {}".format(seed), color=INK)
        if old is not None:
            keep = old[:, 1] > 0
            top.step(old[keep, 1], old[keep, 2], where="post", color=ORANGE, lw=1.5, label="ADR-041 (old rates)")
            # 最大团簇只有一两个缺陷时它的最高点就是随便哪个缺陷的位置，从 ≥ 3 个缺陷起画
            grown = (old[:, 3] >= 3) & (old[:, 0] <= MARKS[-1])
            bottom.plot(old[grown, 0], old[grown, 4], color=ORANGE, lw=1.5, label="ADR-041 cluster top")
        if data is not None:
            keep = data["t_s"] > 0
            top.step(data["t_s"][keep], data["J_A_cm2"][keep], where="post", color=BLUE, lw=1.5,
                     label="ADR-046 (this run)")
            if data.size > 2:
                f, times = fit_transient(Trajectory(data["t_s"], data["J_A_cm2"]))
                top.plot(times, f(times), color=INK, lw=1.0, ls="--",
                         label="power-law fit n={:.2f}, R²={:.2f}".format(f.exponent, f.r_squared))
            n = data["n_generated"]
            middle.plot(n, data["G_total_s"], color=BLUE, lw=1.5, label="total ΣG")
            middle.plot(n, data["G_cells_s"], color=ORANGE, lw=1.5, ls="--", label="electrode term (cells)")
            middle.plot(n, data["G_traps_s"], color=AQUA, lw=1.5, ls=":", label="trap term (source traps)")
            born = data[n > 0]
            bottom.scatter(born["n_generated"], born["birth_z_nm"], s=6, color=AQUA, lw=0, alpha=0.7,
                           label="birth z of each new vacancy")
            grown = data["percolation_size"] >= 3
            bottom.plot(n[grown], data["percolation_top_nm"][grown], color=BLUE, lw=1.5, label="ADR-046 cluster top")
        top.set_xscale("log")
        top.set_yscale("log")
        top.set_xlabel("stress time t [s]", color=INK)
        middle.set_yscale("log")
        middle.set_xlabel("generated vacancies", color=INK)
        bottom.axhline(LZ - DC, color=INK, lw=1, ls="--")
        bottom.text(0.03, LZ - DC - 0.15, "anode contact (Lz − d_c)", color=INK, fontsize=8, va="top",
                    transform=bottom.get_yaxis_transform())
        bottom.set_ylim(0, LZ)
        bottom.set_xlim(0, MARKS[-1])
        bottom.set_xlabel("generated vacancies", color=INK)
        for ax in (top, middle, bottom):
            style(ax)
    axes[0, 0].set_ylabel("J [A/cm²]", color=INK)
    axes[1, 0].set_ylabel("generation rate ΣG [1/s]", color=INK)
    axes[2, 0].set_ylabel("z [nm]", color=INK)
    for row in range(3):
        axes[row, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    for col in range(1, 4):
        axes[0, col].legend(handles=[h for h in axes[0, col].get_lines() if h.get_linestyle() == "--"],
                            frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(os.path.join(HERE, "ext1000_summary.png"), dpi=130)


def charge_thermal_figure(runs):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), constrained_layout=True)
    for col, seed in enumerate(SEEDS):
        current, top, bottom = axes[:, col]
        current.set_title("seed {}".format(seed), color=INK)
        data, old = runs.get(seed), load_adr041(seed)
        # J 对空位数：去掉生成的时间标尺，只看电流对缺陷累积的响应
        if old is not None:
            window = old[:, 0] <= MARKS[-1]
            current.plot(old[window, 0], old[window, 2], color=ORANGE, lw=1.5, label="ADR-041 (old rates)")
        if data is not None:
            n = data["n_generated"]
            current.plot(n, data["J_A_cm2"], color=BLUE, lw=1.5, label="ADR-046 (this run)")
            top.plot(n, data["empty_all"], color=BLUE, lw=1.5, label="net space charge / e (all empty defects)")
            top.plot(n, data["empty_generated"], color=ORANGE, lw=1.5, ls="--", label="empty generated vacancies")
            top.plot(n, n, color=MUTED, lw=0.8, ls=":", label="all generated vacancies")
            # 介质净吸热时升幅 ≤ 0，对数轴上画不出来：置 NaN 让曲线在那里断开，不把两段连起来
            rise = data["T_peak_K"] - AMBIENT
            weighted = data["T_effective_K"] - AMBIENT
            bottom.plot(n, np.where(rise > 0, rise, np.nan), color=BLUE, lw=1.5,
                        label="peak of temperature field − T_amb")
            bottom.plot(n, np.where(weighted > 0, weighted, np.nan), color=ORANGE, lw=1.5, ls="--",
                        label="heat-weighted rise at defects")
        current.set_yscale("log")
        bottom.set_yscale("log")
        for ax in (current, top, bottom):
            ax.set_xlabel("generated vacancies", color=INK)
            ax.set_xlim(0, MARKS[-1])
            style(ax)
    axes[0, 0].set_ylabel("J [A/cm²]", color=INK)
    axes[1, 0].set_ylabel("count", color=INK)
    axes[2, 0].set_ylabel("temperature rise [K]", color=INK)
    for row in range(3):
        axes[row, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(os.path.join(HERE, "ext1000_charge_thermal.png"), dpi=130)


if __name__ == "__main__":
    runs = {seed: d for seed, d in ((s, load_steps(s)) for s in SEEDS) if d is not None}
    reproduction_table(runs)
    numerical_table(runs)
    physics_table(runs)
    summary_figure(runs)
    charge_thermal_figure(runs)
