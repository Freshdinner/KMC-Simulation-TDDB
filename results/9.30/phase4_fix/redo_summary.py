"""汇总 mf_vs_kmc_redo 的结果，口径同 ADR-031 的表。"""
import csv, glob
import numpy as np
D = "results/9.30/phase4_fix/"
kmc = list(csv.DictReader(open("results/9.29/tddb_phase2_mf_vs_kmc.csv", encoding="utf-8")))
def load(v, case):
    files = [D + "redo_{}_3e18.csv".format(v)] if case == "box20_3e18" else sorted(glob.glob(D + "redo_{}_3e19_*.csv".format(v)))
    out = {}
    for f in files:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            out[(int(r["seed"]), r["voltage_V"], r["temperature_C"])] = r
    return out
for case in ("box20_3e18", "box20_3e19"):
    head, new = load("head", case), load("new", case)
    same = sum(float(head[k]["J_mf"]) == float(head[k]["J_mf_csv"]) for k in head)
    print(case, "HEAD 逐位复现 CSV 旧平均场:", same, "/", len(head), " 新平均场全部收敛:", all(r["converged"] == "1" for r in new.values()),
          " 迭代次数", min(int(r["iterations"]) for r in new.values()), "-", max(int(r["iterations"]) for r in new.values()))
    for V, T in sorted({(k[1], k[2]) for k in new}):
        keys = sorted(k for k in new if k[1] == V and k[2] == T)
        old = np.array([float(head[k]["J_mf"]) for k in keys]); nw = np.array([float(new[k]["J_mf"]) for k in keys])
        line = "  {} V {} C M={}: J_old={:.4e} J_new={:.4e} new/old={:.5f} (per-real {:.4f}-{:.4f})".format(
            V, T, len(keys), old.mean(), nw.mean(), nw.mean() / old.mean(), (nw / old).min(), (nw / old).max())
        for solver in ("kmc", "kmc_fast"):
            rows = {int(r["seed"]): r for r in kmc if r["case"] == case and r["solver"] == solver and r["voltage_V"] == V and r["temperature_C"] == T}
            use = [k for k in keys if solver == "kmc" or rows[k[0]]["validated"] == "True"]
            jk = np.array([float(rows[k[0]]["J_A_cm2"]) for k in use])
            jn = np.array([float(new[k]["J_mf"]) for k in use]); jo = np.array([float(head[k]["J_mf"]) for k in use])
            line += " | {}: n={} new {:.4f} ({:.3f}-{:.3f}) old {:.4f}".format(solver, len(use), jn.mean() / jk.mean(), (jn / jk).min(), (jn / jk).max(), jo.mean() / jk.mean())
        if case == "box20_3e19":
            rows = {int(r["seed"]): r for r in kmc if r["case"] == case and r["solver"] == "kmc_fast"}
            jk = np.array([float(rows[k[0]]["J_A_cm2"]) for k in keys]); jn = np.array([float(new[k]["J_mf"]) for k in keys])
            jo = np.array([float(head[k]["J_mf"]) for k in keys])
            line += " | kmc_fast all 16: new {:.4f} ({:.3f}-{:.3f}) old {:.4f}".format(jn.mean() / jk.mean(), (jn / jk).min(), (jn / jk).max(), jo.mean() / jk.mean())
        print(line)
