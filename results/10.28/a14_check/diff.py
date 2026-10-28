"""A1–A4 前后对拍结果的比较：`result_ref_s{SEED}.npz`（改动前的源码 src_ref）与 `result_new_s{SEED}.npz`（改动后的 src）。
写到 diff.txt。

usage: diff.py
"""
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []


def relative(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(scale > 0, np.abs(a - b) / np.where(scale > 0, scale, 1.0), 0.0), initial=0.0))


for path in sorted(glob.glob(os.path.join(HERE, "result_ref_s*.npz"))):
    seed = path.rsplit("_s", 1)[1][:-4]
    other = os.path.join(HERE, "result_new_s{}.npz".format(seed))
    if not os.path.exists(other):
        continue
    ref, new = np.load(path), np.load(other)
    lines.append("== seed {}".format(seed))
    lines.append("  求 F：{} -> {} 次；每次中位数 {:.2f} -> {:.2f} s；稳态 {:.1f} -> {:.1f} s；生成率 {:.1f} -> {:.1f} s".format(
        int(ref["iterations"]), int(new["iterations"]), np.median(ref["flux_times"]), np.median(new["flux_times"]),
        float(ref["steady_wall"]), float(new["steady_wall"]), float(ref["rates_wall"]), float(new["rates_wall"])))
    lines.append("  峰值工作集（一步之后）{:.0f} -> {:.0f} MB（读入状态与耦合矩阵后 {:.0f} / {:.0f} MB）".format(
        float(ref["step_peak_MB"]), float(new["step_peak_MB"]), float(ref["base_peak_MB"]), float(new["base_peak_MB"])))
    lines.append("  残差 {:.2e} / {:.2e}".format(float(ref["residual"]), float(new["residual"])))
    for key in ("occupancy", "transport_current", "gap_current", "filament_current", "heat", "temperature",
                "peak_temperature", "rates"):
        lines.append("  {:<18s} 最大相对差 {:.1e}".format(key, relative(ref[key], new[key])))
    lines.append("  ΣG 相对差 {:.1e}".format(relative(ref["rates"].sum(), new["rates"].sum())))
    if "events" in ref.files and "events" in new.files:
        lines.append("  轨迹 {} 步：断键位点序列{}（{} / {}）；时刻最大相对差 {:.1e}；电流密度最大相对差 {:.1e}；"
                     "机时 {:.0f} -> {:.0f} s，峰值工作集 {:.0f} -> {:.0f} MB".format(
                         len(ref["step_times"]), "相同" if np.array_equal(ref["events"], new["events"]) else "不同",
                         ref["events"].tolist(), new["events"].tolist(), relative(ref["step_times"], new["step_times"]),
                         relative(ref["step_currents"], new["step_currents"]), float(ref["trajectory_wall"]),
                         float(new["trajectory_wall"]), float(ref["trajectory_peak_MB"]), float(new["trajectory_peak_MB"])))
text = "\n".join(lines)
print(text)
open(os.path.join(HERE, "diff.txt"), "w", encoding="utf-8").write(text + "\n")
