"""汇总 j0_L{20,30,40,50}.npz（t = 0 的 J 分布）与 step_cost_L{20,30}.log（TDDB 每步耗时与峰值内存），写 summary.txt。
40、50 nm 的每步耗时与峰值内存按 20、30 nm 的实测对面积线性外推。"""
import os
import re

import numpy as np

from box import HERE

lines = ["t = 0 的 J（3.6 V / 25 °C，簇级平均场，seed 1–128，hx = 0.25 nm，其余同 examples/zro2_tddb.yaml）[A/cm²]",
         "盒子      N   中位数  算术平均 ± 标准误  几何平均  σ(log10 J)  min      p10     p90    max   承载最多缺陷份额中位数"]
for side in (20, 30, 40, 50):
    data = np.load(os.path.join(HERE, "j0_L{}.npz".format(side)))
    J = data["J"]
    lines.append("{:2d}×{:2d} nm {:3d}  {:.3f}   {:.3f} ± {:.3f}      {:.3f}     {:.2f}      {:.1e}  {:.3f}   {:.2f}   {:.2f}   {:.0%}".format(
        side, side, int(data["n_defects"][0]), np.median(J), J.mean(), J.std(ddof=1) / np.sqrt(J.size),
        10 ** np.log10(J).mean(), np.log10(J).std(), J.min(), np.percentile(J, 10), np.percentile(J, 90), J.max(),
        np.median(data["share"])))

lines += ["", "TDDB 外层每步（seed 1，开热反馈，tddb.run_tddb 原样调用）"]
area, wall, peak = [], [], []
for side in (20, 30):
    text = open(os.path.join(HERE, "step_cost_L{}.log".format(side)), encoding="utf-8").read()
    steps = [float(v) for v in re.findall(r"step wall ([\d.]+) s", text)]
    top = float(re.search(r"peak ws (\d+) MB", text).group(1))
    area.append(side ** 2)
    wall.append(np.mean(steps[1:]))  # 第 0 行含建泊松求解器与初始耦合矩阵
    peak.append(top)
    lines.append("{:2d}×{:2d} nm 实测：每步 {:.0f} s，峰值工作集 {:.0f} MB".format(side, side, wall[-1], top))
wall_fit, peak_fit = np.polyfit(area, wall, 1), np.polyfit(area, peak, 1)
for side in (40, 50):
    lines.append("{:2d}×{:2d} nm 外推：每步约 {:.0f} s，峰值工作集约 {:.1f} GB".format(
        side, side, np.polyval(wall_fit, side ** 2), np.polyval(peak_fit, side ** 2) / 1024))

text = "\n".join(lines)
print(text)
with open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write(text + "\n")
