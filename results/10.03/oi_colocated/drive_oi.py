"""what-if（不改 src）：生成事件按电荷守恒记账——前驱俘获的那个电子留在 O 间隙离子上（O_i⁻），O_i⁻ 不动、与空位重合。

usage: drive_oi.py SEED [MAX_VAC] [COMPLIANCE_A_cm2]

现模型里新空位出生占据 = 1、净电荷 0，O_i 不带电：每次生成丢掉一个 −e，等价于 O_i⁻ 出生即漂到阳极。这里取另一个极限：
O_i⁻ 留在出生位置。与空位重合时两者合起来就是一个 Z = 0 的缺陷——净电荷 q·(0 − p)，占据时 −e、空态时中性；
空态时留下的是中性复合体，PF 发射没有库仑势垒降低（R8 的 β ∝ √Z = 0），与这个图像自洽。
实现：替换 tddb 模块里的 `layer_bulk_defects`，生成空位的 charge_state 取 0；初始缺陷不变（Z = +1）。
其余照搬 `results/10.03/ext1000/drive.py`（同一份驱动，结果写进本目录）。
"""
import os
import runpy
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage import tddb as tddb_module  # noqa: E402

original = tddb_module.layer_bulk_defects


def layer_bulk_defects(device, layer_index, positions, rng):
    defects = original(device, layer_index, positions, rng)
    return replace(defects, charge_state=np.zeros_like(defects.charge_state))


tddb_module.layer_bulk_defects = layer_bulk_defects
os.environ["TDDB_OUT_DIR"] = HERE
print("what-if: generated vacancies charge_state = 0 (O_i- co-located, charge-conserving generation)", flush=True)
runpy.run_path(os.path.join(ROOT, "results", "10.03", "ext1000", "drive.py"), run_name="__main__")
