"""试算：前驱位点浓度 N_T 从 10²⁰ 提到 10²¹ cm⁻³，看生成前沿能不能往膜中部推进、空位能不能连通阴极和阳极。

usage: drive.py SEED

不改代码，也不改 examples/zro2_tddb.yaml：读入配置时换掉三项，再原样运行 ../discrete/drive.py（输出格式、快照与它相同）：
- `tddb.precursor_concentration` = 10²¹ cm⁻³（50×50 nm 盒子约 17500 个前驱位点）；
- `tddb.breakdown_current_density` = ∞：这次看的是空位的空间分布，电流越过 10 A/cm² 不停跑；
- `tddb.stop_time` = 10⁷ s（与 ../discrete/long 相同），空位数上限沿用配置的 3000。
其余工况与 ../discrete/long 相同（50×50 nm、3.6 V / 25 °C、E_A = 1.30 ± 0.10 eV、开热反馈）。结果写在本目录。
"""
import os
import runpy
import sys
from dataclasses import replace

from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))

from kmc_leakage import io  # noqa: E402

PRECURSOR_CONCENTRATION = 1.0e21 / constants.centi ** 3  # [m^-3]
load_config = io.load_config


def what_if_config(path):
    config = load_config(path)
    return replace(config, tddb=replace(config.tddb, precursor_concentration=PRECURSOR_CONCENTRATION,
                                        breakdown_current_density=float("inf")))


io.load_config = what_if_config
sys.argv = [os.path.join(HERE, "..", "discrete", "drive.py"), sys.argv[1], "1e7", HERE]
runpy.run_path(sys.argv[0], run_name="__main__")
