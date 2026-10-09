"""把 examples/zro2_tddb.yaml 的横向盒子改成 L×L nm：泊松网格保持 hx = hy = 0.25 nm（ADR-032），生成单元保持
1.25 nm 见方（20 nm / 16），其余参数不变。"""
import os
import sys
from dataclasses import replace

from scipy import constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kmc_leakage.io import load_config  # noqa: E402

GRID_STEP_NM = 0.25
CELL_NM = 1.25


def box_config(side_nm: float, voltage: float = 3.6):
    config = load_config(os.path.join(ROOT, "examples", "zro2_tddb.yaml"))
    side = side_nm * constants.nano
    n_grid = int(round(side_nm / GRID_STEP_NM))
    n_cell = int(round(side_nm / CELL_NM))
    grid = replace(config.simulation.grid, nx=n_grid, ny=n_grid)
    _, _, gz = config.tddb.generation_grid
    return replace(
        config,
        device=replace(config.device, Lx=side, Ly=side),
        simulation=replace(config.simulation, voltage=voltage, grid=grid),
        tddb=replace(config.tddb, generation_grid=(n_cell, n_cell, gz)),
    )
