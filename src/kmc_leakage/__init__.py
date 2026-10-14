"""MIM 电容器漏电流的动力学蒙特卡洛（KMC）仿真。

物理规格见 docs/physics_model.md 与 docs/rate_equations.md。
内部计算一律 SI 单位（J, m, K, C, s），配置文件的 eV/nm/cm^-3 只在
io.py 的读入边界换算。

能量零点约定：**负极费米能级 = 0**。正极费米能级在偏压 U 下位于 -eU。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
