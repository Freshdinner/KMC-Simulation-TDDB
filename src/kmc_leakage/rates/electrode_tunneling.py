"""R1 —— 电极↔电极（直接隧穿 / FN 隧穿 / 热电子发射，统一处理）。

见 docs/rate_equations.md 的 R1 一节。同一个积分同时覆盖两种物理图像，
按势垒最高点能量做阈值切分后分别统计。

能量零点：**负极费米能级**（正极费米能级位于 -eU）。供给函数

    S(E_t) = ln{ (1+exp(-E_t/k_BT)) / (1+exp(-(E_t+eU)/k_BT)) }

两个指数同号。积分变量 E_t 是电子的横向能量，不能为负，**积分下限取
0**；方向上只计负极→正极，正极→负极的直接隧穿忽略。

两种质量各司其职：前置因子里的 m_e 是**发射极**电子质量（自由电子
质量），P(E_t) 内部的 WKB 计算用**隧穿质量 m_di**。
"""

from typing import Tuple

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid

from .wkb import barrier_maximum, wkb_transmission

# 数值积分参数：积分变量是横向能量，下限恒为 0；上限取势垒顶再往上
# _UPPER_MARGIN_KT 个 k_BT，此处供给函数已衰减到可忽略。收敛性由
# tests/test_rates_limits.py 检验。
_UPPER_MARGIN_KT = 30.0


def supply_function(energy: np.ndarray, voltage: float, temperature: float) -> np.ndarray:
    """R1 的供给函数 ln{(1+exp(-E/k_BT))/(1+exp(-(E+eU)/k_BT))}（无量纲）。

    数值上用 logaddexp 实现，避免大指数溢出。U=0 时恒为 0。
    """
    kT = constants.Boltzmann * temperature
    eU = constants.elementary_charge * voltage
    energy = np.asarray(energy, dtype=float)
    return np.logaddexp(0.0, -energy / kT) - np.logaddexp(0.0, -(energy + eU) / kT)


def r1_electrode_electrode(
    z: np.ndarray,
    conduction_band: np.ndarray,
    area: float,
    voltage: float,
    temperature: float,
    m_emitter: float,
    m_di: float,
    n_points: int = 256,
) -> Tuple[float, float]:
    """R1: 电极↔电极统一速率，见 docs/rate_equations.md。

    R1 = [A·m_e·k_BT/(2π²ℏ³)] · ∫[0,∞] P(E_t)·S(E_t) dE_t

    返回 (直接/FN隧穿部分, 热电子发射部分)，单位 1/s。切分阈值是势垒沿
    隧穿路径的最高点能量：E_t < E_barrier,max 计入直接/FN 隧穿，
    E_t ≥ E_barrier,max 计入热电子发射。只计负极→正极方向。

    参数
    ----
    z : (n,) 介质内 z 坐标 [m]
    conduction_band : (n,) 导带底剖面 E_C(z) [J]（全局能量零点 = 负极费米能级）
    area : 电极面积 [m^2]
    m_emitter : 发射极电子质量 [kg]（自由电子质量）
    m_di : 隧穿有效质量 [kg]，只进 P(E_t) 的 WKB 计算
    n_points : 阈值两侧各自的能量积分点数
    """
    kT = constants.Boltzmann * temperature
    prefactor = area * m_emitter * kT / (2.0 * np.pi ** 2 * constants.hbar ** 3)

    e_top = barrier_maximum(conduction_band)
    e_max = e_top + _UPPER_MARGIN_KT * kT

    def segment(lo: float, hi: float) -> float:
        energies = np.linspace(lo, hi, n_points)
        transmission = wkb_transmission(z, conduction_band, energies, m_di)
        integrand = transmission * supply_function(energies, voltage, temperature)
        return float(prefactor * trapezoid(integrand, energies))

    return segment(0.0, e_top), segment(e_top, e_max)


def effective_richardson_constant(m_cb: float) -> float:
    """有效理查森常数 A* = 4π·m_cb·q·k_B²/h³ [A/(m^2·K^2)]。

    用**介质导带底有效质量 m_cb**（不是隧穿质量 m_di），见
    docs/rate_equations.md R1 一节的独立验证小节。
    """
    return (
        4.0
        * np.pi
        * m_cb
        * constants.elementary_charge
        * constants.Boltzmann ** 2
        / constants.Planck ** 3
    )


def richardson_schottky_current_density(
    temperature: float, voltage: float, barrier_height: float, m_cb: float
) -> float:
    """Richardson-Schottky 热电子发射电流密度 [A/m^2]，R1 热发射部分的独立对照。

    J_TE = A*·T²·[exp(-E_B/k_BT) - exp(-(E_B+eU)/k_BT)]

    写成两项之差是为了和本项目的能量零点约定对齐：E_B 是**负极侧**零场
    势垒高度，正极侧同一势垒顶相对正极费米能级则高出 E_B+eU。提出公因子
    即得 A*T²exp(-E_B/k_BT)·[1-exp(-eU/k_BT)]，与写成 [exp(qV/k_BT)-1]
    的形式是同一式子，区别只在 φ_B 从哪一侧参考。

    不考虑镜像力势垒降低（Δφ），按理想零场势垒处理，与 KMC 侧口径一致。
    """
    kT = constants.Boltzmann * temperature
    eU = constants.elementary_charge * voltage
    return float(
        effective_richardson_constant(m_cb)
        * temperature ** 2
        * (np.exp(-barrier_height / kT) - np.exp(-(barrier_height + eU) / kT))
    )
