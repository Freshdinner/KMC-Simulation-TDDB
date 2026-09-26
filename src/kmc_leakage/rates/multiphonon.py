"""R4 / R5 —— 电极↔陷阱 非弹性（多声子）俘获与逃逸。

见 docs/rate_equations.md::

    R4 = ∫[E,∞] N_C(E')·f_C(E')·P(E')·c_MP(E') dE'
    R5 = ∫[E,∞] N_C(E')·[1-f_C(E')]·P(E')·e_MP(E') dE'

    c_MP(E') = [(4π)²·r_D³·e²·ℏ·F²]/[Eg·2·m_di]·Σ[p=0,∞] δ(E'-E_p)·L_p
    e_MP(E') = exp[-(E'-E)/k_BT]·c_MP(E')

δ 函数把积分塌缩成对声子数 p 的求和，E_p = E + p·ℏω，故
exp[-(E_p-E)/k_BT] = exp(-p·ℏω/k_BT)。

L_p 用标准 Huang-Rhys 多声子线型::

    L_p = exp[-S(2n̄+1)] · [(n̄+1)/n̄]^{p/2} · I_p(2S√(n̄(n̄+1)))

n̄ 为声子玻色占据数，I_p 为第一类修正贝塞尔函数。该线型满足
L_{-p} = L_p·exp(-p·ℏω/k_BT)，这正是 R4/R5 细致平衡的来源。
"""

import numpy as np
from scipy import constants
from scipy.special import ive


def phonon_occupation(phonon_energy: float, temperature: float) -> float:
    """声子玻色占据数 n̄ = 1/(exp(ℏω/k_BT)-1)。"""
    return 1.0 / np.expm1(phonon_energy / (constants.Boltzmann * temperature))


def phonon_transition_probability(
    p: np.ndarray, huang_rhys: float, phonon_energy: float, temperature: float
) -> np.ndarray:
    """p 声子跃迁概率 L_p（无量纲），p ≥ 0。"""
    p = np.asarray(p, dtype=float)
    kT = constants.Boltzmann * temperature
    n_bar = phonon_occupation(phonon_energy, temperature)
    z = 2.0 * huang_rhys * np.sqrt(n_bar * (n_bar + 1.0))
    # ive(p, z) = I_p(z)·exp(-|z|)，把 exp(z) 与前置指数合并避免溢出；
    # 由 2n̄+1 ≥ 2√(n̄(n̄+1)) 可知合并后的指数恒 ≤ p·ℏω/(2k_BT)
    log_prefactor = -huang_rhys * (2.0 * n_bar + 1.0) + z + p * phonon_energy / (2.0 * kT)
    return np.exp(log_prefactor) * ive(p, z)


def multiphonon_coupling(r_D: np.ndarray, field: np.ndarray, band_gap: float, m_di: float) -> np.ndarray:
    """c_MP 的常数因子 (4π)²·r_D³·e²·ℏ·F²/(2·Eg·m_di) [J·m^3/s]。

    `field` 是陷阱处的局域电场 F [V/m]，F=0 时多声子跃迁速率为 0。
    用隧穿质量 m_di 与局域化半径 r_D（每个缺陷各自的值）。
    """
    r_D = np.asarray(r_D, dtype=float)
    field = np.asarray(field, dtype=float)
    return (
        (4.0 * np.pi) ** 2
        * r_D ** 3
        * constants.elementary_charge ** 2
        * constants.hbar
        * field ** 2
        / (2.0 * band_gap * m_di)
    )


def r4_multiphonon_capture(
    coupling: np.ndarray,
    dos: np.ndarray,
    occupancy: np.ndarray,
    transmission: np.ndarray,
    phonon_weights: np.ndarray,
) -> np.ndarray:
    """R4: 电极→陷阱非弹性（多声子）俘获，见 docs/rate_equations.md。

    参数按声子指数 p 展开，沿最后一维求和：
    `dos` = N_C(E_p)，`occupancy` = f_C(E_p)，`transmission` = P(E_p)，
    `phonon_weights` = L_p，`coupling` = c_MP 的常数因子。
    """
    terms = np.asarray(dos) * np.asarray(occupancy) * np.asarray(transmission) * np.asarray(
        phonon_weights
    )
    return np.asarray(coupling) * terms.sum(axis=-1)


def r5_multiphonon_emission(
    coupling: np.ndarray,
    dos: np.ndarray,
    occupancy: np.ndarray,
    transmission: np.ndarray,
    phonon_weights: np.ndarray,
    phonon_index: np.ndarray,
    phonon_energy: float,
    temperature: float,
) -> np.ndarray:
    """R5: 陷阱→电极非弹性（多声子）逃逸，见 docs/rate_equations.md。

    与 R4 共用同一组 N_C/P/L_p，占据因子换成 [1-f_C(E_p)]，并多一个
    玻尔兹曼因子 exp[-(E_p-E)/k_BT] = exp(-p·ℏω/k_BT)——R5/R4 的细致平衡
    关系就来自这里。
    """
    boltzmann = np.exp(
        -np.asarray(phonon_index, dtype=float) * phonon_energy / (constants.Boltzmann * temperature)
    )
    terms = (
        np.asarray(dos)
        * (1.0 - np.asarray(occupancy))
        * np.asarray(transmission)
        * np.asarray(phonon_weights)
        * boltzmann
    )
    return np.asarray(coupling) * terms.sum(axis=-1)
