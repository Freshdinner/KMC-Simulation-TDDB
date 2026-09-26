"""R8 —— 陷阱→导带 Poole-Frenkel 热发射。

见 docs/rate_equations.md::

    R8 = ν · exp(-ED/k_BT) · { (1/β²)·[1+(β-1)·e^β] + 1/2 }
    β  = √[ Z·e³·F / (π·ε0·ε_opt·k_B²·T²) ]

**这里用光学介电常数 ε_opt，不是静态 κ**——κ 只进泊松方程。ν 与 R6/R7
的前置因子是同一个配置项。

两个极限（见测试）：
- F → 0 时 β → 0，花括号项 → 1，R8 退化为无场热发射 ν·exp(-ED/k_BT)
- β ≫ 1 时花括号项 → e^β/β，R8 → ν·exp[-(ED-Δ)/k_BT]/β，
  Δ = √(Z·e³·F/(π·ε0·ε_opt)) = β·k_BT 即场致势垒降低量
"""

import numpy as np
from scipy import constants

# β 小于此值时改用级数展开，避免 1+(β-1)e^β 的灾难性相消
_SERIES_THRESHOLD = 1.0e-3


def pf_beta(
    field: np.ndarray, charge_state: float, eps_opt: float, temperature: float
) -> np.ndarray:
    """R8 的场参数 β = √[Z·e³·F/(π·ε0·ε_opt·k_B²·T²)]（无量纲）。

    β 随场强单调增大，β 大对应高场。`field` 为陷阱处局域电场大小 [V/m]。
    """
    field = np.clip(np.asarray(field, dtype=float), 0.0, None)
    numerator = charge_state * constants.elementary_charge ** 3 * field
    denominator = (
        np.pi
        * constants.epsilon_0
        * eps_opt
        * constants.Boltzmann ** 2
        * temperature ** 2
    )
    return np.sqrt(numerator / denominator)


def pf_bracket(beta: np.ndarray) -> np.ndarray:
    """花括号项 (1/β²)·[1+(β-1)·e^β] + 1/2。"""
    values = np.atleast_1d(np.asarray(beta, dtype=float))
    out = np.empty_like(values)
    small = values < _SERIES_THRESHOLD
    # 展开式 1+(β-1)e^β = Σ_{n≥2} β^n·(n-1)/n!，故花括号项 = 1 + β/3 + β²/8 + …
    b = values[small]
    out[small] = 1.0 + b / 3.0 + b ** 2 / 8.0 + b ** 3 / 30.0
    bl = values[~small]
    out[~small] = (1.0 + (bl - 1.0) * np.exp(bl)) / bl ** 2 + 0.5
    if np.ndim(beta) == 0:
        return float(out[0])
    return out


def r8_pf_emission(
    attempt_frequency: float,
    ED: np.ndarray,
    field: np.ndarray,
    charge_state: float,
    eps_opt: float,
    temperature: float,
) -> np.ndarray:
    """R8: 陷阱→导带 PF 热发射，见 docs/rate_equations.md。

    参数
    ----
    attempt_frequency : 尝试频率 ν [1/s]（与 R6/R7 共用同一配置项）
    ED : 陷阱能级深度（相对导带底）[J]
    field : 陷阱处局域电场 F [V/m]
    charge_state : 陷阱失去电子后留下的电荷 Z
    eps_opt : **光学**介电常数
    """
    kT = constants.Boltzmann * temperature
    beta = pf_beta(field, charge_state, eps_opt, temperature)
    return attempt_frequency * np.exp(-np.asarray(ED, dtype=float) / kT) * pf_bracket(beta)
