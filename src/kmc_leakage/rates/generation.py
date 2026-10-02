"""TDDB 缺陷生成：前驱单步俘获速率 R_1e 与 O 空位生成速率 G。

公式编号 T1–T4 是 TDDB 的编号，见 PROJECT_CONTEXT_TDDB.md 第 3 节，与
docs/rate_equations.md 的 R1–R8 不对应::

    G    = R_1e · P_bb                                       (T1)
    P_bb = exp[ -(E_A - p₀·(2+k)/3·E) / (k_B·T) ]            (T2)
    R_1e = 1/τ_c,12                                          (T4)

- R_1e 是前驱位点俘获一个电子的速率，与项目的 R2（电极→陷阱弹性俘获）是两个量，
  代码里叫 `single_electron_capture_rate`。
- 1/τ_c,12（T5–T11）对全部电子来源求和。式 T5 的 N_{j−1}、f_{j−1} 写的是"电极/陷阱处"：
  - **电极**：复用输运引擎的 R2 + R4，对两侧电极求和，与 kmc.py 处理真实陷阱的口径
    逐项一致。前驱用自己的参数：E_t1 决定能级、局域化半径与 R2 前置因子，E_rel1/ℏω₀
    作为多声子线型 L(m)（T10）的参数。E_rel1 是第一次电子俘获导致 O-Zr-O 键断裂、O 空位
    形成的弛豫能，只在这里进入计算，与输运引擎用的材料 Huang-Rhys 因子 S 无关（ADR-030）。
  - **近旁已占据的陷阱**（初始缺陷与生成的空位，同一物种）：复用引擎的陷阱间通道，R6 的
    几何因子 × R7 的 MA 能量因子，按源陷阱的占据 p 加权（`trap_precursor_capture_rate`，
    ADR-039）。前驱的局域化半径由 E_t1 算，能量差取前驱与源陷阱的平均场能级之差，与内层
    平均场处理陷阱对的口径相同（ADR-038）。这一项沿距离每 r_D/2 衰减 e 倍，调用方逐个源陷阱
    对它做体积分（`tddb.TrapCaptureGeneration`，ADR-042），不与电极项在同一个点上相加。
- G 里的 k 是相对介电常数，取介质的静态 κ（Lorentz 局域场修正，与镜像力无关），
  ε_opt 不进这里。p₀ 是前驱有效偶极矩（T3：p₀ = ½·n·Valence·e·d），配置直接给均值。
- G 的俘获速率作为参数传入：后期换成双电子俘获 R_2e（T4′）时只改调用方。

全部量为 SI 单位。
"""

from typing import NamedTuple, Sequence, Union

import numpy as np
from scipy import constants

from ..bands import fermi_dirac, free_electron_dos
from ..defects import localization_radius
from .elastic_capture import r2_elastic_capture
from .multiphonon import multiphonon_coupling, phonon_transition_probability, r4_multiphonon_capture
from .trap_hopping import r7_inelastic_hop
from .wkb import wkb_transmission


class CaptureElectrode(NamedTuple):
    """一侧电极到前驱的隧穿路径与电极参数（R_1e 对两侧电极求和用）。

    取法同 kmc.py 对真实陷阱的处理：路径从电极界面沿 z 直线连到前驱，势垒取横向
    平均剖面在路径上的导带底（叠层含界面台阶），`m_di_path` 是沿路径按所在层取的
    路径传播质量。
    """

    path: np.ndarray  # (N, P) 电极→前驱路径采样点 z [m]
    barrier: np.ndarray  # (N, P) 路径上的导带底 [J]
    m_di_path: np.ndarray  # (N, P) 路径传播质量 m_di(s) [kg]
    fermi_level: float  # 电极费米能级 [J]（全局零点 = 阴极费米能级）
    band_bottom: float  # 电极导带底 [J]
    mass: float  # 电极侧电子质量 [kg]


def trap_precursor_capture_rate(
    attempt_frequency: np.ndarray,
    distance: np.ndarray,
    trap_r_D: np.ndarray,
    precursor_r_D: float,
    trap_energy: np.ndarray,
    precursor_energy: np.ndarray,
    trap_occupancy: np.ndarray,
    temperature: Union[float, np.ndarray],
) -> np.ndarray:
    """前驱从一个近旁陷阱俘获电子的速率 [1/s]：式 T4/T5 的陷阱项，p·R7(陷阱 → 前驱)。

    就是引擎的陷阱间通道（docs/rate_equations.md 的 R6/R7）把目标换成前驱：
    p·ν0·exp[−r·(1/r_D,陷阱 + 1/r_D,前驱)]·Θ，Θ = exp[−(E_前驱 − E_陷阱)/k_BT]（上坡）或 1。
    p 是源陷阱的平均场占据（式 T5 里 f_{j−1}），能量差取两端平均场能级之差，与内层平均场
    处理陷阱对的口径相同（ADR-038）；前驱是空的虚拟位点，r_D,前驱 由 E_t1 算。名字避开项目的
    R2（电极→陷阱弹性俘获），见 PROJECT_CONTEXT_TDDB.md §2 第 1 条。
    """
    return np.asarray(trap_occupancy, dtype=float) * r7_inelastic_hop(
        np.asarray(attempt_frequency, dtype=float),
        distance,
        trap_r_D,
        precursor_r_D,
        trap_energy,
        precursor_energy,
        temperature,
    )


def single_electron_capture_rate(
    conduction_band: np.ndarray,
    field: np.ndarray,
    electrodes: Sequence[CaptureElectrode],
    ionization_energy: float,
    relaxation_energy: float,
    phonon_energy: float,
    band_gap: float,
    m_di: float,
    temperature: Union[float, np.ndarray],
    n_phonon: int,
) -> np.ndarray:
    """R_1e 的电极项：前驱从两侧电极单步俘获一个电子的速率 [1/s]（式 T4，T5–T11）。

    R_1e = Σ_电极 [R2 + R4] + Σ_陷阱 p·R7，这里给前一项：两侧电极的弹性俘获 R2 与多声子俘获 R4。
    陷阱项逐对由 `trap_precursor_capture_rate` 给出，调用方按源陷阱积分（ADR-042）。

    `conduction_band` 是前驱处的导带底 E_C = B − e·φ，前驱能级 E = E_C − E_t1；
    `field` 是前驱处局域电场大小，进 R4 的 c_MP。`ionization_energy` = E_t1（能级
    相对局部导带底的深度）、`relaxation_energy` = E_rel1 是前驱参数，
    `phonon_energy` = ℏω₀、`band_gap`、`m_di` 取宿主介质的值。R4 对声子数
    p = 0..n_phonon−1 求和，n_phonon 要覆盖前驱的多声子线型：E_rel1/ℏω₀ ≈ 1 时引擎的
    12 项已收敛，E_rel1 = 1.0 eV（E_rel1/ℏω₀ = 25）时要约 100 项（ADR-030）。
    `temperature` 可以是标量，也可以是与 `conduction_band` 等长的逐前驱温度（Phase 4 按
    单元局域温度求值，§6.6），声子线型与电极费米分布都取这个温度。
    """
    energy = np.asarray(conduction_band, dtype=float) - ionization_energy
    temperature = np.asarray(temperature, dtype=float)
    column = temperature[..., None]  # 标量 → (1,)，逐前驱 (N,) → (N, 1)，对声子数广播
    phonon_index = np.arange(n_phonon)
    # 前驱俘获线型 L(m)（式 T10）的参数取 E_rel1/ℏω₀，与材料的 Huang-Rhys 因子 S 无关
    lineshape_parameter = relaxation_energy / phonon_energy
    weights = phonon_transition_probability(
        phonon_index, lineshape_parameter, phonon_energy, column
    )
    energies_p = energy[:, None] + phonon_index * phonon_energy
    precursor_r_D = localization_radius(ionization_energy, m_di)
    coupling = multiphonon_coupling(precursor_r_D, field, band_gap, m_di)

    rate = np.zeros(energy.shape)
    for side in electrodes:
        transmission = wkb_transmission(side.path, side.barrier, energy, side.m_di_path)
        occupancy = fermi_dirac(energy, side.fermi_level, temperature)
        rate = rate + r2_elastic_capture(
            energy - side.band_bottom, ionization_energy, side.mass, m_di, occupancy, transmission
        )
        transmission_p = wkb_transmission(
            side.path[:, None, :], side.barrier[:, None, :], energies_p, side.m_di_path[:, None, :]
        )
        occupancy_p = fermi_dirac(energies_p, side.fermi_level, column)
        dos_p = free_electron_dos(energies_p - side.band_bottom, side.mass)
        rate = rate + r4_multiphonon_capture(coupling, dos_p, occupancy_p, transmission_p, weights)
    return rate


def effective_activation_energy(
    activation_energy: float, dipole_moment: float, kappa: float, field: np.ndarray
) -> np.ndarray:
    """场加速后的断键有效势垒 E_A − p₀·(2+k)/3·E [J]（式 T2 指数里的那一项）。

    `kappa` 是静态相对介电常数（Lorentz 局域场因子 (2+k)/3），`field` 是局域
    电场大小 [V/m]，`dipole_moment` = p₀ [C·m]。
    """
    lorentz = (2.0 + kappa) / 3.0
    return activation_energy - dipole_moment * lorentz * np.asarray(field, dtype=float)


def bond_break_probability(
    activation_energy: float,
    dipole_moment: float,
    kappa: float,
    field: np.ndarray,
    temperature: Union[float, np.ndarray],
) -> np.ndarray:
    """断键概率 P_bb = exp[−(E_A − p₀·(2+k)/3·E)/k_BT]（式 T2）。

    有效势垒降到 0 以下时断键不再需要热激活，P_bb 饱和为 1，生成率只剩俘获
    速率限制（PROJECT_CONTEXT_TDDB.md §8.1）。`temperature` 可以是与 `field` 等长的
    逐前驱温度。
    """
    barrier = effective_activation_energy(activation_energy, dipole_moment, kappa, field)
    return np.exp(-np.clip(barrier, 0.0, None) / (constants.Boltzmann * temperature))


def bond_break_generation_rate(
    capture_rate: np.ndarray,
    activation_energy: float,
    dipole_moment: float,
    kappa: float,
    field: np.ndarray,
    temperature: Union[float, np.ndarray],
) -> np.ndarray:
    """单个前驱的 O 空位生成速率 G = R_1e·P_bb [1/s]（式 T1）。

    `capture_rate` 是前驱俘获电子的速率：单步注入机制传 R_1e
    （`single_electron_capture_rate`），后期双电子机制传 R_2e。体积生成率
    N_T·G 由调用方乘前驱浓度得到（§6.1 模型 C）。
    """
    return np.asarray(capture_rate, dtype=float) * bond_break_probability(
        activation_energy, dipole_moment, kappa, field, temperature
    )
