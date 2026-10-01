"""速率方程（R1–R8）与 TDDB 缺陷生成速率（T1–T4）。

R 编号与公式见 docs/rate_equations.md（本项目内部编号，不对应外部文献）；
T 编号见 PROJECT_CONTEXT_TDDB.md 第 3 节，实现在 generation.py。
每个函数都是纯函数：输入物理量与参数，返回速率数值，不依赖全局状态、
不做 I/O。全部量为 SI 单位。

约定提醒：
- 势垒一律按零场高度 E_B 处理，**不含镜像力势垒降低**，WKB 积分在未经
  圆化的直线/分段线性势垒上做
- ε_opt **只**出现在 R8；其余公式出现 ε_opt 即为实现错误
- 速率公式里的质量是隧穿质量 m_di，不是裸带边质量 m_cb/m_vb
- 尝试频率 ν0 是同一个配置项，R6/R7/R8 共用
"""

from .elastic_capture import elastic_prefactor, r2_elastic_capture, r3_elastic_emission
from .electrode_tunneling import (
    effective_richardson_constant,
    r1_electrode_electrode,
    richardson_schottky_current_density,
    supply_function,
)
from .generation import (
    CaptureElectrode,
    CaptureTraps,
    bond_break_generation_rate,
    bond_break_probability,
    effective_activation_energy,
    single_electron_capture_rate,
    trap_precursor_capture_rate,
)
from .multiphonon import (
    multiphonon_coupling,
    phonon_transition_probability,
    r4_multiphonon_capture,
    r5_multiphonon_emission,
)
from .pf_emission import pf_beta, r8_pf_emission
from .trap_hopping import (
    miller_abrahams_factor,
    r6_elastic_hop,
    r7_inelastic_hop,
    segmented_overlap_factor,
)
from .wkb import barrier_maximum, wkb_transmission

__all__ = [
    "CaptureElectrode",
    "CaptureTraps",
    "barrier_maximum",
    "bond_break_generation_rate",
    "bond_break_probability",
    "effective_activation_energy",
    "effective_richardson_constant",
    "elastic_prefactor",
    "miller_abrahams_factor",
    "multiphonon_coupling",
    "pf_beta",
    "phonon_transition_probability",
    "r1_electrode_electrode",
    "r2_elastic_capture",
    "r3_elastic_emission",
    "r4_multiphonon_capture",
    "r5_multiphonon_emission",
    "r6_elastic_hop",
    "r7_inelastic_hop",
    "r8_pf_emission",
    "richardson_schottky_current_density",
    "segmented_overlap_factor",
    "single_electron_capture_rate",
    "supply_function",
    "trap_precursor_capture_rate",
    "wkb_transmission",
]
