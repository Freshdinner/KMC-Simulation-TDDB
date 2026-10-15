"""temperature.md 式 (15)（镜像热源解析解）与 thermal.solve_temperature 的对拍，以及 1000 K 所需功率的估算。

不改 src/。网格步长与 zro2_tddb.yaml 相同（hx = hy = 0.25 nm、hz = 0.1 nm、Lz = 8 nm），横向盒子缩到 30 nm
（两电极之间点源的横向影响按 exp(−πρ/Lz) 衰减，30 nm 处约 10⁻⁵）。
"""

from dataclasses import replace

import numpy as np
from scipy import constants

from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.io import load_config
from kmc_leakage.thermal import power_map, solve_temperature

NM = constants.nano
K_TH = 2.0  # W/(m·K)，ADR-040
T_AMB = 298.15
VOLTAGE = 3.6
G0 = 2 * constants.e ** 2 / constants.h

config = load_config("examples/zro2_tddb.yaml")
device = replace(config.device, Lx=30 * NM, Ly=30 * NM)
grid = replace(config.simulation.grid, nx=120, ny=120, nz=80)
solver = PoissonSolver(device, grid)
L = device.Lz


def analytic(points, sources, powers, n_img=100):
    """式 (15)：两端恒温平板之间点热源的镜像和，另加 ±1 个横向周期像（与求解器的周期边界对应）。"""
    points = np.atleast_2d(points)
    dT = np.zeros(len(points))
    n = np.arange(-n_img, n_img + 1)
    for s, p in zip(sources, powers):
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                dx = points[:, 0, None] - s[0] - ix * device.Lx
                dy = points[:, 1, None] - s[1] - iy * device.Ly
                rho2 = dx ** 2 + dy ** 2
                zp = points[:, 2, None] - s[2] + 2 * n * L  # 同号像（n = 0 为源本身）
                zm = points[:, 2, None] + s[2] + 2 * n * L  # 异号像
                with np.errstate(divide="ignore"):
                    dT += p / (4 * np.pi * K_TH) * (np.sum(1 / np.sqrt(rho2 + zp ** 2), axis=1)
                                                    - np.sum(1 / np.sqrt(rho2 + zm ** 2), axis=1))
    return dT


def grid_temperature(sources, powers):
    return solve_temperature(solver, power_map(solver, np.asarray(sources), np.asarray(powers)), K_TH, T_AMB)


# 1. 单个点源 1 µW，放在网格节点上（盒子中心、膜中部）
P1 = 1e-6
src = np.array([15 * NM, 15 * NM, 4 * NM])
T = grid_temperature([src], [P1])
print("== 1. 单点源 1 µW，求解器 vs 式 (15) ==")
for d_nm in (0.25, 0.5, 1.0, 2.0, 3.0):
    for axis, name in ((0, "横向"), (2, "z 向")):
        p = src.copy()
        p[axis] += d_nm * NM
        num = float(solver.interpolate(T, p[None])[0]) - T_AMB
        ana = float(analytic(p, [src], [P1])[0])
        print(f"  {name} {d_nm:4.2f} nm: 求解器 {num:9.3f} K  式(15) {ana:9.3f} K  相对差 {num / ana - 1:+.2%}")
self_num = float(solver.interpolate(T, src[None])[0]) - T_AMB
regular = float(analytic(src + np.array([1e-14, 0, 0]), [src], [P1])[0]) - P1 / (4 * np.pi * K_TH * 1e-14)
r_eff = P1 / (4 * np.pi * K_TH * (self_num - regular))
print(f"  源点处：求解器 ΔT = {self_num:.1f} K；等效截断半径 r_eff = {r_eff / NM:.3f} nm"
      f"（式 (15) 在源点发散，要取截断半径）")
for r0 in (0.1, 0.2, 0.3):
    print(f"  式 (15) 截断 r0 = {r0} nm：ΔT = {P1 / (4 * np.pi * K_TH * r0 * NM) + regular:.1f} K")
print(f"  ⇒ 单个陷阱升 1000 K 要 P ≈ {1000 / self_num * P1 * 1e6:.2f} µW（网格），"
      f"r0 = 0.2 nm 时 {1000 / (1 / (4 * np.pi * K_TH * 0.2 * NM)) * 1e6:.2f} µW")

# 2. 贯穿细丝：空位链 0.26 nm 间距，从负极到正极，功率均分到每个空位（扩散型细丝的焦耳热）
z_chain = np.arange(0.26, 7.75, 0.26) * NM
chain = np.column_stack([np.full_like(z_chain, 15.13 * NM), np.full_like(z_chain, 14.91 * NM), z_chain])
P_unit = 1e-3
Tc = grid_temperature(chain, np.full(len(chain), P_unit / len(chain)))
peak_per_W = (Tc.max() - T_AMB) / P_unit
ana_chain = analytic(chain[len(chain) // 2] + np.array([0.5 * NM, 0, 0]), chain, np.full(len(chain), P_unit / len(chain)))[0]
num_chain = float(solver.interpolate(Tc, (chain[len(chain) // 2] + np.array([0.5 * NM, 0, 0]))[None])[0]) - T_AMB
print("\n== 2. 贯穿空位链（{} 个，0.26 nm 间距），总功率均分 ==".format(len(chain)))
print(f"  离链 0.5 nm 处（膜中部）1 mW 下：求解器 {num_chain:.0f} K，式 (15) {ana_chain:.0f} K")
print(f"  峰值升温 / 总功率 = {peak_per_W:.3g} K/W；升 1000 K 要 {1000 / peak_per_W * 1e6:.1f} µW"
      f" ⇒ 全部 I·V 放在链上时 I = {1000 / peak_per_W / VOLTAGE * 1e6:.1f} µA")
P_G0 = G0 * VOLTAGE ** 2
print(f"  一个量子电导 G₀·V² = {P_G0 * 1e3:.2f} mW 全部放在链上：峰值升温 {peak_per_W * P_G0:.0f} K（线性外推，无热导率温变）")

# 3. TAT 链的通量上限：相邻空位 R6/R7 速率 ν0·exp(−2r/r_D)，单个空位最多每秒过 ν0 量级个电子
nu0 = 1e13
m_di = 1.2 * constants.m_e
r_D = constants.hbar / np.sqrt(2 * m_di * 1.15 * constants.e)
print("\n== 3. TAT 链能承载的电流与升温上限 ==")
print(f"  r_D = {r_D / NM:.3f} nm")
for r in (0.26, 0.30, 0.50):
    k = nu0 * np.exp(-2 * r * NM / r_D)
    I = constants.e * k / 4  # 一维排斥跃迁链半满时的最大通量 k/4
    print(f"  间距 {r:.2f} nm：跃迁 {k:.2e} s⁻¹，链电流 ≤ {I * 1e9:6.2f} nA，"
          f"全部 I·V 放在链上升温 ≤ {peak_per_W * I * VOLTAGE:.3g} K")
I_nu0 = constants.e * nu0
print(f"  绝对上限 e·ν0 = {I_nu0 * 1e6:.2f} µA：升温 ≤ {peak_per_W * I_nu0 * VOLTAGE:.0f} K")

# 4. 热反馈起作用的门槛：有效势垒 0.70 eV（3.6 V，E_A = 1.30 eV、p₀ = 1 eÅ）下 P_bb 抬高 10 倍要升多少 K
E_eff = 0.70 * constants.e
kB = constants.k
dT10 = 1 / (1 / T_AMB - kB * np.log(10) / E_eff) - T_AMB
print("\n== 4. 热正反馈门槛 ==")
print(f"  P_bb ×10 需 ΔT = {dT10:.1f} K；×10³ 需 {1 / (1 / T_AMB - kB * np.log(1e3) / E_eff) - T_AMB:.0f} K")
print(f"  尖端单点源 ΔT = {dT10:.0f} K 要 P ≈ {dT10 / self_num * P1 * 1e9:.0f} nW（网格截断）")
J_box = 100.0  # A/cm²，ADR-055/056 停时的量级
I_box = J_box * 50e-7 * 50e-7
print(f"  对照：50×50 nm 盒子 J = 100 A/cm² 时总电流 {I_box * 1e9:.1f} nA，U·I = {I_box * VOLTAGE * 1e9:.1f} nW")

# 5. 热扩散时间（稳态近似的依据）
rho_cp = 5.68e3 * 455.0  # ZrO₂ 密度 × 比热 [J/(m³·K)]
D = K_TH / rho_cp
print("\n== 5. 热扩散时间 ==")
print(f"  D = {D:.2e} m²/s；Lz = 8 nm：τ = {L ** 2 / D:.1e} s；1 nm 热点：τ = {(1 * NM) ** 2 / D:.1e} s")
