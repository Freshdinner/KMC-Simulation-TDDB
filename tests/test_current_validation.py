"""F04 —— 三估计量、全局守恒、验证状态与分层归因（ADR-021 F04 / ADR-014）。

- 全局电荷守恒 (I_C−I_A)Δt = q·ΔN_inside（构造精确，捕获计数 bug）；
- 验证状态：正向且窗口/局部稳态达标 → validated；非正向收集 → 未通过验证；
  批次均值法相对标准误超限（批次间电流不一致）→ 未通过验证；
- 分层归因闭合 Σ_{ℓ,Γ} I_{ℓ,Γ} = I_TAT+I_PF，I_total = Σ I_{ℓ,Γ} + I_R1；
- 界面缺陷单独且只计一次、R1 单列；贯通到 sweep / ensemble / CSV；
- 块 4：R8_eff 沿用 Phase 3 的界面单侧场（不另取场）。
"""

import csv

import numpy as np
import pytest
from scipy import constants

from conftest import make_config, make_defect_config, make_device
from kmc_leakage.config import GridConfig, SweepConfig
from kmc_leakage.kmc import (
    _BATCH_REL_TOL,
    MECHANISMS,
    KMCSimulation,
    _validate_result,
    batch_relative_error,
)
from test_pf_collection import _zaz_device, _zaz_ensemble

# ZAZ 4/1/4 nm 的界面在 4、5 nm，需 hz=0.25 nm（nz=36）才对齐到节点（ADR-011）
_ZAZ_GRID = GridConfig(16, 16, 36)

Q = constants.elementary_charge


def _run(**kwargs):
    return KMCSimulation(make_config(**kwargs)).run()


# --------------------------------------------------------------------------
# 全局电荷守恒（F04）
# --------------------------------------------------------------------------
def test_global_charge_conservation_identity():
    """(I_C − I_A)·Δt = q·ΔN_inside：负极净注入 − 正极净收集 = 器件内电子数变化。

    这是逐事件记账的恒等式：只要阴/正极事件都数全，两边精确相等（整数）。
    """
    result = _run(voltage=3.0, n_events=2000, seed=5)
    assert result.elapsed_time > 0.0
    lhs = (result.current_counts_negative - result.current_counts) * result.elapsed_time
    assert lhs == pytest.approx(Q * result.delta_n_inside, rel=1e-9, abs=1e-30)
    # ΔN_inside 也确实是个整数（器件内只有陷阱电子，无导带态）
    assert isinstance(result.delta_n_inside, int)


# --------------------------------------------------------------------------
# 验证状态（F04）
# --------------------------------------------------------------------------
def test_forward_point_with_enough_events_is_validated():
    """正向、窗口充分、占据与条件稳态 p̂ 接近 → 通过验证。"""
    result = _run(voltage=4.0, n_events=3000, seed=13)
    assert result.validated is True
    assert result.validation_notes == ()
    assert result.measured_events >= 100


def test_zero_bias_is_not_validated():
    """零偏压不满足正向收集假设 → 标未通过验证并给出原因。"""
    result = _run(voltage=0.0, n_events=800, seed=7)
    assert result.validated is False
    assert any("正向收集" in note for note in result.validation_notes)


def test_batch_relative_error_is_zero_for_steady_current():
    """各批次电流相同（Q_k ∝ T_k）→ 批次均值法相对标准误为 0。"""
    duration = np.array([1.0, 2.0, 0.5, 3.0])
    charge = 7.0 * duration
    assert batch_relative_error(charge, duration, 7.0) == pytest.approx(0.0, abs=1e-15)


def test_batch_relative_error_flags_single_batch_spike():
    """一个批次的电流是其余的 100 倍（低频通路只在一段里被填充）→ 相对标准误大。

    10 个等时长批次，9 个电流为 1、1 个为 100：Ī = 10.9，
    SE = √[10/9·(9·9.9² + 89.1²)]/10 = 9.9，相对标准误 = 9.9/10.9 ≈ 0.908。
    """
    duration = np.ones(10)
    charge = np.ones(10)
    charge[4] = 100.0
    mean = charge.sum() / duration.sum()
    rel = batch_relative_error(charge, duration, mean)
    assert rel == pytest.approx(9.9 / 10.9, rel=1e-12)
    assert rel > _BATCH_REL_TOL


def test_batch_relative_error_needs_two_batches():
    """有效批次（时长 > 0）不足 2 个时无从估计，返回 0。"""
    assert batch_relative_error(np.array([3.0, 0.0]), np.array([1.0, 0.0]), 3.0) == 0.0


def test_batch_inconsistency_fails_validation():
    """批次相对标准误超过上限 → 未通过验证并给出原因；其余判据达标时仅此一条。"""
    args = dict(
        n_defects=10, forward=True, elapsed=1e-6, measured_events=3200,
        occ_mean_abs_dev=0.0, n_positive_events=0, current=1e-15, current_counts_total=0.0,
    )
    ok, notes = _validate_result(batch_rel_error=0.5 * _BATCH_REL_TOL, **args)
    assert ok is True and notes == ()
    ok, notes = _validate_result(batch_rel_error=2.0 * _BATCH_REL_TOL, **args)
    assert ok is False
    assert len(notes) == 1 and "批次间电流不一致" in notes[0]


def test_r1_only_device_is_validated():
    """全器件无缺陷（R1-only）没有统计估计量要验证，R1 是确定性解 → 通过。"""
    device = make_device(defects=make_defect_config(concentration_cm3=0.0))
    result = KMCSimulation(make_config(device=device, voltage=3.0, n_events=200)).run()
    assert len(result.defects) == 0
    assert result.validated is True
    assert result.validation_notes == ()


# --------------------------------------------------------------------------
# 分层归因（ADR-014）
# --------------------------------------------------------------------------
def test_by_layer_closes_single_layer():
    """单层：by_layer 只有 layer_0，之和 = TAT+PF，加 R1 = 总电流。"""
    result = _run(voltage=3.0, n_events=1000, seed=9)
    trap = result.by_mechanism["tat"] + result.by_mechanism["pf"]
    r1 = result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]
    assert list(result.by_layer) == ["layer_0"]
    assert sum(result.by_layer.values()) == pytest.approx(trap, rel=1e-12)
    assert sum(result.by_layer.values()) + r1 == pytest.approx(result.current, rel=1e-12)


def test_by_layer_zaz_separates_layers_and_interfaces():
    """ZAZ：by_layer 分出层0/层2体缺陷与界面0/界面1（各一栏、只计一次）；闭合成立。"""
    config = make_config(
        device=_zaz_device(),
        voltage=3.0, n_events=800, seed=3, grid=_ZAZ_GRID,
    )
    result = KMCSimulation(config, defects=_zaz_ensemble()).run()
    assert set(result.by_layer) == {"layer_0", "layer_2", "interface_0", "interface_1"}
    trap = result.by_mechanism["tat"] + result.by_mechanism["pf"]
    assert sum(result.by_layer.values()) == pytest.approx(result.by_defect.sum(), rel=1e-12)
    assert sum(result.by_layer.values()) == pytest.approx(trap, rel=1e-12)
    # 界面缺陷只计一次：4 个缺陷分别落在 4 个栏里，逐缺陷之和 = 分层之和
    assert result.by_defect.size == 4


def test_by_layer_threads_through_sweep_and_csv(tmp_path):
    """by_layer + F04 诊断从单点贯通到 sweep 与 CSV。"""
    from kmc_leakage.sweep import run_sweep, write_csv

    config = make_config(
        n_events=150,
        sweep=SweepConfig(voltages=np.array([2.0, 3.0]), temperatures=np.array([298.15])),
    )
    result = run_sweep(config)
    assert "layer_0" in result.by_layer
    assert result.by_layer["layer_0"].shape == (1, 2)
    # 分层归因 + R1 = 总电流密度（逐点）
    r1 = result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]
    layer_sum = sum(result.by_layer.values())
    assert np.allclose(layer_sum + r1, result.current_density, rtol=1e-12)

    path = tmp_path / "sweep.csv"
    write_csv(result, str(path))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "J_layer_0_A_cm2" in rows[0]
    assert "J_direct_A_cm2" in rows[0] and "validated" in rows[0]
    assert "delta_n_inside" in rows[0] and "measured_events" in rows[0]


def test_ensemble_sweep_carries_by_layer_and_cross_check(tmp_path):
    """系综扫描聚合 by_layer 与 ΔJ 交叉校验、逐点验证计数，并贯通到 CSV。"""
    from kmc_leakage.ensemble import run_ensemble_sweep, write_ensemble_csv

    config = make_config(
        n_events=150, seed=1,
        sweep=SweepConfig(voltages=np.array([3.0]), temperatures=np.array([298.15])),
        ensemble=_ensemble_of(2),
    )
    result = run_ensemble_sweep(config)
    assert "layer_0" in result.by_layer
    assert result.delta_j.shape == (1, 1)
    assert result.delta_j_sem.shape == (1, 1)
    assert result.n_validated.shape == (1, 1)
    assert 0 <= int(result.n_validated[0, 0]) <= 2

    path = tmp_path / "ensemble.csv"
    write_ensemble_csv(result, str(path))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "J_layer_0_A_cm2" in rows[0]
    assert "dJ_cross_A_cm2" in rows[0] and "dJ_cross_sem_A_cm2" in rows[0]
    assert "n_validated" in rows[0] and "n_realizations" in rows[0]
    assert int(rows[0]["n_realizations"]) == 2


def _ensemble_of(count):
    from kmc_leakage.config import EnsembleConfig

    return EnsembleConfig(seeds=tuple(range(1, count + 1)))


# --------------------------------------------------------------------------
# 块 4：R8_eff 沿用 Phase 3 的界面单侧场（ADR-017 §2）
# --------------------------------------------------------------------------
def test_r8_uses_interface_sided_field():
    """块 4：R8_eff 沿用 Phase 3 的界面单侧场（不另取场）。

    R4/R5 的 c_MP 与 R8 都读 `_electrostatics` 返回的同一份 field_at。把界面缺陷的
    field_side 从 ±1（单侧差分）改成 0（中心差分），界面缺陷处的场随之改变、体缺陷
    不变——证明这份场确实按 field_side 取，R8_eff 不另立一套。
    """
    from dataclasses import replace

    from kmc_leakage.kmc import _B_R8

    ens_sided = _zaz_ensemble()  # 界面缺陷 field_side = ±1
    ens_central = replace(ens_sided, field_side=np.zeros(4, dtype=int))  # 全用中心差分
    device = _zaz_device()
    occ = np.zeros(4, dtype=bool)

    def field_and_r8(ens):
        sim = KMCSimulation(
            make_config(device=device, voltage=3.0, n_events=5, seed=1, grid=_ZAZ_GRID),
            defects=ens,
        )
        _, field_at, _ = sim._electrostatics(occ)
        blocks, _ = sim._rates(occ)
        return field_at, blocks[_B_R8]

    field_sided, r8_sided = field_and_r8(ens_sided)
    field_central, r8_central = field_and_r8(ens_central)

    # 界面缺陷（index 2,3）单侧 vs 中心差分场不同；体缺陷（0,1）相同
    assert not np.allclose(field_sided[2:], field_central[2:], rtol=1e-6)
    assert np.allclose(field_sided[:2], field_central[:2], rtol=1e-12, atol=0.0)
    # R8_eff 跟着这份场走：界面缺陷的 R8 随场改变（正极侧界面1 P_A=1，纯粹是场变）
    assert not np.isclose(r8_sided[3], r8_central[3], rtol=1e-6)
