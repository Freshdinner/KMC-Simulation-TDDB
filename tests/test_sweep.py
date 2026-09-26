"""I-V / I-T 扫描与结果输出的检验。"""

import csv

import numpy as np
import pytest
from scipy import constants

from conftest import EV, make_config, make_defect_config, make_device
from kmc_leakage.config import DefectLevelConfig, DefectSpatialConfig, SweepConfig
from kmc_leakage.defects import generate_defects
from kmc_leakage.kmc import MECHANISMS
from kmc_leakage.sweep import run_sweep, write_csv


def _small_sweep():
    return make_config(
        n_events=150,
        sweep=SweepConfig(
            voltages=np.array([2.0, 3.0]), temperatures=np.array([298.15, 348.15])
        ),
    )


def test_sweep_shapes_and_monotonicity():
    result = run_sweep(_small_sweep())
    assert result.current_density.shape == (2, 2)  # (温度点, 电压点)
    for name in MECHANISMS:
        assert result.by_mechanism[name].shape == (2, 2)
    # 同一温度下电流随电压单调上升
    assert np.all(np.diff(result.current_density, axis=1) > 0.0)
    # 同一电压下电流随温度单调上升
    assert np.all(np.diff(result.current_density, axis=0) > 0.0)


def test_sweep_reuses_one_defect_realization():
    """整个扫描必须是同一个器件，否则不同工作点无法比较。"""
    result = run_sweep(_small_sweep())
    positions = result.defects.positions
    for row in result.results:
        for point in row:
            assert point.defects is result.defects
            assert np.shares_memory(point.defects.positions, positions)


def test_sweep_honours_gaussian_level_and_interface_enrichment():
    """扫描与单点走同一条缺陷生成链路，两种分布模式在扫描里必须一样生效。

    `run_sweep` 自己调 `generate_defects`，所以要防的是它漏传 `config.defects`
    的字段、把高斯能级或界面富集悄悄退化成 fixed/uniform。
    """
    defects = make_defect_config(
        level=DefectLevelConfig(mode="gaussian", ED=1.15 * EV, sigma=0.15 * EV),
        spatial=DefectSpatialConfig(mode="interface", enrichment=5.0, decay_length=1e-9),
    )
    config = make_config(
        n_events=150,
        defects=defects,
        sweep=SweepConfig(voltages=np.array([2.0, 3.0]), temperatures=np.array([298.15])),
    )
    result = run_sweep(config)

    # 扫描用的缺陷与直接按同一配置、同一种子生成的逐位相同
    # （坐标是 1e-9、能级是 1e-19 量级，比 allclose 的默认 atol=1e-8 还小，
    #  这里必须逐位比较，否则断言会被 atol 吞掉、永远成立）
    expected = generate_defects(config.device, np.random.default_rng(config.simulation.seed))
    assert np.array_equal(result.defects.positions, expected.positions)
    assert np.array_equal(result.defects.ED, expected.ED)

    # 高斯能级生效：ED 分散，r_D = ℏ/√(2·m_di·E_D) 随之逐缺陷不同
    assert result.defects.ED.std(ddof=1) > 0.05 * EV
    assert result.defects.r_D.max() > result.defects.r_D.min()

    # 界面富集生效：同种子下换成 uniform（同几何、不同分布），z 采样就不一样
    uniform = generate_defects(
        make_device(defects=make_defect_config()), np.random.default_rng(config.simulation.seed)
    )
    z_enriched = result.defects.positions[:, 2]
    assert not np.allclose(z_enriched, uniform.positions[:, 2], rtol=1e-6, atol=0.0)

    # 两种模式下机制分解仍然可加
    total = sum(result.by_mechanism[name] for name in MECHANISMS)
    assert np.allclose(total, result.current_density, rtol=1e-12)


def test_sweep_mechanism_decomposition_sums_to_total():
    result = run_sweep(_small_sweep())
    total = sum(result.by_mechanism[name] for name in MECHANISMS)
    assert np.allclose(total, result.current_density, rtol=1e-12)


def test_progress_callback_is_called_for_every_point():
    seen = []
    run_sweep(_small_sweep(), progress=lambda done, total, r: seen.append((done, total)))
    assert [d for d, _ in seen] == [1, 2, 3, 4]
    assert all(total == 4 for _, total in seen)


def test_write_csv_round_trip(tmp_path):
    result = run_sweep(_small_sweep())
    path = tmp_path / "sweep.csv"
    write_csv(result, str(path))

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["temperature_C"] == "25"  # 输出边界换算回 °C
    # CSV 里是 A/cm²，与内部 SI 值相差 centi²
    assert float(rows[0]["J_total_A_cm2"]) == pytest.approx(
        result.current_density[0, 0] * constants.centi ** 2, rel=1e-5
    )
    mechanism_sum = sum(float(rows[0]["J_{}_A_cm2".format(name)]) for name in MECHANISMS)
    assert mechanism_sum == pytest.approx(float(rows[0]["J_total_A_cm2"]), rel=1e-5)


def test_sweep_requires_sweep_section():
    config = make_config(sweep=None)
    with pytest.raises(ValueError):
        run_sweep(config)


def test_plot_prefix_writes_iv_and_it_figures(tmp_path):
    """CLI 的 `--plot-prefix`：扫描结果直接出两张曲线图，落盘发生在 CLI 侧。"""
    from kmc_leakage.__main__ import _write_sweep_plots

    result = run_sweep(_small_sweep())
    _write_sweep_plots(
        str(tmp_path / "scan"),
        result.voltages,
        result.temperatures,
        result.current_density,
    )
    for kind in ("iv", "it"):
        assert (tmp_path / "scan_{}.png".format(kind)).stat().st_size > 0
