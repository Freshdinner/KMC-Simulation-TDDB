"""系综平均的检验。

口径：M 个独立缺陷实现的**算术平均**，误差棒取 σ/√M。这里钉住三件事——
平均确实是算术平均、每个种子确实换了一套缺陷、R1 这类与缺陷无关的分量
在系综里不产生散布。
"""

import csv

import numpy as np
import pytest

from conftest import make_config
from kmc_leakage.config import EnsembleConfig, SweepConfig
from kmc_leakage.defects import generate_defects
from kmc_leakage.ensemble import run_ensemble, run_ensemble_sweep, write_ensemble_csv
from kmc_leakage.kmc import MECHANISMS, KMCSimulation

SEEDS = (7, 8, 9)


def _config(seeds=SEEDS, **kwargs):
    return make_config(ensemble=EnsembleConfig(seeds=tuple(seeds)), **kwargs)


def _sweep_config(seeds=SEEDS):
    return _config(
        seeds=seeds,
        n_events=150,
        sweep=SweepConfig(voltages=np.array([2.0, 3.0]), temperatures=np.array([298.15])),
    )


# ----------------------------------------------------------------------
# 单点系综
# ----------------------------------------------------------------------
def test_mean_is_the_arithmetic_mean_of_the_realizations():
    result = run_ensemble(_config())
    assert result.n_realizations == len(SEEDS)
    assert len(result.results) == len(SEEDS)
    assert result.current_density == pytest.approx(float(np.mean(result.samples)), rel=1e-12)


def test_each_seed_gives_an_independent_defect_realization():
    """换种子换掉的是整套缺陷坐标——这正是系综要平均的涨落来源。"""
    config = _config()
    result = run_ensemble(config)
    for seed, point in zip(SEEDS, result.results):
        expected = generate_defects(config.device, np.random.default_rng(seed))
        assert np.allclose(point.defects.positions, expected.positions)
    positions = [point.defects.positions for point in result.results]
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            assert not np.allclose(positions[a], positions[b])


def test_single_seed_ensemble_reproduces_the_plain_run():
    """M=1 时系综退化为普通单点仿真，逐位相同。"""
    config = make_config(seed=7)
    assert config.seeds == (7,)
    result = run_ensemble(config)
    reference = KMCSimulation(config).run()
    assert result.current_density == reference.current_density
    assert result.current_density_sem == 0.0  # 一个样本给不出离散度


def test_standard_error_is_sigma_over_sqrt_m():
    result = run_ensemble(_config())
    samples = result.samples
    expected = np.std(samples, ddof=1) / np.sqrt(samples.size)
    assert result.current_density_sem == pytest.approx(expected, rel=1e-12)


def test_mechanism_decomposition_sums_to_the_ensemble_mean():
    """分量之和 = 总电流这条一致性检验，在取过平均之后同样成立。"""
    result = run_ensemble(_config())
    total = sum(result.by_mechanism[name] for name in MECHANISMS)
    assert total == pytest.approx(result.current_density, rel=1e-12)


def test_r1_components_do_not_scatter_across_realizations():
    """R1 按构造与缺陷实现无关：M 个实现给出同一个值，标准误恒为 0。"""
    result = run_ensemble(_config())
    for name in ("direct_fn", "thermionic"):
        values = [point.by_mechanism[name] for point in result.results]
        assert len(set(values)) == 1
        assert result.by_mechanism_sem[name] == 0.0
    # 同一批运行里，跟缺陷有关的分量确实在动
    assert result.by_mechanism_sem["tat"] > 0.0


def test_progress_callback_is_called_once_per_realization():
    seen = []
    run_ensemble(_config(), progress=lambda done, total, seed, r: seen.append((done, total, seed)))
    assert seen == [(1, 3, 7), (2, 3, 8), (3, 3, 9)]


# ----------------------------------------------------------------------
# 扫描系综
# ----------------------------------------------------------------------
def test_ensemble_sweep_averages_each_operating_point():
    result = run_ensemble_sweep(_sweep_config())
    assert result.current_density.shape == (1, 2)  # (温度点, 电压点)
    assert result.samples.shape == (3, 1, 2)
    assert np.allclose(result.current_density, result.samples.mean(axis=0), rtol=1e-12)
    # 每个实现内部仍是同一套缺陷跑遍所有工作点
    for single in result.sweeps:
        for row in single.results:
            for point in row:
                assert point.defects is single.defects


def test_ensemble_sweep_mechanism_decomposition_sums_to_total():
    result = run_ensemble_sweep(_sweep_config())
    total = sum(result.by_mechanism[name] for name in MECHANISMS)
    assert np.allclose(total, result.current_density, rtol=1e-12)


def test_ensemble_sweep_progress_reports_realization_and_point():
    seen = []
    run_ensemble_sweep(
        _sweep_config(), progress=lambda i, n, done, total, r: seen.append((i, n, done, total))
    )
    assert seen == [(i, 3, done, 2) for i in (1, 2, 3) for done in (1, 2)]


def test_ensemble_csv_keeps_the_per_realization_values(tmp_path):
    """CSV 里同时写平均、标准误和每个实现的原始值，换平均口径不必重跑。"""
    from scipy import constants

    result = run_ensemble_sweep(_sweep_config())
    path = tmp_path / "ensemble.csv"
    write_ensemble_csv(result, str(path))

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2  # 1 个温度 × 2 个电压
    per_cm2 = constants.centi ** 2
    row = rows[0]
    assert float(row["J_mean_A_cm2"]) == pytest.approx(
        result.current_density[0, 0] * per_cm2, rel=1e-5
    )
    assert float(row["J_sem_A_cm2"]) == pytest.approx(
        result.current_density_sem[0, 0] * per_cm2, rel=1e-5
    )
    samples = [float(row["J_seed{}_A_cm2".format(seed)]) for seed in SEEDS]
    assert np.mean(samples) == pytest.approx(float(row["J_mean_A_cm2"]), rel=1e-5)
    mechanism_sum = sum(float(row["J_{}_A_cm2".format(name)]) for name in MECHANISMS)
    assert mechanism_sum == pytest.approx(float(row["J_mean_A_cm2"]), rel=1e-5)
