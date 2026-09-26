"""系综平均：多个缺陷实现的算术平均。

一次 KMC 只看到一套具体的缺陷坐标与能级，而漏电流对这套坐标很敏感——
实测 σ(log₁₀J) = 0.23–0.40。要给出可与实验比绝对值的 J，得跑 M 个独立
实现再取**算术平均**（不是几何平均：电流是可加量，实验测到的也是整片
电极上大量局部通道之和）。

一个种子决定一次实现：`simulation.seed` 同时驱动缺陷生成与 KMC 轨迹，
所以换种子换掉的是"缺陷摆在哪、能级多深"加上轨迹噪声，两者合起来正是
要平均掉的涨落。每个实现独立做一遍静电预计算（缺陷位置变了，耦合矩阵
就得重算），实现之间没有任何共享状态。

误差棒取平均值的标准误 σ/√M（σ 为实现间样本标准差，ddof=1）。M=1 时
按 0 处理——一个样本给不出离散度。

R1（`direct_fn` / `thermionic`）按构造与缺陷实现无关，M 个实现给出逐位
相同的值，平均后不变、标准误恒为 0。这可以当作系综代码没有串味的检查。
"""

import csv
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import constants

from .config import Config
from .kmc import MECHANISMS, KMCSimulation, SimulationResult
from .sweep import SweepResult, run_sweep


@dataclass
class EnsembleResult:
    """单个工作点上 M 个实现的系综平均。电流一律是**电流密度** [A/m²]。"""

    seeds: Tuple[int, ...]
    voltage: float  # [V]
    temperature: float  # [K]
    current_density: float  # M 个实现的算术平均 [A/m^2]
    current_density_sem: float  # 平均值的标准误 σ/√M [A/m^2]
    by_mechanism: Dict[str, float]  # 各机制的平均电流密度，四项之和 = current_density
    by_mechanism_sem: Dict[str, float]
    by_layer: Dict[str, float]  # 分层归因平均电流密度 [A/m^2]（ADR-014）
    by_layer_sem: Dict[str, float]
    delta_j: float  # 成对差 ΔJ = J_cond − (J_counts+R1) 的系综均值 [A/m^2]（F04 交叉校验）
    delta_j_sem: float  # ΔJ 的标准误（跨种子散布，Q2）
    n_validated: int  # 通过验证的实现个数（F04）
    results: List[SimulationResult]  # 每个实现的原始结果，顺序与 seeds 一致

    @property
    def n_realizations(self) -> int:
        """系综里的实现个数 M。"""
        return len(self.seeds)

    @property
    def samples(self) -> np.ndarray:
        """各实现的总电流密度 [A/m^2]，平均前的原始样本。"""
        return np.array([r.current_density for r in self.results])


@dataclass
class EnsembleSweepResult:
    """I-V / I-T 扫描的系综平均。矩阵均为 (温度点数, 电压点数)。"""

    seeds: Tuple[int, ...]
    voltages: np.ndarray  # [V]
    temperatures: np.ndarray  # [K]
    current_density: np.ndarray  # 算术平均 [A/m^2]
    current_density_sem: np.ndarray  # 平均值的标准误 [A/m^2]
    by_mechanism: Dict[str, np.ndarray]  # 各机制平均电流密度 [A/m^2]
    by_mechanism_sem: Dict[str, np.ndarray]
    by_layer: Dict[str, np.ndarray]  # 分层归因平均电流密度 [A/m^2]（ADR-014）
    by_layer_sem: Dict[str, np.ndarray]
    delta_j: np.ndarray  # ΔJ = J_cond − (J_counts+R1) 的系综均值 [A/m^2]（F04）
    delta_j_sem: np.ndarray
    n_validated: np.ndarray  # 每个工作点通过验证的实现个数（int 矩阵）
    sweeps: List[SweepResult]  # 每个实现的完整扫描结果，顺序与 seeds 一致

    @property
    def n_realizations(self) -> int:
        """系综里的实现个数 M。"""
        return len(self.seeds)

    @property
    def samples(self) -> np.ndarray:
        """各实现的电流密度矩阵，形状 (M, 温度点数, 电压点数) [A/m^2]。"""
        return np.array([s.current_density for s in self.sweeps])


def _mean_sem(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """沿实现轴（axis=0）取算术平均与平均值的标准误 σ/√M。"""
    mean = samples.mean(axis=0)
    m = samples.shape[0]
    if m < 2:
        return mean, np.zeros_like(mean)
    return mean, samples.std(axis=0, ddof=1) / np.sqrt(m)


def _layer_union(layer_dicts) -> List[str]:
    """多个实现的分层归因键的并集，保持首次出现顺序（layer_* 在前、interface_* 在后）。"""
    keys: List[str] = []
    for d in layer_dicts:
        for key in d:
            if key not in keys:
                keys.append(key)
    return keys


def _r1(result: SimulationResult) -> float:
    """R1 两分量之和 [A]（电极↔电极，单列、不进分层）。"""
    return result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]


def _seeded(config: Config, seed: int) -> Config:
    """把配置里的种子换成指定值——缺陷实现与 KMC 轨迹都跟着换。"""
    return replace(config, simulation=replace(config.simulation, seed=seed))


def run_ensemble(
    config: Config,
    progress: Optional[Callable[[int, int, int, SimulationResult], None]] = None,
) -> EnsembleResult:
    """按 `config.seeds` 逐个实现跑单点 KMC，返回算术平均。

    `progress(done, total, seed, result)` 每跑完一个实现调用一次，便于 CLI
    打印进度；不需要时传 None。
    """
    seeds = config.seeds
    area = config.device.area
    results: List[SimulationResult] = []
    for seed in seeds:
        result = KMCSimulation(_seeded(config, seed)).run()
        results.append(result)
        if progress is not None:
            progress(len(results), len(seeds), seed, result)

    density, density_sem = _mean_sem(np.array([r.current_density for r in results]))
    mechanism = {
        name: _mean_sem(np.array([r.by_mechanism[name] / area for r in results]))
        for name in MECHANISMS
    }
    layer_keys = _layer_union([r.by_layer for r in results])
    layer = {
        key: _mean_sem(np.array([r.by_layer.get(key, 0.0) / area for r in results]))
        for key in layer_keys
    }
    delta = _mean_sem(
        np.array([(r.current - (r.current_counts + _r1(r))) / area for r in results])
    )
    return EnsembleResult(
        seeds=seeds,
        voltage=config.simulation.voltage,
        temperature=config.simulation.temperature,
        current_density=float(density),
        current_density_sem=float(density_sem),
        by_mechanism={name: float(value[0]) for name, value in mechanism.items()},
        by_mechanism_sem={name: float(value[1]) for name, value in mechanism.items()},
        by_layer={key: float(value[0]) for key, value in layer.items()},
        by_layer_sem={key: float(value[1]) for key, value in layer.items()},
        delta_j=float(delta[0]),
        delta_j_sem=float(delta[1]),
        n_validated=int(sum(r.validated for r in results)),
        results=results,
    )


def run_ensemble_sweep(
    config: Config,
    progress: Optional[Callable[[int, int, int, int, SimulationResult], None]] = None,
) -> EnsembleSweepResult:
    """按 `config.seeds` 逐个实现跑完整扫描，逐工作点取算术平均。

    一个实现内部仍是同一套缺陷跑遍所有工作点（`run_sweep` 的既有口径），
    所以扫描曲线的形状由单个器件决定，系综只在工作点之间纵向平均。

    `progress(realization, n_realizations, done, total, result)` 每完成一个
    工作点调用一次。
    """
    seeds = config.seeds
    sweeps: List[SweepResult] = []
    for index, seed in enumerate(seeds, start=1):

        def relay(done: int, total: int, result: SimulationResult, index: int = index) -> None:
            if progress is not None:
                progress(index, len(seeds), done, total, result)

        sweeps.append(run_sweep(_seeded(config, seed), progress=relay))

    area = config.device.area
    shape = sweeps[0].current_density.shape
    density, density_sem = _mean_sem(np.array([s.current_density for s in sweeps]))
    mechanism = {
        name: _mean_sem(np.array([s.by_mechanism[name] for s in sweeps])) for name in MECHANISMS
    }
    layer_keys = _layer_union([s.by_layer for s in sweeps])
    layer = {
        key: _mean_sem(np.array([s.by_layer.get(key, np.zeros(shape)) for s in sweeps]))
        for key in layer_keys
    }
    dj_samples = np.array([
        [[(p.current - (p.current_counts + _r1(p))) / area for p in row] for row in s.results]
        for s in sweeps
    ])
    delta_j, delta_j_sem = _mean_sem(dj_samples)
    n_validated = np.sum(
        [[[int(p.validated) for p in row] for row in s.results] for s in sweeps], axis=0
    )
    return EnsembleSweepResult(
        seeds=seeds,
        voltages=sweeps[0].voltages,
        temperatures=sweeps[0].temperatures,
        current_density=density,
        current_density_sem=density_sem,
        by_mechanism={name: value[0] for name, value in mechanism.items()},
        by_mechanism_sem={name: value[1] for name, value in mechanism.items()},
        by_layer={key: value[0] for key, value in layer.items()},
        by_layer_sem={key: value[1] for key, value in layer.items()},
        delta_j=delta_j,
        delta_j_sem=delta_j_sem,
        n_validated=n_validated,
        sweeps=sweeps,
    )


def write_ensemble_csv(result: EnsembleSweepResult, path: str) -> None:
    """把系综扫描结果写成 CSV：平均值、标准误、各机制平均、各实现原始值。

    与 `sweep.write_csv` 一样是输出边界，电流密度换算成 A/cm²、温度换算成
    °C。每个实现的原始电流密度也一并写出（`J_seed<种子>_A_cm2`），换一种
    平均口径时不必重跑。
    """
    per_cm2 = constants.centi ** 2  # A/m^2 -> A/cm^2
    samples = result.samples
    layer_keys = list(result.by_layer.keys())
    n_realizations = result.n_realizations
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["voltage_V", "temperature_C", "J_mean_A_cm2", "J_sem_A_cm2"]
            + ["J_{}_A_cm2".format(name) for name in MECHANISMS]
            + ["J_{}_A_cm2".format(key) for key in layer_keys]
            + ["dJ_cross_A_cm2", "dJ_cross_sem_A_cm2", "n_validated", "n_realizations"]
            + ["J_seed{}_A_cm2".format(seed) for seed in result.seeds]
        )
        for it, temperature in enumerate(result.temperatures):
            for iv, voltage in enumerate(result.voltages):
                writer.writerow(
                    ["{:.6g}".format(voltage), "{:.6g}".format(temperature - constants.zero_Celsius)]
                    + [
                        "{:.6e}".format(result.current_density[it, iv] * per_cm2),
                        "{:.6e}".format(result.current_density_sem[it, iv] * per_cm2),
                    ]
                    + [
                        "{:.6e}".format(result.by_mechanism[name][it, iv] * per_cm2)
                        for name in MECHANISMS
                    ]
                    + [
                        "{:.6e}".format(result.by_layer[key][it, iv] * per_cm2)
                        for key in layer_keys
                    ]
                    + [
                        "{:.6e}".format(result.delta_j[it, iv] * per_cm2),
                        "{:.6e}".format(result.delta_j_sem[it, iv] * per_cm2),
                        "{:d}".format(int(result.n_validated[it, iv])),
                        "{:d}".format(n_realizations),
                    ]
                    + ["{:.6e}".format(value * per_cm2) for value in samples[:, it, iv]]
                )
