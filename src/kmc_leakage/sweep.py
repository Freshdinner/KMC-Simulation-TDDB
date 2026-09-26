"""I-V / I-T 扫描。

缺陷实现与静电耦合矩阵只依赖缺陷位置，与偏压/温度无关，因此整个扫描
复用同一套预计算——这既保证不同工作点比较的是同一个器件，也省掉每点
N+1 次泊松求解。各点使用相同的随机数种子（common random numbers），
扫描曲线上的统计涨落因此是相关的，曲线更平滑且可复现。
"""

import csv
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import constants

from .config import Config
from .defects import DefectEnsemble, generate_defects
from .electrostatics import PoissonSolver
from .kmc import MECHANISMS, ElectrostaticCoupling, KMCSimulation, SimulationResult


@dataclass
class SweepResult:
    """I-V / I-T 扫描结果。矩阵均为 (温度点数, 电压点数)。"""

    voltages: np.ndarray  # [V]
    temperatures: np.ndarray  # [K]
    current_density: np.ndarray  # [A/m^2]
    by_mechanism: Dict[str, np.ndarray]  # 各机制电流密度 [A/m^2]
    by_layer: Dict[str, np.ndarray]  # 分层归因电流密度 [A/m^2]（layer_<id>/interface_<id>）
    results: List[List[SimulationResult]]
    defects: Optional[DefectEnsemble] = None


def _by_layer_keys(results: List[List[SimulationResult]]) -> List[str]:
    """扫描里出现过的所有分层归因键，保持首次出现顺序（layer_* 在前、interface_* 在后）。"""
    keys: List[str] = []
    for row in results:
        for point in row:
            for key in point.by_layer:
                if key not in keys:
                    keys.append(key)
    return keys


def run_sweep(
    config: Config,
    defects: Optional[DefectEnsemble] = None,
    progress: Optional[Callable[[int, int, SimulationResult], None]] = None,
) -> SweepResult:
    """按配置里的 sweep 范围逐点跑 KMC。

    `progress(done, total, result)` 每完成一个工作点调用一次，便于 CLI 打印
    进度；不需要时传 None。
    """
    if config.sweep is None:
        raise ValueError("配置中没有 sweep 段，无法扫描")
    device = config.device
    rng = np.random.default_rng(config.simulation.seed)
    defects = defects if defects is not None else generate_defects(device, rng)

    solver = PoissonSolver(device, config.simulation.grid)
    coupling = ElectrostaticCoupling(solver, defects.positions, defects.field_side)

    voltages = np.asarray(config.sweep.voltages, dtype=float)
    temperatures = np.asarray(config.sweep.temperatures, dtype=float)
    area = device.area
    shape = (temperatures.size, voltages.size)
    density = np.zeros(shape)
    by_mechanism = {name: np.zeros(shape) for name in MECHANISMS}
    results: List[List[SimulationResult]] = []

    total = temperatures.size * voltages.size
    done = 0
    for it, temperature in enumerate(temperatures):
        row: List[SimulationResult] = []
        for iv, voltage in enumerate(voltages):
            point = replace(
                config,
                simulation=replace(config.simulation, voltage=float(voltage), temperature=float(temperature)),
            )
            result = KMCSimulation(
                point, defects=defects, coupling=coupling, solver=solver
            ).run()
            density[it, iv] = result.current_density
            for name in MECHANISMS:
                by_mechanism[name][it, iv] = result.by_mechanism[name] / area
            row.append(result)
            done += 1
            if progress is not None:
                progress(done, total, result)
        results.append(row)

    by_layer = {key: np.zeros(shape) for key in _by_layer_keys(results)}
    for it in range(temperatures.size):
        for iv in range(voltages.size):
            for key, matrix in by_layer.items():
                matrix[it, iv] = results[it][iv].by_layer.get(key, 0.0) / area

    return SweepResult(
        voltages=voltages,
        temperatures=temperatures,
        current_density=density,
        by_mechanism=by_mechanism,
        by_layer=by_layer,
        results=results,
        defects=defects,
    )


def write_csv(result: SweepResult, path: str) -> None:
    """把扫描结果写成 CSV。

    这里是输出边界，电流密度换算成惯用的 A/cm²、温度换算成 °C，换算系数
    同样取自 scipy.constants。除总电流与各机制外，还写出分层归因（ADR-014）与
    F04 诊断（三个估计量、全局守恒的 ΔN_inside、测量窗口、验证状态）。
    """
    per_cm2 = constants.centi ** 2  # A/m^2 -> A/cm^2
    layer_keys = list(result.by_layer.keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["voltage_V", "temperature_C", "J_total_A_cm2"]
            + ["J_{}_A_cm2".format(name) for name in MECHANISMS]
            + ["J_{}_A_cm2".format(key) for key in layer_keys]
            + ["J_direct_A_cm2", "J_counts_total_A_cm2", "delta_n_inside",
               "measured_events", "n_events", "validated"]
        )
        for it, temperature in enumerate(result.temperatures):
            for iv, voltage in enumerate(result.voltages):
                point = result.results[it][iv]
                r1 = point.by_mechanism["direct_fn"] + point.by_mechanism["thermionic"]
                to_density = per_cm2 / point.area
                writer.writerow(
                    ["{:.6g}".format(voltage), "{:.6g}".format(temperature - constants.zero_Celsius)]
                    + ["{:.6e}".format(result.current_density[it, iv] * per_cm2)]
                    + [
                        "{:.6e}".format(result.by_mechanism[name][it, iv] * per_cm2)
                        for name in MECHANISMS
                    ]
                    + [
                        "{:.6e}".format(result.by_layer[key][it, iv] * per_cm2)
                        for key in layer_keys
                    ]
                    + [
                        "{:.6e}".format(point.current_direct * to_density),
                        "{:.6e}".format((point.current_counts + r1) * to_density),
                        "{:d}".format(point.delta_n_inside),
                        "{:d}".format(point.measured_events),
                        "{:d}".format(point.n_events),
                        "{}".format(int(point.validated)),
                    ]
                )
