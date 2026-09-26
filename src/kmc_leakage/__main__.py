"""命令行入口。

    python -m kmc_leakage run   --config examples/single_run.yaml
    python -m kmc_leakage sweep --config examples/iv_sweep.yaml [--out results.csv]
                                [--plot-prefix scan]
    python -m kmc_leakage plot  --config examples/single_run.yaml --kind defects|bands
"""

from typing import Dict, Optional

import numpy as np
import typer
from scipy import constants

from .bands import band_profile, trap_levels
from .defects import generate_defects
from .electrostatics import PoissonSolver
from .ensemble import EnsembleResult, run_ensemble, run_ensemble_sweep, write_ensemble_csv
from .io import load_config
from .kmc import MECHANISM_LABELS, MECHANISMS, KMCSimulation, SimulationResult
from .sweep import run_sweep, write_csv

app = typer.Typer(add_completion=False, help="MIM 电容器漏电流 KMC 仿真")

_PER_CM2 = constants.centi ** 2  # A/m^2 -> A/cm^2


def _report(result: SimulationResult) -> None:
    """打印单点结果摘要。"""
    to_density = _PER_CM2 / result.area
    typer.echo(
        "U = {:.2f} V, T = {:.1f} °C".format(
            result.voltage, result.temperature - constants.zero_Celsius
        )
    )
    typer.echo(
        "  漏电流密度 J = {:.4e} A/cm²   (I = {:.4e} A, 统计窗口 {:.3e} s)".format(
            result.current_density * _PER_CM2, result.current, result.elapsed_time
        )
    )
    typer.echo("  按机制分解：")
    fractions = result.mechanism_fractions
    for name in MECHANISMS:
        typer.echo(
            "    {:<12s} J = {:>12.4e} A/cm²   占比 {:>8.3%}".format(
                MECHANISM_LABELS[name], result.by_mechanism[name] * to_density, fractions[name]
            )
        )
    if result.by_layer:
        typer.echo("  按层/界面归因：")
        for key, value in result.by_layer.items():
            typer.echo("    {:<12s} J = {:>12.4e} A/cm²".format(key, value * to_density))
    order = np.argsort(result.by_defect)[::-1][:5]
    typer.echo("  贡献最大的 5 个缺陷（z 位置 / 电流占比）：")
    for i in order:
        z_nm = result.defects.positions[i, 2] / constants.nano
        share = result.by_defect[i] / result.current if result.current else 0.0
        typer.echo("    #{:<3d} z = {:5.2f} nm   占比 {:>8.3%}".format(int(i), z_nm, share))
    r1 = result.by_mechanism["direct_fn"] + result.by_mechanism["thermionic"]
    typer.echo(
        "  三估计量：J_条件 = {:.3e}，J_直接 = {:.3e}，J_数事件+R1 = {:.3e} A/cm²（阳极事件 {} 次）".format(
            result.current * to_density,
            result.current_direct * to_density,
            (result.current_counts + r1) * to_density,
            result.n_anode_events,
        )
    )
    status = "通过验证" if result.validated else "未通过验证：" + "；".join(result.validation_notes)
    typer.echo(
        "  验证状态：{}（测量事件 {}，ΔN_inside {}，守恒 (I_C−I_A)Δt=q·ΔN）".format(
            status, result.measured_events, result.delta_n_inside
        )
    )


def _report_ensemble(result: EnsembleResult) -> None:
    """打印系综平均结果摘要。逐个实现的 J 在跑的过程中已经打过，这里只给平均。"""
    typer.echo(
        "系综平均（M = {} 个缺陷实现）：J = {:.4e} A/cm²   标准误 ±{:.3e}（相对 {:.1%}）".format(
            result.n_realizations,
            result.current_density * _PER_CM2,
            result.current_density_sem * _PER_CM2,
            result.current_density_sem / result.current_density if result.current_density else 0.0,
        )
    )
    samples = result.samples * _PER_CM2
    spread = "  各实现散布：min {:.4e}   max {:.4e} A/cm²".format(samples.min(), samples.max())
    if np.all(samples > 0.0):
        spread += "   σ(log₁₀J) = {:.3f}".format(float(np.std(np.log10(samples), ddof=1)))
    typer.echo(spread)
    typer.echo("  按机制分解（系综平均）：")
    for name in MECHANISMS:
        mean = result.by_mechanism[name] * _PER_CM2
        share = result.by_mechanism[name] / result.current_density if result.current_density else 0.0
        typer.echo(
            "    {:<12s} J = {:>12.4e} A/cm²   占比 {:>8.3%}   标准误 ±{:.3e}".format(
                MECHANISM_LABELS[name], mean, share, result.by_mechanism_sem[name] * _PER_CM2
            )
        )
    if result.by_layer:
        typer.echo("  按层/界面归因（系综平均）：")
        for key, value in result.by_layer.items():
            typer.echo(
                "    {:<12s} J = {:>12.4e} A/cm²   标准误 ±{:.3e}".format(
                    key, value * _PER_CM2, result.by_layer_sem[key] * _PER_CM2
                )
            )
    typer.echo(
        "  交叉校验 ΔJ = J_条件−(J_数事件+R1) = {:.3e} ± {:.3e} A/cm²；通过验证 {}/{} 个实现".format(
            result.delta_j * _PER_CM2, result.delta_j_sem * _PER_CM2,
            result.n_validated, result.n_realizations,
        )
    )


@app.command()
def run(config: str = typer.Option(..., "--config", help="YAML 配置文件路径")) -> None:
    """单点仿真。配置里有 ensemble 段时逐个种子跑完取算术平均。"""
    cfg = load_config(config)
    if len(cfg.seeds) == 1:
        _report(KMCSimulation(cfg).run())
        return

    typer.echo(
        "U = {:.2f} V, T = {:.1f} °C   系综：{} 个缺陷实现".format(
            cfg.simulation.voltage,
            cfg.simulation.temperature - constants.zero_Celsius,
            len(cfg.seeds),
        )
    )

    def progress(done: int, total: int, seed: int, result: SimulationResult) -> None:
        typer.echo(
            "  [{:>2d}/{:<2d}] seed {:<8d} J = {:.4e} A/cm²".format(
                done, total, seed, result.current_density * _PER_CM2
            )
        )

    _report_ensemble(run_ensemble(cfg, progress=progress))


def _write_sweep_plots(
    prefix: str,
    voltages: np.ndarray,
    temperatures: np.ndarray,
    density: np.ndarray,
    by_mechanism: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """把扫描结果画成 I-V 与 I-T 两张曲线图并落盘。

    绘图函数只返回 Figure，保存动作在这里发生。`by_mechanism` 让两张图各自
    多出 TAT 与 PF 两族曲线。
    """
    from .visualization import plot_it_curves, plot_iv_curves

    args = (voltages, temperatures, density, by_mechanism)
    figures = (("iv", plot_iv_curves(*args)), ("it", plot_it_curves(*args)))
    for kind, figure in figures:
        path = "{}_{}.png".format(prefix, kind)
        figure.savefig(path, dpi=150)
        typer.echo("已写入 {}".format(path))


@app.command()
def sweep(
    config: str = typer.Option(..., "--config", help="YAML 配置文件路径"),
    out: Optional[str] = typer.Option(None, "--out", help="结果 CSV 输出路径"),
    plot_prefix: Optional[str] = typer.Option(
        None, "--plot-prefix", help="曲线图输出前缀，写出 <前缀>_iv.png 与 <前缀>_it.png"
    ),
) -> None:
    """I-V / I-T 扫描。配置里有 ensemble 段时逐个种子扫完取算术平均。"""
    cfg = load_config(config)

    def line(prefix: str, done: int, total: int, result: SimulationResult) -> None:
        typer.echo(
            "{}[{:>3d}/{:<3d}] U = {:.2f} V, T = {:6.1f} °C -> J = {:.4e} A/cm²".format(
                prefix,
                done,
                total,
                result.voltage,
                result.temperature - constants.zero_Celsius,
                result.current_density * _PER_CM2,
            )
        )

    if len(cfg.seeds) == 1:
        result = run_sweep(cfg, progress=lambda d, t, r: line("", d, t, r))
        if out is not None:
            write_csv(result, out)
            typer.echo("已写入 {}".format(out))
        if plot_prefix is not None:
            _write_sweep_plots(
                plot_prefix,
                result.voltages,
                result.temperatures,
                result.current_density,
                by_mechanism=result.by_mechanism,
            )
        return

    def progress(index: int, n: int, done: int, total: int, result: SimulationResult) -> None:
        line("[实现 {}/{}] ".format(index, n), done, total, result)

    ensemble = run_ensemble_sweep(cfg, progress=progress)
    typer.echo(
        "系综平均（M = {} 个缺陷实现，seed = {}）".format(
            ensemble.n_realizations, ", ".join(str(s) for s in ensemble.seeds)
        )
    )
    for it, temperature in enumerate(ensemble.temperatures):
        for iv, voltage in enumerate(ensemble.voltages):
            typer.echo(
                "  U = {:.2f} V, T = {:6.1f} °C -> J = {:.4e} ± {:.3e} A/cm²".format(
                    voltage,
                    temperature - constants.zero_Celsius,
                    ensemble.current_density[it, iv] * _PER_CM2,
                    ensemble.current_density_sem[it, iv] * _PER_CM2,
                )
            )
    if out is not None:
        write_ensemble_csv(ensemble, out)
        typer.echo("已写入 {}".format(out))
    if plot_prefix is not None:
        _write_sweep_plots(
            plot_prefix,
            ensemble.voltages,
            ensemble.temperatures,
            ensemble.current_density,
            ensemble.by_mechanism,
        )


@app.command()
def plot(
    config: str = typer.Option(..., "--config", help="YAML 配置文件路径"),
    kind: str = typer.Option("defects", "--kind", help="defects | bands"),
    out: Optional[str] = typer.Option(None, "--out", help="图片输出路径（默认 <kind>.png）"),
) -> None:
    """出图。绘图函数只返回 Figure，保存动作在这里发生。"""
    from .visualization import plot_band_diagram, plot_defect_distribution

    cfg = load_config(config)
    # 系综 M 个实现（种子每次运行随机抽取）：代表实现取抽到的头一个用于三维散点，
    # 深度直方图汇总全部 M 个再除以 M 压噪。
    seed = cfg.seeds[0]
    realizations = [generate_defects(cfg.device, np.random.default_rng(s)) for s in cfg.seeds]
    defects = realizations[0]

    if kind == "defects":
        figure = plot_defect_distribution(cfg.device, defects, realizations=realizations)
    elif kind == "bands":
        from .bands import top_boundary_potential

        solver = PoissonSolver(cfg.device, cfg.simulation.grid)
        phi = solver.solve(
            np.zeros(cfg.simulation.grid.shape),
            top_boundary_potential(cfg.device, cfg.simulation.voltage),
        )
        profile = band_profile(
            cfg.device, solver.z, solver.lateral_average(phi), cfg.simulation.voltage
        )
        levels = trap_levels(
            defects.band_reference, solver.interpolate(phi, defects.positions), defects.ED
        )
        figure = plot_band_diagram(profile, defects.positions[:, 2], levels)
    else:
        raise typer.BadParameter("--kind 只支持 defects 或 bands，收到 {!r}".format(kind))

    path = out if out is not None else "{}.png".format(kind)
    figure.savefig(path, dpi=150)
    if kind == "defects":
        typer.echo(
            "已写入 {}（三维散点 seed = {}，深度直方图汇总 M = {} 个实现）".format(
                path, seed, len(realizations)
            )
        )
    else:
        typer.echo("已写入 {}（seed = {} 的缺陷实现）".format(path, seed))


def main() -> None:
    """入口函数。"""
    app()


if __name__ == "__main__":
    main()
