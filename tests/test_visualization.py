"""可视化模块的接口约定检验。

绘图函数只接受已经算好的数据，返回 Figure，不落盘、不触发仿真。

叠层部分（ADR-016 ①②，Phase 6）：缺陷图叠画层边界与界面位置、界面缺陷单独
标色，能带图在界面处画真实台阶（界面双值，不连斜坡）。两项都不改函数签名，
单层画出来与叠层改造前逐位相同。
"""

import dataclasses
import pathlib

import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from scipy import constants

from conftest import make_config, make_defect_config, make_device
from kmc_leakage.bands import band_profile, top_boundary_potential, trap_levels
from kmc_leakage.config import DefectLevelConfig, DefectSpatialConfig
from kmc_leakage.defects import generate_defects
from kmc_leakage.electrostatics import PoissonSolver
from kmc_leakage.io import load_config
from kmc_leakage.visualization import (
    plot_band_diagram,
    plot_defect_distribution,
    plot_it_curves,
    plot_iv_curves,
)

EV = constants.electron_volt
PER_CM2 = constants.centi ** 2
NM = constants.nano
ROOT = pathlib.Path(__file__).resolve().parents[1]
# ADR-016 的叠画颜色：界面缺陷单独标色，层边界/界面位置用灰色虚线
INTERFACE_DEFECT_COLOR = "#c00000"

# 一份假扫描结果：2 个温度 × 3 个电压，J 随电压、温度都上升
VOLTAGES = np.array([2.0, 3.0, 4.0])
TEMPERATURES = np.array([298.15, 398.15])
DENSITY = np.array([[1e-4, 3e-4, 9e-4], [2e-3, 6e-3, 1.8e-2]])  # [A/m^2]


def _scatters(axes):
    """散点（scatter）产生 PathCollection，hlines 产生 LineCollection，这里只数前者。"""
    return sum(isinstance(item, PathCollection) for item in axes.collections)


def _luminance(line):
    """线条颜色的 sRGB 相对亮度，用来判断深浅方向（越小越深）。"""
    rgb = np.array(to_rgb(line.get_color()))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(np.dot(linear, (0.2126, 0.7152, 0.0722)))


@pytest.fixture
def defects():
    return generate_defects(make_device(), np.random.default_rng(0))


def _stack_scene(t2=2, voltage=3.0):
    """一份 t₂ 叠层示例的（器件, 缺陷, 能带剖面, 缺陷能级）——只算数据、不出图。"""
    config = load_config(str(ROOT / "examples" / "stack_t2_{}nm.yaml".format(t2)))
    grid = dataclasses.replace(config.simulation.grid, nx=8, ny=8)  # hz 不变，界面仍对齐
    device = config.device
    defects = generate_defects(device, np.random.default_rng(12345))
    solver = PoissonSolver(device, grid)
    phi = solver.solve(np.zeros(grid.shape), top_boundary_potential(device, voltage))
    profile = band_profile(device, solver.z, solver.lateral_average(phi), voltage)
    levels = trap_levels(defects.band_reference, solver.interpolate(phi, defects.positions), defects.ED)
    return device, defects, profile, levels


@pytest.fixture
def profile_and_levels(defects):
    config = make_config()
    solver = PoissonSolver(config.device, config.simulation.grid)
    phi = solver.solve(
        np.zeros(config.simulation.grid.shape),
        top_boundary_potential(config.device, config.simulation.voltage),
    )
    profile = band_profile(
        config.device, solver.z, solver.lateral_average(phi), config.simulation.voltage
    )
    levels = trap_levels(defects.band_reference, solver.interpolate(phi, defects.positions), defects.ED)
    return profile, levels


def test_defect_plot_returns_figure_with_two_panels(defects):
    figure = plot_defect_distribution(make_device(), defects)
    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 2  # 三维散点 + 直方图（外加 colorbar 的轴）
    assert figure.axes[0].get_zlabel() == "z (nm)"


def test_defect_plot_colors_by_level(defects):
    """能级着色：高斯模式下颜色范围应当展开。"""
    device = make_device(defects=make_defect_config(level=DefectLevelConfig("gaussian", 1.15 * EV, 0.15 * EV)))
    gaussian = generate_defects(device, np.random.default_rng(1))
    figure = plot_defect_distribution(device, gaussian)
    collection = figure.axes[0].collections[0]
    assert np.ptp(collection.get_array()) > 0.05  # eV


def test_defect_plot_pools_realizations():
    """深度直方图汇总 M 个实现除以 M：柱高×bin 体积之和 = 每实现平均体缺陷数。"""
    device = make_device(
        defects=make_defect_config(
            concentration_cm3=3.0e18,
            spatial=DefectSpatialConfig(mode="interface", enrichment=6.0, decay_length=1e-9),
        )
    )
    reals = [generate_defects(device, np.random.default_rng(s)) for s in range(6)]
    figure = plot_defect_distribution(device, reals[0], realizations=reals)
    hist = figure.axes[-1]

    # 柱高已除以 M：柱高×bin 体积之和 = 每实现平均体缺陷数（体缺陷数逐实现确定，取任一）
    # 直方图只分箱可采样窗口 [standoff, Lz−standoff]，bin 宽按窗口宽算
    heights = np.array([patch.get_height() for patch in hist.containers[0]])
    standoff = device.stack.layers[0].defects.min_separation
    width_m = (device.Lz - 2 * standoff) / len(heights)
    bin_volume_cm3 = device.Lx * device.Ly * width_m / constants.centi ** 3
    per_realization = int((reals[0].interface_id < 0).sum())
    assert (heights * bin_volume_cm3).sum() == pytest.approx(per_realization, rel=1e-6)


def test_band_diagram_contains_bands_and_defects(profile_and_levels, defects):
    profile, levels = profile_and_levels
    figure = plot_band_diagram(profile, defects.positions[:, 2], levels)
    assert isinstance(figure, Figure)
    axes = figure.axes[0]
    assert len(axes.lines) >= 2  # E_C 与 E_V
    assert _scatters(axes) == 1  # 缺陷能级散点
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("E_C" in label for label in labels)


def test_band_diagram_works_without_defects(profile_and_levels):
    profile, _ = profile_and_levels
    figure = plot_band_diagram(profile)
    assert isinstance(figure, Figure)
    assert _scatters(figure.axes[0]) == 0  # 电极费米能级用的是 hlines，不算散点


def test_band_diagram_respects_energy_zero_convention(profile_and_levels):
    """能量零点 = 阴极费米能级；阳极费米能级在 -eU。"""
    profile, _ = profile_and_levels
    assert profile.E_F_cathode == 0.0
    assert profile.E_F_anode == pytest.approx(-constants.elementary_charge * profile.voltage)
    # 界面处能带是直线（不做镜像力圆化）：无空间电荷时 E_C 严格线性
    slope = np.diff(profile.E_C) / np.diff(profile.z)
    assert np.allclose(slope, slope[0], rtol=1e-10)


# --------------------------------------------------------------------------
# 叠层叠画（ADR-016 ①②，Phase 6）
# --------------------------------------------------------------------------
def test_stack_defect_plot_marks_layers_and_colors_interface_defects():
    """① 缺陷图：两幅都叠画层边界与界面位置，界面缺陷单独标色、单独计数。"""
    device, defects, _, _ = _stack_scene()
    figure = plot_defect_distribution(device, defects)
    axes3d, axes = figure.axes[0], figure.axes[-1]  # 中间那个 axes 是能级 colorbar
    interior = device.stack.layer_edges[1:-1] / NM

    # 三维散点分成两族：体缺陷按能级着色，界面缺陷单独标色
    assert _scatters(axes3d) == 2
    bulk, interface = axes3d.collections[0], axes3d.collections[1]
    assert np.ptp(bulk.get_array()) > 0.0  # 体缺陷仍按 E_D 着色
    assert to_rgb(interface.get_facecolor()[0][:3]) == to_rgb(INTERFACE_DEFECT_COLOR)
    # 界面平面画成盒子上的一圈虚线，每个界面一圈
    plane_z = [np.unique(line.get_data_3d()[2])[0] for line in axes3d.lines]
    assert np.allclose(sorted(plane_z), interior)

    # 深度剖面：层边界竖线 + 层材料名 + 每个界面的界面缺陷个数（直方图只有这些竖线）
    line_x = sorted({line.get_xdata()[0] for line in axes.lines})
    assert np.allclose(line_x, interior)  # 灰色层边界与红色界面线落在同一处
    assert any(line.get_color() == INTERFACE_DEFECT_COLOR for line in axes.lines)
    texts = [text.get_text().strip() for text in axes.texts]
    assert [layer.material for layer in device.stack.layers] == texts[:3]
    assert texts[3:] == ["25", "25"]  # 每个界面 Nₛ·A = 25 个界面缺陷

    # 柱子只统计体缺陷：界面缺陷是面密度，与体密度不是一个量纲
    # 直方图只分箱可采样窗口 [standoff, Lz−standoff]，bin 宽按窗口宽算
    heights = np.array([patch.get_height() for patch in axes.containers[0]])
    standoff = max(l.defects.min_separation for l in device.stack.layers if l.defects is not None)
    width_m = (device.Lz - 2 * standoff) / len(heights)
    bin_volume_cm3 = device.Lx * device.Ly * width_m / constants.centi ** 3
    assert heights.sum() * bin_volume_cm3 == pytest.approx(int((defects.interface_id < 0).sum()))


def test_stack_band_diagram_draws_true_steps_not_ramps():
    """② 能带图：界面处同一个 z 上有两个能量值，台阶高度 = ΔE_C，不连成斜坡。"""
    device, defects, profile, levels = _stack_scene()
    axes = plot_band_diagram(profile, defects.positions[:, 2], levels).axes[0]
    z1_nm = device.stack.interface_positions[0] / NM
    delta_Ec = device.stack.delta_Ec(0) / EV

    conduction = axes.lines[0]
    x, y = conduction.get_xdata(), conduction.get_ydata()
    at_interface = np.nonzero(np.isclose(x, z1_nm))[0]
    assert at_interface.size == 2  # 界面双值：同一个 z 存了两侧各一个 E_C
    below, above = at_interface  # 先下侧（ZrO₂）后上侧（Al₂O₃）
    # 台阶不被插成斜坡：两个值横坐标相同（连线是竖直的），高度差就是 ΔE_C
    assert x[below] == x[above]
    assert y[above] - y[below] == pytest.approx(delta_Ec, rel=1e-9)
    # 两侧的值分别是各自层的导带底 B_ℓ − eφ(z₁)，不是插值出来的居中值
    node = int(np.rint(device.stack.interface_positions[0] / (profile.z[1] - profile.z[0])))
    assert y[below] == pytest.approx(profile.E_C_lower[0] / EV, rel=1e-12)
    assert y[above] == pytest.approx(profile.E_C[node] / EV, rel=1e-12)
    # 价带顶同样有台阶（带隙也跳变），界面位置另画一条参考线
    assert np.isclose(axes.lines[1].get_xdata(), z1_nm).sum() == 2
    guides = [line.get_xdata()[0] for line in axes.lines[2:]]
    assert set(np.round(device.stack.interface_positions / NM, 9)).issubset(np.round(guides, 9))


def test_single_layer_plots_are_unchanged_by_the_stack_overlays(profile_and_levels, defects):
    """退化用例（ADR-018）：单层没有内部界面，两幅图与叠层改造前逐位相同。"""
    profile, levels = profile_and_levels
    axes = plot_band_diagram(profile, defects.positions[:, 2], levels).axes[0]
    # E_C / E_V 逐位画在原网格上（没有插入界面双值点），竖线只有两侧电极那两条
    assert np.array_equal(axes.lines[0].get_xdata(), profile.z / NM)
    assert np.array_equal(axes.lines[0].get_ydata(), profile.E_C / EV)
    assert np.array_equal(axes.lines[1].get_ydata(), profile.E_V / EV)
    assert [line.get_xdata()[0] for line in axes.lines[2:]] == [0.0, profile.z[-1] / NM]

    figure = plot_defect_distribution(make_device(), defects)
    axes3d, hist = figure.axes[0], figure.axes[-1]
    assert _scatters(axes3d) == 1 and len(axes3d.lines) == 0  # 只有体缺陷一族，没有界面平面
    # 直方图没有 stack 的层边界/界面竖线，也没有层名/界面计数（单层只有柱子）
    assert len(hist.lines) == 0 and len(hist.texts) == 0


def test_plot_functions_do_not_touch_the_filesystem(tmp_path, monkeypatch, defects):
    """约定：绘图函数内部不保存文件，也不调用 plt.show()。"""
    import matplotlib.pyplot as plt

    def fail(*args, **kwargs):
        raise AssertionError("绘图函数内部不应触发 show/savefig")

    monkeypatch.setattr(plt, "show", fail)
    monkeypatch.setattr(Figure, "savefig", fail)
    plot_defect_distribution(make_device(), defects)
    plot_iv_curves(VOLTAGES, TEMPERATURES, DENSITY)
    plot_it_curves(VOLTAGES, TEMPERATURES, DENSITY)
    assert not list(tmp_path.iterdir())


def test_iv_curves_one_curve_per_temperature():
    """I-V：横轴电压，一条曲线一个温度；纵轴取对数并换算成 A/cm²。"""
    figure = plot_iv_curves(VOLTAGES, TEMPERATURES, DENSITY)
    assert isinstance(figure, Figure)
    axes = figure.axes[0]
    assert axes.get_yscale() == "log"
    assert len(axes.lines) == TEMPERATURES.size
    for it, line in enumerate(axes.lines):
        assert np.array_equal(line.get_xdata(), VOLTAGES)
        assert np.allclose(line.get_ydata(), DENSITY[it] * PER_CM2, rtol=1e-12)
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert labels == ["T = 25.0 °C", "T = 125.0 °C"]  # 图例里温度换算成 °C


def test_it_curves_slice_the_same_data_by_voltage():
    """I-T：同一份扫描结果的另一个切法——横轴温度（°C），一条曲线一个电压。"""
    figure = plot_it_curves(VOLTAGES, TEMPERATURES, DENSITY)
    axes = figure.axes[0]
    assert axes.get_yscale() == "log"
    assert len(axes.lines) == VOLTAGES.size
    for iv, line in enumerate(axes.lines):
        assert np.allclose(line.get_xdata(), TEMPERATURES - constants.zero_Celsius)
        assert np.allclose(line.get_ydata(), DENSITY[:, iv] * PER_CM2, rtol=1e-12)
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert labels == ["U = 2.00 V", "U = 3.00 V", "U = 4.00 V"]


@pytest.mark.parametrize("plot", [plot_iv_curves, plot_it_curves])
def test_curves_draw_sem_error_bars(plot):
    """给了标准误就画 ±σ/√M 误差棒，棒的上下端 = J ± SEM（换算成 A/cm²），两端带小横线
    （端帽）；不给就没有竖线段。"""
    sem = 0.1 * DENSITY
    axes = plot(VOLTAGES, TEMPERATURES, DENSITY, sem).axes[0]
    assert len(axes.collections) == len(axes.containers)  # 每条曲线一组竖线段
    for container in axes.containers:
        line, caps, (bars,) = container.lines
        y = np.asarray(line.get_ydata())
        segments = np.array([seg[:, 1] for seg in bars.get_segments()])
        assert np.allclose(segments[:, 0], 0.9 * y, rtol=1e-12)
        assert np.allclose(segments[:, 1], 1.1 * y, rtol=1e-12)
        # 两端小横线：下端帽、上端帽各一个 Line2D，位置与棒端一致
        assert len(caps) == 2 and all(cap.get_markersize() > 0 for cap in caps)
        assert np.allclose(caps[0].get_ydata(), 0.9 * y, rtol=1e-12)
        assert np.allclose(caps[1].get_ydata(), 1.1 * y, rtol=1e-12)
    axes = plot(VOLTAGES, TEMPERATURES, DENSITY).axes[0]
    assert len(axes.collections) == 0


def test_curves_work_with_a_single_point():
    """只扫一个工作点（1 温度 × 1 电压）时两张图都还能出。"""
    for plot in (plot_iv_curves, plot_it_curves):
        figure = plot(np.array([2.0]), np.array([298.15]), np.array([[1e-4]]))
        assert len(figure.axes[0].lines) == 1


@pytest.mark.parametrize("plot", [plot_iv_curves, plot_it_curves])
def test_curves_show_only_the_total_current(plot):
    """只画总电流：每行（温度 / 电压）一条曲线，图例里没有分机制条目。"""
    with pytest.raises(TypeError):
        plot(VOLTAGES, TEMPERATURES, DENSITY, {"tat": DENSITY, "pf": DENSITY})
    axes = plot(VOLTAGES, TEMPERATURES, DENSITY).axes[0]
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert len(axes.lines) == len(labels)
    assert not any(name in label for label in labels for name in ("total", "TAT", "PF"))


@pytest.mark.parametrize("plot", [plot_iv_curves, plot_it_curves])
def test_rows_are_shaded_light_to_dark_in_blue(plot):
    """行序（温度 / 电压）由 Blues 色系的深浅承担：逐行变深，浅 = 低、深 = 高。"""
    axes = plot(VOLTAGES, TEMPERATURES, DENSITY).axes[0]
    for line in axes.lines:
        assert int(np.argmax(to_rgb(line.get_color()))) == 2  # 蓝通道最大
    shades = [_luminance(line) for line in axes.lines]
    assert np.all(np.diff(shades) < 0.0), shades  # 逐行变暗
    # 最浅那档在白底上仍看得见（对比度 ≥ 2:1 对应相对亮度 ≤ 0.475）
    assert shades[0] <= 0.475
