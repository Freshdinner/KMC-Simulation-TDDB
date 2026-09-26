"""可视化。

与仿真主流程解耦：绘图函数只接受**已经算好的数据**（器件配置、缺陷列表、
能带剖面数组），不触发仿真、不读配置文件、不落盘；每个函数返回
matplotlib 的 `Figure`，内部不调用 `plt.show()`——保存还是显示由调用方
决定。

新增绘图项只新增函数，不改已有函数签名。

能带按 docs 的势垒形状约定绘制：界面处画直线，**不画镜像力圆化**。

图上文字一律用英文：matplotlib 在没有配置中文字体的机器上会把中文渲染成
方框，这里不去改全局 rcParams，也就不依赖任何字体。
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import Blues, Greens, Oranges
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  注册 3d 投影
from scipy import constants

from .bands import BandProfile
from .config import DeviceConfig
from .defects import DefectEnsemble, _global_standoff

_NM = constants.nano
_EV = constants.electron_volt
_PER_CM2 = constants.centi ** 2  # A/m^2 -> A/cm^2

# I-V / I-T 图上画哪几条曲线，以及各自的色系、线型、标记。
#
# **颜色分机制**（总电流蓝、TAT 橙、PF 绿），行（I-V 的温度 / I-T 的电压）
# 取同一色系里的深浅——两个维度分到两个通道上。线型与标记同样跟机制走，
# 是冗余的第二通道：识别机制不必只靠颜色（色觉障碍、黑白打印下仍可读）。
#
# 只画 TAT 与 PF 两个机制分量。R1 的两个分量（直接/FN 隧穿、热电子发射）
# 比总电流小 7–15 个数量级，画进同一张对数图会把纵轴拉开十几个十倍程，
# 反而看不清 TAT 与 PF 的相对关系；四项完整分解看 CSV。
_TOTAL_STYLE = ("total", Blues, "-", "o")
_COMPONENT_STYLES = (("tat", "TAT", Oranges, "--", "s"), ("pf", "PF", Greens, ":", "^"))

# 行序在色系里的取样区间：浅 = 低温/低压，深 = 高温/高压。下限 0.50 是为了
# 让最浅那档在白底上仍有 2.1:1 以上的对比度（再浅就快看不见了），上限留在
# 0.95 而不是 1.0 是为了最深那档不糊成黑色。行数 n=1 时取区间中点。
_SHADE_RANGE = (0.50, 0.95)

# 叠层叠画用的两个颜色：界面缺陷（独立物种，单独标色）与层边界/界面位置
_INTERFACE_DEFECT_COLOR = "#c00000"
_LAYER_EDGE_COLOR = "0.45"


def _shades(cmap: Colormap, n: int) -> List[Tuple[float, float, float, float]]:
    """一个色系里按行数取 n 档深浅，从浅到深。"""
    lo, hi = _SHADE_RANGE
    positions = np.linspace(lo, hi, n) if n > 1 else np.array([0.5 * (lo + hi)])
    return [cmap(float(p)) for p in positions]


def _new_figure(figsize: Tuple[float, float]) -> Figure:
    """建一个带 canvas 的 Figure（不经 pyplot，便于测试与批量出图）。"""
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


def plot_defect_distribution(
    device: DeviceConfig,
    defects: DefectEnsemble,
    n_bins: int = 20,
    realizations: Optional[Sequence[DefectEnsemble]] = None,
) -> Figure:
    """器件体内缺陷分布：三维散点（按能级着色）+ 沿 z 的数密度直方图。

    三维散点画 `defects` 这**一个代表实现**；深度直方图汇总 `realizations` 给的
    **M 个实现**再除以 M（每实现平均密度），压低单实现的泊松噪声——默认 N≈68、
    每 bin 才几个缺陷，单实现的直方图基本是散粒噪声，看不出界面富集。`realizations`
    缺省时退回只用 `defects` 一个实现（M=1，与改造前逐位相同）。

    横轴只取**可采样窗口** [standoff, Lz−standoff]、从 standoff 起分箱：两端 standoff
    内按数值稳定性要求本就不放缺陷（`min_separation`），画进来只是空白，裁掉后
    边缘那根柱子恰好落在富集峰所在的可放置平面上、不再被死区体积摊低。

    叠层（层数 > 1）时两幅图都叠画层边界与界面位置，**界面缺陷单独标色**
    （ADR-016 ①）：它们是独立物种、按面密度 Nₛ 放在界面平面上，与体缺陷
    的体密度不是同一个量纲，所以直方图的柱子只统计体缺陷，界面缺陷改用
    界面处的竖线加个数标注。单层没有内部界面，两幅图与叠层改造前逐位相同。
    """
    figure = _new_figure((11.0, 4.5))
    is_interface = np.asarray(defects.interface_id) >= 0
    bulk, interface = ~is_interface, is_interface
    layer_edges = device.stack.layer_edges / _NM
    inner_edges = layer_edges[1:-1]  # 内部界面 = 层边界去掉两端

    axes3d = figure.add_subplot(1, 2, 1, projection="3d")
    x, y, z = (defects.positions[bulk, i] / _NM for i in range(3))
    scatter = axes3d.scatter(x, y, z, c=defects.ED[bulk] / _EV, cmap="viridis", s=28, depthshade=False)
    lx, ly = device.Lx / _NM, device.Ly / _NM
    for edge in inner_edges:  # 界面平面画成盒子上的一圈虚线
        axes3d.plot(
            [0.0, lx, lx, 0.0, 0.0], [0.0, 0.0, ly, ly, 0.0], [edge] * 5,
            color=_LAYER_EDGE_COLOR, ls="--", lw=0.8,
        )
    if np.any(interface):
        axes3d.scatter(
            *(defects.positions[interface, i] / _NM for i in range(3)),
            color=_INTERFACE_DEFECT_COLOR, marker="D", s=20, depthshade=False,
            label="interface defects",
        )
        axes3d.legend(loc="upper left", fontsize=8)
    axes3d.set_xlim(0.0, lx)
    axes3d.set_ylim(0.0, ly)
    axes3d.set_zlim(0.0, device.Lz / _NM)
    axes3d.set_xlabel("x (nm)")
    axes3d.set_ylabel("y (nm)")
    axes3d.set_zlabel("z (nm)")
    axes3d.set_title("Defect positions (N = {})".format(len(defects)))
    figure.colorbar(scatter, ax=axes3d, shrink=0.7, pad=0.12, label="$E_D$ (eV)")

    axes = figure.add_subplot(1, 2, 2)
    pool = list(realizations) if realizations is not None else [defects]
    bulk_z = np.concatenate(
        [e.positions[np.asarray(e.interface_id) < 0, 2] for e in pool]
    ) / _NM
    # 横轴只取可采样窗口 [standoff, Lz−standoff]：两端 standoff 内按数值稳定性要求
    # 本就不放缺陷，画进来只是空白，直方图从 standoff 起分箱、图上不留空
    standoff = _global_standoff(device) / _NM
    z_lo, z_hi = standoff, device.Lz / _NM - standoff
    counts, edges = np.histogram(bulk_z, bins=n_bins, range=(z_lo, z_hi))
    width = edges[1] - edges[0]
    bin_volume_cm3 = device.Lx * device.Ly * (width * _NM) / constants.centi ** 3
    axes.bar(
        0.5 * (edges[:-1] + edges[1:]),
        counts / (len(pool) * bin_volume_cm3),  # 除以 M：每实现平均密度
        width=0.9 * width,
        color="#4878a8",
    )
    axes.set_xlabel("z (nm)")
    axes.set_ylabel(r"defect density (cm$^{-3}$)")
    axes.set_title("Depth profile")
    axes.set_xlim(z_lo, z_hi)
    _annotate_layers(axes, device, defects, layer_edges)
    figure.tight_layout()
    return figure


def _annotate_layers(
    axes, device: DeviceConfig, defects: DefectEnsemble, layer_edges: np.ndarray
) -> None:
    """在深度剖面上叠画层边界、层材料名与各界面的界面缺陷个数（ADR-016 ①）。

    单层（只有两端边界）时什么都不画，深度剖面与叠层改造前逐位相同。
    """
    layers = device.stack.layers
    if len(layers) < 2:
        return
    top = axes.get_ylim()[1]
    for edge in layer_edges[1:-1]:
        axes.axvline(edge, color=_LAYER_EDGE_COLOR, ls="--", lw=1.0)
    for i, layer in enumerate(layers):
        axes.text(
            0.5 * (layer_edges[i] + layer_edges[i + 1]), 0.96 * top, layer.material,
            ha="center", va="top", fontsize=8, color="0.25",
        )
    interface_id = np.asarray(defects.interface_id)
    for j, edge in enumerate(layer_edges[1:-1]):
        n_interface = int(np.count_nonzero(interface_id == j))
        if n_interface == 0:
            continue
        axes.axvline(edge, color=_INTERFACE_DEFECT_COLOR, lw=1.5)
        axes.text(
            edge, 0.80 * top, " {}".format(n_interface),
            ha="left", va="top", fontsize=8, color=_INTERFACE_DEFECT_COLOR,
        )


def plot_band_diagram(
    profile: BandProfile,
    defect_z: Optional[np.ndarray] = None,
    defect_levels: Optional[np.ndarray] = None,
) -> Figure:
    """沿 z 的能带剖面：电极费米能级、导带底/价带顶、缺陷能级散点。

    叠层的导带/价带在内部界面处画**真实台阶**（同一个 z 上取两侧两个值，
    ADR-016 ②），不把台阶连成斜坡；界面位置另画一条竖直参考线。单层没有
    内部界面，画出来与叠层改造前逐位相同。

    参数
    ----
    profile : 已经算好的一维能带剖面（bands.band_profile 的返回值）
    defect_z : 缺陷 z 坐标 [m]，可选
    defect_levels : 缺陷能级（电子总能量）[J]，与 defect_z 一一对应，可选
    """
    figure = _new_figure((7.0, 5.0))
    axes = figure.add_subplot(1, 1, 1)
    z_steps, E_C, E_V = profile.stepped
    z_nm = profile.z / _NM
    lz = z_nm[-1]

    axes.plot(z_steps / _NM, E_C / _EV, color="#1f4e79", lw=2.0, label=r"$E_C$")
    axes.plot(z_steps / _NM, E_V / _EV, color="#7f3f00", lw=2.0, label=r"$E_V$")
    for interface in np.asarray(profile.interface_z) / _NM:
        axes.axvline(interface, color=_LAYER_EDGE_COLOR, ls="--", lw=0.8)
    # 电极费米能级画在介质两侧的电极区域内
    margin = 0.12 * lz
    axes.hlines(profile.E_F_cathode / _EV, -margin, 0.0, color="k", lw=2.0)
    axes.hlines(
        profile.E_F_anode / _EV, lz, lz + margin, color="k", lw=2.0, label=r"electrode $E_F$"
    )
    axes.axvline(0.0, color="0.5", lw=0.8)
    axes.axvline(lz, color="0.5", lw=0.8)

    if defect_z is not None and defect_levels is not None:
        axes.scatter(
            np.asarray(defect_z) / _NM,
            np.asarray(defect_levels) / _EV,
            s=18,
            color="#c00000",
            zorder=3,
            label="defect levels",
        )

    axes.set_xlim(-margin, lz + margin)
    axes.set_xlabel("z (nm)")
    axes.set_ylabel("E (eV), zero at cathode $E_F$")
    axes.set_title("Band diagram (U = {:.2f} V)".format(profile.voltage))
    axes.legend(loc="best", fontsize=9)
    figure.tight_layout()
    return figure


class _Component(NamedTuple):
    """总电流之外画在同一张图上的机制分量，行布局与总电流矩阵一致。"""

    label: str
    cmap: Colormap
    linestyle: str
    marker: str
    rows: np.ndarray  # [A/m^2]


def _components(
    by_mechanism: Optional[Dict[str, np.ndarray]], transpose: bool
) -> Tuple[_Component, ...]:
    """从机制分解字典里挑出要画的分量；`transpose` 给 I-T 的行布局用。

    输入就是 `SweepResult` / `EnsembleSweepResult` 的 `by_mechanism`
    （形状 (温度点数, 电压点数) 的矩阵字典），传 None 表示不画分量。
    """
    if by_mechanism is None:
        return ()
    out = []
    for name, label, cmap, linestyle, marker in _COMPONENT_STYLES:
        rows = np.asarray(by_mechanism[name], dtype=float)
        out.append(_Component(label, cmap, linestyle, marker, rows.T if transpose else rows))
    return tuple(out)


def _log_safe(values: np.ndarray) -> np.ndarray:
    """把非正值换成 nan——对数轴画不出 J ≤ 0。

    总电流恒为正（R1 按构造非零），但 TAT / PF 分量在统计窗口内没有对应
    事件时恰好是 0。留成 nan 会在曲线上断开，比让 matplotlib 静默丢点更
    容易看出"这个工作点上没有这个机制"。
    """
    out = np.array(values, dtype=float)
    out[~(out > 0.0)] = np.nan
    return out


def _current_curves(
    x: np.ndarray,
    rows: np.ndarray,
    labels: Sequence[str],
    x_label: str,
    title: str,
    components: Sequence[_Component] = (),
) -> Figure:
    """一族电流密度曲线：`rows` 的每一行画一条，纵轴取对数、换算成 A/cm²。

    `components` 非空时每个分量在**同一张图**上再画一族曲线。颜色分机制
    （总电流蓝、TAT 橙、PF 绿），行序在各自色系里取深浅，见 `_SHADE_RANGE`。
    图例分两段——先各行的温度/电压（画成总电流那条线，深浅即行序），再拿
    三条 `Line2D` 代理标出机制的颜色与线型，否则 行数 × 3 条曲线会把图例
    胀成一整块。
    """
    figure = _new_figure((7.0, 5.0))
    axes = figure.add_subplot(1, 1, 1)
    n = len(labels)
    _, total_cmap, total_linestyle, total_marker = _TOTAL_STYLE
    palettes = [_shades(cmap, n) for cmap in (total_cmap,) + tuple(c.cmap for c in components)]
    for i, label in enumerate(labels):
        series = [(rows[i], total_linestyle, total_marker, label)]
        series += [(c.rows[i], c.linestyle, c.marker, "_nolegend_") for c in components]
        for (values, linestyle, marker, entry), palette in zip(series, palettes):
            axes.plot(
                x,
                _log_safe(np.asarray(values, dtype=float) * _PER_CM2),
                label=entry,
                color=palette[i],
                linestyle=linestyle,
                marker=marker,
                ms=4.0,
                lw=1.5,
            )
    axes.set_yscale("log")
    axes.set_xlabel(x_label)
    axes.set_ylabel(r"J (A/cm$^2$)")
    axes.set_title(title)
    axes.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    handles = list(axes.get_legend_handles_labels()[0])
    if components:
        styles = [_TOTAL_STYLE] + [(c.label, c.cmap, c.linestyle, c.marker) for c in components]
        handles += [
            # 图例代理取色系中点，与只有一行时实际画的颜色是同一档
            Line2D([], [], color=_shades(cmap, 1)[0], ls=ls, marker=marker, ms=4.0, label=name)
            for name, cmap, ls, marker in styles
        ]
    axes.legend(handles=handles, loc="best", fontsize=9, ncol=2 if len(handles) > 6 else 1)
    figure.tight_layout()
    return figure


def plot_iv_curves(
    voltages: np.ndarray,
    temperatures: np.ndarray,
    current_density: np.ndarray,
    by_mechanism: Optional[Dict[str, np.ndarray]] = None,
) -> Figure:
    """I-V 曲线：一条曲线一个温度点，纵轴 J 取对数。

    见 docs/physics_model.md 的"可视化接口"一节。

    参数
    ----
    voltages : 电压 [V]，长度 n_V
    temperatures : 温度 [K]，长度 n_T
    current_density : 电流密度 [A/m²]，形状 (n_T, n_V)，即 `SweepResult` /
        `EnsembleSweepResult` 的 `current_density`
    by_mechanism : 机制分解 [A/m²]，`SweepResult.by_mechanism` 那样的矩阵字典。
        传入时把 **TAT 与 PF 两项画在同一张图上**（颜色分机制、深浅分温度）；
        传 None 就只画总电流
    """
    temperatures = np.asarray(temperatures, dtype=float)
    labels = ["T = {:.1f} °C".format(t - constants.zero_Celsius) for t in temperatures]
    return _current_curves(
        np.asarray(voltages, dtype=float),
        np.asarray(current_density, dtype=float),
        labels,
        "U (V)",
        "I-V characteristics",
        _components(by_mechanism, transpose=False),
    )


def plot_it_curves(
    voltages: np.ndarray,
    temperatures: np.ndarray,
    current_density: np.ndarray,
    by_mechanism: Optional[Dict[str, np.ndarray]] = None,
) -> Figure:
    """I-T 曲线：一条曲线一个电压点，横轴温度换算成 °C，纵轴 J 取对数。

    参数与 `plot_iv_curves` 完全相同（同一份扫描结果的另一个切法：这里按电压
    分曲线、沿温度轴展开）。机制分量矩阵跟着一起转置，TAT 与 PF 同样画在这
    张图上、深浅分电压。
    """
    voltages = np.asarray(voltages, dtype=float)
    density = np.asarray(current_density, dtype=float)
    return _current_curves(
        np.asarray(temperatures, dtype=float) - constants.zero_Celsius,
        density.T,  # 行从温度换成电压
        ["U = {:.2f} V".format(v) for v in voltages],
        "T (°C)",
        "I-T characteristics",
        _components(by_mechanism, transpose=True),
    )
