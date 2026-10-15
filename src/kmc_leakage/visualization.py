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

from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import Blues, tab10
from matplotlib.colors import Colormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # 同时注册 3d 投影
from scipy import constants

from .bands import BandProfile
from .config import DeviceConfig
from .defects import DefectEnsemble, _global_standoff

_NM = constants.nano
_EV = constants.electron_volt
_PER_CM2 = constants.centi ** 2  # A/m^2 -> A/cm^2

# I-V / I-T 图只画总电流，行（I-V 的温度 / I-T 的电压）取 Blues 色系里的深浅。
# 分机制的电流不上图，完整分解看 CSV。
_TOTAL_CMAP = Blues

# 行序在色系里的取样区间：浅 = 低温/低压，深 = 高温/高压。下限 0.50 是为了
# 让最浅那档在白底上仍有 2.1:1 以上的对比度（再浅就快看不见了），上限留在
# 0.95 而不是 1.0 是为了最深那档不糊成黑色。行数 n=1 时取区间中点。
_SHADE_RANGE = (0.50, 0.95)

# 叠层叠画用的两个颜色：界面缺陷（独立物种，单独标色）与层边界/界面位置
_INTERFACE_DEFECT_COLOR = "#c00000"
_LAYER_EDGE_COLOR = "0.45"

# 强耦合簇标注：不属于任何簇的缺陷画黑色，簇间用 tab10 定性色轮换区分。去掉其中的
# 蓝色（下标 0，与深度剖面柱子同色，圆点落在柱子上看不见）与灰色（下标 7，免得和
# 黑色混淆）；簇多于 8 个时颜色会重复，靠位置区分
_UNCLUSTERED_COLOR = "k"
_CLUSTER_COLORS = [tab10(i) for i in range(tab10.N) if i not in (0, 7)]
# 深度剖面上簇成员圆点的行位置（轴内比例）：第 i 个簇画在 START + i·STEP 高度
_CLUSTER_ROW_START, _CLUSTER_ROW_STEP = 0.03, 0.035


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
    figure, _, _ = _draw_defect_distribution(device, defects, n_bins, realizations)
    figure.tight_layout()
    return figure


def plot_defect_clusters(
    device: DeviceConfig,
    defects: DefectEnsemble,
    clusters: Sequence[np.ndarray],
    n_bins: int = 20,
    realizations: Optional[Sequence[DefectEnsemble]] = None,
) -> Figure:
    """缺陷分布图 + 强耦合簇（ADR-022）标注。

    `clusters` 是某个工作点下的强耦合簇（每个簇一组全局缺陷编号，如
    `KMCSimulation.cluster_members`），由调用方算好传入。三维散点不按能级着色、不画
    colorbar：不属于任何簇的缺陷画黑色，同一个簇的成员画同一种颜色（簇间颜色轮换），
    界面缺陷仍以菱形区分。深度剖面的柱子与原图相同，底部每个簇占一行、同色圆点标出
    成员的 z 位置（簇常跨界面，画成 z 范围竖带会盖住整层，故只点成员）。没有簇时
    全部画黑色，标题注明 0 个。
    """
    colors = [_CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i in range(len(clusters))]
    point_colors = np.tile(to_rgba(_UNCLUSTERED_COLOR), (len(defects), 1))
    for members, color in zip(clusters, colors):
        point_colors[members] = color
    figure, axes3d, axes = _draw_defect_distribution(
        device, defects, n_bins, realizations, point_colors
    )
    rows = axes.get_xaxis_transform()  # x 取数据坐标、y 取轴内比例
    for row, (members, color) in enumerate(zip(clusters, colors)):
        z = defects.positions[members, 2] / _NM
        axes.plot(
            z, np.full(z.size, _CLUSTER_ROW_START + row * _CLUSTER_ROW_STEP),
            transform=rows, ls="none", marker="o", ms=5, color=color, mec="white", mew=0.6,
        )
    n_members = int(sum(len(members) for members in clusters))
    axes3d.set_title(
        "N = {}, {} coupled clusters ({} traps)".format(
            len(defects), len(clusters), n_members
        )
    )
    figure.tight_layout()
    return figure


def _draw_defect_distribution(
    device: DeviceConfig,
    defects: DefectEnsemble,
    n_bins: int,
    realizations: Optional[Sequence[DefectEnsemble]],
    point_colors: Optional[np.ndarray] = None,
) -> Tuple[Figure, Axes3D, Axes]:
    """画缺陷分布底图，返回 (figure, 三维散点 axes, 深度剖面 axes)，供两个公开函数共用。

    `point_colors`（N×4 RGBA，逐缺陷）缺省时三维散点按能级着色、界面缺陷单独标色、
    配能级 colorbar；给了就一律按它着色、不画 colorbar，界面缺陷只靠菱形区分。
    """
    figure = _new_figure((11.0, 4.5))
    is_interface = np.asarray(defects.interface_id) >= 0
    bulk, interface = ~is_interface, is_interface
    layer_edges = device.stack.layer_edges / _NM
    inner_edges = layer_edges[1:-1]  # 内部界面 = 层边界去掉两端

    axes3d = figure.add_subplot(1, 2, 1, projection="3d")
    x, y, z = (defects.positions[bulk, i] / _NM for i in range(3))
    if point_colors is None:
        scatter = axes3d.scatter(
            x, y, z, c=defects.ED[bulk] / _EV, cmap="viridis", s=28, depthshade=False
        )
    else:
        axes3d.scatter(x, y, z, color=point_colors[bulk], s=28, depthshade=False)
    lx, ly = device.Lx / _NM, device.Ly / _NM
    for edge in inner_edges:  # 界面平面画成盒子上的一圈虚线
        axes3d.plot(
            [0.0, lx, lx, 0.0, 0.0], [0.0, 0.0, ly, ly, 0.0], [edge] * 5,
            color=_LAYER_EDGE_COLOR, ls="--", lw=0.8,
        )
    if np.any(interface):
        interface_colors = (
            _INTERFACE_DEFECT_COLOR if point_colors is None else point_colors[interface]
        )
        axes3d.scatter(
            *(defects.positions[interface, i] / _NM for i in range(3)),
            color=interface_colors, marker="D", s=20, depthshade=False,
        )
        # 图例用代理句柄：簇着色时界面缺陷颜色各不相同，图例只说明菱形这个形状
        handle_color = _INTERFACE_DEFECT_COLOR if point_colors is None else "k"
        axes3d.legend(
            handles=[Line2D([], [], color=handle_color, marker="D", ms=5, ls="none")],
            labels=["interface defects"], loc="upper left", fontsize=8,
        )
    axes3d.set_xlim(0.0, lx)
    axes3d.set_ylim(0.0, ly)
    axes3d.set_zlim(0.0, device.Lz / _NM)
    axes3d.set_xlabel("x (nm)")
    axes3d.set_ylabel("y (nm)")
    axes3d.set_zlabel("z (nm)")
    axes3d.set_title("Defect positions (N = {})".format(len(defects)))
    if point_colors is None:
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
    return figure, axes3d, axes


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
    axes.hlines(profile.E_F_negative / _EV, -margin, 0.0, color="k", lw=2.0)
    axes.hlines(
        profile.E_F_positive / _EV, lz, lz + margin, color="k", lw=2.0, label=r"electrode $E_F$"
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
    axes.set_ylabel("E (eV), zero at negative-electrode $E_F$")
    axes.set_title("Band diagram (U = {:.2f} V)".format(profile.voltage))
    axes.legend(loc="best", fontsize=9)
    figure.tight_layout()
    return figure


def _current_curves(
    x: np.ndarray,
    rows: np.ndarray,
    labels: Sequence[str],
    x_label: str,
    title: str,
    errors: Optional[np.ndarray] = None,
) -> Figure:
    """一族总电流密度曲线：`rows` 的每一行画一条，纵轴取对数、换算成 A/cm²。

    行序在 Blues 色系里取深浅（浅 = 低温/低压、深 = 高温/高压），见 `_SHADE_RANGE`。
    给了 `errors`（与 `rows` 同形状）就画 ±errors 的竖直误差棒。
    """
    figure = _new_figure((7.0, 5.0))
    axes = figure.add_subplot(1, 1, 1)
    palette = _shades(_TOTAL_CMAP, len(labels))
    for i, label in enumerate(labels):
        yerr = None if errors is None else np.asarray(errors[i], dtype=float) * _PER_CM2
        axes.errorbar(
            x,
            np.asarray(rows[i], dtype=float) * _PER_CM2,
            yerr=yerr,
            label=label,
            color=palette[i],
            linestyle="-",
            marker="o",
            ms=4.0,
            lw=1.5,
            elinewidth=1.0,
            capsize=3.0,  # 误差棒两端画小横线（端帽是额外的 Line2D，会混进 axes.lines）
        )
    axes.set_yscale("log")
    axes.set_xlabel(x_label)
    axes.set_ylabel(r"J (A/cm$^2$)")
    axes.set_title(title)
    axes.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    axes.legend(loc="best", fontsize=9, ncol=2 if len(labels) > 6 else 1)
    figure.tight_layout()
    return figure


def plot_iv_curves(
    voltages: np.ndarray,
    temperatures: np.ndarray,
    current_density: np.ndarray,
    current_density_sem: Optional[np.ndarray] = None,
) -> Figure:
    """I-V 曲线：一条曲线一个温度点，纵轴总电流密度 J 取对数。

    见 docs/physics_model.md 的"可视化接口"一节。

    参数
    ----
    voltages : 电压 [V]，长度 n_V
    temperatures : 温度 [K]，长度 n_T
    current_density : 电流密度 [A/m²]，形状 (n_T, n_V)，即 `SweepResult` /
        `EnsembleSweepResult` 的 `current_density`
    current_density_sem : 平均值的标准误 [A/m²]，形状同上，即
        `EnsembleSweepResult.current_density_sem`；给了就画 ±σ/√M 误差棒
    """
    temperatures = np.asarray(temperatures, dtype=float)
    labels = ["T = {:.1f} °C".format(t - constants.zero_Celsius) for t in temperatures]
    return _current_curves(
        np.asarray(voltages, dtype=float),
        np.asarray(current_density, dtype=float),
        labels,
        "U (V)",
        "I-V characteristics",
        None if current_density_sem is None else np.asarray(current_density_sem, dtype=float),
    )


def plot_current_transient(
    time: np.ndarray,
    current_density: np.ndarray,
    end_time: float,
    fit: Optional[Tuple[float, float, np.ndarray]] = None,
) -> Figure:
    """TDDB 应力下的电流演化 J(t)，双对数坐标（PROJECT_CONTEXT_TDDB.md §7-8）。

    参数
    ----
    time : 各状态的起始时刻 [s]，time[0] = 0 是初始缺陷集合，即 `TDDBResult.time`
    current_density : 各状态的电流密度 [A/m²]，与 `time` 等长；状态 k 保持到 time[k+1]，
        最后一个保持到 `end_time`
    end_time : 轨迹终点 [s]
    fit : 幂律拟合 (n, A, 取样时刻)，J = A·tⁿ（A 以 A/m² 计），画在取样时刻的范围上并标出 n

    t = 0 在对数轴上画不出来，初始状态的电流画成水平虚线；第一次生成之后的状态画成阶梯。
    """
    figure = _new_figure((7.0, 5.0))
    axes = figure.add_subplot(1, 1, 1)
    _draw_current_transient(axes, time, current_density, end_time, fit)
    axes.set_xlabel("t (s)")
    axes.legend(loc="best", fontsize=9)
    figure.tight_layout()
    return figure


def _draw_state_steps(axes: Axes, time: np.ndarray, values: np.ndarray, end_time: float, label: str) -> None:
    """t = 0 的状态画成水平虚线，之后的状态画成阶梯（最后一段到 `end_time`），横轴取对数。"""
    axes.axhline(values[0], color="0.45", ls="--", lw=1.0, label="t = 0 " + label)
    if time.size > 1:
        edges = np.append(time[1:], end_time)
        axes.step(
            edges, np.append(values[1:], values[-1]), where="post",
            color=_shades(_TOTAL_CMAP, 1)[0], lw=1.5, label="simulation",
        )
        axes.set_xscale("log")


def _draw_current_transient(
    axes: Axes,
    time: np.ndarray,
    current_density: np.ndarray,
    end_time: float,
    fit: Optional[Tuple[float, float, np.ndarray]],
) -> np.ndarray:
    """在 `axes` 上画 J(t) 的阶梯与幂律拟合（双对数），返回换算成 A/cm² 的电流密度。"""
    time = np.asarray(time, dtype=float)
    density = np.asarray(current_density, dtype=float) * _PER_CM2
    _draw_state_steps(axes, time, density, end_time, "background")
    if fit is not None:
        exponent, prefactor, times = fit
        times = np.asarray(times, dtype=float)
        axes.plot(
            times, prefactor * times ** exponent * _PER_CM2,
            color="#c00000", ls="--", lw=1.2, label="power law, n = {:.3g}".format(exponent),
        )
    axes.set_yscale("log")
    axes.set_ylabel(r"J (A/cm$^2$)")
    axes.set_title("Current under constant-voltage stress")
    axes.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    return density


def plot_stress_transient(
    time: np.ndarray,
    current_density: np.ndarray,
    temperature: np.ndarray,
    end_time: float,
    fit: Optional[Tuple[float, float, np.ndarray]] = None,
    breakdown_time: Optional[float] = None,
) -> Figure:
    """TDDB 应力下的电流 J(t) 与温度 T(t)，标出击穿点（PROJECT_CONTEXT_TDDB.md §7-8）。

    参数
    ----
    time, current_density, end_time, fit : 同 `plot_current_transient`
    temperature : 各状态的温度 [K]，与 `time` 等长（`TDDBResult.peak_temperature`）
    breakdown_time : 贯穿通路形成的时刻 t_BD（`TDDBResult.breakdown_time`）；给出时在击穿记录 (t_BD, J) 画标记并在
        图例里写出时刻，None 不画

    上图是双对数的 J(t)（与 `plot_current_transient` 相同），下图是温度（°C），横轴共用。
    """
    time = np.asarray(time, dtype=float)
    figure = _new_figure((7.0, 7.0))
    top = figure.add_subplot(2, 1, 1)
    bottom = figure.add_subplot(2, 1, 2, sharex=top)
    density = _draw_current_transient(top, time, current_density, end_time, fit)
    if breakdown_time is not None:
        record = int(np.searchsorted(time, breakdown_time))  # 击穿记录：第一条时刻为 t_BD 的记录
        top.plot(
            [breakdown_time], [density[record]], ls="none", marker="X", ms=9, color="k",
            label="breakdown, t = {:.3g} s".format(breakdown_time),
        )
    top.legend(loc="best", fontsize=9)

    celsius = np.asarray(temperature, dtype=float) - constants.zero_Celsius
    _draw_state_steps(bottom, time, celsius, end_time, "state")
    bottom.set_xlabel("t (s)")
    bottom.set_ylabel("T (°C)")
    bottom.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    bottom.legend(loc="best", fontsize=9)
    figure.tight_layout()
    return figure


def plot_it_curves(
    voltages: np.ndarray,
    temperatures: np.ndarray,
    current_density: np.ndarray,
    current_density_sem: Optional[np.ndarray] = None,
) -> Figure:
    """I-T 曲线：一条曲线一个电压点，横轴温度换算成 °C，纵轴总电流密度 J 取对数。

    参数与 `plot_iv_curves` 完全相同（同一份扫描结果的另一个切法：这里按电压
    分曲线、沿温度轴展开）。
    """
    voltages = np.asarray(voltages, dtype=float)
    density = np.asarray(current_density, dtype=float)
    return _current_curves(
        np.asarray(temperatures, dtype=float) - constants.zero_Celsius,
        density.T,  # 行从温度换成电压
        ["U = {:.2f} V".format(v) for v in voltages],
        "T (°C)",
        "I-T characteristics",
        None if current_density_sem is None else np.asarray(current_density_sem, dtype=float).T,
    )
