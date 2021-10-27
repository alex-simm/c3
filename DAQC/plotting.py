from typing import List
import numpy as np
from matplotlib import pyplot as plt, colors, cm, figure
import plotly.graph_objects as go


def plotSignal(time, signal, filename=None, spectrum_threshold=1e-4) -> figure.Figure:
    """
    Plots a time dependent drive signal and its normalised frequency spectrum.

    Parameters
    ----------
    time
        timestamps
    signal
        signal value
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    spectrum_threshold:
        If not None, only the part of the normalised spectrum whose absolute square
        is larger than this value will be plotted.

    Returns
    -------

    """
    # plot time domain
    time = time.flatten()
    signal = signal.flatten()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].set_title("Signal")
    axs[0].plot(time, signal)
    axs[0].set_xlabel("Time")

    # calculate frequency spectrum
    freq_signal = np.fft.rfft(signal)
    if np.abs(np.max(freq_signal)) > 1e-14:
        normalised = freq_signal / np.max(freq_signal)
    else:
        normalised = freq_signal
    freq = np.fft.rfftfreq(len(time), time[-1] / len(time))

    # cut spectrum if necessary
    if spectrum_threshold is not None:
        limits = np.flatnonzero(np.abs(normalised) ** 2 > spectrum_threshold)
        freq = freq[limits[0] : limits[-1]]
        normalised = normalised[limits[0] : limits[-1]]

    # plot frequency domain
    axs[1].set_title("Spectrum")
    axs[1].plot(freq, normalised.real, label="Re")
    axs[1].plot(freq, normalised.imag, label="Im")
    axs[1].plot(freq, np.abs(normalised) ** 2, label="Square")
    axs[1].set_xlabel("frequency")
    axs[1].legend()

    # show and save
    plt.tight_layout()
    if filename:
        print("saving plot in " + filename)
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    else:
        plt.show()
    plt.close()

    return fig


def plotComplexMatrix(
    M: np.array,
    colourMap: colors.Colormap,
    xlabels: List[str] = None,
    ylabels: List[str] = None,
) -> figure.Figure:
    """
    Plots a complex matrix as a 3d bar plot, where the radius is the bar height and the phase defines
    the bar colour.

    Parameters
    ----------
    M : np.array
      the matrix to plot
    colourMap : matplotlib.colors.Colormap
      a Colormap to be used for the phases
    xlabels : List[str]
      labels for the x-axis
    ylabels : List[str]
      labels for the y-axis

    Returns
    -------
    matplotlib.figure.Figure
      the figure in which the plot was created
    """
    z1 = np.absolute(M)
    z2 = np.angle(M)

    # mesh
    lx = z1.shape[1]
    ly = z1.shape[0]
    xpos, ypos = np.meshgrid(
        np.arange(0.25, lx + 0.25, 1), np.arange(0.25, ly + 0.25, 1)
    )
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    # bar sizes
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz1 = z1.flatten()
    dz2 = z2.flatten()

    # plot the bars
    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")
    for idx, cur_zpos in enumerate(zpos):
        color = colourMap((dz2[idx] + np.pi) / (2 * np.pi))
        axis.bar3d(
            xpos[idx],
            ypos[idx],
            cur_zpos,
            dx[idx],
            dy[idx],
            dz1[idx],
            alpha=1,
            color=color,
        )

    # view, ticks and labels
    axis.view_init(elev=30, azim=-15)
    axis.set_xticks(np.arange(0.5, lx + 0.5, 1))
    axis.set_yticks(np.arange(0.5, ly + 0.5, 1))
    tickFontSize = 13 - 2 * (len(xlabels) / 8)
    if xlabels is not None:
        axis.w_xaxis.set_ticklabels(xlabels, fontsize=tickFontSize)
    if ylabels is not None:
        axis.w_yaxis.set_ticklabels(ylabels, fontsize=tickFontSize, rotation=-65)

    # colour bar
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colourMap),
        ax=axis,
        shrink=0.6,
        pad=0.1,
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    )
    cbar.ax.set_yticklabels(["$-\\pi$", "$\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

    return fig


def plotComplexMatrixAbsOrPhase(
    M: np.array,
    colourMap: colors.Colormap,
    xlabels: List[str] = None,
    ylabels: List[str] = None,
    phase: bool = True,
):
    """
    Plots the phase or absolute value of a complex matrix as a 2d colour plot.

    Parameters
    ----------
    M : np.array
      the matrix to plot
    colourMap : matplotlib.colors.Colormap
      a Colormap to be used for the phases
    xlabels : List[str]
      labels for the x-axis
    ylabels : List[str]
      labels for the y-axis
    phase : bool
      whether the phase or the absolute value should be plotted

    Returns
    -------
    matplotlib.figure.Figure
      the figure in which the plot was created
    """
    data = np.angle(M) if phase else np.abs(M)

    # grid
    lx = M.shape[1]
    ly = M.shape[0]
    extent = [0.5, lx + 0.5, 0.5, ly + 0.5]

    # plot
    fig = plt.figure()
    axis = fig.add_subplot(111)
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    axis.imshow(
        data,
        cmap=colourMap,
        norm=norm,
        interpolation=None,
        extent=extent,
        aspect="auto",
        origin="lower",
    )

    # ticks and labels
    axis.set_xticks(np.arange(1, lx + 1, 1))
    axis.set_yticks(np.arange(1, ly + 1, 1))
    if xlabels is not None:
        axis.xaxis.set_ticklabels(xlabels, fontsize=12, rotation=-90)
    if ylabels is not None:
        axis.yaxis.set_ticklabels(ylabels, fontsize=12)

    # colour bar
    if phase:
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    else:
        norm = colors.Normalize(vmin=0, vmax=np.max(data))
        ticks = np.linspace(0, np.max(data), 5)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colourMap),
        ax=axis,
        shrink=0.8,
        pad=0.1,
        ticks=ticks,
    )
    if phase:
        cbar.ax.set_yticklabels(["$-\\pi$", "$\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

    return fig


def plot_dynamics_plotly(exp, psi_init, seq, filename, goal=-1):
    model = exp.pmap.model
    dUs = exp.partial_propagators
    psi_t = psi_init.numpy()
    pop_t = exp.populations(psi_t, model.lindbladian)
    for gate in seq:
        for du in dUs[gate]:
            psi_t = np.matmul(du.numpy(), psi_t)
            pops = exp.populations(psi_t, model.lindbladian)
            pop_t = np.append(pop_t, pops, axis=1)

    ts = exp.ts
    dt = ts[1] - ts[0]
    ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])

    legends = model.state_labels
    fig = go.Figure()
    for i in range(len(pop_t.T[0])):
        fig.add_trace(
            go.Scatter(x=ts / 1e-9, y=pop_t.T[:, i], mode="lines", name=str(legends[i]))
        )

    fig.show()
    # fig.write_html(filename+".html")

    pass
