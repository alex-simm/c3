from typing import List
import numpy as np
from matplotlib import pyplot as plt, colors, cm
import plotly.graph_objects as go

from four_level_transmons.utilities import getQubitsPopulation
from c3.experiment import Experiment
import tensorflow as tf


def plotSignal(time, signal, filename=None):
    """
    Plots a time dependent drive signal.

    Parameters
    ----------
    time
        timestamps
    signal
        the function values
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    time = time.flatten()
    signal = signal.flatten()
    plt.figure()
    plt.plot(time, signal)
    plt.xlabel("Time")
    plt.ylabel("Signal")

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotSignalSpectrum(
    time: np.array,
    signal: np.array,
    spectrum_threshold: float = 1e-4,
    filename: str = None,
):
    """
    Plots the normalised frequency spectrum of a time-dependent signal.

    Parameters
    ----------
    time: np.array
        timestamps
    signal: np.array
        signal value
    spectrum_threshold: float
        If not None, only the part of the normalised spectrum whose absolute square
        is larger than this value will be plotted.
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    time = time.flatten()
    signal = signal.flatten()
    plt.figure()

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
    plt.plot(freq, normalised.real, label="Re")
    plt.plot(freq, normalised.imag, label="Im")
    plt.plot(freq, np.abs(normalised) ** 2, label="Square")
    plt.xlabel("frequency")
    plt.legend()

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotSignalAndSpectrum(
    time, signal, spectrum_threshold: float = 1e-4, filename=None
):
    """
    Plots a time dependent drive signal and its frequency spectrum.

    Parameters
    ----------
    time
        timestamps
    signal
        the function values
    spectrum_threshold: float
        If not None, only the part of the normalised spectrum whose absolute square
        is larger than this value will be plotted.
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    time = time.flatten()
    signal = signal.flatten()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # time domain
    ax[0].plot(time, signal)
    ax[0].set_xlabel("Time")
    ax[0].set_xlabel("Signal")

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
    ax[1].plot(freq, normalised.real, label="Re")
    ax[1].plot(freq, normalised.imag, label="Im")
    ax[1].plot(freq, np.abs(normalised) ** 2, label="Square")
    ax[1].set_xlabel("frequency")
    ax[1].legend()

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotComplexMatrix(
    M: np.array,
    colourMap: str = "nipy_spectral",
    xlabels: List[str] = None,
    ylabels: List[str] = None,
    filename: str = None,
):
    """
    Plots a complex matrix as a 3d bar plot, where the radius is the bar height and the phase defines
    the bar colour.

    Parameters
    ----------
    M : np.array
      the matrix to plot
    colourMap : str
      a name of a colormap to be used for the phases
    xlabels : List[str]
      labels for the x-axis
    ylabels : List[str]
      labels for the y-axis
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
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
    colours = cm.get_cmap(colourMap)
    for idx, cur_zpos in enumerate(zpos):
        color = colours((dz2[idx] + np.pi) / (2 * np.pi))
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
    if xlabels is not None:
        axis.w_xaxis.set_ticklabels(xlabels, fontsize=13 - 2 * (len(xlabels) / 8))
    if ylabels is not None:
        axis.w_yaxis.set_ticklabels(
            ylabels, fontsize=13 - 2 * (len(ylabels) / 8), rotation=-65
        )

    # colour bar
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colours),
        ax=axis,
        shrink=0.6,
        pad=0.1,
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    )
    cbar.ax.set_yticklabels(["$-\\pi$", "$\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotComplexMatrixAbsOrPhase(
    M: np.array,
    colourMap: str = "nipy_spectral",
    xlabels: List[str] = None,
    ylabels: List[str] = None,
    phase: bool = True,
    filename: str = None,
):
    """
    Plots the phase or absolute value of a complex matrix as a 2d colour plot.

    Parameters
    ----------
    M : np.array
      the matrix to plot
    colourMap : str
      name of a colour map to be used for the phases
    xlabels : List[str]
      labels for the x-axis
    ylabels : List[str]
      labels for the y-axis
    phase : bool
      whether the phase or the absolute value should be plotted
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
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
    colours = cm.get_cmap(colourMap)
    axis.imshow(
        data,
        cmap=colours,
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
        cm.ScalarMappable(norm=norm, cmap=colours),
        ax=axis,
        shrink=0.8,
        pad=0.1,
        ticks=ticks,
    )
    if phase:
        cbar.ax.set_yticklabels(["$-\\pi$", "$\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotPopulation(
    exp: Experiment,
    population: np.array,
    sequence: List[str],
    labels: List[str] = None,
    usePlotly=False,
    vertical_lines=False,
    filename: str = None,
):
    """
    Plots time dependent populations. They need to be calculated with `runTimeEvolution` first.
    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    population: np.array
        The population
    sequence: List[str]
        List of gate names that will be applied to the state
    labels: List[str]
        Optional list of names for the levels. If none, the default list from the experiment will be used.
    usePlotly: bool
        Whether to use Plotly or Matplotlib
    vertical_lines: bool
        If true, this add a dotted vertical line after each gate
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    # calculate the time dependent level population
    model = exp.pmap.model

    # timestamps
    dt = exp.ts[1] - exp.ts[0]
    ts = np.linspace(0.0, dt * population.shape[1], population.shape[1])

    legend_labels = labels if labels else model.state_labels
    labelX = "Time [ns]"
    labelY = "Population"

    # create the plot
    if usePlotly:
        fig = go.Figure()
        for i in range(len(population.T[0])):
            fig.add_trace(
                go.Scatter(
                    x=ts / 1e-9,
                    y=population.T[:, i],
                    mode="lines",
                    name=str(legend_labels[i]),
                )
            )
        fig.update_layout(xaxis_title=labelX, yaxis_title=labelY)
    else:
        fig, axs = plt.subplots(1, 1, figsize=[10, 5])
        axs.plot(ts / 1e-9, population.T)

        # set plot properties
        axs.tick_params(direction="in", left=True, right=True, top=False, bottom=True)
        axs.set_xlabel(labelX)
        axs.set_ylabel(labelY)
        plt.legend(
            legend_labels,
            ncol=int(np.ceil(model.tot_dim / 15)),
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
        )
        plt.tight_layout()

    # plot vertical lines; TODO: does not work with Plotly yet!
    if (not usePlotly) and vertical_lines and len(sequence) > 0:
        gate_steps = [exp.partial_propagators[g].shape[0] for g in sequence]
        for i in range(1, len(gate_steps)):
            gate_steps[i] += gate_steps[i - 1]
        gate_times = gate_steps * dt
        if usePlotly:
            for t in gate_times:
                fig.add_vline(
                    x=t / 1e-9, line_width=1, line_dash="dot", line_color="black"
                )
        else:
            plt.vlines(
                x=gate_times / 1e-9,
                ymin=tf.reduce_min(population),
                ymax=tf.reduce_max(population),
                linestyles=":",
                colors="black",
            )

    # show and save
    if usePlotly:
        if filename:
            fig.write_image(filename)
        else:
            fig.show()
    else:
        if filename:
            plt.savefig(filename, bbox_inches="tight", dpi=100)
            plt.close()
        else:
            plt.show()


def plotSplittedPopulation(
    exp: Experiment,
    population: np.array,
    sequence: List[str],
    vertical_lines=False,
    filename: str = None,
) -> None:
    """
    Plots time dependent populations for multiple qubits in separate plots.

    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    population: np.array
        The population
    sequence: List[str]
        List of gate names that will be applied to the state
    vertical_lines: bool
        If true, this add a dotted vertical line after each gate
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    -------
    """
    # calculate the time dependent level population
    model = exp.pmap.model
    dims = [s.hilbert_dim for s in model.subsystems.values()]
    splitted = getQubitsPopulation(population, dims)

    # timestamps
    dt = exp.ts[1] - exp.ts[0]
    ts = np.linspace(0.0, dt * population.shape[1], population.shape[1])

    # positions of vertical lines
    gate_steps = [exp.partial_propagators[g].shape[0] for g in sequence]
    for i in range(1, len(gate_steps)):
        gate_steps[i] += gate_steps[i - 1]
    gate_times = gate_steps * dt

    # create both subplots
    fig, axs = plt.subplots(1, len(splitted), sharey="all")
    for idx, ax in enumerate(axs):
        ax.plot(ts / 1e-9, splitted[idx].T)
        if vertical_lines:
            ax.vlines(
                gate_times / 1e-9,
                tf.reduce_min(population),
                tf.reduce_max(population),
                linestyles=":",
                colors="black",
            )

        # set plot properties
        ax.tick_params(direction="in", left=True, right=True, top=False, bottom=True)
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Population")
        ax.legend([str(x) for x in np.arange(dims[idx])])

    plt.tight_layout()

    # show and save
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    else:
        plt.show()
    plt.close()
