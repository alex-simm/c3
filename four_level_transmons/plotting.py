import string
from typing import List, Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt, colors, cm, lines

from four_level_transmons.utilities import getQubitsPopulation
from c3.experiment import Experiment
import tensorflow as tf


def plotData(x, y, xlabel: str = None, ylabel: str = None, filename: str = None):
    """
    Plots a set of data points.

    Parameters
    ----------
    x,y
        data values
    xlabel, ylabel:str
        Optional labels for the axes.
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    if isinstance(x, (np.ndarray, np.generic)):
        x = x.flatten()
    if isinstance(y, (np.ndarray, np.generic)):
        y = y.flatten()
    plt.figure()
    plt.plot(x, y)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotSignal(
        time: np.array,
        real: np.array,
        imag=None,
        envelope=None,
        pwcTimes=None,
        min_signal_limit=0.2e9,
        filename=None,
):
    """
    Plots a time dependent real or complex signal.

    Parameters
    ----------
    time
        timestamps
    real
        real part of the function value
    imag
        imaginary part of the function value, or None
    envelope
        Envelope of the signal. If not none, plot this array as a dashed line.
    pwcTimes
        Timestamps of a PWC signal. If this and envelope are not none, plots the timestamps as circles.
    min_signal_limit
        If not null, the limits for the y axis in the time domain will be at least this.
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    time = time.flatten()
    real = real.flatten()
    if imag is not None:
        imag = imag.flatten()

    fig, ax = plt.subplots(1, 1)
    drawSignal(
        ax[0] if (ax is List) > 0 else ax,
        time,
        real=real,
        imag=imag,
        envelope=envelope,
        pwcTimes=pwcTimes,
        min_signal_limit=min_signal_limit,
    )

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotSpectrum(
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
        real signal values
    spectrum_threshold: float
        If not None, only the part of the normalised spectrum whose absolute square
        is larger than this value will be plotted.
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    time = time.flatten()
    signal = signal.flatten()

    fig, ax = plt.subplots(1, 1)
    drawSpectrum(ax[1], time, signal, spectrum_threshold)

    fig.canvas.draw()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotSignalAndSpectrum(
        time: np.array,
        real: np.array,
        imag=None,
        envelope=None,
        pwcTimes=None,
        spectralThreshold: float = 1e-5,
        min_signal_limit=None,
        filename=None,
        states: List[Tuple[float, str]] = None,
        spectralCutoff: Tuple[float, float]=None
):
    """
    Plots a time dependent drive signal and its frequency spectrum.

    Parameters
    ----------
    time
        timestamps
    real
        real part of the signal values
    imag
        imaginary part of the signal values, or None
    envelope
        Envelope of the signal. If not none, plot this array as a dashed line.
    pwcTimes
        Timestamps of a PWC signal. If this and envelope are not none, plots the timestamps as circles.
    spectralThreshold: float
        If not None, only the part of the normalised spectrum whose absolute square
        is larger than this value will be plotted.
    min_signal_limit
        If not null, the limits for the y axis in the time domain will be at least this.
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    time = time.flatten()
    real = real.flatten()
    if imag is not None:
        imag = imag.flatten()
        spectrumSignal = real ** 2 + imag ** 2
    else:
        spectrumSignal = real
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    drawSignal(
        ax[0],
        time,
        real=real,
        imag=imag,
        envelope=envelope,
        pwcTimes=pwcTimes,
        min_signal_limit=min_signal_limit,
    )
    drawSpectrum(
        ax[1],
        time,
        signal=spectrumSignal,
        spectralThreshold=spectralThreshold,
        states=states,
        spectralCutoff=spectralCutoff
    )

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def drawSignal(
        axes: plt.Axes,
        time: np.array,
        real: np.array,
        imag=None,
        envelope=None,
        pwcTimes=None,
        min_signal_limit=None,
):
    """
    Draws real part, imaginary part, and absolute square of a complex signal into an Axes object.
    """
    if imag is not None:
        absSq = np.abs(real) ** 2 + np.abs(imag) ** 2
        axes.plot(time, real, label="Re")
        axes.plot(time, imag, label="Im")
        axes.plot(time, absSq, label="AbsSq")
        axes.legend()
        maxVal = np.max(
            [np.max(np.abs(real)), np.max(np.abs(real)), np.max(np.abs(real))]
        )
    else:
        maxVal = np.max(np.abs(real))
        axes.plot(time, real)

    if envelope is not None:
        axes.plot(envelope[0].flatten(), envelope[1].flatten(), "--", color="black")
        if pwcTimes is not None:
            indices = [(np.abs(time - t)).argmin() for t in pwcTimes]
            axes.plot(envelope[0][indices], envelope[1][indices], "o", color="black")

    axes.set_xlabel("Time [ns]")
    axes.set_ylabel("Amplitude [Hz]")
    if min_signal_limit is not None:
        limit = max(min_signal_limit, 1.1 * maxVal)
        axes.set_ylim(-limit, limit)


def drawSpectrum(
        axes: plt.Axes,
        time: np.array,
        signal: np.array,
        spectralThreshold: float = 1e-4,
        states: List[Tuple[float, str]] = None,
        spectralCutoff: Tuple[float, float] = None
):
    """
    Draws the frequency spectrum of a time signal into an Axes object.
    """
    # calculate frequency spectrum
    if np.iscomplex(signal[0]):
        freq_signal = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(len(time), time[-1] / len(time)))
    else:
        freq_signal = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(len(time), time[-1] / len(time))
    freq_signal_abs = np.abs(freq_signal)
    if np.max(freq_signal_abs) > 1e-14:
        normalised = freq_signal / np.max(freq_signal_abs)
    else:
        normalised = freq_signal

    # cut spectrum if necessary
    if spectralThreshold is not None:
        limits = np.flatnonzero(np.abs(normalised) ** 2 > spectralThreshold)
        if len(limits) > 1:
            start = max(limits[0] - 1, 0)
            end = min(limits[-1] + 1, len(freq) - 1)
            freq = freq[start: end]
            normalised = normalised[start: end]
    if spectralCutoff is not None:
        leftCut = np.argwhere(freq > spectralCutoff[0]).flatten()
        leftIdx = leftCut[0] if len(leftCut) > 0 else 0
        rightCut = np.argwhere(freq < spectralCutoff[1]).flatten()
        rightIdx = rightCut[-1] if len(rightCut) > 0 else -1
        freq = freq[leftIdx:rightIdx]
        normalised = normalised[leftIdx:rightIdx]

    # plot frequency domain
    axes.plot(freq, normalised.real, label="Re")
    axes.plot(freq, normalised.imag, label="Im")
    axes.plot(freq, np.abs(normalised) ** 2, label="Square")
    axes.set_xlabel("frequency")
    axes.legend()

    # add vertical lines for transition frequencies
    x_bounds = axes.get_xlim()
    if states is not None:
        # only transitions within a range
        inRange = [s for s in states if x_bounds[0] < s[0] < x_bounds[1]]
        energies = np.array([s[0] for s in inRange])
        labels = np.array([s[1] for s in inRange])

        # binned transitions within the range
        bins = np.linspace(x_bounds[0], x_bounds[1], 40)
        digitised = np.digitize(energies, bins)
        binnedEnergies = [energies[digitised == i] for i in range(1, len(bins))]
        binnedLabels = [labels[digitised == i] for i in range(1, len(bins))]

        # a tick for each transition energy
        for E in energies:
            axes.vlines(E, 1.0, 1.1, colors=["black"], linestyles="-")

        filtered = list(filter(lambda x: len(x[0]) > 0, zip(binnedLabels, binnedEnergies)))
        letters = iter(list(string.ascii_uppercase))
        for i in range(len(filtered)):
            label = " / ".join(list(filtered[i][0]))
            mean = np.array(filtered[i][1]).mean()
            '''
            # plot rotated transition labels above each tick 
            axes.annotate(
                text=label,
                xy=((mean - x_bounds[0]) / (x_bounds[1] - x_bounds[0]), 1.01),
                xycoords="axes fraction",
                verticalalignment="bottom",
                horizontalalignment="right",
                rotation=270,
            )
            '''
            # plot transition labels below the plot with a letter at each tick
            letter = next(letters)
            axes.annotate(
                text=letter,
                xy=((mean - x_bounds[0]) / (x_bounds[1] - x_bounds[0]), 1.01),
                xycoords="axes fraction",
                verticalalignment="bottom",
                horizontalalignment="center"
            )
            axes.annotate(f"{letter}: {label}", (0, 0), (0, -40 - i * 10), xycoords='axes fraction',
                          textcoords='offset points', va='top')


def plotComplexMatrix(
        M: np.array,
        colourMap: str = "nipy_spectral",
        xlabels: List[str] = None,
        ylabels: List[str] = None,
        zlimits: Tuple[int, int] = (0, 1),
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
    zlimits : Tuple[int, int]
      Limit for the z-axis. If none, the limits will be set automatically.
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
    if zlimits is not None:
        axis.set_zlim(zlimits[0], zlimits[1])

    # colour bar
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colours),
        ax=axis,
        shrink=0.6,
        pad=0.1,
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    )
    cbar.ax.set_yticklabels(["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

    # show and save
    fig.canvas.draw()
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
    colours = cm.get_cmap(colourMap)
    if phase:
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    else:
        norm = colors.Normalize(vmin=0, vmax=1)
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
        ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    else:
        ticks = np.linspace(0, 1, 5)
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


def plotComplexMatrixHinton(
    M: np.array,
    maxAbsolute: float = None,
    xlabels: List[str] = None,
    ylabels: List[str] = None,
    colourMap: str = "nipy_spectral",
    gridColour: str = None,
    filename: str = None,
):
    # grid
    lx = M.shape[1]
    ly = M.shape[0]

    # figure
    fig = plt.figure(facecolor='white', figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor("white")
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_xlim(-0.5, M.shape[0] - 0.5)
    ax.set_ylim(-0.5, M.shape[1] - 0.5)

    # add patches
    colours = cm.get_cmap(colourMap)
    if maxAbsolute is None:
        maxAbsolute = 2 ** np.ceil(np.log2(np.abs(M).max()))

    for (x, y), w in np.ndenumerate(M):
        color = colours((np.angle(w) + np.pi) / (2 * np.pi))
        size = min(1, np.sqrt(np.absolute(w) / maxAbsolute))
        rect = plt.Rectangle(
            (x - size / 2, y - size / 2), size, size, facecolor=color, edgecolor=color
        )
        ax.add_patch(rect)

    # plot grid lines
    if gridColour:
        for x in range(1, M.shape[0]):
            ax.add_line(
                lines.Line2D(
                    [x - 0.5, x - 0.5],
                    [-0.5, M.shape[1] - 0.5],
                    lw=1,
                    color=gridColour,
                    axes=ax,
                )
            )
        for y in range(1, M.shape[1]):
            ax.add_line(
                lines.Line2D(
                    [-0.5, M.shape[0] - 0.5],
                    [y - 0.5, y - 0.5],
                    lw=1,
                    color=gridColour,
                    axes=ax,
                )
            )

    ax.autoscale_view()
    ax.invert_yaxis()

    # ticks and labels
    ax.set_xticks(np.arange(0, lx, 1))
    ax.set_yticks(np.arange(0, ly, 1))
    if xlabels is not None:
        ax.xaxis.set_ticklabels(xlabels, fontsize=12, rotation=-90)
    if ylabels is not None:
        ax.yaxis.set_ticklabels(ylabels, fontsize=12)

    # colour bar
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colours),
        ax=ax,
        shrink=1,
        pad=0.1,
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    )
    cbar.ax.set_yticklabels(["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

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
        vertical_lines=False,
        filename: str = None,
        labelX: str = "Time [ns]",
        labelY: str = "Population"
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

    # create the plot
    fig, axs = plt.subplots(1, 1, figsize=[8, 5])
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

    # plot vertical lines
    if vertical_lines and len(sequence) > 0:
        gate_steps = [exp.partial_propagators[g].shape[0] for g in sequence]
        for i in range(1, len(gate_steps)):
            gate_steps[i] += gate_steps[i - 1]
        gate_times = gate_steps * dt
        plt.vlines(
            x=gate_times / 1e-9,
            ymin=tf.reduce_min(population),
            ymax=tf.reduce_max(population),
            linestyles=":",
            colors="black",
        )

    # show and save
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotObservable(
        exp: Experiment,
        values: np.array,
        sequence: List[str],
        name: str,
        vertical_lines=False,
        filename: str = None,
):
    """
    Plots time dependent values of an observable. They need to be calculated with `runTimeEvolution` first.
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
    ts = np.linspace(0.0, dt * values.shape[0], values.shape[0])

    # create the plot
    fig, axs = plt.subplots(1, 1, figsize=[8, 5])
    axs.plot(ts / 1e-9, values)

    # set plot properties
    axs.tick_params(direction="in", left=True, right=True, top=False, bottom=True)
    axs.set_xlabel("Time [ns]")
    axs.set_ylabel(name)
    plt.tight_layout()

    # plot vertical lines
    if vertical_lines and len(sequence) > 0:
        gate_steps = [exp.partial_propagators[g].shape[0] for g in sequence]
        for i in range(1, len(gate_steps)):
            gate_steps[i] += gate_steps[i - 1]
        gate_times = gate_steps * dt
        plt.vlines(
            x=gate_times / 1e-9,
            ymin=tf.reduce_min(values),
            ymax=tf.reduce_max(values),
            linestyles=":",
            colors="black",
        )

    # show and save
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
        plt.close()
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
    plt.show()
    plt.close()
