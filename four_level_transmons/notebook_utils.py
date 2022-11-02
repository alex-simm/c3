import os
from typing import List, Tuple
import scipy.linalg

from c3.experiment import Experiment
from c3.libraries import chip, fidelities
from c3.optimizers.optimalcontrol import OptimalControl
from c3.signal import gates
from four_level_transmons.DataOutput import DataOutput
from four_level_transmons.plotting import *
from four_level_transmons.utilities import *


def printSignal(exper: Experiment, qubits: List[chip.Qubit],
                gate: gates.Instruction, output: DataOutput,
                states: List[Tuple[float, str]] = None):
    """
    Plots the drive signal and its spectrum for each qubit in the list separately.

    Parameters
    ----------
    exper
        a complete Experiment object
    qubits
        a list of qubits from the experiment for which to plot the signals
    gate
        the instruction for which to plot the signals
    output
        the target output
    """
    signals = exper.pmap.generator.generate_signals(gate)
    for i, qubit in enumerate(qubits):
        # generate signal
        drive = getDrive(exper.pmap.model, qubit).name
        signal = signals[drive]
        ts = signal["ts"].numpy()
        values = signal["values"].numpy()

        # save data
        peakFrequencies, peakValues = findFrequencyPeaks(ts, values, 10, normalise=False)
        sortIndices = np.argsort(peakValues)
        print(f"peaks: {drive}")
        for idx in sortIndices:
            print(f"\t{peakFrequencies[idx]:e} (amp={peakValues[idx]:e})")
        output.save([ts, values], f"signal_t{i + 1}")

        # plot
        plotSignalAndSpectrum(ts, real=values, filename=output.createFileName(f"signal_t{i + 1}", "svg"),
                              spectralThreshold=None)
        plotSignalAndSpectrum(ts, real=values, filename=output.createFileName(f"signal_detail_t{i + 1}", "svg"),
                              spectralThreshold=1e-4)
        plotSignalAndSpectrum(ts, real=values, filename=output.createFileName(f"signal_detail_states_t{i + 1}", "svg"),
                              spectralThreshold=1e-4, states=states)


def printTimeEvolution(exper: Experiment, init: tf.Tensor, gate: gates.Instruction,
                       labels: List[str], output: DataOutput):
    """
    Plots the time evolution of a specific initial state under the effect of a gate.
    """
    populations = calculatePopulation(exper, init, [gate.get_key()])
    output.save(populations, "population")
    plotPopulation(exper, populations, sequence=[gate.get_key()],
                   labels=labels, filename=output.createFileName("population", "svg"))
    if len(exper.pmap.model.subsystems) > 1:
        plotSplittedPopulation(exper, populations, [gate.get_key()], filename=output.createFileName("population", "svg"))


def printMatrix(M: np.array, labels: List[str], name: str, output: DataOutput):
    """
    Plots a complex matrix as a Hinton diagram.
    """
    # plotComplexMatrix(M, xlabels=labels, ylabels=labels, filename=output.createFileName(name))
    plotComplexMatrixHinton(M, maxAbsolute=1, xlabels=labels, ylabels=labels, gridColour="gray",
                            filename=output.createFileName(name, "svg"), colourMap='hsv')
    # plotComplexMatrixAbsOrPhase(M, xlabels=labels, ylabels=labels, phase=True,
    #                            filename=output.createFileName(name + "_phase"))
    # plotComplexMatrixAbsOrPhase(M, xlabels=labels, ylabels=labels, phase=False,
    #                            filename=output.createFileName(name + "_abs"))


def printPropagator(exper: Experiment, gate: gates.Instruction,
                    labels: List[str], output: DataOutput,
                    savePartials=False):
    """
    Saves the propagator of a gate to a file and plots it as a Hinton diagram.
    """
    U = exper.propagators[gate.get_key()]
    output.save(U, "propagator")
    printMatrix(U, labels, "propagator", output)
    if savePartials:
        output.save(exper.partial_propagators[gate.get_key()], "partial_propagators")


def optimise(output: DataOutput, qubits: List[chip.PhysicalComponent],
             exp: Experiment, algorithm, options, gate: gates.Instruction, goalFunction=None) -> List[float]:
    """
    Runs the optimisation of an experiment using a specific algorithm.
    """
    # set up the optimiser
    if goalFunction is None:
        goalFunction = fidelities.unitary_infid_set
    opt = OptimalControl(
        dir_path=output.getDirectory(),
        fid_func=goalFunction,
        fid_subspace=[q.name for q in qubits],
        pmap=exp.pmap,
        algorithm=algorithm,
        options=options,
        run_name=gate.name,
        fid_func_kwargs={
            "active_levels": 4
        }
    )
    exp.set_opt_gates([gate.get_key()])
    opt.set_exp(exp)

    # add the callback
    infidelities = []

    def fidelityCallback(index, fidelity, previousFidelity):
        print(index, fidelity, f"{previousFidelity - fidelity:e}")
        infidelities.append(fidelity)

    opt.set_callback(fidelityCallback)

    # run optimisation
    opt.optimize_controls()
    print(opt.current_best_goal)
    exp.pmap.print_parameters()

    return infidelities


def printAllSignals(exper: Experiment, qubit: chip.Qubit, output: DataOutput, directory: str):
    """
    Debugging function: prints the generated signal after each device in the chain separately.
    """
    try:
        os.mkdir(output.getDirectory() + "/" + directory)
    except:
        pass
    drive = getDrive(exper.pmap.model, qubit)
    outputs = exper.pmap.generator.global_signal_stack[drive.name]

    for name, values in outputs.items():
        filename = output.createFileName(directory + f"/device_{drive.name}_{name}", "svg")
        time = values["ts"].numpy()
        if "values" in values:
            signal = values["values"].numpy()
            print(signal)
            plotSignalAndSpectrum(time, signal, min_signal_limit=None, filename=filename, spectralThreshold=None)
        else:
            print(values["inphase"].numpy())
            print(values["quadrature"].numpy())
            plotSignalAndSpectrum(time, real=values["inphase"].numpy(), imag=values["quadrature"].numpy(),
                                  min_signal_limit=None, spectralThreshold=None, filename=filename)


def getEnergiesFromModel(experiment: Experiment, gate: gates.Instruction, labels: List[str]) -> List[Tuple[float, str]]:
    """
    Returns a list of all energies of the model, including a Stark shift, combined with the corresponding state labels.
    The list of labels is expected to be sorted by increasing energy.
    """
    # obtain eigenvalues from full (Stark-shifted) Hamiltonian
    signal = experiment.pmap.generator.generate_signals(gate)
    H = experiment.pmap.model.get_Hamiltonian(signal)
    evals,evecs = scipy.linalg.eig(H)
    evals = evals.real / (2 * np.pi)

    # combine eigenvalues with labels
    indices = [np.argmax(np.round(evecs[i], 2)) for i in range(len(evals))]
    stateEnergies = []
    for i, x in enumerate(labels):
        if x is not None:
            energy = evals[indices[i]]
            stateEnergies.append((energy, x))
    return stateEnergies


def getEnergiesFromFile(filename: str, labels: List[str]) -> List[Tuple[float, str]]:
    """
    Reads a list of all energies of the model, including a Stark shift by the drive, combined with the corresponding
    state labels from a file.
    """
    energies = np.load(filename)
    return list(zip(energies, labels))


def calculateTransitions(energies: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """
    Calculates and returns all transitions (resonances) between all pairs of energies from the list, combined with a
    corresponding label.
    """
    items = sorted(energies, key=lambda x: x[0])
    transitions = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j:
                E = items[j][0] - items[i][0]
                if E > 0:
                    transitions.append((E, items[i][1] + " - " + items[j][1]))
    return transitions
