import os
from typing import List, Tuple

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
    # output.save(exper.partial_propagators[gate.get_key()], "partial_propagators")
    # os.system('bzip2 -9 "' + output.createFileName('partial_propagators.npy') + '"')
    printMatrix(U, labels, "propagator", output)
    # Uprojected = tf_project_to_comp(U, dims=[5, 5], index=[0, 1], outdims=[4, 4])
    # printMatrix(Uprojected, labels, "propagator_projected", output)
    if savePartials:
        output.save(exper.partial_propagators[gate.get_key()], "partial_propagators")


def optimise(output: DataOutput, qubits: List[chip.PhysicalComponent],
             exp: Experiment, algorithm, options, gate: gates.Instruction) -> List[float]:
    """
    Runs the optimisation of an experiment using a specific algorithm.
    """
    # set up the optimiser
    opt = OptimalControl(
        dir_path=output.getDirectory(),
        fid_func=fidelities.unitary_infid_set,
        #fid_func=diagonal_infidelity_set,
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
        #if name.startswith("AWG"):
        #    re = values["inphase"].numpy()
        #    im = values["quadrature"].numpy()
        #    plotSignalAndSpectrum(time[:1000], real=re[:1000], min_signal_limit=None,
        #                          spectralThreshold=5e-4,
        #                          filename=output.createFileName(directory + f"/device_{drive.name}_{name}_real", "svg"))
        #    plotSignalAndSpectrum(time[:1000], real=im[:1000], min_signal_limit=None,
        #                          spectralThreshold=5e-4,
        #                          filename=output.createFileName(directory + f"/device_{drive.name}_{name}_imag", "svg"))

        print(name+":")
        if "values" in values:
            signal = values["values"].numpy()
            print(signal)
            plotSignalAndSpectrum(time, signal, min_signal_limit=None, filename=filename, spectralThreshold=None)
        else:
            print(values["inphase"].numpy())
            print(values["quadrature"].numpy())
            plotSignalAndSpectrum(time, real=values["inphase"].numpy(), imag=values["quadrature"].numpy(),
                                  min_signal_limit=None, spectralThreshold=None, filename=filename)
