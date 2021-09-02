import pathlib
from typing import Callable
import numpy as np
import utils as utils
import tensorflow as tf
import c3.signal.pulse as pulse
import c3.utils.qt_utils as qt_utils
import c3.libraries.constants as constants
import c3.libraries.fidelities as fidelities
import c3.libraries.chip as chip
from c3.parametermap import ParameterMap
from c3.experiment import Experiment
from c3.signal.gates import Instruction


class SingleQubitExperiment:
    __directory: str
    __file_suffix: str
    __qubit: chip.Qubit
    __experiment: Experiment
    __gate: Instruction
    __t_final: float
    __init_state: tf.Tensor

    def __init__(self, file_suffix: str, directory: str = None):
        self.__file_suffix = file_suffix

        if directory is None:
            directory = "./output"
        output_dir = pathlib.Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.__directory = directory

    def prepare(
        self,
        t_final: float,
        freq: float,
        anharm: float,
        carrier_freq: float,
        envelope: pulse.Envelope,
    ):
        self.__t_final = t_final
        occupied_levels = [0, 2]

        # model
        self.__qubit = utils.createQubit(1, 5, freq, -anharm)
        model = utils.createModel([self.__qubit])
        generator = utils.createGenerator(model)
        energies = self.__qubit.get_Hamiltonian().numpy().diagonal().real
        print(
            "energies: ",
            [
                (energies[i + 1] - energies[i]) / (2 * np.pi)
                for i in range(len(energies) - 1)
            ],
        )

        # gate
        # envelope = createPWCGaussianPulse(t_final, t_final / 4, 30)
        ideal = qt_utils.np_kron_n(
            [
                constants.Id,
                constants.x90p,
            ]
        )
        self.__gate = utils.createSingleQubitGate(
            "lower-X", t_final, carrier_freq, envelope, model, self.__qubit, ideal
        )
        gates = [self.__gate]

        # experiment
        gate_names = list(map(lambda g: g.get_key(), gates))
        self.__experiment = Experiment(
            pmap=ParameterMap(instructions=gates, model=model, generator=generator)
        )
        self.__experiment.set_opt_gates(gate_names)
        self.__experiment.compute_propagators()

        # initial state
        self.__init_state = utils.createState(model, occupied_levels)
        # state = init_state.numpy().flatten()
        # print("initial state=", state, ", occupation=", exp.populations(state, model.lindbladian).numpy())

        # time evolution and signal before optimisation
        sequence = ["lower-X[0]"]
        signal = generator.generate_signals(self.__gate)[
            utils.getDrive(model, self.__qubit).name
        ]
        utils.plotSignal(
            signal["ts"].numpy(),
            signal["values"].numpy(),
            self.__directory + f"/signal_before_{self.__file_suffix}.png",
            spectrum_cut=1e-4,
        )
        peakFrequencies, peakValues = utils.findFrequencyPeaks(
            signal["ts"].numpy(), signal["values"].numpy(), 4
        )
        print("peaks: ", np.sort(peakFrequencies))
        populations = utils.runTimeEvolutionDefault(
            self.__experiment, self.__init_state, sequence
        )
        utils.plotOccupations(
            self.__experiment,
            populations,
            sequence,
            filename=self.__directory + f"/populations_before_{self.__file_suffix}.png",
            level_names=[
                "$|0,0\\rangle$",
                "$|0,1\\rangle$",
                "$|1,0\\rangle$",
                "$|1,1\\rangle$",
                "leakage",
            ],
        )

    def optimise(
        self, optimisable_params: dict, algorithm: Callable, algorithm_options: dict
    ):
        # optimise
        optimisable_gates = list(filter(lambda g: g.get_key() != "id[]", [self.__gate]))
        gateset_opt_map = utils.createOptimisableParameterMap(
            self.__experiment, optimisable_gates, optimisable_params
        )
        # callback = lambda fidelity: print(fidelity)
        params_before, final_fidelity, params_after = utils.optimise(
            self.__experiment,
            optimisable_gates,
            optimisable_parameters=gateset_opt_map,
            fidelity_fctn=fidelities.unitary_infid_set,
            # fidelity_fctn=test_fidelity,
            fidelity_params={
                # 'psi_0': init_state[:active_levels],
                "active_levels": 4,
                # 'num_gates': len(gates)
                # "qubit": q1,
                # "generator": generator,
                # "drive": getDrive(model, q1)
            },
            # callback=callback,
            algorithm=algorithm,
            algo_options=algorithm_options,
            log_dir=(
                self.__directory
                + ("/log_{0}_{1:.2f}/".format(self.__file_suffix, self.__t_final * 1e9))
            ),
        )
        print("before:\n", params_before)
        print("after:\n", params_after)
        print("fidelity:\n", final_fidelity)

        # time evolution and signal after optimisation
        sequence = [self.__gate.get_key()]
        pmap = self.__experiment.pmap
        signal = pmap.generator.generate_signals(self.__gate)[
            utils.getDrive(pmap.model, self.__qubit).name
        ]
        utils.plotSignal(
            signal["ts"].numpy(),
            signal["values"].numpy(),
            self.__directory + f"/signal_after_{self.__file_suffix}.png",
            spectrum_cut=1e-4,
        )
        peakFrequencies, peakValues = utils.findFrequencyPeaks(
            signal["ts"].numpy(), signal["values"].numpy(), 4
        )
        print("peaks: ", np.sort(peakFrequencies))
        populations = utils.runTimeEvolutionDefault(
            self.__experiment, self.__init_state, sequence
        )
        utils.plotOccupations(
            self.__experiment,
            populations,
            sequence,
            filename=self.__directory + f"/populations_after_{self.__file_suffix}.png",
            level_names=[
                "$|0,0\\rangle$",
                "$|0,1\\rangle$",
                "$|1,0\\rangle$",
                "$|1,1\\rangle$",
                "leakage",
            ],
        )
        return params_after
