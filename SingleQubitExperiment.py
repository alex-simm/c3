import pathlib
from typing import Callable, Dict, List
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
    __file_prefix: str
    __file_suffix: str
    __qubit: chip.Qubit
    __experiment: Experiment
    __gate: Instruction
    __t_final: float
    __init_state: tf.Tensor
    __sequence: List[str]

    def __init__(
        self, file_prefix: str = None, file_suffix: str = None, directory: str = None
    ):
        self.__file_prefix = file_prefix
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

        # gate
        ideal = qt_utils.np_kron_n([constants.Id, constants.x90p])
        self.__gate = utils.createSingleQubitGate(
            "lower-X", t_final, carrier_freq, envelope, model, self.__qubit, ideal
        )
        self.__sequence = [self.__gate.get_key()]

        # experiment
        gates = [self.__gate]
        gate_names = list(map(lambda g: g.get_key(), gates))
        self.__experiment = Experiment(
            pmap=ParameterMap(instructions=gates, model=model, generator=generator)
        )
        self.__experiment.set_opt_gates(gate_names)
        self.__experiment.compute_propagators()

        # initial state
        self.__init_state = utils.createState(model, occupied_levels)

    def getEnergies(self):
        return self.__qubit.get_Hamiltonian().numpy().diagonal().real

    def generateSignal(self) -> Dict[str, tf.Tensor]:
        """
        Makes the generator generate a signal and returns it.
        """
        return self.__experiment.pmap.generator.generate_signals(self.__gate)[
            utils.getDrive(self.__experiment.pmap.model, self.__qubit).name
        ]

    def plotSignal(self, name: str) -> None:
        """
        Generates a signal, plots it, and saves it to a file with the given name in the experiment's directory.
        """
        signal = self.generateSignal()
        utils.plotSignal(
            signal["ts"].numpy(),
            signal["values"].numpy(),
            self.__createFileName(name),
            spectrum_cut=1e-4,
        )

    def runTimeEvolution(self) -> np.array:
        return utils.runTimeEvolutionDefault(
            self.__experiment, self.__init_state, self.__sequence
        )

    def plotTimeEvolution(self, name: str) -> None:
        populations = self.runTimeEvolution()
        utils.plotOccupations(
            self.__experiment,
            populations,
            self.__sequence,
            filename=self.__createFileName(name),
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
            log_dir=self.__directory
            + ("/log_{0}_{1:.2f}/".format(self.__file_suffix, self.__t_final * 1e9)),
        )
        print("before:\n", params_before)
        print("after:\n", params_after)
        print("fidelity:\n", final_fidelity)
        return params_after

    def __createFileName(self, name):
        s = self.__directory + "/"
        if self.__file_prefix is not None and len(self.__file_prefix) > 0:
            s += self.__file_prefix + "_"
        s += name
        if self.__file_suffix is not None and len(self.__file_suffix) > 0:
            s += self.__file_suffix + "_"
        return s + ".png"
