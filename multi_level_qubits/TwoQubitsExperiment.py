from typing import Callable, Dict, List, Tuple
import numpy as np
import utils as utils
import tensorflow as tf
import c3.signal.pulse as pulse
import c3.libraries.chip as chip
from c3.parametermap import ParameterMap
from c3.experiment import Experiment
from c3.signal.gates import Instruction
from multi_level_qubits.DataOutput import DataOutput


class TwoQubitsExperiment(DataOutput):
    __qubit1: chip.Qubit
    __qubit2: chip.Qubit
    __experiment: Experiment
    __gate: Instruction
    __t_final: float
    __init_state: tf.Tensor

    def __init__(
        self, directory: str, file_prefix: str = None, file_suffix: str = None
    ):
        super(TwoQubitsExperiment, self).__init__(directory, file_prefix, file_suffix)

    def prepare(
        self,
        t_final: float,
        freqs: Tuple[float, float],
        anharms: Tuple[float, float],
        carrier_freq: float,
        envelope: pulse.Envelope,
        ideal_gate: np.array,
        occupied_levels: List[int],
        useDrag=True,
    ):
        self.__t_final = t_final

        # model
        self.__qubit1 = utils.createQubit(1, 5, freqs[0], -anharms[0])
        self.__qubit2 = utils.createQubit(2, 5, freqs[1], -anharms[1])
        model = utils.createModel([self.__qubit1, self.__qubit2])
        generator = utils.createGenerator(model, useDrag)

        # gate
        self.__gate = utils.createSingleQubitGate(
            "gate", t_final, carrier_freq, envelope, model, self.__qubit1, ideal_gate
        )

        # experiment
        gate_names = list(map(lambda g: g.get_key(), [self.__gate]))
        self.__experiment = Experiment(
            pmap=ParameterMap(
                instructions=[self.__gate], model=model, generator=generator
            )
        )
        self.__experiment.set_opt_gates(gate_names)
        self.__experiment.compute_propagators()

        # initial state
        self.__init_state = utils.createState(model, occupied_levels)

    def getInitialState(self) -> tf.Tensor:
        return self.__init_state

    def getEnergies(self) -> np.array:
        return self.__experiment.pmap.model.get_Hamiltonian().numpy().diagonal().real

    def getQubits(self) -> Tuple[chip.Qubit, chip.Qubit]:
        return self.__qubit1, self.__qubit2

    def getExperiment(self) -> Experiment:
        return self.__experiment

    def generateSignal(self) -> Dict[str, tf.Tensor]:
        """
        Makes the generator generate a signal and returns it.
        """
        return self.__experiment.pmap.generator.generate_signals(self.__gate)[
            utils.getDrive(self.__experiment.pmap.model, self.__qubit1).name
        ]

    def saveSignal(self, name: str) -> None:
        signal = self.generateSignal()
        np.save(
            super().createFileName(name, ""),
            [signal["ts"].numpy(), signal["values"].numpy()],
        )

    def plotSignal(self, name: str) -> None:
        """
        Generates a signal, plots it, and saves it to a file with the given name in the experiment's directory.
        """
        signal = self.generateSignal()
        utils.plotSignal(
            signal["ts"].numpy(),
            signal["values"].numpy(),
            super().createFileName(name, "png"),
            spectrum_cut=1e-4,
        )

    def runTimeEvolution(self) -> np.array:
        gate_names = list(map(lambda g: g.get_key(), [self.__gate]))
        return utils.runTimeEvolutionDefault(
            self.__experiment, self.__init_state, gate_names
        )

    def saveTimeEvolution(self, name: str) -> None:
        populations = self.runTimeEvolution()
        np.save(super().createFileName(name, ""), populations)

    def plotTimeEvolution(self, name: str) -> None:
        populations = self.runTimeEvolution()
        gate_names = list(map(lambda g: g.get_key(), [self.__gate]))
        utils.plotOccupations(
            self.__experiment,
            populations,
            gate_names,
            filename=super().createFileName(name, "png"),
            # level_names=[
            #    "$|0,0\\rangle$",
            #    "$|0,1\\rangle$",
            #    "$|1,0\\rangle$",
            #    "$|1,1\\rangle$",
            #    "leakage",
            # ],
        )

    def savePropagator(self, name: str) -> None:
        U = self.__experiment.propagators[self.__gate.get_key()]
        np.save(super().createFileName(name, ""), U)

    def plotPropagator(self, name: str) -> None:
        U = self.__experiment.propagators[self.__gate.get_key()]
        utils.plotMatrix(
            U,
            super().createFileName(name, "png"),
            super().createFileName(name + "_phase", "png"),
            super().createFileName(name + "_abs", "png"),
        )

    def saveIdealPropagator(self, name) -> None:
        utils.plotMatrix(
            self.__gate.ideal,
            super().createFileName(name, "png"),
            super().createFileName(name + "_phases", "png"),
        )

    def optimise(
        self,
        optimisable_params: dict,
        algorithm: Callable,
        algorithm_options: dict,
        fidelity_function: Callable,
        fidelity_params: dict,
        callback: Callable = None,
    ):
        # optimise
        optimisable_gates = list(filter(lambda g: g.get_key() != "id[]", [self.__gate]))
        gateset_opt_map = utils.createOptimisableParameterMap(
            self.__experiment, optimisable_gates, optimisable_params
        )
        params_before, final_fidelity, params_after = utils.optimise(
            self.__experiment,
            optimisable_gates,
            optimisable_parameters=gateset_opt_map,
            fidelity_fctn=fidelity_function,
            fidelity_params=fidelity_params,
            callback=callback,
            algorithm=algorithm,
            algo_options=algorithm_options,
            log_dir="./" + super().createFileName("log") + "/",
        )
        print("before:\n", params_before)
        print("after:\n", params_after)
        print("fidelity:\n", final_fidelity)
        return params_after
