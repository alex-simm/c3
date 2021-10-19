from typing import Callable, Dict
import numpy as np
import utils as utils
import tensorflow as tf
import c3.signal.pulse as pulse
import c3.libraries.chip as chip
from c3.parametermap import ParameterMap
from c3.experiment import Experiment
from c3.signal.gates import Instruction
from DataOutput import DataOutput


class SingleQubitExperiment(DataOutput):
    __directory: str
    __file_prefix: str
    __file_suffix: str
    __qubit: chip.Qubit
    __experiment: Experiment
    __gate: Instruction
    __t_final: float
    __init_state: tf.Tensor

    def __init__(
        self, directory: str, file_prefix: str = None, file_suffix: str = None
    ):
        super().__init__(directory, file_prefix, file_suffix)

    def prepare(
        self,
        t_final: float,
        freq: float,
        anharm: float,
        carrier_freq: float,
        envelope: pulse.Envelope,
        ideal_gate: np.array,
        useDrag=True,
    ):
        self.__t_final = t_final
        occupied_levels = [0, 2]

        # model
        self.__qubit = utils.createQubit(1, 5, freq, -anharm)
        model = utils.createModel([self.__qubit])
        generator = utils.createGenerator(model, useDrag)

        # gate
        utils.plotMatrix(
            ideal_gate,
            self.__createFileName("ideal_propagator_a", "png"),
            self.__createFileName("ideal_propagator_b", "png"),
        )
        gate = utils.createSingleQubitGate(
            "gate", t_final, carrier_freq, envelope, model, self.__qubit, ideal_gate
        )
        self.__gates = [gate]

        # experiment
        gate_names = list(map(lambda g: g.get_key(), self.__gates))
        self.__experiment = Experiment(
            pmap=ParameterMap(
                instructions=self.__gates, model=model, generator=generator
            )
        )
        self.__experiment.set_opt_gates(gate_names)
        self.__experiment.compute_propagators()

        # initial state
        self.__init_state = utils.createState(model, occupied_levels)

    def getInitialState(self):
        return self.__init_state

    def getEnergies(self):
        return self.__qubit.get_Hamiltonian().numpy().diagonal().real

    def getQubit(self) -> chip.Qubit:
        return self.__qubit

    def getExperiment(self) -> Experiment:
        return self.__experiment

    def generateSignal(self) -> Dict[str, tf.Tensor]:
        """
        Makes the generator generate a signal and returns it.
        """
        return self.__experiment.pmap.generator.generate_signals(self.__gate)[
            utils.getDrive(self.__experiment.pmap.model, self.__qubit).name
        ]

    def saveSignal(self, name: str) -> None:
        signal = self.generateSignal()
        np.save(
            self.__createFileName(name, ""),
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
            self.__createFileName(name, "png"),
            spectrum_cut=1e-4,
        )

    def runTimeEvolution(self) -> np.array:
        gate_names = list(map(lambda g: g.get_key(), self.__gates))
        return utils.runTimeEvolutionDefault(
            self.__experiment, self.__init_state, gate_names
        )

    def saveTimeEvolution(self, name: str) -> None:
        populations = self.runTimeEvolution()
        np.save(self.__createFileName(name, ""), populations)

    def plotTimeEvolution(self, name: str) -> None:
        populations = self.runTimeEvolution()
        gate_names = list(map(lambda g: g.get_key(), self.__gates))
        utils.plotOccupations(
            self.__experiment,
            populations,
            gate_names,
            filename=self.__createFileName(name, "png"),
            level_names=[
                "$|0,0\\rangle$",
                "$|0,1\\rangle$",
                "$|1,0\\rangle$",
                "$|1,1\\rangle$",
                "leakage",
            ],
        )

    def savePropagator(self, name: str) -> None:
        for gate in self.__gates:
            U = self.__experiment.propagators[gate.get_key()]
            np.save(self.__createFileName(name + "-" + gate.get_key(), ""), U)

    def plotPropagator(self, name: str) -> None:
        for gate in self.__gates:
            U = self.__experiment.propagators[gate.get_key()]
            utils.plotMatrix(
                U,
                self.__createFileName(name + "-" + gate.get_key() + "_a", "png"),
                self.__createFileName(name + "-" + gate.get_key() + "_b", "png"),
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
        optimisable_gates = list(filter(lambda g: g.get_key() != "id[]", self.__gates))
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
            log_dir="./" + self.__createFileName("log") + "/",
        )
        print("before:\n", params_before)
        print("after:\n", params_after)
        print("fidelity:\n", final_fidelity)
        return params_after
