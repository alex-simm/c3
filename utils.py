import os
import tempfile
from typing import List, Dict, Callable, cast, Tuple

import numpy as np
import tensorflow as tf

import c3.generator.devices as devices
import c3.libraries.chip as chip
import c3.libraries.envelopes as envelopes
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.tasks as tasks
import c3.signal.pulse as pulse
from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.generator.generator import Generator
from c3.libraries import algorithms
from c3.model import Model
from c3.optimizers.optimalcontrol import OptimalControl
from c3.signal.gates import Instruction

"""
TODO:
- is sideband necessary? (createCarrier, createGaussianEnvelope, createSingleQubitGate)
- generalise createState to more than one qubit
"""


# ========================== general utilities ==========================
def scaled_quantity(value: float, delta: float, unit: str) -> Quantity:
    """
    Creates a quantity between (1+delta)*value and (1-delta)*value.
    """
    if value == 0:
        upper = lower = 0.0
    else:
        upper = (1.0 + delta) * value
        lower = (1.0 - delta) * value
    return Quantity(
        value=value, min_val=min(upper, lower), max_val=max(upper, lower), unit=unit
    )


def filterValues(data: Dict, valueType) -> List:
    """
    Returns a list of all values in the dict of a specific type.
    """
    return list(filter(lambda q: type(q) == valueType, data.values()))


def getDrive(model: Model, qubit: chip.Qubit) -> chip.Drive:
    """
    Returns the drive line that is connected to the qubit, or None if no drive is connected to it.
    """
    drives = [
        cast(chip.Drive, d) for d in model.couplings.values() if type(d) == chip.Drive
    ]
    connected = list(filter(lambda d: qubit.name in d.connected, drives))
    return connected[0] if len(connected) > 0 else None


# ====================== creating models and experiments ======================
def createQubit(
    index: int, levels: int, frequency: float, anharmonicity: float
) -> chip.Qubit:
    """
    Creates a single qubit.

    Parameters
    ----------
    index : int
        Index of the qubit which is only used for its name.
    levels : int
        Number of energy levels
    frequency : float
        resonance frequency
    anharmonicity : float
        anharmonicity of the duffing oscillator
    """
    t1 = 27e-6
    t2star = 39e-6
    temp = 50e-3
    return chip.Qubit(
        name=f"Q{index}",
        desc=f"Qubit {index}",
        freq=scaled_quantity(frequency, 0.001, "Hz 2pi"),
        anhar=scaled_quantity(anharmonicity, 0.5, "Hz 2pi"),
        hilbert_dim=levels,
        t1=Quantity(value=t1, min_val=1e-6, max_val=90e-6, unit="s"),
        t2star=Quantity(value=t2star, min_val=10e-6, max_val=90e-3, unit="s"),
        temp=Quantity(value=temp, min_val=0.0, max_val=0.12, unit="K"),
    )


def createCoupling(q1: chip.Qubit, q2: chip.Qubit, strength: float) -> chip.Coupling:
    """
    Creates a coupling between two qubits.
    """
    return chip.Coupling(
        name=f"{q1.name}-{q2.name}",
        desc="coupling",
        comment=f"Coupling qubit {q1.name} to qubit {q2.name}",
        connected=[q1.name, q2.name],
        strength=Quantity(
            value=strength, min_val=-1 * 1e3, max_val=200e6, unit="Hz 2pi"
        ),
        hamiltonian_func=hamiltonians.int_XX,
    )


def createDrive(index: int, qubit: chip.Qubit) -> chip.Drive:
    """
    Creates a drive line in x-direction for the qubit.

    Parameters
    ----------
    index : int
        Index of the drive or qubit, only used for the name.
    qubit : int
        The qubit to which this drive is connection.
    """
    return chip.Drive(
        name=f"d{index + 1}",
        desc=f"Drive {index + 1}",
        comment=f"Drive line {index + 1} on qubit {qubit.name}",
        connected=[qubit.name],
        hamiltonian_func=hamiltonians.x_drive,
    )


def createModel(
    qubits: List[chip.Qubit], couplings: List[chip.Coupling] = None
) -> Model:
    """
    Creates a model for a set of qubits and couplings. This function creates a drive line for each qubit and additional
    tasks (initialise ground state).
    """
    # create a drive line for each qubit
    line_components: List[chip.LineComponent] = [
        createDrive(idx, qubit) for idx, qubit in enumerate(qubits)
    ]

    if couplings is not None:
        line_components.extend(couplings)

    # additional tasks for the model
    init_ground = tasks.InitialiseGround(
        init_temp=Quantity(value=50e-3, min_val=-0.001, max_val=0.22, unit="K")
    )

    # create the model
    model = Model(qubits, line_components, [init_ground])
    model.set_lindbladian(False)
    model.set_dressed(True)
    model.set_FR(False)
    return model


def createGenerator(model: Model) -> Generator:
    """
    Creates a generator with a simple device chain for each drive line in the model.
    """
    sim_res = 100e9
    awg_res = 2e9

    # create chain for each drive line
    drives = [d for d in model.couplings.values() if type(d) == chip.Drive]
    chain = ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"]
    chains = {f"{d.name}": chain for d in drives}

    # enable drag pulse for the AWG
    awg = devices.AWG(name="awg", resolution=awg_res, outputs=1)
    awg.enable_drag_2()

    return Generator(
        devices={
            "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": awg,
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Response": devices.Response(
                name="resp",
                rise_time=Quantity(
                    value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"
                ),
                resolution=sim_res,
                inputs=1,
                outputs=1,
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="v_to_hz",
                V_to_Hz=Quantity(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chains=chains,
    )


def createCarrier(frequency: float) -> pulse.Carrier:
    """
    Creates a carrier signal with optimisable frequency and framechange.
    """
    sideband = 50e6
    return pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params={
            "freq": scaled_quantity(frequency + sideband, 0.2, "Hz 2pi"),
            "framechange": Quantity(
                value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"
            ),
        },
    )


def createGaussianPulse(t_final: float) -> pulse.Envelope:
    """
    Creates a Gaussian pulse that can be used as envelope for the carrier frequency on a single drive line.
    """
    sideband = 50e6
    gauss_params_single = {
        "amp": Quantity(value=0.5, min_val=0.2, max_val=0.6, unit="V"),
        "t_final": scaled_quantity(t_final, 0.5, "s"),
        "sigma": Quantity(
            value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"
        ),
        "xy_angle": Quantity(
            value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Quantity(
            value=-sideband - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"
        ),
        "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
    }

    return pulse.Envelope(
        name="gauss",
        desc="Gaussian envelope",
        params=gauss_params_single,
        shape=envelopes.gaussian_nonorm,
    )


def createNoDrivePulse(t_final: float) -> pulse.Envelope:
    """
    Creates an empty envelope, i.e. a pulse that does nothing.
    """
    return pulse.Envelope(
        name="no_drive",
        params={"t_final": scaled_quantity(t_final, 0.1, "s")},
        shape=envelopes.no_drive,
    )


def createSingleQubitGate(
    name: str,
    t_final: float,
    frequency: float,
    envelope: pulse.Envelope,
    model: Model,
    qubit: chip.Qubit,
    ideal: np.array,
) -> Instruction:
    """
    Creates a gate that is realised by acting with the specified envelope on one qubit.

    Parameters
    ----------
    name: str
        Name of this gate
    t_final:
        Time after which the pulse of this gate should be over
    frequency:
        Carrier frequency
    envelope:
        Shape of the envelope
    model:
        Full model of the system.
    qubit:
        The qubit on which this gate should act.
    ideal:
        Matrix representation of the ideal gate.
    """
    sideband = 50e6

    # find the drive line for the qubit
    drives = filterValues(model.couplings, chip.Drive)
    drive_names = list(map(lambda d: d.name, drives))
    active_drive = list(filter(lambda d: qubit.name in d.connected, drives))[0]

    gate = Instruction(
        name=name,
        targets=[qubit.index],
        t_start=0.0,
        t_end=t_final,
        channels=drive_names,
        ideal=ideal,
    )

    # add carrier and envelope for each drive line
    for drive in drive_names:
        carrier = createCarrier(frequency)
        if drive == active_drive.name:
            gate.add_component(envelope, drive)
            gate.add_component(carrier, drive)
        else:
            gate.add_component(createNoDrivePulse(t_final), drive)
            gate.add_component(carrier, drive)
            gate.comps[drive]["carrier"].params["framechange"].set_value(
                (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
            )

    return gate


def createState(model: Model, occupied_basis_states: List[int]) -> tf.Tensor:
    """
    Returns a state as a tensor in which specific basis states are occupied.
    """
    psi_init = np.array([[0] * model.tot_dim], dtype=float)
    psi_init[0][occupied_basis_states] = 1
    psi_init /= np.linalg.norm(psi_init)
    return tf.transpose(tf.constant(psi_init, tf.complex128))


# ========================== running and optimising experiments ==========================
def runTimeEvolution(
    experiment: Experiment,
    psi_init: tf.Tensor,
    gate_sequence: List[str],
    population_function: Callable,
) -> np.array:
    """
    Runs the time evolution of the experiment and returns the populations.

    Parameters
    ----------
    experiment:
        The experiment containing the model and propagators
    psi_init:
        Initial state
    gate_sequence:
        List of gate names that will be applied to the state
    population_function:
        A function that is called in each time step to calculate the population from the state

    Returns
    -------
    np.array
        The population vector for each time step.
    """
    dUs = experiment.partial_propagators
    psi_t = psi_init.numpy()

    # run the time evolution for each gate
    pop_t = population_function(psi_t)
    for gate in gate_sequence:
        for du in dUs[gate]:
            psi_t = np.matmul(du.numpy(), psi_t)
            pops = population_function(psi_t)
            pop_t = np.append(pop_t, pops, axis=1)
    return pop_t


def runTimeEvolutionDefault(
    experiment: Experiment, psi_init: tf.Tensor, gate_sequence: List[str]
) -> np.array:
    """
    Runs the time evolution of the experiment and returns the populations. Same as `runTimeEvolution`. Uses
    the absolute square of the wave function entries as population.

    See Also
        runTimeEvolution
    """
    model = experiment.pmap.model
    return runTimeEvolution(
        experiment,
        psi_init,
        gate_sequence,
        lambda psi_t: experiment.populations(psi_t, model.lindbladian),
    )


def optimise(
    experiment: Experiment,
    gates: List[Instruction],
    optimisable_parameters: List,
    fidelity_fctn: Callable,
    fidelity_params: Dict,
    callback: Callable = None,
    log_dir: str = None,
) -> Tuple:
    """
    Optimises gates in an experiment with respect to a given fidelity function.

    Parameters
    ----------
    experiment:
        The experiment containing the model
    gates:
        A list of gates that should be optimised
    optimisable_parameters:
        A list of parameters that can be optimised
    fidelity_fctn:
        The infidelity function that is used for optimisation
    fidelity_params:
        Additional parameters for the fidelity function
    callback:
        A callback that is called after each optimisation step
    log_dir:
        Directory to which the log files will be written. Defaults to the system's temp directory.

    Returns
    -------
    The parameters before and after optimisation and the final infidelity
    """
    # find drive lines corresponding to the qubits
    model = experiment.pmap.model
    qubits = filterValues(model.subsystems, chip.Qubit)
    qubit_names = [q.name for q in qubits]
    experiment.set_opt_gates([g.get_key() for g in gates])
    experiment.pmap.set_opt_map(optimisable_parameters)

    # create optimiser
    if log_dir is None:
        log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")
    opt = OptimalControl(
        dir_path=log_dir,
        fid_func=fidelity_fctn,
        fid_func_kwargs=fidelity_params,
        fid_subspace=qubit_names,
        pmap=experiment.pmap,
        algorithm=algorithms.lbfgs,
        options={"maxfun": 150},
    )
    opt.set_exp(experiment)
    if callback:
        opt.set_callback(callback)

    # start optimisation
    params_before = experiment.pmap.str_parameters()
    opt.optimize_controls()
    result = opt.current_best_goal
    params_after = experiment.pmap.str_parameters()

    return params_before, params_after, result
