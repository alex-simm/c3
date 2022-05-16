import itertools
from typing import List, Tuple, cast, Dict, Callable
import numpy as np
import tensorflow as tf
import copy
from c3.c3objs import Quantity as Qty
from c3.experiment import Experiment
from c3.generator.generator import Generator
import c3.libraries.chip as chip
import c3.generator.devices as devices
import c3.signal.pulse as pulse
import c3.signal.gates as gates
import c3.libraries.hamiltonians as hamiltonians
from scipy.signal import find_peaks

from c3.model import Model


def createQubits(
    qubit_levels_list: List[int],
    freq_list: List[float],
    anharm_list: List[float],
    t1_list: List[float],
    t2star_list: List[float],
    qubit_temp: float,
) -> List[chip.Qubit]:
    """
    Creates and returns a list of qubits.

    Parameters
    ----------
    qubit_levels_list: List[int]
        List of number of levels in each qubit.
    freq_list: List[Float]
        List of frequency of each qubit.
    anharm_list: List[Float]
        List of anharmonicity of each qubit.
    t1_list: List[Float]
        List of T1 values for each qubit.
    t2star_list: List[Float]
        List of T2* values for each qubit.
    qubit_temp: Float
        Temperature of the qubits

    Returns
    -------
    qubits: List[chip.Transmon]
        An list of transmons
    """

    qubits = []
    for i in range(len(qubit_levels_list)):
        qubits.append(
            chip.Qubit(
                name=f"Q{i + 1}",
                desc=f"Qubit {i + 1}",
                freq=Qty(
                    value=freq_list[i],
                    min_val=freq_list[i] - 5e6,
                    max_val=freq_list[i] + 5e6,
                    unit="Hz 2pi",
                ),
                anhar=Qty(
                    value=anharm_list[i], min_val=-380e6, max_val=-120e6, unit="Hz 2pi"
                ),
                hilbert_dim=qubit_levels_list[i],
                t1=Qty(value=t1_list[i], min_val=1e-6, max_val=90e-6, unit="s"),
                t2star=Qty(
                    value=t2star_list[i], min_val=10e-6, max_val=90e-6, unit="s"
                ),
                temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
            )
        )

    return qubits


def createTransmons(
    qubit_levels_list: List[int],
    freq_list: List[float],
    anharm_list: List[float],
    t1_list: List[float],
    t2star_list: List[float],
    phi_list: List[float],
    phi0_list: List[float],
    d_list: List[float],
    qubit_temp: float,
) -> List[chip.Transmon]:
    """
    Creates and returns a list of transmons.

    Parameters
    ----------
    qubit_levels_list: List[int]
        List of number of levels in each qubit.
    freq_list: List[Float]
        List of frequency of each qubit.
    anharm_list: List[Float]
        List of anharmonicity of each qubit.
    t1_list: List[Float]
        List of T1 values for each qubit.
    t2star_list: List[Float]
        List of T2* values for each qubit.
    phi_list: List[Float]
        List of current flux values for each qubit.
    phi0_list: List[Float]
        List of flux bias for each qubit.
    d_list: List[Float]
        List of junction asymmetry values for each qubit.
    qubit_temp: Float
        Temperature of the qubits

    Returns
    -------
    qubits: List[chip.Transmon]
        An list of transmons
    """

    qubits = []
    for i in range(len(qubit_levels_list)):
        qubits.append(
            chip.Transmon(
                name=f"Q{i + 1}",
                desc=f"Qubit {i + 1}",
                freq=Qty(
                    value=freq_list[i],
                    min_val=freq_list[i] - 5e6,
                    max_val=freq_list[i] + 5e6,
                    unit="Hz 2pi",
                ),
                anhar=Qty(
                    value=anharm_list[i], min_val=-380e6, max_val=-120e6, unit="Hz 2pi"
                ),
                hilbert_dim=qubit_levels_list[i],
                t1=Qty(value=t1_list[i], min_val=1e-6, max_val=90e-6, unit="s"),
                t2star=Qty(
                    value=t2star_list[i], min_val=10e-6, max_val=90e-6, unit="s"
                ),
                temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
                phi=Qty(value=phi_list[i], max_val=5.0, min_val=0.0, unit="Wb"),
                phi_0=Qty(value=phi0_list[i], max_val=11.0, min_val=9.0, unit="Wb"),
                d=Qty(value=d_list[i], max_val=0.1, min_val=-0.1, unit=""),
            )
        )

    return qubits


def createChainCouplings(
    coupling_strength: List[float],
    qubits: List[chip.Transmon],
) -> List[chip.Coupling]:
    """
    Creates nearest neighbour couplings between the qubits in 1D chain. This directly couples the qubits without
    intermediate couplers and assumes a linear chain architecture of qubits.

    Parameters
    ----------
    coupling_strength: List[float]
        List of coupling strength between adjacent Qubits. For n qubits, this needs to be a list of n-1 values.
    qubits: List[chip.Transmon]
        List of qubits

    Returns
    -------
    List of couplings
    """
    if len(coupling_strength) < len(qubits) - 1:
        raise Exception("not enough coupling constants")

    couplings = []

    for i in range(len(qubits) - 1):
        couplings.append(
            chip.Coupling(
                name=f"{qubits[i].name}-{qubits[i + 1].name}",
                desc="Coupling",
                comment=f"Coupling between {qubits[i].name} and {qubits[i + 1].name}",
                connected=[qubits[i].name, qubits[i + 1].name],
                strength=Qty(
                    value=coupling_strength[i],
                    min_val=-1 * 1e3,
                    max_val=200e6,
                    unit="Hz 2pi",
                ),
                hamiltonian_func=hamiltonians.int_XX,
            )
        )

    return couplings


def createChainCouplingsWithCouplers(
    coupling_strength: List[float],
    qubits: List[chip.Transmon],
    couplers: List[chip.Transmon],
) -> List[chip.Coupling]:
    """
    Creates and returns a list of Nearest neighbour (NN) couplings where each pair of qubits is coupled via a coupler.
    This assumes a linear chain architecture of Qubits and couplers.

    Parameters
    ----------
    coupling_strength: List[float]
        List of coupling strength between the Qubits and couplers. For n qubits, this list needs to contain 2(n-1)
        values, two for each coupler.
    qubits: List[chip.Transmon]
        List of qubits
    couplers: List[chip.Transmon]
        List of couplers. For n qubits, this list needs to contain n-1 couplers.

    Returns
    -------
    List of couplings
    """
    if len(couplers) < len(qubits) - 1:
        raise Exception("not enough couplers")
    if len(coupling_strength) < 2 * len(couplers):
        raise Exception("not enough coupling constants")

    g_NN_array = []

    for i in range(2 * len(couplers)):
        coupler_index = int(np.floor(i / 2))
        qubit_index = int(np.floor((i + 1) / 2))
        g_NN_array.append(
            chip.Coupling(
                name=f"{qubits[qubit_index].name}-{couplers[coupler_index].name}",
                desc="Coupling",
                comment=f"Coupling between {qubits[qubit_index].name} and {couplers[coupler_index].name}",
                connected=[
                    qubits[qubit_index].name,
                    couplers[coupler_index].name,
                ],
                strength=Qty(
                    value=coupling_strength[i],
                    min_val=-1 * 1e3,
                    max_val=200e6,
                    unit="Hz 2pi",
                ),
                hamiltonian_func=hamiltonians.int_XX,
            )
        )

    return g_NN_array


def createDrives(qubits: List[chip.Transmon]) -> List[chip.Drive]:
    """
    Creates and returns a drive line for each qubit in the list.

    Parameters
    ----------
    qubits : List[chip.PhysicalComponent]
        One drive line is generated for each qubit in this list. The qubits are not altered.

    Returns
    -------
    List[chip.Drive]
        The drive lines in the same order as the list of qubits.
    """
    drives = []
    for i in range(len(qubits)):
        drives.append(
            chip.Drive(
                name=f"d{i + 1}",
                desc=f"Drive {i + 1}",
                comment=f"Drive line on qubit {qubits[i].name}",
                connected=[qubits[i].name],
                hamiltonian_func=hamiltonians.x_drive,
            )
        )
    return drives


def createGenerator(
        drives: List[chip.Drive],
        sim_res: float = 100e9,
        awg_res: float = 2e9,
        useDrag: bool = False,
        usePWC: bool = False
) -> Generator:
    """
    Creates and returns the generator.

    Parameters
    ----------
    drives: List[chip.Drive]
        List of drives on the Qubits
    sim_res: float
        Resolution of the simulation
    awg_res: float
        Resolution of AWG
    useDrag: bool
        Whether to enable DRAG in the AWG

    Returns
    -------
    Generator
    """
    chain = {
        "LO": [],
        "AWG": [],
        "DigitalToAnalog": ["AWG"],
        "ResponseFFT": ["DigitalToAnalog"],
        "Mixer": ["LO", "ResponseFFT"],
        "VoltsToHertz": ["Mixer"],
    }
    chains = {f"{d.name}": chain for d in drives}

    generator = Generator(
        devices={
            "LO": devices.LO(name="lo", lo_index=1, resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name="awg", awg_index=1, resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "ResponseFFT": devices.ResponseFFT(
                name="resp",
                rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
                resolution=sim_res,
                inputs=1,
                outputs=1,
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="V_to_Hz",
                V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chains=chains,
    )

    if useDrag:
        print("enabling DRAG2")
        generator.devices["AWG"].enable_drag_2()
    if usePWC:
        print("enabling PWC in AWG")
        generator.devices["AWG"].enable_pwc()

    return generator


def createGenerator2LOs(
        drives: List[chip.Drive],
        sim_res: float = 100e9,
        awg_res: float = 2e9,
        useDrag: bool = False,
        usePWC: bool = False
) -> Generator:
    """
    Creates and returns the generator.

    Parameters
    ----------
    drives: List[chip.Drive]
        List of drives on the Qubits
    sim_res: float
        Resolution of the simulation
    awg_res: float
        Resolution of AWG
    useDrag: bool
        Whether to enable DRAG in the AWG

    Returns
    -------
    Generator
    """
    chain = {
        "LO1": [],
        "LO2": [],
        "AWG1": [],
        "AWG2": [],
        "DigitalToAnalog1": ["AWG1"],
        "DigitalToAnalog2": ["AWG2"],
        "ResponseFFT1": ["DigitalToAnalog1"],
        "ResponseFFT2": ["DigitalToAnalog2"],
        "Mixer1": ["LO1", "ResponseFFT1"],
        "Mixer2": ["LO2", "ResponseFFT2"],
        "RealMixer": ["Mixer1", "Mixer2"],
        "VoltsToHertz": ["RealMixer"],
    }
    chains = {f"{d.name}": chain for d in drives}

    generator = Generator(
        devices={
            "LO1": devices.LO(lo_index=1, name="lo1", resolution=sim_res, outputs=1),
            "LO2": devices.LO(lo_index=2, name="lo2", resolution=sim_res, outputs=1),
            "AWG1": devices.AWG(
                awg_index=1, name="awg1", resolution=awg_res, outputs=1
            ),
            "AWG2": devices.AWG(
                awg_index=2, name="awg2", resolution=awg_res, outputs=1
            ),
            "DigitalToAnalog1": devices.DigitalToAnalog(
                name="dac1", resolution=sim_res, inputs=1, outputs=1
            ),
            "DigitalToAnalog2": devices.DigitalToAnalog(
                name="dac2", resolution=sim_res, inputs=1, outputs=1
            ),
            "ResponseFFT1": devices.ResponseFFT(
                name="resp1",
                rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
                resolution=sim_res,
                inputs=1,
                outputs=1,
            ),
            "ResponseFFT2": devices.ResponseFFT(
                name="resp2",
                rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
                resolution=sim_res,
                inputs=1,
                outputs=1,
            ),
            "Mixer1": devices.Mixer(name="mixer1", inputs=2, outputs=1),
            "Mixer2": devices.Mixer(name="mixer2", inputs=2, outputs=1),
            "RealMixer": devices.RealMixer(name="realmixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="V_to_Hz",
                V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chains=chains,
    )

    if useDrag:
        print("enabling DRAG")
        generator.devices["AWG1"].enable_drag_2()
        generator.devices["AWG2"].enable_drag_2()
    if usePWC:
        print("enabling PWC in AWG")
        generator.devices["AWG1"].enable_pwc()
        generator.devices["AWG2"].enable_pwc()

    return generator


def createGeneratorNLOs(
        drives: List[chip.Drive],
        N: int,
        sim_res: float = 100e9,
        awg_res: float = 2e9,
        useDrag: bool = False,
        usePWC: bool = False
) -> Generator:
    """
    Creates and returns the generator.

    Parameters
    ----------
    drives: List[chip.Drive]
        List of drives on the Qubits
    N: int
        The number of LOs and AWGs.
    sim_res: float
        Resolution of the simulation
    awg_res: float
        Resolution of AWG
    useDrag: bool
        Whether to enable DRAG in the AWG

    Returns
    -------
    Generator
    """

    # create chain
    chain: Dict[str, List[str]] = {}
    for n in range(1, N + 1):
        chain[f"LO{n}"] = []
        chain[f"AWG{n}"] = []
        chain[f"DigitalToAnalog{n}"] = [f"AWG{n}"]
        chain[f"ResponseFFT{n}"] = [f"DigitalToAnalog{n}"]
        chain[f"Mixer{n}"] = [f"LO{n}", f"ResponseFFT{n}"]
    chain["RealMixer"] = [f"Mixer{n}" for n in range(1, N + 1)]
    chain["VoltsToHertz"] = ["RealMixer"]
    chains = {f"{d.name}": chain for d in drives}

    # create devices
    devs: Dict[str, devices.Device] = {}
    for n in range(1, N + 1):
        devs[f"LO{n}"] = devices.LO(
            lo_index=n, name=f"lo{n}", resolution=sim_res, outputs=1
        )
        devs[f"AWG{n}"] = devices.AWG(
            awg_index=n, name=f"awg{n}", resolution=awg_res, outputs=1
        )
        devs[f"DigitalToAnalog{n}"] = devices.DigitalToAnalog(
            name=f"dac{n}", resolution=sim_res, inputs=1, outputs=1
        )
        devs[f"ResponseFFT{n}"] = devices.ResponseFFT(
            name=f"resp{n}",
            rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
            resolution=sim_res,
            inputs=1,
            outputs=1,
        )
        devs[f"Mixer{n}"] = devices.Mixer(name=f"mixer{n}", inputs=2, outputs=1)
    devs["RealMixer"] = devices.RealMixer(name="realmixer", inputs=N, outputs=1)
    devs["VoltsToHertz"] = devices.VoltsToHertz(
        name="V_to_Hz",
        V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
        inputs=1,
        outputs=1,
    )

    generator = Generator(devices=devs, chains=chains)
    if useDrag:
        print("enabling DRAG2")
        for n in range(1, N + 1):
            generator.devices[f"AWG{n}"].enable_drag_2()
    if usePWC:
        print("enabling PWC in AWG")
        for n in range(1, N + 1):
            generator.devices[f"AWG{n}"].enable_pwc()

    return generator


def createCarriers(qubit_freqs: List[float], sideband: float) -> List[pulse.Carrier]:
    """
    Creates and returns a carrier for each qubit.

    Parameters
    ----------
    qubit_freqs: List[float]
        List of frequencies of the qubits.
        For tunable qubits, one can use model.get_qubit_freqs()
        to find the frequency of the qubits at current flux.
    sideband: float
        Frequency of the sideband.

    Returns
    -------
    List of carriers for each qubit
    """
    carrier_array = []

    framechange = 0.0  # (-sideband * t_final) % (2 * np.pi)
    for i in range(len(qubit_freqs)):
        f = qubit_freqs[i] + sideband
        carrier_parameters = {
            "freq": Qty(value=f, min_val=0.95 * f, max_val=1.02 * f, unit="Hz 2pi"),
            "framechange": Qty(
                value=framechange, min_val=-np.pi, max_val=3 * np.pi, unit="rad"
            ),
        }
        carrier_array.append(
            pulse.Carrier(
                name=f"carrier{i + 1}",
                desc="Frequency of the local oscillator",
                params=carrier_parameters,
            )
        )

    return carrier_array


def createSingleQubitGate(
    name: str,
    drives: List[chip.Drive],
    carriers: List[pulse.Carrier],
    target: int,
    t_final: float,
    target_pulse: pulse.Envelope,
    non_target_pulse: pulse.Envelope,
    sideband: float,
    xy_angle: float = None,
    ideal: np.array = None,
) -> gates.Instruction:
    """
    Creates an instruction that represents a single qubit gate. This applies the target pulse to the target qubit's
    drive line and the "non-target pulse" to all other qubits.

    Parameters
    ----------
    name: str
        Name of the gate
    drives: List[chip.Drive]
        List of drive lines for each qubit
    carriers: List[pulse.Carrier]
        list of carrier pulses for each qubit
    target: int
        index of the target qubit on which the gate should act
    t_final: float
        Final gate time for single qubit rotation gates
    target_pulse: pulse.Envelope
        pulse for single qubit rotation gates
    non_target_pulse: pulse.Envelope
        pulse for correcting phase
    sideband: float
        Frequency of sideband
    xy_angle: float
        The angle by which the target's drive should be shifted to create a different gate
    ideal: np.array
        Matrix representation of the ideal gate, if not specified by the name.

    Returns
    -------
    Single qubit rotation gates for one qubit on the chip.
    """
    gate = gates.Instruction(
        name=name,
        targets=[target],
        t_start=0.0,
        t_end=t_final,
        channels=[d.name for d in drives],
        ideal=ideal,
    )
    for i in range(len(drives)):
        if i == target:
            gate.add_component(copy.deepcopy(target_pulse), drives[i].name)
            gate.add_component(copy.deepcopy(carriers[i]), drives[i].name)
            if xy_angle is not None:
                gate.comps[drives[i].name][target_pulse.name].params[
                    "xy_angle"
                ].set_value(xy_angle)
        else:
            carrier = copy.deepcopy(carriers[i])
            carrier.params["framechange"].set_value(
                (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
            )
            gate.add_component(copy.deepcopy(non_target_pulse), drives[i].name)
            gate.add_component(carrier, drives[i].name)

    return gate


def createTwoQubitsGate(
    name: str,
    drives: List[chip.Drive],
    carriers: List[pulse.Carrier],
    targets: List[int],
    t_final: float,
    target_pulse: pulse.Envelope,
    non_target_pulse: pulse.Envelope,
    sideband: float,
    ideal: np.array = None,
) -> gates.Instruction:
    """
    Creates a gate that acts on a list of qubits. This applies a copy of the target pulse to all the target qubit's
    drive lines and the "non-target pulse" to all other qubits.

    Parameters
    ----------
    name: str
        Name of the gate
    drives: List[chip.Drive]
        List of drive lines for each qubit
    carriers: List[pulse.Carrier]
        list of carrier pulses for each qubit
    targets: List[int]
        indices of the target qubits on which the gate should act
    t_final: float
        Final gate time for single qubit rotation gates
    target_pulse: pulse.Envelope
        pulse for single qubit rotation gates
    non_target_pulse: pulse.Envelope
        pulse for correcting phase
    sideband: float
        Frequency of sideband
    ideal: np.array
        Matrix representation of the ideal gate, if not specified by the name.

    Returns
    -------
    Single qubit rotation gates for one qubit on the chip.
    """
    gate = gates.Instruction(
        name=name,
        targets=targets,
        t_start=0.0,
        t_end=t_final,
        channels=[d.name for d in drives],
        ideal=ideal,
    )

    for i in range(len(drives)):
        if i in targets:
            gate.add_component(copy.deepcopy(target_pulse), drives[i].name)
            gate.add_component(copy.deepcopy(carriers[i]), drives[i].name)
        else:
            carrier = copy.deepcopy(carriers[i])
            carrier.params["framechange"].set_value(
                (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
            )
            gate.add_component(copy.deepcopy(non_target_pulse), drives[i].name)
            gate.add_component(carrier, drives[i].name)

    return gate


def getDrive(model: Model, subsystem: chip.PhysicalComponent) -> chip.Drive:
    """
    Returns the drive line that is connected to the subsystem, or None if no drive is connected to it.
    """
    drives = [
        cast(chip.Drive, d) for d in model.couplings.values() if type(d) == chip.Drive
    ]
    connected = list(filter(lambda d: subsystem.name in d.connected, drives))
    return connected[0] if len(connected) > 0 else None


def generateSignal(
        experiment: Experiment, gate: gates.Instruction, subsystem: chip.PhysicalComponent
) -> Dict[str, tf.Tensor]:
    """
    Makes the generator generate a signal for the specific subsystem and returns it.
    """
    drive = getDrive(experiment.pmap.model, subsystem).name
    return experiment.pmap.generator.generate_signals(gate)[drive]


def getOutputFromDevice(
        gen: Generator, gate: gates.Instruction, channel: str, deviceName: str
):
    gen.generate_signals(gate)
    return gen.getDeviceOutput(channel, deviceName)


def getEnvelope(gen: Generator, gate: gates.Instruction, channel: str):
    full_signal = gen.generate_signals(gate)[channel]
    values1 = full_signal["values"].numpy()
    envelope = gen.getDeviceOutput(channel, "ResponseFFT")
    values2 = envelope["inphase"].numpy()
    # TODO: this isn't good!
    if np.max(np.abs(values2)) != 0:
        factor = np.max(np.abs(values1)) / np.max(np.abs(values2))
    else:
        factor = 1
    return (
        envelope["ts"].numpy(),
        factor * envelope["inphase"].numpy(),
        factor * envelope["quadrature"].numpy(),
    )


def findPeaks(x: np.array, y: np.array, N: int) -> Tuple[np.array, np.array]:
    """
    Returns the x and y values of the N largest local maxima in y.
    """
    peaks = find_peaks(y)[0]
    peakX = x[peaks]
    peakY = y[peaks]
    maxIndices = peakY.argsort()[-N:][::-1]
    return peakX[maxIndices], peakY[maxIndices]


def findFrequencyPeaks(
    time: np.array, signal: np.array, N: int, normalise: bool = False
) -> Tuple[np.array, np.array]:
    """
    Calculates the Fourier transformation of the signal and returns the N highest peaks from the frequency spectrum.
    Parameters
    ----------
    time: array of time stamps
    signal: array of signal values at the time stamps
    N: number of peaks to return
    normalise: whether the spectrum should be normalised
    Returns
    -------
    a tuple of frequencies and peak values for all N peaks
    """
    x = time.flatten()
    y = signal.flatten()

    freq_signal = np.fft.rfft(y)
    if normalise and np.abs(np.max(freq_signal)) > 1e-14:
        normalised = freq_signal / np.max(freq_signal)
    else:
        normalised = freq_signal
    freq = np.fft.rfftfreq(len(x), x[-1] / len(x))

    return findPeaks(freq, np.abs(normalised) ** 2, N)


def calculatePopulation(
    exp: Experiment, psi_init: tf.Tensor, sequence: List[str]
) -> np.array:
    """
    Calculates the time dependent population starting from a specific initial state.

    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    psi_init: tf.Tensor
        Initial state vector
    sequence: List[str]
        List of gate names that will be applied to the state

    Returns
    -------
    np.array
       two-dimensional array, first dimension: time, second dimension: population of the levels
    """
    # calculate the time dependent level population
    model = exp.pmap.model
    dUs = exp.partial_propagators
    if type(psi_init).__module__ != 'numpy':
        psi_init = psi_init.numpy()
    psi_t = psi_init
    pop_t = exp.populations(psi_t, model.lindbladian)
    for gate in sequence:
        for du in dUs[gate]:
            psi_t = np.matmul(du, psi_t)
            pops = exp.populations(psi_t, model.lindbladian)
            pop_t = np.append(pop_t, pops, axis=1)
    return pop_t


def calculateObservable(
        exp: Experiment, psi_init, sequence: List[str], function: Callable
) -> np.array:
    """
    Calculates the value of a state dependent function during time evolution starting from a specific initial state.

    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    psi_init: tf.Tensor
        Initial state vector
    sequence: List[str]
        List of gate names that will be applied to the state

    Returns
    -------
    np.array
       two-dimensional array, first dimension: time, second dimension: entropy
    """
    # calculate the time dependent level population
    dUs = exp.partial_propagators
    if type(psi_init).__module__ != 'numpy':
        psi_init = psi_init.numpy()
    psi_t = psi_init
    pop_t = np.array([function(psi_t)])
    for gate in sequence:
        for idx, du in enumerate(dUs[gate]):
            psi_t = np.matmul(du, psi_t)
            pop_t = np.append(pop_t, function(psi_t))
    return pop_t


def getQubitsPopulation(population: np.array, dims: List[int]) -> np.array:
    """
    Splits the population of all levels of a system into the populations of levels per subsystem.
    Parameters
    ----------
    population: np.array
        The time dependent population of each energy level. First dimension: level index, second dimension: time.
    dims: List[int]
        The number of levels for each subsystem.
    Returns
    -------
    np.array
        The time-dependent population of energy levels for each subsystem. First dimension: subsystem index, second
        dimension: level index, third dimension: time.
    """
    numQubits = len(dims)

    # create a list of all levels
    qubit_levels = []
    for dim in dims:
        qubit_levels.append(list(range(dim)))
    combined_levels = list(itertools.product(*qubit_levels))

    # calculate populations
    qubitsPopulations = np.zeros((numQubits, dims[0], population.shape[1]))
    for idx, levels in enumerate(combined_levels):
        for i in range(numQubits):
            qubitsPopulations[i, levels[i]] += population[idx]
    return qubitsPopulations


def partialTrace(M: np.ndarray, qubitsToKeep) -> np.ndarray:
    """
    Calculates the partial trace of a matrix.

    Parameters
    ----------
    M: np.ndarray
        Density matrix
    qubitsToKeep: list
        Index of qubit to be kept after taking the trace
    Returns
    -------
    np.ndarray
        Density matrix after taking partial trace
    """
    numQubits = int(np.log2(M.shape[0]))
    qubitAxis = [(i, numQubits + i) for i in range(numQubits) if i not in qubitsToKeep]
    minusFactor = [(i, 2 * i) for i in range(len(qubitAxis))]
    minusQubitAxis = [
        (q[0] - m[0], q[1] - m[1]) for q, m in zip(qubitAxis, minusFactor)
    ]
    Mres = np.reshape(M, [2, 2] * numQubits)
    numQubitsLeft = numQubits - len(qubitAxis)
    for i, j in minusQubitAxis:
        Mres = np.trace(Mres, axis1=i, axis2=j)
    if numQubitsLeft > 1:
        Mres = np.reshape(Mres, [2 ** numQubitsLeft] * 2)
    return Mres


def partialTraceTF(tensor: tf.Tensor, keep_indices: List[int]) -> tf.Tensor:
    numQubits = int(np.log2(tensor.shape[0]))
    M = tf.reshape(tensor, shape=[2, 2] * numQubits)
    ndim = M.ndim // 2

    keep_set = set(keep_indices)
    keep_map = dict(zip(keep_indices, sorted(keep_indices)))
    left_indices = [keep_map[i] if i in keep_set else i for i in range(ndim)]
    right_indices = [ndim + i if i in keep_set else i for i in left_indices]
    single_indices = [i for i in left_indices if i not in right_indices] \
                     + [i for i in right_indices if i not in left_indices]

    letters = np.array(['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'])
    equation = ''.join(letters[left_indices + right_indices])
    equation += '->' + ''.join(letters[single_indices])
    rhoPartial = tf.einsum(equation, M)

    return tf.reshape(rhoPartial, [2 ** len(keep_indices)] * 2)


def densityMatrix(state: np.array):
    return np.outer(state, np.conjugate(state))


def densityMatrixTF(state: tf.Tensor):
    return tf.tensordot(state, tf.math.conj(state), axes=0)


def entanglementEntropy(rho: np.array):
    vals = np.linalg.eigvalsh(rho.data)
    nzvals = vals[np.where(vals > 1e-15)]
    logvals = np.log2(nzvals)
    return -np.sum(nzvals * logvals)


def entanglementEntropyTF(rho: tf.Tensor):
    vals = tf.cast(tf.math.real(tf.linalg.eigvalsh(rho)), tf.float64)
    nzvals = vals[tf.math.greater(vals, tf.ones_like(vals) * 1e-15)]
    logvals = tf.math.log(nzvals) / tf.cast(tf.math.log(2.0), tf.float64)
    return -tf.reduce_sum(nzvals * logvals)


def scaleQuantity(q: Qty, factor: float) -> Qty:
    return Qty(value=q.get_value() * factor, min_val=q.get_limits()[0] * factor, max_val=q.get_limits()[1] * factor,
               unit=q.unit)


def scaleGaussianEnvelope(envelope: pulse.Envelope, factor: float) -> pulse.Envelope:
    """
    Scales a gaussian envelope such that the integrated area stays constant. The final time and sigma
    are multiplied by the factor, the amplitude is divided by it.
    -------
    """
    envelope.params['amp'] = scaleQuantity(envelope.params['amp'], 1.0 / factor)
    envelope.params['t_final'] = scaleQuantity(envelope.params['t_final'], factor)
    envelope.params['sigma'] = scaleQuantity(envelope.params['sigma'], factor)
    return envelope
