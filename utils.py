from typing import List, Dict
import c3.libraries.chip as chip
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.tasks as tasks
from c3.c3objs import Quantity
from c3.model import Model


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
