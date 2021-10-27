from typing import List

import numpy as np

# Main C3 objects
from c3.c3objs import Quantity as Qty

# Building blocks
import c3.libraries.chip as chip

# Libs and helpers
import c3.libraries.hamiltonians as hamiltonians


def intialize_qubits(
    Num_qubits,
    qubit_levels_list,
    freq_list,
    anharm_list,
    t1_list,
    t2star_list,
    phi_list,
    phi0_list,
    d_list,
    qubit_temp,
):
    qubit_array = []
    for i in range(Num_qubits):
        qubit_array.append(
            chip.Transmon(
                name="Q" + str(i + 1),
                desc="Qubit  " + str(i + 1),
                freq=Qty(value=freq_list[i], min_val=2e9, max_val=8e9, unit="Hz 2pi"),
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

    return qubit_array


def initialize_coupler(
    Num_coupler,
    coupler_levels_list,
    freq_list,
    anharm_list,
    t1_list,
    t2star_list,
    phi_list,
    phi0_list,
    d_list,
    qubit_temp,
):
    coupler_array = []
    for i in range(Num_coupler):
        coupler_array.append(
            chip.Transmon(
                name="C" + str(i + 1),
                desc="Coupler  " + str(i + 1),
                freq=Qty(
                    value=freq_list[i], min_val=3.995e9, max_val=4.005e9, unit="Hz 2pi"
                ),
                anhar=Qty(
                    value=anharm_list[i], min_val=-380e6, max_val=-120e6, unit="Hz 2pi"
                ),
                hilbert_dim=coupler_levels_list[i],
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


def generate_couplings(
    Num_qubits,
    Num_coupler,
    NN_coupling_strength,
    NNN_coupling_strength,
    qubit_array,
    coupler_array,
):

    ## For chain architecture of qubits and couplers

    g_NN_array = []
    g_NNN_array = []

    for i in range(Num_coupler + Num_qubits - 1):
        coupler_index = int(np.floor(i / 2) + 1)
        qubit_index = int(np.floor((i + 1) / 2) + 1)
        g_NN_array.append(
            chip.Coupling(
                name=qubit_array[qubit_index].name
                + "-"
                + coupler_array[coupler_index].name,
                desc="Coupling",
                comment="Coupling between "
                + qubit_array[qubit_index].name
                + " and "
                + coupler_array[coupler_index].name,
                connected=[
                    qubit_array[qubit_index].name,
                    coupler_array[coupler_index].name,
                ],
                strength=Qty(
                    value=NN_coupling_strength[i],
                    min_val=-1 * 1e3,
                    max_val=200e6,
                    unit="Hz 2pi",
                ),
                hamiltonian_func=hamiltonians.int_XX,
            )
        )

    for i in range(len(NNN_coupling_strength)):
        g_NNN_array.append(
            chip.Coupling(
                name=qubit_array[i].name + "-" + qubit_array[i + 1].name,
                desc="Coupling",
                comment="Coupling between "
                + qubit_array[i].name
                + " and "
                + qubit_array[i + 1].name,
                connected=[qubit_array[i].name, qubit_array[i + 1].name],
                strength=Qty(
                    value=NNN_coupling_strength[i],
                    min_val=-1 * 1e3,
                    max_val=200e6,
                    unit="Hz 2pi",
                ),
                hamiltonian_func=hamiltonians.int_XX,
            )
        )

    return g_NN_array + g_NNN_array


def createDrives(qubits: List[chip.PhysicalComponent]) -> List[chip.Drive]:
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
                name=f"d{i}",
                desc=f"Drive {i}",
                comment=f"Drive line on qubit {qubits[i].name}",
                connected=[qubits[i].name],
                hamiltonian_func=hamiltonians.x_drive,
            )
        )
    return drives
