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
    return coupler_array


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
        coupler_index = int(np.floor(i / 2))
        qubit_index = int(np.floor((i + 1) / 2))
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


def initialize_qubit_drives(Num_qubits, qubit_array):
    drive_array = []
    for i in range(Num_qubits):
        drive_array.append(
            chip.Drive(
            name="d" + str(i+1),
            desc="Drive " + str(i+1),
            comment = "Drive line" +str(i+1) +  "on qubit "+str(i+1),
            connected = [qubit_array[i].name],
            hamiltonian_func = hamiltonians.x_drive
            )
        )

    return drive_array


def build_confusion_matrix(Num_qubits, m00_arr, m01_arr, qubit_levels):
    ## For now this code is written assuming same number of levels for all the qubits

    one_zeros = np.array([0] * qubit_levels)
    zero_ones = np.array([1] * qubit_levels)
    one_zeros[0] = 1
    zero_ones[0] = 0

    confusion_row_arr = []
    for i in range(Num_qubits):
        val = one_zeros * m00_arr[i] + zero_ones * m01_arr[i]

        min_val = one_zeros * 0.8 + zero_ones * 0.0
        max_val = one_zeros * 1.0 + zero_ones * 0.2

        confusion_row_arr.append( Qty(value=val, min_val=min_val, max_val=max_val, unit="") )

    conf_matrix = tasks.ConfusionMatrix(confusion_row_arr)

    return conf_matrix



def build_generator(Num_qubits, drive_array, sim_res, awg_res, v2hz):
    sim_res = 100e9
    awg_res = 2e9
    chain = {}
    for i in range(Num_qubits):
        chain[drive_array[i].name] = ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"]

    generator = Gnr(
        devices = {
                "LO": devices.LO(name="lo", resolution = sim_res, outputs= 1),
                "AWG": devices.AWG(name = "awg", resolution = awg_res, outputs = 1),
                "DigitalToAnalog": devices.DigitalToAnalog(
                        name = "dac",
                        resolution = sim_res,
                        inputs = 1,
                        outputs = 1
                ),
                "Response": devices.Response(
                        name = "resp",
                        rise_time = Qty(
                                value = 0.3e-9,
                                min_val = 0.05e-9,
                                max_val = 0.6e-9,
                                unit = "s"
                        ),
                        resolution = sim_res,
                        inputs = 1,
                        outputs = 1
                ),
                "Mixer": devices.Mixer(name = "mixer", inputs = 2, outputs = 1),
                "VoltsToHertz": devices.VoltsToHertz(
                        name = "V_to_Hz",
                        V_to_Hz = Qty(
                                value = v2hz,
                                min_val = 0.9e9,
                                max_val = 1.1e9,
                                unit = "Hz/V"
                        ),
                        inputs= 1,
                        outputs = 1
                )
        },
        
        chains = chain

    )

    return generator



def build_carriers(Num_qubits, qubit_freqs, sideband):
    lo_freq_array = [qubit_freqs[i] + sideband for i in range(Num_qubits) ]

    carrier_array = []

    for i in range(Num_qubits):

            carrier_parameters = {
                    "freq": Qty(
                            value = lo_freq_array[i],
                            min_val = 1e9,
                            max_val = 8e9,
                            unit = "Hz 2pi"
                    ),
                    "framechange": Qty(
                            value = 0.0,
                            min_val = -np.pi,
                            max_val = 3 * np.pi,
                            unit = "rad"
                    )
            }

            carrier_array.append( 
                    pulse.Carrier(
                            name = "carrier",
                            desc = "Frequency of the local oscillator",
                            params = carrier_parameters
                    )
            )

    return carrier_array


def build_single_qubit_XY_gates(Num_qubits, drive_array, carrier_array,t_final, drive_pulse, nodrive_pulse, sideband):
    rx90p_gate_array = []

    for i in range(Num_qubits):
        rx90p_q = gates.Instruction(
                name = "rx90p", targets = [i], t_start = 0.0, t_end = t_final, channels=[drive_array[i].name,drive_array[(i+1)%Num_qubits].name]
        )
        rx90p_q.add_component(drive_pulse, drive_array[i].name)
        rx90p_q.add_component(carrier_array[i], drive_array[i].name)
        rx90p_q.add_component(nodrive_pulse,drive_array[(i+1)%Num_qubits].name)
        rx90p_q.add_component(copy.deepcopy(carrier_array[(i+1)%Num_qubits]), drive_array[(i+1)%Num_qubits].name)
        rx90p_q.comps[drive_array[(i+1)%Num_qubits].name]["carrier"].params["framechange"].set_value(
                (-sideband * t_final) * 2 * np.pi % (2*np.pi)
        )
        rx90p_gate_array.append(rx90p_q)


    ry90p_gate_array = []
    rx90m_gate_array = []
    ry90m_gate_array = []
    for i in range(Num_qubits):
            ry90p_q = copy.deepcopy(rx90p_gate_array[i])
            ry90p_q.name = "ry90p"
            rx90m_q = copy.deepcopy(rx90p_gate_array[i])
            rx90m_q.name = "rx90m"
            ry90m_q = copy.deepcopy(rx90p_gate_array[i])
            ry90m_q.name = "ry90m"
            ry90p_q.comps[drive_array[i].name]["gauss"].params["xy_angle"].set_value(0.5*np.pi)
            rx90m_q.comps[drive_array[i].name]["gauss"].params["xy_angle"].set_value(np.pi)
            ry90m_q.comps[drive_array[i].name]["gauss"].params["xy_angle"].set_value(1.5*np.pi)

            ry90p_gate_array.append(ry90p_q)
            ry90m_gate_array.append(ry90m_q)
            rx90m_gate_array.append(rx90m_q)


    single_qubit_gates = rx90p_gate_array + rx90m_gate_array + ry90p_gate_array + ry90m_gate_array
        
    return single_qubit_gates
