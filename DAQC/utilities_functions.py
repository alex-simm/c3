import numpy as np
import copy 
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.signal.gates as gates
import c3.libraries.chip as chip
import c3.signal.pulse as pulse
import c3.libraries.tasks as tasks

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes
import c3.utils.qt_utils as qt_utils
import c3.utils.tf_utils as tf_utils


def intialize_qubits(Num_qubits, qubit_levels_list, freq_list, anharm_list, t1_list, t2star_list, phi_list, phi0_list, d_list, qubit_temp):
    qubit_array = []
    for i in range(Num_qubits):
        qubit_array.append( 
                chip.Transmon(
                        name = "Q" + str(i+1),
                        desc = "Qubit  "+ str(i+1),
                        freq = Qty(
                                value = freq_list[i],
                                min_val = 2e9,
                                max_val = 8e9,
                                unit = "Hz 2pi"
                        ),
                        anhar = Qty(
                                value = anharm_list[i],
                                min_val = -380e6,
                                max_val = -120e6,
                                unit = "Hz 2pi"
                        ),
                        hilbert_dim = qubit_levels_list[i],
                        t1 = Qty(
                                value = t1_list[i],
                                min_val = 1e-6,
                                max_val = 90e-6,
                                unit = "s"
                        ),
                        t2star=Qty(
                                value=t2star_list[i],
                                min_val=10e-6,
                                max_val=90e-6,
                                unit="s"
                        ),
                        temp=Qty(
                                value=qubit_temp,
                                min_val=0.0,
                                max_val=0.12,
                                unit="K"
                        ),
                        phi=Qty(
                                value=phi_list[i],
                                max_val=5.0,
                                min_val=0.0,
                                unit = "Wb"
                        ),
                        phi_0 = Qty(
                                value = phi0_list[i],
                                max_val = 11.0,
                                min_val = 9.0,
                                unit = "Wb"
                        ),
                        d = Qty(
                                value=d_list[i],
                                max_val=0.1,
                                min_val=-0.1,
                                unit = ""
                        )
                        
                )
        )

    return qubit_array

def initialize_coupler(Num_coupler, coupler_levels_list, freq_list, anharm_list, t1_list, t2star_list, phi_list, phi0_list, d_list, qubit_temp):
    for i in range(Num_coupler):
        coupler_array = []
        coupler_array.append( 
                chip.Transmon(
                        name = "C" + str(i+1),
                        desc = "Coupler  "+ str(i+1),
                        freq = Qty(
                                value = freq_list[i],
                                min_val = 3.995e9,
                                max_val = 4.005e9,
                                unit = "Hz 2pi"
                        ),
                        anhar = Qty(
                                value = anharm_list[i],
                                min_val = -380e6,
                                max_val = -120e6,
                                unit = "Hz 2pi"
                        ),
                        hilbert_dim = coupler_levels_list[i],
                        t1 = Qty(
                                value = t1_list[i],
                                min_val = 1e-6,
                                max_val = 90e-6,
                                unit = "s"
                        ),
                        t2star=Qty(
                                value=t2star_list[i],
                                min_val=10e-6,
                                max_val=90e-6,
                                unit="s"
                        ),
                        temp=Qty(
                                value=qubit_temp,
                                min_val=0.0,
                                max_val=0.12,
                                unit="K"
                        ),
                        phi=Qty(
                                value=phi_list[i],
                                max_val=5.0,
                                min_val=0.0,
                                unit = "Wb"
                        ),
                        phi_0 = Qty(
                                value = phi0_list[i],
                                max_val = 11.0,
                                min_val = 9.0,
                                unit = "Wb"
                        ),
                        d = Qty(
                                value=d_list[i],
                                max_val=0.1,
                                min_val=-0.1,
                                unit = ""
                        )
                        
                )
        )

def generate_couplings(Num_qubits, Num_coupler, NN_coupling_strength, NNN_coupling_strength, qubit_array, coupler_array):
    
    ## Change the two loops to one loop
    
    g_NN_array = []
    g_NNN_array = []
    for i in range(Num_coupler):
        for j in range(Num_qubits):
            if j == i + 1 or j == i: 
                g_NN_array.append(
                    chip.Coupling(
                        name=qubit_array[j].name + "-" + coupler_array[i].name,
                        desc="Coupling",
                        comment="Coupling between "+ qubit_array[j].name + " and " + coupler_array[i].name,
                        connected=[qubit_array[j].name, coupler_array[i].name],
                        strength=Qty(
                                value=NN_coupling_strength[i],
                                min_val=-1*1e3,
                                max_val=200e6,
                                unit="Hz 2pi"
                        ),
                        hamiltonian_func=hamiltonians.int_XX
                    )
                )
        else:
            continue

        

