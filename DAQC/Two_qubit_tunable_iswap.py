# %% [markdown]
# # Two Tunable Qubits + One Tunable Coupler: Engtangling gates

# %% [markdown]
# ### Imports

# %%
#!pip install c3-toolset
#!pip install matplotlib
#!pip install plotly
#!pip install nbformat --upgrade

# %%
import os
import tempfile
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.eager.context import num_gpus
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
from c3.optimizers.optimalcontrol import OptimalControl

#%matplotlib widget
import plotly.graph_objects as go
from utilities_functions import *
from plotting import *

# %% [markdown]
# ### Define Qubits and coupler

# %% [markdown]
# Define Qubits, Coupler, Drives and Couplings

# %%
qubit_levels = [3, 3]
qubit_frequencies = [4.16e9, 4.0e9]
anharmonicities = [-220e6, -210e6]
t1 = [60e-6, 30e-6]
t2star = [66e-6, 5e-6]
qubit_temp = [50e-3, 50e-3]
transmon_phi = [0.0, 0.0]
transmon_phi_0 = [10.0, 10.0]
transmon_d = [0.0, 0.0]


qubits = createQubits(qubit_levels, qubit_frequencies, anharmonicities, t1,
                      t2star, transmon_phi, transmon_phi_0, transmon_d, qubit_temp)

coupler_levels = 3
frequency_coupler = 5.45e9
anharmonicity_coupler = -90e6
t1_coupler = 10e-6
t2star_coupler = 1e-6
coupler_temp = 50e-3
phi_coupler = 3.0227
phi_0_coupler = 10.0
d_coupler = 0.0

c1 = chip.Transmon(
    name="C1",
    desc="Tunable coupler coupling Qubit 1 and Qubit 2",
    freq=Qty(value=frequency_coupler, min_val=2e9, max_val=8e9, unit="Hz 2pi"),
    anhar=Qty(value=anharmonicity_coupler, min_val=-380e6, max_val=-50e6, unit="Hz 2pi"),
    hilbert_dim=coupler_levels,
    t1=Qty(value=t1_coupler, min_val=1e-6, max_val=90e-6, unit="s"),
    t2star=Qty(value=t2star_coupler, min_val=1e-6, max_val=90e-6, unit="s"),
    temp=Qty(value=coupler_temp, min_val=0.0, max_val=0.12, unit="K"),
    phi=Qty(value=phi_coupler, max_val=5.0, min_val=0.0, unit="Wb"),
    phi_0=Qty(value=phi_0_coupler, max_val=11.0, min_val=9.0, unit="Wb"),
    d=Qty(value=d_coupler, max_val=0.1, min_val=-0.1, unit="")
)

# Couple the qubits to the coupler
coupling_strengths = [72.5e6, 71.5e6]
couplings = createChainCouplingsWithCouplers(coupling_strengths, qubits, [c1])

drives = createDrives(qubits)

# %% [markdown]
# Define the model using the Qubits, Couplers, Drives and Couplings

# %%
model = Mdl(
    qubits + [c1],
    drives + couplings
)
model.set_lindbladian(False)
model.set_dressed(False)

# %% [markdown]
# Generator

# %%
sim_res = 100e9
awg_res = 2e9
v2hz = 1e9

chain = ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"]
chains = {f"{d.name}": chain for d in drives}
chains["Q1"] = ["AWG", "DigitalToAnalog", "Response"]
chains["Q2"] = ["AWG", "DigitalToAnalog", "Response"]
chains["C1"] = ["AWG", "DigitalToAnalog", "Response"]
generator = Gnr(
    devices={
        "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
        "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
        "DigitalToAnalog": devices.DigitalToAnalog(
            name="dac", resolution=sim_res, inputs=1, outputs=1
        ),
        "Response": devices.Response(
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
        "fluxbias":devices.FluxTuning(
            name="fluxbias",
            inputs=1,
            outputs=1,
            params={
                "phi_0":Qty(value=10.0, min_val=9.0, max_val=11.0, unit="Wb"),
                "phi":Qty(value=0.0, min_val=0.0, max_val=5.0, unit="Hz 2pi"),
                "omega_0":Qty(value=7.172e9, min_val=6e9, max_val=10e9, unit="Hz 2pi"),
                "anhar":Qty(value=-250e6, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
                "d":Qty(value=0, min_val=-0.324, max_val=0.396, unit="")
            }
        )
    },
    chains=chains,
)
generator.devices["AWG"].enable_drag_2()

# %%
generator.chains

# %% [markdown]
# Define pulses and carriers used for applying the gates

# %%
t_final = 50e-9
sideband = 50e6
nodrive_env = pulse.Envelope(
    name="no_drive",
    params={
        "t_final": Qty(
            value=        t_final,
            min_val=0.5 * t_final,
            max_val=1.5 * t_final,
            unit="s"
        )
    },
    shape=envelopes.no_drive
)


params_flux = {
        "amp": Qty(
            value=2.0,
            min_val=0.1,
            max_val=5.0,
            unit="V"
        ),
        "t_up": Qty(
            value=2.0e-9,
            min_val=0.1e-9,
            max_val=5.0e-9,
            unit="s"
        ),
        "t_down": Qty(
            value=t_final + 0.01e-9,
            min_val=0.1e-9,
            max_val=t_final + 1.0e-9,
            unit="s"
        ),
        "risefall": Qty(
            value=1.0e-9,
            min_val=0.1e-9,
            max_val=5.0e-9,
            unit="s"
        ),
        'xy_angle': Qty(
            value=0.0,
            min_val=-0.5 * np.pi,
            max_val=2.5 * np.pi,
            unit='rad'
        ),
        'freq_offset': Qty(
            value=-sideband - 3e6 ,
            min_val=-56 * 1e6 ,
            max_val=-52 * 1e6 ,
            unit='Hz 2pi'
        ),
        'delta': Qty(
            value=-1,
            min_val=-5,
            max_val=3,
            unit=""
        ),
        't_final': Qty(
            value=t_final,
            min_val=0.5 * t_final,
            max_val=1.5 * t_final,
            unit="s"
        )
}

gauss_params = {
    'amp': Qty(value=0.0, min_val=0.0, max_val=3, unit="V"),
    't_final': Qty(value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"),
    'sigma': Qty(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
    'xy_angle': Qty(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit='rad'),
    'freq_offset': Qty(value=-sideband - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit='Hz 2pi'),
    'delta': Qty(value=-1, min_val=-5, max_val=3, unit="")
}


flux_pulse = pulse.Envelope(
                name = "flux",
                params = params_flux,
                shape = envelopes.flattop
)

gaussian_pulse = pulse.Envelope(
    name="gauss",
    desc="Gaussian envelope",
    params=gauss_params,
    shape=envelopes.gaussian_nonorm
)

qubit_freqs = model.get_qubit_freqs()
carriers = createCarriers(qubit_freqs[:2], sideband)
carrier_coupler = createCarriers([qubit_freqs[-1]],sideband)[0]

# %% [markdown]
# Plot the variation of energy with external flux to find out the parameters for gate

# %%
def PlotAvoidedCrossingDiagram(flux_range, c_flux, q1_flux, q2_flux, tune_coupler, tune_q1, tune_q2):    
    model.set_dressed(True)
    model.update_model()
    
    c1.params["phi"] = Qty(value=c_flux, min_val=0.0, max_val=5.0, unit="Wb")
    qubits[0].params["phi"] = Qty(value=q1_flux, min_val=0.0, max_val=5.0, unit="Wb")
    qubits[1].params["phi"] = Qty(value=q2_flux, min_val=0.0, max_val=5.0, unit="Wb")
    model.update_model()

    states = [(1,0,0),(0,1,0),(0,0,1)]
    state_labels = model.get_state_indeces(states)
    energies = [[] for i in range(len(states))]
    freq_coupler_arr = []

    for i in flux_range:
        if tune_coupler:
            c1.params["phi"] = Qty(value=i, min_val=0.0, max_val=5.0, unit="Wb")
        elif tune_q1:
            qubits[0].params["phi"] = Qty(value=i, min_val=0.0, max_val=5.0, unit="Wb")
        elif tune_q2:
            qubits[1].params["phi"] = Qty(value=i, min_val=0.0, max_val=5.0, unit="Wb")
        else:
            print("Choose which component to tune")
            return 0
        
        model.update_model()
        eigen_energies = model.eigenframe.numpy()
        freq_coupler_arr.append(c1.get_freq()/(2*np.pi*1e9))
        for j in range(len(states)):
            energies[j].append(eigen_energies[state_labels[j]]/(2*np.pi*1e9))

    fig = go.Figure()
    for i in range(len(states)):
        fig.add_trace(go.Scatter(x=flux_range, y = energies[i], name = str(states[i]), mode = "lines"))
    fig.show()

    #plt.plot(flux_range, freq_coupler_arr)

    c1.params["phi"] = Qty(value=0.0, min_val=0.0, max_val=5.0, unit="Wb")
    qubits[0].params["phi"] = Qty(value=0.0, min_val=0.0, max_val=5.0, unit="Wb")
    qubits[1].params["phi"] = Qty(value=0.0, min_val=0.0, max_val=5.0, unit="Wb")
    model.update_model()

    model.set_dressed(False)
    model.update_model()

"""
PlotAvoidedCrossingDiagram(
    flux_range = np.linspace(0.0, 4.0, 100), 
    c_flux = 3.0465, 
    q1_flux = 1.2401, 
    q2_flux = 0.0, 
    tune_coupler = False, 
    tune_q1 = False, 
    tune_q2 = True
)
"""

# %%
print("Qubit1 Frequency =", qubits[0].get_freq(phi_sig=1.2401).numpy()/(2*np.pi*1e9))
print("Qubit2 Frequency =", qubits[1].get_freq(phi_sig=0.0).numpy()/(2*np.pi*1e9))
print("Coupler Frequency =", c1.get_freq(phi_sig=3.0465).numpy()/(2*np.pi*1e9))


# %% [markdown]
# Define the Gate using these parameters

# %%
qubit1_pulse = copy.deepcopy(gaussian_pulse)
qubit1_pulse.params["amp"] = Qty(value=0.0, min_val=0.0, max_val=5.0, unit="V")

qubit2_pulse = copy.deepcopy(gaussian_pulse)
qubit2_pulse.params["amp"] = Qty(value=0.0, min_val=0.0, max_val=5.0, unit="V")

qubit1_fluxpulse = copy.deepcopy(flux_pulse)
qubit1_fluxpulse.params["amp"] = Qty(value=1.2401, min_val=0.1, max_val=5.0, unit="V")
qubit1_fluxcarrier = copy.deepcopy(carriers[0])
qubit1_fluxcarrier.params["freq"] = Qty(value=0, min_val=0, max_val=8e9, unit="Hz 2pi")

qubit2_fluxpulse = copy.deepcopy(flux_pulse)
qubit2_fluxpulse.params["amp"] = Qty(value=0.0, min_val=0.0, max_val=5.0, unit="V")
qubit2_fluxcarrier = copy.deepcopy(carriers[1])
qubit2_fluxcarrier.params["freq"] = Qty(value=0, min_val=0, max_val=8e9, unit="Hz 2pi")


coupler_fluxpulse = copy.deepcopy(flux_pulse)
coupler_fluxpulse.params["amp"] = Qty(value=3.0465, min_val=0.1, max_val=5.0, unit="V")
coupler_fluxcarrier = copy.deepcopy(carrier_coupler)
coupler_fluxcarrier.params["freq"] = Qty(value=0, min_val=0, max_val=8e9, unit="Hz 2pi")

iswap = gates.Instruction(
    name = "iswap", targets = [0, 1], t_start = 0.0, t_end = t_final, channels=["Q1", "Q2", "C1", "d1", "d2"]
)

iswap.add_component(qubit1_pulse, "d1")
iswap.add_component(copy.deepcopy(carriers[0]), "d1")

iswap.add_component(qubit2_pulse, "d2")
iswap.add_component(copy.deepcopy(carriers[1]), "d2")

iswap.add_component(qubit1_fluxpulse, "Q1")
#iswap.add_component(qubit1_fluxcarrier, "Q1")

iswap.add_component(qubit2_fluxpulse, "Q2")
#iswap.add_component(copy.deepcopy(carriers[1]), "Q2")

iswap.add_component(coupler_fluxpulse,"C1")
#iswap.add_component(coupler_fluxcarrier, "C1")


two_qubit_gates = [iswap]

# %% [markdown]
# Define the experiment and compute the unitaries

# %%
parameter_map = PMap(instructions=two_qubit_gates, model=model, generator=generator)
parameter_map.load_values("test_0.7.txt")
exp = Exp(pmap=parameter_map)


model.set_dressed(False)
model.use_FR = False
exp.use_control_fields = False 

exp.set_opt_gates(['iswap[0, 1]'])
unitaries = exp.compute_propagators()

# %%
exp.write_config("DAQC_two_qubit_gates_50ns.hjson")

# %%
psi_init = [[0] * model.tot_dim]
psi_init[0][9] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ['iswap[0, 1]']
plotPopulation(exp, init_state, sequence, usePlotly=False, filename="Before_opt_iswap_50ns.png")

# %%
exp.write_config("DAQC_two_qubit_gates_50ns.hjson")
parameter_map.store_values("Two_qubit_gate_50ns.c3log")

# %% [markdown]
# ### Optimisation

# %%
print("----------------------------------------------")
print("-----------Starting optimal control-----------")
opt_gates = ['iswap[0, 1]']
parameter_map.set_opt_map([
    [('iswap[0, 1]', "Q1", "flux", "amp")],
    [('iswap[0, 1]', "Q1", "flux", "t_up")],
    [('iswap[0, 1]', "Q1", "flux", "t_down")],
    [('iswap[0, 1]', "Q1", "flux", "risefall")],
    [('iswap[0, 1]', "Q1", "flux", "xy_angle")],
    [('iswap[0, 1]', "Q1", "flux", "freq_offset")],
    [('iswap[0, 1]', "Q1", "flux", "delta")],

    [('iswap[0, 1]', "Q2", "flux", "amp")],
    [('iswap[0, 1]', "Q2", "flux", "t_up")],
    [('iswap[0, 1]', "Q2", "flux", "t_down")],
    [('iswap[0, 1]', "Q2", "flux", "risefall")],
    [('iswap[0, 1]', "Q2", "flux", "xy_angle")],
    [('iswap[0, 1]', "Q2", "flux", "freq_offset")],
    [('iswap[0, 1]', "Q2", "flux", "delta")],

    [('iswap[0, 1]', "C1", "flux", "amp")],
    [('iswap[0, 1]', "C1", "flux", "t_up")],
    [('iswap[0, 1]', "C1", "flux", "t_down")],
    [('iswap[0, 1]', "C1", "flux", "risefall")],
    [('iswap[0, 1]', "C1", "flux", "xy_angle")],
    [('iswap[0, 1]', "C1", "flux", "freq_offset")],
    [('iswap[0, 1]', "C1", "flux", "delta")],

    [('iswap[0, 1]', "d1", "gauss", "amp")],
    [('iswap[0, 1]', "d1", "gauss", "sigma")],
    [('iswap[0, 1]', "d1", "gauss", "xy_angle")],
    [('iswap[0, 1]', "d1", "gauss", "freq_offset")],
    [('iswap[0, 1]', "d1", "gauss", "delta")],

    [('iswap[0, 1]', "d2", "gauss", "amp")],
    [('iswap[0, 1]', "d2", "gauss", "sigma")],
    [('iswap[0, 1]', "d2", "gauss", "xy_angle")],
    [('iswap[0, 1]', "d2", "gauss", "freq_offset")],
    [('iswap[0, 1]', "d2", "gauss", "delta")],
])


parameter_map.print_parameters()

opt = OptimalControl(
    dir_path="./output",
    fid_func=fidelities.unitary_infid_set,
    fid_subspace=["Q1", "Q2"],
    pmap=parameter_map,
    algorithm=algorithms.lbfgs,
    options={"maxfun": 250},
    run_name="iswap_trial_50ns"
)
exp.set_opt_gates(opt_gates)
opt.set_exp(exp)

# %%
opt.optimize_controls()
opt.current_best_goal

# %%
print(opt.current_best_goal)
print(parameter_map.print_parameters())


plotPopulation(exp, init_state, sequence, usePlotly=False, filename="After_opt_iswap_50ns.png")
parameter_map.store_values("Two_qubit_gates_50ns.c3log")

# %%
parameter_map.print_parameters()
print("----------------------------------------------")
print("-----------Finished optimal control-----------")
# %%



