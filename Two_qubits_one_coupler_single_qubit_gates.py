#%%
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

#%matplotlib widget

import plotly.express as px
import plotly.graph_objects as go

# %%

qubit_levels = 3
freq_q1 = 5e9
anharm_q1 = -210e6
t1_q1 = 27e-6
t2star_q1 = 39e-6
qubit_temp = 50e-3
phi_1 = 0.1
phi_0_1 = 10.0
d_1 = 0.0


q1 = chip.Transmon(
        name = "Q1",
        desc = "Qubit  1",
        freq = Qty(
                value = freq_q1,
                min_val = 4.995e9,
                max_val = 5.005e9,
                unit = "Hz 2pi"
        ),
        anhar = Qty(
                value = anharm_q1,
                min_val = -380e6,
                max_val = -120e6,
                unit = "Hz 2pi"
        ),
        hilbert_dim = qubit_levels,
        t1 = Qty(
                value = t1_q1,
                min_val = 1e-6,
                max_val = 90e-6,
                unit = "s"
        ),
        t2star=Qty(
                value=t2star_q1,
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
                value=phi_1,
                max_val=5.0,
                min_val=0.0,
                unit = "Wb"
        ),
        phi_0 = Qty(
            value = phi_0_1,
            max_val = 11.0,
            min_val = 9.0,
            unit = "Wb"
        ),
        d = Qty(
            value=d_1,
            max_val=0.1,
            min_val=-0.1,
            unit = ""
        )
)

# %%

qubit_levels = 3
freq_q2 = 5.6e9
anharm_q2 = -240e6
t1_q2 = 23e-6
t2star_q2 = 31e-6
qubit_temp = 50e-3
phi_2 = 0.1  
phi_0_2 = 10.0
d_2 = 0.0

q2 = chip.Transmon(
        name = "Q2",
        desc = "Qubit  2",
        freq = Qty(
                value = freq_q2,
                min_val = 5.595e9,
                max_val = 5.605e9,
                unit = "Hz 2pi"
        ),
        anhar = Qty(
                value = anharm_q2,
                min_val = -380e6,
                max_val = -120e6,
                unit = "Hz 2pi"
        ),
        hilbert_dim = qubit_levels,
        t1 = Qty(
                value = t1_q2,
                min_val = 1e-6,
                max_val = 90e-6,
                unit = "s"
        ),
        t2star=Qty(
                value=t2star_q2,
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
                value=phi_2,
                max_val=5.0,
                min_val=0.0,
                unit = "Wb"
        ),
        phi_0 = Qty(
            value = phi_0_2,
            max_val = 11.0,
            min_val = 9.0,
            unit = "Wb"
        ),
        d = Qty(
            value=d_2,
            max_val=0.1,
            min_val=-0.1,
            unit = ""
        )        
)

# %%

qubit_levels = 3
freq_coupler = 4e9
anharm_coupler = -200e6
t1_coupler = 20e-6
t2star_coupler = 30e-6
qubit_temp = 50e-3
phi_coupler = 0.1  
phi_0_coupler = 10.0
d_c = 0.0

c1 = chip.Transmon(
        name = "C1",
        desc = "Tunable coupler coupling Qubit 1 and Qubit 2",
        freq = Qty(
                value = freq_coupler,
                min_val = 3.995e9,
                max_val = 4.005e9,
                unit = "Hz 2pi"
        ),
        anhar = Qty(
                value = anharm_coupler,
                min_val = -380e6,
                max_val = -120e6,
                unit = "Hz 2pi"
        ),
        hilbert_dim = qubit_levels,
        t1 = Qty(
                value = t1_coupler,
                min_val = 1e-6,
                max_val = 90e-6,
                unit = "s"
        ),
        t2star=Qty(
                value=t2star_coupler,
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
                value=phi_coupler,
                max_val=5.0,
                min_val=0.0,
                unit = "Wb"
        ),
        phi_0 = Qty(
            value = phi_0_coupler,
            max_val = 11.0,
            min_val = 9.0,
            unit = "Wb"
        ),
        d = Qty(
            value=d_c,
            max_val=0.1,
            min_val=-0.1,
            unit = ""
        )
)

# %%

coupling_strength_1 = 20e6
coupling_strength_2 = 25e6
q1c1 = chip.Coupling(
        name="Q1-C1",
        desc="Coupling",
        comment="Coupling between qubit 1 and Coupler",
        connected=["Q1","C1"],
        strength=Qty(
                value=coupling_strength_1,
                min_val=-1*1e3,
                max_val=200e6,
                unit="Hz 2pi"
        ),
        hamiltonian_func=hamiltonians.int_XX
)

c1q2 = chip.Coupling(
        name="C1-Q2",
        desc="Coupling",
        comment="Coupling between qubit 2 and Coupler",
        connected=["C1","Q2"],
        strength=Qty(
                value=coupling_strength_2,
                min_val=-1*1e3,
                max_val=200e6,
                unit="Hz 2pi"
        ),
        hamiltonian_func=hamiltonians.int_XX
)

# %%


drive1 = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment = "Drive line 1 on qubit 1",
        connected = ["Q1"],
        hamiltonian_func = hamiltonians.x_drive
)

drive2 = chip.Drive(
        name="d2",
        desc="Drive 2",
        comment = "Drive line 2 on qubit 2",
        connected = ["Q2"],
        hamiltonian_func = hamiltonians.x_drive
)
""""
drive_coupler = chip.Drive(
        name = "dc",
        desc = "Drive 1",
        comment = "Drive line on the coupler",
        connected = ["C1"],
        hamiltonian_func = hamiltonians.x_drive
)
"""

# %%

m00_q1 = 0.97
m01_q1 = 0.04
m00_q2 = 0.96
m01_q2 = 0.05
one_zeros = np.array([0] * qubit_levels)
zero_ones = np.array([1] * qubit_levels)
one_zeros[0] = 1
zero_ones[0] = 0

val1 = one_zeros * m00_q1 + zero_ones * m01_q1
val2 = one_zeros * m00_q2 + zero_ones * m01_q2

min_val = one_zeros * 0.8 + zero_ones * 0.0
max_val = one_zeros * 1.0 + zero_ones * 0.2

confusion_row1 = Qty(value=val1, min_val=min_val, max_val=max_val, unit="")
confusion_row2 = Qty(value=val2, min_val=min_val, max_val=max_val, unit="")

conf_matrix = tasks.ConfusionMatrix(Q1=confusion_row1, Q2=confusion_row2)


# %%

init_temp = 50e-3
init_ground = tasks.InitialiseGround(
        init_temp = Qty(
                value = init_temp,
                min_val = -0.001,
                max_val = 0.22,
                unit = "K"
        )
)

# %%

model = Mdl(
        [q1,c1,q2],
        [drive1, drive2, q1c1, c1q2],
        [conf_matrix, init_ground]
)
model.set_lindbladian(False)
model.set_dressed(True)

# %%
sim_res = 100e9
awg_res = 2e9
lo = devices.LO(name="lo", resolution=sim_res)
awg = devices.AWG(name = "awg", resolution=awg_res)
mixer = devices.Mixer(name = "mixer")

#%%

current_flux_1 = 4.1
current_flux_2 = 4.1
current_flux_c = 4.1

flux_line_params_1 = {
        "phi_0" : Qty(
                    value =   10.0,
                    min_val = 9.0,
                    max_val = 11.0,
                    unit = ""
        ),
        "phi" : Qty(
                value = current_flux_1,
                min_val = 0.0,
                max_val = 5.0,
                unit = ""
        ),
        "omega_0" : Qty(
                value =  freq_q1,
                min_val= freq_q1*0.95,
                max_val= freq_q1*1.05,
                unit = "Hz 2pi"
        ),
        "anhar" : Qty(
                value = anharm_q1,
                min_val = -380e6,
                max_val = -120e6,
                unit = "Hz 2pi"
        ),
        "d" : Qty(
                value = 0,
                min_val = -0.324,
                max_val = 0.396,
                unit = ""
        )
}

flux_line_params_2 = {
        "phi_0" : Qty(
                    value =   10.0,
                    min_val = 9.0,
                    max_val = 11.0,
                    unit = ""
        ),
        "phi" : Qty(
                value = current_flux_2,
                min_val = 0.0,
                max_val = 5.0,
                unit = ""
        ),
        "omega_0" : Qty(
                value =  freq_q2,
                min_val= freq_q2*0.95,
                max_val= freq_q2*1.05,
                unit = "Hz 2pi"
        ),
        "anhar" : Qty(
                value = anharm_q2,
                min_val = -380e6,
                max_val = -120e6,
                unit = "Hz 2pi"
        ),
        "d" : Qty(
                value = 0,
                min_val = -0.324,
                max_val = 0.396,
                unit = ""
        )
}


flux_line_params_c = {
        "phi_0" : Qty(
                    value =   10.0,
                    min_val = 9.0,
                    max_val = 11.0,
                    unit = ""
        ),
        "phi" : Qty(
                value = current_flux_c,
                min_val = 0.0,
                max_val = 5.0,
                unit = ""
        ),
        "omega_0" : Qty(
                value =  freq_coupler,
                min_val= freq_coupler*0.95,
                max_val= freq_coupler*1.05,
                unit = "Hz 2pi"
        ),
        "anhar" : Qty(
                value = anharm_coupler,
                min_val = -380e6,
                max_val = -120e6,
                unit = "Hz 2pi"
        ),
        "d" : Qty(
                value = 0,
                min_val = -0.324,
                max_val = 0.396,
                unit = ""
        )
}



flux_line_1 = devices.FluxTuning(
                name = "FL1",
                desc = "Flux line 1",
                params = flux_line_params_1,
                resolution = sim_res
)

flux_line_2 = devices.FluxTuning(
                name = "FL2",
                desc = "Flux line 2",
                params = flux_line_params_2,
                resolution = sim_res
)

flux_line_c = devices.FluxTuning(
                name = "FLC",
                desc = "Flux line for Coupler",
                params = flux_line_params_c,
                resolution = sim_res
)

# %%

resp = devices.Response(
        name = "resp",
        rise_time = Qty(
                value = 0.3e-9,
                min_val = 0.05e-9,
                max_val = 0.6e-9,
                unit = "s"
        ),
        resolution = sim_res

)

# %%
dig_to_an = devices.DigitalToAnalog(
                name = "dac",
                resolution = sim_res
)

# %%
v2hz = 1e9

v_to_hz = devices.VoltsToHertz(
                name = "V_to_Hz",
                V_to_Hz = Qty(
                        value = v2hz,
                        min_val = 0.9e9,
                        max_val = 1.1e9,
                        unit = "Hz/V"
                )
)

# %%
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
                ),
                "FluxBias": devices.FluxTuning(
                        name = "FL",
                        desc = "Flux line",
                        params = flux_line_params_1,
                        resolution = sim_res
                )
        },
        
        chains = {
                "d1": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
                "d2": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"]
        }

)
# %%

t_final = 7e-9
sideband = 50e6
gauss_params_single = {
        "amp": Qty(
                value = 0.5,
                min_val = 0.2,
                max_val = 0.6,
                unit = "V"
        ),
        "t_final": Qty(
                value = t_final,
                min_val = 0.5 * t_final,
                max_val = 1.5 * t_final,
                unit = "s"
        ),
        "sigma": Qty(
                value = t_final/4,
                min_val = t_final/8,
                max_val = t_final/2,
                unit = "s"
        ),
        "xy_angle": Qty(
                value = 0.0,
                min_val = -0.5 * np.pi,
                max_val = 2.5 * np.pi,
                unit = "rad"
        ),
        "freq_offset": Qty(
                value = -sideband - 3e6,
                min_val = -56 * 1e6,
                max_val = -52 * 1e6,
                unit = "Hz 2pi"
        ),
        "delta": Qty(
                value = -1,
                min_val = -5,
                max_val = 3,
                unit = ""
        )
}


# %%

gauss_env_single = pulse.Envelope(
        name = "gauss",
        desc = "Gaussian comp for single qubit gates",
        params = gauss_params_single,
        shape = envelopes.gaussian_nonorm
)

# %%
nodrive_env = pulse.Envelope(
        name = "no_drive",
        params = {
                "t_final": Qty(
                        value = t_final,
                        min_val = 0.5 * t_final,
                        max_val = 1.5 * t_final,
                        unit = "s"
                )
        },
        shape = envelopes.no_drive
)
#%%
qubit_freqs = model.get_qubit_freqs()
# %%
lo_freq_q1 = qubit_freqs[0] + sideband
carrier_parameters = {
        "freq": Qty(
                value = lo_freq_q1,
                min_val = 1e9,
                max_val = 6e9,
                unit = "Hz 2pi"
        ),
        "framechange": Qty(
                value = 0.0,
                min_val = -np.pi,
                max_val = 3 * np.pi,
                unit = "rad"
        )
}

carr = pulse.Carrier(
        name = "carrier",
        desc = "Frequency of the local oscillator",
        params = carrier_parameters
)

# %%

lo_freq_q2 = qubit_freqs[2] + sideband
carr_2 = copy.deepcopy(carr)
carr_2.params["freq"].set_value(lo_freq_q2)


#%%
lo_freq_c1 = qubit_freqs[1] + sideband
carr_c1 = copy.deepcopy(carr)
carr_c1.params["freq"].set_value(lo_freq_c1)


# %%

rx90p_q1 = gates.Instruction(
        name = "rx90p", targets = [0], t_start = 0.0, t_end = t_final, channels=["d1","d2"]
)

rx90p_q2 = gates.Instruction(
        name ="rx90p", targets = [2], t_start = 0.0, t_end = t_final, channels = ["d1","d2"]
)

rx90p_q1.add_component(gauss_env_single, "d1")
rx90p_q1.add_component(carr, "d1")

rx90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
rx90p_q2.add_component(carr_2, "d2")


# %%


rx90p_q1.add_component(nodrive_env,"d2")
rx90p_q1.add_component(copy.deepcopy(carr_2), "d2")
rx90p_q1.comps["d2"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) * 2 * np.pi % (2*np.pi)
)


rx90p_q2.add_component(nodrive_env,"d1")
rx90p_q2.add_component(copy.deepcopy(carr), "d1")
rx90p_q2.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) * 2 * np.pi % (2*np.pi)
)



# %%

ry90p_q1 = copy.deepcopy(rx90p_q1)
ry90p_q1.name = "ry90p"
rx90m_q1 = copy.deepcopy(rx90p_q1)
rx90m_q1.name = "rx90m"
ry90m_q1 = copy.deepcopy(rx90p_q1)
ry90m_q1.name = "ry90m"
ry90p_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5*np.pi)
rx90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
ry90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5*np.pi)
single_q_gates = [rx90p_q1, ry90p_q1, rx90m_q1, ry90m_q1]

ry90p_q2 = copy.deepcopy(rx90p_q2)
ry90p_q2.name = "ry90p"
rx90m_q2 = copy.deepcopy(rx90p_q2)
rx90m_q2.name = "rx90m"
ry90m_q2 = copy.deepcopy(rx90p_q2)
ry90m_q2.name = "ry90m"
ry90p_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(0.5*np.pi)
rx90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(np.pi)
ry90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(1.5*np.pi)
single_q_gates.extend([rx90p_q2, ry90p_q2, rx90m_q2, ry90m_q2])
# %%
parameter_map = PMap(instructions=single_q_gates, model = model, generator = generator)
# %%
exp = Exp(pmap = parameter_map)
# %%
exp.set_opt_gates(['rx90p[0]', 'rx90p[2]'])
# %%
unitaries = exp.compute_propagators()
# %%

psi_init = [[0] * (qubit_levels**3)]
psi_init[0][0] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))

# %%
barely_a_seq = ['rx90p[0]']
# %%
def plot_dynamics(exp, psi_init, seq,filename, goal = -1):
        model = exp.pmap.model
        dUs = exp.partial_propagators
        psi_t = psi_init.numpy()
        pop_t = exp.populations(psi_t, model.lindbladian)
        for gate in seq:
                for du in dUs[gate]:
                        psi_t = np.matmul(du.numpy(), psi_t)
                        pops = exp.populations(psi_t, model.lindbladian)
                        pop_t = np.append(pop_t, pops, axis = 1)
        
        ts = exp.ts
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        """
        fig, axs = plt.subplots(1,1)
        axs.plot(ts/1e-9, pop_t.T)
        axs.grid(linestyle="--")
        axs.tick_params(
            direction="in", left=True, right=True, top=True, bottom=True
        )
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        plt.legend(model.state_labels)
        """
        #print(pop_t.T[:,0])
        legends = model.state_labels
        fig = go.Figure()
        for i in range(len(pop_t.T[0])):
            fig.add_trace(go.Scatter(x = ts/1e-9, y = pop_t.T[:,i], mode = "lines", name = str(legends[i]) ))
            
        fig.show()
        fig.write_html(filename+".html")

        pass



# %%
plot_dynamics(exp, init_state, barely_a_seq, "Before_opt_single_Trial2")
 
# %%
plot_dynamics(exp, init_state, barely_a_seq * 10, "Before_opt_10_Trial2")
# %%

generator.devices["AWG"].enable_drag_2()

# %%

opt_gates = ["rx90p[0]","rx90p[2]"]
gateset_opt_map = [
        [
                ("rx90p[0]", "d1", "gauss", "amp" ),
        ],
        [
                ("rx90p[0]", "d1", "gauss", "freq_offset" ),
        ],
        [
                ("rx90p[0]", "d1", "gauss", "xy_angle" ),
        ],
        [
                ("rx90p[0]", "d1", "gauss", "delta" ),
        ],
        [
                ("rx90p[0]", "d2", "carrier", "framechange" ),
        ],
        [
                ("rx90p[2]", "d2", "gauss", "amp" ),
        ],
        [
                ("rx90p[2]", "d2", "gauss", "freq_offset" ),
        ],
        [
                ("rx90p[2]", "d2", "gauss", "xy_angle" ),
        ],
        [
                ("rx90p[2]", "d2", "gauss", "delta" ),
        ],
        [
                ("rx90p[2]", "d1", "carrier", "framechange" ),
        ],
        [
                ("Q1", "phi"),
        ],
        [
                ("Q2", "phi"),
        ],
        [
                ("C1", "phi"),
        ],
]
parameter_map.set_opt_map(gateset_opt_map)

# %%

parameter_map.print_parameters()

# %%

from c3.optimizers.optimalcontrol import OptimalControl

# %%

import os
import tempfile

log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

opt = OptimalControl(
        dir_path = log_dir,
        fid_func = fidelities.average_infid_set,
        fid_subspace = ["Q1", "C1" , "Q2"],
        pmap = parameter_map,
        algorithm = algorithms.lbfgs,
        options = {"maxfun": 150},
        run_name = "better_x90"
)

# %%

exp.set_opt_gates(opt_gates)
opt.set_exp(exp)

# %%

opt.optimize_controls()

# %%

opt.current_best_goal
# %%
plot_dynamics(exp, init_state, barely_a_seq, "After_opt_single_Trial2")
# %%
plot_dynamics(exp, init_state, barely_a_seq * 5, "After_opt_5_Trial2")
# %%
parameter_map.print_parameters()

# %%
