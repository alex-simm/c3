#%%
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
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

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes
from c3.optimizers.optimalcontrol import OptimalControl

from plotting import *
from utilities_functions import *
#%%

t_final = 50e-9
ts = np.linspace(0, t_final, 100)

exp = Exp()
exp.read_config("DAQC_iswap_pwc.hjson")
parameter_map = exp.pmap
model = parameter_map.model
#parameter_map.load_values("best_point_open_loop_iswap.txt")
parameter_map.load_values("best_point_open_loop_pwc.txt")

"""
pmap_dict = parameter_map.asdict()

Q1_pulse = pmap_dict['iswap[0, 1]']["drive_channels"]["d1"]["gauss"]
Q2_pulse = pmap_dict['iswap[0, 1]']["drive_channels"]["d2"]["gauss"]
Q1_flux_pulse = pmap_dict['iswap[0, 1]']["drive_channels"]["Q1"]["flux"]
Q2_flux_pulse = pmap_dict['iswap[0, 1]']["drive_channels"]["Q2"]["flux"]
C1_flux_pulse = pmap_dict['iswap[0, 1]']["drive_channels"]["C1"]["flux"]


Q1_pulse_carrier = pmap_dict['iswap[0, 1]']["drive_channels"]["d1"]["carrier"]
Q2_pulse_carrier = pmap_dict['iswap[0, 1]']["drive_channels"]["d2"]["carrier"]

Q1_pulse_shape = Q1_pulse.shape(ts, Q1_pulse.params)
Q2_pulse_shape = Q2_pulse.shape(ts, Q2_pulse.params)


Q1_flux_pulse_shape = Q1_flux_pulse.shape(ts, Q1_flux_pulse.params)
Q2_flux_pulse_shape = Q2_flux_pulse.shape(ts, Q2_flux_pulse.params)
C1_flux_pulse_shape = C1_flux_pulse.shape(ts, C1_flux_pulse.params)


Q1_pulse_pwc_params = {
    'amp': Qty(value=Q1_pulse.params["amp"], min_val=0.0, max_val=5, unit="V"),
    "t_bin_start": Qty(value=0.0, min_val=0.0, max_val=1e-9, unit="s"),
    "t_bin_end": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "inphase": Qty(value=Q1_pulse_shape, min_val=0.0, max_val=5.0, unit=""),
    "t_final": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "xy_angle": Qty(value=Q1_pulse.params["xy_angle"], min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Qty(value=Q1_pulse.params["freq_offset"],min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(Q1_pulse.params["delta"],min_val=-5,max_val=3,unit="")
}


Q1_pwc_pulse = pulse.Envelope(
    name="pwc_pulse",
    desc="PWC pulse for Qubit 1",
    params=Q1_pulse_pwc_params,
    shape=envelopes.pwc_shape
)

Q2_pulse_pwc_params = {
    'amp': Qty(value=Q2_pulse.params["amp"], min_val=0.0, max_val=5, unit="V"),
    "t_bin_start": Qty(value=0.0, min_val=0.0, max_val=1e-9, unit="s"),
    "t_bin_end": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "inphase": Qty(value=Q2_pulse_shape, min_val=0.0, max_val=5.0, unit=""),
    "t_final": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "xy_angle": Qty(value=Q2_pulse.params["xy_angle"], min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Qty(value=Q2_pulse.params["freq_offset"],min_val=-56 * 1e6,max_val=0,unit="Hz 2pi"),
    "delta": Qty(Q2_pulse.params["delta"],min_val=-5,max_val=3,unit="")
}


Q2_pwc_pulse = pulse.Envelope(
    name="pwc_pulse",
    desc="PWC pulse for Qubit 2",
    params=Q2_pulse_pwc_params,
    shape=envelopes.pwc_shape
)

Q1_flux_pulse_pwc_params = {
    'amp': Qty(value=Q1_flux_pulse.params["amp"], min_val=0.0, max_val=5, unit="V"),
    "t_bin_start": Qty(value=0.0, min_val=0.0, max_val=1e-9, unit="s"),
    "t_bin_end": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "inphase": Qty(value=Q1_flux_pulse_shape, min_val=0.0, max_val=5.0, unit=""),
    "t_final": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "xy_angle": Qty(value=Q1_flux_pulse.params["xy_angle"], min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Qty(value=Q1_flux_pulse.params["freq_offset"],min_val=-56 * 1e6,max_val=0,unit="Hz 2pi"),
    "delta": Qty(Q1_flux_pulse.params["delta"],min_val=-5,max_val=3,unit="")
}


Q1_flux_pwc_pulse = pulse.Envelope(
    name="pwc_pulse",
    desc="PWC Flux pulse for Qubit 1",
    params=Q1_flux_pulse_pwc_params,
    shape=envelopes.pwc_shape
)


Q2_flux_pulse_pwc_params = {
    'amp': Qty(value=Q2_flux_pulse.params["amp"], min_val=0.0, max_val=5, unit="V"),
    "t_bin_start": Qty(value=0.0, min_val=0.0, max_val=1e-9, unit="s"),
    "t_bin_end": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "inphase": Qty(value=Q2_flux_pulse_shape, min_val=0.0, max_val=5.0, unit=""),
    "t_final": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "xy_angle": Qty(value=Q2_flux_pulse.params["xy_angle"], min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Qty(value=Q2_flux_pulse.params["freq_offset"],min_val=-56 * 1e6,max_val=0,unit="Hz 2pi"),
    "delta": Qty(Q2_flux_pulse.params["delta"],min_val=-5,max_val=3,unit="")
}


Q2_flux_pwc_pulse = pulse.Envelope(
    name="pwc_pulse",
    desc="PWC Flux pulse for Qubit 2",
    params=Q2_flux_pulse_pwc_params,
    shape=envelopes.pwc_shape
)

C1_flux_pulse_pwc_params = {
    'amp': Qty(value=C1_flux_pulse.params["amp"], min_val=0.0, max_val=5, unit="V"),
    "t_bin_start": Qty(value=0.0, min_val=0.0, max_val=1e-9, unit="s"),
    "t_bin_end": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "inphase": Qty(value=C1_flux_pulse_shape, min_val=0.0, max_val=5.0, unit=""),
    "t_final": Qty(value=t_final, min_val=1e-9, max_val=t_final+10e-9, unit="s"),
    "xy_angle": Qty(value=C1_flux_pulse.params["xy_angle"], min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Qty(value=C1_flux_pulse.params["freq_offset"],min_val=-56 * 1e6,max_val=0,unit="Hz 2pi"),
    "delta": Qty(C1_flux_pulse.params["delta"],min_val=-5,max_val=3,unit="")
}


C1_flux_pwc_pulse = pulse.Envelope(
    name="pwc_pulse",
    desc="PWC Flux pulse for Coupler",
    params=C1_flux_pulse_pwc_params,
    shape=envelopes.pwc_shape
)

iswap = gates.Instruction(
    name = "iswap", targets = [0, 1], t_start = 0.0, t_end = t_final, channels=["Q1", "Q2", "C1", "d1", "d2"]
)

iswap.add_component(Q1_pwc_pulse, "d1")
iswap.add_component(Q1_pulse_carrier, "d1")

iswap.add_component(Q2_pwc_pulse, "d2")
iswap.add_component(Q2_pulse_carrier, "d2")

iswap.add_component(Q1_flux_pwc_pulse, "Q1")

iswap.add_component(Q2_flux_pwc_pulse, "Q2")

iswap.add_component(C1_flux_pwc_pulse,"C1")


two_qubit_gates = [iswap]

generator = parameter_map.generator
parameter_map = PMap(instructions=two_qubit_gates, model=model, generator=generator)
"""

exp = Exp(pmap=parameter_map)


model.set_dressed(False)
model.use_FR = False
exp.use_control_fields = False 

exp.set_opt_gates(['iswap[0, 1]'])
unitaries = exp.compute_propagators()
#%%
#exp.write_config("DAQC_iswap_pwc.hjson")

#%%
psi_init = [[0] * model.tot_dim]
psi_init[0][9] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ['iswap[0, 1]']
plotPopulation(exp, init_state, sequence, usePlotly=False, filename="Before_opt_iswap_PWC2.png")



print("----------------------------------------------")
print("-----------Starting optimal control-----------")
opt_gates = ['iswap[0, 1]']
parameter_map.set_opt_map([
    [('iswap[0, 1]', "Q1", "pwc_pulse", "inphase")],
    [('iswap[0, 1]', "Q2", "pwc_pulse", "inphase")],
    [('iswap[0, 1]', "C1", "pwc_pulse", "inphase")],
    [('iswap[0, 1]', "d1", "pwc_pulse", "inphase")],
    [('iswap[0, 1]', "d2", "pwc_pulse", "inphase")],
])


parameter_map.print_parameters()

opt = OptimalControl(
    dir_path="./output",
    fid_func=fidelities.unitary_infid_set,
    fid_subspace=["Q1", "Q2"],
    pmap=parameter_map,
    algorithm=algorithms.lbfgs,
    options={"maxfun": 250},
    run_name="iswap_trial_PWC"
)
exp.set_opt_gates(opt_gates)
opt.set_exp(exp)
opt.optimize_controls()
print(opt.current_best_goal)
print(parameter_map.print_parameters())


plotPopulation(exp, init_state, sequence, usePlotly=False, filename="After_opt_iswap_PWC2.png")
plotSplittedPopulation(exp, init_state, sequence, filename="After_opt_split_iswap_PWC2.png")


parameter_map.store_values("Two_qubit_gates_PWC2.c3log")

print("----------------------------------------------")
print("-----------Finished optimal control-----------")


# %%
