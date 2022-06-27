# Libs and helpers

import c3.libraries.algorithms as algorithms
import c3.libraries.envelopes
import c3.utils.qt_utils as qt_utils
# Main C3 objects
from c3.libraries import constants
from c3.model import Model as Mdl
from c3.signal.pulse import EnvelopeDrag
from four_level_transmons.custom_envelopes import *
from four_level_transmons.utilities import *


def generateSignalFromConfig(config: dict, sim_res: float = 50e9, awg_res: float = 2e9, useDRAG: bool = False,
                   usePWC: bool = False, numPWCPieces: int = 50, t_final=None):
    # Main parameters
    qubit_levels = [4, 4]
    qubit_frequencies = [5e9, 4.5e9]
    anharmonicities = [-300e6, -250e6]
    t1s = [25e-6, 25e-6]
    t2stars = [35e-6, 35e-6]
    qubit_temps = 50e-3
    couplingStrength = 20e6

    # Create model and generator
    qubits = createQubits(qubit_levels, qubit_frequencies, anharmonicities, t1s, t2stars, qubit_temps)
    coupling = createChainCouplings([couplingStrength], qubits)
    drives = createDrives(qubits)
    model = Mdl(qubits, coupling + drives)
    model.set_lindbladian(False)
    model.set_dressed(True)
    model.set_FR(False)
    generator = createGenerator2LOs(drives, sim_res=sim_res, awg_res=awg_res, lowPassFrequency=None)

    # Copy carrier frequencies and envelopes from configuration
    envelopesForDrive = {d.name: [] for d in drives}
    carriersForDrive = {d.name: [] for d in drives}
    t_end = t_final
    for idx in [0, 1]:
        dstIdx = idx
        srcIdx = idx
        driveNameSrc = drives[srcIdx].name
        driveNameDst = drives[dstIdx].name
        stored_params_d = config["drive_channels"][driveNameSrc]
        for i in range(0, 2):
            env = copy.deepcopy(stored_params_d[f"envelope_{driveNameSrc}_{i + 1}"])
            if useDRAG and not isinstance(env, EnvelopeDrag):
                env = convertToDRAG(env)
            if usePWC:
                if env.shape != c3.libraries.envelopes.pwc:
                    print("convertToPWC")
                    env = convertToPWC(env, numPWCPieces)
                elif len(env.params["inphase"]) != numPWCPieces:
                    print("resamplePWC")
                    env = resamplePWC(env, numPWCPieces)
            if t_final is not None and env.params['t_final'].get_value() != t_final:
                env = scaleGaussianEnvelope(env, t_final / env.params['t_final'].get_value())
            else:
                t_end = env.params['t_final']
            env.name = f"envelope_{driveNameDst}_{i + 1}"
            envelopesForDrive[driveNameDst].append(env)

            carrier = copy.deepcopy(stored_params_d[f"carrier_{driveNameSrc}_{i + 1}"])
            carrier.name = f"carrier_{driveNameDst}_{i + 1}"
            carriersForDrive[driveNameDst].append(carrier)

    # Create a dummy gate
    gate = gates.Instruction(
        name="dummy",
        # name="unity",
        targets=[0, 1],
        t_start=0.0,
        t_end=t_end,
        channels=[d.name for d in drives],
        ideal=qt_utils.np_kron_n([constants.Id] * 4)
    )
    for drive in drives:
        for env in envelopesForDrive[drive.name]:
            gate.add_component(copy.deepcopy(env), drive.name)
        for carrier in carriersForDrive[drive.name]:
            gate.add_component(copy.deepcopy(carrier), drive.name)
    print("carrier: ", [[carrier.params["freq"] for carrier in carriers] for carriers in carriersForDrive.values()])
    print("amp: ", [[env.params["amp"] for env in envelopes] for envelopes in envelopesForDrive.values()])

    # Generate the signal
    return generator.generate_signals(gate)
