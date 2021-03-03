import numpy as np
import tensorflow as tf
from typing import Any, Dict
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from c3.utils.tf_utils import (
    tf_super,
    tf_choi_to_chi,
    tf_abs,
    super_to_choi,
    tf_project_to_comp,
)
import pytest


@pytest.fixture()
def get_test_circuit() -> QuantumCircuit:
    """fixture for sample Quantum Circuit

    Returns
    -------
    QuantumCircuit
        A circuit with a Hadamard, a C-X
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture()
def get_bell_circuit() -> QuantumCircuit:
    """fixture for Quantum Circuit to make Bell
    State |11> + |00>

    Returns
    -------
    QuantumCircuit
        A circuit with a Hadamard, a C-X and 2 Measures
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture()
def get_bad_circuit() -> QuantumCircuit:
    """fixture for Quantum Circuit with
    unsupported operations

    Returns
    -------
    QuantumCircuit
        A circuit with a Conditional
    """
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    qc.x(q[0]).c_if(c, 0)
    qc.measure(q, c)
    return qc


@pytest.fixture()
def get_6_qubit_circuit() -> QuantumCircuit:
    """fixture for 6 qubit Quantum Circuit

    Returns
    -------
    QuantumCircuit
        A circuit with an X on qubit 1
    """
    qc = QuantumCircuit(6, 6)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    return qc


@pytest.fixture()
def get_result_qiskit() -> Dict[str, Dict[str, Any]]:
    """Fixture for returning sample experiment result

    Returns
    -------
    Dict[str, Dict[str, Any]]
            A dictionary of results for physics simulation and perfect gates
            A result dictionary which looks something like::

            {
            "name": name of this experiment (obtained from qobj.experiment header)
            "seed": random seed used for simulation
            "shots": number of shots used in the simulation
            "data":
                {
                "counts": {'0x9: 5, ...},
                "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                },
            "status": status string for the simulation
            "success": boolean
            "time_taken": simulation time of this single experiment
            }

    """
    # Result of physics based sim for applying X on qubit 0 in 6 qubits
    perfect_counts = {"110000": 1000}

    counts_dict = {
        "c3_qasm_perfect_simulator": perfect_counts,
    }
    return counts_dict


@pytest.fixture
def get_error_process():
    """Fixture for a constant unitary

    Returns
    -------
    np.array
        Unitary on a large Hilbert space that needs to be projected down correctly and
        compared to an ideal representation in the computational space.
    """
    U_actual = (
        1
        / np.sqrt(2)
        * np.array(
            [
                [1, 0, 0, -1.0j, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, -1.0j, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1.0j, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, -1.0j, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 45, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )

    lvls = [3, 3]
    U_ideal = (
        1
        / np.sqrt(2)
        * np.array(
            [[1, 0, -1.0j, 0], [0, 1, 0, -1.0j], [-1.0j, 0, 1, 0], [0, -1.0j, 0, 1]]
        )
    )
    Lambda = tf.matmul(
        tf.linalg.adjoint(tf_project_to_comp(U_actual, lvls, to_super=False)), U_ideal
    )
    return Lambda


@pytest.fixture
def get_average_fidelitiy(get_error_process):
    lvls = [3, 3]
    Lambda = get_error_process
    d = 4
    err = tf_super(Lambda)
    choi = super_to_choi(err)
    chi = tf_choi_to_chi(choi, dims=lvls)
    fid = tf_abs((chi[0, 0] / d + 1) / (d + 1))
    return fid
