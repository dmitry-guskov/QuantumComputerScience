import itertools
import numpy as np
import scipy
import socket
import subprocess
import time
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
import qiskit.quantum_info as qi
from  qiskit_aer import Aer
from qiskit_aer import AerSimulator, StatevectorSimulator, QasmSimulator


def get_amplitudes(circuit):
    if not isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        raise ValueError("Unknown circuit type")
    return qi.Statevector(circuit)

def get_counts(circuit, num_shots=100):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        backend = AerSimulator()
        job = execute(circuit, backend, shots=num_shots)
        result = job.result()
        counts = result.get_counts(circuit)
    else:
        raise ValueError("Unknown circuit type")
    return counts



def get_amplitudes(circuit):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        backend = StatevectorSimulator()
        job = execute(circuit, backend)
        amplitudes = job.result().get_statevector(circuit)
    else:
        raise ValueError("Unknown circuit type")
    return amplitudes


def get_single_measurement_counts(circuit, num_shots=100):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        backend = QasmSimulator()
        job = execute(circuit, backend, shots=num_shots)
        result = job.result()
        counts = result.get_counts(circuit)
    else:
        raise ValueError("Unknown circuit type")
    return counts


def get_classical_bits(circuit):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        classical_bits = circuit.cregs[0].size
    else:
        raise ValueError("Unknown circuit type")
    return classical_bits


def get_circuit_length(circuit):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        program_length = sum(circuit.count_ops().values())
    else:
        raise ValueError("Unknown circuit type")
    return program_length


def my_kron(mats):
    ans = np.kron(mats[-2],mats[-1])
    for i in range(len(mats)-2,0,-1):
        ans = np.kron(mats[i],ans)
    return ans
def get_gap(H):
    eigh = np.linalg.eigvals(H)
    gap = eigh[-1] - eigh[-2]
    print(gap,eigh)
    return gap

