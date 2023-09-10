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



## TODO: add the following in the abstract(meaning oop) way 
# def pauli_x(qubit_index, coeff, n_qubits):
#     eye = np.eye((n_qubits)) # the i^th row of the identity matrix is the correct parameter for \sigma_i 
#     return qi.SparsePauliOp(Pauli((np.zeros(n_qubits), eye[qubit_index])), coeff)

# def pauli_z(qubit_index, coeff, n_qubits):
#     eye = np.eye((n_qubits)) # the i^th row of the identity matrix is the correct parameter for \sigma_i 
    # return qi.SparsePauliOp(Pauli(( eye[qubit_index], np.zeros(n_qubits))), coeff)

# def product_pauli_z(q1, q2, coeff, n_qubits):
#     eye = np.eye((n_qubits))
#     return qi.SparsePauliOp(Pauli((eye[q1], np.zeros(n_qubits))).dot( Pauli((eye[q2], np.zeros(n_qubits)))), coeff)

# Hc = sum([product_pauli_z(0,1,-1,2),pauli_z(0,-0.5,2)])


# def evolve(hamiltonian, angle):
#     return PauliEvolutionGate(hamiltonian, time = angle, synthesis=SuzukiTrotter(reps=3))
#     # return hamiltonian.evolve(None, angle, 'circuit', 1,
#     #                           quantum_registers=quantum_registers,
#     #                           expansion_mode='suzuki',
#     #                           expansion_order=3)

# def create_circuit(circuit_init, beta, gamma, Hc, Hm, p):
#     if isinstance(circuit_init, qiskit.circuit.quantumcircuit.QuantumCircuit):
#         circuit = QuantumCircuit.copy(circuit_init)
#         for i in range(p):
#             circuit.append(evolve(Hc, beta[i]),[0,1])
#             circuit.append(evolve(Hm,gamma[i]),[0,1])
#     return circuit



# def make_evaluate_circuit(Hc, Hm, circuit_init, p):
#     def evaluate_circuit(beta_gamma):
#         n = len(beta_gamma)//2
#         circuit = create_circuit(circuit_init, beta_gamma[:n], beta_gamma[n:], Hc, Hm, p)
#         if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
#             state = qi.Statevector(circuit)
#             return np.real((state.expectation_value(Hc)))
#     return evaluate_circuit



# evaluate_circuit = make_evaluate_circuit(Hc,Hm, circuit_init,p)

# result = minimize(evaluate_circuit, np.concatenate([beta, gamma]), method='CG')
# result 



# n_spins = 10
# n_samples = 1000
# h = {v: np.random.uniform(-2, 2) for v in range(n_spins)}
# J = {}
# for u, v in itertools.combinations(h, 2):
#     if np.random.random() < .05:
#         J[(u, v)] = np.random.uniform(-1, 1)
# model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
# sampler = dimod.SimulatedAnnealingSampler()
# temperature_0 = 1
# response = sampler.sample(model, beta_range=[1/temperature_0, 1/temperature_0], num_reads=n_samples)
# energies_0 = [solution.energy for solution in response.data()]
# temperature_1 = 10
# response = sampler.sample(model, beta_range=[1/temperature_1, 1/temperature_1], num_reads=n_samples)
# energies_1 = [solution.energy for solution in response.data()]
# temperature_2 = 100
# response = sampler.sample(model, beta_range=[1/temperature_2, 1/temperature_2], num_reads=n_samples)
# energies_2 = [solution.energy for solution in response.data()]



# def plot_probabilities(energy_samples, temperatures):
#     fig, ax = plt.subplots()
#     for i, (energies, T) in enumerate(zip(energy_samples, temperatures)):
#         probabilities = np.exp(-np.array(sorted(energies))/T)
#         Z = probabilities.sum()
#         probabilities /= Z
#         ax.plot(energies, probabilities, linewidth=3, label = "$T_" + str(i+1)+"$")
#     minimum_energy = min([min(energies) for energies in energy_samples])
#     maximum_energy = max([max(energies) for energies in energy_samples])
#     ax.set_xlim(minimum_energy, maximum_energy)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xlabel('Energy')
#     ax.set_ylabel('Probability')
#     ax.legend()
#     plt.show()

# plot_probabilities([energies_0, energies_1, energies_2], 
#                    [temperature_0, temperature_1, temperature_2])



# from qiskit_aer import Aer
# from qiskit.algorithms.minimum_eigensolvers import QAOA
# from qiskit.algorithms.optimizers import COBYLA
# from qiskit_aer.primitives import Sampler as AerSampler

# p = 1

# # qaoa = QAOA(ising_model, optimizer, p, operator_mode='matrix')

# sampler = AerSampler(backend_options={"method": "statevector"},
#                     run_options={"shots": 100, "seed": 42}
#                                                             )
# optimizer = COBYLA()
# qaoa = QAOA(sampler, optimizer, reps=p)

# result = qaoa.compute_minimum_eigenvalue(ising_model)
# print(result)
# # backend = get_aer_backend('statevector_simulator')
# # quantum_instance = QuantumInstance(backend, shots=100)
# # result = qaoa.run(quantum_instance)
# result.eigenstate
# qi.Pauli.to




# from sklearn.linear_model import Perceptron
# model_1 = Perceptron(max_iter=1000, tol=1e-3)
# model_1.fit(X_train, y_train)
# print('accuracy (train): %5.2f'%(metric(y_train, model_1.predict(X_train))))
# print('accuracy (test): %5.2f'%(metric(y_test, model_1.predict(X_test))))
# from sklearn.svm import SVC
# model_2 = SVC(kernel='rbf', gamma='auto')
# model_2.fit(X_train, y_train)
# print('accuracy (train): %5.2f'%(metric(y_train, model_2.predict(X_train))))
# print('accuracy (test): %5.2f'%(metric(y_test, model_2.predict(X_test))))
# from sklearn.ensemble import AdaBoostClassifier
# model_3 = AdaBoostClassifier(n_estimators=3)
# model_3.fit(X_train, y_train)
# print('accuracy (train): %5.2f'%(metric(y_train, model_3.predict(X_train))))
# print('accuracy (test): %5.2f'%(metric(y_test, model_3.predict(X_test))))
# models = [model_3, model_2, model_1]
# n_models = len(models)
# predictions = np.array([h.predict(X_train) for h in models], dtype=np.float64)
# λ = 1
# q = predictions @ predictions.T/(n_models ** 2)

# qii = len(X_train) / (n_models ** 2) + λ - 2 * predictions @ y_train/(n_models)

# q[np.diag_indices_from(q)] = qii
# Q = {}
# for i in range(n_models):
#     for j in range(i, n_models):
#         Q[(i, j)] = q[i, j]
# Q
# import dimod
# sampler = dimod.SimulatedAnnealingSampler()
# response = sampler.sample_qubo(Q, num_reads=10)
# weights = list(response.first.sample.values())
# def predict(models, weights, X):

#     n_data = len(X)
#     T = 0
#     y = np.zeros(n_data)
#     for i, h in enumerate(models):
#         y0 = weights[i] * h.predict(X)  # prediction of weak classifier
#         y += y0
#         T += np.sum(y0)
#     # print(T / (n_data*len(models)))
#     # print(y)
#     # TODO: why do we need this term?
#     y = np.sign(y - T / (n_data*len(models)))

#     return y
# print('accuracy (train): %5.2f'%(metric(y_train, predict(models, weights, X_train))))
# print('accuracy (test): %5.2f'%(metric(y_test, predict(models, weights, X_test))))
# weights
# h, J, offset = dimod.qubo_to_ising(Q)
# h,J,offset
# from qiskit.quantum_info import Pauli
# import qiskit.quantum_info as qi
# num_nodes = q.shape[0]
# pauli_list = []
# coef_list = []

# wp = np.zeros(num_nodes)
# for i in range(num_nodes):
#     vp = np.zeros(num_nodes)
#     vp[i] = 1
#     pauli_list.append(Pauli((vp, wp)))
#     coef_list.append(h[i])
#     for j in range(i+1, num_nodes):
#         if q[i, j] != 0:
#             vp = np.zeros(num_nodes)
#             vp[i] = 1
#             vp[j] = 1
#             pauli_list.append(Pauli((vp, wp)))
#             coef_list.append(-J[i,j])  # minus here to fix issues 
# ising_model = qi.SparsePauliOp(pauli_list,np.array(coef_list))
# ising_model
# # pauli_list
# from qiskit_aer import Aer
# from qiskit.algorithms.minimum_eigensolvers import QAOA
# from qiskit.algorithms.optimizers import COBYLA
# from qiskit_aer.primitives import Sampler as AerSampler

# p = 1

# # qaoa = QAOA(ising_model, optimizer, p, operator_mode='matrix')

# sampler = AerSampler(backend_options={"method": "statevector"},
#                     run_options={"shots": 100, "seed": 42}
#                                                             )
# optimizer = COBYLA()
# qaoa = QAOA(sampler, optimizer, reps=p)

# result = qaoa.compute_minimum_eigenvalue(ising_model)
# print(result)
# # backend = get_aer_backend('statevector_simulator')
# # quantum_instance = QuantumInstance(backend, shots=100)
# # result = qaoa.run(quantum_instance)
# # weights = [ 1-int(a) for a in result.best_measurement['bitstring']][::-1]
# weights = [int(a) for a in result.best_measurement['bitstring']]

# # k = np.argmax(result['eigvecs'][0])
# # k = np.argmax(result.eigenstate)
# # k
# # weights = np.zeros(num_nodes)
# # for i in range(num_nodes):
# #     weights[i] = k % 2
# #     k >>= 1
# weights
# print('accuracy (train): %5.2f'%(metric(y_train, predict(models, weights, X_train))))
# print('accuracy (test): %5.2f'%(metric(y_test, predict(models, weights, X_test))))






############################ instead ################    https://qiskit.org/documentation/stable/0.28/aqua_tutorials/Qiskit%20Algorithms%20Migration%20Guide.html
# from qiskit.aqua import get_aer_backend, QuantumInstance
# from qiskit.aqua.algorithms import QAOA
# from qiskit.aqua.components.optimizers import COBYLA
# from qiskit.aqua.translators.ising import max_cut


#from qiskit import Aer
# from qiskit.utils import QuantumInstance
# from qiskit.algorithms import QAOA
# from qiskit.algorithms.optimizers import COBYLA
# qiskit.optimization.applications.ising.max_cut