# Defined functions

import argparse
import time
import random
import itertools
import numpy as np
import cvxpy as cp
import math
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from qutip import *
from qiskit import *
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, Operator, Pauli, partial_trace, state_fidelity, random_density_matrix
from qiskit.visualization import plot_histogram, plot_state_city, plot_bloch_multivector, plot_state_paulivec, plot_state_hinton, plot_state_qsphere
from qiskit.tools.monitor import job_monitor
from typing import TYPE_CHECKING, List, Dict, Any, Union, Iterable

import os
#os.environ["OMP_NUM_THREADS"] = "10"


#----------------------------------------------------------------------------------------------------------------------
# Generate measurement dataset
def meas(qubits_meas_basis:List[int], state:DensityMatrix, num_meas:int, N:int) -> List[str]:
    ''' Given a Pauli basis (0-Z, 1-X, 2-Y), do measurement and return its outcome
    Args:
        qubits_meas_basis: A list representing measuring basis, e.g.: [0,0,0] is 'ZZZ'
        state: A quantum state from Qiskit
        num_meas: number of measurements performed in this basis
    Yeilds:
        outcome: A list of strings, of which each element is an instance of measurement
    E.g.:
        INPUT: [0,0,0], state, N=3, num_meas=2
        OUTPUT: [000, 000] (in order of qubit 012)
    '''
    outcome = []
    circ_meas = QuantumCircuit(N)

    if num_meas>0:
        for i in range(N):
            if qubits_meas_basis[i] == '1':
                circ_meas.ry(-math.pi / 2, i)
            elif qubits_meas_basis[i] == '2':
                circ_meas.rx(math.pi / 2, i)

        U_meas = Operator(circ_meas)
        state_temp = state.evolve(U_meas)
        for j in range(num_meas):
            str_tmp = state_temp.measure()[0]
            outcome.append(str_tmp[::-1])  # Take the reverse

    # Note: in qiskit, qubit number counts from the left,
    # e.g.: '00101' means we measure qubit0 a '1'.
    return outcome

def number_to_Pauli(pauli_num_str:str, N:int) -> str:
    ''' Given a number string, return the corresponding Pauli string
        0-Z, 1-X, 2-Y
    E.g.:
        INPUT: '01200' (in order of qubit 01234)
        OUTPUT: 'ZZYXZ' (in order of qubit 01234)
    '''
    pauli_num_list = list(pauli_num_str)
    pauli_basis_list = list(pauli_num_str)
    for i in range(N):
        if pauli_num_list[i] == '1':
            pauli_basis_list[i] = 'X'
        elif pauli_num_list[i] == '2':
            pauli_basis_list[i] = 'Y'
        else:
            pauli_basis_list[i] = 'Z'
    return ''.join(pauli_basis_list)

def random_distribute(N_meas:int, N:int) -> List[int]:
    '''N_meas is the total number of measurements for all basis
    '''
    quotient = N_meas//3**N
    remainder = N_meas%3**N
    num_of_meas_list = quotient*np.ones(3**N)
    
    tmp = list(range(0,3**N))
    lucky_dog = random.sample(tmp, int(remainder))

    for i in range(remainder):
        num_of_meas_list[lucky_dog[i]] = num_of_meas_list[lucky_dog[i]]+1

    return num_of_meas_list

def generate_meas_dataset(state:DensityMatrix, N_meas:int, N:int) -> Dict[str,List[str]]:
    '''Generate measurement dataset for a N-qubit quantum state
    Args:
        state: A quantum state from Qiskit
        N_meas: total number of measurements for all basis
        N: number of qubits of the state
    Yeilds:
        Dict_meas_outcome
    '''
    Dict_meas_outcome = dict()
    num_meas_list = random_distribute(N_meas, N) # A list of integers, of which each element represent the number of measurement for one basis
    for i in range(3 ** N):
        qubits_meas_basis = tenToAny(i, N, 3)
        meas_outcome_string = meas(qubits_meas_basis, state, int(num_meas_list[i]), N)
        Dict_meas_outcome[number_to_Pauli(''.join(qubits_meas_basis), N)] = meas_outcome_string
    return Dict_meas_outcome

def tenToAny(origin:int, N:int, n:int) -> List[str]:
    # 10进制转换为n进制list
    list = []
    while True:
        s = origin // n
        tmp = origin % n
        list.append(tmp)
        if s == 0:
            break
        origin = s
    list.reverse()
    list = [str(each) for each in list]
    while len(list) < N:
        list.insert(0, '0')
    return list

def generate_PauliStrList(N:int) -> List[str]:
    ''' Given the number of qubits N, return its corresponding Pauli vector.
    E.g.:
        INPUT: N=2
        OUTPUT: ['II','IX',...'ZZ']
    '''
    Pauli_str_list = []
    for i in range(4 ** N):
        pauli_num_list = tenToAny(i, N, 4)
        pauli_basis_list = pauli_num_list
        for j in range(N):
            if pauli_num_list[j] == '0':
                pauli_basis_list[j] = 'I'
            elif pauli_num_list[j] == '1':
                pauli_basis_list[j] = 'X'
            elif pauli_num_list[j] == '2':
                pauli_basis_list[j] = 'Y'
            else:
                pauli_basis_list[j] = 'Z'
        Pauli_str_list.append(''.join(pauli_basis_list))

    return Pauli_str_list

def generate_sub_PauliStrList(N:int, index_list:List[int]) -> List[str]:
    # Less-complexity version
    ''' Given a index (list) of qubits, retrun the Pauli vectors of this sub system.
    E.g.:
        INPUT: PauliStrList=['III',...'ZZZ'], index=[0,2]
        OUTPUT: ['III','IIX','IIY','IIZ','XII','XIX',...'ZIZ']
    '''
    base_string = 'I' * N
    output_strings = []

    for combination in itertools.product('IXYZ', repeat=len(index_list)):
        if all(c == 'I' for c in combination):
            continue

        temp_string = list(base_string)
        for index, char in zip(index_list, combination):
            temp_string[index] = char

        output_strings.append(''.join(temp_string))

    return output_strings

def parity_check(meas_string:List[str]) -> List[int]:
    ''' Given a measurement outcome binary string array,
        return 0 if #1 in the string is even, otherwise return 1 for each element
    E.g.:
        INPUT: ['0001', '0101', '0000']
        OUTPUT: [-1, 1, 1]
    '''
    num_of_meas = len(meas_string)
    meas_parity = np.zeros(num_of_meas)
    for i in range(num_of_meas):
        temp = bin(int(meas_string[i], 2)).count("1")
        if temp % 2 == 0:
            meas_parity[i] = 1
        else:
            meas_parity[i] = -1
    return meas_parity

def exp_var_calculator(measurement_dataset:Dict[str,List[str]], pauli_basis_str:str) -> (float,float):
    ''' Given a Pauli basis (on partial qubits, e.g.: XIXZY, IIIXX, ZIIII, etc.) and dataset,
        return its applicable measurement outcome expectation value and variance.
    '''
    num_negatives = pauli_basis_str.count('-')
    phase = (-1) ** num_negatives
    cleaned_pauli_string = pauli_basis_str.replace('-', '')

    #measurement_dataset = {key: value for key, value in measurement_dataset.items() if value} # For reducing the complexity
    output = list([])
    for key in measurement_dataset:
        if cleaned_pauli_string.count('I') == sum(char1 != char2 for char1, char2 in zip(cleaned_pauli_string, key)):
            output = measurement_dataset[key] + output

    while cleaned_pauli_string.find('I') != -1:
        index_I = cleaned_pauli_string.find('I')
        cleaned_pauli_string = cleaned_pauli_string[:index_I] + cleaned_pauli_string[(index_I + 1):]
        for j in range(len(output)):
            words = output[j]
            output[j] = words[:index_I] + words[(index_I + 1):]

    meas_outcome = parity_check(output)
    N_meas_sub = len(output)
    
    if N_meas_sub == 0:
        expectation_value = 0
        variance = 0
    else: 
        expectation_value = np.average(meas_outcome)
        variance = np.var(meas_outcome)

    return phase*expectation_value, variance

def num_meas_sub_calculator(measurement_dataset:Dict[str,List[str]], pauli_basis_str:str) -> int:
    ''' Given a Pauli basis (on partial qubits, e.g.: XIXZY, IIIXX, ZIIII, etc.) and dataset,
        return the number of measurements performed in this basis
    '''
    output = list([])
    for key in measurement_dataset:
        if pauli_basis_str.count('I') == sum(char1 != char2 for char1, char2 in zip(pauli_basis_str, key)):
            output = measurement_dataset[key] + output

    while pauli_basis_str.find('I') != -1:
        index_I = pauli_basis_str.find('I')
        pauli_basis_str = pauli_basis_str[:index_I] + pauli_basis_str[(index_I + 1):]
        for j in range(len(output)):
            words = output[j]
            output[j] = words[:index_I] + words[(index_I + 1):]

    return len(output)

def pauliToMatrix(pauli_str:str) -> qutip.qobj.Qobj:
    '''Given a Pauli string basis (str),
       output its corresponding matrix representation (Qobj data).
    '''
    num_negatives = pauli_str.count('-')
    phase = (-1) ** num_negatives
    cleaned_pauli_string = pauli_str.replace('-', '')

    pauli_basis_list = list()
    for basis in cleaned_pauli_string:
        if basis == 'I':
            pauli_basis_list.append(qeye(2))
        elif basis == 'X':
            pauli_basis_list.append(sigmax())
        elif basis == 'Y':
            pauli_basis_list.append(sigmay())
        else:
            pauli_basis_list.append(sigmaz())
    return tensor(pauli_basis_list)*phase

def q_tomography_dm(qubit_index:List[int], measurement_dataset:Dict[str,List[str]], N:int) -> qutip.qobj.Qobj:
    ''' Do quantum tomography for certain qubits according to the index,
        output the constructed density matrix.
    '''
    density_matrix = 0
    for basis in generate_sub_PauliStrList(N, qubit_index):
        expectation, variance = exp_var_calculator(measurement_dataset, basis)
        sub_basis = ''.join([basis[i] for i in qubit_index])
        density_matrix += expectation * pauliToMatrix(sub_basis)
    density_matrix += tensor([qeye(2)] * len(qubit_index))
    return 1 / (2 ** len(qubit_index)) * density_matrix

def q_tomography_vec(qubit_index:List[int], measurement_dataset:Dict[str,List[str]], N:int) -> List[float]:
    ''' Do quantum tomography for certain qubits according to the index,
        output a list of expectation value.
    '''
    bloch_vec = []
    for basis in generate_sub_PauliStrList(N, qubit_index):
        expectation, variance = exp_var_calculator(measurement_dataset, basis)
        bloch_vec.append(expectation)
    return bloch_vec

def Wald_interval(qubit_index:List[int], confidence_level:float, 
                  measurement_dataset:Dict[str,List[str]], N:int) -> List[float]:
    ''' Given a qubit index (e.g. [0,1,2], in order of 01234...),
        return the corresponding Wald_interval for each expectation value
    '''
    error_rate = 1 - confidence_level
    z = norm.ppf(1 - error_rate / 2)  # Quantile of the input confidence level for binomial distribution
    mean_vec = np.array(q_tomography_vec(qubit_index, measurement_dataset, N))
    p_vec = 0.5 * (1 + mean_vec)

    basis_list = generate_sub_PauliStrList(N, qubit_index)
    sigma = []
    for i in range(len(basis_list)):
        num_meas_sub = num_meas_sub_calculator(measurement_dataset, basis_list[i])
        sigma.append(2 * z * ((p_vec[i] * (1 - p_vec[i]) / num_meas_sub) ** 0.5))

    return sigma

def Wald_interval_bisection(coef:float, qubit_index:List[int], confidence_level:float, 
                            measurement_dataset:Dict[str,List[str]], N:int) -> List[float]:
    ''' Given a qubit index (in order of 01234...),
        return the corresponding Wald_interval for each expectation value.
        But here "bisection" means we add an additional coefficient,
        so that we can use bisection method to find the solution of the SDP within a certain domain defined by a threshold.

        Wald method: CI is z*\sqrt(p(1-p)/n)
    '''
    error_rate = 1 - confidence_level
    z = norm.ppf(1 - error_rate / 2)  # Quantile of the input confidence level for binomial distribution
    mean_vec = np.array(q_tomography_vec(qubit_index, measurement_dataset, N))
    p_vec = 0.5 * (1 + mean_vec)

    basis_list = generate_sub_PauliStrList(N, qubit_index)
    sigma = []
    for i in range(len(basis_list)):
        num_meas_sub = num_meas_sub_calculator(measurement_dataset, basis_list[i])
        variance = p_vec[i] * (1 - p_vec[i]) # variance should be p(1-p)/n, here I don't divide for simplicity
        if num_meas_sub == 0:
            sigma.append(float(1*coef))
        if num_meas_sub != 0:
            if variance == 0:
                sigma.append(2 * coef * z * ((0.25/num_meas_sub)**0.5)) # since p(1-p) <= 0.25
            else:
                sigma.append(2 * coef * z * (variance/num_meas_sub)**0.5)
    return sigma, mean_vec

# def Wilson_interval_bisection(coef:float, qubit_index:List[int], confidence_level:float, 
#                             measurement_dataset:Dict[str,List[str]], N:int) -> List[float]:
#     ''' Given a qubit index (in order of 01234...),
#         return the corresponding Wilson_interval for each expectation value.
#         But here "bisection" means we add an additional coefficient,
#         so that we can use bisection method to find the solution of the SDP within a certain domain defined by a threshold
#     '''
#     error_rate = 1 - confidence_level
#     z = norm.ppf(1 - error_rate / 2)  # Quantile of the input confidence level for binomial distribution

#     mean_vec = np.array(q_tomography_vec(qubit_index, measurement_dataset, N))
#     p_vec = 0.5 * (1 + mean_vec)

#     basis_list = generate_sub_PauliStrList(N, qubit_index)
#     sigma = []
#     for i in range(len(basis_list)):
#         num_meas_sub = num_meas_sub_calculator(measurement_dataset, basis_list[i])
#         sigma.append(2*coef*z/(1+z*z/num_meas_sub)*math.sqrt((p_vec[i]*(1-p_vec[i]) + z*z/(4*num_meas_sub)) / num_meas_sub))
#     sigma = np.nan_to_num(sigma, nan=1)
#     return sigma

def Bloch_vec(qiskit_state:DensityMatrix, qubit_index:List[int], N:int) -> List[float]:
    ''' Given a qiskit quantum state and the qubit index,
        return the Bloch vector of the reduced state according to the index
    '''
    output = []
    for basis in generate_sub_PauliStrList(N, qubit_index):
        basis = basis[::-1]
        output.append(qiskit_state.expectation_value(oper=Pauli(basis), qargs=None))
    return output

def qubit_swap(N:int, state_43210:DensityMatrix) -> DensityMatrix:
    circSWAP = QuantumCircuit(N)
    for i in range(int(N / 2)):
        circSWAP.swap(i, N - 1 - i)
    U_SWAP = Operator(circSWAP)
    state_01234 = state_43210.evolve(U_SWAP)
    return state_01234

def generate_random_dm(purity:float, N:int) -> np.ndarray:
    '''Generate a random density matrix with a certain purity
    '''
    qiskit_state = DensityMatrix(random_statevector(2 ** N))
    PauliStrList = generate_PauliStrList(N)[1:]

    Bloch_vector = []
    for basis in PauliStrList:
        Bloch_vector.append(qiskit_state.expectation_value(oper=Pauli(basis), qargs=None))
    Bloch_vector_noisy = math.sqrt(((2 ** N) * purity - 1) / (2 ** N - 1)) * np.array(Bloch_vector)

    density_matrix = tensor([qeye(2)] * N)
    for i in range(4 ** N - 1):
        density_matrix += Bloch_vector_noisy[i] * pauliToMatrix(PauliStrList[i])
    return 1 / (2 ** N) * np.array(density_matrix)


#----------------------------------------------------------------------------------------------------------------------
# Hamiltonian and plot
def Hamiltonian_matrix(H:List[str], model_type:str) -> List[str]:
    '''Given a list of Pauli string for each subsystem,
       output a list of their matrix representation.
    '''
    Hamiltonian_matrix = 0
    for i in range(len(H)):
        Hamiltonian_matrix = Hamiltonian_matrix + (pauliToMatrix(H[i]))
    if model_type=='open':
        return Hamiltonian_matrix
    if model_type=='closed':
        return Hamiltonian_matrix

def Hamiltonian_global(H_local_list:List[str], N:int, M:int, K:int, model_type:str) -> List[str]:
    '''Given the Hamiltonian of local subsystem (list of Pauli strings)
       return the Hamiltonian of global system (list of Pauli strings)
    '''
    H_global = []
    if model_type=='open':
        for i in range(K):
            for h in H_local_list:
                H_global.append(i * 'I' + h + (N - M - i) * 'I')
    if model_type=='closed':
        for i in range(K-1):
            for h in H_local_list:
                H_global.append(i * 'I' + h + (N - M - i) * 'I')
        # Add the local Hamiltonian between the head and tail qubits (only works for M=2) 
        for h in H_local_list:
            H_global.append(h[1] + (N - M)*'I' + h[0])

    return H_global

def ground_state(H_matrix:np.ndarray) -> (float, np.ndarray):
    '''Given a matrix representation of a Hamiltonian,
       find the ground state energy, i.e. the minimum eigenvalue of the matrix,
       and the ground state density matrix
    '''
    H_matrix = np.array(H_matrix)
    eigenvalue, eigenvector = np.linalg.eigh(H_matrix)

    tmp = np.argsort(eigenvalue)
    ground_state_energy = eigenvalue[tmp[0]]
    ground_state_vec = np.array(eigenvector[:, tmp[0]])

    ground_state_dm = np.outer(ground_state_vec, np.conj(ground_state_vec))

    return ground_state_energy, ground_state_dm

def N_meas_list_func(start:int, end:int, num:int) -> List[int]:
    '''Generate a list of number of measurement for the loop
    '''
    a = pow(end / start, 1 / (num - 1))
    N_meas_list = [start]
    for i in range(num - 1):
        N_meas_list.append(math.floor(a * N_meas_list[-1]))

    return N_meas_list

# def gs_energy_estimate(measurement_dataset:Dict[str,List[str]], confidence_level:float, H_global_list:List[str]) -> (float,float):
#     '''Given the Pauli decomposition of the Hamiltonian of interest and measurement dataset
#        return the expectaion value of the Hamiltonian (with confidence interval)
#     '''
#     E_min = 0
#     E_max = 0
#     error_rate = 1 - confidence_level
#     z = norm.ppf(1 - error_rate / 2)  # Quantile of the input confidence level for binomial distribution
    
#     for pauli_basis_str in H_global_list:
#         exp, var = exp_var_calculator(measurement_dataset, pauli_basis_str)
#         num_meas_sub = num_meas_sub_calculator(measurement_dataset, pauli_basis_str)
#         p_value = 0.5 * (1 + exp)
#         sigma = z * ((p_value * (1 - p_value) / num_meas_sub) ** 0.5)
#         E_min = E_min + exp - 2*sigma
#         E_max = E_max + exp + 2*sigma

#     return E_min, E_max

def gs_energy_estimate(measurement_dataset:Dict[str,List[str]], 
                       confidence_level:float, 
                       H_global_list:List[str],
                       model_type:str) -> (float,float):
    '''Given the Pauli decomposition of the Hamiltonian of interest and measurement dataset
       return the expectaion value of the Hamiltonian (with confidence interval)
       This version is more rigorous.
    '''
    E_sum = 0
    var_sum = 0
    error_rate = 1 - confidence_level
    z = norm.ppf(1 - error_rate / 2)  # Quantile of the input confidence level for binomial distribution
    
    for pauli_basis_str in H_global_list:
        exp, var = exp_var_calculator(measurement_dataset, pauli_basis_str)
        num_meas_sub = num_meas_sub_calculator(measurement_dataset, pauli_basis_str)
        if model_type == 'open':
            E_sum = E_sum + exp
        if model_type == 'closed':
            E_sum = E_sum - exp
        var_sum = var_sum + var/num_meas_sub

    E_min = E_sum - 2.58*var_sum**0.5
    E_max = E_sum + 2.58*var_sum**0.5

    return E_min, E_max

def lower_bound_with_SDP(H:np.ndarray, 
                         N:int, M:int, G:int, K:int, P:int, 
                         PauliStrList_part:List[str], PauliStrList_Gbody:[List[str]],
                         model_type:str) -> float:
    '''Solve the SDP minimization problem with constraints C0 and C0+C1
    '''

    if model_type=='open':
        K_3body = N-G+1 # Number of 3-body subsystems
    if model_type=='closed':
        K_3body = K # Number of 3-body subsystems
    P_3body = 4**G-1 # Number of Pauli basis for 3-body subsystems
    ep = cp.Variable((K, P))
    ep_C1 = cp.Variable((K_3body, 4**G-1))


    # Define SDP variables---------------------------------------------------------------------------------------------
    # Physical: dm_tilde
    dm_tilde = []
    for k in range(K):
        dm_tilde.append( np.array(tensor([qeye(2)] * M)) / 2 **M )
    for k in range(K):
        for p in range(P):
            dm_tilde[k] = dm_tilde[k] + cp.multiply(ep[k, p], np.array(pauliToMatrix(PauliStrList_part[p])))
    # Global: dm_tilde_C1
    dm_tilde_C1 = []
    for k in range(K_3body):
        dm_tilde_C1.append( np.array(tensor([qeye(2)] * G)) / 2 ** G )
    for k in range(K_3body): 
        for p in range(P_3body):
            dm_tilde_C1[k] = dm_tilde_C1[k] + cp.multiply(ep_C1[k, p], np.array(pauliToMatrix(PauliStrList_Gbody[p])))

            
    # Define SDP constraints---------------------------------------------------------------------------------------------
    # Physical constraints
    constraints_C0 = []
    for i in range(K):  # non-negative eigenvalues
        constraints_C0 += [dm_tilde[i] >> 1e-8]
    if model_type=='open':
        for i in range(K - 1):  # physically compatitble
            constraints_C0 += [cp.partial_trace(dm_tilde[i], dims=[2] * M, axis=0) ==
                            cp.partial_trace(dm_tilde[i+1], dims=[2] * M, axis=M - 1)]
    if model_type=='closed':
        for i in range(K - 2):  # physically compatitble
            constraints_C0 += [cp.partial_trace(dm_tilde[i], dims=[2] * M, axis=0) ==
                            cp.partial_trace(dm_tilde[i+1], dims=[2] * M, axis=M - 1)]
        # Add the physically compatitble constraints between the head and tail local systems (only works for M=2)
        constraints_C0 += [cp.partial_trace(dm_tilde[K-2], dims=[2] * M, axis=0) ==
                            cp.partial_trace(dm_tilde[K-1], dims=[2] * M, axis=0)]
        constraints_C0 += [cp.partial_trace(dm_tilde[K-1], dims=[2] * M, axis=M-1) ==
                            cp.partial_trace(dm_tilde[0], dims=[2] * M, axis=M-1)]
    
    # Global constraints
    constraints_C1 = []
    for i in range(K_3body): 
        constraints_C1 += [dm_tilde_C1[i] >> 1e-8]  # non-negative eigenvalues
    if model_type=='open':
        for i in range(N-G+1):
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[i], dims=[4,2], axis=1) == dm_tilde[i]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[i], dims=[2,4], axis=0) == dm_tilde[i+1]]
    if model_type=='closed':
        for i in range(N-G+1):
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[i], dims=[4,2], axis=1) == dm_tilde[i]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[i], dims=[2,4], axis=0) == dm_tilde[i+1]]
        if K>=4: # Add the globally compatitble constraints between the head and tail local systems (only works for M=2)
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-2], dims=[2,4], axis=0) == dm_tilde[K-2]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-2], dims=[2,2,2], axis=1) == dm_tilde[K-1]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-1], dims=[4,2], axis=1) == dm_tilde[0]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-1], dims=[2,2,2], axis=1) == dm_tilde[K-1]]
            # (only works for M=2, G=3)


    # Solve SDP with conditions C1+C0---------------------------------------------------------------------------------------------
    H_exp01 = 0
    for i in range(K):
        H_exp01 = H_exp01 + H @ dm_tilde[i]
    prob_C01 = cp.Problem(
        cp.Minimize(
            cp.real(
                cp.trace(
                    H_exp01
                )
            )
        ), constraints_C0 + constraints_C1
    )
    energy_C01 = prob_C01.solve(solver=cp.SCS, verbose=False)

    return energy_C01


#----------------------------------------------------------------------------------------------------------------------
# SDP problem variables and constraints
def SDP_variables_C0(ep:cp.expressions.variable.Variable, 
                     measurement_dataset:Dict[str,List[str]], 
                     N:int, M:int, K:int, P:int, 
                     PauliStrList_part:List[str],
                     model_type:str) -> cp.atoms.affine.add_expr.AddExpression:
    '''Define SDP variables'''
    dm = []

    if model_type=='open':
        for k in range(K):  # K: number of subsystems
            index = list(range(k, k + M, 1))  # [k, k+1, ...]
            dm.append(np.array(q_tomography_dm(index, measurement_dataset, N)))
        dm_hat = dm.copy()
        dm_tilde = dm
        for k in range(K):
            for p in range(P):
                dm_tilde[k] = dm_tilde[k] + cp.multiply(ep[k, p], np.array(pauliToMatrix(PauliStrList_part[p])))
    
    if model_type=='closed':
        for k in range(K):  # K: number of subsystems
            index = [(k + i) % K for i in range(M)]  # [k, k+1, ...]
            dm.append(np.array(q_tomography_dm(index, measurement_dataset, N)))
        dm_hat = dm.copy()
        dm_tilde = dm
        for k in range(K):
            for p in range(P):
                dm_tilde[k] = dm_tilde[k] + cp.multiply(ep[k, p], np.array(pauliToMatrix(PauliStrList_part[p])))
    
    return dm_tilde, dm_hat

def constraints_C0(ep:cp.expressions.variable.Variable, 
                   coef:float, 
                   dm_tilde: cp.atoms.affine.add_expr.AddExpression, 
                   measurement_dataset:Dict[str,List[str]], 
                   N:int, M:int, K:int, P:int,
                   model_type:str) -> list:
    '''Define the constraints of the SDP for bisection method:
       1. non-negative eigenvalues
       2. physically compatitble
    '''
    constraints = []
    for i in range(K):  # non-negative eigenvalues
        constraints += [dm_tilde[i] >> 1e-8]

    if model_type=='open':
        for i in range(K - 1):  # physically compatitble
            constraints += [cp.partial_trace(dm_tilde[i], dims=[2] * M, axis=0) ==
                            cp.partial_trace(dm_tilde[i+1], dims=[2] * M, axis=M - 1)]
    if model_type=='closed':
        for i in range(K - 1):  # physically compatitble
            constraints += [cp.partial_trace(dm_tilde[i], dims=[2] * M, axis=0) ==
                            cp.partial_trace(dm_tilde[i+1], dims=[2] * M, axis=M - 1)]
        # Add the physically compatitble constraints between the head and tail local systems (only works for M=2)
        constraints += [cp.partial_trace(dm_tilde[K-1], dims=[2] * M, axis=0) ==
                            cp.partial_trace(dm_tilde[0], dims=[2] * M, axis=M - 1)]
        
    # if model_type=='closed':
    #     for i in range(K - 2):  # physically compatitble
    #         constraints += [cp.partial_trace(dm_tilde[i], dims=[2] * M, axis=0) ==
    #                         cp.partial_trace(dm_tilde[i+1], dims=[2] * M, axis=M - 1)]
    #     # Add the physically compatitble constraints between the head and tail local systems (only works for M=2)
    #     constraints += [cp.partial_trace(dm_tilde[K-2], dims=[2] * M, axis=0) ==
    #                         cp.partial_trace(dm_tilde[K-1], dims=[2] * M, axis=0)]
    #     constraints += [cp.partial_trace(dm_tilde[K-1], dims=[2] * M, axis=M-1) ==
    #                         cp.partial_trace(dm_tilde[0], dims=[2] * M, axis=M-1)]

    sigma = np.zeros((K, P))
    mean_vec = np.zeros((K, P))
    if model_type=='open':
        for i in range(K):
            index = list(range(i, i + M, 1))  # [i, i+1, ...]
            sigma[i], mean_vec[i] = Wald_interval_bisection(coef, index, 0.95, measurement_dataset, N)
        constraints += [ep >= -sigma, ep <= sigma]
        constraints += [ep+mean_vec >= -1, ep+mean_vec <= 1]
    if model_type=='closed':
        for i in range(K):
            index = [(i + j) % K for j in range(M)]  # [i, i+1, ...]
            sigma[i], mean_vec[i] = Wald_interval_bisection(coef, index, 0.95, measurement_dataset, N)
        constraints += [ep >= -sigma, ep <= sigma]
        constraints += [ep+mean_vec >= -1, ep+mean_vec <= 1]
  
    return constraints


def constraints_interval(ep:cp.expressions.variable.Variable,
                        coef:float, 
                        dm_tilde:cp.atoms.affine.add_expr.AddExpression, 
                        measurement_dataset:Dict[str,List[str]], 
                        N:int, M:int, K:int, P:int) -> list:
    '''Define the constraints of the SDP for bisection method:
       1. non-negative eigenvalues
    '''
    constraints = []
    sigma = np.zeros((K, P))
    for i in range(K):
        index = list(range(i, i + M, 1))  # [i, i+1, ...]
        sigma[i] = Wald_interval_bisection(coef, index, 0.95, measurement_dataset, N)
    constraints += [ep >= -sigma, ep <= sigma]

    return constraints

def SDP_variables_verify(ep_verify:cp.expressions.variable.Variable, 
                         measurement_dataset:Dict[str,List[str]], 
                         N:int, 
                         PauliStrList:List[str]) -> cp.atoms.affine.add_expr.AddExpression:
    '''Define the varibles of global verification problem
    '''
    dm_tilde_full = np.array(tensor([qeye(2)] * N)) / 2 ** N
    for i in range(4 ** N - 1):
        dm_tilde_full = dm_tilde_full + cp.multiply(ep_verify[i], np.array(pauliToMatrix(PauliStrList[i])))
    return dm_tilde_full

def constraints_verify(ep_verify:cp.expressions.variable.Variable, 
                       coef:float, 
                       dm_tilde:cp.atoms.affine.add_expr.AddExpression, 
                       dm_tilde_full:cp.atoms.affine.add_expr.AddExpression, 
                       measurement_dataset:Dict[str,List[str]], 
                       N:int, M:int, K:int) -> list:
    '''Define the constraints of global verification problem:
       1. non-negative eigenvalues
       2. there exists a global state whose reduced states are the corresponding subsystems' states
    '''
    constraints_verify = []
    constraints_verify += [dm_tilde_full >> 1e-8]  # non-negative eigenvalues

    # global verification
    constraints_verify += [cp.partial_trace(dm_tilde_full, dims=[2 ** M, 2 ** (N - M)], axis=1) == dm_tilde[0]]
    constraints_verify += [cp.partial_trace(dm_tilde_full, dims=[2 ** (N - M), 2 ** M], axis=0) == dm_tilde[-1]]
    # constraints_verify += [cp.partial_trace(cp.partial_trace(dm_tilde_full, dims=[2,4,2], axis=0), dims=[4,2], axis=1) == dm_tilde[1]]

    if K >= 3:
        for i in range(K - 2):
            constraints_verify += [cp.partial_trace(
                cp.partial_trace(dm_tilde_full, dims=[2 ** (i + 1), 2 ** M, 2 ** (N - M - i - 1)], axis=2),
                dims=[2 ** (i + 1), 2 ** M], axis=0) == dm_tilde[i + 1]]

    constraints_verify += [ep_verify >= -1, ep_verify <= 1]

    return constraints_verify

def SDP_variables_C1(ep_C1:cp.expressions.variable.Variable, 
                     measurement_dataset:Dict[str,List[str]], 
                     N:int, G:int, K:int,
                     PauliStrList_Gbody:List[str],
                     model_type:str) -> cp.atoms.affine.add_expr.AddExpression:
    '''Define SDP variables for C1
    '''
    if model_type=='open':
        K_3body = N-G+1 # Number of 3-body subsystems
    if model_type=='closed':
        if N==3:
            K_3body = 1
        else:
            K_3body = K # Number of 3-body subsystems
    P_3body = 4**G-1 # Number of Pauli basis for 3-body subsystems
    
    dm_tilde_C1 = []
    
    for k in range(K_3body):
        dm_tilde_C1.append( np.array(tensor([qeye(2)] * G)) / 2 ** G )

    for k in range(K_3body): 
        for p in range(P_3body):
            dm_tilde_C1[k] = dm_tilde_C1[k] + cp.multiply(ep_C1[k, p], np.array(pauliToMatrix(PauliStrList_Gbody[p])))
    
    return dm_tilde_C1

def constraints_C1(ep_C1:cp.expressions.variable.Variable, 
                   coef:float, 
                   dm_tilde:cp.atoms.affine.add_expr.AddExpression, 
                   dm_tilde_C1:cp.atoms.affine.add_expr.AddExpression, 
                   measurement_dataset:Dict[str,List[str]], 
                   N:int, M:int, G:int, K:int,
                   model_type:str) -> list:
    '''Define the constraints of 3-body SDP problem (C1):
       1. non-negative eigenvalues
       2. there exists a 3-body global state whose reduced states are the corresponding subsystems' states
    '''
    if model_type=='open':
        K_3body = N-G+1 # Number of 3-body subsystems
    if model_type=='closed':
        if N==3:
            K_3body = 1
        else:
            K_3body = K # Number of 3-body subsystems
    P_3body = 4**G-1 # Number of Pauli basis for 3-body subsystems

    constraints_C1 = []
    for i in range(K_3body): 
        constraints_C1 += [dm_tilde_C1[i] >> 1e-8]  # non-negative eigenvalues
    
    if model_type=='open':
        for i in range(K_3body):
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[i], dims=[4,2], axis=1) == dm_tilde[i]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[i], dims=[2,4], axis=0) == dm_tilde[i+1]]
    if model_type=='closed':
        if N==3:
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[0], dims=[2,2,2], axis=2) == dm_tilde[0]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[0], dims=[2,2,2], axis=1) == dm_tilde[2]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[0], dims=[2,2,2], axis=0) == dm_tilde[1]]
        else:
            for i in range(K_3body-1): # the first N-2 sub-global system
                constraints_C1 += [cp.partial_trace(dm_tilde_C1[i], dims=[2,2,2], axis=2) == dm_tilde[i]]
                constraints_C1 += [cp.partial_trace(dm_tilde_C1[i], dims=[2,2,2], axis=0) == dm_tilde[i+1]]
            # Add the globally compatitble constraints between the head and tail local systems (only works for M=2)
            # constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-2], dims=[2,4], axis=0) == dm_tilde[K-1]]
            # constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-2], dims=[4,2], axis=1) == dm_tilde[K-2]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[K_3body-1], dims=[2,2,2], axis=0) == dm_tilde[0]]
            constraints_C1 += [cp.partial_trace(dm_tilde_C1[K_3body-1], dims=[2,2,2], axis=2) == dm_tilde[K-1]]
                #constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-2], dims=[2,4], axis=0) == dm_tilde[K-2]]
                # constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-2], dims=[2,2,2], axis=1) == dm_tilde[K-1]]
                # constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-1], dims=[4,2], axis=1) == dm_tilde[0]]
                # constraints_C1 += [cp.partial_trace(dm_tilde_C1[K-1], dims=[2,2,2], axis=1) == dm_tilde[K-1]]
                # # (only works for M=2, G=3)
    
    return constraints_C1

def SDP_variables_C2(ep_C2:cp.expressions.variable.Variable, 
                     measurement_dataset:Dict[str,List[str]], 
                     N:int, G:int, 
                     PauliStrList_Gbody:List[int]) -> cp.atoms.affine.add_expr.AddExpression:
    '''Define SDP variables for C2:
    '''
    K_3body = N-G+1 # Number of 3-body subsystems
    P_3body = 4**G-1 # Number of Pauli basis for 3-body subsystems

    dm = []
    for k in range(K_3body):  # K: number of subsystems
        index = list(range(k, k+G, 1))  # [k, k+1, ...]
        dm.append(np.array(q_tomography_dm(index, measurement_dataset, N)))
    dm_hat_C2 = dm
    dm_tilde_C2 = dm
    for k in range(K_3body):
        for p in range(P_3body):
            dm_tilde_C2[k] = dm_tilde_C2[k] + cp.multiply(ep_C2[k, p], np.array(pauliToMatrix(PauliStrList_Gbody[p])))

    return dm_tilde_C2

def constraints_C2(ep_C2:cp.expressions.variable.Variable, 
                   coef:float, 
                   dm_tilde:cp.atoms.affine.add_expr.AddExpression, 
                   dm_tilde_C2:cp.atoms.affine.add_expr.AddExpression, 
                   measurement_dataset:Dict[str,List[str]], 
                   N:int, M:int, G:int) -> list:
    '''Define the constraints of the SDP for bisection method:
       1. there exists a 3-body global state whose reduced states are the corresponding subsystems' states
       2. in the confidence interval
    '''
    K_3body = N-G+1 # Number of 3-body subsystems
    P_3body = 4**G-1 # Number of Pauli basis for 3-body subsystems

    constraints_C2 = []
    for i in range(K_3body):
        constraints_C2 += [cp.partial_trace(dm_tilde_C2[i], dims=[4,2], axis=1) == dm_tilde[i]]
        constraints_C2 += [cp.partial_trace(dm_tilde_C2[i], dims=[2,4], axis=0) == dm_tilde[i+1]]

    sigma = np.zeros((K_3body, P_3body))
    for i in range(K_3body):
        index = list(range(i, i+G, 1))  # [i, i+1, ...]
        sigma[i] = Wald_interval_bisection(coef, index, 0.95, measurement_dataset, N)
    constraints_C2 += [ep_C2 >= -sigma, ep_C2 <= sigma]

    return constraints_C2

# def constraints_CWM(ep, coef, dm_tilde, dm_tilde_C1, measurement_dataset, N, M, G):
#     '''Define the constraints of weak monotonicity (CWM):
#        1. Weak monotonicity: For any state rho_ABC on systems ABC, we have: S(A|B) + S(A|C) >= 0
#        2. Here we only consider WM for each 3-body global state
       
#        Comments: 
#        Unlike the classical conditional entropy, the conditional quantum entropy can be negative.
#     '''
#     K_3body = N-G+1 # Number of 3-body subsystems
#     P_3body = 4**G-1 # Number of Pauli basis for 3-body subsystems

#     # constraints_WM = []
#     # for i in range(K_3body):
#     #     constraints_WM += [( cp.von_neumann_entr(cp.partial_trace(dm_tilde_C1[i], dims=[2]*G, axis=G-1))+
#     #                        cp.von_neumann_entr(cp.partial_trace(dm_tilde_C1[i], dims=[2]*G, axis=0)) ) 
#     #                        >= 
#     #                        ( cp.von_neumann_entr(cp.partial_trace(
#     #                             cp.partial_trace(dm_tilde_C1[i], dims=[2]*G, axis=0), dims=[2]*(G-1), axis=0
#     #                        )) + 
#     #                        cp.von_neumann_entr(cp.partial_trace(
#     #                             cp.partial_trace(dm_tilde_C1[i], dims=[2]*G, axis=G-1), dims=[2]*(G-1), axis=G-2
#     #                        )) )
#     #                        ]

#     constraints_WM = []
#     for i in range(K): # non-negative eigenvalues
#         constraints_WM += [dm_tilde[i] >> 1e-8]  
#     for i in range(K_3body):
#         constraints_WM += [cp.von_neumann_entr(dm_tilde[i])+cp.von_neumann_entr(dm_tilde[i+1]) >= 
#                         cp.von_neumann_entr(cp.partial_trace(dm_tilde[i], dims=[2] * M, axis=M-1))+
#                         cp.von_neumann_entr(cp.partial_trace(dm_tilde[i+1], dims=[2] * M, axis=0))]

#     return constraints_WM


#----------------------------------------------------------------------------------------------------------------------
# Solve the SDP problems
def SDP_solver_min(coef:float, 
                   ep:cp.expressions.variable.Variable, ep_C1:cp.expressions.variable.Variable, 
                   dm_tilde:cp.atoms.affine.add_expr.AddExpression, dm_tilde_C1:cp.atoms.affine.add_expr.AddExpression, 
                   H:np.ndarray, 
                   measurement_dataset:Dict[str,List[str]], 
                   N:int, M:int, G:int, K:int, P:int,
                   model_type:str) -> (float, float, float):
    '''Solve the SDP minimization problem with constraints C0 and C0+C1
    '''
    
    dm_tilde_copy0 = dm_tilde
    dm_tilde_copy1 = dm_tilde
    dm_tilde_copy01 = dm_tilde
    dm_tilde_copyWM = dm_tilde

    # Solve SDP with conditions C0
    constraints0 = constraints_C0(ep, coef, dm_tilde_copy0, measurement_dataset, N, M, K, P, model_type)
    H_exp0 = 0
    for i in range(K):
        H_exp0 = H_exp0 + H @ dm_tilde_copy0[i]
    prob_C0 = cp.Problem(
        cp.Minimize(
            cp.real(
                cp.trace(
                    H_exp0
                )
            )
        ), constraints0
    )
    energy_C0 = prob_C0.solve(solver=cp.SCS, verbose=False)
    if prob_C0.status != cp.OPTIMAL:
        energy_C0 = float('inf') 
    
    # # Solve SDP with conditions C1
    # constraints1 = constraints_C1(ep_C1, coef, dm_tilde_copy1, dm_tilde_C1, measurement_dataset, N, M, G, K, model_type)
    # H_exp1 = 0
    # for i in range(K):
    #     H_exp1 = H_exp1 + H @ dm_tilde_copy1[i]
    # prob_C1 = cp.Problem(
    #     cp.Minimize(
    #         cp.real(
    #             cp.trace(
    #                 H_exp1
    #             )
    #         )
    #     ), constraints1
    # )
    # energy_C1 = prob_C1.solve(solver=cp.SCS, verbose=False)
    # if prob_C1.status != cp.OPTIMAL:
    #     energy_C1 = float('inf')
    energy_C1 = 0

    # Solve SDP with conditions C1+C0
    constraints_0 = constraints_C0(ep, coef, dm_tilde_copy01, measurement_dataset, N, M, K, P, model_type)
    constraints_1 = constraints_C1(ep_C1, coef, dm_tilde_copy01, dm_tilde_C1, measurement_dataset, N, M, G, K, model_type)
    H_exp01 = 0
    for i in range(K):
        H_exp01 = H_exp01 + H @ dm_tilde_copy01[i]
    prob_C01 = cp.Problem(
        cp.Minimize(
            cp.real(
                cp.trace(
                    H_exp01
                )
            )
        ), constraints_0 + constraints_1
    )
    energy_C01 = prob_C01.solve(solver=cp.SCS, verbose=False)
    if prob_C01.status != cp.OPTIMAL:
        energy_C01 = float('inf') 

    return energy_C0, energy_C1, energy_C01

def SDP_solver_max(coef:float, 
                   ep:cp.expressions.variable.Variable, ep_C1:cp.expressions.variable.Variable, 
                   dm_tilde:cp.atoms.affine.add_expr.AddExpression, dm_tilde_C1:cp.atoms.affine.add_expr.AddExpression, 
                   H:np.ndarray, 
                   measurement_dataset:Dict[str,List[str]], 
                   N:int, M:int, G:int, K:int, P:int,
                   model_type:str) -> (float, float, float):
    '''Solve the SDP maximization problem with constraints C0 and C0+C1
    '''
    
    dm_tilde_copy0 = dm_tilde
    dm_tilde_copy1 = dm_tilde
    dm_tilde_copy01 = dm_tilde

    # Solve SDP with conditions C0
    constraints0 = constraints_C0(ep, coef, dm_tilde_copy0, measurement_dataset, N, M, K, P, model_type)
    H_exp0 = 0
    for i in range(K):
        H_exp0 = H_exp0 + H @ dm_tilde_copy0[i]
    prob_C0 = cp.Problem(
        cp.Maximize(
            cp.real(
                cp.trace(
                    H_exp0
                )
            )
        ), constraints0
    )
    energy_C0 = prob_C0.solve(solver=cp.SCS, verbose=False)
    if prob_C0.status != cp.OPTIMAL:
        energy_C0 = float('inf') 

    # Solve SDP with conditions C1
    # constraints1 = constraints_C1(ep_C1, coef, dm_tilde_copy1, dm_tilde_C1, measurement_dataset, N, M, G, K, model_type)
    # H_exp1 = 0
    # for i in range(K):
    #     H_exp1 = H_exp1 + H @ dm_tilde_copy1[i]
    # prob_C1 = cp.Problem(
    #     cp.Maximize(
    #         cp.real(
    #             cp.trace(
    #                 H_exp1
    #             )
    #         )
    #     ), constraints1
    # )
    # energy_C1 = prob_C1.solve(solver=cp.SCS, verbose=False)
    # if prob_C1.status != cp.OPTIMAL:
    #     energy_C1 = float('inf')
    energy_C1 = 0

    # Solve SDP with conditions C1+C0
    constraints_0 = constraints_C0(ep, coef, dm_tilde_copy01, measurement_dataset, N, M, K, P, model_type)
    constraints_1 = constraints_C1(ep_C1, coef, dm_tilde_copy01, dm_tilde_C1, measurement_dataset, N, M, G, K, model_type)
    H_exp01 = 0
    for i in range(K):
        H_exp01 = H_exp01 + H @ dm_tilde_copy01[i]
    prob_C01 = cp.Problem(
        cp.Maximize(
            cp.real(
                cp.trace(
                    H_exp01
                )
            )
        ), constraints_0 + constraints_1
    )
    energy_C01 = prob_C01.solve(solver=cp.SCS, verbose=False)
    if prob_C01.status != cp.OPTIMAL:
        energy_C01 = float('inf') 


    return energy_C0, energy_C1, energy_C01

def biSection_search_min(higher_bound:float, threshold:float, 
                         ep:cp.expressions.variable.Variable, ep_C1:cp.expressions.variable.Variable, 
                         dm_tilde:cp.atoms.affine.add_expr.AddExpression, dm_tilde_C1:cp.atoms.affine.add_expr.AddExpression, 
                         H:np.ndarray, 
                         measurement_dataset:Dict[str,List[str]], 
                         N:int, M:int, G:int, K:int, P:int,
                         model_type:str) -> (float,float,float,float):
    '''Use bi-search method to find the minimum value of the relaxation such that there exists at least one solution in the search space,
       with an accuracy of 'threshold'
    '''

    low = 0
    high = higher_bound
    max_iter = 10
   
    energy_C0, energy_C1, energy_C01 = SDP_solver_min(high, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)
    coef = high
    
    # If no solution exists within the initial higher bounds, increase the higher bound.
    while (math.isinf(energy_C0) or math.isinf(energy_C1) or math.isinf(energy_C01)) and max_iter > 0:
        low = high
        high = 2*high
        max_iter = max_iter-1
        energy_C0, energy_C1, energy_C01 = SDP_solver_min(high, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)

    # If still no solution after expanding the bounds, return an error message.
    if max_iter == 0:
        return "No solution found within the search bounds and maximum iterations."
    
    # Perform the binary search within the updated bounds.
    while abs(high - low) >= threshold:
        coef = low + abs(high - low) / 2
        energy_C0_result, energy_C1_result, energy_C01_result = SDP_solver_min(coef, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)
        if (math.isinf(energy_C0_result) or math.isinf(energy_C1) or math.isinf(energy_C01_result)):
            low = coef
        else:
            high = coef
            energy_C0 = energy_C0_result
            energy_C1 = energy_C1_result
            energy_C01 = energy_C01_result

    # # Perform the binary search within the updated bounds.
    # while abs(high - low) > threshold or (math.isinf(energy_C0) or math.isinf(energy_C01)):
    #     coef = low + abs(high - low) / 2
    #     energy_C0, energy_C01 = SDP_solver_min(coef, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, K, P)
    #     if (math.isinf(energy_C0) or math.isinf(energy_C01)):
    #         low = coef
    #     else:
    #         high = coef

    return energy_C0, energy_C1, energy_C01, coef

def biSection_search_max(higher_bound:float, threshold:float, 
                         ep:cp.expressions.variable.Variable, ep_C1:cp.expressions.variable.Variable, 
                         dm_tilde:cp.atoms.affine.add_expr.AddExpression, dm_tilde_C1:cp.atoms.affine.add_expr.AddExpression, 
                         H:np.ndarray, 
                         measurement_dataset:Dict[str,List[str]], 
                         N:int, M:int, G:int, K:int, P:int,
                         model_type:str) -> (float,float,float,float):
    '''Use bi-search method to find the minimum value of the relaxation such that there exists at least one solution in the search space,
       with an accuracy of 'threshold'
    '''

    low = 0
    high = higher_bound
    max_iter = 10
   
    energy_C0, energy_C1, energy_C01 = SDP_solver_max(high, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)
    coef = high
    
    # If no solution exists within the initial higher bounds, increase the higher bound.
    while (math.isinf(energy_C0) or math.isinf(energy_C1) or math.isinf(energy_C01)) and max_iter > 0:
        low = high
        high = 2*high
        max_iter = max_iter-1
        energy_C0, energy_C1, energy_C01 = SDP_solver_max(high, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)

    # If still no solution after expanding the bounds, return an error message.
    if max_iter == 0:
        return "No solution found within the search bounds and maximum iterations."
    
    # Perform the binary search within the updated bounds.
    while abs(high - low) >= threshold:
        coef = low + abs(high - low) / 2
        energy_C0_result, energy_C1_result, energy_C01_result = SDP_solver_max(coef, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)
        if (math.isinf(energy_C0_result) or math.isinf(energy_C1) or math.isinf(energy_C01_result)):
            low = coef
        else:
            high = coef
            energy_C0 = energy_C0_result
            energy_C1 = energy_C1_result
            energy_C01 = energy_C01_result

    # # Perform the binary search within the updated bounds.
    # while abs(high - low) > threshold or (math.isinf(energy_C0) or math.isinf(energy_C01)):
    #     coef = low + abs(high - low) / 2
    #     energy_C0, energy_C01 = SDP_solver_min(coef, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, K, P)
    #     if (math.isinf(energy_C0) or math.isinf(energy_C01)):
    #         low = coef
    #     else:
    #         high = coef

    return energy_C0, energy_C1, energy_C01, coef

def biSection_gs(coef_gs:float, threshold:float, 
                 ep:cp.expressions.variable.Variable, ep_C1:cp.expressions.variable.Variable, 
                 dm_tilde:cp.atoms.affine.add_expr.AddExpression, dm_tilde_C1:cp.atoms.affine.add_expr.AddExpression, 
                 H:np.ndarray, 
                 measurement_dataset:Dict[str,List[str]], 
                 N:int, M:int, K:int, P:int,
                 model_type:str) -> (float,float,float):
    '''Use bi-search method to find the approximation of ground state energy
    '''

    low = 0
    max_iter = 6
    high = coef_gs
   
    energy_C0, energy_C01 = SDP_solver_min(high, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, K, P, model_type)
    coef = high
    
    # If no solution exists within the initial higher bounds, increase the higher bound.
    while (math.isinf(energy_C0) or math.isinf(energy_C01)) and max_iter > 0:
        low = high
        high = 2*high
        max_iter = max_iter-1
        energy_C0, energy_C01 = SDP_solver_min(high, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, K, P, model_type)
    energy_C0_gs = energy_C0 # Store the SDP solutions when the value of relaxation equals the higher bound (to approch the ground energy)
    energy_C01_gs = energy_C01 # Store the SDP solutions when the value of relaxation equals the higher bound (to approch the ground energy)

    # If still no solution after expanding the bounds, return an error message.
    if max_iter == 0:
        return "No solution found within the search bounds and maximum iterations."

    return energy_C0_gs, energy_C01_gs, coef


#----------------------------------------------------------------------------------------------------------------------
# Main functions
def jordi_min(repetition:int, 
              N_meas_list:List[int], 
              higher_bound:float, threshold:float, 
              N:int, M:int, G:int, K:int, P:int, 
              model_type:str,
              PauliStrList_part:List[str], PauliStrList_Gbody:List[str],
              H_local_matrix:np.ndarray, 
              H_global_list:List[str]) -> (List[float],List[float],List[float],List[float],List[float],List[float]):
    '''Solve the SDP minimization and maximization problem with the two different constraints for a list of number of measurement
    '''

    E_min = []
    E_max = []

    E_min_C0 = []
    E_min_C1 = []
    E_min_C01 = []
    coef_min = []

    for N_meas in N_meas_list:
        path = f'meas_dataset/{model_type}/N={N}/N{N}_Meas{N_meas}.npy'
        data = np.load(path, allow_pickle=True)
        measurement_dataset = data[repetition]
        measurement_dataset = {key: value for key, value in measurement_dataset.items() if value} # For reducing the complexity
        N_meas_sub = N_meas * (3 ** (N - M))

        ep = cp.Variable((K, P))
        if model_type=='open':
            K_3body = N-G+1 # Number of 3-body subsystems
        if model_type=='closed':
            K_3body = K # Number of 3-body subsystems
        ep_C1 = cp.Variable((K_3body, 4**G-1))

        dm_tilde, dm_hat = SDP_variables_C0(ep, measurement_dataset, N, M, K, P, PauliStrList_part, model_type)
        dm_tilde_C1 = SDP_variables_C1(ep_C1, measurement_dataset, N, G, K, PauliStrList_Gbody, model_type)

        # Energy with SDP - minimum
        E_min_C0_value, E_min_C1_value, E_min_C01_value, coef_min_value = biSection_search_min(higher_bound, threshold, 
                                                                               ep, ep_C1, dm_tilde, dm_tilde_C1, 
                                                                               H_local_matrix, measurement_dataset, 
                                                                               N, M, G, K, P,
                                                                               model_type
                                                                               )
        E_min_C0.append(E_min_C0_value)
        E_min_C1.append(E_min_C1_value)
        E_min_C01.append(E_min_C01_value)
        coef_min.append(coef_min_value)
        
        # Average energy calculated from measurements
        E_min_and_max = gs_energy_estimate(measurement_dataset, 0.99, H_global_list, model_type)
        E_min.append(E_min_and_max[0])
        E_max.append(E_min_and_max[1])

        print("Case N_meas =", N_meas, "finished")

    return E_min_C0, E_min_C1, E_min_C01, coef_min, E_min, E_max

def jordi_max(repetition:int, 
              N_meas_list:List[int], 
              higher_bound:float, threshold:float, 
              N:int, M:int, G:int, K:int, P:int, 
              model_type:str,
              PauliStrList_part:List[str], PauliStrList_Gbody:List[str],
              H_local_matrix:np.ndarray, 
              H_global_list:List[str]) -> (List[float],List[float],List[float],List[float]):
    '''Solve the SDP minimization and maximization problem with the two different constraints for a list of number of measurement
    '''

    E_max_C0 = []
    E_max_C1 = []
    E_max_C01 = []
    coef_max = []

    for N_meas in N_meas_list:
        path = f'meas_dataset/{model_type}/N={N}/N{N}_Meas{N_meas}.npy'
        data = np.load(path, allow_pickle=True)
        measurement_dataset = data[repetition]
        measurement_dataset = {key: value for key, value in measurement_dataset.items() if value} # For reducing the complexity
        N_meas_sub = N_meas * (3 ** (N - M))
        
        ep = cp.Variable((K, P))
        if model_type=='open':
            K_3body = N-G+1 # Number of 3-body subsystems
        if model_type=='closed':
            K_3body = K # Number of 3-body subsystems
        ep_C1 = cp.Variable((K_3body, 4**G-1))
 
        dm_tilde, dm_hat = SDP_variables_C0(ep, measurement_dataset, N, M, K, P, PauliStrList_part, model_type)
        dm_tilde_C1 = SDP_variables_C1(ep_C1, measurement_dataset, N, G, K, PauliStrList_Gbody, model_type)
        
        # Energy with SDP - maximum
        E_max_C0_value, E_max_C1_value, E_max_C01_value, coef_max_value = biSection_search_max(higher_bound, threshold, 
                                                                               ep, ep_C1, dm_tilde, dm_tilde_C1, 
                                                                               H_local_matrix, measurement_dataset, 
                                                                               N, M, G, K, P,
                                                                               model_type
                                                                               )
        E_max_C0.append(E_max_C0_value)
        E_max_C1.append(E_max_C1_value)
        E_max_C01.append(E_max_C01_value)
        coef_max.append(coef_max_value)

        print("Case N_meas =", N_meas, "finished")

    return E_max_C0, E_max_C1, E_max_C01, coef_max

def get_SDP_dataset_min(num_of_shot:int, 
                        N_meas_list:List[int], 
                        higher_bound:float, threshold:float, 
                        N:int, M:int, G:int, K:int, P:int, 
                        model_type:str,
                        PauliStrList_part:List[str], PauliStrList_Gbody:List[str],
                        H_local_matrix:np.ndarray, 
                        H_global_list:List[str]) -> dict:
    '''Get the dataset of the solution of the SDP problems
    '''

    data = {}
    data['E_min_C0'] = []
    data['E_min_C1'] = []
    data['E_min_C01'] = []
    data['coef_min'] = []

    data['E_min'] = []
    data['E_max'] = []
    
    for repetition in range(num_of_shot):
        E_min_C0, E_min_C1, E_min_C01, coef_min, E_min, E_max = jordi_min(
            repetition, 
            N_meas_list, higher_bound, threshold, 
            N, M, G, K, P,
            model_type,
            PauliStrList_part, PauliStrList_Gbody,
            H_local_matrix, H_global_list
        )
        data['E_min_C0'].append(E_min_C0)
        data['E_min_C1'].append(E_min_C1)
        data['E_min_C01'].append(E_min_C01)
        data['coef_min'].append(coef_min)
        data['E_min'].append(E_min)
        data['E_max'].append(E_max)
    return data

def get_SDP_dataset_max(num_of_shot:int, 
                        N_meas_list:List[int], 
                        higher_bound:float, threshold:float, 
                        N:int, M:int, G:int, K:int, P:int, 
                        model_type:str,
                        PauliStrList_part:List[str], PauliStrList_Gbody:List[str],
                        H_local_matrix:np.ndarray, 
                        H_global_list:List[str]) -> dict:
    '''Get the dataset of the solution of the SDP problems
    '''

    data = {}
    data['E_max_C0'] = []
    data['E_max_C1'] = []
    data['E_max_C01'] = []
    data['coef_max'] = []

    for repetition in range(num_of_shot):
        E_max_C0, E_max_C1, E_max_C01, coef_max = jordi_max(
            repetition, 
            N_meas_list, higher_bound, threshold, 
            N, M, G, K, P,
            model_type,
            PauliStrList_part, PauliStrList_Gbody,
            H_local_matrix, H_global_list
        )
        data['E_max_C0'].append(E_max_C0)
        data['E_max_C1'].append(E_max_C1)
        data['E_max_C01'].append(E_max_C01)
        data['coef_max'].append(coef_max)
    return data

def process_SDP_dataset(data:dict, num_of_shot:int, 
                        num_data_point:int) -> (Dict[str,np.ndarray],Dict[str,np.ndarray]):
    '''Given the dataset of SDP problem results,
       return the mean value and standard deviation
    '''
    E_mean = {}
    E_std = {}

    for key in data:
        tmp = np.array(data[key])
        E_mean[key] = np.mean(tmp, axis=0)
        E_std[key] = np.std(tmp, axis=0) / num_of_shot ** 0.5

    return E_mean, E_std


#----------------------------------------------------------------------------------------------------------------------
# Moniter the time complexity with fixed G as N increases
import time
def time_cost_min_C01(H:np.ndarray, 
                      measurement_dataset:Dict[str,List[str]], 
                      N:int, M:int, G:int, K:int, P:int, 
                      PauliStrList_part:List[str]) -> (float,float):
    '''Solve the SDP using bi-section method
    '''
    ep = cp.Variable((K, P))
    ep_C1 = cp.Variable((N-G+1, 4**G-1))
    ep_C2 = cp.Variable((N-G+1, 4**G-1))
    coef = 1

    # Solve SDP with conditions C1+C0
    dm_tilde01, dm_hat01 = SDP_variables_C0(ep, measurement_dataset, N, M, K, P, PauliStrList_part)
    dm_tilde_C1 = SDP_variables_C1(ep_C1, measurement_dataset, N, G)
    constraints0 = constraints_C0(ep, coef, dm_tilde01, measurement_dataset, N, M, K, P)
    constraints1 = constraints_C1(ep_C1, coef, dm_tilde01, dm_tilde_C1, measurement_dataset, N, M, G)
    
    H_exp01 = 0
    for i in range(K):
        H_exp01 = H_exp01 + H @ dm_tilde01[i]
    prob_C1 = cp.Problem(
        cp.Minimize(
            cp.real(
                cp.trace(
                    H_exp01
                )
            )
        ), constraints0 + constraints1
    )
    energy_C1 = prob_C1.solve(solver=cp.SCS, verbose=False)

    return energy_C1, coef

def time_cost_max_C01(H:np.ndarray, 
                      measurement_dataset:Dict[str,List[str]], 
                      N:int, M:int, G:int, K:int, P:int, 
                      PauliStrList_part:List[str]) -> (float,float):
    '''Solve the SDP using bi-section method
    '''
    ep = cp.Variable((K, P))
    ep_C1 = cp.Variable((N-G+1, 4**G-1))
    ep_C2 = cp.Variable((N-G+1, 4**G-1))
    coef = 1

    # Solve SDP with conditions C1+C0
    dm_tilde01, dm_hat01 = SDP_variables_C0(ep, measurement_dataset, N, M, K, P, PauliStrList_part)
    dm_tilde_C1 = SDP_variables_C1(ep_C1, measurement_dataset, N, G)
    constraints0 = constraints_C0(ep, coef, dm_tilde01, measurement_dataset, N, M, K, P)
    constraints1 = constraints_C1(ep_C1, coef, dm_tilde01, dm_tilde_C1, measurement_dataset, N, M, G)
    
    H_exp01 = 0
    for i in range(K):
        H_exp01 = H_exp01 + H @ dm_tilde01[i]
    prob_C1 = cp.Problem(
        cp.Maximize(
            cp.real(
                cp.trace(
                    H_exp01
                )
            )
        ), constraints0 + constraints1
    )
    energy_C1 = prob_C1.solve(solver=cp.SCS, verbose=False)

    return energy_C1, coef

def time_cost_min_C0(H:np.ndarray, 
                     measurement_dataset:Dict[str,List[str]], 
                     N:int, M:int, G:int, K:int, P:int, 
                     PauliStrList_part:List[str]) -> (float,float):
    '''Solve the SDP using bi-section method
    '''
    ep = cp.Variable((K, P))
    coef = 1

    # Solve SDP with conditions C0
    dm_tilde0, dm_hat0 = SDP_variables_C0(ep, measurement_dataset, N, M, K, P, PauliStrList_part)
    constraints0 = constraints_C0(ep, coef, dm_tilde0, measurement_dataset, N, M, K, P)
    
    H_exp0 = 0
    for i in range(K):
        H_exp0 = H_exp0 + H @ dm_tilde0[i]
    prob_C0 = cp.Problem(
        cp.Minimize(
            cp.real(
                cp.trace(
                    H_exp0
                )
            )
        ), constraints0
    )
    energy_C0 = prob_C0.solve(solver=cp.SCS, verbose=False)

    return energy_C0, coef

def time_cost_max_C0(H:np.ndarray, 
                     measurement_dataset:Dict[str,List[str]], 
                     N:int, M:int, G:int, K:int, P:int, 
                     PauliStrList_part:List[str]) -> (float,float):
    '''Solve the SDP using bi-section method
    '''
    ep = cp.Variable((K, P))
    coef = 1

    # Solve SDP with conditions C0
    dm_tilde0, dm_hat0 = SDP_variables_C0(ep, measurement_dataset, N, M, K, P, PauliStrList_part)
    constraints0 = constraints_C0(ep, coef, dm_tilde0, measurement_dataset, N, M, K, P)
    
    H_exp0 = 0
    for i in range(K):
        H_exp0 = H_exp0 + H @ dm_tilde0[i]
    prob_C0 = cp.Problem(
        cp.Maximize(
            cp.real(
                cp.trace(
                    H_exp0
                )
            )
        ), constraints0
    )
    energy_C0 = prob_C0.solve(solver=cp.SCS, verbose=False)

    return energy_C0, coef