import argparse
import time
import random
import itertools
import numpy as np
import cvxpy as cp
import math
import scipy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from qutip import *
from qiskit import *
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, Operator, Pauli, partial_trace, state_fidelity, random_density_matrix, random_statevector
from qiskit.visualization import plot_histogram, plot_state_city, plot_bloch_multivector, plot_state_paulivec, plot_state_hinton, plot_state_qsphere
from qiskit.tools.monitor import job_monitor
import os

from SDPforVQE import *
from SDPforVQE import *

def pauli_product(single_pauli_str1, single_pauli_str2):
    '''Compute the product of Pauli operators in an analytical way
    Example 1:
        INPUT: 'XI', 'IY' 
        OUTPUT: [1, 'XY'] (Because XI times IY is XY)
    Example 2:
        INPUT: 'X', 'Y' 
        OUTPUT: [1j, 'Z'] (Because X times Y is iZ)
    '''

    if single_pauli_str1 == 'I':
        return [1, single_pauli_str2]
    if single_pauli_str2 == 'I':
        return [1, single_pauli_str1]

    if single_pauli_str1 == 'X':
        if single_pauli_str2 == 'X':
            return [1, 'I']
        elif single_pauli_str2 == 'Y':
            return [1j, 'Z']
        elif single_pauli_str2 == 'Z':
            return [-1j, 'Y']
        
    if single_pauli_str1 == 'Y':
        if single_pauli_str2 == 'X':
            return [-1j, 'Z']
        elif single_pauli_str2 == 'Y':
            return [1, 'I']
        elif single_pauli_str2 == 'Z':
            return [1j, 'X']
        
    if single_pauli_str1 == 'Z':
        if single_pauli_str2 == 'X':
            return [1j, 'Y']
        elif single_pauli_str2 == 'Y':
            return [-1j, 'X']
        elif single_pauli_str2 == 'Z':
            return [1, 'I']

def pauli_commutator(pauli_str1, pauli_str2):
    '''Compute the commutator of two Pauli operators in an analytical way
    Example 1:
        INPUT: 'XI', 'IY' 
        OUTPUT: 0 (Because [XI,IY]=0)
    Example 2:
        INPUT: 'X', 'Y' 
        OUTPUT: [2j, Z] (Because [X,Y]=2iZ)
    '''
    
    commutator = pauliToMatrix(pauli_str1)*pauliToMatrix(pauli_str2) - pauliToMatrix(pauli_str2)*pauliToMatrix(pauli_str1)
    
    if np.all(np.array(commutator) == 0):
        return 0 
    else:
        commutator_str = ''
        coef = 2
        for i in range(len(pauli_str1)):
            result = pauli_product(pauli_str1[i], pauli_str2[i])
            commutator_str = commutator_str + result[1]
            coef = coef*result[0]
    
        return coef, commutator_str

def qiskit_statevec_map(statevec_qiskit, N):
    '''Qiskit orders qubits in a endian way, 
       this function is used to convert a state vector that written in endian ordering to a normal ordering
    '''
    statevec_qiskit = np.array(statevec_qiskit)
    statevec_normal = np.zeros_like(statevec_qiskit, dtype=complex)
    
    for i in range(2**N):
        binary_index = format(i, f'0{N}b')  # Convert the index to an N-bit binary string
        reversed_index = int(binary_index[::-1], 2)  # Reverse the binary string and convert it back to an integer
        
        statevec_normal[reversed_index] = statevec_qiskit[i]

    return Statevector(statevec_normal)

def available_h_set(N, M, K):
    '''Get all the accessible operations according to the layout of the Hamiltonian of insterest'''

    h_set = set({})

    for i in range(N):
        h_set.add('I'*i + 'X' + 'I'*(N-i-1))
        h_set.add('I'*i + 'Y' + 'I'*(N-i-1))
        h_set.add('I'*i + 'Z' + 'I'*(N-i-1))
    
    PauliStrList_part = generate_PauliStrList(M)[1:]
    for k in range(K):
        for basis in PauliStrList_part:
            h_set.add('I'*k + basis + 'I'*(N-k-M))

    return h_set

def get_reduced_pauli_strings(pauli_str):
    '''Given a Pauli string, eliminate all "I", and thus leaves the reduced Pauli string
    Example:
        INPUT: 'XIZIIY' 
        OUTPUT: 'XZY'
    '''
    non_trivial_indices = []
    for i, char in enumerate(pauli_str): # Iterate through the characters in the Pauli string
        if char in ['X', 'Y', 'Z']: # Check if the character represents a non-trivial Pauli operator
            non_trivial_indices.append(i) # Append the index to the list of non-trivial indices

    return non_trivial_indices

def get_all_relevant_indices(h_set, H_global_list):
    '''Get the set of qubit indices of all the relevant reduced density matrices, 
       which are used to calculate A=<hHh-H> and B=i<hH-Hh>
    Args:
       h_set: Set of accessible Pauli operators in the lab
       H_global_list: A list Pauli strings which describes the global Hamiltonian
    '''
    set_of_indices = set({})

    for h in h_set:
        for H in H_global_list:
            tmp = pauli_commutator(h, H)
            if tmp != 0:
                relevant_index = get_reduced_pauli_strings(tmp[1])
                set_of_indices.add(tuple(relevant_index))
                
    return set_of_indices

def get_rdm_dict(dm_Mbody, meas_dataset, 
            h_set, H_global_list, N, M, K):
    '''Get the dictionary for all relevant reduced density matrices required to calculate A=<hHh-H> and B=i<hH-Hh> 
    Args:
       dm_Mbody: A list of M-body density matrices, which can be obtained by SDP or just tomography
       meas_dataset: measurement dataset
       h_set: Set of accessible Pauli operators in the lab
       H_global_list: A list Pauli strings which describes the global Hamiltonian
    '''
    set_of_indices = get_all_relevant_indices(h_set, H_global_list)
    
    # Make a dictionary for all the relevant reduced density matrices
    dm_dict = {}
    for index in set_of_indices:
        dm_dict[tuple(index)] = np.array(q_tomography_dm(list(index), meas_dataset, N))
    
    # Replace the reduced density matrices that are associated with local Hamiltonians with dm_Mbody 
    # dm_Mbody can be obtained by SDP or just tomography
    for k in range(K):  # K: number of subsystems
        index = list(range(k, k + M, 1))  # [k, k+1, ...]
        dm_dict[tuple(index)] = dm_Mbody[index[0]]
        
    return dm_dict

def evolved_rdm(layer_operators, rdm_index, rdm):
    '''Given layer operator and a reduced density matrix (and its index),
       compute the evolved reduced density matrix
    Args: 
       layer_operators: A layer of unitary operators
       rdm: density matrix to be evovled
       rdm_index: the qubit indices of the rdm
    '''
    register = list(range(len(rdm_index)))
    
    for operator_index in layer_operators: # Loop over every unitary operator in this layer

        overlap_index = list(set(tuple(rdm_index)) & set(operator_index))
        extra_I_index = list(set(tuple(rdm_index)) - set(operator_index))

        if len(overlap_index) > 0:
            h_best, B, t_opt, decrease = layer_operators[operator_index] # Fetch the operator associated with operator_index
            reduced_h = ''.join(h_best[i] for i in rdm_index) # Get the reduced Pauli strings (e.g.: 'XYIII' to 'XY')
            
            # Get the reduced operator 
            if (B < 0 and t_opt < 0):
                reduced_U = Operator( scipy.linalg.expm( -1j * (t_opt+math.pi/2) * np.array(pauliToMatrix(reduced_h)) ) )
            elif (B <= 0 and t_opt >= 0):
                reduced_U = Operator( scipy.linalg.expm( -1j * t_opt * np.array(pauliToMatrix(reduced_h)) ) )
            elif (B >= 0 and t_opt <= 0):
                reduced_U = Operator( scipy.linalg.expm( 1j * (-t_opt) * np.array(pauliToMatrix(reduced_h)) ) )
            elif (B > 0 and t_opt > 0):
                reduced_U = Operator( scipy.linalg.expm( 1j * (-t_opt+math.pi/2) * np.array(pauliToMatrix(reduced_h)) ) )
            
            # Evolve
            rdm = np.array( DensityMatrix(np.array(rdm)).evolve(reduced_U, register) )
 
    return rdm
    
def get_current_rdm(layer_operators, dm_dict):
    '''Given the layer operators and measurement dataset,
       comupte all the M-body reduced density matrices and single-qubit density matrices
    Args: 
       layer_operators: A layer of unitary operators
       dm_dict: The dictionary that stores all relevant reduced density matrices required to calculate A=<hHh-H> and B=i<hH-Hh> 
    '''
    dm_dict_new = {}
    for rdm_index in dm_dict: # Loop through every rdm in dm_dict
        rdm = dm_dict[rdm_index]
        dm_dict_new[tuple(rdm_index)] = evolved_rdm(layer_operators, rdm_index, rdm) # Get the evolved reduced density matrix
        
    return dm_dict_new

def find_h_best(dm_dict, h_set, H_global_list, N, M, K):
    '''Given the accessible set operations and the reduced density matrices,
       find the 'cooling' operation which leads to the maximal decrease of energy expection
    Args: 
       dm_dict: The dictionary that stores all relevant reduced density matrices required to calculate A=<hHh-H> and B=i<hH-Hh> 
       h_set: Set of accessible Pauli operators in the lab
       H_global_list: A list Pauli strings which describes the global Hamiltonian
    '''
    
    h_cool = [] # Set of h such that i<[h,H]> is smaller than 0 and the corresponding optimal time t_opt
    
    for h in h_set: # Loop through every Pauli operator in the accessible set h_set
        
        # Compute commutator [h,H]
        commutator_1st_list = [] # [h,H]
        for H in H_global_list:
            tmp = pauli_commutator(h, H)
            if tmp != 0:
                commutator_1st_list.append(tmp)

        # Get B = i<[h,H]>
        B_tmp = 0
        for commutator in commutator_1st_list:
            relevant_index = get_reduced_pauli_strings(commutator[1])
            rho = dm_dict[tuple(relevant_index)] # reduced density matrices
            commutator_sub = commutator[1].replace('I', '')
            exp = np.trace(np.matmul(np.array(pauliToMatrix(commutator_sub)), np.array(rho)))
            B_tmp += exp*commutator[0]
        B = B_tmp*1j

        # Compute commutator [h,[h,H]]
        commutator_2nd_list = [] # [h,[h,H]]
        for h_H in commutator_1st_list:
            coef_1st = h_H[0]
            if pauli_commutator(h, h_H[1]) != 0:
                coef_2nd, commutator_2nd = pauli_commutator(h, h_H[1])
                commutator_2nd_list.append( (coef_1st*coef_2nd, commutator_2nd) )
                
        # Get A = -1/2*<[h,[h,H]]>
        A_tmp = 0
        for commutator in commutator_2nd_list:
            relevant_index = get_reduced_pauli_strings(commutator[1])
            rho = dm_dict[tuple(relevant_index)] # reduced density matrices
            commutator_sub = commutator[1].replace('I', '')
            exp = np.trace(np.matmul(np.array(pauliToMatrix(commutator_sub)), np.array(rho)))
            A_tmp += exp*commutator[0]
        A = -1/2*A_tmp

        #if A.real != 0 and B.real != 0:
        if A.real != 0:
            t_opt = 1/2*math.atan(-B.real/A.real) # The optimal time to evolve with
            decrease = 0.5*(A.real-math.sqrt(A.real**2+B.real**2)) # The decrease
            h_cool.append( (h, B.real, t_opt, decrease) ) # (Pauli string of h, B, t_opt, decrease)

        # Select the operator h which leads to the most decrease
        if len(h_cool)>0:
            h_best, B, t_opt, decrease = max(h_cool, key=lambda x: abs(x[3]))
        else:
            h_best, B, t_opt, decrease = 'I'*N, 0, 0, 0
  
    return h_best, B, t_opt, decrease

def find_compatible_paulis(N, qubit_index):
    '''Find the set of Pauli operators that support qubits associated with the input qubit_index
    Args: 
       N: number of qubits of the whole system
       qubit_index: indices of qubits
    Example:
       INPUT: N=3, qubit_index=[2]
       OUTPUT: {IIX, IIY, IIZ}
    '''
    
    h_set = generate_PauliStrList(len(qubit_index))[1:]
    identity_string = 'I'*N
    
    new_h_set = []

    for pauli_string in h_set:
        new_string_list = list(identity_string)

        for i in qubit_index:
            new_string_list[i] = pauli_string[qubit_index.index(i)]

        new_h_set.append(''.join(new_string_list))

    return new_h_set

def find_incompatible_paulis(pauli_set, given_pauli):
    '''Find the set of Pauli operators that support qubits NOT associated with the input qubit_index
    Args: 
       pauli_set: set of pauli operator string
       given_pauli: a pauli string
    Example:
       INPUT: pauli_set = {XXI, YYY, ZZI, ZYY}, given_pauli={IIX}
       OUTPUT: {XXI, ZZI}
    '''
    # Find indices of non-'I' characters in the given Pauli string
    non_I_indices = [i for i, char in enumerate(given_pauli) if char != 'I']

    # Find Pauli strings in the set with 'I' at these indices
    incompatible_paulis = set()
    for pauli in pauli_set:
        if all(pauli[i] == 'I' for i in non_I_indices):
            incompatible_paulis.add(pauli)

    return incompatible_paulis

def find_layer_operator(dm_dict, h_set, H_global_list, N, M, K,
                        num_of_sweep):
    '''Find a layer operator of algorithmic cooling
    Args:
       dm_dict: The dictionary that stores all relevant reduced density matrices required to calculate A=<hHh-H> and B=i<hH-Hh> 
       h_set: Set of accessible Pauli operators in the lab
       H_global_list: A list Pauli strings which describes the global Hamiltonian
    '''
    
    layer_operators_list = [] # A list of operators that forms the layer
                              # Each element of this list refers to one 'sub-layer' of this layer operator
                              # For example, we have only one 'sub-layer' in this list if num_of_sweep=0, and two if num_of_sweep=1
    layer_operators = {} # Initialize the first sweep
    while len(h_set) != 0:
        # Get the operator which gives the most decrease
        h_best, B, t_opt, decrease = find_h_best(dm_dict, h_set, H_global_list, N, M, K)
        layer_operators[ tuple([i for i, char in enumerate(h_best) if char != 'I']) ] = tuple((h_best, B, t_opt, decrease))
        
        # Evolve the reduced density matrices with this newly-getted operator
        new_operator = {} # Get the unitary operator we get
        new_operator[ tuple([i for i, char in enumerate(h_best) if char != 'I']) ] = tuple((h_best, B, t_opt, decrease))
        dm_dict = get_current_rdm(new_operator, dm_dict)
        
        # Updated the accessible set of operators
        h_set = find_incompatible_paulis(h_set, h_best)
    
    layer_operators_list.append(layer_operators)
    
    # Get the layout of this layer operator
    layout = list(layer_operators.keys())
    
    # Do the sweep
    for i in range(num_of_sweep):

        sweep_operators = {}

        for qubit_index in layout:

            h_set = find_compatible_paulis(N, list(qubit_index))

            h_best, B, t_opt, decrease = find_h_best(dm_dict, h_set, H_global_list, N, M, K)
            sweep_operators[ tuple([i for i, char in enumerate(h_best) if char != 'I']) ] = tuple((h_best, B, t_opt, decrease))
            
            new_operator = {}
            new_operator[ tuple([i for i, char in enumerate(h_best) if char != 'I']) ] = tuple((h_best, B, t_opt, decrease))
            dm_dict = get_current_rdm(new_operator, dm_dict)

        layer_operators_list.append(sweep_operators)


    return layer_operators_list

def get_dm(N, qiskit_state, qubit_index):
    '''Given a multi-qubit state and index, 
       get the density matrix of the reduced state associated with the index
    Args:
       N: number of qubits of the whole system
       qiskit_state: a quantum state with qiskit structure
       qubit_index: indices of qubits of interest
    '''

    basis_list = generate_sub_PauliStrList(N, qubit_index)
    
    # First get the Bloch vector
    Bloch_vec = []
    for basis in basis_list:
        basis = basis[::-1] # Qiskit use endian, so we take the reverse here
        Bloch_vec.append(qiskit_state.expectation_value(oper=Pauli(basis), qargs=None))
    
    # Now compute the reduced density matrix
    dm = 0
    for i in range(4**len(qubit_index)-1):
        basis = basis_list[i]
        sub_basis = ''.join([basis[i] for i in qubit_index])
        dm += Bloch_vec[i] * pauliToMatrix(sub_basis)
    dm += tensor([qeye(2)] * len(qubit_index))

    return 1 / (2 ** len(qubit_index)) * dm

def get_HF_state(H_global_list, H_global_matrix, N, M, K):
    '''Get the product state with the lowest <H>
    Args:
       H_global_list: A list Pauli strings which describes the global Hamiltonian
       H_global_matrix: matrix representation of the global Hamiltonian in computational basis
    '''

    # Initialize state to a random tensor product state
    input_state = np.array(DensityMatrix(random_statevector(2))) # Make the first qubit a random pure state
    for i in range(N-1):
        statevec = random_statevector(2)
        dm = DensityMatrix(statevec)
        input_state = np.kron(input_state, np.array(dm))
    input_state = DensityMatrix(input_state) # This is an N-qubit random tensor product state

    h_set = available_h_set(N, M=1, K=N)
    set_of_indices = get_all_relevant_indices(h_set, H_global_list)

    exp_H_value = np.real(np.trace( np.matmul(input_state, H_global_matrix) ))
    exp_H_value_list = [exp_H_value]
    diff = 10
    
    qc = QuantumCircuit(N)

    while abs(diff) > 1e-3:

        exp_H_value_old = exp_H_value
        
        # Make a dictionary for all the relevant reduced density matrices
        dm_dict = {}
        for qubit_index in set_of_indices:
            dm_dict[tuple(qubit_index)] = np.array( get_dm(N, input_state, qubit_index) )
            
        # Find the layer operators that gives a decrease to <H>
        layer_operators_list = find_layer_operator(dm_dict, h_set, H_global_list, N, M, K, num_of_sweep=0)
    
        # Evolve the state with the obtained layer operators
        for i in range(len(layer_operators_list)):
            layer_operators = layer_operators_list[i]

            for qubit_index in layer_operators:
                register = list(qubit_index)
                h_best, B, t_opt, decrease = layer_operators[qubit_index]
                h_best_reduced = ''.join(char for char in h_best if char != 'I')
            
                if (B < 0 and t_opt < 0):
                    U = Operator( scipy.linalg.expm( -1j * (t_opt+math.pi/2) * np.array(pauliToMatrix(h_best_reduced)) ) )
                elif (B <= 0 and t_opt >= 0):
                    U = Operator( scipy.linalg.expm( -1j * t_opt * np.array(pauliToMatrix(h_best_reduced)) ) )
                elif (B >= 0 and t_opt <= 0):
                    U = Operator( scipy.linalg.expm( 1j * (-t_opt) * np.array(pauliToMatrix(h_best_reduced)) ) )
                elif (B > 0 and t_opt > 0):
                    U = Operator( scipy.linalg.expm( 1j * (-t_opt+math.pi/2) * np.array(pauliToMatrix(h_best_reduced)) ) )
                    
                input_state = input_state.evolve(U, register[::-1])
                qc.append(U, register[::-1])
        qc.barrier()

        exp_H_value = np.real(np.trace( np.matmul(input_state, H_global_matrix) ))
        exp_H_value_list.append(exp_H_value)
        diff = exp_H_value - exp_H_value_old

    qc.draw('mpl')

    return DensityMatrix(input_state), exp_H_value_list
    
def SDP_solver_min_C01(coef:float, 
                   ep:cp.expressions.variable.Variable, ep_C1:cp.expressions.variable.Variable, 
                   dm_tilde:cp.atoms.affine.add_expr.AddExpression, dm_tilde_C1:cp.atoms.affine.add_expr.AddExpression, 
                   H:np.ndarray, 
                   measurement_dataset:Dict[str,List[str]], 
                   N:int, M:int, G:int, K:int, P:int,
                   model_type:str) -> (float,float,float):
    '''Solve the SDP minimization problem with constraints C0 and C0+C1
    '''
    
    dm_tilde_copy01 = dm_tilde

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
  
    dm_SDP = []
    for i in range(K):
        dm_SDP.append(dm_tilde_copy01[i].value)


    return energy_C01, dm_SDP

def biSection_search_min_C01(higher_bound:float, threshold:float, 
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
   
    energy_C01, dm_SDP = SDP_solver_min_C01(high, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)
    coef = high
    
    # If no solution exists within the initial higher bounds, increase the higher bound.
    while (math.isinf(energy_C01)) and max_iter > 0:
        low = high
        high = 2*high
        max_iter = max_iter-1
        energy_C01, dm_SDP = SDP_solver_min_C01(high, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)

    # If still no solution after expanding the bounds, return an error message.
    if max_iter == 0:
        return "No solution found within the search bounds and maximum iterations."
    
    # Perform the binary search within the updated bounds.
    while abs(high - low) >= threshold:
        coef = low + abs(high - low) / 2
        energy_C01_result, dm_SDP = SDP_solver_min_C01(coef, ep, ep_C1, dm_tilde, dm_tilde_C1, H, measurement_dataset, N, M, G, K, P, model_type)
        if (math.isinf(energy_C01_result)):
            low = coef
        else:
            high = coef
            energy_C01 = energy_C01_result

    return energy_C01, dm_SDP, coef

def algorithmic_cooling(input_state, N_opt, N_meas,
                        N, M, G, K, P, 
                        PauliStrList_part, PauliStrList_Gbody, h_set,
                        H_global_list, H_local_matrix,
                        higher_bound, threshold,
                        model_type,
                        SDP_tag, num_of_sweep):
    '''Do the algorithmic cooling
    '''
    # Get the energy of the initial input state
    E = 0
    for pauli_basis in H_global_list:
        E = E + input_state.expectation_value(oper=Pauli(pauli_basis), qargs=None)

    # Define lists for saving the results
    expH_dm_iter = [E]
    expH_dm_SDPvalue_iter = [E]

    # Start the optimization
    for i in tqdm(range(N_opt)):
        # Meas
        meas_dataset = generate_meas_dataset(input_state, N_meas, N)

        # Get the density matrix by tomography and SDP, respectively. 
        # Also get the minimized energy by SDP.
        ep = cp.Variable((K, P)) # SDP variables
        if model_type=='open':
            K_3body = N-G+1 # Number of 3-body subsystems
        if model_type=='closed':
            K_3body = K # Number of 3-body subsystems
        ep_C1 = cp.Variable((K_3body, 4**G-1)) # SDP variables for global verification
        dm_tilde, dm_hat = SDP_variables_C0(ep, meas_dataset, 
                                            N, M, K, P, 
                                            PauliStrList_part, model_type)
        
        # Solve the SDP problem
        if SDP_tag:
            dm_tilde_C1 = SDP_variables_C1(ep_C1, meas_dataset, 
                                    N, G, K, 
                                    PauliStrList_Gbody, model_type)
            # Energy with SDP - minimum
            E_min_C01_value, dm_SDP, coef_min_value = biSection_search_min_C01(higher_bound, threshold, 
                                                                                    ep, ep_C1, dm_tilde, dm_tilde_C1, 
                                                                                    H_local_matrix, meas_dataset, 
                                                                                    N, M, G, K, P,
                                                                                    model_type
                                                                                    ) # dm_SDP is the density matrices by SDP

            expH_dm_SDPvalue_iter.append(E_min_C01_value) # Save the solved SDP min. value
            dm_dict = get_rdm_dict(dm_SDP, meas_dataset, h_set, H_global_list, N, M, K)
            layer_operators_list = find_layer_operator(dm_dict, h_set, H_global_list, N, M, K, num_of_sweep)
        else:
            dm_dict = get_rdm_dict(dm_hat, meas_dataset, h_set, H_global_list, N, M, K)
            layer_operators_list = find_layer_operator(dm_dict, h_set, H_global_list, N, M, K, num_of_sweep)

        # Get the expectation value <H>
        exp_H_new = 0
        for H in H_global_list:
            exp, var = exp_var_calculator(meas_dataset, H)
            exp_H_new += exp
        expH_dm_iter.append(exp_H_new) # Save the expectation value <H>

        # Evolve the state with the calculated layer operator 
        for i in range(len(layer_operators_list)):
                
            layer_operators = layer_operators_list[i]

            for qubit_index in layer_operators:
                register = list(qubit_index)
                h_best, B, t_opt, decrease = layer_operators[qubit_index]
                h_best_reduced = ''.join(char for char in h_best if char != 'I')
            
                if (B < 0 and t_opt < 0):
                    U = Operator( scipy.linalg.expm( -1j * (t_opt+math.pi/2) * np.array(pauliToMatrix(h_best_reduced)) ) )
                elif (B <= 0 and t_opt >= 0):
                    U = Operator( scipy.linalg.expm( -1j * t_opt * np.array(pauliToMatrix(h_best_reduced)) ) )
                elif (B >= 0 and t_opt <= 0):
                    U = Operator( scipy.linalg.expm( 1j * (-t_opt) * np.array(pauliToMatrix(h_best_reduced)) ) )
                elif (B > 0 and t_opt > 0):
                    U = Operator( scipy.linalg.expm( 1j * (-t_opt+math.pi/2) * np.array(pauliToMatrix(h_best_reduced)) ) )
                    
                input_state = input_state.evolve(U, register[::-1])
    
    return expH_dm_iter, expH_dm_SDPvalue_iter

def get_figure(avg_expH, std_expH, 
               avg_expH_enhanced, std_expH_enhanced, 
               avg_expH_enhanced_SDPvalue, std_expH_enhanced_SDPvalue, 
               ground_state_energy,
               initial_guess, N, N_opt, N_meas):
    '''Draw and plot the figure
    '''
    
    plt.figure()
    iteration_list = list(range(N_opt+1))
    
    # Plot the theoretical ground state energy
    plt.axhline(y = ground_state_energy, color='r', linestyle='-', linewidth=1.25, label='GS energy')

    # Plot the case without SDP
    plt.plot(iteration_list, avg_expH, linewidth=0.75, marker='s', markersize=2.5, label='Tomography')
    plt.fill_between(iteration_list, 
                    avg_expH-std_expH, avg_expH+std_expH, 
                    alpha=0.3)

    # Plot the case with SDP
    plt.plot(iteration_list, avg_expH_enhanced, linewidth=0.75, marker='s', markersize=2.5, label='SDP-Enhanced')
    plt.plot(iteration_list, avg_expH_enhanced_SDPvalue, linewidth=0.75, marker='o', markersize=2.5, label='SDP-Enhanced-min.')
    plt.fill_between(iteration_list, 
                    avg_expH_enhanced-std_expH_enhanced, avg_expH_enhanced+std_expH_enhanced, 
                    alpha=0.3)
    plt.fill_between(iteration_list, 
                    avg_expH_enhanced_SDPvalue-std_expH_enhanced_SDPvalue, avg_expH_enhanced_SDPvalue+std_expH_enhanced_SDPvalue, 
                    alpha=0.3)


    titlename = 'Initial:' + initial_guess + ', $N=$' + str(N) + ', $N_{meas}=$' + str(int(N_meas))
    plt.title(titlename)
    plt.xlabel('Number of iterations')
    plt.ylabel('Energy expectation')
    plt.legend()

    figurename = 'N' + str(N) + '_Meas' + str(int(N_meas)) + '_'+ initial_guess +  '.pdf'
    plt.savefig(figurename)

    # Show the plot
    plt.show()

def main(initial_guess, N_opt, N_meas, num_of_shots,
         N, M, G, H_local_list, model_type):
    
    # Get the useful parameters and constants for this function
    if model_type == 'open':
        K = N-M+1 # Number of subsystems
    if model_type == 'closed':
        K = N
    P = 4**M-1 # Number of Pauli basis for each subsystem
    PauliStrList_part = generate_PauliStrList(M)[1:]
    PauliStrList_Gbody = generate_PauliStrList(G)[1:]
    h_set = available_h_set(N, M, K)
    H_global_list = Hamiltonian_global(H_local_list, N, M, K, model_type) # Pauli string representation of the Hamiltonian of the whole system
    H_local_matrix = np.array( Hamiltonian_matrix(H_local_list, model_type) ) # Matrix representation of the local Hamiltonian of subsystems
    H_global_matrix = np.array( Hamiltonian_matrix(H_global_list, model_type) ) # Matrix representation of the Hamiltonian of the whole system
    higher_bound = 1 # Starting trial value for the bi-search method
    threshold = 1 # Accuracy of the minimum relaxation value 
    ground_state_energy, ground_state_dm = ground_state(H_global_matrix) 

    # Initial state
    if initial_guess == 'HF':
        input_state, exp_H_value_list = get_HF_state(H_global_list, H_global_matrix, N, M, K) # The HF state (the product state with the lowest <H>)
    if initial_guess == '00':
        tmp = np.zeros(2**N)
        tmp[0] = 1
        input_state = DensityMatrix( qiskit_statevec_map( Statevector(tmp), N ) ) # |000> state
    if initial_guess == '++':
        tmp = np.zeros(2**N)
        tmp[0] = 1
        input_state = DensityMatrix( qiskit_statevec_map( Statevector(tmp), N ) ) # |000> state
        Hadamard = Operator(np.array(1/2**0.5*(qutip.sigmax()+qutip.sigmaz()))) 
        for i in range(N):
            input_state = input_state.evolve(Hadamard, [i])
    
    # Define lists for saving the results
    expH_matrix = []
    expH_enhanced_matrix = []
    expH_enhanced_SDPvalue_matrix = []

    # Do the algorithmic cooling for many times
    for i in tqdm(range(num_of_shots)):
        # Without SDP
        expH, expH_SDPvalue = algorithmic_cooling(input_state, N_opt, N_meas,
                                                  N, M, G, K, P, 
                                                  PauliStrList_part, PauliStrList_Gbody, h_set,
                                                  H_global_list, H_local_matrix,
                                                  higher_bound, threshold,
                                                  model_type,
                                                  SDP_tag=False, num_of_sweep=0)
        expH_matrix.append(expH)
        # With SDP
        expH_enhanced, expH_enhanced_SDPvalue = algorithmic_cooling(input_state, N_opt, N_meas,
                                                                    N, M, G, K, P, 
                                                                    PauliStrList_part, PauliStrList_Gbody, h_set,
                                                                    H_global_list, H_local_matrix,
                                                                    higher_bound, threshold,
                                                                    model_type,
                                                                    SDP_tag=True, num_of_sweep=0)
        expH_enhanced_matrix.append(expH_enhanced)
        expH_enhanced_SDPvalue_matrix.append(expH_enhanced_SDPvalue)

    # Calculate the avg and std of cases WITHOUT SDP
    avg_expH= np.mean(np.array(expH_matrix), axis=0)
    std_expH = np.std(np.array(expH_matrix), axis=0)/(num_of_shots**0.5)

    # Calculate the avg and std of cases WITH SDP
    avg_expH_enhanced= np.mean(np.array(expH_enhanced_matrix), axis=0)
    std_expH_enhanced= np.std(np.array(expH_enhanced_matrix), axis=0)/(num_of_shots**0.5)
    avg_expH_enhanced_SDPvalue = np.mean(np.array(expH_enhanced_SDPvalue_matrix), axis=0)
    std_expH_enhanced_SDPvalue = np.std(np.array(expH_enhanced_SDPvalue_matrix), axis=0)/(num_of_shots**0.5)
    
    # Draw and save the figure
    get_figure(avg_expH, std_expH, 
               avg_expH_enhanced, std_expH_enhanced, 
               avg_expH_enhanced_SDPvalue, std_expH_enhanced_SDPvalue, 
               ground_state_energy,
               initial_guess, N, N_opt, N_meas)

    return avg_expH, std_expH, avg_expH_enhanced, std_expH_enhanced, avg_expH_enhanced_SDPvalue, std_expH_enhanced_SDPvalue, ground_state_energy
        


H_local_list = ['XX','YY'] # Pauli string representation of the local Hamiltonian of subsystems
model_type = 'open'
M = 2 # Number of qubits of subsystems
G = 3 # Number of qubits of partial global system (C1)

N_opt = 15 # Number of iterations of cooling
num_of_shots = 50 # Number of experiments we do 

data = []
for N in [3,4,5,6,7,8]: # Number of qubits of the entire system
    for N_meas in [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]: # Number of measurements in all basis each loop
        for initial_guess in ['HF', '++']:
            avg_expH, std_expH, avg_expH_enhanced, std_expH_enhanced, avg_expH_enhanced_SDPvalue, std_expH_enhanced_SDPvalue, ground_state_energy = main(initial_guess, N_opt, N_meas, num_of_shots, 
                                                                                                                                                N, M, G, H_local_list, model_type)
            for i in list(range(N_opt+1)):
                # Save data to Panda DataFrame
                df = {
                    'N_opt': i,
                    'avg_expH': avg_expH[i],
                    'std_expH': std_expH[i],
                    'avg_expH_enhanced': avg_expH_enhanced[i],
                    'std_expH_enhanced': std_expH_enhanced[i],
                    'avg_expH_enhanced_SDPvalue': avg_expH_enhanced_SDPvalue[i],
                    'std_expH_enhanced_SDPvalue': std_expH_enhanced_SDPvalue[i],
                    'N': N,
                    'N_meas': N_meas,
                    'Initial_state': initial_guess
                }
                data.append(df)

df = pd.DataFrame(data)
df.to_csv('experiment_data.csv', index=False) # Save the DataFrame to a CSV file