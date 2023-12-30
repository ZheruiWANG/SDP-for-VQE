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
import os

from SDPforVQE import generate_PauliStrList, Hamiltonian_global, Hamiltonian_matrix, ground_state, lower_bound_with_SDP, N_meas_list_func
from SDPforVQE import get_SDP_dataset_min, get_SDP_dataset_max, process_SDP_dataset




model_type = 'closed'
#model_type = 'open'

N = 3 # Number of qubits of the entire system
M = 2 # Number of qubits of subsystems
G = 3 # Number of qubits of partial global system (C1)
if model_type == 'open':
    K = N-M+1 # Number of subsystems
if model_type == 'closed':
    K = N
P = 4**M-1 # Number of Pauli basis for each subsystem

PauliStrList = generate_PauliStrList(N)[1:]
PauliStrList_part = generate_PauliStrList(M)[1:]
PauliStrList_Gbody = generate_PauliStrList(G)[1:]

H_local_list = ['XX','YY'] # Pauli string representation of the local Hamiltonian of subsystems
H_global_list = Hamiltonian_global(H_local_list, N, M, K, model_type) # Pauli string representation of the Hamiltonian of the whole system
H_local_matrix = np.array( Hamiltonian_matrix(H_local_list, model_type) ) # Matrix representation of the local Hamiltonian of subsystems
H_global_matrix = np.array( Hamiltonian_matrix(H_global_list, model_type) ) # Matrix representation of the Hamiltonian of the whole system

ground_state_energy, ground_state_dm = ground_state(H_global_matrix) 
q_state = DensityMatrix(ground_state_dm) 
lower_bound = lower_bound_with_SDP(H_local_matrix, N, M, G, K, P, PauliStrList_part, PauliStrList_Gbody, model_type)

num_data_point = 15 # number of N_meas that we select to run
N_meas_list = N_meas_list_func(100, 100000, num_data_point) # A list of number of measurement performed in all basis
num_of_shot = 100 # Number of repeatation of the experiment

higher_bound = 0.2 # Starting trial value for the bi-search method
threshold = 0.001 # Accuracy of the minimum relaxation value 
data_min = get_SDP_dataset_min(num_of_shot=num_of_shot,
                       N_meas_list=N_meas_list,
                       higher_bound=higher_bound,
                       threshold=threshold,
                       N=N,
                       M=M,
                       G=G,
                       K=K,
                       P=P,
                       model_type=model_type,
                       PauliStrList_part=PauliStrList_part,
                       PauliStrList_Gbody=PauliStrList_Gbody,
                       H_local_matrix=H_local_matrix, 
                       H_global_list=H_global_list)
data_max = get_SDP_dataset_max(num_of_shot=num_of_shot,
                       N_meas_list=N_meas_list,
                       higher_bound=higher_bound,
                       threshold=threshold,
                       N=N,
                       M=M,
                       G=G,
                       K=K,
                       P=P, 
                       model_type=model_type,
                       PauliStrList_part=PauliStrList_part,
                       PauliStrList_Gbody=PauliStrList_Gbody,
                       H_local_matrix=H_local_matrix, 
                       H_global_list=H_global_list)

E_mean_min, E_std_min = process_SDP_dataset(data_min, num_of_shot, num_data_point)
E_mean_max, E_std_max = process_SDP_dataset(data_max, num_of_shot, num_data_point)

name = model_type + '_N' + str(N) + '_threshold' + str(threshold)
filename_min = '%s_min.npy' % name
filename_max = '%s_max.npy' % name

np.save(filename_min, data_min)
np.save(filename_max, data_max)