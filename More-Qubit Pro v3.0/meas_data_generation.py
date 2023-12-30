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

from SDPforVQE import generate_PauliStrList, Hamiltonian_global, Hamiltonian_matrix, ground_state
from SDPforVQE import N_meas_list_func, generate_meas_dataset, random_distribute




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
H_local = np.array( Hamiltonian_matrix(H_local_list, model_type) ) # Matrix representation of the local Hamiltonian of subsystems
H_global = np.array( Hamiltonian_matrix(H_global_list, model_type) ) # Matrix representation of the Hamiltonian of the whole system

ground_state_energy, ground_state_dm = ground_state(H_global) 
q_state = DensityMatrix(ground_state_dm) 

# Create a folder if it doesn't exist
meas_dataset_filename = "meas_dataset"
os.makedirs(meas_dataset_filename, exist_ok=True)
folder_name = os.path.join('meas_dataset/'+model_type, f"N={N}")
os.makedirs(folder_name, exist_ok=True)





num_data_point = 15 # number of N_meas that we select to run
N_meas_list = N_meas_list_func(100, 100000, num_data_point) # A list of number of measurement performed in all basis
num_of_shot = 100 # Number of repeatation of the experiment

# Generate the dataset for N_meas=N_meas_max number of measurements
N_meas_max = N_meas_list[-1]
data_full = []
for i in range(num_of_shot):   
    data_full.append(generate_meas_dataset(q_state, N_meas_max, N))
path = f'meas_dataset/{model_type}/N={N}/N{N}_Meas{N_meas_max}.npy'
np.save(path, data_full)

for N_meas in N_meas_list[:-1]:
    data = []
    num_meas_list = random_distribute(N_meas, N)
    for i in range(num_of_shot):   
        selected_dict = {}
        data_dict = data_full[i] # Get the data for the i-th shot of experiment
        for j, key in enumerate(data_dict):
            selected_strings = random.sample(data_dict[key], int(num_meas_list[j])) # Randomly select strings from the list for this key
            selected_dict[key] = selected_strings # Add the selected strings to the new dictionary
        data.append(selected_dict)
    path = f'meas_dataset/{model_type}/N={N}/N{N}_Meas{N_meas}.npy'
    np.save(path, data)