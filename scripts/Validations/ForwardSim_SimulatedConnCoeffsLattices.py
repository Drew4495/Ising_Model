from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
from typing import Dict, Tuple, Type, Callable
from src.src_Ising_ver2_1 import *


#======================================================================================================================#
#region
"""
Helper Functions
"""
##############################
##############################

def plot_dldJ_and_J_fit(dl_dJ, J, dl_dh, h):
    """
    Plot the following:
    1) frobenius norm for dl_dJ
    2) diff in frobenius norm for dl_dJ
    3) frobenius norm for J
    4) diff in J,
    5) l2 norm for dl_dh
    6) diff in l2 norm for dl_dh
    7) l2 norm for h
    8) diff in h
    The figures do not necessarily have the same number of iterations, so the length of each subplot
    is dependent on that specific argument array
    :param dl_dJ: dl_dJ (3d matrix of dl_dJ matrices). Shape: (num_iterations - 1 maybe?, num_units, num_units)
    :param J: J (3d matrix of J matrices). Shape: (num_iterations, num_units, num_units)
    """
    # Error checks. Make sure 2nd and 3rd dimension of dl_dJ and J are the same
    if dl_dJ.shape[1] !=  dl_dJ.shape[2] or J.shape[1] != J.shape[2]:
        raise ValueError("dl_dJ and J must have the same number of units (2nd and 3rd dimension). "
                         "Make sure the first dimension is the num of iterations")
    # Initialize plotting
    fig, axs = plt.subplots(4, 2, figsize=(10, 15))

    # Calculate frobenius norm of dl_dJ over iterations
    dl_dJ_fro_norms =  np.linalg.norm(dl_dJ, axis=(1, 2), ord="fro")
    # Plot frobenius norm of dl_dJ over iterations
    axs[0,0].plot(np.arange(dl_dJ.shape[0]), dl_dJ_fro_norms)
    axs[0,0].set_title("dl_dJ | Frobenius norm over iterations")
    axs[0,0].set_xlabel("Iteration")
    axs[0,0].set_ylabel("Frobenius norm of dl_dJ")
    axs[0,0].grid()
    # Plot difference in frobenius norm of dl_dJ over iterations
    axs[0,1].plot(np.arange(dl_dJ.shape[0] - 1), np.diff(dl_dJ_fro_norms))
    axs[0,1].set_title("dl_dJ | Difference in Frobenius norm over iterations")
    axs[0,1].set_xlabel("Iteration")
    axs[0,1].set_ylabel("Difference in Frobenius norm of dl_dJ")
    axs[0,1].grid()

    # Calculate frobenius norm of J over iterations
    J_fro_norms = np.linalg.norm(J, axis=(1, 2), ord="fro")
    # Plot frobenius norm of J over iterations
    axs[1,0].plot(np.arange(J.shape[0]), J_fro_norms)
    axs[1,0].set_title("J | Frobenius norm over iterations")
    axs[1,0].set_xlabel("Iteration")
    axs[1,0].set_ylabel("Frobenius norm of J")
    axs[1,0].grid()
    # Plot Fro norm of difference in J over iterations
    axs[1,1].plot(np.arange(J.shape[0] - 1), np.linalg.norm(np.diff(J, axis=0), axis=(1,2), ord="fro"))
    axs[1,1].set_title("J | Difference over iterations")
    axs[1,1].set_xlabel("Iteration")
    axs[1,1].set_ylabel("Difference in J")
    axs[1,1].grid()

    # Plot l2 norm of dl_dh over iterations
    dl_dh_l2_norms = np.linalg.norm(dl_dh, axis=1, ord=2)
    # Plot frobenius norm of dl_dh over iterations
    axs[2,0].plot(np.arange(dl_dh.shape[0]), dl_dh_l2_norms)
    axs[2,0].set_title("dl_dh | L2 norm over iterations")
    axs[2,0].set_xlabel("Iteration")
    axs[2,0].set_ylabel("L2 Norm")
    axs[2,0].grid()
    # Plot difference in dl_dh L2 norm over iterations
    axs[2,1].plot(np.arange(dl_dh.shape[0] -1), np.diff(dl_dh_l2_norms, axis=0))
    axs[2,1].set_title("dl_dh | Difference in L2 norm over iterations")
    axs[2,1].set_xlabel("Iteration")
    axs[2,1].set_ylabel("Diff in L2 norm of dl_dh")
    axs[2,1].grid()

    # Plot l2 norm of h over iterations
    h_l2_norms = np.linalg.norm(h, axis=1, ord=2)
    # Plot frobenius norm of h over iterations
    axs[3,0].plot(np.arange(h.shape[0]), h_l2_norms)
    axs[3,0].set_title("h | L2 norm over iterations")
    axs[3,0].set_xlabel("Iteration")
    axs[3,0].set_ylabel("L2 Norm")
    axs[3,0].grid()
    # Plot L2 norm of differences in h over iterations
    axs[3,1].plot(np.arange(h.shape[0] -1), np.linalg.norm(np.diff(h, axis=0), axis=1, ord=2))
    axs[3,1].set_title("h | Difference over iterations")
    axs[3,1].set_xlabel("Iteration")
    axs[3,1].set_ylabel("Diff in h")
    axs[3,1].grid()

    plt.tight_layout()
    return fig, axs




##############################
##############################
#endregion
#======================================================================================================================#




#======================================================================================================================#
#region
"""
Forward Simulate, save & plot order parameters
"""
##############################
##############################

### Load dataset
J = np.load("data/simulated_data/ConnectivityMatrices_J/Constant_Weights/J_ConstantWeights1_20x20.npy")

### Define parameters for forward simulation
num_units = J.shape[0]
num_sim_steps = 1000
h_zeros = np.zeros((num_units,))
sim_measurements_dict = {
    "energy": 1,
    "magnetization": 1,
    "specific_heat": "global",
    "magnetic_susceptibility": "global"
}
#betas = np.concatenate(
#    ([0.01], np.arange(0.05, 2.05, 0.05))
#)
betas = np.arange(0.2, 0.7, 0.02)
specific_heat = np.zeros(len(betas))
magnetic_susceptibility = np.zeros(len(betas))
magnetization = np.zeros(len(betas))

### Forward simulate
for i, beta in enumerate(betas):
    print(f"\nSimulating beta = {beta}")
    start_time = time.time()
    model = Ising_Model()
    sim_out = model.simulate_ising(
        conn_coeffs=J,
        local_fields=None,
        beta=beta,
        num_sim_steps = num_sim_steps,
        num_iters_per_sim=50,
        num_burn_in_steps=1000,
        sim_measurements_dict=sim_measurements_dict
    )
    break
    # Save order parameters
    magnetization = sim_out["sim_measurements"]["magnetization"]
    specific_heat[i] = sim_out["sim_measurements"]["specific_heat"]
    magnetic_susceptibility[i] = sim_out["sim_measurements"]["magnetic_susceptibility"]
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

### Save results


### Extract an plot order parameters
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(betas, specific_heat, label="Specific Heat")
axs[1].plot(betas, magnetic_susceptibility, label="Magnetic Susceptibility")
plt.show()



##############################
##############################
#endregion
#======================================================================================================================#
