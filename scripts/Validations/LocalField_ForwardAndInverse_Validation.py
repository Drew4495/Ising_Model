from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
from typing import Dict, Tuple, Type
from src.src_Ising_ver2_1 import *




#======================================================================================================================#
#region
"""
Ground-truth generation helpers
"""
##############################
##############################

def generate_ground_truth(
    N: int,
    coupling_scale: float = 0.5,
    field_scale: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return symmetric *J_true* and local field *h_true*."""
    rng = np.random.RandomState(seed)
    J = rng.randn(N, N)
    J = 0.5 * (J + J.T)
    J *= coupling_scale
    np.fill_diagonal(J, 0)
    h = rng.randn(N) * field_scale
    return J, h

##############################
##############################
#endregion
#======================================================================================================================#




#======================================================================================================================#
#region
"""
Helpers functions
"""
##############################
##############################

# Function to plot dl_dJ and J (frobenius norm for dl_dJ, diff in frobenius norm for dl_dJ, diff in J)
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

# Show an example of the function
#dl_dJ = np.random.rand(100, 20, 20)
#J = np.random.rand(100, 20, 20)
#dl_dh = np.random.rand(100, 20, 1)
#h = np.random.rand(100, 20, 1)
#fig, axs = plot_dldJ_and_J_fit(dl_dJ, J, dl_dh, h)
#plt.show()


##############################
##############################
#endregion
#======================================================================================================================#




#======================================================================================================================#
#region
"""
Forward simulate ground-truth and save
"""
##############################
##############################

# Simulate Ground Truth: Beta = 1.0, all local_fields the same
# -------------------------------------------------------------
### Define parameters
conn_coeffs, local_fields = generate_ground_truth(N=20,
                                                  coupling_scale = 0.5,
                                                  field_scale = 0.3,
                                                  seed=42)
num_sim_steps = 1000
sim_measurements_dict = {"units_array": 1}

### Simulate
model_sim = Ising_Model()
sim_out = model_sim.simulate_ising(
    conn_coeffs=conn_coeffs,
    local_fields=local_fields,
    beta=1.0,
    num_sim_steps=num_sim_steps,
    num_iters_per_sim=1,
    num_burn_in_steps=0,
    sim_measurements_dict=sim_measurements_dict
)

### Save simulated data
filepath = f"data/simulated_data/forwardsim_beta1_localfields.pkl"
os.makedirs(os.path.dirname(filepath), exist_ok=True)
with open(filepath, "wb") as f:
    pickle.dump(sim_out, f)

##############################
##############################
#endregion
#======================================================================================================================#




#======================================================================================================================#
#region
"""
Inverse fit ground truth data and evaluate
"""
##############################
##############################

# Load in data and initialize parameters
# -------------------------------------------------------------
filepath = f"data/simulated_data/forwardsim_beta1_localfields.pkl"
with open(filepath, "rb") as f:
    gt_data = pickle.load(f)

time_series = gt_data["sim_measurements"]["units_array"]
beta = 1.0
hyper_df = pd.DataFrame([{"beta_inverse": beta}])
max_iter = 50000
lr = 0.01
log_settings_objective_function_dict = {
    "dl_dh": 1,
    "h": 1,
    "J": 1,
    "dl_dJ": 1
}



# Run inverse fit
# -------------------------------------------------------------
model_fit = Ising_Model()
fit_out = model_fit.fit_inverse_ising(
    func_ts=time_series,
    hyperparams_df_one_set=hyper_df,
    conn_coeffs=None,  # start J at zeros
    objective_function="MJPL_symmetric_localfield",
    optimization_method="gradient",
    max_iter=max_iter,
    learning_rate=lr,
    convergence_threshold=1e-4,
    log_settings_objective_function_dict=log_settings_objective_function_dict,
    skip_error_checks=False,
)


# Evaluate model fit
# -------------------------------------------------------------
# Add extra key to dict
fit_out["validation_estimates"] = {}

# Estimated params
J_est = fit_out["fitted_conn_coeffs"]
h_est = fit_out["fitted_local_fields"]

# True params
J_true = gt_data["params"]["conn_coeffs"]
h_true = gt_data["params"]["local_fields"]

# Evaluation metrics (upper‑triangle for J)
tri = np.triu_indices_from(J_true, k=1)
corr_J = np.corrcoef(J_true[tri], J_est[tri])[0, 1]
corr_h = np.corrcoef(h_true, h_est)[0, 1]
rmse_J = np.sqrt(np.mean((J_true[tri] - J_est[tri]) ** 2))
rmse_h = np.sqrt(np.mean((h_true - h_est) ** 2))

fit_out["validation_estimates"] = {
    "J_true": J_true,
    "h_true": h_true,
    "J_est": J_est,
    "h_est": h_est,
    "corr_J": corr_J,
    "corr_h": corr_h,
    "rmse_J": rmse_J,
    "rmse_h": rmse_h,
}

### Save fit output
filepath = f"data/simulated_data/InverseFit_MJPL_symmetric_localfield_beta1.pkl"
os.makedirs(os.path.dirname(filepath), exist_ok=True)
with open(filepath, "wb") as f:
    pickle.dump(fit_out, f)

### Plot estimates with helper function
dl_dJ = np.array(fit_out["objective_function_output_dict"]["storage_values_objective_function"]["dl_dJ"])
dl_dh = np.array(fit_out["objective_function_output_dict"]["storage_values_objective_function"]["dl_dh"])
J = np.array(fit_out["objective_function_output_dict"]["storage_values_objective_function"]["J"])
h = np.array(fit_out["objective_function_output_dict"]["storage_values_objective_function"]["h"])
fig, axs = plot_dldJ_and_J_fit(dl_dJ = dl_dJ,
                               J = J,
                               dl_dh = dl_dh,
                               h = h,
                               )

### Save figure and show
filepath = f"images/simulated_data/InverseFit_Paramaters_and_Gradients_MJPL_symmetric_localfield_beta1.png"
os.makedirs(os.path.dirname(filepath), exist_ok=True)
plt.savefig(filepath)
plt.show()


##############################
##############################
#endregion
#======================================================================================================================#




#======================================================================================================================#
#region
"""
Calculate Order Parameters and Plot
"""
##############################
##############################

### Load in fit_out data
filepath = f"data/simulated_data/InverseFit_MJPL_symmetric_localfield_beta1.pkl"
with open(filepath, "rb") as f:
    fit_out = pickle.load(f)

J_est = fit_out["fitted_conn_coeffs"]
h_est = fit_out["fitted_local_fields"]

### Define parameters for forward simulation
sim_measurements_dict = {
    "energy": 1,
    "magnetization": 1,
    "specific_heat": "global",
    "magnetic_susceptibility": "global"}

### Simulate
model = Ising_Model()
sim_out = model.simulate_ising(
    conn_coeffs=J_est,
    local_fields=h_est,
    beta=0.44,
    num_sim_steps=1000,
    num_iters_per_sim=1,
    num_burn_in_steps=0,
    sim_measurements_dict=sim_measurements_dict
)

## Now, let's do for a range of  betas, store the order parameters, and plot
betas = np.arange(0.1, 2.05, 0.05)
h_zeros = np.zeros(len(h_est))
specific_heat = np.zeros(len(betas))
magnetic_susceptibility = np.zeros(len(betas))
for i, beta in enumerate(betas):
    sim_out = model.simulate_ising(
        conn_coeffs=J_est,
        local_fields=h_est,
        beta=beta,
        num_sim_steps=1000,
        num_iters_per_sim=1,
        num_burn_in_steps=0,
        sim_measurements_dict=sim_measurements_dict
    )
    # Extract order parameters
    specific_heat[i] = sim_out["sim_measurements"]["specific_heat"]
    magnetic_susceptibility[i] = sim_out["sim_measurements"]["magnetic_susceptibility"]



### Extract and Plot Order Parameters
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(betas, specific_heat, label="Specific Heat")
axs[1].plot(betas, magnetic_susceptibility, label="Magnetic Susceptibility")
plt.show()

##############################
##############################
#endregion
#======================================================================================================================#


