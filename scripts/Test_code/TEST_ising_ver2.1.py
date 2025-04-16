import numpy as np
import pandas as pd
from scipy.io import loadmat
from src.src_Ising_ver2_1 import *


### Load in all files
wd = "/Users/aburns/Codebook/Projects/uic_eib_lld_damek/"
data_path = f"{wd}data/"
mat_dict = loadmat(f"{data_path}/igor_test_data/lambd_beta2.mat")
lambda_beta_values = mat_dict["lambd_beta2"]
mat_dict = loadmat(f"{data_path}/igor_test_data/bold.mat")
FC_timeseries = mat_dict["ts_sz"]       #164 times, 132 regions, 3 subjects
FC_timeseries = np.transpose(FC_timeseries, axes=(1,0,2))
mat_dict = loadmat(f"{data_path}/igor_test_data/fc.mat")
FC = mat_dict["fc"]
mat_dict = loadmat(f"{data_path}/igor_test_data/normsc.mat")
SC = mat_dict['normsc']

### Preprocess lambda_beta_values (there are some repeats)
_, unique_indices = np.unique(lambda_beta_values, axis=0, return_index=True)                    # Find unique rows and their corresponding indices
unique_rows = lambda_beta_values[np.sort(unique_indices)]                                       # Extract the unique rows using the indices
lambda_beta_values = unique_rows[np.lexsort((unique_rows[:, 1], unique_rows[:, 0]))]

# Prepare data for "fit and optimize"
df_lambda_beta = pd.DataFrame(lambda_beta_values, columns = ["lambda", "beta_inverse"])
df_forward_hyperparams = df_lambda_beta[["beta_inverse"]].iloc[0:10]
df_forward_hyperparams.columns = ["beta_forward"]
FC_ts = FC_timeseries[:,:,0]
SC_0 = SC[:,:,0]
df_test_lambda_beta = df_lambda_beta.iloc[0:2,0:2]

### Instantiate model
test_IsingModel = Ising_Model(
    func_ts = FC_ts,
    conn_coeffs = SC_0
)

### Settings for inverse and forward issing
fit_inverse_ising_param_dict = {
        "objective_function": "MJPL",
        "optimization_method": "gradient",
        "max_iter": 50000,
        "learning_rate": 0.1,
        "log_settings_dict": None,
        "skip_error_checks": True
    }

simulate_ising_param_dict = {
        "num_sim_steps": 200,
        #"beta": 1.0,  # Must be off for fit_an_optimize
        "num_iters_per_sim": 1,
        "num_burn_in_steps": 0,
        "sim_measurements_dict": None,
        "empirical_FC": None,
        #"verbose_printouts": True,
        "verbose_num_sim_steps": True
    }

FC_corr_log_settings_dict = {
    "FC_corr": 1
}

log_settings_objective_function_dict = {
    #"dl_dJ": 1,
}

# Fit and optimize
test_result_dict = test_IsingModel.fit_and_optimize_inverse_ising(
    func_ts = FC_ts,
    #hyperparams_df = df_test_lambda_beta,
    df_hyperparams_inverse = df_test_lambda_beta,
    df_hyperparams_forward = df_forward_hyperparams,
    conn_coeffs= SC_0,
    fit_inverse_ising_param_dict = fit_inverse_ising_param_dict,
    simulate_ising_param_dict = simulate_ising_param_dict,
    FC_corr_log_settings_dict = FC_corr_log_settings_dict,
    log_settings_objective_function_dict = log_settings_objective_function_dict
)
