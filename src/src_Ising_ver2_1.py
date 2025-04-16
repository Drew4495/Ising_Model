##############################
##############################
#region Import libraries
"""
Import libraries
"""
##############################
##############################

import numpy as np
import time
import pandas as pd
from scipy.stats import zscore
import warnings
import numbers


##############################
##############################
#endregion
##############################
##############################




##############################
##############################
#region Thoughts on improving
"""
Thoughts on improving
"""
##############################
##############################

"""
ABSOLUTE FIXES! DO RIGHT AWAY!
1. Add two different betas to the hyperparam dict (Beta_inverse, beta_forward) and record the FC_corrs this way
    Make sure when I store SC_corr, it stores the SC_Corr for all the beta_inverse and lambda values even with different
    beta_forwards

Improvement ideas
1. Add timing printouts in fit_and_optimize
2. Add saving options for each hyperparameter in case it is interrupted
3. Add "interval" for MH step printout in Ising simulation
4. Check to see if empirical_FC actually needs to be inputted. Probably delete this by default
5. Add helper method to calculate EIR (globally, regionally, and any parcellation)
6. I could probably save time by not storing so much in fitted_cache in fit_and_optimize. Probably just need to store fitted_J
7. The _get_df_hyperparams_FCcorr_SCcorr funciton could be replace with a cusotm funciton and df.apply() for speed boosts


Thoughts on correcting why this version isn't working:
1. Go through a debugging step first and store dl_dJ and FC_corr intervals = num_sims / intervals = 200 
    so maybe num_sims = 20000, intervals = 100
2. I am calculating FC_empirical directly on the empirical time series not the binarized version 
     If i have to do this, this is a weakness of the model I need to write down

"""

##############################
##############################
#endregion
##############################
##############################







##############################
##############################
#region Ising EIB Model Class
"""
Ising EIB Model Class
"""
##############################
##############################

class Ising_Model:
    def __init__(self,
                 func_ts = None,
                 conn_coeffs = None,
                 random_state=111):
        """
        1. Initialize func_ts and conn_coeffs.
        2. Initialize random_state.
        3. We do NOT binarize timeseries here; user can decide how to do that externally.
        """
        self.func_ts = func_ts
        self.conn_coeffs = conn_coeffs
        self.rs = np.random.RandomState(seed=random_state) # Is this necessary?


    ###################################
    # Ising simulation  #
    ###################################

    def simulate_ising(self,
                       conn_coeffs,
                       num_sim_steps = 500,
                       beta = 1.0,
                       num_iters_per_sim = 1,
                       num_burn_in_steps = 0,
                       sim_measurements_dict = None,
                       empirical_FC = None,
                       #verbose_printouts = False,
                       verbose_num_sim_steps = False):
        """
        Simulates an Ising model given a single beta value (inverse temperature)
        and a connectivity matrix 'conn_coeffs'.

        Parameters
        ----------
        conn_coeffs : np.ndarray, shape (N, N)
            A NxN symmetric matrix of connectivity coefficients for N units.
            If None, uses self.conn_coeffs (if available).
        num_sim_steps : int
            Number of simulation steps at which we'll measure data.
        beta : float, optional
            One value for beta (inverse temperature). Default = 1.0.
        num_iters_per_sim : int, optional
            Number of Metropolis-Hastings sub-iterations to perform for each
            simulation step. Default = 1.
        num_burn_in_steps : int, optional
            Number of MH steps before recording measurements. Default = 0.
        sim_measurements_dict : dict, optional
            Dict specifying which measurements to calculate and how often.
            E.g. {"energy": 1, "magnetization": 2, "units_array": 10}
            means:
              - store 'energy' every 1 step,
              - store 'magnetization' every 2 steps,
              - store 'units_array' every 10 steps.
        empirical_FC : np.ndarray, shape (N, N), optional
            Optional real/empirical FC if we want to compute correlation
            (e.g., "FC_corr"). If "FC_corr" is in sim_measurements_dict,
            we should pass this. Default = None.
        verbose_printouts : bool
            If True, prints additional debug info.
        verbose_num_sim_steps : bool
            If True, prints progress every sim step.

        NOTE TO OWNER: UPDATE VERBOSE SIM STEPS FOR MORE CUSTOMIZATION

        Returns
        -------
        results_dict : dict
            {
              "final_units_array":  Final spin state after all steps,
              "params":             dictionary of the function parameters,
              "sim_measurements":   dictionary of lists of stored measurements
            }
        """
        params_dict = locals()

        # Error Handling and basic argument checks
        # -------------------------------------------------------------
        if conn_coeffs is None:
            if self.conn_coeffs is None:
                raise ValueError("No connectivity matrix provided or found in self.conn_coeffs.")
            conn_coeffs = self.conn_coeffs

        if sim_measurements_dict is None:
            sim_measurements_dict = {}

        # Basic setup (Define variables and storage_dicts)
        # -------------------------------------------------------------
        ### Define variables and storage dicts
        params_dict = {
            "num_sim_steps": num_sim_steps,
            "beta": beta,
            "num_iters_per_sim": num_iters_per_sim,
            "num_burn_in_steps": num_burn_in_steps,
            "sim_measurements_dict": sim_measurements_dict,
            "empirical_FC_provided": (empirical_FC is not None)
        }

        num_units = conn_coeffs.shape[0]
        storage_dict = {
            measurement_name: [] for measurement_name in sim_measurements_dict.keys()
        }

        # MCMC simulation
        # -------------------------------------------------------------
        ### Initialize random unit state
        units_array = self._get_random_units_array(num_units)

        ### Burn-in
        current_energy = None
        for _ in range(num_burn_in_steps):
            units_array, current_energy = self._make_MH_sweep(
                units_array = units_array,
                conn_coeffs = conn_coeffs,
                beta = beta,
                current_energy = current_energy
            )

        ### Sampling - store measurements each sim_step
        for sim_step in range(num_sim_steps):
            # Run iters_per_sim
            for _ in range(num_iters_per_sim):
                [units_array, current_energy] = self._make_MH_sweep(
                    units_array = units_array,
                    conn_coeffs = conn_coeffs,
                    beta = beta,
                    current_energy = current_energy)

            # Store sim measurement by interval
            for measurement_name, interval in sim_measurements_dict.items():
                if (sim_step % interval) == 0:
                    measurement_value = self._calculate_sim_measurement(
                        measurement_name,
                        units_array = units_array,
                        conn_coeffs = conn_coeffs,
                        beta = beta,
                        current_energy = current_energy
                    )
                    storage_dict[measurement_name].append(measurement_value)

            # Optional printing to track progress
            if verbose_num_sim_steps and (sim_step % 50 == 0):
                print(f"MH step {sim_step}/{num_sim_steps} completed")

        ### Convert list of arrays to matrices
        if "units_array" in storage_dict.keys():
            storage_dict["units_array"] = np.array(storage_dict["units_array"]).T

        # Build final results dict and return
        # -------------------------------------------------------------
        results_dict = {"final_units_array": units_array,
                        "params": params_dict,
                        "sim_measurements": storage_dict}

        return results_dict



    ###################################
    # Inverse Ising and sub-functions#
    ###################################

    def fit_and_optimize_inverse_ising(self,
                                       func_ts,
                                       df_hyperparams_inverse,
                                       df_hyperparams_forward,
                                       #hyperparams_df,
                                       conn_coeffs = None,
                                       fit_inverse_ising_param_dict = None,
                                       simulate_ising_param_dict = None,
                                       FC_corr_log_settings_dict = None,
                                       log_settings_dict = None,
                                       log_settings_objective_function_dict = None,
                                       ):

        # Ensure param dicts have defaults if none
        # -------------------------------------------------------------
        if fit_inverse_ising_param_dict is None:
            fit_inverse_ising_param_dict = {}

        if simulate_ising_param_dict is None:
            simulate_ising_param_dict = {}

        # Instantiate variables and storage dicts
        # -------------------------------------------------------------
        # Create a consolidated hyperparams df with unique ids for each inverse combo and each inverse-forward combo
        hyperparams_df = self._create_hyperparam_df_inverse_fit_and_optimize(
            df_inverse = df_hyperparams_inverse,
            df_forward = df_hyperparams_forward
        )

        # Create storage dictionary where each key is an inverse hyperparam combo and the forward results are within
        storage_dict = self._create_storage_dict_from_merged_df_inverse_fit_and_optimize(hyperparams_df)
        storage_dict["df_hyperparams_all"] = hyperparams_df

        # Create log_settings_dict if doesn't exist
        if log_settings_dict is None:
            log_settings_dict = {}

        # Ensure "keep_units_history" is set to False or raise warning
        if log_settings_dict.get("keep_units_history") is True:
            # Raise warning
            warnings.warn("log_settings_dict['keep_units_history'] is set to True in fit_and_optimize_inverse_ising."
                          "Storing units/spin history for each hyperparameter combination and consumes significant "
                          "memory. Consider setting to False.")
        else:
            # Set to False to minimize storage overhead
            log_settings_dict["keep_units_history"] = False


        # Start Global Time for whole pipeline (inverse and forward)
        # -------------------------------------------------------------
        start_time_all_fit_and_optimize = time.time()

        # Binarize func_ts
        # -------------------------------------------------------------
        func_ts_bin = self._binarize_func_ts(func_ts)

        # 1) Fit inverse ising,
        # -------------------------------------------------------------
        # Maintain a cache/set so we only fit each (lambda, beta_inverse) once
        fitted_cache = {} # (lam, beta_inv) -> dict with {fitted_conn_coeffs, SC_corr, etc.}

        # Start time for inverse fitting for all hparams
        start_time_all_inverse_fitting = time.time()

        for idx, hyperparam_row in hyperparams_df.iterrows():
            lam = hyperparam_row["lambda"]
            beta_inverse = hyperparam_row["beta_inverse"]
            id_inv = hyperparam_row["ID_inverse_hyperparams"]

            # Skip fitting is lambda and beta_inverse have already been used
            if (lam, beta_inverse) in fitted_cache:
                continue

            # Select hyperparams row for fitting
            row_for_fit = hyperparam_row[["lambda", "beta_inverse"]].to_frame().T
            row_for_fit.columns = ["lambda", "beta_inverse"]

            # Fit inverse ising for one param combo and store
            fitted_results_inverse_ising_dict = self.fit_inverse_ising(
                func_ts = func_ts_bin,
                hyperparams_df_one_set = row_for_fit,
                conn_coeffs = conn_coeffs,
                log_settings_objective_function_dict = log_settings_objective_function_dict,
                **fit_inverse_ising_param_dict
            )

            # Calc SC_corr
            fitted_conn_coeffs = fitted_results_inverse_ising_dict["fitted_conn_coeffs"]
            fitted_con_coeffs_signmatched = (
                    np.abs(fitted_conn_coeffs.flatten()) * np.sign(conn_coeffs.flatten())
            )
            SC_corr = np.corrcoef(conn_coeffs.flatten(),
                                  fitted_con_coeffs_signmatched
                                  )[0,1]

            # Store in the cache
            fitted_cache[(lam, beta_inverse)] = {
                "fitted_conn_coeffs": fitted_conn_coeffs,
                "SC_corr": SC_corr,
                "fitted_results_inverse_ising": fitted_results_inverse_ising_dict
            }

            # Also store in storage_dict
            storage_dict["hyperparameters_results"][id_inv]["SC_corr"] = SC_corr
            storage_dict["hyperparameters_results"][id_inv]["fitted_results_inverse_ising"] = fitted_results_inverse_ising_dict

        # Get elapsed time for all inverse fittings
        elapsed_time_all_inverse_fitting = self._get_elapsed_time(start_time = start_time_all_inverse_fitting)


        # 2) Run ising simulation and calc FC_corr for each hyperparam combo
        # -------------------------------------------------------------
        # Calculate empirical FC_corr (may consider changing this to non-binarized version)
        empirical_FC = self._calc_FC(
            func_time_series = func_ts_bin,
            allow_self_connections = False
        )

        # Start time for forward sim of all hparams
        start_time_all_forward_sims = time.time()

        # Loop through each forward sim hparam combo
        for idx, hyperparam_row in hyperparams_df.iterrows():
            # Extract params
            lam = hyperparam_row["lambda"]
            beta_inverse = hyperparam_row["beta_inverse"]
            beta_forward = hyperparam_row["beta_forward"]
            id_inv = hyperparam_row["ID_inverse_hyperparams"]
            id_fwd = hyperparam_row["ID_forward_hyperparams"]

            # Get fitted J from cached
            fitted_conn_coeffs = fitted_cache[(lam, beta_inverse)]["fitted_conn_coeffs"]

            # DELETE: Format row as df
            # DELETE: hyperparam_row = hyperparam_row.to_frame().T

            # Prepare the simulate_ising_kwargs and params
            # Ensure units/spin history is kept at each iteration
            if "units_array" not in simulate_ising_param_dict:
                simulate_ising_param_dict["sim_measurements_dict"] = {
                    "units_array": 1
                }
            # Ensure beta  and empirical_FC is not in ising_param_dict
            if "beta" in simulate_ising_param_dict:
                del simulate_ising_param_dict["beta"]
            if "empirical_FC" in simulate_ising_param_dict:
                del simulate_ising_param_dict["empirical_FC"]

            # Run Ising simulation (forward problem)
            results_ising_simulation_results_dict = self.simulate_ising(
                conn_coeffs = fitted_conn_coeffs,
                beta = beta_forward,
                empirical_FC = empirical_FC,
                **simulate_ising_param_dict
            )

            # Set log_settings_dict for FC_corr is "whole" if not otherwise specified
            if FC_corr_log_settings_dict is None:
                FC_corr_log_settings_dict = {"FC_corr": "whole"}

            # Calculate FC_corr
            func_ts_sim = results_ising_simulation_results_dict["sim_measurements"]["units_array"]
            FC_corr_dict = self._calc_FC_corr(
                FC_empirical = empirical_FC,
                func_time_series = func_ts_sim,
                log_settings_dict = FC_corr_log_settings_dict
            )

            # Delete spin history unless specified
            results_ising_simulation_results_dict["sim_measurements"]["units_array"] = "deleted to preserve memory"

            # Store results
            storage_dict["hyperparameters_results"][id_inv]["sim_forward_results"][id_fwd]["FC_corr"] = FC_corr_dict["FC_corr_whole"]
            storage_dict["hyperparameters_results"][id_inv]["sim_forward_results"][id_fwd]["FC_corr_detailed"] = FC_corr_dict
            storage_dict["hyperparameters_results"][id_inv]["sim_forward_results"][id_fwd]["results_ising_simulation"] = results_ising_simulation_results_dict


        # 3) Choose optimized FC_corr and SC_corr
        # -------------------------------------------------------------
        # Prepare hyperparams and FC_corr and SC_corr in one dataframe
        df_hyperparams_FCcorr_SCcorr = self._get_df_hyperparams_FCcorr_SCcorr(
            fitted_and_optimized_inverse_ising_storage_dict = storage_dict
        )

        # Find opt FC_corr for each inverse fitting and store
        cache = {}
        for idx, row in df_hyperparams_FCcorr_SCcorr.iterrows():
            # Extract params
            lam = row["lambda"]
            beta_inverse = row["beta_inverse"]
            id_inv = row["ID_inverse_hyperparams"]

            # Skip if inverse fit already calculated
            if id_inv in cache:
                continue

            # subset df
            df_subset_inv = df_hyperparams_FCcorr_SCcorr[
                df_hyperparams_FCcorr_SCcorr["ID_inverse_hyperparams"] == id_inv
                ]


            # Find max id
            id_fwd_optimal_beta_per_inv_fit = self._choose_hyperparams_FCcorr_SCcorr(
                df_hyperparams_FCcorr_SCcorr = df_subset_inv,
                id_colname = "ID_forward_hyperparams",
                method = "average"
            )


            # Store in dict
            storage_dict["hyperparameters_results"][id_inv]["optimal_forward_sim_results"] = {
                "opt_FC_corr": storage_dict["hyperparameters_results"][id_inv]["sim_forward_results"][id_fwd_optimal_beta_per_inv_fit]["FC_corr"],
                "opt_forward_id": id_fwd_optimal_beta_per_inv_fit,
                "opt_beta_forward": storage_dict["hyperparameters_results"][id_inv]["sim_forward_results"][id_fwd_optimal_beta_per_inv_fit]["hyperparameters_forward"]["beta_forward"]
            }


        # Find optimized FC_corr and SC_corr based off choice of optimization method
        id_unique_optimal_hyperparams = self._choose_hyperparams_FCcorr_SCcorr(
            df_hyperparams_FCcorr_SCcorr = df_hyperparams_FCcorr_SCcorr,
            id_colname = "ID_unique_hyperparams",
            method = "average"
        )

        id_inv_optimal_hyperparams = df_hyperparams_FCcorr_SCcorr.loc[
            df_hyperparams_FCcorr_SCcorr["ID_unique_hyperparams"] == id_unique_optimal_hyperparams,
            ["ID_inverse_hyperparams"]
        ].iloc[0,0]

        # Create dict for optimized hyperparams
        optimal_hyperparams_dict = {
            "fitted conn_coeffs (rssc)": storage_dict["hyperparameters_results"][id_inv_optimal_hyperparams]["fitted_results_inverse_ising"]["fitted_conn_coeffs"],
            "FC_corr": storage_dict["hyperparameters_results"][id_inv_optimal_hyperparams]["optimal_forward_sim_results"]["opt_FC_corr"],
            "SC_corr": storage_dict["hyperparameters_results"][id_inv_optimal_hyperparams]["SC_corr"],
            "opt_hyperparams": df_hyperparams_FCcorr_SCcorr.loc[
                df_hyperparams_FCcorr_SCcorr["ID_unique_hyperparams"] == id_unique_optimal_hyperparams,
                ["lambda", "beta_inverse", "beta_forward"]
                ]
        }

        # 4) Make final result dictionary
        # -------------------------------------------------------------
        storage_dict["optimal_hyperparams"] = optimal_hyperparams_dict
        storage_dict["df_hyperparams_all"] = df_hyperparams_FCcorr_SCcorr
        result_dict = storage_dict

        return result_dict


    def fit_inverse_ising(self,
                          func_ts,
                          hyperparams_df_one_set,
                          conn_coeffs = None,
                          objective_function = "MJPL_with_conn_regularization",
                          optimization_method = "gradient",
                          max_iter = 50000,
                          learning_rate = 0.1,
                          convergence_threshold = None,
                          log_settings_dict = None,
                          log_settings_objective_function_dict = None,
                          skip_error_checks = False):
        """
        Main entry point for fitting an inverse Ising model with a chosen objective function
        and optimization method.

        Parameters
        ----------
        func_ts : np.ndarray
            Binarized time series data, shape = (N, T).
        hyperparams_df_one_set : pd.DataFrame or dict
            Must have exactly one row (or be a dict)
        conn_coeffs : np.ndarray, optional
            Empirical or structural connectivity matrix, shape = (N, N).
        objective_function : str
            e.g., "MJPL".
        optimization_method : str
            e.g., "gradient".
        max_iter : int
            Maximum iterations for optimization.
        learning_rate : float
            Step size for gradient ascent.
        convergence_threshold : float or None
            If not None, used in convergence checks (optional).
        log_settings_dict : dict or None
            e.g. {"time": 1, "log_settings_dict_objective_function": {"dl_dJ": 10, "J": 50}}
        skip_error_checks : bool
            If True, skip input checks (e.g. for binarization, shape, etc.).

        Returns
        -------
        return_dict : dict
        """

        # Error Handling
        # -------------------------------------------------------------
        if not skip_error_checks:
            # Ensure func_ts is provided and raise note if conn_coeffs is not set
            if func_ts is None:
                raise ValueError("func_ts must be provided in 'fit_inverse_ising'")

            # Check that func_ts is binarized or raise error
            unique_vals = np.unique(func_ts)
            if len(unique_vals) != 2:
                raise ValueError(f"func_ts in 'fit_inverse_ising' is not binarized. "
                                 f"unique values are {unique_vals}")

            #Ensure hyperparams_df_one_set only has one row of values
            if hyperparams_df_one_set.shape[0] != 1:
                raise ValueError("hyperparams_df_one_set in 'fit_inverse_ising' does not have one row")

        # Instantiate variables and storage dicts
        # -------------------------------------------------------------
        # Instantiate log_settings
        if log_settings_dict is None:
            log_settings_dict = {}
        storage_dict = {
            var_name: [] for var_name in log_settings_dict
        }

        # Choose and run Objective Function and Optimization Method
        # -------------------------------------------------------------
        start_time_obj_fxn = time.time()
        objective_function_output_dict = self._choose_and_optimize_objective_function(
            objective_function=objective_function,
            optimization_method=optimization_method,
            func_ts = func_ts,
            conn_coeffs = conn_coeffs,
            hyperparams_df_one_set=hyperparams_df_one_set,
            max_iter = max_iter,
            learning_rate = learning_rate,
            convergence_threshold = convergence_threshold,
            log_settings_dict = log_settings_objective_function_dict
        )
        if "time" in log_settings_dict:
            elapsed_time_obj_fxn = self._get_elapsed_time(start_time_obj_fxn)
        else:
            elapsed_time_obj_fxn = "Not measured. To measure and store set log_settings_dict['time'] = 1."

        # Return
        # -------------------------------------------------------------
        return_dict = {
            "fitted_conn_coeffs": objective_function_output_dict["J_fitted"],
            "fitted_local_fields": objective_function_output_dict.get("h_fitted", None),
            "objective_function_output_dict": objective_function_output_dict,
            "elapsed_time_obj_fxn": elapsed_time_obj_fxn
        }

        return return_dict


    def _choose_and_optimize_objective_function(self,
                                                objective_function,
                                                optimization_method,
                                                func_ts,
                                                conn_coeffs,
                                                hyperparams_df_one_set,
                                                max_iter,
                                                beta_colname = "beta_inverse",
                                                lambda_colname = "lambda",
                                                learning_rate=0.1,
                                                convergence_threshold=None,
                                                log_settings_dict = None
                                                ):
        """
        Based on the chosen objective_function and optimization_method,
        calls the appropriate sub-routine.
        """
        # MJPL with gradient ascent/descent
        # -------------------------------------------------------------
        if objective_function == "MJPL" and optimization_method == "gradient":
            # Unpack beta and lambda
            beta = hyperparams_df_one_set[beta_colname].item()
            objective_function_output_dict = self._MJPL_gradient(
                ts = func_ts,
                J = conn_coeffs,
                beta = beta,
                max_iter = max_iter,
                lr = learning_rate,
                log_settings_dict = log_settings_dict
            )
            return objective_function_output_dict

        # MJPL with connectivity (structural) regularization with gradient ascent/descent
        # -------------------------------------------------------------
        elif objective_function == "MJPL_with_conn_regularization" and optimization_method == "gradient":
            # Unpack beta and lambda
            beta, lambda_val = hyperparams_df_one_set[beta_colname].item(), hyperparams_df_one_set[lambda_colname].item()

            # Call objective function
            objective_function_output_dict = self._MJPL_conn_regularization_gradient(
                ts = func_ts,
                J = conn_coeffs,
                beta = beta,
                lambda_val = lambda_val,
                max_iter = max_iter,
                lr = learning_rate,
                convergence_threshold = convergence_threshold,
                log_settings_dict = log_settings_dict
            )
            return objective_function_output_dict


        # MJPL with local field (symmetric, no structural regularization)
        elif objective_function == "MJPL_symmetric_localfield" and optimization_method == "gradient":
            # 1) Extract beta (and any other relevant hyperparams) from hyperparams_df_one_set:
            beta_val = hyperparams_df_one_set[beta_colname].item()

            # 2) Decide how you want to initialize h and J.
            #    Option A: Just start them at zero (common choice):
            num_units = func_ts.shape[0]
            h_init = np.zeros(num_units)
            J_init = np.zeros((num_units, num_units))

            # 3) Call gradient method
            objective_function_output_dict = self._MJPL_Symmetric_LocalField_gradient(
                ts = func_ts,
                h = h_init,
                J = J_init,
                beta = beta_val,
                max_iter = max_iter,
                lr = learning_rate,
                convergence_threshold = convergence_threshold,
                log_settings_dict = log_settings_dict
            )
            return objective_function_output_dict

        # Error Handling - if an objective function and optimization method aren't recognized
        # -------------------------------------------------------------
        else:
            raise NotImplementedError(
                f"Objective function '{objective_function}' "
                f"with method {optimization_method} is not implemented"
            )


    def _get_df_hyperparams_FCcorr_SCcorr(self, fitted_and_optimized_inverse_ising_storage_dict):
        # Add two columns to df_hyperparams
        results_df = fitted_and_optimized_inverse_ising_storage_dict["df_hyperparams_all"]
        results_df[["FC_corr", "SC_corr"]] = None

        # Redo below to fill in df
        for idx, row in results_df.iterrows():
            id_inv = row["ID_inverse_hyperparams"]
            id_fwd = row["ID_forward_hyperparams"]
            results_inverse = fitted_and_optimized_inverse_ising_storage_dict["hyperparameters_results"][id_inv]
            SC_corr = results_inverse["SC_corr"]
            results_df.loc[idx,"SC_corr"] = SC_corr
            FC_corr = results_inverse["sim_forward_results"][id_fwd]["FC_corr"]
            results_df.loc[idx, "FC_corr"] = FC_corr

        return results_df

    def _choose_hyperparams_FCcorr_SCcorr(self,
                                          df_hyperparams_FCcorr_SCcorr,
                                          id_colname,
                                          method = "average"
                                          ):
        df = df_hyperparams_FCcorr_SCcorr.copy()

        if method == "average":
            # Create average column
            df["avg_FCcorr_SCcorr"] = (( df["FC_corr"] +
                                          df["SC_corr"] )
                                         / 2)
            # Get id based on max of average
            max_idx = df["avg_FCcorr_SCcorr"].idxmax()
            max_id = df.loc[max_idx, id_colname]
            return max_id

        else:
            raise ValueError("Method not recognized in _choose_hyperparams_FCcorr_SCcorr")




    ###################################
    # Likelihood and optimization functions#
    ###################################
    def _MJPL_gradient(self,
                        ts,
                        J,
                        beta,
                        max_iter,
                        lr,
                        log_settings_dict = None):
        # Error Handling
        # -------------------------------------------------------------
        # Raise error if ts is not binarized.

        # Instantiate storage dicts
        # -------------------------------------------------------------
        if log_settings_dict is None:
            log_settings_dict = {}
        storage_dict = {
            var_name: [] for var_name in log_settings_dict
        }

        # Instantiate variables for optimization
        # -------------------------------------------------------------
        # Calculate corr(UNDERSTAND WHY I AM DOING THIS)
        num_units, num_timepoints = ts.shape
        corr = (ts @ ts.T) / num_timepoints

        # Copy empirical J and create a zeroed J to start optimization
        J_empirical = J.copy()
        np.fill_diagonal(J_empirical, 0)
        J = np.zeros((num_units, num_units))

        # Run gradient ascent for all iterations
        # -------------------------------------------------------------
        for iter_num in range(max_iter):
            # Store current J for convergence check
            J_prev = np.copy(J)

            # Compute gradient
            try:
                dl_dJ = (-corr) + \
                        ((0.5 * ts @ np.tanh(beta * (J @ ts)).T) / num_timepoints) + \
                        ((0.5 * (ts @ np.tanh(beta * (J @ ts)).T).T) / num_timepoints)
                dl_dJ = dl_dJ - np.diag(np.diag(dl_dJ))
            except Warning as e:
                error_info = {
                    "error": str(e),
                    "iteration num": iter_num,
                    "J": J,
                    "dl_dJ": dl_dJ,
                    "func_ts": ts
                }
                print(str(e))
                return error_info

            # Store variables if required
            variables_dict = {
                "dl_dJ": dl_dJ,
                "J": J
            }

            for var_name, interval in log_settings_dict.items():
                if (iter_num % interval) == 0:
                    storage_dict[var_name].append(variables_dict[var_name])

            # Update J Matrix
            J = J - (lr * dl_dJ)

            # Break if convergence is achieved
            if iter_num > 0 and np.allclose(J, J_prev):
                break

        # Return
        # -------------------------------------------------------------
        # Consolidate J output and storage values
        return_dict = {
            "J_fitted": J,
            "storage_values_objective_function": storage_dict
        }

        return return_dict


    def _MJPL_conn_regularization_gradient(self,
                       ts,
                       J,
                       beta,
                       lambda_val,
                       max_iter,
                       lr,
                       convergence_threshold,
                       log_settings_dict = None):
        """
               Maximizes the Maximum Joint Pseudolikehood ("MJPL") with connectivity regularizaiton
               objective function via gradient ascent.

               Parameters
               ----------
               ts : np.ndarray, shape (N, T)
                   Binarized time series.
               J : np.ndarray, shape (N, N)
                   "Empirical" coupling, e.g. structural matrix or prior J.
               beta : float
               lambda_val : float
               max_iter : int
               lr : float
               convergence_threshold : float or None
               log_settings_dict : dict or None
                   e.g. {"dl_dJ": 10, "J": 50} meaning store dl_dJ every 10 iters,
                   store J every 50 iters.

               Returns
               -------
               return_dict : dict
                  "J_fitted": Final connection coefficients fitted by the model
                  "storage_values_objective_function": Dictionary of values through each iteration
                    of the model (e.g. dl_dJ, J)
               """

        # Error Handling
        # -------------------------------------------------------------
        # Raise error if ts is not binarized.

        # Instantiate storage dicts
        # -------------------------------------------------------------
        if log_settings_dict is None:
            log_settings_dict = {}
        storage_dict = {
            var_name: [] for var_name in log_settings_dict
        }

        # Instantiate variables for optimization
        # -------------------------------------------------------------
        # Calculate corr(UNDERSTAND WHY I AM DOING THIS)
        num_units, num_timepoints = ts.shape
        corr = (ts @ ts.T) / num_timepoints

        # Copy empirical J and create a zeroed J to start optimization
        J_empirical = J.copy()
        np.fill_diagonal(J_empirical, 0)
        J = np.zeros((num_units, num_units))

        # Run gradient ascent for all iterations
        # -------------------------------------------------------------
        for iter_num in range(max_iter):
            # Store current J for convergence check
            J_prev = np.copy(J)

            # Compute gradient
            try:
                dl_dJ = (-corr) + \
                        ((0.5 * ts @ np.tanh(beta * (J @ ts)).T) / num_timepoints) + \
                        ((0.5 * (ts @ np.tanh(beta * (J @ ts)).T).T) / num_timepoints)
                dl_dJ = dl_dJ - np.diag(np.diag(dl_dJ))
            except Warning as e:
                error_info = {
                    "error": str(e),
                    "iteration num": iter_num,
                    "J": J,
                    "dl_dJ": dl_dJ,
                    "func_ts": ts
                }
                print(str(e))
                return error_info

            # Store variables if required
            variables_dict = {
                "dl_dJ": dl_dJ,
                "J": J
            }

            for var_name, interval in log_settings_dict.items():
                if (iter_num % interval) == 0:
                        storage_dict[var_name].append(variables_dict[var_name])

            # Update J Matrix
            J = J - (lr * dl_dJ) - (lr * lambda_val * (J - np.sign(J) * J_empirical))

            # Break if convergence is achieved
            if iter_num > 0 and np.allclose(J, J_prev):
                break

        # Return
        # -------------------------------------------------------------
        # Consolidate J output and storage values
        return_dict = {
            "J_fitted": J,
            "storage_values_objective_function": storage_dict
        }

        return return_dict


    def _MJPL_Symmetric_LocalField_gradient(self,
                                            ts,
                                            h,
                                            J,
                                            beta,
                                            max_iter,
                                            lr,
                                            convergence_threshold,
                                            log_settings_dict=None
                                            ):
        """
        Performs gradient ascent on the log pseudolikelihood for an Ising model
        with local fields h_i and pairwise couplings J_ij, ensuring J is symmetric
        with zero diagonal.

        Parameters
        ----------
        ts : np.ndarray of shape (N, T)
            Binarized time series (spins ±1).
        h : np.ndarray of shape (N,)
            Initial local fields. h[i] is the local field for spin i.
        J : np.ndarray of shape (N, N)
            Initial coupling matrix. Must be square and match the number of spins N.
            Diagonal entries are assumed to be 0; we enforce J to remain symmetric.
        beta : float
            Inverse temperature.
        max_iter : int
            Maximum number of gradient steps.
        lr : float
            Learning rate (step size) for gradient ascent.
        convergence_threshold : float or None
            If not None, used for convergence check based on change in parameters
            from iteration to iteration (L2 norm).
        log_settings_dict : dict or None
            Dictionary that specifies which variables to log and how often.
            Example: {"h": 100, "dl_dh": 100, "J": 500, "dl_dJ": 500}
            logs those items every specified number of iterations.

        Returns
        -------
        return_dict : dict
            {
              "J_fitted": Final optimized coupling matrix J,
              "h_fitted": Final optimized local field vector h,
              "storage_values_objective_function": dict
                 Contains any logged variables collected over iterations.
            }
        """
        # Error Handling
        # -------------------------------------------------------------
        if ts.ndim != 2:
            raise ValueError("Time series `ts` must be a 2D array.")
        if h.shape[0] != ts.shape[0]:
            raise ValueError("Local field `h` must have the same size as the number of units in `ts`.")
        if J.shape[0] != J.shape[1] or J.shape[0] != ts.shape[0]:
            raise ValueError("Coupling matrix `J` must be square and match the number of units in `ts`.")

        # Instantiate storage dicts
        # -------------------------------------------------------------
        if log_settings_dict is None:
            log_settings_dict = {}
        storage_dict = {
            var_name: [] for var_name in log_settings_dict
        }

        # Gradient Loop
        # -------------------------------------------------------------
        num_units, num_timepoints = ts.shape
        for iter_num in range(max_iter):
            # Keep a copy of previous params for convergence check
            h_prev = h.copy()
            J_prev = J.copy()

            # --- Subsection: actual gradient calculations ---
            # Effective fields: h_i + Σ_j (J_ij * s_j)
            effective_field = h[:, np.newaxis] + np.dot(J, ts)  # (N, T)
            # Model's expected spin = tanh(β * effective_field)
            expected_spin = np.tanh(beta * effective_field)  # (N, T)
            # Error term = (s_i - tanh(...))
            error_term = ts - expected_spin  # (N, T)
            # Gradient wrt h
            #     grad_h = β * (1/T) * sum_{t=1..T} [ s_i(t) - tanh(...) ]
            grad_h = beta * np.sum(error_term, axis=1) / T  # shape (N,)
            # Gradient wrt J
            #     We'll compute a partial first, then symmetrize:
            #         grad_J_part[i,j] = sum over t of [ (s_i - tanh(...)) * s_j ]
            grad_J_from_i = np.dot(error_term, ts.T)  # (N, N)
            # grad_J_from_j = np.dot(time_series, error_term.T)  # shape: (N, N) | = sum(s_j * (s_i - tanh(beta * ef_field)), axis=1)
            # I think I could redo this to be more efficient by using the the fact that grad_J_from_j is just the transpose of grad_J_from_i
            grad_J = beta * (grad_J_from_i + grad_J_from_j.T) / (2.0 * T)  # (N, N)

            # --- Subsection: Optionally log intermediate variables ---
            variables_dict = {
                "dl_dh": grad_h,
                "dl_dJ": grad_J,
                "h": h,
                "J": J
            }
            for var_name, interval in log_settings_dict.items():
                if (iter_num % interval) == 0 and var_name in variables_dict:
                    storage_dict[var_name].append(variables_dict[var_name].copy())

            # --- Subsection: Update Parameters ---
            h += lr * grad_h
            J += lr * grad_J
            # Enforce symmetry and zero diagonal
            J = 0.5 * (J + J.T)
            np.fill_diagonal(J, 0.0)

            # --- Subsection: Convergence Check ---
            if convergence_threshold is not None:
                h_diff = np.linalg.norm(h - h_prev)
                J_diff = np.linalg.norm(J - J_prev)
                if h_diff < convergence_threshold and J_diff < convergence_threshold:
                    break

        # Return final results
        return_dict = {
            "h_fitted": h,
            "J_fitted": J,
            "storage_values_objective_function": storage_dict
        }
        return return_dict




    ###################################
    # Helper Functions #
    ###################################
    def _get_random_units_array(self, num_units):
        return np.random.choice([-1,1], size = num_units)


    def _get_elapsed_time(self, start_time):
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time


    def _binarize_func_ts(self, func_ts):

        # Check that func_ts is 2D
        if func_ts.ndim != 2:
            raise ValueError("func_ts in '_binarize_func_ts' is not 2-dimensional")

        # Raise warning if axis 0 is greater than axis 1
        if func_ts.shape[0] > func_ts.shape[1]:
            warnings.warn("func_ts in 'binarize_func_ts' has 0 axis length greater than 1 axis."
                          "Uncommon. Check that units/ROIs/spins are 0 axis and time is 1 axis")

        # Binarize time series
        ts = func_ts[:, ~np.all(func_ts == 0, axis=0)] # Remove time points that are zero for all ROIs
        ts_zscored = zscore(ts, axis=1, nan_policy='omit') # z-score normalization (across time for each ROI)
        bin_ts = np.sign(ts_zscored)

        return bin_ts


    def _calc_FC_corr(self,
                      FC_empirical,
                      func_time_series,
                      log_settings_dict):

        # Error Handling
        # -------------------------------------------------------------
        # Ensure empirical_FC and time_series have same number of units
        if FC_empirical.shape[0] != func_time_series.shape[0]:
            raise ValueError("in _calc_FC_corr, the empirical_FC and time_series do not have same number of units. "
                             "Check that empirical_FC is NxN and time_series is Nxt")


        # Instantiate storage_dict and log_settings_dict
        # -------------------------------------------------------------
        if log_settings_dict is None:
            log_settings_dict = {
                "FC_corr": "whole"
            }
        storage_dict = {
            "FC_corr_whole": None,
            "FC_corr_interval_values": None,
            "FC_corr_interval": None
        }

        # Calculate FC_sim for time_series as "whole"
        # -------------------------------------------------------------
        if log_settings_dict.get("FC_corr") == "whole":
            # Calc and store just whole FC
            FC_sim = self._calc_FC(
                func_time_series = func_time_series,
                allow_self_connections = False
            )

            #Calc FC_corr
            FC_corr = np.corrcoef(FC_empirical.flatten(), FC_sim.flatten())[0,1]
            storage_dict["FC_corr_whole"] = FC_corr

        # Calculate FC_sim for time_series as "interval"
        # -------------------------------------------------------------
        elif isinstance(log_settings_dict.get("FC_corr"), numbers.Number):
            # Extract and store interval
            interval = log_settings_dict.get("FC_corr")
            storage_dict["FC_corr_interval"] = interval

            # Create iterable for storage (only store correlations for >= 3)
            num_timepoints = func_time_series.shape[1]
            time_splits = self._get_time_splits_for_FC_corr(
                total_timepoints = num_timepoints,
                interval = interval,
            )

            # Calc FC then FC_corr and store for each interval
            FC_corr_interval_values = []
            for t in time_splits:
                # skip if we have < 3 time points for correlation
                if t < 3:
                    continue
                # Calc FC_sim for each interval
                ts_chunk = func_time_series[:, :t]
                FC_sim_chunk = self._calc_FC(
                    func_time_series = ts_chunk,
                    allow_self_connections = False
                )

                # Calc FC_corr and store for each interval
                FC_corr = np.corrcoef(FC_empirical.flatten(), FC_sim_chunk.flatten())[0,1]
                FC_corr_interval_values.append(FC_corr)

            # Store FC_corr list in storage_dict
            storage_dict["FC_corr_interval_values"] = FC_corr_interval_values
            storage_dict["FC_corr_whole"] = FC_corr_interval_values[-1]

        # Error handling: if "whole" or numeric isn't given for FC_corr setting
        # -------------------------------------------------------------
        else:
            raise ValueError("Value in log_settings_dict['FC_corr'] in _calc_FC_corr is not 'whole' or a numeric "
                             "(representing the interval for storing)")

        return storage_dict


    def _calc_FC(self,
                 func_time_series,
                 allow_self_connections = False):
        num_timepoints = func_time_series.shape[1]
        FC = (func_time_series @ func_time_series.T) / num_timepoints

        if not allow_self_connections:
            np.fill_diagonal(FC, 0)

        return FC


    def _get_time_splits_for_FC_corr(self,
                                     total_timepoints,
                                     interval):
        """
        Small helper to build a list of chunk endpoints for interval-based correlation.
        e.g., if total_timepoints=20, interval=7, we might return [7, 14, 20].
        If last timepoint not divisibel by interval, it will be included.
        """
        splits = list(range(interval, total_timepoints+1, interval))
        if splits[-1] < total_timepoints:
            splits.append(total_timepoints)
            warnings.warn(f"Warning: {interval} is not a divisor of total timepoints: {total_timepoints}."
                          f"The last value will be stored even if not a multiple.")
        return splits


    def _create_hyperparam_df_inverse_fit_and_optimize(self, df_inverse, df_forward):
        # Add unique id for inverse df and forward_df
        df_inverse["ID_inverse_hyperparams"] = [
            f"inverse_{i}" for i in range(len(df_inverse))
        ]
        df_forward["ID_forward_hyperparams"] = [
            f"forward_{i}" for i in range(len(df_forward))
        ]

        # Add a key = 1 column to both dfs
        df_inverse["key"], df_forward["key"] = 1, 1

        # merge on that key -> a cartesian product
        df_merged = pd.merge(df_inverse, df_forward, on="key").drop(columns = "key")

        # Reorder column ascending order and ascending
        df_merged = df_merged.sort_values(by = ["lambda", "beta_inverse", "beta_forward"],
                                          ascending = [True, True, True])
        df_merged = df_merged.reset_index(drop = True)

        # Add unique id for merged_df
        df_merged["ID_unique_hyperparams"] = [
            f"unique_{i}" for i in range(len(df_merged))
        ]

        # Reorder column order
        columns_order = ["ID_unique_hyperparams",
                         "ID_inverse_hyperparams",
                         "ID_forward_hyperparams",
                         "lambda",
                         "beta_inverse",
                         "beta_forward"
                         ]
        df_merged = df_merged[columns_order]

        return df_merged


    def _create_storage_dict_from_merged_df_inverse_fit_and_optimize(self, df_merged):
        # Initialize level 1 of storage dict
        storage_dict = {
            "hyperparameters_results": {}
        }

        # Get all unique "ID_inverse_hyperparams" from df_merged
        unique_id_inverse = df_merged["ID_inverse_hyperparams"].unique()

        for id_inv in unique_id_inverse:
            # subset the rows that share this ID inverse
            df_inv_subset = df_merged[df_merged["ID_inverse_hyperparams"] == id_inv].copy()

            # Take first row params since they are all the same
            first_row = df_inv_subset.iloc[0]

            # Build a small dict for the inverse hyperparams
            inverse_hparams_dict = {
                "lambda": first_row["lambda"],
                "beta_inverse": first_row["beta_inverse"]
            }

            # Initialize second level dict for ID_inverse
            storage_dict["hyperparameters_results"][id_inv] = {
                "hyperparameters_inverse": inverse_hparams_dict,
                "sim_forward_results": {}
            }

            # For each row in df_inv_subset, create a forward_sim sub_dict
            for _, row in df_inv_subset.iterrows():
                id_fwd = row["ID_forward_hyperparams"]

                # Build a dict for the forward hyperparams
                forward_hparams_dict = {
                    "beta_forward": row["beta_forward"]
                }

                # Store placeholders for simulation results
                storage_dict["hyperparameters_results"][id_inv]["sim_forward_results"][id_fwd] = {
                    "hyperparameters_forward": forward_hparams_dict,
                }

        return storage_dict



    ###################################
    # Metropolis Hastings #
    ###################################
    def _make_MH_step(self,
                      units_array,
                      conn_coeffs,
                      beta,
                      unit_to_flip=None,
                      current_energy=None):
        """
        Flip one unit (at index=unit_to_flip) using Metropolis acceptance.
        Return the possibly-updated unit array and the new energy.
        """
        num_units = len(units_array)

        # Propose unit flip
        if unit_to_flip is None:
            unit_to_flip = self.rs.randint(num_units)

        # Calculate energy pre and post unit flip and calculate difference
        if current_energy is None:
            current_energy = self._calc_energy(units_array, conn_coeffs)
        units_array[unit_to_flip] *= -1
        proposed_energy = self._calc_energy(units_array, conn_coeffs)
        energy_diff = proposed_energy - current_energy

        # Accept or reject spin flip based on Boltzmann probability
        if energy_diff > 0 and np.exp(-beta * energy_diff) < self.rs.rand():
            # Reject flip
            units_array[unit_to_flip] *= -1  # Revert the flip
            energy = current_energy
        else:
            # Accept flip
            energy = proposed_energy

        return [units_array, energy]


    def _make_MH_sweep(self,
                       units_array,
                       conn_coeffs,
                       beta,
                       current_energy=None):
        """
        Perform ONE *full sweep* of the spins in random order.
        That is, we attempt to flip every spin exactly once.
        """
        # Select and order units to flip
        num_units = len(units_array)
        unit_indices = self.rs.permutation(num_units)

        # Calculate energy if we don't have it
        if current_energy is None:
            current_energy = self._calc_energy(units_array, conn_coeffs)

        # Flip each unit
        for idx in unit_indices:
            # Attempt a single unit flip (step)
            units_array, current_energy = self._make_MH_step(
                units_array,
                beta=beta,
                unit_to_flip=idx,
                current_energy=current_energy,
                conn_coeffs=conn_coeffs
            )

        return [units_array, current_energy]





    ###################################
    # Simulation Measurements Functions #
    ###################################
    def _calculate_sim_measurement(self,
                                   measurement_name,
                                   units_array = None,
                                   conn_coeffs = None,
                                   current_energy = None,
                                   beta = None):
        """
        Helper function to be used for each simulation step. Given a specified measurement_name,
        the appropriate function is called and calculated
        """
        if measurement_name == "units_array":
            "Return a copy of the units_array"
            return units_array.copy()

        elif measurement_name == "energy":
            if current_energy is not None:
                return current_energy
            else:
                return self._calc_energy(
                    units_array=units_array,
                    conn_coeffs=conn_coeffs)

        elif measurement_name == "magnetization":
            return self._calc_magnetization()

        return None


    def _calc_energy(self, units_array, conn_coeffs):
        """I need to double check this is correct. I am worried I am overcounting the spins"""
        return -0.5 * np.sum(conn_coeffs * np.outer(units_array, units_array))


    def _calc_specific_heat(self):
        return None


    def _calc_magnetization(self, units_array):
        magnetization = np.mean(units_array)
        return magnetization


    def _calc_magnetic_suscpetibility(self):
        return None



##############################
##############################
#endregion
##############################
##############################