# validate_localfield_mjpl.py
"""
Utility functions for **validating** the new
`MJPL_symmetric_local_field` routine **from inside any notebook or
script** (no `if __name__ == "__main__"` guard).  The core entry point
is

```python
validate_localfield_mjpl(Ising_Model)
```

which returns a small dictionary containing correlation and RMSE
metrics plus the ground‑truth and estimated parameters.  The helper
also shows the minimal tweaks you need to make to the `Ising_Model`
class so its *forward* simulation can include local fields.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Type

###############################################################################
# --------------------------------------------------------------------------- #
# 0)  TWO‑LINE PATCH YOU NEED INSIDE YOUR  Ising_Model  CLASS
# --------------------------------------------------------------------------- #
#   A) Add a  local_fields=None  argument to  simulate_ising , _make_MH_step ,
#      _make_MH_sweep , and _calc_energy.
#
#   B) Wherever energy is evaluated, include the local‑field term:
#          energy = -0.5 * s^T J s    -   h^T s
#
#      A quick drop‑in replacement for your _calc_energy is:
#
# def _calc_energy(self, spins: np.ndarray, J: np.ndarray, h: np.ndarray|None=None):
#     pair_term = -0.5 * np.sum(J * np.outer(spins, spins))
#     field_term = 0.0 if h is None else -np.dot(h, spins)
#     return pair_term + field_term
#
#   C) Update  _make_MH_step  so  delta_E  also accounts for  h  (see code
#      below for reference).  After those three micro‑patches your existing
#      forward simulation becomes perfectly usable for local‑field models.
###############################################################################

# --------------------------------------------------------------------------- #
# 1) Ground‑truth generation helpers
# --------------------------------------------------------------------------- #

def generate_ground_truth(
    N: int,
    coupling_scale: float = 0.5,
    field_scale: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return symmetric *J_true* and local field *h_true*."""
    rng = np.random.RandomState(seed)
    J = rng.randn(N, N) * coupling_scale
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0)
    h = rng.randn(N) * field_scale
    return J, h


# --------------------------------------------------------------------------- #
# 2) High‑level validation routine
# --------------------------------------------------------------------------- #

def validate_localfield_mjpl(
    Model: Type,  # the Ising_Model class
    N: int = 20,
    T: int = 5000,
    beta: float = 1.0,
    max_iter: int = 3000,
    lr: float = 0.05,
    convergence_threshold: float = 1e-4,
    seed: int = 42,
) -> Dict:
    """Train & evaluate MJPL‑symmetric‑local‑field on synthetic data."""

    # 2.1 Generate ground truth and simulate spins with *your* forward code
    J_true, h_true = generate_ground_truth(N, seed=seed)

    model_sim = Model()
    sim_out = model_sim.simulate_ising(
        conn_coeffs=J_true,
        beta=beta,
        num_sim_steps=T,
        num_iters_per_sim=1,
        num_burn_in_steps=1000,
        sim_measurements_dict={"units_array": 1},
        # NEW PARAM YOU ADDED:
        local_fields=h_true,
    )
    ts = sim_out["sim_measurements"]["units_array"]  # shape (N, T)

    # 2.2 Fit inverse Ising with local‑field objective
    hyper_df = pd.DataFrame([{"beta_inverse": beta}])
    model_fit = Model()
    fit_res = model_fit.fit_inverse_ising(
        func_ts=ts,
        hyperparams_df_one_set=hyper_df,
        conn_coeffs=None,  # start J at zeros
        objective_function="MJPL_symmetric_localfield",
        optimization_method="gradient",
        max_iter=max_iter,
        learning_rate=lr,
        convergence_threshold=convergence_threshold,
        skip_error_checks=True,
    )

    J_est = fit_res["fitted_conn_coeffs"]
    h_est = fit_res["fitted_local_fields"]

    # 2.3 Evaluation metrics (upper‑triangle for J)
    tri = np.triu_indices_from(J_true, k=1)
    corr_J = np.corrcoef(J_true[tri], J_est[tri])[0, 1]
    rmse_J = np.sqrt(np.mean((J_true[tri] - J_est[tri]) ** 2))
    rmse_h = np.sqrt(np.mean((h_true - h_est) ** 2))

    return {
        "J_true": J_true,
        "h_true": h_true,
        "J_est": J_est,
        "h_est": h_est,
        "corr_J": corr_J,
        "rmse_J": rmse_J,
        "rmse_h": rmse_h,
    }


# --------------------------------------------------------------------------- #
# 3) Tiny example of programmatic use (no __main__ block)
# --------------------------------------------------------------------------- #
# >>> from src_Ising_ver2_1 import Ising_Model
# >>> from validate_localfield_mjpl import validate_localfield_mjpl
# >>> stats = validate_localfield_mjpl(Ising_Model)
# >>> print(stats["corr_J"], stats["rmse_J"], stats["rmse_h"])
###############################################################################
