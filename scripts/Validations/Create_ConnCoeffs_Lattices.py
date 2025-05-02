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
Helper functions to generate lattices
"""
##############################
##############################

def make_square_lattice_J(num_rows: int,
                          *,
                          num_cols: int = None,
                          distribution: Union[str, Callable[[int], np.ndarray]] = "constant",
                          distribution_params: Dict = None,
                          periodic: bool = True,
                          rescale: bool = False,
                          symmetric: bool = True
) -> np.ndarray:
    """"""
    # ---------- Basic Checks -----------
    if num_cols is None:
        num_cols = num_rows

    if distribution_params is None:
        distribution_params = {}

    # ---------- Choose a Sampling Fxn -----------
    if callable(distribution):
        sample_func = distribution
    elif distribution == "constant":
        conn_value = distribution_params.get("conn_value", 1.0)
        sample_func = lambda n: np.full((n,), conn_value, dtype=float)
    elif distribution == "normal":
        mu = distribution_params.get("mu", 0.0)
        sigma = distribution_params.get("sigma", 1.0)
        rng = distribution_params.get("rng_seed", np.random.default_rng())
        sample_func = lambda n: rng.normal(mu, sigma, n)
    elif distribution == "uniform":
        a, b = distribution_params.get("a", -1.0), distribution_params.get("b", 1.0)
        rng = distribution_params.get("rng_seed", np.random.default_rng())
        sample_func = lambda n: rng.uniform(a, b, n)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # ---------- Build neighbor list -----------
    num_units = num_rows * num_cols
    # Create a 2D index array for the lattice. Each (i,j) gets a unique SINGLE index (i * num_cols + j)
    idx = np.arange(num_units).reshape(num_rows, num_cols)

    # Get right and down neighbors by "rolling" the index array (can be periodic boundary conditions or not)
    bonds_right = np.roll(idx, -1, axis=1) if periodic else idx[:, 1:]
    bonds_down = np.roll(idx, -1, axis=0) if periodic else idx[1:, :]

    # Concatenate to create a (num_bonds x 2) array all bond indices
    bonds_ij = np.concatenate(
        [np.stack([idx.ravel(), bonds_right.ravel()], axis=1),
         np.stack([idx.ravel(), bonds_down.ravel()], axis=1)
         ] if periodic
        else [
            np.stack([idx[:, :-1].ravel(), bonds_right[:, :-1].ravel()], axis=1,),
            np.stack([idx[:-1, :].ravel(), bonds_down[:-1, :].ravel()], axis=1)
        ]
    )

    # Get shape of undirected bonds_ij
    num_bonds_undirected = bonds_ij.shape[0]

    # ---------- Draw weights -----------
    weights = sample_func(num_bonds_undirected)

    if symmetric:
        # Duplicate the weights for undirected bonds
        rows = np.concatenate([bonds_ij[:, 0], bonds_ij[:, 1]])
        cols = np.concatenate([bonds_ij[:, 1], bonds_ij[:, 0]])
        weights = np.concatenate([weights, weights])
    else:
        # Independent reverse weights
        rows = bonds_ij[:, 0]
        cols = bonds_ij[:, 1]
        rows = np.concatenate([rows, cols])
        cols = np.concatenate([cols, rows[:num_bonds_undirected]])
        weights = np.concatenate([weights, sample_func(num_bonds_undirected)])

    # ---------- Build sparse matrix -----------
    J = np.zeros((num_units, num_units), dtype=float)
    J[rows, cols] = weights
    np.fill_diagonal(J, 0.0) # No self-interactions

    # ---------- Optional rescaling -----------
    """According to ChatGPT, this is a common convention so that Beta stays in scale as lattice size 
    arbitratily changes. Keeps the critical temp relatively constant since changing the scale of J changes 
    the scale of Beta"""
    if rescale:
        sd = weights.std()
        if sd > 0:
            J *= 1.0 / sd

    return J

##############################
##############################
#endregion
#======================================================================================================================#




#======================================================================================================================#
#region
"""
Build Lattices and save
"""
##############################
##############################


##############################
##############################
#endregion
#======================================================================================================================#

### Practice building one lattice for 20x20 with constant weights of 1
J = make_square_lattice_J(num_rows=20,
                          num_cols=20,
                          distribution="constant",
                          distribution_params={"conn_value": 1.0},
                          periodic=True,
                          rescale=False,
                          symmetric=True)


### Build 2D lattice for different N's: Constant weights
# N = 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100
# weights = 1,-1, 0.5, -0.5, 5, -5, 10, -10
N = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for n in N:
    for conn_value in [1, -1, 0.5, -0.5, 5, -5, 10, -10]:
        J = make_square_lattice_J(num_rows=n,
                                  num_cols=n,
                                  distribution="constant",
                                  distribution_params={"conn_value": conn_value},
                                  periodic=True,
                                  rescale=False,
                                  symmetric=True)
        ### Save J
        filepath = f"data/simulated_data/ConnectivityMatrices_J/Constant_Weights/J_ConstantWeights_ConnValue{conn_value}_{n}x{n}.npy"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, J)

### Gaussian Connectivity Coefficients
# mu, sigma = [(0,1), (1,0.5), (-1,0.5), (5,0.5), (-5,0.5), (0, 5), (5, 10), (-5, 10)]
N = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for n in N:
    for mu, sigma in [(0,1), (1,0.5), (-1,0.5), (5,0.5), (-5,0.5), (0, 5), (5, 10), (-5, 10)]:
        J = make_square_lattice_J(num_rows=n,
                                  num_cols=n,
                                  distribution="normal",
                                  distribution_params={"mu": mu, "sigma": sigma},
                                  periodic=True,
                                  rescale=False,
                                  symmetric=True)
        ### Save J
        filepath = f"data/simulated_data/ConnectivityMatrices_J/Normal_Weights/J_NormalWeights_mu{mu}_sigma{sigma}_{n}x{n}.npy"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, J)

### Uniform Connectivity Coefficients
# a, b = [(-1,1), (0,1), (-1,0), (0,5), (-5,0), (5,10), (-5,10)]
N = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for n in N:
    for a, b in [(-1,1), (0,1), (-1,0), (0,5), (-5,0), (5,10), (-5,10)]:
        J = make_square_lattice_J(num_rows=n,
                                  num_cols=n,
                                  distribution="uniform",
                                  distribution_params={"a": a, "b": b},
                                  periodic=True,
                                  rescale=False,
                                  symmetric=True)
        ### Save J
        filepath = f"data/simulated_data/ConnectivityMatrices_J/Uniform_Weights/J_UniformWeights_a{a}_b{b}_{n}x{n}.npy"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, J)


