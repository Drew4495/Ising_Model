# Imports
# -------------------------------------------------------------
import numpy as np



# Define function to calculate gradient and return new h and Jij
# -------------------------------------------------------------
def gradient_MPL_with_local_field(time_series, h_current, J_current, beta):
    """
    Compute gradients of the log pseudolikelihood for an Ising model.

        Parameters:
          time_series : numpy.ndarray
              Data matrix of shape (N, T) with spins ±1.
          h_current : numpy.ndarray
              Local fields, shape (N,).
          J_current : numpy.ndarray
              Coupling matrix, shape (N, N) with J[i,i]=0.
          beta : float
              Inverse temperature.

        Returns:
          grad_h : numpy.ndarray
              Gradient with respect to h (shape (N,)).
          grad_J : numpy.ndarray
              Gradient with respect to J (shape (N, N)).
    """
    N, T = time_series.shape
    # Compute effective field, which is the sum of h and the spin-spin interactions
    effective_field_empirical = h_current[:, np.newaxis] + np.dot(J_current, time_series)  # shape: (N, T) | = h_i + sum(J_ij * s_j))

    # Compute model's predicted effective field
    expected_spin_model = np.tanh(beta * effective_field_empirical)  # shape: (N, T)  | = tanh(beta * (h_i + sum(J_ij * s_j)))

    # Calculate error term: Used in h and Jij gradient calculation
    error_term = time_series - expected_spin_model   # (s_i - tanh(beta * ef_field)), shape: (N, T)

    # Gradient w.r.t. h: average over all timepoints so gradient doesn't scale with T
    grad_h = beta * np.sum(error_term, axis=1) / T  # shape: (N, T) -> (N,1)

    # Gradient w.r.t J: average over all timepoints
    grad_J_from_i = np.dot(error_term, time_series.T)  # shape: (N, N) | = sum(s_j * (s_i - tanh(beta * ef_field)), axis=1)
    grad_J_from_j = np.dot(time_series, error_term.T)  # shape: (N, N) | = sum(s_j * (s_i - tanh(beta * ef_field)), axis=1)
    # I think I could redo this to be more efficient by using the the fact that grad_J_from_j is just the transpose of grad_J_from_i
    grad_J = beta * (grad_J_from_i + grad_J_from_j) / T  # shape: (N, N)

    return grad_h, grad_J


# Gradient ascent procedure
# -------------------------------------------------------------
# Define initial values
num_iters = 1000
num_units = 30
num_timepts = 1000
h_initial = np.full((num_units,), 1)
J_initial = np.full((num_units, num_units), 1)
learn_rate = 0.001
print_iter = 50

# Create an example time series: random ±1 values of shape (num_units, num_timepts)
time_series = np.random.choice([-1, 1], size=(num_units, num_timepts))

# Define ascent procedure
h_current = h_initial.copy()
J_current = J_initial.copy()
# Enforce no self-interactions
np.fill_diagonal(J_current, 0)

#Gradient ascent
for iter in range(num_iters):
    # Compute gradients
    grad_h, grad_J = gradient_MPL_with_local_field(
        time_series=time_series,
        h_current=h_current,
        J_current=J_current,
        beta =1
    )

    # Update params
    h_current = h_current + learn_rate * grad_h
    J_current = J_current + learn_rate * grad_J

    # Enforce symmetry and remove self-interactions
    J_current = (J_current + J_current.T) / 2
    np.fill_diagonal(J_current, 0)

    # Print every print_iter
    if iter % print_iter == 0:
        grad_norm = np.linalg.norm(grad_h) + np.linalg.norm(grad_J)
        print(f"Iteration {iter}: Gradient norm = {grad_norm:.4f}")


