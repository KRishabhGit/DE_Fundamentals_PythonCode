import time
import numpy as np
from math import inf
from scipy.optimize import differential_evolution, NonlinearConstraint
from tabulate import tabulate
from matplotlib import pyplot as plt
from pandas import DataFrame as df
from matplotlib import font_manager

# Start timer
start = time.time()

# ---------------------------- Input Data ---------------------------- #
# Displacement values (in cm)
x = np.arange(-4, 4.2, 0.2)

# Sensor output corresponding to displacements in x
T1 = 20
T2 = np.array([-9.82,-9.48,-9.14,-8.78,-8.38,-7.98,-7.58,-7.12,-6.68,
               -6.18,-5.7,-5.18,-4.64,-4.12,-3.56,-3.0,-2.42,-1.86,
               -1.28,-0.7,-0.1,0.4,0.98,1.54,2.1,2.64,3.18,3.7,4.22,
               4.7,5.2,5.66,6.14,6.6,7.02,7.44,7.84,8.22,8.58,8.94,9.28])

# Nonlinear function output (F)
F = (T2 / T1)

# ------------- Linear Model Fitting for Best-Fit Line Computation ------------- #
def TPBFL(x, F, A1, A2, K1, K2):
    """Compute parameters of best-fit line for linearized output."""
    F_L = K1 * (F / (F + A1)) + K2 * (F / (F + A2))
    X = np.column_stack((np.ones(len(x)), x))
    b = np.linalg.inv(X.T @ X) @ X.T @ F_L
    return b.tolist()

# -------------------- Nonlinearity Calculation -------------------- #
def Calculate_NL(x, F, A1, A2, K1, K2):
    """Calculate % nonlinearity between F_L and best-fit line."""
    F_L = K1 * (F / (F + A1)) + K2 * (F / (F + A2))
    c, m = TPBFL(x, F, A1, A2, K1, K2)
    y = m * x + c
    NL = 100 * max(abs(F_L - y)) / abs(max(F_L) - min(F_L))
    return NL

# -------------------- Objective Function -------------------- #
def to_minimise(R):
    """Objective function: minimize nonlinearity."""
    A1, A2, K1, K2 = R
    return Calculate_NL(x, F, A1, A2, K1, K2)

# ----------------------- Constraints ------------------------ #
def constr_f1(params):
    """Constraint on slope of best-fit line."""
    A1, A2, K1, K2 = params
    return abs(TPBFL(x, F, A1, A2, K1, K2)[1])

def constr_f2(params):
    """Constraint on nonlinearity value."""
    A1, A2, K1, K2 = params
    return Calculate_NL(x, F, A1, A2, K1, K2)

nlc1 = NonlinearConstraint(constr_f1, 0.1, inf) # Constraint: Absolute slope of best-fit line must be ≥ 0.1
nlc2 = NonlinearConstraint(constr_f2, 0.0001, 100) # Constraint: Nonlinearity must be between 0.0001 and 100
constraints = (nlc1, nlc2)

# -------------------- Differential Evolution -------------------- #
bounds = [(-100, 100), (-100, 100), (-1, 1), (-1, 1)] # A1, A2 ∈ [-100, 100]; K1, K2 ∈ [-1, 1]

result = differential_evolution(
    to_minimise, # Objective function for optimizer
    bounds,
    constraints=constraints,
    maxiter=500, 
    popsize=25, # max-iterations and population size to ensure thorough search.
    polish=True
)
# ------------------------ Output Results ------------------------- #
A1, A2, K1, K2 = result.x
c, m = TPBFL(x, F, A1, A2, K1, K2)
F_L = K1 * (F / (F + A1)) + K2 * (F / (F + A2))

results_dict = {result.fun: [A1, A2, K1, K2, abs(m), result.fun]}
sorted_nl = sorted(results_dict)
to_print = [results_dict[i] for i in sorted_nl]

# Display the results 
print(tabulate(to_print, ["A1", "A2", "K1", "K2", "abs(m)", "NL"], tablefmt="fancy_grid")) # optimal parameter set in a clean table.
print("Best-Fit Line Coefficients:", TPBFL(x, F, A1, A2, K1, K2))
print("Optimizer Result Object:", result)
print("Total Optimization Time (s):", time.time() - start)
print("Final Nonlinearity (%):", Calculate_NL(x, F, A1, A2, K1, K2))

# ------------------------------- Plotting ------------------------------- #
from matplotlib import font_manager
# Set custom font for mathtext and general text
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['font.family'] = 'Times New Roman'

# Plot the linearized output and the best-fit line
fig, ax1 = plt.subplots(figsize=(9, 5))
line1, = ax1.plot(x, F_L, 's-', color='black', linewidth=2, label=r'Output $F_L$')
line2 = ax1.axline((0, c), slope=m, linestyle='--', linewidth=2, color='red', label='Best-Fit Line')

ax1.set_xlabel('Displacement (in cm)', fontsize=20)
ax1.set_ylabel(r'Output $F_L$ of RELD', fontsize=20)
ax1.tick_params(axis='both', labelsize=20, labelcolor='black')
# ax1.set_ylim(-0.9, 0.9)
ax1.grid(True)
ax1.legend(fontsize=18)
fig.tight_layout()
plt.show()