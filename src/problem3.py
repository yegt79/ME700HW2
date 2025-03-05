# input_file.py (with plotting)
import numpy as np
from scipy.linalg import eig
import Direct_Stiffness_Method as dsm
from Elastic_Critical_Solver import ElasticCriticalLoadSolver

# Define nodes
nodes = [
    dsm.create_node(0.0, 0.0, 0.0, 0, bc=[True, True, True, True, True, True]),  # Node 0: Fixed
    dsm.create_node(11.0, 0.0, 0.0, 1),  # Node 1
    dsm.create_node(11.0, 23.0, 0.0, 2),  # Node 2
    dsm.create_node(26.0, 23.0, 0.0, 3),  # Node 3
    dsm.create_node(26.0, 10.0, 0.0, 4),  # Node 4
    dsm.create_node(11.0, 10.0, 0.0, 5, bc=[True, True, True, True, True, True]),  # Node 5: Fixed
]

# Material and section properties for Type A elements
E_a = 100000
nu_a = 0.3
r = 1.0
A_a = np.pi * r * 2.0
Iy_a = np.pi * r**4 * 4.0 / 4.0
Iz_a = np.pi * r**4 * 4.0 / 4.0
J_a = np.pi * r**4 * 4.0 / 2.0
local_z_a = None  # Default local_z for Type A (None as specified)

# Material and section properties for Type B elements (if needed, not used here)
E_b = 50000
nu_b = 0.3
b = 0.5
h = 1.0
A_b = b * h
Iy_b = b * h**3 * 3.0 / 12.0
Iz_b = b**3 * h * 3.0 / 12.0
J_b = 0.028610026041666667 * b * h * (b**2 + h**2)
local_z_b = np.array([[0, 0], [1, 1]])  # Not used unless Type B elements are defined

# Create elements (all Type A)
elements = [
    dsm.create_element(nodes[0], nodes[1], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E0
    dsm.create_element(nodes[1], nodes[2], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E1
    dsm.create_element(nodes[2], nodes[3], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E2
    dsm.create_element(nodes[3], nodes[4], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E3
    dsm.create_element(nodes[4], nodes[5], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E4
    dsm.create_element(nodes[5], nodes[0], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E5
    dsm.create_element(nodes[1], nodes[5], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E6
    dsm.create_element(nodes[2], nodes[4], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E7
    dsm.create_element(nodes[0], nodes[2], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E8
    dsm.create_element(nodes[1], nodes[3], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E9
    dsm.create_element(nodes[2], nodes[5], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E10
    dsm.create_element(nodes[3], nodes[0], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E11
    dsm.create_element(nodes[4], nodes[1], E_a, nu_a, A_a, Iy_a, Iz_a, J_a, v_temp=local_z_a),  # E12
]

# Define loads (applied at Node 3)
loads = {3: np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])}  # Fy = -1

# Initialize the solver
solver = ElasticCriticalLoadSolver(nodes, elements, loads, use_interaction_terms=False)

# Solve eigenvalue problem and get full eigenvectors
eigenvalues, eigenvectors = solver.solve_eigenvalue_problem()

# Select the smallest positive eigenvalue (Elastic Critical Load Factor λ)
critical_load_factor = np.min(eigenvalues[eigenvalues > 0]) if np.any(eigenvalues > 0) else np.min(eigenvalues)
print(f"Elastic Critical Load Factor (λ): {critical_load_factor:.6f}")
print(f"Eigenvector shape: {eigenvectors.shape}")

# Plot the interpolated deformed shape using the critical mode
#solver.plot_buckling_mode(eigenvectors[:, np.argmin(eigenvalues)], scale=10.0)

# Report the Elastic Critical Load Factor
print("\nReport:")
print(f"Elastic Critical Load Factor (λ): {critical_load_factor:.6f}")
print("Plot of structure interpolated deformed shape: (See plot above)")
