# input_file.py (with plotting)
import numpy as np
from scipy.linalg import eig
import Direct_Stiffness_Method as dsm
from Elastic_Critical_Solver import ElasticCriticalLoadSolver

# Define nodes
nodes = [
    dsm.create_node(0.0, 0.0, 0.0, 0, bc=[True, True, True, True, True, True]),
    dsm.create_node(30.0, 40.0, 0.0, 1)
]

# Material and section properties
A = np.pi
E = 1000
nu = 0.3
Iy = np.pi / 4
Iz = np.pi / 4
J = np.pi / 2

# Create element
element = dsm.create_element(nodes[0], nodes[1], E, nu, A, Iy, Iz, J)
elements = [element]

# Define loads
loads = {1: np.array([-3/5, -4/5, 0.0, 0.0, 0.0, 0.0])}

# Initialize the solver
solver = ElasticCriticalLoadSolver(nodes, elements, loads, use_interaction_terms=False)

# Solve static problem
displacements, reactions = dsm.calculate_structure_response(nodes, elements, loads)
print("Displacements:", displacements)
print("Reactions:", reactions)

# Check internal forces
internal_forces = solver.compute_internal_forces_and_moments(displacements)
Fx2 = internal_forces[(0, 1)][6]
print("Fx2 (raw from solver):", Fx2)

# Solve eigenvalue problem and get full eigenvectors
eigenvalues, eigenvectors = solver.solve_eigenvalue_problem()

# Select the smallest positive eigenvalue (Euler buckling)
critical_load_factor = np.min(eigenvalues)
critical_mode_index = np.argmin(eigenvalues)  # Index of Mode 2 (0.781945)
print(f"Critical Load Factor (Euler Buckling): {critical_load_factor:.6f}")
print(f"Eigenvector shape: {eigenvectors.shape}")  # Should be (12, 6)

# Plot the Euler buckling mode (Mode 2, index 1 after sorting)
solver.plot_buckling_mode(eigenvectors[:, critical_mode_index], scale=10.0)
