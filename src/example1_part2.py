import numpy as np
import Direct_Stiffness_Method as dsm
import functions as fu
import Elastic_Critical_Solver as ecs

# Define nodes
nodes = [
    {'node_id': 0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'bc': [True, True, True, True, True, True]},
    {'node_id': 1, 'x': 30.0, 'y': 40.0, 'z': 0.0, 'bc': [False, False, False, False, False, False]}
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

# Instantiate solver
solver = ecs.CriticalLoadSolver(nodes, elements, loads, include_interactions=False)

# Compute and print results
displacements, _ = dsm.calculate_structure_response(nodes, elements, loads)
print("Displacements:", displacements)
internal_forces = solver.compute_internal_forces_and_moments(displacements)
print("Force and Moment for Element (0,1):", ecs.moments_from_forces(internal_forces, (0, 1)))
eigenvalues, eigenvectors = solver.compute_eigenvalues()
print("Critical Load Factor:", eigenvalues[0] if eigenvalues.size > 0 else "None")

# Plot (optional, kept as per your code)
solver.plot_mode_shape(eigenvectors[:, 0], scale_factor=0.5)
