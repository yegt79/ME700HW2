import numpy as np
import Direct_Stiffness_Method as dsm
import functions as fu
import Elastic_Critical_Solver as ecs

# Define nodes for a vertical beam: Node 0 at (0,0,0), Node 1 at (0,0,5)
nodes = [
    {'node_id': 0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'bc': [True, True, True, True, True, True]},  # Fixed base
    {'node_id': 1, 'x': 0.0, 'y': 0.0, 'z': 5.0, 'bc': [False, False, False, False, False, False]}  # Free top
]

# Material and section properties
A = 0.01       # Cross-sectional area (m²)
E = 200e9      # Young's modulus (Pa, steel)
nu = 0.3       # Poisson's ratio (steel)
Iy = 0.0001    # Moment of inertia about y (m⁴)
Iz = 0.0001    # Moment of inertia about z (m⁴)
J = 0.0002     # Torsion constant (m⁴)

# Create element using DSM's create_element function
element = dsm.create_element(nodes[0], nodes[1], E, nu, A, Iy, Iz, J)
elements = [element]

# Define loads: 1000 N compressive load at Node 1 (downward)
loads = {
    1: np.array([0.0, 0.0, -1000.0, 0.0, 0.0, 0.0])  # [Fx, Fy, Fz, Mx, My, Mz]
}

# Instantiate and run the CriticalLoadSolver
solver = ecs.CriticalLoadSolver(nodes, elements, loads, include_interactions=False)
eigenvalues, eigenvectors = solver.compute_eigenvalues()

# Output results
print("Critical Load Factors:", eigenvalues[:3])

# Plot the first buckling mode
solver.plot_mode_shape(eigenvectors[:, 0], scale_factor=0.5)
