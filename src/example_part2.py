import numpy as np
import Direct_Stiffness_Method as dsm
import functions as fu
import Elastic_Critical_Solver as ecs

# Define nodes: N0 at (0,0,0) fixed, N1 at (30,40,0) free
nodes = [
    {'node_id': 0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'bc': [True, True, True, True, True, True]},  # N0 fixed
    {'node_id': 1, 'x': 30.0, 'y': 40.0, 'z': 0.0, 'bc': [False, False, False, False, False, False]}  # N1 free
]

# Material and section properties
A = np.pi       # Cross-sectional area (m²)
E = 1000        # Young's modulus (Pa)
nu = 0.3        # Poisson's ratio
Iy = np.pi / 4  # Moment of inertia about y (m⁴)
Iz = np.pi / 4  # Moment of inertia about z (m⁴)
J = np.pi / 2   # Torsion constant (m⁴)

# Create element using DSM's create_element function
element = dsm.create_element(nodes[0], nodes[1], E, nu, A, Iy, Iz, J)
elements = [element]

# Define loads: F = [-3/5, -4/5, 0] N at N1 (1 N magnitude)
loads = {
    1: np.array([-3/5, -4/5, 0.0, 0.0, 0.0, 0.0])  # [Fx, Fy, Fz, Mx, My, Mz], 1 N total
}

# Instantiate and run the CriticalLoadSolver
solver = ecs.CriticalLoadSolver(nodes, elements, loads, include_interactions=False)

# Debug
displacements, _ = dsm.calculate_structure_response(nodes, elements, loads)
print("Displacements:", displacements)
internal_forces = solver.compute_internal_forces_and_moments(displacements)
print("Internal Forces for Element (0,1):", internal_forces[(0, 1)])
fx2, mx2, my1, mz1, my2, mz2 = ecs.moments_from_forces(internal_forces, (0, 1))
print("Fx2:", fx2)
print("Global Elastic Stiffness Matrix (sample):", dsm.assemble_global_stiffness_matrix(nodes, elements)[:6, :6])
print("Global Geometric Stiffness Matrix (sample):", solver.global_geo_matrix[:6, :6])

# Compute eigenvalues
eigenvalues, eigenvectors = solver.compute_eigenvalues()

# Output results
print("Critical Load Factors:", eigenvalues[:3])

# Plot the first buckling mode (save for cluster)
solver.plot_mode_shape(eigenvectors[:, 0], scale_factor=0.5)
