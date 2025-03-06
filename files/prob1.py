# prob1.py
# A script to define inputs, import Direct_Stiffness_Methodd.py, and run beam analysis

import numpy as np
from Direct_Stiffness_Methodd import BeamComponent, BoundaryCondition, BeamSolver

# Define inputs
# Material properties: Steel beam (E in Pa, A in m^2, Iy/Iz/J in m^4)
E = 1000  # Young's Modulus (1000 Pa)
nu = 0.3  # Poisson's Ratio
A = np.pi * 1.0**2 # Cross-sectional area (approx 6.2832 m^2)
Iy = np.pi / 4.0  # Moment of inertia about y-axis (approx 3.1416 m^4)
Iz = np.pi / 4.0  # Moment of inertia about z-axis (approx 3.1416 m^4)
J = np.pi / 2.0  # Polar moment of inertia (approx 6.2832 m^4)

# Nodes: [x, y, z, node_id] in meters
nodes = np.array([
    [0.0, 0.0, 0.0, 0],  # Node 0 at origin
    [30.0, 40.0, 0.0, 1],  # Node 1 at (30m, 40m, 0m)
])

# Elements: [node1_id, node2_id]
elements = np.array([
    [0, 1],  # Element connecting node 0 to node 1
])

# Boundary conditions: {node_id: (UX, UY, UZ, RX, RY, RZ)}
fixed_nodes = {
    0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Fixed support at node 0
    1: (None, None, None, None, None, None),  # Pinned support at node 1
}

# Loads: {node_id: (FX, FY, FZ, MX, MY, MZ)} in Newtons and Newton-meters
# Force vector F = [3/5, -4/5, 0], magnitude 1000 N
fx = 3/5
fy = 4/5
loads = {
    1: (fx, fy, 0.0, 0.0, 0.0, 0.0),  # Force at node 1
}

# Set up the beam and boundary conditions
beam = BeamComponent(nodes, elements, E, nu, A, Iy, Iz, J)
bc = BoundaryCondition(fixed_nodes)
for node_id, load in loads.items():
    bc.apply_load(node_id, load)

# Run the analysis
solver = BeamSolver(beam, bc)

# Static analysis
displacements, reactions = solver.solve()
print("Static Analysis Results:")
solver.display_results(displacements, reactions)
print("Internal Forces per Element:")
for elem_idx, forces in solver.internal_forces.items():
    print(f"Element {elem_idx}: {np.round(forces, 5)}")

# Buckling analysis
eigvals, eigvecs = solver.solve_buckling()
print("\nBuckling Analysis Results:")
print("Critical Load Factors:", np.round(eigvals, 5))
print("First Buckling Mode Shape:", np.round(eigvecs[:, 0], 5))
