# beam_analysis.py
# A script to define inputs, import dsm.py, and run beam analysis

import numpy as np
from dsm import BeamComponent, BoundaryCondition, BeamSolver

# Define inputs
# Material properties: Steel beam (E in Pa, A in m^2, Iy/Iz/J in m^4)
E = 200e9  # Young's Modulus (200 GPa)
nu = 0.3   # Poisson's Ratio
A = 0.01   # Cross-sectional area (100 cm^2)
Iy = 1e-4  # Moment of inertia about y-axis (10000 cm^4)
Iz = 1e-4  # Moment of inertia about z-axis (10000 cm^4)
J = 2e-4   # Polar moment of inertia (20000 cm^4)

# Nodes: [x, y, z, node_id] in meters
nodes = np.array([
    [0.0, 0.0, 0.0, 0],  # Node 0 at origin
    [1.0, 0.0, 0.0, 1],  # Node 1 at 1m along x-axis
])

# Elements: [node1_id, node2_id]
elements = np.array([
    [0, 1],  # Element connecting node 0 to node 1
])

# Boundary conditions: {node_id: (UX, UY, UZ, RX, RY, RZ)}
fixed_nodes = {
    0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Fixed support at node 0
    1: (0.0, 0.0, 0.0, None, None, None),  # Pinned support at node 1
}

# Loads: {node_id: (FX, FY, FZ, MX, MY, MZ)} in Newtons and Newton-meters
loads = {
    1: (0.0, 0.0, -1000.0, 0.0, 0.0, 0.0),  # 1000N downward force at node 1
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
