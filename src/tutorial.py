# tutorial.py

import numpy as np
import Direct_Stiffness_Method as dsm

# Example properties for the beam element
E = 210e9   # Young's modulus in Pa
A = 0.01    # Cross-sectional area in m^2
Iy = 4.0e-6  # Moment of inertia about the y-axis in m^4
Iz = 6.0e-6  # Moment of inertia about the z-axis in m^4
J = 8.0e-6  # Polar moment of inertia in m^4
nodes = 3    # Number of nodes in the beam element

# Coordinates of the nodes (x, y, z positions)
node_positions = [
    [0, 0, 0],  # Node 1
    [5, 0, 0],  # Node 2
    [10, 0, 0]  # Node 3
]

# Create a BeamElement object
beam_element = BeamElement(E, A, Iy, Iz, J, nodes)

# Compute the global stiffness matrix
K_global = beam_element.stiffness_matrix(node_positions)

# Display the global stiffness matrix
print(K_global)
