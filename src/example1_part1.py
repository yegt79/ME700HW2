import math
import numpy as np
from functions import rotation_matrix_3D, transformation_matrix_3D
from Direct_Stiffness_Method import create_node, create_element, calculate_structure_response

# Node Definitions
node0 = create_node(0, 0, 10, 0, bc=[True, True, True, True, True, True])  # Fully fixed node
node1 = create_node(15, 0, 10, 1, bc=[False, False, False, False, False, False])  # Free node
node2 = create_node(15, 0, 0, 2, bc=[True, True, True, False, False, False])  # Fixed in translations only

# Material Properties
E = 1000  # Young's Modulus
nu = 0.3  # Poisson's Ratio
A = 0.5  # Cross-sectional Area
Iy = 0.5**3 / 12  # Moment of Inertia about y-axis
Iz = 0.5 / 12  # Moment of Inertia about z-axis
J = 0.02861  # Polar Moment of Inertia

# Create Elements
element1 = create_element(node0, node1, E, nu, A, Iy, Iz, J)  # From node0 to node1
element2 = create_element(node1, node2, E, nu, A, Iy, Iz, J)  # From node1 to node2

# Add elements to structure
elements = [element1, element2]
nodes = [node0, node1, node2]

# Create Load Vector
loads = {
    1: np.array([0.1, 0.05, -0.07, 0.05, -0.1, 0.25])  # Apply load to node 1
}

# Calculate structure response
displacements, reactions = calculate_structure_response(nodes, elements, loads)

# Print Results
print("Nodal Displacements and Rotations:")
for i, node in enumerate(nodes):
    print(f"Node {node['node_id']}: [u: {displacements[i*6]:.10f}, v: {displacements[i*6+1]:.10f}, w: {displacements[i*6+2]:.10f}, "
          f"rot_x: {displacements[i*6+3]:.10f}, rot_y: {displacements[i*6+4]:.10f}, rot_z: {displacements[i*6+5]:.10f}]")

print("\nReaction Forces and Moments at Supports:")
for i, node in enumerate(nodes):
    if any(node['bc']):  # Only display reactions for nodes with boundary conditions
        print(f"Node {node['node_id']}: [Fx: {reactions[i*6]:.10f}, Fy: {reactions[i*6+1]:.10f}, Fz: {reactions[i*6+2]:.10f}, "
              f"Mx: {reactions[i*6+3]:.10f}, My: {reactions[i*6+4]:.10f}, Mz: {reactions[i*6+5]:.10f}]")
