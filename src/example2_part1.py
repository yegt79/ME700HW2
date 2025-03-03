import math
import numpy as np
from functions import rotation_matrix_3D, transformation_matrix_3D
from Direct_Stiffness_Method import create_node, create_element, calculate_structure_response

# Node Definitions
node0 = create_node(0, 0, 0, 0, bc=[False, False, True, False, False, False])  # fixed in z
node1 = create_node(-5, 1, 10, 1, bc=[False, False, False, False, False, False])  # Free node
node2 = create_node(-1, 5, 13, 2, bc=[False, False, False, False, False, False])  # Free node
node3 = create_node(-3, 7, 11, 3, bc=[True, True, True, True, True, True]) # fixed
node4 = create_node(6, 9, 5, 4, bc=[True, True, True, False, False, False]) # pinned

# Material Properties
E = 500  # Young's Modulus
nu = 0.3  # Poisson's Ratio
A = math.pi  # Cross-sectional Area
Iy = math.pi/4  # Moment of Inertia about y-axis
Iz = math.pi/4 # Moment of Inertia about z-axis
J = math.pi/2  # Polar Moment of Inertia

# Create Elements
element0 = create_element(node0, node1, E, nu, A, Iy, Iz, J)
element1 = create_element(node1, node2, E, nu, A, Iy, Iz, J)
element2 = create_element(node2, node3, E, nu, A, Iy, Iz, J)
element3 = create_element(node2, node4, E, nu, A, Iy, Iz, J)

# Add elements to structure
elements = [element0, element1, element2, element3]
nodes = [node0, node1, node2, node3, node4]

# Create Load Vector
loads = {
    1: np.array([0.1, -0.05, -0.07, 0, 0, 0]),  # Apply Force to node 1
    2: np.array([0, 0, 0, 0.5, -0.1, 0.3])  # Apply Moment to node 2
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
