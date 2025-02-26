import math
import numpy as np
from Direct_Stiffness_Method import Node, Element, Structure, calculate_structure_response

# Node Definitions
node0 = Node(0, 0, 10, 0, bc=[True, True, True, True, True, True])  # Fully fixed node
node1 = Node(15, 0, 10, 1, bc=[False, False, False, False, False, False])  # Free node
node2 = Node(15, 0, 0, 2, bc=[True, True, True, False, False, False])  # Fixed in translations only

# Define Force and Moment
Force = np.array([0.1, 0.05, -0.07])  # Force in x, y, z direction at node 1
Moment = np.array([0.05, -0.1, 0.25])  # Moment about x, y, z axis at node 1

# Material Properties
E = 1000  # Young's Modulus
nu = 0.3  # Poisson's Ratio
A = 0.5  # Cross-sectional Area
Iy = 0.5**3 / 12  # Moment of Inertia about y-axis
Iz = 0.5 / 12  # Moment of Inertia about z-axis
J = 0.02861  # Polar Moment of Inertia

# Create Elements
element1 = Element(node0, node1, E, nu, A, Iy, Iz, J)  # From node0 to node1
element2 = Element(node1, node2, E, nu, A, Iy, Iz, J)  # From node1 to node2

# Add elements to structure
elements = [element1, element2]
nodes = [node0, node1, node2]

# Create Load Vector
loads = {
    1: np.array([0.1, 0.05, -0.07, 0.05, -0.1, 0.25])  # Apply load to node 1
}

# Boundary Conditions
supports = {
    0: [True, True, True, True, True, True],  # Fully fixed
    1: [False, False, False, False, False, False],  # Free
    2: [True, True, True, False, False, False]  # Fixed in translations only
}

# Calculate structure response
solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
displacements, reactions = solver.solve()

# Reshape and print results
disp_matrix = displacements.reshape((-1, 6))
reac_matrix = reactions.reshape((-1, 6))

disp_dict = {node: disp_matrix[i] for i, node in enumerate(nodes)}
react_dict = {node: reac_matrix[i] for i, node in enumerate(nodes)}

print("Nodal Displacements and Rotations:")
for node, disp in disp_dict.items():
    print(f"Node {node}: [u: {disp[0]:.10f}, v: {disp[1]:.10f}, w: {disp[2]:.10f}, "
          f"rot_x: {disp[3]:.10f}, rot_y: {disp[4]:.10f}, rot_z: {disp[5]:.10f}]")

print("\nReaction Forces and Moments at Supports:")
for node, react in react_dict.items():
    if node in supports:
        print(f"Node {node}: [Fx: {react[0]:.10f}, Fy: {react[1]:.10f}, Fz: {react[2]:.10f}, "
              f"Mx: {react[3]:.10f}, My: {react[4]:.10f}, Mz: {react[5]:.10f}]")
