import numpy as np
import functions as fu  # Assuming functions.py is already available

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

# Create Element
element1 = Element(node0, node1, E, nu, A, 15, Iy, Iz, J)  # From node0 to node1
element2 = Element(node1, node2, E, nu, A, 15, Iy, Iz, J)  # From node1 to node2

# Add elements to structure
elements = [element1, element2]

# Create Structure
structure = Structure(nodes=[node0, node1, node2], elements=elements)

# Apply Loads (adding Force and Moment to node1's load vector)
load_vector = np.zeros(12 * len([node0, node1, node2]))  # 12 DOFs per node
load_vector[12 * 1:12 * 1 + 3] = Force  # Apply force to node1's translation DOFs
load_vector[12 * 1 + 3:12 * 1 + 6] = Moment  # Apply moment to node1's rotational DOFs

structure.apply_loads(load_vector)

# Solve for displacements and reactions
displacements = structure.solve_for_displacements()
reactions = structure.solve_for_reactions(displacements)

# Output results
print("Displacements:", displacements)
print("Reactions:", reactions)
