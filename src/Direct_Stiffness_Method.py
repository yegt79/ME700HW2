import math
import numpy as np
import functions as fu

class Node:
    def __init__(self, x, y, z, node_id, bc=None):
        self.x = x
        self.y = y
        self.z = z
        self.node_id = node_id
        self.bc = bc if bc else [False, False, False, False, False, False]  # 6 DOFs: UX, UY, UZ, RX, RY, RZ
        self.displacement = None  # Store displacement
        self.reaction = None  # Store reaction

class Element:
    def __init__(self, node1, node2, E, nu, A, Iy, Iz, J):
        self.node1 = node1
        self.node2 = node2
        self.E = E
        self.nu = nu
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.L = self.calculate_length()
        self.local_stiffness_matrix = self.compute_local_stiffness_matrix()

    def calculate_length(self):
        # Calculate the distance between the two nodes (Euclidean distance)
        x1, y1, z1 = self.node1.x, self.node1.y, self.node1.z
        x2, y2, z2 = self.node2.x, self.node2.y, self.node2.z
        # Euclidean distance formula
        L = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return L

    def compute_local_stiffness_matrix(self):
        return fu.local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, self.L, self.Iy, self.Iz, self.J)

class Structure:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.node_index_map = {node.node_id: i for i, node in enumerate(nodes)}
        self.ndof = len(nodes) * 6  # Total DOFs
        self.global_stiffness_matrix = self.assemble_global_stiffness_matrix()
        self.load_vector = np.zeros(self.ndof)  # 6 DOFs per node
        self.boundary_conditions = self.apply_boundary_conditions()

    def assemble_global_stiffness_matrix(self):
        K = np.zeros((self.ndof, self.ndof))  # Global stiffness matrix
        
        for element in self.elements:
            k_local = element.local_stiffness_matrix
            node1_id = element.node1.node_id
            node2_id = element.node2.node_id
            
            idx1 = self.node_index_map[node1_id] * 6
            idx2 = self.node_index_map[node2_id] * 6

            # Correctly map element stiffness to global matrix
            dof_indices = np.array([idx1, idx1+1, idx1+2, idx1+3, idx1+4, idx1+5,
                                    idx2, idx2+1, idx2+2, idx2+3, idx2+4, idx2+5])
            
            for i in range(12):
                for j in range(12):
                    K[dof_indices[i], dof_indices[j]] += k_local[i, j]

        return K

    def apply_boundary_conditions(self):
        boundary_conditions = np.copy(self.global_stiffness_matrix)
        for node in self.nodes:
            for i in range(6):  # Only iterate over the first 6 DOFs (translations + rotations)
                if node.bc[i]:  # If this DOF is fixed
                    # Set the entire row and column to zero for this DOF
                    boundary_conditions[12 * node.node_id + i, :] = 0
                    boundary_conditions[:, 12 * node.node_id + i] = 0
                    # Set diagonal to a large value to prevent singularity
                    boundary_conditions[12 * node.node_id + i, 12 * node.node_id + i] = 1e10
        return boundary_conditions

    def apply_loads(self, load_vector):
        self.load_vector = load_vector

    def partition_matrices(self):
        # Determine free and supported DOFs
        free_dofs = []
        supported_dofs = []
        for i, node in enumerate(self.nodes):
            for j in range(6):
                if node.bc[j]:  # This DOF is fixed
                    supported_dofs.append(i * 12 + j)
                else:  # This DOF is free
                    free_dofs.append(i * 12 + j)

        # Partitioning the stiffness matrix
        K_ff = self.global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]
        K_fs = self.global_stiffness_matrix[np.ix_(free_dofs, supported_dofs)]
        K_sf = self.global_stiffness_matrix[np.ix_(supported_dofs, free_dofs)]
        K_ss = self.global_stiffness_matrix[np.ix_(supported_dofs, supported_dofs)]
        
        # Partition the load vector
        F_f = self.load_vector[free_dofs]
        F_s = self.load_vector[supported_dofs]

        return K_ff, K_fs, K_sf, K_ss, F_f, F_s, free_dofs, supported_dofs

    def solve_for_displacements(self, K_ff, F_f):
        # Solve for displacements of free DOFs
        delta_f = np.linalg.solve(K_ff, F_f)
        return delta_f

    def solve_for_reactions(self, K_sf, delta_f):
        # Solve for reactions (forces at supported DOFs)
        F_s = np.dot(K_sf, delta_f)
        return F_s

def calculate_structure_response(nodes, elements, load_vector):
    structure = Structure(nodes, elements)
    structure.apply_loads(load_vector)

    # Partition the global stiffness matrix
    K_ff, K_fs, K_sf, K_ss, F_f, F_s, free_dofs, supported_dofs = structure.partition_matrices()

    # Solve for the displacements at the free DOFs
    delta_f = structure.solve_for_displacements(K_ff, F_f)

    # Solve for the reactions (forces and moments at the supported DOFs)
    F_s = structure.solve_for_reactions(K_sf, delta_f)

    # Assign the results to the nodes
    for i, node in enumerate(nodes):
        node.displacement = np.zeros(12)
        if i in free_dofs:
            node.displacement = delta_f[free_dofs.index(i)]
        node.reaction = np.zeros(12)
        if i in supported_dofs:
            node.reaction = F_s[supported_dofs.index(i)]

    # Return the displacements and reactions
    node_displacements = {node.node_id: node.displacement for node in nodes}
    node_reactions = {node.node_id: node.reaction for node in nodes}

    return node_displacements, node_reactions
