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
        self.displacement = np.zeros(12)  # Store displacement (12 DOFs per node)
        self.reaction = np.zeros(12)  # Store reaction (12 DOFs per node)

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
        # Assuming 'fu.local_elastic_stiffness_matrix_3D_beam' is defined in your 'functions.py'
        return fu.local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, self.L, self.Iy, self.Iz, self.J)

class Structure:
    def __init__(self, nodes, elements, loads=None, supports=None):
        self.nodes = nodes
        self.elements = elements
        self.loads = loads if loads else {}
        self.supports = supports if supports else {}
        self.node_index_map = {node.node_id: i for i, node in enumerate(nodes)}
        self.ndof = len(nodes) * 6  # Total DOFs
        self.global_stiffness_matrix = self.assemble_global_stiffness_matrix()
        self.load_vector = self.assemble_load_vector()
        self.boundary_conditions = self.apply_boundary_conditions()

    def assemble_global_stiffness_matrix(self):
        """Assembles the global stiffness matrix correctly mapping local DOFs to global DOFs."""
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

    def assemble_load_vector(self):
        """Assembles the global load vector ensuring correct placement of nodal loads."""
        if not isinstance(self.loads, dict):
            raise TypeError("Loads must be provided as a dictionary.")

        F = np.zeros(self.ndof)
        for node_id, load in self.loads.items():
            if not isinstance(load, np.ndarray):
                raise TypeError(f"Load at node {node_id} must be a numpy array.")
            if load.shape != (6,):
                raise ValueError(f"Load vector at node {node_id} must have 6 components.")

            idx = self.node_index_map[node_id] * 6
            F[idx:idx+6] += load  # Correct indexing for 6 DOFs per node

        return F

    def apply_boundary_conditions(self):
        """Applies boundary conditions by modifying the global stiffness matrix for fixed DOFs."""
        K_bc = np.copy(self.global_stiffness_matrix)
        
        for node in self.nodes:
            for i in range(6):  # Iterate over all 6 DOFs (translations + rotations)
                if node.bc[i]:  # If DOF is fixed
                    global_idx = self.node_index_map[node.node_id] * 6 + i
                    # Zero out row and column
                    K_bc[global_idx, :] = 0
                    K_bc[:, global_idx] = 0
                    # Set diagonal to a large number to maintain numerical stability
                    K_bc[global_idx, global_idx] = 1e10

        return K_bc

    def solve_for_displacements(self, K_ff, F_f):
        """Solve for displacements at free DOFs."""
        # Check for singularity using condition number
        cond_number = np.linalg.cond(K_ff)
        if cond_number > 1e12:  # Threshold for numerical singularity
            raise np.linalg.LinAlgError(f"Stiffness matrix is nearly singular (Condition Number: {cond_number})")

        # Solve for displacements of free DOFs
        delta_f = np.linalg.solve(K_ff, F_f)
        
        # Check for NaN or Inf in results
        if np.any(np.isnan(delta_f)) or np.any(np.isinf(delta_f)):
            raise np.linalg.LinAlgError("Solution contains NaN or Inf, indicating singular system.")
        
        # Check for unreasonably large values in the displacement vector
        if np.any(np.abs(delta_f) > 1e6):  # Threshold for unreasonably large displacements
            raise np.linalg.LinAlgError("Unreasonably large displacements, indicating numerical instability.")
        
        return delta_f

    def solve_for_reactions(self, K_sf, delta_f):
        """Solve for reactions at supported DOFs."""
        F_s = np.dot(K_sf, delta_f)  # Reaction forces at fixed DOFs
        return F_s

def calculate_structure_response(nodes, elements, load_vector):
    structure = Structure(nodes, elements)
    structure.apply_loads(load_vector)

    # Partition the global stiffness matrix
    K_ff, K_fs, K_sf, K_ss, F_f, F_s, free_dofs, supported_dofs = structure.partition_matrices()

    # Solve for displacements at the free DOFs
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
