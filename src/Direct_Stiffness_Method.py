import numpy as np
import functions as fu

class Node:
    def __init__(self, x, y, z, node_id, is_fixed=False):
        self.x = x
        self.y = y
        self.z = z
        self.node_id = node_id
        self.is_fixed = is_fixed

class Element:
    def __init__(self, node1, node2, E, nu, A, L, Iy, Iz, J):
        self.node1 = node1
        self.node2 = node2
        self.E = E
        self.nu = nu
        self.A = A
        self.L = L
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.local_stiffness_matrix = self.compute_local_stiffness_matrix()

    def compute_local_stiffness_matrix(self):
        return fu.local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, self.L, self.Iy, self.Iz, self.J)

class Structure:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.global_stiffness_matrix = self.assemble_global_stiffness_matrix()
        self.load_vector = np.zeros(12 * len(nodes)) 
        self.boundary_conditions = self.apply_boundary_conditions()

    def assemble_global_stiffness_matrix(self):
        size = 12 * len(self.nodes)
        global_stiffness_matrix = np.zeros((size, size))
        
        for element in self.elements:
            element_matrix = element.local_stiffness_matrix
            global_stiffness_matrix += self.map_to_global_dof(element_matrix, element)
        
        return global_stiffness_matrix

    def map_to_global_dof(self, local_matrix, element):
        size = 12 * len(self.nodes)
        global_matrix = np.zeros((size, size))
        
        node1_id, node2_id = element.node1.node_id, element.node2.node_id
        
        for i in range(12):
            for j in range(12):
                global_row = node1_id * 12 + i
                global_col = node2_id * 12 + j
                global_matrix[global_row, global_col] = local_matrix[i, j]
        
        return global_matrix

    def apply_boundary_conditions(self):
        boundary_conditions = np.copy(self.global_stiffness_matrix)
        for node in self.nodes:
            if node.is_fixed:
                # Set all degrees of freedom for this node to zero in the global stiffness matrix
                for i in range(12):
                    # Set the entire row and column for the fixed DOF to zero
                    boundary_conditions[12 * node.node_id + i, :] = 0
                    boundary_conditions[:, 12 * node.node_id + i] = 0
                # Set the diagonal element to a large value (to avoid division by zero in solving)
                for i in range(12):
                    boundary_conditions[12 * node.node_id + i, 12 * node.node_id + i] = 1e10
        return boundary_conditions

    def apply_loads(self, load_vector):
        self.load_vector = load_vector

    def solve_for_displacements(self):
        displacements = np.linalg.solve(self.global_stiffness_matrix, self.load_vector)
        return displacements

    def solve_for_reactions(self, displacements):
        reactions = np.dot(self.global_stiffness_matrix, displacements)
        return reactions

def calculate_structure_response(nodes, elements, load_vector):
    structure = Structure(nodes, elements)
    structure.apply_loads(load_vector)
    displacements = structure.solve_for_displacements()
    reactions = structure.solve_for_reactions(displacements)
    
    for i, node in enumerate(nodes):
        node.displacement = displacements[12 * i : 12 * (i + 1)]
        node.reaction = reactions[12 * i : 12 * (i + 1)]
    
    node_displacements = {node.node_id: node.displacement for node in nodes}
    node_reactions = {node.node_id: node.reaction for node in nodes}
    
    return node_displacements, node_reactions
