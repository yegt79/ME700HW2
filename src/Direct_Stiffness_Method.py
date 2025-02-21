import numpy as np
import functions as fu

class BeamElement:
    def __init__(self, E, A, Iy, Iz, J, nodes):
        """
        Initialize the beam element with the material properties and nodes.

        Parameters:
        E (float): Young's Modulus
        A (float): Cross-sectional area
        Iy (float): Moment of inertia about the y-axis
        Iz (float): Moment of inertia about the z-axis
        J (float): Polar moment of inertia
        nodes (int): Number of nodes in the beam element
        """
        self.E = E
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.nodes = nodes

    def length(self, node_positions):
        """
        Compute the length of the beam element between two nodes.
        
        Parameters:
        node_positions (list): The coordinates of the nodes.
        
        Returns:
        float: Length of the beam element.
        """
        return np.linalg.norm(np.array(node_positions[-1]) - np.array(node_positions[0]))

    def stiffness_matrix(self, node_positions):
        """
        Compute the global stiffness matrix for the beam element, 
        which is generalized for any number of nodes with 6 DOF each.
        
        Parameters:
        node_positions (list): The coordinates of the nodes.
        
        Returns:
        numpy.ndarray: The global stiffness matrix.
        """
        # Length of the beam
        L = self.length(node_positions)

        # Material properties
        E, A, Iy, Iz, J = self.E, self.A, self.Iy, self.Iz, self.J

        # Initialize the global stiffness matrix
        num_dofs = self.nodes * 6  # 6 DOFs per node (3 translational + 3 rotational)
        K_global = np.zeros((num_dofs, num_dofs))

        # Loop through each element to build the local stiffness matrix and add it to the global matrix
        for i in range(self.nodes - 1):  # (nodes-1) beam elements
            k_local = fu.local_stiffness_matrix(L, E, A, Iy, Iz, J)
            
            # Calculate the node index for the degrees of freedom for the element
            start_dof = 6 * i
            end_dof = 6 * (i + 2)  # 6 DOFs for each node involved in the element
            
            # Add the local stiffness to the global stiffness matrix
            K_global[start_dof:end_dof, start_dof:end_dof] += k_local
        
        return K_global
