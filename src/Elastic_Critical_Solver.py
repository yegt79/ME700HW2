# Elastic_Critical_Solver.py
import numpy as np
from scipy.linalg import eig
import Direct_Stiffness_Method as dsm
import functions as fu

class ElasticCriticalLoadSolver:
    """
    Elastic Critical Load Solver for 3D Frames using the Direct Stiffness Method.
    Computes critical load factors and buckling modes for a given structure.
    """

    def __init__(self, nodes, elements, loads, use_interaction_terms: bool = False):
        """
        Initialize the solver with frame geometry, material properties, loads, and options.

        Parameters
        ----------
        nodes : list
            List of node dictionaries from dsm.create_node.
        elements : list
            List of element dictionaries from dsm.create_element.
        loads : dict
            Dictionary of loads keyed by node_id, with 6-component load vectors.
        use_interaction_terms : bool, optional
            If True, uses geometric stiffness with interaction terms.
            If False, uses geometric stiffness without interaction terms (default).
        """
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.use_interaction_terms = use_interaction_terms
        self.ndof = len(nodes) * 6  # 6 DOF per node
        self.global_geometric_stiffness = np.zeros((self.ndof, self.ndof))
        self.node_index_map = {node['node_id']: i for i, node in enumerate(nodes)}

    def compute_internal_forces_and_moments(self, displacements: np.ndarray) -> dict:
        """
        Compute internal forces and moments for each element based on displacements.

        Parameters
        ----------
        displacements : np.ndarray
            Global displacement vector from static analysis.

        Returns
        -------
        dict
            Internal forces for each element, keyed by (node1_id, node2_id) tuples.
        """
        internal_forces = {}
        for elem in self.elements:
            node1_id = elem['node1']['node_id']
            node2_id = elem['node2']['node_id']
            idx1 = self.node_index_map[node1_id] * 6
            idx2 = self.node_index_map[node2_id] * 6
            local_dofs = np.array([idx1, idx1+1, idx1+2, idx1+3, idx1+4, idx1+5,
                                   idx2, idx2+1, idx2+2, idx2+3, idx2+4, idx2+5])
            local_displacements = displacements[local_dofs]
            Gamma = fu.transformation_matrix_3D(elem['rotation_matrix'])
            local_displacements = Gamma @ local_displacements
            local_forces = elem['local_stiffness_matrix'] @ local_displacements
            internal_forces[(node1_id, node2_id)] = local_forces
        return internal_forces

    def extract_moments_from_internal_forces(self, internal_forces, element) -> tuple:
        """
        Extract axial force and moments from internal forces for a given element.

        Parameters
        ----------
        internal_forces : dict
            Internal forces for each element.
        element : dict
            Element dictionary from dsm.create_element.

        Returns
        -------
        tuple
            (Fx2, Mx2, My1, Mz1, My2, Mz2) extracted forces and moments.
        """
        node1_id = element['node1']['node_id']
        node2_id = element['node2']['node_id']
        forces = internal_forces[(node1_id, node2_id)]
        Fx2 = forces[6]
        Mx2 = forces[9]
        My1 = forces[4]
        Mz1 = forces[5]
        My2 = forces[10]
        Mz2 = forces[11]
        return Fx2, Mx2, My1, Mz1, My2, Mz2

    def assemble_geometric_stiffness(self, displacements: np.ndarray):
        """
        Assemble the global geometric stiffness matrix using local geometric stiffness matrices.

        Parameters
        ----------
        displacements : np.ndarray
            Global displacement vector from the static analysis.
        """
        self.global_geometric_stiffness.fill(0)
        internal_forces = self.compute_internal_forces_and_moments(displacements)

        for elem in self.elements:
            L = elem['L']
            A = elem['A']
            I_rho = elem['J']
            node1_id = elem['node1']['node_id']
            node2_id = elem['node2']['node_id']

            Fx2, Mx2, My1, Mz1, My2, Mz2 = self.extract_moments_from_internal_forces(internal_forces, elem)

            if self.use_interaction_terms:
                k_geo = fu.local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
            else:
                k_geo = fu.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(L, A, I_rho, Fx2)

            idx1 = self.node_index_map[node1_id] * 6
            idx2 = self.node_index_map[node2_id] * 6
            dof_indices = np.array([idx1, idx1+1, idx1+2, idx1+3, idx1+4, idx1+5,
                                    idx2, idx2+1, idx2+2, idx2+3, idx2+4, idx2+5])

            for i in range(12):
                for j in range(12):
                    self.global_geometric_stiffness[dof_indices[i], dof_indices[j]] += k_geo[i, j]

    def solve_eigenvalue_problem(self):
        """
        Solve the generalized eigenvalue problem to find critical load factors.

        Returns
        -------
        eigenvalues : np.ndarray
            Array of critical load factors.
        eigenvectors : np.ndarray
            Array of buckling mode shapes (full DOF size).
        """
        displacements, _ = dsm.calculate_structure_response(self.nodes, self.elements, self.loads)
        K = dsm.assemble_global_stiffness_matrix(self.nodes, self.elements)
        K_reduced, _, free_dof, _ = dsm.apply_boundary_conditions(self.nodes, K, np.zeros(self.ndof))
        self.assemble_geometric_stiffness(displacements)
        Kg_reduced = self.global_geometric_stiffness[np.ix_(free_dof, free_dof)]
        eigenvalues, eigenvectors = eig(K_reduced, -Kg_reduced)
        eigenvalues = np.real(eigenvalues)
        mask = eigenvalues > 0
        eigenvalues = eigenvalues[mask]
        eigenvectors = eigenvectors[:, mask]
        # Map to full DOF space
        full_eigenvectors = np.zeros((self.ndof, eigenvectors.shape[1]))
        full_eigenvectors[free_dof, :] = eigenvectors
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        full_eigenvectors = full_eigenvectors[:, idx]
        return eigenvalues, full_eigenvectors

    def plot_buckling_mode(self, eigenvector, scale: float = 1.0):
        """
        Plot the buckling mode shape corresponding to an eigenvector.

        Parameters
        ----------
        eigenvector : np.ndarray
            The eigenvector representing the buckling mode shape (full DOF size).
        scale : float, optional
            Scaling factor for the deformations (default is 1.0).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        for elem in self.elements:
            node1 = elem['node1']
            node2 = elem['node2']
            x = [node1['x'], node2['x']]
            y = [node1['y'], node2['y']]
            z = [node1['z'], node2['z']]
            ax.plot(x, y, z, color='blue', label='Undeformed', linewidth=2.5)
            idx1 = self.node_index_map[node1['node_id']] * 6
            idx2 = self.node_index_map[node2['node_id']] * 6
            xd = [x[0] + scale * eigenvector[idx1], x[1] + scale * eigenvector[idx2]]
            yd = [y[0] + scale * eigenvector[idx1+1], y[1] + scale * eigenvector[idx2+1]]
            zd = [z[0] + scale * eigenvector[idx1+2], z[1] + scale * eigenvector[idx2+2]]
            ax.plot(xd, yd, zd, color='red', linestyle='--', label='Buckling Mode', linewidth=2.5)
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))
        ax.set_title('Euler Buckling Mode Shape', fontsize=20, fontweight='bold')
        ax.set_xlabel('X', fontsize=15, fontweight='bold')
        ax.set_ylabel('Y', fontsize=15, fontweight='bold')
        ax.set_zlabel('Z', fontsize=15, fontweight='bold')
        plt.show()
