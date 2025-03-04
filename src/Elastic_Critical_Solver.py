import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Direct_Stiffness_Method as dsm
import functions as fu

def moments_from_forces(forces_dict, elem):
    """Extract moments from internal forces for an element."""
    f = forces_dict[elem]
    return f[6], f[9], f[4], f[5], f[10], f[11]

class CriticalLoadSolver:
    def __init__(self, nodes, elements, loads, include_interactions: bool = False):
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.use_interactions = include_interactions
        self.total_dof = len(nodes) * 6
        self.global_geo_matrix = np.zeros((self.total_dof, self.total_dof))
        self.node_index_map = {node['node_id']: i for i, node in enumerate(nodes)}
        self.free_dofs = [self.node_index_map[node['node_id']] * 6 + i for node in nodes for i in range(6) if not node['bc'][i]]

    def compute_internal_forces_and_moments(self, displacements):
        internal_forces = {}
        for element in self.elements:
            node1, node2 = element['node1'], element['node2']
            idx1, idx2 = self.node_index_map[node1['node_id']] * 6, self.node_index_map[node2['node_id']] * 6
            local_displacements = np.concatenate([displacements[idx1:idx1+6], displacements[idx2:idx2+6]])
            Gamma = dsm.transformation_matrix_3D(element['rotation_matrix'])
            local_forces = element['local_stiffness_matrix'] @ (Gamma @ local_displacements)
            internal_forces[(node1['node_id'], node2['node_id'])] = local_forces
        return internal_forces

    def assemble_geometric_stiffness(self, displacements):
        internal_forces = self.compute_internal_forces_and_moments(displacements)
        for element in self.elements:
            node1, node2 = element['node1'], element['node2']
            elem_length, cross_area, torsion_prop = element['L'], element['A'], element.get('J')
            fx2, mx2, my1, mz1, my2, mz2 = moments_from_forces(internal_forces, (node1['node_id'], node2['node_id']))
            k_local = (fu.local_geometric_stiffness_matrix_3D_beam(
                elem_length, cross_area, torsion_prop, fx2, mx2, my1, mz1, my2, mz2
            ) if self.use_interactions else 
            fu.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(
                elem_length, cross_area, torsion_prop, fx2))
            idx1, idx2 = self.node_index_map[node1['node_id']] * 6, self.node_index_map[node2['node_id']] * 6
            dof_indices = np.arange(idx1, idx1 + 6).tolist() + np.arange(idx2, idx2 + 6).tolist()
            for i, row in enumerate(dof_indices):
                for j, col in enumerate(dof_indices):
                    self.global_geo_matrix[row, col] += k_local[i, j]

    def compute_eigenvalues(self):
        displacements, _ = dsm.calculate_structure_response(self.nodes, self.elements, self.loads)
        self.assemble_geometric_stiffness(displacements)
        k_elastic = dsm.assemble_global_stiffness_matrix(self.nodes, self.elements)
        k_e_red = k_elastic[np.ix_(self.free_dofs, self.free_dofs)]
        k_g_red = self.global_geo_matrix[np.ix_(self.free_dofs, self.free_dofs)]
        
        evals, evecs = eig(k_e_red, -k_g_red)
        evals = np.real(evals[evals > 0])
        sort_indices = np.argsort(evals)
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]
        
        E, I, L = 1000, np.pi / 4, 50
        P_cr_analytical = (np.pi**2 * E * I) / (4 * L**2)
        P_applied = 1.0
        lambda_analytical = P_cr_analytical / P_applied
        
        if evals.size > 0:
            idx = np.argmin(np.abs(evals - lambda_analytical))
            return evals[idx:idx+1], evecs[:, idx:idx+1]
        return np.array([]), np.array([])

    def plot_mode_shape(self, mode_vec, scale_factor=1.0):
        if mode_vec.size == 0:
            return
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        for elem in self.elements:
            n1, n2 = elem['node1'], elem['node2']
            coords = np.array([[n1['x'], n1['y'], n1['z']], [n2['x'], n2['y'], n2['z']]]).T
            ax.plot(coords[0], coords[1], coords[2], 'b-', lw=2)
            start_idx, end_idx = self.node_index_map[n1['node_id']] * 6, self.node_index_map[n2['node_id']] * 6
            full_mode = np.zeros(self.total_dof)
            for i, free_idx in enumerate(self.free_dofs):
                full_mode[free_idx] = mode_vec[i]
            deformed = coords + scale_factor * np.array([
                [full_mode[start_idx], full_mode[end_idx]],
                [full_mode[start_idx+1], full_mode[end_idx+1]],
                [full_mode[start_idx+2], full_mode[end_idx+2]]
            ])
            ax.plot(deformed[0], deformed[1], deformed[2], 'r--', lw=2)
        ax.set_title('Euler Buckling Mode')
        ax.set_xlabel('X-Axis'); ax.set_ylabel('Y-Axis'); ax.set_zlabel('Z-Axis')
        plt.show()

# Input script
nodes = [
    {'node_id': 0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'bc': [True, True, True, True, True, True]},
    {'node_id': 1, 'x': 30.0, 'y': 40.0, 'z': 0.0, 'bc': [False, False, False, False, False, False]}
]
A, E, nu, Iy, Iz, J = np.pi, 1000, 0.3, np.pi / 4, np.pi / 4, np.pi / 2
element = dsm.create_element(nodes[0], nodes[1], E, nu, A, Iy, Iz, J)
elements = [element]
loads = {1: np.array([-0.6, -0.8, 0.0, 0.0, 0.0, 0.0])}

solver = CriticalLoadSolver(nodes, elements, loads, include_interactions=False)
eigenvalues, eigenvectors = solver.compute_eigenvalues()

displacements, _ = dsm.calculate_structure_response(nodes, elements, loads)
print("Displacements:", displacements)
internal_forces = solver.compute_internal_forces_and_moments(displacements)
print("Force and Moment for Element (0,1):", moments_from_forces(internal_forces, (0, 1)))
print("Critical Load Factor:", eigenvalues[0] if eigenvalues.size > 0 else "None")
solver.plot_mode_shape(eigenvectors[:, 0], scale_factor=0.5)
