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
        """Initialize with nodes, elements, and loads from dsm."""
        self.nodes = nodes  # List of node dicts
        self.elements = elements  # List of element dicts
        self.loads = loads  # Dict of node_id: load_vector
        self.use_interactions = include_interactions
        self.total_dof = len(nodes) * 6
        self.global_geo_matrix = np.zeros((self.total_dof, self.total_dof))
        self.node_index_map = {node['node_id']: i for i, node in enumerate(nodes)}

    def compute_internal_forces_and_moments(self, displacements):
        """Compute internal forces and moments for each element."""
        internal_forces = {}
        for element in self.elements:
            node1 = element['node1']
            node2 = element['node2']
            idx1 = self.node_index_map[node1['node_id']] * 6
            idx2 = self.node_index_map[node2['node_id']] * 6
            local_displacements = np.concatenate([
                displacements[idx1:idx1+6],
                displacements[idx2:idx2+6]
            ])
            # Transform global displacements to local coordinates
            Gamma = dsm.transformation_matrix_3D(element['rotation_matrix'])
            local_d = Gamma @ local_displacements
            # Compute local forces: F_local = k_local @ d_local
            local_forces = element['local_stiffness_matrix'] @ local_d
            internal_forces[(node1['node_id'], node2['node_id'])] = local_forces
        return internal_forces

    def assemble_geometric_stiffness(self, displacements):
        """Assemble the global geometric stiffness matrix."""
        internal_forces = self.compute_internal_forces_and_moments(displacements)
        
        for element in self.elements:
            node1 = element['node1']
            node2 = element['node2']
            elem_length = element['L']
            cross_area = element['A']
            torsion_prop = element.get('J')  # Use 'J' as in your dsm
            
            fx2, mx2, my1, mz1, my2, mz2 = moments_from_forces(
                internal_forces, (node1['node_id'], node2['node_id'])
            )
            
            k_local = (fu.local_geometric_stiffness_matrix_3D_beam(
                elem_length, cross_area, torsion_prop, fx2, mx2, my1, mz1, my2, mz2
            ) if self.use_interactions else 
            fu.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(
                elem_length, cross_area, torsion_prop, fx2))
            
            idx1 = self.node_index_map[node1['node_id']] * 6
            idx2 = self.node_index_map[node2['node_id']] * 6
            dof_indices = np.concatenate([np.arange(idx1, idx1 + 6), np.arange(idx2, idx2 + 6)])
            
            for i, row in enumerate(dof_indices):
                for j, col in enumerate(dof_indices):
                    self.global_geo_matrix[row, col] += k_local[i, j]

    def compute_eigenvalues(self):
        """Solve for critical load factors and mode shapes."""
        # Solve static problem using dsm
        displacements, _ = dsm.calculate_structure_response(self.nodes, self.elements, self.loads)
        self.assemble_geometric_stiffness(displacements)
        k_elastic = dsm.assemble_global_stiffness_matrix(self.nodes, self.elements)
        
        evals, evecs = eig(k_elastic, self.global_geo_matrix)
        evals = np.real(evals[evals > 0])
        sort_indices = np.argsort(evals)
        return evals[sort_indices], evecs[:, sort_indices]

    def plot_mode_shape(self, mode_vec, scale_factor=1.0):
        """Plot the buckling mode shape."""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for elem in self.elements:
            n1, n2 = elem['node1'], elem['node2']
            coords = np.array([[n1['x'], n1['y'], n1['z']], [n2['x'], n2['y'], n2['z']]]).T
            ax.plot(coords[0], coords[1], coords[2], 'b-', lw=2)
            
            start_idx = self.node_index_map[n1['node_id']] * 6
            end_idx = self.node_index_map[n2['node_id']] * 6
            deformed = coords + scale_factor * np.array([
                [mode_vec[start_idx], mode_vec[end_idx]],
                [mode_vec[start_idx+1], mode_vec[end_idx+1]],
                [mode_vec[start_idx+2], mode_vec[end_idx+2]]
            ])
            ax.plot(deformed[0], deformed[1], deformed[2], 'r--', lw=2)
            
        ax.set_title('Buckling Mode Visualization')
        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')
        plt.show()
