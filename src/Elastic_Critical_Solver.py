import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Direct_Stiffness_Method as dsm
import functions as fu

def moments_ftom_forces(forces_dict, elem):
    """Extract moments from internal forces for an element"""
    f = forces_dict[elem]
    return f[6], f[9], f[4], f[5], f[10], f[11]

class CriticalLoadSolver:
    def __init__(self, frame_solver: Frame3DSolver, include_interactions: bool = False):
        self.solver = frame_solver
        self.use_interactions = include_interactions
        self.total_dof = frame_solver.ndof
        self.global_geo_matrix = np.zeros((self.total_dof, self.total_dof))

    def assemble_geometric_stiffness(self, d):
        internal_forces = self.solver.compute_internal_forces_and_moments(d)
        
        for element in self.solver.elements:
            node_start, node_end, props = element
            coord1 = self.solver.nodes[node_start]
            coord2 = self.solver.nodes[node_end]
            elem_length = np.linalg.norm(coord2 - coord1)
            
            cross_area = props["A"]
            torsion_prop = props.get("I_rho", props["J"])
            
            fx2, mx2, my1, mz1, my2, mz2 = moments_from_forces(internal_forces, (node_start, node_end))
            
            k_local = (fu.local_geometric_stiffness_matrix_3D_beam(
                elem_length, cross_area, torsion_prop, fx2, mx2, my1, mz1, my2, mz2
            ) if self.use_interactions else 
            fu.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(
                elem_length, cross_area, torsion_prop, fx2))
            
            idx1 = self.solver.node_index_map[node_start] * 6
            idx2 = self.solver.node_index_map[node_end] * 6
            dof_indices = np.concatenate([np.arange(idx1, idx1 + 6), np.arange(idx2, idx2 + 6)])
            
            for i, row in enumerate(dof_indices):
                for j, col in enumerate(dof_indices):
                    self.global_geo_matrix[row, col] += k_local[i, j]

    def compute_eigenvalues(self):
        displacements, _ = self.solver.solve()
        self.assemble_geometric_stiffness(displacements)
        k_elastic = self.solver.assemble_stiffness()
        
        evals, evecs = eig(k_elastic, self.global_geo_matrix)
        evals = np.real(evals[evals > 0])
        sort_indices = np.argsort(evals)
        return evals[sort_indices], evecs[:, sort_indices]

    def plot_mode_shape(self, mode_vec, scale_factor=1.0):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for elem in self.solver.elements:
            n1, n2, _ = elem
            coords = np.array([self.solver.nodes[n1], self.solver.nodes[n2]]).T
            ax.plot(coords[0], coords[1], coords[2], 'b-', lw=2)
            
            start_idx = self.solver.node_index_map[n1] * 6
            end_idx = self.solver.node_index_map[n2] * 6
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
