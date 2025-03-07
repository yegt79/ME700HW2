# Direct_Stiffness_Methodd.py
import numpy as np
import scipy.linalg as sp
import functions as fu
from functions import rotation_matrix_3D, transformation_matrix_3D
from typing import List, Dict, Tuple, Optional

class BeamComponent:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray, E: float, nu: float, A: float, Iy: float, Iz: float, J: float):
        self.nodes = np.array(nodes, dtype=float)
        self.elements_raw = elements  # Store raw input
        self._validate_inputs()  # Validate before conversion
        self.elements = np.array(elements, dtype=int)  # Convert after validation
        self.E = E
        self.nu = nu
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self._validate_material_properties()
        self.bc = np.full((self.nodes.shape[0], 6), False)
        self.displacements = np.zeros((self.nodes.shape[0], 6))
        self.reactions = np.zeros((self.nodes.shape[0], 6))
        self.element_properties = self._compute_element_properties()

    def _validate_inputs(self):
        if not isinstance(self.nodes, np.ndarray):
            raise ValueError(f"Nodes must be a NumPy array, got {type(self.nodes)}.")
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 4:
            raise ValueError(f"Nodes must be a 2D array with shape (n_nodes, 4), got {self.nodes.shape}.")
        if not isinstance(self.elements_raw, np.ndarray):
            raise ValueError(f"Elements must be a NumPy array, got {type(self.elements_raw)}.")
        if self.elements_raw.ndim != 2 or self.elements_raw.shape[1] != 2:
            raise ValueError(f"Elements must be a 2D array with shape (n_elements, 2), got {self.elements_raw.shape}.")
        valid_node_ids = set(self.nodes[:, 3].astype(int))
        max_node_id = int(self.nodes[:, 3].max())
        for i, element in enumerate(self.elements_raw):
            node1_id, node2_id = element
            if not isinstance(node1_id, (int, np.integer)) or (isinstance(node1_id, float) and not node1_id.is_integer()) or node1_id < 0 or node1_id > max_node_id or node1_id not in valid_node_ids:
                raise ValueError(f"Element {i} has invalid node1_id {node1_id}; must be an integer in nodes array (max {max_node_id}).")
            if not isinstance(node2_id, (int, np.integer)) or (isinstance(node2_id, float) and not node2_id.is_integer()) or node2_id < 0 or node2_id > max_node_id or node2_id not in valid_node_ids:
                raise ValueError(f"Element {i} has invalid node2_id {node2_id}; must be an integer in nodes array (max {max_node_id}).")

    def _validate_material_properties(self):
        if not isinstance(self.E, (int, float)) or self.E <= 0:
            raise ValueError(f"Young's Modulus (E) must be positive, got {self.E}.")
        if not isinstance(self.nu, (int, float)) or self.nu < -1 or self.nu > 0.5:
            raise ValueError(f"Poisson's Ratio (nu) must be between -1 and 0.5, got {self.nu}.")
        if not isinstance(self.A, (int, float)) or self.A <= 0:
            raise ValueError(f"Cross-sectional area (A) must be positive, got {self.A}.")
        if not isinstance(self.Iy, (int, float)) or self.Iy <= 0:
            raise ValueError(f"Moment of inertia about y-axis (Iy) must be positive, got {self.Iy}.")
        if not isinstance(self.Iz, (int, float)) or self.Iz <= 0:
            raise ValueError(f"Moment of inertia about z-axis (Iz) must be positive, got {self.Iz}.")
        if not isinstance(self.J, (int, float)) or self.J <= 0:
            raise ValueError(f"Polar moment of inertia (J) must be positive, got {self.J}.")

    def _compute_element_properties(self):
        pass

class BoundaryCondition:
    """A class to manage boundary conditions and loads for a beam structure."""
    
    def __init__(self, fixed_nodes: Dict[int, Tuple[Optional[float], Optional[float], Optional[float],
                                                    Optional[float], Optional[float], Optional[float]]]):
        self.fixed_nodes = fixed_nodes
        self.applied_loads = {}
        self._check_fixed_nodes()

    def _check_fixed_nodes(self):
        for node_id, constraints in self.fixed_nodes.items():
            if not isinstance(node_id, int) or node_id < 0:
                raise ValueError(f"Node ID {node_id} must be a non-negative integer.")
            if len(constraints) != 6:
                raise ValueError(f"Constraints for node {node_id} must be a tuple of 6 DOFs.")
            for i, constraint in enumerate(constraints):
                if constraint is not None and not isinstance(constraint, (int, float)):
                    raise ValueError(f"Constraint {constraint} at DOF {i} for node {node_id} must be a number or None.")

    def _check_loads(self):
        for node_id, loads in self.applied_loads.items():
            if not isinstance(node_id, int) or node_id < 0:
                raise ValueError(f"Node ID {node_id} must be a non-negative integer.")
            if len(loads) != 6:
                raise ValueError(f"Loads for node {node_id} must be a tuple of 6 values.")
            for i, load in enumerate(loads):
                if not isinstance(load, (int, float)):
                    raise ValueError(f"Load {load} at DOF {i} for node {node_id} must be a number.")

    def apply_load(self, node_id: int, load: Tuple[float, float, float, float, float, float]):
        self.applied_loads[node_id] = load
        self._check_loads()

    def add_fixed_support(self, node_id: int):
        self.fixed_nodes[node_id] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._check_fixed_nodes()

    def add_pinned_support(self, node_id: int):
        self.fixed_nodes[node_id] = (0.0, 0.0, 0.0, None, None, None)
        self._check_fixed_nodes()

    def get_dof_constraints(self) -> Dict[int, List[bool]]:
        constraints = {}
        for node_id, conditions in self.fixed_nodes.items():
            constraints[node_id] = [cond is not None for cond in conditions]
        return constraints

class BeamSolver:
    """A class to solve the beam structure for displacements, reactions, and buckling analysis."""
    
    def __init__(self, beam: 'BeamComponent', bc: 'BoundaryCondition'):
        self.beam = beam
        self.bc = bc
        self.internal_forces = None
        self.buckling_forces = {}  # Added for buckling mode forces

    def build_stiffness_matrix(self) -> np.ndarray:
        n_nodes = self.beam.nodes.shape[0]
        n_dofs = n_nodes * 6
        K_global = np.zeros((n_dofs, n_dofs))

        for elem_idx, (node1_id, node2_id) in enumerate(self.beam.elements):
            node1_idx = np.where(self.beam.nodes[:, 3] == node1_id)[0][0]
            node2_idx = np.where(self.beam.nodes[:, 3] == node2_id)[0][0]
            node1_coords = self.beam.nodes[node1_idx, :3]
            node2_coords = self.beam.nodes[node2_idx, :3]

            L = np.linalg.norm(node2_coords - node1_coords)
            k_local = fu.local_elastic_stiffness_matrix_3D_beam(
                self.beam.E, self.beam.nu, self.beam.A, L, self.beam.Iy, self.beam.Iz, self.beam.J
            )
            gamma = rotation_matrix_3D(*node1_coords, *node2_coords)
            T = transformation_matrix_3D(gamma)
            k_global = T.T @ k_local @ T

            dofs = np.array([
                node1_idx * 6, node1_idx * 6 + 1, node1_idx * 6 + 2, node1_idx * 6 + 3, node1_idx * 6 + 4, node1_idx * 6 + 5,
                node2_idx * 6, node2_idx * 6 + 1, node2_idx * 6 + 2, node2_idx * 6 + 3, node2_idx * 6 + 4, node2_idx * 6 + 5
            ])
            for i in range(12):
                for j in range(12):
                    K_global[dofs[i], dofs[j]] += k_global[i, j]

        return K_global

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        n_nodes = self.beam.nodes.shape[0]
        K_global = self.build_stiffness_matrix()
        n_dofs = K_global.shape[0]
        displacements = np.zeros(n_dofs)
        reactions = np.zeros(n_dofs)
        force_vector = np.zeros(n_dofs)

        known_dofs = []
        unknown_dofs = []
        for node_id, constraints in self.bc.fixed_nodes.items():
            node_idx = np.where(self.beam.nodes[:, 3] == node_id)[0][0]
            dofs = [node_idx * 6 + i for i in range(6)]
            for i, constraint in enumerate(constraints):
                if constraint is not None:
                    known_dofs.append(dofs[i])
                    displacements[dofs[i]] = constraint

        unknown_dofs = [i for i in range(n_dofs) if i not in known_dofs]

        for node_id, load in self.bc.applied_loads.items():
            node_idx = np.where(self.beam.nodes[:, 3] == node_id)[0][0]
            dofs = [node_idx * 6 + i for i in range(6)]
            for i, val in enumerate(load):
                force_vector[dofs[i]] = val

        K_uu = K_global[np.ix_(unknown_dofs, unknown_dofs)]
        K_uk = K_global[np.ix_(unknown_dofs, known_dofs)]
        K_ku = K_global[np.ix_(known_dofs, unknown_dofs)]
        F_u = force_vector[unknown_dofs]
        F_k = force_vector[known_dofs]
        d_k = displacements[known_dofs]

        d_u = np.linalg.solve(K_uu, F_u - K_uk @ d_k)
        displacements[unknown_dofs] = d_u
        reactions[known_dofs] = K_ku @ d_u + K_global[np.ix_(known_dofs, known_dofs)] @ d_k - F_k

        displacements_reshaped = displacements.reshape(n_nodes, 6)
        reactions_reshaped = reactions.reshape(n_nodes, 6)
        
        self.internal_forces = self.compute_element_forces(displacements_reshaped)
        
        return displacements_reshaped, reactions_reshaped

    def compute_element_forces(self, displacements: np.ndarray) -> Dict[int, np.ndarray]:
        displacements_flat = displacements.flatten()
        element_forces = {}

        for elem_idx, (node1_id, node2_id) in enumerate(self.beam.elements):
            node1_idx = np.where(self.beam.nodes[:, 3] == node1_id)[0][0]
            node2_idx = np.where(self.beam.nodes[:, 3] == node2_id)[0][0]
            node1_coords = self.beam.nodes[node1_idx, :3]
            node2_coords = self.beam.nodes[node2_idx, :3]

            L = np.linalg.norm(node2_coords - node1_coords)
            k_local = fu.local_elastic_stiffness_matrix_3D_beam(
                self.beam.E, self.beam.nu, self.beam.A, L, self.beam.Iy, self.beam.Iz, self.beam.J
            )
            gamma = rotation_matrix_3D(*node1_coords, *node2_coords)
            T = transformation_matrix_3D(gamma)

            dofs = np.array([
                node1_idx * 6, node1_idx * 6 + 1, node1_idx * 6 + 2, node1_idx * 6 + 3, node1_idx * 6 + 4, node1_idx * 6 + 5,
                node2_idx * 6, node2_idx * 6 + 1, node2_idx * 6 + 2, node2_idx * 6 + 3, node2_idx * 6 + 4, node2_idx * 6 + 5
            ])
            disp_elem = displacements_flat[dofs]
            disp_local = T @ disp_elem
            forces_local = k_local @ disp_local
            element_forces[elem_idx] = forces_local

        return element_forces

    def display_results(self, displacements: np.ndarray, reactions: np.ndarray):
        for node_idx in range(self.beam.nodes.shape[0]):
            node_id = int(self.beam.nodes[node_idx, 3])
            disp = np.round(displacements[node_idx], 5)
            rxn = np.round(reactions[node_idx], 5)
            print(f"Node {node_id} Displacements: [UX: {disp[0]}, UY: {disp[1]}, UZ: {disp[2]}, "
                  f"RX: {disp[3]}, RY: {disp[4]}, RZ: {disp[5]}]")
            print(f"Node {node_id} Reactions: [FX: {rxn[0]}, FY: {rxn[1]}, FZ: {rxn[2]}, "
                  f"MX: {rxn[3]}, MY: {rxn[4]}, MZ: {rxn[5]}]")
            print("---")

    def build_geometric_stiffness_matrix(self) -> np.ndarray:
        if self.internal_forces is None:
            raise ValueError("Must run solve() first to compute internal forces for buckling analysis.")

        n_nodes = self.beam.nodes.shape[0]
        n_dofs = n_nodes * 6
        K_geo_global = np.zeros((n_dofs, n_dofs))

        for elem_idx, (node1_id, node2_id) in enumerate(self.beam.elements):
            node1_idx = np.where(self.beam.nodes[:, 3] == node1_id)[0][0]
            node2_idx = np.where(self.beam.nodes[:, 3] == node2_id)[0][0]
            node1_coords = self.beam.nodes[node1_idx, :3]
            node2_coords = self.beam.nodes[node2_idx, :3]

            L = np.linalg.norm(node2_coords - node1_coords)
            Ip = self.beam.Iy + self.beam.Iz
            forces = self.internal_forces[elem_idx]
            Fx2, Fy1, Fz1, Mx1, My1, Mz1, _, Fy2, Fz2, Mx2, My2, Mz2 = forces

            k_geo = fu.local_geometric_stiffness_matrix_3D_beam(
                L, self.beam.A, Ip, Fx2, Mx2, My1, Mz1, My2, Mz2
            )
            gamma = rotation_matrix_3D(*node1_coords, *node2_coords)
            T = transformation_matrix_3D(gamma)
            k_geo_global = T.T @ k_geo @ T

            dofs = np.array([
                node1_idx * 6, node1_idx * 6 + 1, node1_idx * 6 + 2, node1_idx * 6 + 3, node1_idx * 6 + 4, node1_idx * 6 + 5,
                node2_idx * 6, node2_idx * 6 + 1, node2_idx * 6 + 2, node2_idx * 6 + 3, node2_idx * 6 + 4, node2_idx * 6 + 5
            ])
            for i in range(12):
                for j in range(12):
                    K_geo_global[dofs[i], dofs[j]] += k_geo_global[i, j]

        return K_geo_global

    # Added method for buckling forces computation
    def compute_buckling_mode_forces(self, mode_displacements: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute internal forces for a given buckling mode displacement vector."""
        displacements_flat = mode_displacements.flatten()
        element_forces = {}

        for elem_idx, (node1_id, node2_id) in enumerate(self.beam.elements):
            node1_idx = np.where(self.beam.nodes[:, 3] == node1_id)[0][0]
            node2_idx = np.where(self.beam.nodes[:, 3] == node2_id)[0][0]
            node1_coords = self.beam.nodes[node1_idx, :3]
            node2_coords = self.beam.nodes[node2_idx, :3]

            L = np.linalg.norm(node2_coords - node1_coords)
            k_local = fu.local_elastic_stiffness_matrix_3D_beam(
                self.beam.E, self.beam.nu, self.beam.A, L, self.beam.Iy, self.beam.Iz, self.beam.J
            )
            gamma = rotation_matrix_3D(*node1_coords, *node2_coords)
            T = transformation_matrix_3D(gamma)

            dofs = np.array([
                node1_idx * 6, node1_idx * 6 + 1, node1_idx * 6 + 2, node1_idx * 6 + 3, node1_idx * 6 + 4, node1_idx * 6 + 5,
                node2_idx * 6, node2_idx * 6 + 1, node2_idx * 6 + 2, node2_idx * 6 + 3, node2_idx * 6 + 4, node2_idx * 6 + 5
            ])
            disp_elem = displacements_flat[dofs]
            disp_local = T @ disp_elem
            forces_local = k_local @ disp_local
            element_forces[elem_idx] = forces_local

        return element_forces

    # Updated solve_buckling to return three values
    def solve_buckling(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[int, np.ndarray]]]:
        """Solve for buckling eigenvalues, eigenvectors, and corresponding internal forces."""
        if self.internal_forces is None:
            self.solve()

        K_elastic = self.build_stiffness_matrix()
        K_geo = self.build_geometric_stiffness_matrix()

        known_dofs = []
        for node_id, constraints in self.bc.fixed_nodes.items():
            node_idx = np.where(self.beam.nodes[:, 3] == node_id)[0][0]
            dofs = [node_idx * 6 + i for i in range(6)]
            for i, constraint in enumerate(constraints):
                if constraint is not None:
                    known_dofs.append(dofs[i])

        n_dofs = K_elastic.shape[0]
        unknown_dofs = [i for i in range(n_dofs) if i not in known_dofs]

        K_e_ff = K_elastic[np.ix_(unknown_dofs, unknown_dofs)]
        K_g_ff = K_geo[np.ix_(unknown_dofs, unknown_dofs)]

        eigvals, eigvecs = sp.eig(K_e_ff, -K_g_ff)  # Your original convention

        real_pos_mask = np.isreal(eigvals) & (eigvals > 0)
        filtered_eigvals = np.real(eigvals[real_pos_mask])
        filtered_eigvecs = eigvecs[:, real_pos_mask]

        sorted_inds = np.argsort(filtered_eigvals)
        filtered_eigvals = filtered_eigvals[sorted_inds]
        filtered_eigvecs = filtered_eigvecs[:, sorted_inds]

        # Compute buckling forces
        n_nodes = self.beam.nodes.shape[0]
        buckling_forces = {}
        full_eigvecs = np.zeros((n_dofs, filtered_eigvals.size))
        full_eigvecs[unknown_dofs, :] = filtered_eigvecs
        for mode_idx in range(min(3, filtered_eigvals.size)):  # Compute for first 3 modes
            mode_displacements = full_eigvecs[:, mode_idx].reshape(n_nodes, 6)
            buckling_forces[mode_idx] = self.compute_buckling_mode_forces(mode_displacements)

        self.buckling_forces = buckling_forces
        return filtered_eigvals, full_eigvecs, buckling_forces

    # Updated to display only first mode
    def display_buckling_results(self, eigvals: np.ndarray, eigvecs: np.ndarray, buckling_forces: Dict[int, Dict[int, np.ndarray]]):
        """Display buckling eigenvalue, mode shape, and internal forces for the first mode only."""
        print("\nBuckling Analysis Results (First Mode):")
        if len(eigvals) > 0:
            mode_idx = 0  # First mode only
            print(f"Mode 1: Critical Load Multiplier = {eigvals[mode_idx]:.5f}")
            print("Mode Shape (Displacements):")
            mode_disp = eigvecs[:, mode_idx].reshape(self.beam.nodes.shape[0], 6)
            for node_idx in range(self.beam.nodes.shape[0]):
                node_id = int(self.beam.nodes[node_idx, 3])
                disp = np.round(mode_disp[node_idx], 5)
                print(f"Node {node_id}: [UX: {disp[0]}, UY: {disp[1]}, UZ: {disp[2]}, RX: {disp[3]}, RY: {disp[4]}, RZ: {disp[5]}]")
            print("Internal Forces for Buckling Mode 1:")
            for elem_idx, forces in buckling_forces[mode_idx].items():
                forces = np.round(forces, 5)
                print(f"Element {elem_idx}: [Fx1: {forces[0]}, Fy1: {forces[1]}, Fz1: {forces[2]}, Mx1: {forces[3]}, My1: {forces[4]}, Mz1: {forces[5]}, "
                      f"Fx2: {forces[6]}, Fy2: {forces[7]}, Fz2: {forces[8]}, Mx2: {forces[9]}, My2: {forces[10]}, Mz2: {forces[11]}]")
        else:
            print("No buckling modes found.")
