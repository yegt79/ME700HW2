import math
import numpy as np
import functions as fu  # Ensure 'functions.py' has correct stiffness matrix calculations
from functions import rotation_matrix_3D, transformation_matrix_3D

def check_unit_vector(v: np.ndarray):
    """Ensure vector is a unit vector."""
    if np.isclose(np.linalg.norm(v), 1.0):
        return True
    else:
        raise ValueError(f"Input vector is not a unit vector. Norm = {np.linalg.norm(v)}.")

def check_parallel(v1: np.ndarray, v2: np.ndarray):
    """Ensure vectors are not parallel."""
    if np.isclose(np.dot(v1, v2), 1.0) or np.isclose(np.dot(v1, v2), -1.0):
        raise ValueError("Vectors are parallel.")

def create_node(x, y, z, node_id, bc=None):
    """Create a node with displacement and reaction initialized."""
    return {
        'x': x,
        'y': y,
        'z': z,
        'node_id': node_id,
        'bc': bc if bc else [False, False, False, False, False, False],  # UX, UY, UZ, RX, RY, RZ
        'displacement': np.zeros(6),
        'reaction': np.zeros(6)
    }

def create_element(node1: dict, node2: dict, E: float, nu: float, A: float, Iy: float, Iz: float, J: float, v_temp: np.ndarray = None) -> dict:
    """Create an element and compute relevant properties.

    Args:
        node1 (dict): First node of the element.
        node2 (dict): Second node of the element.
        E (float): Young's Modulus.
        nu (float): Poisson's Ratio.
        A (float): Cross-sectional area.
        Iy (float): Moment of inertia about y-axis.
        Iz (float): Moment of inertia about z-axis.
        J (float): Polar moment of inertia.
        v_temp (np.ndarray, optional): Temporary reference vector for rotation matrix. Defaults to None.

    Returns:
        dict: Element properties including stiffness matrices.
    """
    L = calculate_length(node1, node2)
    rotation_matrix = compute_rotation_matrix(node1, node2, v_temp=v_temp)  # Pass v_temp here
    local_stiffness_matrix = compute_local_stiffness_matrix(E, nu, A, L, Iy, Iz, J)
    global_stiffness_matrix = transform_to_global(local_stiffness_matrix, rotation_matrix)
    
    return {
        'node1': node1,
        'node2': node2,
        'E': E,
        'nu': nu,
        'A': A,
        'Iy': Iy,
        'Iz': Iz,
        'J': J,
        'L': L,
        'rotation_matrix': rotation_matrix,
        'local_stiffness_matrix': local_stiffness_matrix,
        'global_stiffness_matrix': global_stiffness_matrix
    }

def calculate_length(node1, node2):
    """Calculate the Euclidean distance between the two nodes."""
    x1, y1, z1 = node1['x'], node1['y'], node1['z']
    x2, y2, z2 = node2['x'], node2['y'], node2['z']
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def compute_rotation_matrix(node1: dict, node2: dict, v_temp: np.ndarray = None) -> np.ndarray:
    """
    Compute the 3D rotation matrix for a beam element based on node coordinates.
    (Based on Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition)

    Args:
        node1 (dict): First node with keys 'x', 'y', 'z'.
        node2 (dict): Second node with keys 'x', 'y', 'z'.
        v_temp (np.ndarray, optional): Temporary reference vector for local coordinate system.
                                       Defaults to None, triggering automatic selection.

    Returns:
        np.ndarray: 3x3 rotation matrix aligning local element axes with global axes.

    Raises:
        TypeError: If node coordinates are not numerical.
        ValueError: If element length is zero, v_temp is not a unit vector, or vectors are parallel.
    """
    # Extract coordinates and validate type
    try:
        x1, y1, z1 = node1['x'], node1['y'], node1['z']
        x2, y2, z2 = node2['x'], node2['y'], node2['z']
    except KeyError:
        raise KeyError("Nodes must have 'x', 'y', and 'z' keys.")
    
    if not all(isinstance(coord, (int, float)) for coord in [x1, y1, z1, x2, y2, z2]):
        raise TypeError("Node coordinates must be numerical values.")

    # Calculate length and check for zero length
    L = calculate_length(node1, node2)
    if np.isclose(L, 0.0):
        raise ValueError("Element length cannot be zero.")

    # Compute local_x (element direction)
    local_x = np.array([(x2 - x1) / L, (y2 - y1) / L, (z2 - z1) / L])

    # Define or validate v_temp
    if v_temp is None:
        if np.isclose(local_x[0], 0.0) and np.isclose(local_x[1], 0.0):
            v_temp = np.array([0.0, 1.0, 0.0])  # Use Y-axis like your friend's code
        else:
            v_temp = np.array([0.0, 0.0, 1.0])  # Use Z-axis
    else:
        # Reuse your existing helper functions (assuming they're in your code)
        if not np.isclose(np.linalg.norm(v_temp), 1.0):
            raise ValueError(f"Input v_temp is not a unit vector. Norm = {np.linalg.norm(v_temp)}.")
        if np.isclose(np.abs(np.dot(local_x, v_temp)), 1.0):
            raise ValueError("v_temp and local_x are parallel.")

    # Compute local_y and local_z
    local_y = np.cross(v_temp, local_x)
    local_y = local_y / np.linalg.norm(local_y)  # Normalize explicitly

    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)  # Normalize explicitly

    # Assemble rotation matrix
    gamma = np.vstack((local_x, local_y, local_z))

    # Validate shape
    if gamma.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    #print(f"Element from {node1['node_id']} to {node2['node_id']}: Rotation Matrix:\n{gamma}")
    return gamma

def compute_local_stiffness_matrix(E, nu, A, L, Iy, Iz, J):
    """Compute the local element stiffness matrix."""
    return fu.local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)

def transform_to_global(local_stiffness_matrix, rotation_matrix):
    """Transform local stiffness matrix to the global coordinate system."""
    Gamma = transformation_matrix_3D(rotation_matrix)
    return Gamma.T @ local_stiffness_matrix @ Gamma

def assemble_global_stiffness_matrix(nodes, elements):
    """Assembles the global stiffness matrix with correct DOF mapping."""
    ndof = len(nodes) * 6  # Total DOFs
    K = np.zeros((ndof, ndof))  

    node_index_map = {node['node_id']: i for i, node in enumerate(nodes)}

    for element in elements:
        k_global = element['global_stiffness_matrix']
        node1_id = element['node1']['node_id']
        node2_id = element['node2']['node_id']

        idx1 = node_index_map[node1_id] * 6
        idx2 = node_index_map[node2_id] * 6

        dof_indices = np.array([idx1, idx1+1, idx1+2, idx1+3, idx1+4, idx1+5,
                                idx2, idx2+1, idx2+2, idx2+3, idx2+4, idx2+5])
        
        for i in range(12):
            for j in range(12):
                K[dof_indices[i], dof_indices[j]] += k_global[i, j]

    return K

def assemble_load_vector(nodes, loads):
    """Assembles the global load vector ensuring correct placement."""
    ndof = len(nodes) * 6
    F = np.zeros(ndof)

    node_index_map = {node['node_id']: i for i, node in enumerate(nodes)}

    for node_id, load in loads.items():
        idx = node_index_map[node_id] * 6
        F[idx:idx+6] += load

    return F

def apply_boundary_conditions(nodes, global_stiffness_matrix, load_vector):
    """Applies boundary conditions and extracts the reduced system."""
    fixed_dof = []
    node_index_map = {node['node_id']: i for i, node in enumerate(nodes)}

    for node in nodes:
        idx = node_index_map[node['node_id']] * 6
        for i, is_fixed in enumerate(node['bc']):
            if is_fixed:
                fixed_dof.append(idx + i)

    fixed_dof = np.array(fixed_dof)
    all_dof = np.arange(global_stiffness_matrix.shape[0])
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    K_reduced = global_stiffness_matrix[np.ix_(free_dof, free_dof)]
    F_reduced = load_vector[free_dof]

    return K_reduced, F_reduced, free_dof, fixed_dof

def solve_system(global_stiffness_matrix, load_vector, K_reduced, F_reduced, free_dof):
    """Solves the system for displacements and reactions."""
    d = np.zeros(global_stiffness_matrix.shape[0])

    try:
        d_free = np.linalg.solve(K_reduced, F_reduced)
        d[free_dof] = d_free
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Error solving system: {e}")

    reactions = global_stiffness_matrix @ d - load_vector

    return d, reactions

def calculate_structure_response(nodes, elements, loads):
    """Creates a structure and solves it."""
    global_stiffness_matrix = assemble_global_stiffness_matrix(nodes, elements)
    load_vector = assemble_load_vector(nodes, loads)
    K_reduced, F_reduced, free_dof, fixed_dof = apply_boundary_conditions(nodes, global_stiffness_matrix, load_vector)
    displacements, reactions = solve_system(global_stiffness_matrix, load_vector, K_reduced, F_reduced, free_dof)

    # Assign displacements and reactions to nodes
    node_index_map = {node['node_id']: i for i, node in enumerate(nodes)}
    for i, node in enumerate(nodes):
        idx = i * 6
        node['displacement'] = displacements[idx:idx+6]
        node['reaction'] = reactions[idx:idx+6]

    return displacements, reactions
