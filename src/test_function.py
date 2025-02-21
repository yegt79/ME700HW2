import numpy as np
import pytest
import functions as fu
from beam_element import BeamElement

# Test 1: Testing BeamElement class initialization
def test_beam_element_initialization():
    E = 200e9  # Young's Modulus in Pascals
    nu = 0.3  # Poisson's ratio
    A = 0.01  # Cross-sectional area in m^2
    Iy = 1e-6  # Moment of inertia about y-axis in m^4
    Iz = 1e-6  # Moment of inertia about z-axis in m^4
    J = 1e-6  # Polar moment of inertia in m^4
    nodes = 2  # Number of nodes for the beam element

    beam = BeamElement(E, nu, A, Iy, Iz, J, nodes)

    # Check if object initialization is correct
    assert beam.E == E
    assert beam.nu == nu
    assert beam.A == A
    assert beam.Iy == Iy
    assert beam.Iz == Iz
    assert beam.J == J
    assert beam.nodes == nodes

# Test 2: Testing length calculation for BeamElement
def test_beam_element_length():
    node_positions = [(0, 0, 0), (3, 4, 0)]  # Coordinates of the two nodes
    E = 200e9  # Young's Modulus in Pascals
    nu = 0.3  # Poisson's ratio
    A = 0.01  # Cross-sectional area in m^2
    Iy = 1e-6  # Moment of inertia about y-axis in m^4
    Iz = 1e-6  # Moment of inertia about z-axis in m^4
    J = 1e-6  # Polar moment of inertia in m^4
    nodes = 2  # Number of nodes for the beam element

    beam = BeamElement(E, nu, A, Iy, Iz, J, nodes)
    length = beam.length(node_positions)

    # The length should be 5 (3-4-5 triangle)
    assert np.isclose(length, 5.0)

# Test 3: Testing stiffness matrix for BeamElement
def test_beam_element_stiffness_matrix():
    node_positions = [(0, 0, 0), (3, 0, 0)]  # Simple 3D beam between two nodes
    E = 200e9  # Young's Modulus in Pascals
    nu = 0.3  # Poisson's ratio
    A = 0.01  # Cross-sectional area in m^2
    Iy = 1e-6  # Moment of inertia about y-axis in m^4
    Iz = 1e-6  # Moment of inertia about z-axis in m^4
    J = 1e-6  # Polar moment of inertia in m^4
    nodes = 2  # Number of nodes for the beam element

    beam = BeamElement(E, nu, A, Iy, Iz, J, nodes)
    stiffness_matrix = beam.stiffness_matrix(node_positions)

    # The stiffness matrix should be 12x12 (6 DOFs per node)
    assert stiffness_matrix.shape == (12, 12)

# Test 4: Testing local stiffness matrix function
def test_local_stiffness_matrix():
    E = 200e9  # Young's Modulus in Pascals
    nu = 0.3  # Poisson's ratio
    A = 0.01  # Cross-sectional area in m^2
    L = 5  # Length of the beam in meters
    Iy = 1e-6  # Moment of inertia about y-axis in m^4
    Iz = 1e-6  # Moment of inertia about z-axis in m^4
    J = 1e-6  # Polar moment of inertia in m^4

    # Test the function that calculates the local stiffness matrix
    k_local = fu.local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)

    # The local stiffness matrix should be 12x12
    assert k_local.shape == (12, 12)

# Test 5: Check transformation matrix function
def test_transformation_matrix():
    # Define two nodes in 3D space
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = 3, 4, 0  # Simple 3D line from (0, 0, 0) to (3, 4, 0)

    # Compute the 3D rotation matrix
    gamma = fu.rotation_matrix_3D(x1, y1, z1, x2, y2, z2)

    # The rotation matrix should be 3x3
    assert gamma.shape == (3, 3)

# Test 6: Test geometric stiffness matrix calculation
def test_local_geometric_stiffness_matrix():
    L = 5  # Length of the beam in meters
    A = 0.01  # Cross-sectional area in m^2
    I_rho = 1e-6  # Polar moment of inertia
    Fx2 = 1000  # Force applied at node 2
    Mx2 = 100  # Moment applied at node 2
    My1 = 50  # Moment applied at node 1
    Mz1 = 50  # Moment applied at node 1
    My2 = 50  # Moment applied at node 2
    Mz2 = 50  # Moment applied at node 2

    # Test the function that calculates the geometric stiffness matrix
    k_g = fu.local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)

    # The geometric stiffness matrix should be 12x12
    assert k_g.shape == (12, 12)

# Test 7: Test the geometric stiffness matrix without interaction terms
def test_local_geometric_stiffness_matrix_without_interaction():
    L = 5  # Length of the beam in meters
    A = 0.01  # Cross-sectional area in m^2
    I_rho = 1e-6  # Polar moment of inertia
    Fx2 = 1000  # Force applied at node 2

    # Test the function that calculates the geometric stiffness matrix without interaction terms
    k_g_without_interaction = fu.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(L, A, I_rho, Fx2)

    # The geometric stiffness matrix should be 12x12
    assert k_g_without_interaction.shape == (12, 12)

if __name__ == "__main__":
    pytest.main()
