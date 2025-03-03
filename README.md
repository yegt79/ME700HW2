# ME700HW2

This project implements a 3D beam buckling analysis using the Direct Stiffness Method (DSM) and critical load solver in Python. It calculates critical load factors and buckling mode shapes for a beam structure under compressive loads, based on formulations from Matrix Structural Analysis by McGuire et al. (2nd Edition).

Files
1. functions.py
Purpose: Contains core matrix computation functions for 3D beam elements.
Key Functions:
local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J): Computes the 12x12 local elastic stiffness matrix for a 3D beam (axial, torsion, and bending terms), per McGuire p. 73.
rotation_matrix_3D(x1, y1, z1, x2, y2, z2, v_temp=None): Calculates the 3x3 rotation matrix from local to global coordinates, handling vertical beams with a reference vector (v_temp), per McGuire Ch. 5.1.
transformation_matrix_3D(gamma): Expands a 3x3 rotation matrix into a 12x12 transformation matrix for 3D beam DOFs, per McGuire Ch. 5.1.
local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2): Computes the 12x12 geometric stiffness matrix with interaction terms, per McGuire p. 258.
local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(L, A, I_rho, Fx2): Simplified geometric stiffness matrix without moment interactions, per McGuire p. 257.
Dependencies: numpy.
2. Direct_Stiffness_Method.py (DSM)
Purpose: Implements the Direct Stiffness Method for static analysis of 3D beam structures.
Key Functions:
create_node(x, y, z, node_id, bc=None): Creates a node dictionary with coordinates, boundary conditions, and zeroed displacement/reaction vectors.
create_element(node1, node2, E, nu, A, Iy, Iz, J): Constructs an element dictionary with material properties, length, rotation matrix, and stiffness matrices (local and global).
assemble_global_stiffness_matrix(nodes, elements): Assembles the global elastic stiffness matrix from element global stiffness matrices.
assemble_load_vector(nodes, loads): Builds the global load vector from applied loads.
apply_boundary_conditions(nodes, global_stiffness_matrix, load_vector): Reduces the system by applying boundary conditions.
calculate_structure_response(nodes, elements, loads): Solves for displacements and reactions, updating node dictionaries.
Dependencies: numpy, math, functions.py.
3. Elastic_Critical_Solver.py (ECS)
Purpose: Computes critical load factors and buckling mode shapes for a 3D beam structure under compressive loads.
Key Class: CriticalLoadSolver(nodes, elements, loads, include_interactions=False):
__init__: Initializes the solver with nodes, elements, loads, and a flag for geometric stiffness interaction terms.
compute_internal_forces_and_moments(displacements): Calculates local forces/moments per element from global displacements.
assemble_geometric_stiffness(displacements): Assembles the global geometric stiffness matrix using functions.py routines.
compute_eigenvalues(): Solves the eigenvalue problem (K_elastic vs. K_geometric) for critical load factors and mode shapes.
plot_mode_shape(mode_vec, scale_factor): Visualizes the buckling mode in 3D.
Dependencies: numpy, scipy.linalg, matplotlib, Direct_Stiffness_Method.py, functions.py.
4. example1.py
Purpose: Demonstrates the usage of the above modules with a simple vertical beam buckling problem.
Structure:
Defines two nodes (fixed base at (0,0,0), free top at (0,0,5)).
Creates one element using dsm.create_element with steel properties (E = 200 GPa, A = 0.01 m², etc.).
Applies a 1000 N compressive load downward at the top node.
Runs CriticalLoadSolver to compute critical load factors and plot the first buckling mode.
Dependencies: numpy, Direct_Stiffness_Method.py, functions.py, Elastic_Critical_Solver.py.
Prerequisites
Python 3.x
Libraries: numpy, scipy, matplotlib
Conda environment (optional): Configure as needed (e.g., env2 in the example).
Usage
Setup:
Place all files (functions.py, Direct_Stiffness_Method.py, Elastic_Critical_Solver.py, example1.py) in the same directory (e.g., /usr4/me700/yegt/ME700HW2-8/src/).
Ensure your Python environment has the required libraries installed.
Run the Example:
bash

Collapse

Wrap

Copy
/projectnb/me700/students/yegt/.conda/envs/env2/bin/python /usr4/me700/yegt/ME700HW2-8/src/example1.py
Outputs critical load factors (e.g., ~16.2 for the first mode) and displays a 3D plot of the buckling mode.
Customization:
Modify example1.py to change:
Node coordinates or boundary conditions ('bc').
Material/section properties (E, A, Iy, Iz, J, nu).
Load magnitudes/directions.
Toggle include_interactions in CriticalLoadSolver for full geometric stiffness effects.
Example Output
For the default example1.py (5m steel column, fixed base, 1000 N compressive load):

Critical Load Factors: Approximately [16.2, ...] (first factor ≈ Euler buckling load / applied load).
Plot: Shows lateral deflection of the beam in the first buckling mode.
Theoretical Basis
Euler Buckling: For a fixed-free column, P_cr = (π² * E * I) / (L_eff²), where L_eff = 0.7L. With I = 0.0001 m⁴, E = 200e9 Pa, L = 5m, expect P_cr ≈ 16,209 N.
Geometric Stiffness: Accounts for load effects on stability, critical for buckling analysis.
Limitations
Assumes linear elastic behavior and small deformations pre-buckling.
Current example is a single element; multi-element structures require additional nodes/elements in example1.py.
local_stiffness_matrix in ECS must match DSM’s output (12x12, 6 DOF per node).
Contributing
Feel free to extend this for more complex structures, improve error handling, or validate against experimental data!
