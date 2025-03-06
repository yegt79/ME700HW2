import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Direct_Stiffness_Methodd import BeamComponent, BoundaryCondition, BeamSolver
# you can put the inputs here

# Define inputs
E = 500  # Young's Modulus (5800 Pa)
nu = 0.3  # Poisson's Ratio
r = 0.5  # Radius (0.5 m)
A = np.pi * r**2  # Cross-sectional area (pi/4 ≈ 0.7854 m^2)
Iy = np.pi * r**4 / 4  # Moment of inertia about y-axis (pi/64 ≈ 0.0491 m^4)
Iz = np.pi * r**4 / 4  # Moment of inertia about z-axis (pi/64 ≈ 0.0491 m^4)
J = np.pi * r**4 / 2  # Polar moment of inertia (pi/32 ≈ 0.0982 m^4)

nodes = np.array([
    [0.0, 0.0, 0.0, 0],    # N0: (0, 0, 0)
    [10.0, 0.0, 0.0, 1],   # N1: (10, 0, 0)
    [10.0, 20.0, 0.0, 2],   # N2: (10, 0, 0)
    [0.0, 20.0, 0.0, 3],   # N3: (0, 20, 0)
    [0.0, 0.0, 25.0, 4],   # N4: (0, 0, 35)
    [10.0, 0, 25.0, 5],  # N5: (10, 20, 0)
    [10.0, 20.0, 25.0, 6], # N6: (10, 20, 35)
    [0.0, 20.0, 25.0, 7],  # N7: (0, 20, 35)
])

elements = np.array([
    [0, 4],  # E0: N0-N4
    [1, 5],  # E1: N1-N5
    [2, 6],  # E2: N2-N6
    [3, 7],  # E3: N3-N7
    [4, 5],  # E4: N4-N5
    [5, 6],  # E5: N5-N6
    [6, 7],  # E6: N6-N7
    [4, 7],  # E7: N4-N7
])

fixed_nodes = {
    0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # N0: Fixed in all DOFs
    1: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # N1: Fixed in all DOFs
    2: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # N2: Fixed in all DOFs
    3: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # N3: Fixed in all DOFs
    4: (None, None, None, None, None, None),  # N4: Free
    5: (None, None, None, None, None, None),  # N5: Free
    6: (None, None, None, None, None, None),  # N6: Free
    7: (None, None, None, None, None, None),  # N7: Free
}

P = -1.0  # Applied load magnitude
loads = {
    4: (0.0, 0.0, -P, 0.0, 0.0, 0.0),  # N4: [0, 0, -P]
    5: (0.0, 0.0, -P, 0.0, 0.0, 0.0),  # N5: [0, 0, -P]
    6: (0.0, 0.0, -P, 0.0, 0.0, 0.0),  # N6: [0, 0, -P]
    7: (0.0, 0.0, -P, 0.0, 0.0, 0.0),  # N7: [0, 0, -P]
}


# --- Analysis and Plotting (minimized for display, but fully functional) ---
# Set up the beam and boundary conditions
beam = BeamComponent(nodes, elements, E, nu, A, Iy, Iz, J)
bc = BoundaryCondition(fixed_nodes)
for node_id, load in loads.items():
    bc.apply_load(node_id, load)

# Run the analysis
solver = BeamSolver(beam, bc)
displacements, reactions = solver.solve()
print("Static Analysis Results:")
solver.display_results(displacements, reactions)
print("Internal Forces per Element:")
for elem_idx, forces in solver.internal_forces.items():
    print(f"Element {elem_idx}: {np.round(forces, 5)}")

eigvals, eigvecs, buckling_forces = solver.solve_buckling()
print("\nBuckling Analysis Results:")
print("Critical Load Factors:", np.round(eigvals, 5))
print("First Buckling Mode Shape (raw):", np.round(eigvecs[:, 0], 5))

# Define plotting functions
def plot_internal_forces(beam, internal_forces):
    print("Plotting internal forces...")
    for elem_idx, forces in internal_forces.items():
        node1_id, node2_id = beam.elements[elem_idx]
        n1_idx = np.where(beam.nodes[:, 3] == node1_id)[0][0]
        n2_idx = np.where(beam.nodes[:, 3] == node2_id)[0][0]
        n1_loc = beam.nodes[n1_idx, :3]
        n2_loc = beam.nodes[n2_idx, :3]
        length = np.linalg.norm(n2_loc - n1_loc)
        x = [0, length]
        print(f"Element {elem_idx} forces: {forces}")
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        titles = ['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$']
        indices = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
        
        for i, (ax, title, idx_pair) in enumerate(zip(axs.flat, titles, indices)):
            y_values = [forces[idx_pair[0]], forces[idx_pair[1]]]
            ax.plot(x, y_values, 'b-o')
            ax.set_title(title)
            ax.set_xlabel('Length (m)')
            ax.set_ylabel('Force (N)' if i < 3 else 'Moment (N·m)')
            ax.grid(True)
            print(f"{title} values: {y_values}")
        
        fig.suptitle(f'Internal Forces and Moments - Element {elem_idx}')
        plt.tight_layout()
        plt.savefig(f'internal_forces_elem_{elem_idx}.png')
        plt.close()

def plot_deformed_structure(beam, displacements, scale=10):
    print("Plotting deformed structure...")
    print(f"Displacements: {displacements}")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for elem in beam.elements:
        n1_idx = np.where(beam.nodes[:, 3] == elem[0])[0][0]
        n2_idx = np.where(beam.nodes[:, 3] == elem[1])[0][0]
        x = [beam.nodes[n1_idx, 0], beam.nodes[n2_idx, 0]]
        y = [beam.nodes[n1_idx, 1], beam.nodes[n2_idx, 1]]
        z = [beam.nodes[n1_idx, 2], beam.nodes[n2_idx, 2]]
        ax.plot(x, y, z, 'k--', label='Original' if elem[0] == 0 else "")
        print(f"Original element {elem}: x={x}, y={y}, z={z}")
    
    deformed_nodes = beam.nodes[:, :3] + displacements[:, :3] * scale
    print(f"Deformed nodes: {deformed_nodes}")
    for elem in beam.elements:
        n1_idx = np.where(beam.nodes[:, 3] == elem[0])[0][0]
        n2_idx = np.where(beam.nodes[:, 3] == elem[1])[0][0]
        x = [deformed_nodes[n1_idx, 0], deformed_nodes[n2_idx, 0]]
        y = [deformed_nodes[n1_idx, 1], deformed_nodes[n2_idx, 1]]
        z = [deformed_nodes[n1_idx, 2], deformed_nodes[n2_idx, 2]]
        ax.plot(x, y, z, 'purple', label='Deformed' if elem[0] == 0 else "")
        print(f"Deformed element {elem}: x={x}, y={y}, z={z}")
    
    ax.scatter(beam.nodes[:, 0], beam.nodes[:, 1], beam.nodes[:, 2], c='blue', label='Nodes')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Frame Deformed Shape (Scale Factor: {scale})')
    ax.legend()
    plt.savefig('deformed_structure.png')
    plt.close()

def hermite_shape_funcs(eta):
    """Hermite shape functions for transverse displacement (v or w) in a beam element."""
    N1 = 1 - 3*eta**2 + 2*eta**3  # Displacement at node 1
    N2 = eta - 2*eta**2 + eta**3   # Rotation at node 1
    N3 = 3*eta**2 - 2*eta**3       # Displacement at node 2
    N4 = -eta**2 + eta**3          # Rotation at node 2
    return N1, N2, N3, N4

def plot_buckling_mode(beam, eigvec, mode_num=0, scale=10, points=50):
    print("Plotting buckling mode...")
    print(f"Input eigenvector: {eigvec}")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Map reduced eigenvector back to full DOF system
    full_eigvec = np.zeros(12)  # 2 nodes * 6 DOFs
    free_dofs = [6, 7, 8, 9, 10, 11]  # Node 1 DOFs (0-5 are fixed at node 0)
    full_eigvec[free_dofs] = eigvec
    mode_full = full_eigvec.reshape(beam.nodes.shape[0], 6)  # [u, v, w, rx, ry, rz] per node
    print(f"Full mode shape: {mode_full}")
    
    for elem_idx, elem in enumerate(beam.elements):
        n1_idx = np.where(beam.nodes[:, 3] == elem[0])[0][0]
        n2_idx = np.where(beam.nodes[:, 3] == elem[1])[0][0]
        n1_loc = beam.nodes[n1_idx, :3]
        n2_loc = beam.nodes[n2_idx, :3]
        L = np.linalg.norm(n2_loc - n1_loc)
        
        # Original geometry
        x_ref = [n1_loc[0], n2_loc[0]]
        y_ref = [n1_loc[1], n2_loc[1]]
        z_ref = [n1_loc[2], n2_loc[2]]
        ax.plot(x_ref, y_ref, z_ref, 'k--', label='Original' if elem_idx == 0 else "")
        print(f"Original element {elem_idx}: x={x_ref}, y={y_ref}, z={z_ref}")
        
        # Hermite interpolation for buckling mode
        t = np.linspace(0, 1, points)
        x_buckled = np.linspace(n1_loc[0], n2_loc[0], points)
        y_buckled = np.linspace(n1_loc[1], n2_loc[1], points)
        z_buckled = np.linspace(n1_loc[2], n2_loc[2], points)
        
        # Extract displacements and rotations
        u1, v1, w1, rx1, ry1, rz1 = mode_full[n1_idx]
        u2, v2, w2, rx2, ry2, rz2 = mode_full[n2_idx]
        
        # Hermite interpolation for v (y-displacement) and w (z-displacement)
        v_vals = []
        w_vals = []
        for eta in t:
            N1, N2, N3, N4 = hermite_shape_funcs(eta)
            v = N1*v1 + N2*L*rz1 + N3*v2 + N4*L*rz2  # y-displacement
            w = N1*w1 + N2*L*ry1 + N3*w2 + N4*L*ry2  # z-displacement (bending about y)
            v_vals.append(v)
            w_vals.append(w)
        
        # Apply scaled displacements
        for i in range(points):
            y_buckled[i] += scale * v_vals[i]
            z_buckled[i] += scale * w_vals[i]
        
        ax.plot(x_buckled, y_buckled, z_buckled, 'purple', label=f'Mode {mode_num+1}' if elem_idx == 0 else "")
        print(f"Buckled element {elem_idx}: x={x_buckled}, y={y_buckled}, z={z_buckled}")
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Frame Buckling Mode {mode_num+1} (λ = {eigvals[mode_num]:.2f}, Scale = {scale})')
    ax.legend()
    plt.savefig(f'buckling_mode_{mode_num}.png')
    plt.close()

# Generate plots
print("Starting plot generation...")
plot_internal_forces(beam, solver.internal_forces)
plot_deformed_structure(beam, displacements, scale=10)
plot_buckling_mode(beam, eigvecs[:, 0], mode_num=0, scale=10, points=50)
print("Plot generation complete. Plots saved as PNG files.")


