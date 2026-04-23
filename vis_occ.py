import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (or load NPY file)
occ_grid = np.load('data/firewoods_sand_dark/voxel_grid.npy')
Nx, Ny, Nz = occ_grid.shape

# Set the specified xy plane boundary region to 1 as well (using vectorized operations)
x_center = Nx // 2 + 0.04 * Nx
y_center = Ny // 2
x_half_width = int(0.06 * Nx)
y_half_width = int(0.25 * Ny)
z_center = 160
z_half_thickness = int(0.02 * Nz)

# Create coordinate grid
i_indices, j_indices, k_indices = np.ogrid[:Nx, :Ny, :Nz]

# Create masks
x_mask = (i_indices >= x_center - x_half_width) & (i_indices <= x_center + x_half_width)
y_mask = (j_indices >= y_center - y_half_width) & (j_indices <= y_center + y_half_width)
z_mask = (k_indices >= z_center - z_half_thickness) & (k_indices <= z_center + z_half_thickness)

# Combine masks and set the region to 1
boundary_mask = x_mask & y_mask & z_mask
occ_grid[boundary_mask] = 1

x, y, z = np.where(occ_grid)

# Create figure with 3D view and three orthogonal projections
fig = plt.figure(figsize=(16, 12))

# 1. 3D view (upper left)
ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
ax_3d.scatter(x, y, z, c='blue', alpha=0.2, s=1)
ax_3d.set_xlim(0, Nx)
ax_3d.set_ylim(0, Ny)
ax_3d.set_zlim(0, Nz)
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('3D View')

# 2. XY plane projection (upper right) - top-down view (projected onto Z axis)
ax_xy = fig.add_subplot(2, 2, 2)
ax_xy.scatter(x, y, c='blue', alpha=0.1, s=0.5)
ax_xy.set_xlim(0, Nx)
ax_xy.set_ylim(0, Ny)
ax_xy.set_xlabel('X')
ax_xy.set_ylabel('Y')
ax_xy.set_title('XY Projection (Top View)')
ax_xy.set_aspect('equal')
ax_xy.grid(True, alpha=0.3)

# 3. XZ plane projection (lower left) - side view (projected onto Y axis)
ax_xz = fig.add_subplot(2, 2, 3)
ax_xz.scatter(x, z, c='blue', alpha=0.1, s=0.5)
ax_xz.set_xlim(0, Nx)
ax_xz.set_ylim(0, Nz)
ax_xz.set_xlabel('X')
ax_xz.set_ylabel('Z')
ax_xz.set_title('XZ Projection (Side View)')
ax_xz.set_aspect('equal')
ax_xz.grid(True, alpha=0.3)

# 4. YZ plane projection (lower right) - front view (projected onto X axis)
ax_yz = fig.add_subplot(2, 2, 4)
ax_yz.scatter(y, z, c='blue', alpha=0.1, s=0.5)
ax_yz.set_xlim(0, Ny)
ax_yz.set_ylim(0, Nz)
ax_yz.set_xlabel('Y')
ax_yz.set_ylabel('Z')
ax_yz.set_title('YZ Projection (Front View)')
ax_yz.set_aspect('equal')
ax_yz.grid(True, alpha=0.3)

plt.suptitle('Occupancy Grid - 3D View and Three Orthogonal Projections', fontsize=14)
plt.tight_layout()
plt.savefig('occupancy_grid.png', dpi=300, bbox_inches='tight')
# plt.show()