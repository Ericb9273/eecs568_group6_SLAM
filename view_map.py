#!/usr/bin/env python3
"""
Point Cloud Map Viewer
=======================
Visualize the SLAM output: global_map.pcd + trajectory.txt

Controls (Open3D viewer):
  Left click + drag:  Rotate
  Scroll:             Zoom
  Middle click + drag: Pan
  R:                  Reset view
  Q / Esc:            Quit

Usage:
  python view_map.py                              # default slam_output/
  python view_map.py --dir slam_output
  python view_map.py --pcd global_map.pcd         # just the map
  python view_map.py --top                        # bird's-eye view
"""

import argparse
import numpy as np
import open3d as o3d
import os

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_map(pcd_path):
    print(f"Loading map: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"  {len(pcd.points)} points")
    return pcd


def color_by_height(pcd):
    """Color points by Z value using viridis colormap."""
    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min if z_max - z_min > 0.01 else 1.0
    z_norm = (z - z_min) / z_range

    if HAS_MPL:
        colors = plt.cm.viridis(z_norm)[:, :3]
    else:
        # Simple blue-to-red gradient
        colors = np.zeros((len(z_norm), 3))
        colors[:, 0] = z_norm       # red increases with height
        colors[:, 2] = 1 - z_norm   # blue decreases with height

    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"  Z range: [{z_min:.2f}, {z_max:.2f}] m")
    return pcd


def load_trajectory(traj_path):
    """Load trajectory and create a red line set."""
    print(f"Loading trajectory: {traj_path}")
    data = np.loadtxt(traj_path)
    positions = data[:, 1:4]
    print(f"  {len(positions)} poses")

    lines = [[i, i+1] for i in range(len(positions) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))

    # Start and end markers as small spheres
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    start_sphere.translate(positions[0])
    start_sphere.paint_uniform_color([0, 0.8, 0])  # green

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    end_sphere.translate(positions[-1])
    end_sphere.paint_uniform_color([0.8, 0, 0])  # red

    return line_set, start_sphere, end_sphere, positions


def create_ground_grid(center, size=50, spacing=5):
    """Create a flat grid on the ground plane for reference."""
    lines = []
    points = []
    idx = 0

    cx, cy = center[0], center[1]
    half = size / 2

    # Lines along X
    for y in np.arange(cy - half, cy + half + spacing, spacing):
        points.append([cx - half, y, 0])
        points.append([cx + half, y, 0])
        lines.append([idx, idx + 1])
        idx += 2

    # Lines along Y
    for x in np.arange(cx - half, cx + half + spacing, spacing):
        points.append([x, cy - half, 0])
        points.append([x, cy + half, 0])
        lines.append([idx, idx + 1])
        idx += 2

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(lines))
    return grid


def main():
    parser = argparse.ArgumentParser(description="View SLAM point cloud map")
    parser.add_argument("--dir", default="slam_output",
                        help="SLAM output directory")
    parser.add_argument("--pcd", default=None,
                        help="Direct path to .pcd file (overrides --dir)")
    parser.add_argument("--traj", default=None,
                        help="Direct path to trajectory.txt")
    parser.add_argument("--no-traj", action="store_true",
                        help="Don't show trajectory")
    parser.add_argument("--no-grid", action="store_true",
                        help="Don't show ground grid")
    parser.add_argument("--top", action="store_true",
                        help="Start with bird's-eye (top-down) view")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Point size (default: 2.0)")
    args = parser.parse_args()

    # Resolve paths
    pcd_path = args.pcd or os.path.join(args.dir, "global_map.pcd")
    traj_path = args.traj or os.path.join(args.dir, "trajectory.txt")

    if not os.path.exists(pcd_path):
        print(f"ERROR: {pcd_path} not found")
        return

    # Load map
    pcd = load_map(pcd_path)
    pcd = color_by_height(pcd)
    geometries = [pcd]

    # Load trajectory
    traj_center = np.array([0, 0, 0])
    if not args.no_traj and os.path.exists(traj_path):
        line_set, start_s, end_s, positions = load_trajectory(traj_path)
        geometries.extend([line_set, start_s, end_s])
        traj_center = positions.mean(axis=0)

    # Ground grid
    if not args.no_grid:
        grid = create_ground_grid(traj_center)
        geometries.append(grid)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SLAM Map Viewer", width=1400, height=900)

    for g in geometries:
        vis.add_geometry(g)

    # Render options
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])  # dark background
    opt.show_coordinate_frame = True

    # Camera setup
    if args.top:
        ctrl = vis.get_view_control()
        ctrl.set_front([0, 0, 1])    # looking down
        ctrl.set_up([0, 1, 0])       # Y is up in screen
        ctrl.set_lookat(traj_center)
        ctrl.set_zoom(0.3)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
