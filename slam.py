#!/usr/bin/env python3
"""
LiDAR-Odometry SLAM with Loop Closure
=======================================
Dataset: SubSurfaceGeoRobo - 80m traverse
Sensors: Velodyne 3D LiDAR (~10 Hz) + Wheel Encoders (~50 Hz)

Architecture:
  1. Wheel odometry provides motion prior (velocity + heading)
  2. Scan-to-submap point-to-plane ICP refines each pose
  3. Keyframes stored for loop closure detection
  4. When robot revisits a location, ICP against old keyframe
  5. Loop closure correction distributed along trajectory

No IMU — accelerometer is useless for position (quadratic drift),
and gyro heading doesn't match odom heading (different reference frames).
Odom is wheel-encoder based: 2D, smooth velocity, reliable heading.

Usage:
  python slam.py --bag /path/to/03_80m_other_sensor.bag --max-scans 500
  python slam.py --bag /path/to/03_80m_other_sensor.bag --view-map
"""

import argparse
import numpy as np
from collections import deque
import time
import os
import copy

import rosbag
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import (quaternion_matrix, quaternion_from_matrix,
                                 euler_from_quaternion)

import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d


###############################################################################
# 1. MATH
###############################################################################

def quat_to_rot(q):
    """[x,y,z,w] -> 3x3"""
    return quaternion_matrix(q)[:3, :3]

def rot_to_quat(R):
    """3x3 -> [x,y,z,w]"""
    mat = np.eye(4); mat[:3, :3] = R
    return quaternion_from_matrix(mat)

def yaw_to_rot(yaw):
    """Yaw angle -> 3x3 rotation about Z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rot_to_yaw(R):
    return np.arctan2(R[1, 0], R[0, 0])

def pose_to_T(x, y, z, yaw):
    """Build 4x4 from position + yaw."""
    T = np.eye(4)
    T[:3, :3] = yaw_to_rot(yaw)
    T[:3, 3] = [x, y, z]
    return T


###############################################################################
# 2. LIDAR -> BASE TRANSFORM (Translation Only)
###############################################################################

def get_calibrated_lidar_to_base():
    """
    Applies the translation offset from 001_Approximate_value.txt, 
    but assumes the point clouds are ALREADY aligned to gravity (Z-up).
    We use an Identity matrix for rotation to prevent double-flipping the axes.
    """
    T = np.eye(4)
    
    # Translation from the calibration file
    T[0, 3] = 0.565
    T[1, 3] = -0.295
    T[2, 3] = 0.4585
    
    print(f"[TF] Using flattened calibration LiDAR->base (Translation Only):\n{T}")
    return T


###############################################################################
# 3. DATA EXTRACTION
###############################################################################

def extract_data(bag_path, max_lidar_scans=None):
    print(f"[DATA] Reading {bag_path}...")
    bag = rosbag.Bag(bag_path)

    lidar_data = []  # (t, Nx3 points)
    odom_data = []   # (t, x, y, z, yaw, vx, wz)
    lidar_count = 0

    for topic, msg, t in bag.read_messages(
        topics=['/robot/front_laser/points',
                '/robot/robotnik_base_control/odom']
    ):
        stamp = msg.header.stamp.to_sec()

        if topic == '/robot/front_laser/points':
            if max_lidar_scans and lidar_count >= max_lidar_scans:
                continue
            points = np.array(
                [[p[0], p[1], p[2]] for p in
                 pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)],
                dtype=np.float64)
            if len(points) > 100:
                lidar_data.append((stamp, points))
                lidar_count += 1
                if lidar_count % 500 == 0:
                    print(f"  LiDAR: {lidar_count}")

        elif topic == '/robot/robotnik_base_control/odom':
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            q = [ori.x, ori.y, ori.z, ori.w]
            _, _, yaw = euler_from_quaternion(q)
            
            # --- FALLBACK: LEFT-HANDED ODOMETRY FIX ---
            # If flattening the TF matrix didn't fix the mirror issue, 
            # uncomment the line below to invert the odometry turning direction.
            # yaw = -yaw 
            
            vx = msg.twist.twist.linear.x
            wz = msg.twist.twist.angular.z
            odom_data.append((stamp, pos.x, pos.y, pos.z, yaw, vx, wz))

    bag.close()
    lidar_data.sort(key=lambda x: x[0])
    odom_data.sort(key=lambda x: x[0])

    print(f"  LiDAR: {len(lidar_data)} scans")
    print(f"  Odom:  {len(odom_data)} samples")
    return lidar_data, odom_data


###############################################################################
# 4. ODOM INTERPOLATOR
###############################################################################

class OdomInterpolator:
    def __init__(self, odom_data):
        t = np.array([d[0] for d in odom_data])
        x = np.array([d[1] for d in odom_data])
        y = np.array([d[2] for d in odom_data])
        z = np.array([d[3] for d in odom_data])
        yaw = np.array([d[4] for d in odom_data])
        vx = np.array([d[5] for d in odom_data])

        # Unwrap yaw for interpolation (avoid jumps at ±pi)
        yaw_unwrapped = np.unwrap(yaw)

        self.f_x = interp1d(t, x, fill_value='extrapolate')
        self.f_y = interp1d(t, y, fill_value='extrapolate')
        self.f_z = interp1d(t, z, fill_value='extrapolate')
        self.f_yaw = interp1d(t, yaw_unwrapped, fill_value='extrapolate')
        self.f_vx = interp1d(t, vx, fill_value='extrapolate')

        # Store origin for alignment
        self.x0 = x[0]
        self.y0 = y[0]
        self.z0 = z[0]
        self.yaw0 = yaw_unwrapped[0]

    def get_pose(self, t):
        """
        Get odom pose at time t, transformed to start-at-origin frame.
        Both position AND heading are rotated so that the robot starts
        at (0,0,0) facing +X. Without the position rotation, the XY
        coordinates stay in the original odom frame while the heading
        is zeroed — every relative transform gets rotated by yaw0.
        """
        dx = float(self.f_x(t)) - self.x0
        dy = float(self.f_y(t)) - self.y0
        dz = float(self.f_z(t)) - self.z0
        yaw = float(self.f_yaw(t)) - self.yaw0

        # Rotate position into the heading=0 frame
        c, s = np.cos(-self.yaw0), np.sin(-self.yaw0)
        x = c * dx - s * dy
        y = s * dx + c * dy

        return x, y, dz, yaw

    def get_relative_transform(self, t1, t2):
        """
        Get the relative 4x4 transform from time t1 to t2.
        This is the odom-predicted motion between two LiDAR scans.
        """
        x1, y1, z1, yaw1 = self.get_pose(t1)
        x2, y2, z2, yaw2 = self.get_pose(t2)

        # Build world poses
        T1 = pose_to_T(x1, y1, z1, yaw1)
        T2 = pose_to_T(x2, y2, z2, yaw2)

        # Relative: T_rel = T1^-1 @ T2
        return np.linalg.inv(T1) @ T2

    def get_velocity(self, t):
        return float(self.f_vx(t))


###############################################################################
# 5. LIDAR PREPROCESSING
###############################################################################

class LiDARPreprocessor:
    def __init__(self, voxel_size=0.15, min_range=1.0, max_range=50.0):
        self.voxel_size = voxel_size
        self.min_range = min_range
        self.max_range = max_range

    def process(self, points):
        ranges = np.linalg.norm(points, axis=1)
        mask = (ranges > self.min_range) & (ranges < self.max_range)
        points = points[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 4, max_nn=30))
        return pcd


###############################################################################
# 6. LOCAL SUBMAP
###############################################################################

class LocalSubmap:
    def __init__(self, max_scans=20, voxel_size=0.25):
        self.max_scans = max_scans
        self.voxel_size = voxel_size
        self.scan_queue = deque()
        self.combined = None

    def add_scan(self, pcd_body, T_world):
        pcd_world = copy.deepcopy(pcd_body).transform(T_world)
        self.scan_queue.append(pcd_world)
        while len(self.scan_queue) > self.max_scans:
            self.scan_queue.popleft()
        self._rebuild()

    def _rebuild(self):
        combined = o3d.geometry.PointCloud()
        for pcd in self.scan_queue:
            combined += pcd
        combined = combined.voxel_down_sample(self.voxel_size)
        combined.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 4, max_nn=30))
        self.combined = combined

    def get(self):
        return self.combined


###############################################################################
# 7. SCAN MATCHER (multi-scale ICP)
###############################################################################

class ScanMatcher:
    def match(self, source, target, init_T=np.eye(4)):
        """Try ICP at normal scale, fall back to coarse if needed."""
        T, f, r = self._icp(source, target, init_T, 1.0, 50)
        if f > 0.3:
            return T, f, r
        # Coarse fallback
        T2, f2, r2 = self._icp(source, target, init_T, 3.0, 30)
        if f2 > f:
            T3, f3, r3 = self._icp(source, target, T2, 1.0, 50)
            return (T3, f3, r3) if f3 > f else (T2, f2, r2)
        return T, f, r

    def _icp(self, src, tgt, init_T, max_d, max_i):
        r = o3d.pipelines.registration.registration_icp(
            src, tgt, max_d, init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_i, relative_fitness=1e-6, relative_rmse=1e-6))
        return r.transformation, r.fitness, r.inlier_rmse


###############################################################################
# 8. LOOP CLOSURE
###############################################################################

class LoopCloser:
    def __init__(self, distance_thresh=5.0, min_interval=100,
                 fitness_thresh=0.4, keyframe_interval=10):
        self.distance_thresh = distance_thresh    # meters to trigger check
        self.min_interval = min_interval          # min scans between closure pair
        self.fitness_thresh = fitness_thresh       # ICP fitness to accept
        self.keyframe_interval = keyframe_interval

        self.keyframes = []  # (scan_idx, pose_4x4, pcd)
        self.matcher = ScanMatcher()
        self.closures = []   # (idx_from, idx_to, T_correction)

    def maybe_add_keyframe(self, scan_idx, pose, pcd):
        """Add keyframe every N scans."""
        if scan_idx % self.keyframe_interval == 0:
            # Downsample for storage
            kf_pcd = pcd.voxel_down_sample(0.3)
            kf_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=1.2, max_nn=30))
            self.keyframes.append((scan_idx, pose.copy(), kf_pcd))

    def check_loop(self, scan_idx, pose, pcd):
        """
        Check if current pose is near any old keyframe.
        Returns (loop_detected, correction_info) or (False, None).
        """
        if len(self.keyframes) < 2:
            return False, None

        current_pos = pose[:3, 3]

        for kf_idx, kf_pose, kf_pcd in self.keyframes:
            # Must be far enough in time (avoid matching neighbors)
            if scan_idx - kf_idx < self.min_interval:
                continue

            # Distance check
            kf_pos = kf_pose[:3, 3]
            dist = np.linalg.norm(current_pos[:2] - kf_pos[:2])  # 2D distance
            if dist > self.distance_thresh:
                continue

            # Try ICP: align current scan to keyframe's world-frame cloud
            # Transform keyframe pcd to world (it's stored in body frame)
            kf_world = copy.deepcopy(kf_pcd).transform(kf_pose)

            T_icp, fitness, rmse = self.matcher.match(pcd, kf_world, pose)

            if fitness > self.fitness_thresh:
                # Loop closure! T_icp is the corrected world pose of current scan
                print(f"\n  *** LOOP CLOSURE: scan {scan_idx} -> keyframe {kf_idx} "
                      f"(dist={dist:.1f}m, fitness={fitness:.3f}) ***")

                correction = {
                    'from_idx': kf_idx,
                    'to_idx': scan_idx,
                    'T_corrected': T_icp.copy(),
                    'T_original': pose.copy(),
                    'kf_pose': kf_pose.copy(),
                }
                self.closures.append(correction)
                return True, correction

        return False, None


def apply_loop_correction(trajectory, correction):
    """
    Distribute loop closure correction linearly along the trajectory.

    When we detect that scan N should actually be at pose T_corrected
    (instead of T_original), we distribute the error evenly across
    all poses from the keyframe (scan K) to scan N.

    This is a simple linear interpolation approach. A full pose graph
    optimizer (g2o, GTSAM) would be better but this is from scratch.
    """
    idx_from = correction['from_idx']
    idx_to = correction['to_idx']
    T_corrected = correction['T_corrected']
    T_original = correction['T_original']

    # Compute the correction delta
    delta_T = T_corrected @ np.linalg.inv(T_original)
    delta_pos = delta_T[:3, 3]
    delta_yaw = rot_to_yaw(delta_T[:3, :3])

    n_poses = idx_to - idx_from
    if n_poses <= 0:
        return trajectory

    print(f"  Distributing correction over {n_poses} poses:")
    print(f"    Position: [{delta_pos[0]:.3f}, {delta_pos[1]:.3f}, {delta_pos[2]:.3f}] m")
    print(f"    Yaw: {np.degrees(delta_yaw):.2f}°")

    # Linearly interpolate the correction
    corrected = list(trajectory)
    for i in range(idx_from, min(idx_to + 1, len(corrected))):
        alpha = (i - idx_from) / n_poses  # 0 at keyframe, 1 at closure
        t_stamp, T_pose = corrected[i]

        # Interpolated correction
        dp = delta_pos * alpha
        dyaw = delta_yaw * alpha

        # Apply
        T_corr = np.eye(4)
        T_corr[:3, :3] = yaw_to_rot(dyaw)
        T_corr[:3, 3] = dp
        T_new = T_corr @ T_pose

        corrected[i] = (t_stamp, T_new)

    return corrected


###############################################################################
# 9. SLAM LOOP
###############################################################################

def run_slam(lidar_data, odom_data, T_lidar_to_base,
             submap_size=20, map_interval=5):
    odom = OdomInterpolator(odom_data)
    preprocessor = LiDARPreprocessor(voxel_size=0.15)
    matcher = ScanMatcher()
    submap = LocalSubmap(max_scans=submap_size, voxel_size=0.25)

    R_l2b = T_lidar_to_base[:3, :3]
    t_l2b = T_lidar_to_base[:3, 3]

    # Initialize global pose from odom (so we're in odom's frame from the start)
    x0, y0, z0, yaw0 = odom.get_pose(lidar_data[0][0])
    global_pose = pose_to_T(x0, y0, z0, yaw0)  # should be identity (origin-aligned)

    trajectory = []
    map_pcds = []
    n_scans = len(lidar_data)
    t0 = time.time()
    low_fit = 0

    print(f"\n[SLAM] Processing {n_scans} scans (submap={submap_size})...")

    for si in range(n_scans):
        t_scan = lidar_data[si][0]
        pts_raw = lidar_data[si][1]

        # ---- LiDAR -> base frame ----
        pts_base = (R_l2b @ pts_raw.T).T + t_l2b
        cur_pcd = preprocessor.process(pts_base)

        # ---- First scan ----
        if si == 0:
            trajectory.append((t_scan, global_pose.copy()))
            submap.add_scan(cur_pcd, global_pose)
            map_pcds.append(copy.deepcopy(cur_pcd).transform(global_pose))
            print(f"  [Scan 0] Init. {len(cur_pcd.points)} pts")
            continue

        # ---- Odom motion prior ----
        t_prev = lidar_data[si - 1][0]
        T_odom_rel = odom.get_relative_transform(t_prev, t_scan)

        # ICP initial guess = last pose + odom motion
        T_init = trajectory[-1][1] @ T_odom_rel

        # ---- Scan-to-submap ICP ----
        T_refined, fitness, rmse = matcher.match(cur_pcd, submap.get(), T_init)

        if fitness < 0.15:
            low_fit += 1
            T_refined = T_init  # fall back to odom prediction

        global_pose = T_refined.copy()

        # ---- Ground plane constraint ----
        ground_alpha = 0.9  # 1.0 = trust ICP fully, 0.0 = force Z=0
        global_pose[2, 3] *= ground_alpha

        # Also constrain roll and pitch (keep Z-axis pointing up)
        R_current = global_pose[:3, :3]
        current_yaw = rot_to_yaw(R_current)
        global_pose[:3, :3] = yaw_to_rot(current_yaw)  # zero out roll/pitch
        trajectory.append((t_scan, global_pose.copy()))

        # ---- Submap ----
        submap.add_scan(cur_pcd, global_pose)

        # ---- Global map ----
        if si % map_interval == 0:
            map_pcds.append(copy.deepcopy(cur_pcd).transform(global_pose))

        # ---- Progress ----
        if si % 200 == 0:
            p = global_pose[:3, 3]
            print(f"  [Scan {si:4d}/{n_scans}] "
                  f"pos=({p[0]:7.2f}, {p[1]:7.2f}, {p[2]:7.2f})  "
                  f"fitness={fitness:.3f}  rmse={rmse:.4f}  "
                  f"t={time.time()-t0:.0f}s")

    total = time.time() - t0
    print(f"\n[SLAM] Done. {n_scans} scans in {total:.1f}s ({n_scans/total:.1f} Hz)")
    print(f"  Low fitness: {low_fit}/{n_scans}")
    return trajectory, map_pcds, []


###############################################################################
# 10. SAVE
###############################################################################

def save_results(trajectory, map_pcds, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    traj_arr = []
    for t, T in trajectory:
        p = T[:3, 3]
        q = rot_to_quat(T[:3, :3])
        traj_arr.append([t, p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
    traj_arr = np.array(traj_arr)
    np.savetxt(os.path.join(output_dir, "trajectory.txt"), traj_arr,
               header="timestamp x y z qx qy qz qw", fmt="%.6f")
    print(f"[SAVE] Trajectory: {len(traj_arr)} poses")

    if map_pcds:
        combined = o3d.geometry.PointCloud()
        for pcd in map_pcds:
            combined += pcd
        combined = combined.voxel_down_sample(0.3)
        map_path = os.path.join(output_dir, "global_map.pcd")
        o3d.io.write_point_cloud(map_path, combined)
        print(f"[SAVE] Map: {len(combined.points)} points")

    return traj_arr


###############################################################################
# 11. VISUALIZATION
###############################################################################

def visualize_all(traj_arr, odom_data, closures, output_dir):
    slam_t = traj_arr[:, 0]
    slam_pos = traj_arr[:, 1:4]
    t_rel = slam_t - slam_t[0]

    # Odom aligned to origin
    odom_t = np.array([d[0] for d in odom_data])
    odom_raw = np.array([[d[1], d[2], d[3]] for d in odom_data])
    idx0 = np.argmin(np.abs(odom_t - slam_t[0]))
    odom_pos = odom_raw - odom_raw[idx0]

    # Rotate odom to match SLAM heading
    # SLAM starts at identity (yaw=0), odom starts at its own yaw
    odom_yaw0 = odom_data[idx0][4]
    c, s = np.cos(-odom_yaw0), np.sin(-odom_yaw0)
    R2d = np.array([[c, -s], [s, c]])
    odom_pos[:, :2] = (R2d @ odom_pos[:, :2].T).T

    # ---- Bird's-eye XY + height ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.plot(slam_pos[:, 0], slam_pos[:, 1], 'b-', lw=2, label='SLAM')
    ax.plot(odom_pos[:, 0], odom_pos[:, 1], 'r--', lw=1.5, alpha=0.6, label='Odom')
    ax.plot(slam_pos[0, 0], slam_pos[0, 1], 'go', ms=10, label='Start')
    ax.plot(slam_pos[-1, 0], slam_pos[-1, 1], 'rs', ms=10, label='End')
    # Mark loop closures
    for lc in closures:
        idx = lc['to_idx']
        if idx < len(slam_pos):
            ax.plot(slam_pos[idx, 0], slam_pos[idx, 1], 'm*', ms=15,
                    zorder=5)
    if closures:
        ax.plot([], [], 'm*', ms=10, label='Loop Closure')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title("Trajectory (Bird's Eye View)")
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_rel, slam_pos[:, 2], 'b-', lw=2, label='SLAM Z')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Z (m)')
    ax.set_title('Height Profile')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_2d.png"), dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved trajectory_2d.png")
    plt.show()

    # ---- 3D trajectory ----
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(slam_pos[:, 0], slam_pos[:, 1], slam_pos[:, 2], 'b-', lw=2, label='SLAM')
    ax.plot(odom_pos[:, 0], odom_pos[:, 1], odom_pos[:, 2],
            'r--', lw=1.5, alpha=0.6, label='Odom')
    ax.scatter(*slam_pos[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(*slam_pos[-1], c='red', s=100, marker='s', label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    plt.savefig(os.path.join(output_dir, "trajectory_3d.png"), dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved trajectory_3d.png")
    plt.show()

    # ---- SLAM vs Odom error ----
    fx = interp1d(odom_t, odom_pos[:, 0], fill_value='extrapolate')
    fy = interp1d(odom_t, odom_pos[:, 1], fill_value='extrapolate')
    odom_interp = np.column_stack([fx(slam_t), fy(slam_t)])
    error = np.linalg.norm(slam_pos[:, :2] - odom_interp, axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_rel, error, 'b-', lw=1.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('2D Difference (m)')
    ax.set_title('SLAM vs Odom Position Difference (XY only)')
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95,
            f"Mean: {error.mean():.2f} m\nMax: {error.max():.2f} m",
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_odom.png"), dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved error_vs_odom.png")
    plt.show()


def view_3d_map(output_dir):
    map_path = os.path.join(output_dir, "global_map.pcd")
    if not os.path.exists(map_path):
        print("[VIS] No map found."); return

    pcd = o3d.io.read_point_cloud(map_path)
    print(f"[VIS] Map: {len(pcd.points)} points")

    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(z_norm)[:, :3])

    geometries = [pcd]
    traj_path = os.path.join(output_dir, "trajectory.txt")
    if os.path.exists(traj_path):
        traj = np.loadtxt(traj_path)[:, 1:4]
        lines = [[i, i+1] for i in range(len(traj)-1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(traj)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
        geometries.append(ls)

    o3d.visualization.draw_geometries(geometries,
        window_name="SLAM Map + Trajectory", width=1280, height=720)


###############################################################################
# 12. ENTRY POINT
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="LiDAR-Odom SLAM + Loop Closure")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--output", default="slam_output")
    parser.add_argument("--submap-size", type=int, default=20)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--view-map", action="store_true")
    args = parser.parse_args()

    # Replaced extract_lidar_to_base with the hardcoded calibration function
    T_l2b = get_calibrated_lidar_to_base()
    
    lidar_data, odom_data = extract_data(args.bag, args.max_scans)
    trajectory, map_pcds, closures = run_slam(
        lidar_data, odom_data, T_l2b, submap_size=args.submap_size)
    traj_arr = save_results(trajectory, map_pcds, args.output)

    if not args.no_vis:
        visualize_all(traj_arr, odom_data, closures, args.output)
    if args.view_map:
        view_3d_map(args.output)

    d = np.linalg.norm(traj_arr[-1, 1:4] - traj_arr[0, 1:4])
    print(f"\n[SUMMARY]")
    print(f"  SLAM displacement: {d:.2f} m")
    print(f"  Loop closures: {len(closures)}")
    if closures:
        print(f"  End-to-start distance: {d:.2f} m (should be near 0 for a loop)")


if __name__ == "__main__":
    main()