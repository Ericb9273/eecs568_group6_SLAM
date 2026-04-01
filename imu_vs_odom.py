#!/usr/bin/env python3
"""
IMU Dead-Reckoning vs Odom Diagnostic
=======================================
Goal: Figure out what /robot/robotnik_base_control/odom actually is.

We run THREE integration methods on the raw IMU data:
  1. Full dead-reckoning: integrate accel (gravity-compensated) for position
  2. Gyro-only heading: integrate gyro for yaw, compare vs odom heading
  3. Velocity from accel: single-integrate accel, compare vs odom velocity

Then we analyze the odom itself:
  - Does Z stay at 0? (wheel encoders are 2D → Z=0 always)
  - Is velocity smooth or noisy? (encoders = smooth, IMU = noisy)
  - Does heading match gyro integration? (yes = uses gyro, no = pure wheels)

Usage:
  python imu_vs_odom.py --bag /path/to/03_80m_other_sensor.bag
"""

import argparse
import numpy as np
import rosbag
from tf.transformations import euler_from_quaternion, quaternion_matrix, quaternion_from_matrix
import matplotlib.pyplot as plt


###############################################################################
# Math
###############################################################################

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def exp_so3(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + skew(phi)
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

def quat_to_rot(q):
    """[x,y,z,w] -> 3x3"""
    return quaternion_matrix(q)[:3, :3]

def rot_to_yaw(R):
    """Extract yaw (rotation about Z) from a rotation matrix."""
    return np.arctan2(R[1, 0], R[0, 0])


###############################################################################
# Data extraction
###############################################################################

def extract_data(bag_path):
    print(f"[DATA] Reading {bag_path}...")
    bag = rosbag.Bag(bag_path)

    # IMU mounted 180° about X → R_imu_to_base = diag(1, -1, -1)
    R_i2b = np.diag([1.0, -1.0, -1.0])

    imu_data = []   # (t, acc_base, gyro_base)
    odom_data = []  # (t, pos, quat_xyzw)

    for topic, msg, t in bag.read_messages(
        topics=['/robot/imu/data', '/robot/robotnik_base_control/odom']
    ):
        stamp = msg.header.stamp.to_sec()

        if topic == '/robot/imu/data':
            acc = R_i2b @ np.array([msg.linear_acceleration.x,
                                     msg.linear_acceleration.y,
                                     msg.linear_acceleration.z])
            gyro = R_i2b @ np.array([msg.angular_velocity.x,
                                      msg.angular_velocity.y,
                                      msg.angular_velocity.z])
            imu_data.append((stamp, acc, gyro))

        elif topic == '/robot/robotnik_base_control/odom':
            pos = np.array([msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            msg.pose.pose.position.z])
            ori = np.array([msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w])
            # Also grab twist (velocity in body frame)
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            vz = msg.twist.twist.linear.z
            wz = msg.twist.twist.angular.z
            odom_data.append((stamp, pos, ori, np.array([vx, vy, vz]), wz))

    bag.close()
    imu_data.sort(key=lambda x: x[0])
    odom_data.sort(key=lambda x: x[0])

    print(f"  IMU:  {len(imu_data)} samples "
          f"({(imu_data[-1][0]-imu_data[0][0]):.1f}s, "
          f"~{len(imu_data)/(imu_data[-1][0]-imu_data[0][0]):.0f} Hz)")
    print(f"  Odom: {len(odom_data)} samples "
          f"({(odom_data[-1][0]-odom_data[0][0]):.1f}s, "
          f"~{len(odom_data)/(odom_data[-1][0]-odom_data[0][0]):.0f} Hz)")
    return imu_data, odom_data


###############################################################################
# IMU dead reckoning: full gravity-compensated integration
###############################################################################

def imu_dead_reckoning(imu_data):
    """
    Proper IMU dead-reckoning:
      1. Estimate initial orientation from first 500 samples (gravity alignment)
      2. Estimate gyro bias and accel bias from static period
      3. For each IMU sample:
         a. Update orientation by integrating bias-corrected gyro
         b. Rotate bias-corrected accel to world frame
         c. Subtract gravity
         d. Integrate for velocity and position

    Returns arrays of (timestamps, positions, velocities, yaw_angles)
    """
    print("\n[IMU] Running dead-reckoning...")

    # ---- Static calibration from first 500 samples ----
    n_cal = min(500, len(imu_data))
    accels_cal = np.array([imu_data[i][1] for i in range(n_cal)])
    gyros_cal = np.array([imu_data[i][2] for i in range(n_cal)])

    g_body = accels_cal.mean(axis=0)
    g_mag = np.linalg.norm(g_body)
    b_gyro = gyros_cal.mean(axis=0)
    b_accel = np.zeros(3)  # we handle gravity explicitly

    print(f"  Gravity in base frame: [{g_body[0]:.4f}, {g_body[1]:.4f}, {g_body[2]:.4f}]")
    print(f"  Gravity magnitude: {g_mag:.4f} m/s^2")
    print(f"  Gyro bias: [{b_gyro[0]:.6f}, {b_gyro[1]:.6f}, {b_gyro[2]:.6f}] rad/s")

    # Initial orientation: align measured gravity with world +Z
    g_hat = g_body / g_mag
    z_w = np.array([0.0, 0.0, 1.0])
    v_cross = np.cross(g_hat, z_w)
    s = np.linalg.norm(v_cross)
    c = np.dot(g_hat, z_w)
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([-1, 1, -1])
    else:
        vx = skew(v_cross)
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    g_check = R @ g_body
    print(f"  Gravity after alignment: [{g_check[0]:.4f}, {g_check[1]:.4f}, {g_check[2]:.4f}]")

    gravity_world = np.array([0.0, 0.0, -g_mag])

    # ---- Integration ----
    pos = np.zeros(3)
    vel = np.zeros(3)

    timestamps = []
    positions = []
    velocities = []
    yaws = []

    for i in range(1, len(imu_data)):
        dt = imu_data[i][0] - imu_data[i-1][0]
        if dt <= 0 or dt > 0.05:
            continue

        acc_raw = imu_data[i][1]
        gyro_raw = imu_data[i][2]

        # Bias-corrected
        gyro_c = gyro_raw - b_gyro
        acc_c = acc_raw - b_accel

        # Update orientation
        R = R @ exp_so3(gyro_c * dt)

        # World-frame acceleration (gravity-compensated)
        a_world = R @ acc_c + gravity_world

        # Integrate
        pos = pos + vel * dt + 0.5 * a_world * dt**2
        vel = vel + a_world * dt

        timestamps.append(imu_data[i][0])
        positions.append(pos.copy())
        velocities.append(vel.copy())
        yaws.append(rot_to_yaw(R))

    timestamps = np.array(timestamps)
    positions = np.array(positions)
    velocities = np.array(velocities)
    yaws = np.array(yaws)

    print(f"  Final position: ({positions[-1, 0]:.1f}, {positions[-1, 1]:.1f}, {positions[-1, 2]:.1f})")
    print(f"  Final velocity: ({velocities[-1, 0]:.1f}, {velocities[-1, 1]:.1f}, {velocities[-1, 2]:.1f})")

    return timestamps, positions, velocities, yaws


###############################################################################
# Gyro-only heading integration (no accel, just rotation)
###############################################################################

def gyro_heading_only(imu_data):
    """Just integrate gyro for heading — no position."""
    n_cal = min(500, len(imu_data))
    gyros_cal = np.array([imu_data[i][2] for i in range(n_cal)])
    b_gyro = gyros_cal.mean(axis=0)

    # Start with gravity-aligned orientation
    accels_cal = np.array([imu_data[i][1] for i in range(n_cal)])
    g_body = accels_cal.mean(axis=0)
    g_hat = g_body / np.linalg.norm(g_body)
    z_w = np.array([0.0, 0.0, 1.0])
    v_cross = np.cross(g_hat, z_w)
    s = np.linalg.norm(v_cross)
    c = np.dot(g_hat, z_w)
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([-1, 1, -1])
    else:
        vx = skew(v_cross)
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    timestamps = []
    yaws = []

    for i in range(1, len(imu_data)):
        dt = imu_data[i][0] - imu_data[i-1][0]
        if dt <= 0 or dt > 0.05:
            continue
        gyro_c = imu_data[i][2] - b_gyro
        R = R @ exp_so3(gyro_c * dt)
        timestamps.append(imu_data[i][0])
        yaws.append(rot_to_yaw(R))

    return np.array(timestamps), np.array(yaws)


###############################################################################
# Analyze the odom topic itself
###############################################################################

def analyze_odom(odom_data):
    """Print diagnostic info about the odom topic."""
    print("\n[ODOM] Analyzing /robot/robotnik_base_control/odom...")

    positions = np.array([d[1] for d in odom_data])
    quats = np.array([d[2] for d in odom_data])
    lin_vels = np.array([d[3] for d in odom_data])
    ang_vels = np.array([d[4] for d in odom_data])
    times = np.array([d[0] for d in odom_data])

    # Z analysis
    z = positions[:, 2]
    print(f"  Z range: [{z.min():.6f}, {z.max():.6f}] m")
    print(f"  Z std:   {z.std():.6f} m")
    if z.std() < 0.001:
        print(f"  → Z is constant/near-zero → LIKELY 2D (wheel encoders)")
    else:
        print(f"  → Z varies → LIKELY 3D (IMU-based or fused)")

    # Velocity analysis
    speed = np.linalg.norm(lin_vels, axis=1)
    print(f"\n  Speed range: [{speed.min():.4f}, {speed.max():.4f}] m/s")
    print(f"  Speed mean:  {speed.mean():.4f} m/s")

    # Velocity smoothness (jerk = derivative of acceleration)
    if len(speed) > 10:
        dt_avg = np.mean(np.diff(times))
        accel_from_speed = np.diff(speed) / dt_avg
        jerk = np.diff(accel_from_speed) / dt_avg
        print(f"  Acceleration std: {accel_from_speed.std():.4f} m/s^2")
        print(f"  Jerk std:         {jerk.std():.4f} m/s^3")
        if accel_from_speed.std() < 0.5:
            print(f"  → Velocity is smooth → LIKELY wheel encoders")
        else:
            print(f"  → Velocity is noisy → LIKELY IMU-derived")

    # Angular velocity analysis
    print(f"\n  Angular vel (yaw) range: [{ang_vels.min():.4f}, {ang_vels.max():.4f}] rad/s")
    print(f"  Angular vel (yaw) std:   {ang_vels.std():.4f} rad/s")

    # Heading from odom quaternions
    yaws = []
    for q in quats:
        _, _, yaw = euler_from_quaternion(q)
        yaws.append(yaw)
    yaws = np.array(yaws)

    # Total distance
    dp = np.diff(positions, axis=0)
    total_dist = np.sum(np.linalg.norm(dp, axis=1))
    displacement = np.linalg.norm(positions[-1] - positions[0])
    print(f"\n  Total path length: {total_dist:.2f} m")
    print(f"  Net displacement:  {displacement:.2f} m")
    print(f"  Start: ({positions[0, 0]:.2f}, {positions[0, 1]:.2f}, {positions[0, 2]:.2f})")
    print(f"  End:   ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f}, {positions[-1, 2]:.2f})")

    return times, positions, yaws, lin_vels


###############################################################################
# Plotting
###############################################################################

def plot_comparison(imu_t, imu_pos, imu_vel, imu_yaw,
                    gyro_t, gyro_yaw,
                    odom_t, odom_pos, odom_yaw, odom_vel,
                    output_dir="diagnostic_output"):
    os.makedirs(output_dir, exist_ok=True)

    # Align time to start at 0
    t0 = min(imu_t[0], odom_t[0])
    imu_t_rel = imu_t - t0
    gyro_t_rel = gyro_t - t0
    odom_t_rel = odom_t - t0

    # Align odom position to start at origin
    odom_pos_aligned = odom_pos - odom_pos[0]

    # =====================================================================
    # PLOT 1: XY trajectory comparison
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    ax.plot(odom_pos_aligned[:, 0], odom_pos_aligned[:, 1],
            'r-', lw=2, label='Odom', alpha=0.8)
    ax.plot(imu_pos[:, 0], imu_pos[:, 1],
            'b-', lw=1.5, label='IMU Dead-Reckoning', alpha=0.7)
    ax.plot(0, 0, 'go', ms=10, label='Start')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('XY Trajectory: IMU Dead-Reckoning vs Odom')
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Zoom into first 30 seconds where IMU might still be reasonable
    mask_imu = imu_t_rel < 30
    mask_odom = odom_t_rel < 30
    ax.plot(odom_pos_aligned[mask_odom, 0], odom_pos_aligned[mask_odom, 1],
            'r-', lw=2, label='Odom (first 30s)', alpha=0.8)
    ax.plot(imu_pos[mask_imu, 0], imu_pos[mask_imu, 1],
            'b-', lw=1.5, label='IMU (first 30s)', alpha=0.7)
    ax.plot(0, 0, 'go', ms=10)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('First 30 Seconds (before IMU drifts)')
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xy_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved xy_comparison.png")
    plt.show()

    # =====================================================================
    # PLOT 2: Position components over time
    # =====================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    labels = ['X', 'Y', 'Z']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(odom_t_rel, odom_pos_aligned[:, i], 'r-', lw=2, label=f'Odom {label}')
        ax.plot(imu_t_rel, imu_pos[:, i], 'b-', lw=1, alpha=0.7, label=f'IMU {label}')
        ax.set_ylabel(f'{label} (m)')
        ax.legend(); ax.grid(True, alpha=0.3)

    axes[0].set_title('Position Components: IMU vs Odom')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "position_components.png"), dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved position_components.png")
    plt.show()

    # =====================================================================
    # PLOT 3: Heading (yaw) comparison — this is the key diagnostic
    # =====================================================================
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    ax = axes[0]
    ax.plot(odom_t_rel, np.degrees(odom_yaw), 'r-', lw=2, label='Odom Heading')
    ax.plot(gyro_t_rel, np.degrees(gyro_yaw), 'b-', lw=1.5, alpha=0.7, label='Gyro Integration')
    ax.set_ylabel('Yaw (degrees)')
    ax.set_title('Heading: Gyro Integration vs Odom')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Heading difference
    from scipy.interpolate import interp1d
    f_odom_yaw = interp1d(odom_t, odom_yaw, fill_value='extrapolate')
    odom_yaw_at_gyro = f_odom_yaw(gyro_t)
    yaw_diff = np.degrees(gyro_yaw - odom_yaw_at_gyro)
    # Wrap to [-180, 180]
    yaw_diff = (yaw_diff + 180) % 360 - 180

    ax = axes[1]
    ax.plot(gyro_t_rel, yaw_diff, 'g-', lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Difference (deg)')
    ax.set_title(f'Gyro - Odom Heading (mean: {np.mean(np.abs(yaw_diff)):.2f}°)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heading_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved heading_comparison.png")
    plt.show()

    # =====================================================================
    # PLOT 4: Velocity comparison
    # =====================================================================
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Odom speed (from twist)
    odom_speed = np.linalg.norm(odom_vel, axis=1)
    # IMU speed
    imu_speed = np.linalg.norm(imu_vel, axis=1)

    ax = axes[0]
    ax.plot(odom_t_rel, odom_speed, 'r-', lw=2, label='Odom Speed')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Odom Velocity Profile (from twist)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(imu_t_rel, imu_speed, 'b-', lw=1, alpha=0.7, label='IMU Speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('IMU Integrated Velocity')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "velocity_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved velocity_comparison.png")
    plt.show()

    # =====================================================================
    # PLOT 5: Raw IMU data (first 10 seconds)
    # =====================================================================
    mask = imu_t_rel < 10
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    accels = np.array([imu_data_global[i][1] for i in range(sum(mask))])
    gyros = np.array([imu_data_global[i][2] for i in range(sum(mask))])

    ax = axes[0]
    ax.plot(imu_t_rel[mask][:len(accels)], accels[:, 0], label='ax')
    ax.plot(imu_t_rel[mask][:len(accels)], accels[:, 1], label='ay')
    ax.plot(imu_t_rel[mask][:len(accels)], accels[:, 2], label='az')
    ax.set_ylabel('m/s^2'); ax.set_title('Raw Accel (base frame, first 10s)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(imu_t_rel[mask][:len(gyros)], gyros[:, 0], label='gx')
    ax.plot(imu_t_rel[mask][:len(gyros)], gyros[:, 1], label='gy')
    ax.plot(imu_t_rel[mask][:len(gyros)], gyros[:, 2], label='gz')
    ax.set_ylabel('rad/s'); ax.set_title('Raw Gyro (base frame, first 10s)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_imu.png"), dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved raw_imu.png")
    plt.show()

    # =====================================================================
    # Summary verdict
    # =====================================================================
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    # Z test
    odom_z_std = np.std(odom_pos_aligned[:, 2])
    if odom_z_std < 0.001:
        print(f"  Odom Z std = {odom_z_std:.6f} → Z is flat → 2D source (encoders)")
        z_verdict = "ENCODERS"
    else:
        print(f"  Odom Z std = {odom_z_std:.4f} → Z varies → 3D source (IMU or fused)")
        z_verdict = "IMU/FUSED"

    # Heading test
    mean_yaw_diff = np.mean(np.abs(yaw_diff))
    if mean_yaw_diff < 2.0:
        print(f"  Heading diff = {mean_yaw_diff:.2f}° → Odom heading matches gyro → uses gyro")
        heading_verdict = "GYRO"
    elif mean_yaw_diff < 10.0:
        print(f"  Heading diff = {mean_yaw_diff:.2f}° → Partially matches → fused or filtered gyro")
        heading_verdict = "FUSED"
    else:
        print(f"  Heading diff = {mean_yaw_diff:.2f}° → Doesn't match → independent heading source")
        heading_verdict = "INDEPENDENT"

    # Velocity test
    if odom_speed.std() < 0.1 and odom_speed.max() < 2.0:
        print(f"  Odom speed smooth (std={odom_speed.std():.4f}) → LIKELY encoders")
        vel_verdict = "ENCODERS"
    else:
        print(f"  Odom speed (std={odom_speed.std():.4f}) → check plot for noise level")
        vel_verdict = "UNKNOWN"

    # Overall
    print(f"\n  Z says: {z_verdict}")
    print(f"  Heading says: {heading_verdict}")
    print(f"  Velocity says: {vel_verdict}")

    if z_verdict == "ENCODERS":
        print("\n  → VERDICT: Odom is most likely WHEEL ENCODER based")
        print("    (2D position from encoders, heading possibly fused with gyro)")
    else:
        print("\n  → VERDICT: Odom uses 3D information (IMU integration or sensor fusion)")


###############################################################################
# Main
###############################################################################

import os

# Global reference for raw IMU plot
imu_data_global = None

def main():
    global imu_data_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", default="diagnostic_output")
    args = parser.parse_args()

    imu_data, odom_data = extract_data(args.bag)
    imu_data_global = imu_data

    # Analyze odom
    odom_t, odom_pos, odom_yaw, odom_vel = analyze_odom(odom_data)

    # IMU dead-reckoning
    imu_t, imu_pos, imu_vel, imu_yaw = imu_dead_reckoning(imu_data)

    # Gyro-only heading
    gyro_t, gyro_yaw = gyro_heading_only(imu_data)

    # Plot everything
    plot_comparison(imu_t, imu_pos, imu_vel, imu_yaw,
                    gyro_t, gyro_yaw,
                    odom_t, odom_pos, odom_yaw, odom_vel,
                    args.output)


if __name__ == "__main__":
    main()
