"""HRP end-effector action semantics and the isolated Trossen adapter math."""

from __future__ import annotations

from math import acos, cos, isfinite, sin

import numpy as np


def rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert an angle-axis vector to a 3x3 rotation matrix."""
    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("rotation_vector must contain three finite values")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        skew = np.array(
            [[0.0, -vector[2], vector[1]],
             [vector[2], 0.0, -vector[0]],
             [-vector[1], vector[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3) + skew
    axis = vector / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + sin(angle) * skew + (1.0 - cos(angle)) * (skew @ skew)


def euler_xyz_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert dataset roll/pitch/yaw to a rotation matrix.

    The LeRobot dataset explicitly records the last three Cartesian fields as
    roll, pitch, and yaw.  They are intrinsic XYZ angles, equivalently
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.  They must not be passed to the
    Trossen angle-axis interface as though they were a rotation vector.
    """
    angles = np.asarray(euler_xyz, dtype=np.float64)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("euler_xyz must contain three finite values")
    roll, pitch, yaw = angles
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to its principal angle-axis vector."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("matrix must be a finite 3x3 rotation matrix")
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = acos(cosine)
    if angle < 1e-8:
        return 0.5 * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
    if np.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * sin(angle))
    return axis * angle


def hrp_state(cartesian_pose: np.ndarray, gripper_position: float) -> np.ndarray:
    """Build the official 7-D HRP state: Cartesian pose plus gripper position."""
    pose = np.asarray(cartesian_pose, dtype=np.float32)
    if pose.shape != (6,) or not np.isfinite(pose).all() or not isfinite(gripper_position):
        raise ValueError("HRP state requires a finite 6-D pose and gripper position")
    return np.concatenate((pose, np.array([gripper_position], dtype=np.float32)))


def dataset_euler_pose_to_hrp_pose(cartesian_pose: np.ndarray) -> np.ndarray:
    """Map dataset ``xyz + roll/pitch/yaw`` to Trossen ``xyz + angle-axis``."""
    pose = np.asarray(cartesian_pose, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Dataset Cartesian pose must contain six finite values")
    rotation_vector = matrix_to_rotation_vector(euler_xyz_to_matrix(pose[3:6]))
    return np.concatenate((pose[:3], rotation_vector)).astype(np.float32)


def measured_hrp_velocity(
    current_state: np.ndarray,
    next_state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Recover HRP base-frame Cartesian/gripper velocity from two observations."""
    current = np.asarray(current_state, dtype=np.float64)
    following = np.asarray(next_state, dtype=np.float64)
    if (
        current.shape != (7,)
        or following.shape != (7,)
        or not np.isfinite(current).all()
        or not np.isfinite(following).all()
        or not isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("Velocity reconstruction requires finite 7-D states and dt > 0")
    linear = (following[:3] - current[:3]) / dt
    current_rotation = rotation_vector_to_matrix(current[3:6])
    next_rotation = rotation_vector_to_matrix(following[3:6])
    # Spatial angular velocity is expressed in the fixed robot base frame.
    angular = matrix_to_rotation_vector(next_rotation @ current_rotation.T) / dt
    gripper = np.array([(following[6] - current[6]) / dt])
    return np.concatenate((linear, angular, gripper)).astype(np.float32)


def clip_hrp_action(
    action: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Apply an external driver action envelope without changing the policy."""
    velocity = np.asarray(action, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if (
        velocity.shape != (7,)
        or low.shape != (7,)
        or high.shape != (7,)
        or not np.isfinite(np.concatenate((velocity, low, high))).all()
        or np.any(low > high)
    ):
        raise ValueError("HRP action and bounds must be finite ordered 7-D vectors")
    return np.clip(velocity, low, high).astype(np.float32)
