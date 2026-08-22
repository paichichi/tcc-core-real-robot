from __future__ import annotations

import numpy as np
import pytest

from tcc_real_robot.hrp_action_space import (
    clip_hrp_action,
    dataset_euler_pose_to_hrp_pose,
    euler_xyz_to_matrix,
    hrp_state,
    matrix_to_rotation_vector,
    measured_hrp_velocity,
    rotation_vector_to_matrix,
)


def test_velocity_bounds_reject_legacy_joint_position_limits() -> None:
    with pytest.raises(ValueError, match="contain zero"):
        clip_hrp_action(
            np.zeros(7),
            np.array([-0.6, 0.5, 0.5, -1.5, -0.2, -0.9, -0.001]),
            np.array([0.6, 2.5, 2.3, 0.7, 0.5, 0.9, 0.04]),
        )


def test_rotation_vector_round_trip() -> None:
    vector = np.array([0.1, -0.2, 0.3])
    recovered = matrix_to_rotation_vector(rotation_vector_to_matrix(vector))
    assert np.allclose(recovered, vector, atol=1e-8)


def test_dataset_euler_pose_is_converted_to_trossen_angle_axis() -> None:
    dataset_pose = np.array([0.3, -0.1, 0.2, 0.2, -0.3, 0.4])
    driver_pose = dataset_euler_pose_to_hrp_pose(dataset_pose)
    assert np.allclose(driver_pose[:3], dataset_pose[:3])
    assert np.allclose(
        rotation_vector_to_matrix(driver_pose[3:6]),
        euler_xyz_to_matrix(dataset_pose[3:6]),
        atol=1e-7,
    )


def test_dataset_euler_wrap_does_not_create_false_angular_velocity() -> None:
    current_pose = dataset_euler_pose_to_hrp_pose(
        np.array([0.0, 0.0, 0.0, -3.0536284, 1.3513319, -3.1360779])
    )
    following_pose = dataset_euler_pose_to_hrp_pose(
        np.array([0.0, 0.0, 0.0, -3.0693729, 1.3424587, 3.1317558])
    )
    current = hrp_state(current_pose, 0.0)
    following = hrp_state(following_pose, 0.0)
    velocity = measured_hrp_velocity(current, following, 0.05)
    assert np.linalg.norm(velocity[3:6]) < 0.2


def test_measured_velocity_integrates_to_the_next_driver_pose() -> None:
    dt = 0.05
    current = hrp_state(
        dataset_euler_pose_to_hrp_pose(
            np.array([0.30, -0.10, 0.20, 0.2, -0.3, 0.4])
        ),
        0.01,
    )
    following = hrp_state(
        dataset_euler_pose_to_hrp_pose(
            np.array([0.31, -0.12, 0.23, 0.22, -0.28, 0.37])
        ),
        0.013,
    )
    velocity = measured_hrp_velocity(current, following, dt)

    integrated_position = current[:3] + velocity[:3] * dt
    integrated_rotation = (
        rotation_vector_to_matrix(velocity[3:6] * dt)
        @ rotation_vector_to_matrix(current[3:6])
    )
    integrated_gripper = current[6] + velocity[6] * dt

    assert np.allclose(integrated_position, following[:3], atol=1e-7)
    assert np.allclose(
        integrated_rotation,
        rotation_vector_to_matrix(following[3:6]),
        atol=1e-7,
    )
    assert np.isclose(integrated_gripper, following[6], atol=1e-7)


def test_measured_velocity_is_expressed_in_base_frame() -> None:
    current = hrp_state(np.zeros(6), 0.01)
    following = hrp_state(
        np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.1]), 0.012
    )
    velocity = measured_hrp_velocity(current, following, 0.05)
    assert np.allclose(velocity, [0.2, -0.4, 0.6, 0.0, 0.0, 2.0, 0.04])


def test_driver_action_envelope_is_external_to_policy() -> None:
    clipped = clip_hrp_action(
        np.array([-2.0, -0.5, 0.0, 0.5, 2.0, 0.0, 0.1]),
        np.full(7, -1.0),
        np.full(7, 1.0),
    )
    assert np.allclose(clipped, [-1.0, -0.5, 0.0, 0.5, 1.0, 0.0, 0.1])
