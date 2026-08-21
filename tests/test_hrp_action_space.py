from __future__ import annotations

import numpy as np

from tcc_real_robot.hrp_action_space import (
    clip_hrp_action,
    hrp_state,
    matrix_to_rotation_vector,
    measured_hrp_velocity,
    rotation_vector_to_matrix,
)


def test_rotation_vector_round_trip() -> None:
    vector = np.array([0.1, -0.2, 0.3])
    recovered = matrix_to_rotation_vector(rotation_vector_to_matrix(vector))
    assert np.allclose(recovered, vector, atol=1e-8)


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
