from types import SimpleNamespace

import pytest

from tcc_real_robot.reporting import (
    format_cartesian_step_report,
    format_current_position_hold_report,
    format_folded_pose_return_report,
    format_gripper_cycle_report,
    format_position_hold_report,
    format_preflight_report,
    format_whole_arm_cycle_report,
    format_workspace_point_report,
)
from tcc_real_robot.robot_inspection import (
    inspect_robot,
    monitor_robot,
    preflight_robot,
    run_cartesian_step_test,
    run_current_position_hold_test,
    run_folded_pose_return,
    run_gripper_cycle_test,
    run_position_hold_test,
    run_whole_arm_cycle_test,
)


class FakeDriver:
    def __init__(self) -> None:
        self.configure_args = None
        self.cleaned_up = False
        self.mode_calls: list[object] = []
        self.position_calls: list[tuple[list[float], float, bool]] = []
        self.arm_mode_calls: list[object] = []
        self.gripper_mode_calls: list[object] = []
        self.arm_position_calls: list[tuple[list[float], float, bool]] = []
        self.gripper_position_calls: list[tuple[float, float, bool]] = []
        self.cartesian_position_calls: list[tuple[list[float], object, float, bool, int]] = []
        self.joint_modes = [0] * 7
        self.arm_positions = [0.0] * 6
        self.gripper_position = 0.02
        self.cartesian_positions = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]

    def configure(self, *args: object) -> None:
        self.configure_args = args

    def get_driver_version(self) -> str:
        return "v1.9.0"

    def get_controller_version(self) -> str:
        return "v1.9.2"

    def get_num_joints(self) -> int:
        return 7

    def get_arm_positions(self) -> list[float]:
        return self.arm_positions.copy()

    def get_gripper_position(self) -> float:
        return self.gripper_position

    def get_all_positions(self) -> list[float]:
        return self.arm_positions.copy() + [self.gripper_position]

    def get_all_rotor_temperatures(self) -> list[float]:
        return [30.0] * 7

    def get_all_driver_temperatures(self) -> list[float]:
        return [31.0] * 7

    def get_cartesian_positions(self) -> list[float]:
        return self.cartesian_positions.copy()

    def get_joint_limits(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                position_min=-3.0,
                position_max=3.0,
                position_tolerance=0.01,
                velocity_max=1.0,
                velocity_tolerance=0.1,
                effort_max=10.0,
                effort_tolerance=1.0,
            )
            for _ in range(7)
        ]

    def get_error_information(self) -> str:
        return "none"

    def set_all_modes(self, mode: object) -> None:
        self.mode_calls.append(mode)
        value = 1 if mode == "position" else 0
        self.joint_modes = [value] * 7

    def set_arm_modes(self, mode: object) -> None:
        self.arm_mode_calls.append(mode)
        value = 1 if mode == "position" else 0
        self.joint_modes[:6] = [value] * 6

    def set_gripper_mode(self, mode: object) -> None:
        self.gripper_mode_calls.append(mode)
        self.joint_modes[6] = 1 if mode == "position" else 0

    def set_all_positions(
        self, positions: list[float], goal_time: float, blocking: bool
    ) -> None:
        self.position_calls.append((positions, goal_time, blocking))
        self.arm_positions = positions[:6].copy()
        self.gripper_position = positions[6]

    def set_arm_positions(
        self, positions: list[float], goal_time: float, blocking: bool
    ) -> None:
        self.arm_position_calls.append((positions.copy(), goal_time, blocking))
        self.arm_positions = positions.copy()

    def set_gripper_position(
        self, position: float, goal_time: float, blocking: bool
    ) -> None:
        self.gripper_position_calls.append((position, goal_time, blocking))
        self.gripper_position = position

    def set_cartesian_positions(
        self,
        positions: list[float],
        interpolation_space: object,
        *,
        goal_time: float,
        blocking: bool,
        num_trajectory_check_samples: int,
    ) -> None:
        self.cartesian_position_calls.append(
            (
                positions.copy(),
                interpolation_space,
                goal_time,
                blocking,
                num_trajectory_check_samples,
            )
        )
        self.cartesian_positions = positions.copy()

    def get_modes(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(value=value) for value in self.joint_modes]

    def cleanup(self, reboot_controller: bool) -> None:
        assert reboot_controller is False
        self.cleaned_up = True


def make_api(driver: FakeDriver) -> SimpleNamespace:
    return SimpleNamespace(
        Model=SimpleNamespace(wxai_v0="model"),
        StandardEndEffector=SimpleNamespace(wxai_v0_follower="end-effector"),
        Mode=SimpleNamespace(position="position", idle="idle"),
        InterpolationSpace=SimpleNamespace(cartesian="cartesian"),
        TrossenArmDriver=lambda: driver,
    )


def make_config() -> dict[str, object]:
    return {
        "robot": {
            "driver_model": "wxai_v0",
            "end_effector": "wxai_v0_follower",
            "controller_ip": "192.168.1.2",
            "expected_driver_series": "1.9",
            "expected_firmware_series": "1.9",
            "home_name": "dataset_collection_home",
            "home_source": "test_fixture",
            "home_arm_positions_rad": [0.1] * 6,
            "folded_pose_name": "test_folded_pose",
            "folded_pose_source": "test_fixture",
            "folded_arm_positions_rad": [0.0] * 6,
        },
        "diagnostic_tests": {
            "position_hold": {
                "duration_s": 0.001,
                "sample_rate_hz": 1000.0,
                "max_arm_drift_rad": 0.02,
                "max_gripper_drift_m": 0.002,
            },
            "current_position_hold": {
                "goal_time_s": 0.001,
                "duration_s": 0.001,
                "sample_rate_hz": 1000.0,
                "max_arm_error_rad": 0.02,
                "max_gripper_error_m": 0.002,
            },
            "gripper_cycle": {
                "open_delta_m": 0.005,
                "goal_time_s": 0.001,
                "hold_duration_s": 0.001,
                "sample_rate_hz": 1000.0,
                "max_tracking_error_m": 0.001,
            },
            "whole_arm_cycle": {
                "joint_deltas_rad": [0.03] * 6,
                "goal_time_s": 0.001,
                "hold_duration_s": 0.001,
                "sample_rate_hz": 1000.0,
                "max_tracking_error_rad": 0.02,
                "joint_limit_margin_rad": 0.01,
            },
            "folded_pose_return": {
                "goal_time_s": 0.001,
                "expected_start_tolerance_rad": 0.03,
                "max_tracking_error_rad": 0.03,
            },
            "cartesian_step": {
                "home_goal_time_s": 0.001,
                "max_home_tracking_error_rad": 0.03,
                "goal_time_s": 0.001,
                "hold_duration_s": 0.001,
                "max_translation_step_m": 0.01,
                "max_downward_step_m": 0.005,
                "max_tracking_error_m": 0.005,
                "trajectory_check_samples": 1000,
            },
        },
    }


def test_inspection_is_read_only_and_cleans_up() -> None:
    driver = FakeDriver()
    state = inspect_robot(make_api(driver), make_config(), timeout=3.0)

    assert driver.configure_args == (
        "model",
        "end-effector",
        "192.168.1.2",
        False,
        3.0,
    )
    assert driver.cleaned_up is True
    assert state["arm_positions_rad"] == [0.0] * 6
    assert state["gripper_position_m"] == 0.02


def test_inspection_rejects_unexpected_firmware_series_and_cleans_up() -> None:
    driver = FakeDriver()
    driver.get_controller_version = lambda: "v1.10.0"  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Firmware"):
        inspect_robot(make_api(driver), make_config(), timeout=3.0)

    assert driver.cleaned_up is True


def test_monitor_reads_only_and_cleans_up() -> None:
    driver = FakeDriver()
    samples: list[dict[str, object]] = []

    summary = monitor_robot(
        make_api(driver),
        make_config(),
        timeout=3.0,
        duration=0.001,
        rate_hz=1000.0,
        on_sample=samples.append,
    )

    assert len(samples) == 1
    assert summary["samples"] == 1
    assert summary["observed_rate_hz"] == 0.0
    assert summary["max_arm_change_rad"] == 0.0
    assert driver.cleaned_up is True


def test_preflight_passes_valid_idle_state_and_cleans_up() -> None:
    driver = FakeDriver()

    report = preflight_robot(make_api(driver), make_config(), timeout=3.0)

    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["positions"][-1] == 0.02
    assert driver.cleaned_up is True


def test_preflight_accepts_position_inside_controller_tolerance() -> None:
    driver = FakeDriver()
    driver.get_all_positions = lambda: [-3.005] + [0.0] * 6  # type: ignore[method-assign]

    report = preflight_robot(make_api(driver), make_config(), timeout=3.0)

    assert report["passed"] is True
    assert report["checks"]["positions_within_limits_and_tolerance"] is True
    assert driver.cleaned_up is True


def test_preflight_rejects_position_outside_limits() -> None:
    driver = FakeDriver()
    driver.get_all_positions = lambda: [4.0] + [0.0] * 6  # type: ignore[method-assign]

    report = preflight_robot(make_api(driver), make_config(), timeout=3.0)

    assert report["passed"] is False
    assert report["checks"]["positions_within_limits_and_tolerance"] is False
    assert driver.cleaned_up is True


def test_position_hold_changes_only_modes_and_cleans_up() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]

    report = run_position_hold_test(make_api(driver), make_config(), timeout=3.0)

    assert report["passed"] is True
    assert driver.mode_calls == ["position", "idle"]
    assert report["peak_arm_drift_rad"] == 0.0
    assert report["peak_gripper_drift_m"] == 0.0
    assert driver.cleaned_up is True


def test_current_position_hold_commands_unchanged_target_and_cleans_up() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]

    report = run_current_position_hold_test(
        make_api(driver), make_config(), timeout=3.0
    )

    target = [0.0] * 6 + [0.02]
    assert report["passed"] is True
    assert driver.mode_calls == ["position", "idle"]
    assert driver.position_calls == [(target, 0.001, True)]
    assert report["target_positions"] == target
    assert report["peak_arm_error_rad"] == 0.0
    assert report["peak_gripper_error_m"] == 0.0
    assert report["peak_joint_errors"] == [0.0] * 7
    assert report["samples_observed"] == 1
    assert report["failure_reasons"] == []
    assert driver.cleaned_up is True


def test_current_position_hold_returns_to_idle_when_command_fails() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]

    def fail_command(
        positions: list[float], goal_time: float, blocking: bool
    ) -> None:
        raise RuntimeError("command failed")

    driver.set_all_positions = fail_command  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="command failed"):
        run_current_position_hold_test(make_api(driver), make_config(), timeout=3.0)

    assert driver.mode_calls == ["position", "idle"]
    assert driver.cleaned_up is True


def test_current_position_hold_stops_and_reports_excessive_error() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]
    positions = iter(
        [
            [0.0] * 6 + [0.02],
            [0.03] + [0.0] * 5 + [0.02],
        ]
    )
    driver.get_all_positions = lambda: next(positions)  # type: ignore[method-assign]

    report = run_current_position_hold_test(
        make_api(driver), make_config(), timeout=3.0
    )

    assert report["passed"] is False
    assert report["samples_observed"] == 1
    assert report["peak_joint_errors"][0] == 0.03
    assert report["failure_reasons"] == [
        "Arm error 0.030000 rad exceeded 0.020000 rad"
    ]
    assert driver.mode_calls == ["position", "idle"]
    assert driver.cleaned_up is True


def test_gripper_cycle_moves_only_gripper_and_returns() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]

    report = run_gripper_cycle_test(make_api(driver), make_config(), timeout=3.0)

    assert report["passed"] is True
    assert driver.arm_mode_calls == []
    assert driver.gripper_mode_calls == ["position", "idle"]
    assert driver.gripper_position_calls == [
        (0.025, 0.001, True),
        (0.02, 0.001, True),
    ]
    assert report["return_error_m"] == 0.0
    assert driver.cleaned_up is True


def test_whole_arm_cycle_moves_all_arm_joints_and_returns() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]

    report = run_whole_arm_cycle_test(
        make_api(driver), make_config(), timeout=3.0
    )

    assert report["passed"] is True
    assert driver.arm_mode_calls == ["position", "idle"]
    assert driver.gripper_mode_calls == []
    assert driver.arm_position_calls == [
        ([0.03] * 6, 0.001, True),
        ([0.0] * 6, 0.001, True),
    ]
    assert report["peak_outbound_errors_rad"] == [0.0] * 6
    assert report["return_errors_rad"] == [0.0] * 6
    assert driver.cleaned_up is True


def test_cartesian_step_moves_positive_z_and_returns() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]

    report = run_cartesian_step_test(
        make_api(driver),
        make_config(),
        timeout=3.0,
        axis="z",
        distance_m=0.01,
    )

    origin = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
    target = origin.copy()
    target[2] += 0.01
    assert report["passed"] is True
    assert driver.arm_mode_calls == ["position", "idle"]
    assert driver.gripper_mode_calls == []
    assert driver.arm_position_calls == [([0.1] * 6, 0.001, True)]
    assert driver.cartesian_position_calls == [
        (target, "cartesian", 0.001, True, 1000),
        (origin, "cartesian", 0.001, True, 1000),
    ]
    assert report["target_errors"] == [0.0] * 6
    assert report["return_errors"] == [0.0] * 6
    assert driver.cleaned_up is True


def test_folded_pose_return_requires_home_and_moves_only_the_arm() -> None:
    driver = FakeDriver()
    driver.arm_positions = [0.1] * 6
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]

    report = run_folded_pose_return(make_api(driver), make_config(), timeout=3.0)

    assert report["passed"] is True
    assert driver.arm_mode_calls == ["position", "idle"]
    assert driver.gripper_mode_calls == []
    assert driver.arm_position_calls == [([0.0] * 6, 0.001, True)]
    assert report["tracking_errors_rad"] == [0.0] * 6
    assert driver.cleaned_up is True


def test_folded_pose_return_refuses_to_move_when_not_at_home() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="not at dataset_collection_home"):
        run_folded_pose_return(make_api(driver), make_config(), timeout=3.0)

    assert driver.arm_position_calls == []
    assert driver.cleaned_up is True


def test_cartesian_step_rejects_excessive_downward_motion() -> None:
    driver = FakeDriver()

    with pytest.raises(ValueError, match="exceeds"):
        run_cartesian_step_test(
            make_api(driver),
            make_config(),
            timeout=3.0,
            axis="z",
            distance_m=-0.01,
        )

    assert driver.configure_args is None


def test_cartesian_step_stops_before_cartesian_motion_when_home_is_not_reached() -> None:
    driver = FakeDriver()
    driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]
    driver.set_arm_positions = lambda *args: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Home tracking error"):
        run_cartesian_step_test(
            make_api(driver),
            make_config(),
            timeout=3.0,
            axis="z",
            distance_m=0.01,
        )

    assert driver.cartesian_position_calls == []
    assert driver.arm_mode_calls == ["position", "idle"]
    assert driver.cleaned_up is True


def test_text_reports_include_operator_summary() -> None:
    driver = FakeDriver()
    preflight = preflight_robot(make_api(driver), make_config(), timeout=3.0)
    preflight["captured_at"] = "2026-08-17T07:00:00+00:00"
    preflight_text = format_preflight_report(preflight)
    assert "Overall: PASS" in preflight_text
    assert "Joint state and limits" in preflight_text

    hold_driver = FakeDriver()
    hold_driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]
    hold = run_position_hold_test(make_api(hold_driver), make_config(), timeout=3.0)
    hold["captured_at"] = "2026-08-17T07:00:00+00:00"
    hold_text = format_position_hold_report(hold)
    assert "Overall: PASS" in hold_text
    assert "No position, velocity, or effort target was sent." in hold_text

    current_driver = FakeDriver()
    current_driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]
    current = run_current_position_hold_test(
        make_api(current_driver), make_config(), timeout=3.0
    )
    current["captured_at"] = "2026-08-17T07:00:00+00:00"
    current_text = format_current_position_hold_report(current)
    assert "Overall: PASS" in current_text
    assert "sent back unchanged as the target" in current_text
    assert "Failure reasons\n---------------\nNone" in current_text

    gripper_driver = FakeDriver()
    gripper_driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]
    gripper = run_gripper_cycle_test(
        make_api(gripper_driver), make_config(), timeout=3.0
    )
    gripper["captured_at"] = "2026-08-17T07:00:00+00:00"
    assert "Overall: PASS" in format_gripper_cycle_report(gripper)

    arm_driver = FakeDriver()
    arm_driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]
    arm = run_whole_arm_cycle_test(make_api(arm_driver), make_config(), timeout=3.0)
    arm["captured_at"] = "2026-08-17T07:00:00+00:00"
    assert "Overall: PASS" in format_whole_arm_cycle_report(arm)

    cartesian_driver = FakeDriver()
    cartesian_driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]
    cartesian = run_cartesian_step_test(
        make_api(cartesian_driver),
        make_config(),
        timeout=3.0,
        axis="z",
        distance_m=0.01,
    )
    cartesian["captured_at"] = "2026-08-17T07:00:00+00:00"
    assert "Overall: PASS" in format_cartesian_step_report(cartesian)

    folded_driver = FakeDriver()
    folded_driver.arm_positions = [0.1] * 6
    folded_driver.get_error_information = lambda: "No error"  # type: ignore[method-assign]
    folded = run_folded_pose_return(
        make_api(folded_driver), make_config(), timeout=3.0
    )
    folded["captured_at"] = "2026-08-17T07:00:00+00:00"
    assert "Overall: PASS" in format_folded_pose_return_report(folded)

    workspace = preflight_robot(make_api(FakeDriver()), make_config(), timeout=3.0)
    workspace["label"] = "task_center"
    workspace["captured_at"] = "2026-08-17T07:00:00+00:00"
    workspace_text = format_workspace_point_report(workspace)
    assert "Label: task_center" in workspace_text
    assert "sent no motion command" in workspace_text
