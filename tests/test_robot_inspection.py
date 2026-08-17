from types import SimpleNamespace

import pytest

from tcc_real_robot.robot_inspection import inspect_robot, monitor_robot, preflight_robot


class FakeDriver:
    def __init__(self) -> None:
        self.configure_args = None
        self.cleaned_up = False

    def configure(self, *args: object) -> None:
        self.configure_args = args

    def get_driver_version(self) -> str:
        return "v1.9.0"

    def get_controller_version(self) -> str:
        return "v1.9.2"

    def get_num_joints(self) -> int:
        return 7

    def get_modes(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(value="idle")] * 7

    def get_arm_positions(self) -> list[float]:
        return [0.0] * 6

    def get_gripper_position(self) -> float:
        return 0.02

    def get_all_positions(self) -> list[float]:
        return [0.0] * 6 + [0.02]

    def get_all_rotor_temperatures(self) -> list[float]:
        return [30.0] * 7

    def get_all_driver_temperatures(self) -> list[float]:
        return [31.0] * 7

    def get_cartesian_positions(self) -> list[float]:
        return [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]

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

    def cleanup(self, reboot_controller: bool) -> None:
        assert reboot_controller is False
        self.cleaned_up = True


def make_api(driver: FakeDriver) -> SimpleNamespace:
    return SimpleNamespace(
        Model=SimpleNamespace(wxai_v0="model"),
        StandardEndEffector=SimpleNamespace(wxai_v0_follower="end-effector"),
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
        }
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


def test_preflight_rejects_position_outside_limits() -> None:
    driver = FakeDriver()
    driver.get_all_positions = lambda: [4.0] + [0.0] * 6  # type: ignore[method-assign]

    report = preflight_robot(make_api(driver), make_config(), timeout=3.0)

    assert report["passed"] is False
    assert report["checks"]["positions_within_limits"] is False
    assert driver.cleaned_up is True
