from types import SimpleNamespace

import pytest

from tcc_real_robot.policy_home import PolicyHomeSession


class FakeHomeDriver:
    def __init__(self) -> None:
        self.modes = [0] * 7
        self.arm = [0.0] * 6
        self.gripper = 0.02
        self.arm_commands: list[tuple[list[float], float, bool]] = []
        self.gripper_commands: list[tuple[float, float, bool]] = []
        self.cleaned = False

    def configure(self, *args: object) -> None:
        self.configure_args = args

    def get_driver_version(self) -> str:
        return "v1.9.3"

    def get_controller_version(self) -> str:
        return "v1.9.2"

    def get_error_information(self) -> str:
        return "No error"

    def get_modes(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(value=value) for value in self.modes]

    def get_joint_limits(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(position_min=-3.2, position_max=3.2) for _ in range(6)
        ] + [SimpleNamespace(position_min=0.0, position_max=0.04)]

    def set_arm_modes(self, mode: str) -> None:
        self.modes[:6] = [1 if mode == "position" else 0] * 6

    def set_gripper_mode(self, mode: str) -> None:
        self.modes[6] = 1 if mode == "position" else 0

    def set_arm_positions(
        self, target: list[float], goal_time: float, blocking: bool
    ) -> None:
        self.arm_commands.append((target.copy(), goal_time, blocking))
        self.arm = target.copy()

    def set_gripper_position(
        self, target: float, goal_time: float, blocking: bool
    ) -> None:
        self.gripper_commands.append((target, goal_time, blocking))
        self.gripper = target

    def get_all_positions(self) -> list[float]:
        return self.arm + [self.gripper]

    def cleanup(self, reboot_controller: bool) -> None:
        assert reboot_controller is False
        self.cleaned = True


def make_api(driver: FakeHomeDriver) -> SimpleNamespace:
    return SimpleNamespace(
        Model=SimpleNamespace(wxai_v0="model"),
        StandardEndEffector=SimpleNamespace(wxai_v0_follower="end-effector"),
        Mode=SimpleNamespace(position="position", idle="idle"),
        TrossenArmDriver=lambda: driver,
    )


def make_config() -> dict:
    return {
        "robot": {
            "driver_model": "wxai_v0",
            "end_effector": "wxai_v0_follower",
            "controller_ip": "192.168.1.4",
            "expected_driver_series": "1.9",
            "expected_firmware_series": "1.9",
            "home_arm_positions_rad": [0.0, 1.0, 0.5, 0.6, 0.0, 0.0],
            "home_gripper_position_m": 0.0,
        },
        "policy_evaluation": {
            "home_goal_time_s": 10.0,
            "gripper_goal_time_s": 2.0,
            "max_home_tracking_error_rad": 0.03,
            "max_home_gripper_error_m": 0.002,
        },
    }


def test_prepare_home_holds_then_restores_idle() -> None:
    driver = FakeHomeDriver()
    session = PolicyHomeSession(make_api(driver), make_config(), timeout=20.0)

    result = session.prepare()

    assert result.observed == result.target
    assert result.max_arm_error_rad == 0.0
    assert driver.modes == [1] * 7
    assert driver.arm_commands[0][1:] == (10.0, True)
    assert driver.gripper_commands == [(0.0, 2.0, True)]

    session.close()
    assert driver.modes == [0] * 7
    assert driver.cleaned is True


def test_prepare_home_rejects_non_idle_start() -> None:
    driver = FakeHomeDriver()
    driver.modes[0] = 1
    session = PolicyHomeSession(make_api(driver), make_config(), timeout=20.0)

    with pytest.raises(RuntimeError, match="must start idle"):
        session.prepare()

    assert driver.cleaned is True
