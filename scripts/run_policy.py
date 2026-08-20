#!/usr/bin/env python3
"""Run a trained dual-camera policy with safety-gated home staging."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import numpy as np

from tcc_real_robot.config import load_yaml
from tcc_real_robot.model_assets import resolve_model_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pinned backbone-policy pair from two RGB cameras with "
            "shadow mode or explicitly gated, bounded real-robot rollout."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robot.yaml"))
    parser.add_argument("--backbone", default="ours_rn50")
    parser.add_argument("--demonstrations", type=int, default=80)
    parser.add_argument(
        "--task",
        default="carrot",
        help="Configured task name, or carrot/pineapple/starfruit/strawberry.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("realsense-sdk", "v4l2"),
        default="realsense-sdk",
        help="Camera API. RealSense SDK is the default and safest for RGB streams.",
    )
    parser.add_argument(
        "--cam-main-serial",
        help="RealSense serial for cam_main; defaults to robot config.",
    )
    parser.add_argument(
        "--cam-wrist-serial",
        help="RealSense serial for cam_wrist; defaults to robot config.",
    )
    parser.add_argument(
        "--cam-main",
        help="Legacy V4L2 main camera index/path; only used with --camera-backend v4l2.",
    )
    parser.add_argument(
        "--cam-wrist",
        help="Legacy V4L2 wrist camera index/path; only used with --camera-backend v4l2.",
    )
    parser.add_argument(
        "--tcc-source-root",
        type=Path,
        default=Path("/home/robotarm/TCC-core"),
        help="TCC-Core checkout; defaults to the fixed robot-computer path.",
    )
    parser.add_argument("--hub-cache-dir", type=Path)
    parser.add_argument("--offline", dest="offline", action="store_true", default=True)
    parser.add_argument(
        "--online",
        dest="offline",
        action="store_false",
        help="Allow Hugging Face network access instead of using the local cache.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--camera-startup-delay", type=float, default=1.0)
    parser.add_argument("--camera-read-attempts", type=int, default=3)
    parser.add_argument("--camera-retry-delay", type=float, default=0.25)
    parser.add_argument("--camera-min-channel-std", type=float, default=2.0)
    parser.add_argument("--camera-max-pair-skew-ms", type=float, default=50.0)
    parser.add_argument(
        "--camera-read-mode",
        choices=("latest", "synchronous"),
        default="latest",
        help=(
            "Use background latest-frame capture to overlap 30 FPS acquisition "
            "with inference, or the legacy synchronous read path."
        ),
    )
    parser.add_argument("--inference-warmup-steps", type=int)
    parser.add_argument("--controller-timeout", type=float, default=20.0)
    parser.add_argument(
        "--execute-home",
        action="store_true",
        help=(
            "Move arm and gripper slowly to dataset_collection_home and hold "
            "there during shadow evaluation. Policy actions are not executed."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for a future safety-gated actuation implementation.",
    )
    parser.add_argument(
        "--execute-clipped-step",
        action="store_true",
        help=(
            "Execute a short policy rollout after clipping every command to "
            "small per-step and cumulative-from-home limits. Requires "
            "--execute-home and --emergency-stop-ready."
        ),
    )
    parser.add_argument(
        "--execute-policy",
        action="store_true",
        help=(
            "Run the fixed real-robot preset: dataset home, bounded policy "
            "actuation, and the configured full rollout length. Still requires "
            "--emergency-stop-ready."
        ),
    )
    parser.add_argument(
        "--emergency-stop-ready",
        action="store_true",
        help="Acknowledge that the physical emergency stop is immediately ready.",
    )
    return parser.parse_args()


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


class SynchronizedCameras:
    """Acquire a close-in-time frame pair using OpenCV grab/retrieve."""

    def __init__(
        self,
        cv2: Any,
        main_source: str,
        wrist_source: str,
        width: int,
        height: int,
        fps: float,
        *,
        startup_delay_s: float = 1.0,
        read_attempts: int = 3,
        retry_delay_s: float = 0.25,
        minimum_channel_std: float = 2.0,
        maximum_pair_skew_ms: float = 50.0,
    ) -> None:
        if startup_delay_s < 0:
            raise ValueError("startup_delay_s must be non-negative")
        if read_attempts <= 0:
            raise ValueError("read_attempts must be positive")
        if retry_delay_s < 0:
            raise ValueError("retry_delay_s must be non-negative")
        if minimum_channel_std < 0:
            raise ValueError("minimum_channel_std must be non-negative")
        if maximum_pair_skew_ms <= 0:
            raise ValueError("maximum_pair_skew_ms must be positive")
        self.cv2 = cv2
        self.read_attempts = read_attempts
        self.retry_delay_s = retry_delay_s
        self.minimum_channel_std = minimum_channel_std
        self.maximum_pair_skew_ms = maximum_pair_skew_ms
        self.last_pair_skew_ms: float | None = None
        self.main = cv2.VideoCapture(_camera_source(main_source), cv2.CAP_V4L2)
        self.wrist = cv2.VideoCapture(_camera_source(wrist_source), cv2.CAP_V4L2)
        for name, capture in (("cam_main", self.main), ("cam_wrist", self.wrist)):
            if not capture.isOpened():
                self.close()
                raise RuntimeError(f"Could not open {name}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.main_properties = self._properties(self.main)
        self.wrist_properties = self._properties(self.wrist)
        for name, properties in (
            ("cam_main", self.main_properties),
            ("cam_wrist", self.wrist_properties),
        ):
            actual_width = float(properties["width"])
            actual_height = float(properties["height"])
            actual_fps = float(properties["fps"])
            if (
                abs(actual_width - width) > 0.5
                or abs(actual_height - height) > 0.5
                or abs(actual_fps - fps) > 0.1
            ):
                self.close()
                raise RuntimeError(
                    f"{name} negotiated {actual_width:.0f}x{actual_height:.0f} "
                    f"@ {actual_fps:.3f} FPS, expected strict dataset profile "
                    f"{width}x{height} @ {fps:.3f} FPS"
                )
        if startup_delay_s:
            time.sleep(startup_delay_s)

    def _properties(self, capture: Any) -> dict[str, object]:
        fourcc_value = int(capture.get(self.cv2.CAP_PROP_FOURCC))
        fourcc = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
        )
        return {
            "width": capture.get(self.cv2.CAP_PROP_FRAME_WIDTH),
            "height": capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": capture.get(self.cv2.CAP_PROP_FPS),
            "fourcc": fourcc,
        }

    def read_rgb_pair(self) -> tuple[np.ndarray, np.ndarray]:
        last_status = "no attempts made"
        for attempt in range(1, self.read_attempts + 1):
            main_grabbed = bool(self.main.grab())
            main_grabbed_at = time.monotonic()
            wrist_grabbed = bool(self.wrist.grab())
            wrist_grabbed_at = time.monotonic()
            pair_skew_ms = abs(wrist_grabbed_at - main_grabbed_at) * 1000.0
            if main_grabbed and wrist_grabbed:
                main_ok, main_bgr = self.main.retrieve()
                wrist_ok, wrist_bgr = self.wrist.retrieve()
                main_retrieved = bool(main_ok and main_bgr is not None)
                wrist_retrieved = bool(wrist_ok and wrist_bgr is not None)
                if main_retrieved and wrist_retrieved:
                    main_rgb = self.cv2.cvtColor(main_bgr, self.cv2.COLOR_BGR2RGB)
                    wrist_rgb = self.cv2.cvtColor(wrist_bgr, self.cv2.COLOR_BGR2RGB)
                    main_std = np.std(main_rgb, axis=(0, 1))
                    wrist_std = np.std(wrist_rgb, axis=(0, 1))
                    main_valid = float(np.max(main_std)) >= self.minimum_channel_std
                    wrist_valid = (
                        float(np.max(wrist_std)) >= self.minimum_channel_std
                    )
                    skew_valid = pair_skew_ms <= self.maximum_pair_skew_ms
                    if main_valid and wrist_valid and skew_valid:
                        self.last_pair_skew_ms = pair_skew_ms
                        return main_rgb, wrist_rgb
                    last_status = (
                        f"attempt={attempt}, main_frame="
                        f"{'PASS' if main_valid else 'FLAT'}, wrist_frame="
                        f"{'PASS' if wrist_valid else 'FLAT'}, "
                        f"pair_skew_ms={pair_skew_ms:.3f}, "
                        f"pair_skew={'PASS' if skew_valid else 'FAIL'}, "
                        f"main_channel_std={main_std.round(3).tolist()}, "
                        f"wrist_channel_std={wrist_std.round(3).tolist()}"
                    )
                    if attempt < self.read_attempts and self.retry_delay_s:
                        time.sleep(self.retry_delay_s)
                    continue
                last_status = (
                    f"attempt={attempt}, main_grab=PASS, wrist_grab=PASS, "
                    f"main_retrieve={'PASS' if main_retrieved else 'FAIL'}, "
                    f"wrist_retrieve={'PASS' if wrist_retrieved else 'FAIL'}, "
                    f"pair_skew_ms={pair_skew_ms:.3f}"
                )
            else:
                last_status = (
                    f"attempt={attempt}, "
                    f"main_grab={'PASS' if main_grabbed else 'FAIL'}, "
                    f"wrist_grab={'PASS' if wrist_grabbed else 'FAIL'}, "
                    f"pair_skew_ms={pair_skew_ms:.3f}"
                )
            if attempt < self.read_attempts and self.retry_delay_s:
                time.sleep(self.retry_delay_s)
        raise RuntimeError(
            "Failed to read a synchronized camera pair after "
            f"{self.read_attempts} attempts ({last_status})"
        )

    def close(self) -> None:
        for capture in (getattr(self, "main", None), getattr(self, "wrist", None)):
            if capture is not None:
                capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def resolve_task(task: str, task_names: list[str]) -> tuple[int, str]:
    aliases = {
        "carrot": "pick_and_place_carrot_100",
        "pineapple": "pick_and_place_pineapple_100",
        "starfruit": "pick_and_place_starfruit_100",
        "strawberry": "pick_and_place_strawberry_100",
    }
    resolved = aliases.get(task.lower(), task)
    if resolved not in task_names:
        raise ValueError(f"Unknown task {task!r}; expected one of {task_names}")
    return task_names.index(resolved), resolved


def assert_shadow_only(robot_config: dict[str, Any], execute: bool) -> None:
    """Fail closed until action semantics and workspace limits are certified."""
    if not execute:
        return
    contract = robot_config.get("action_contract", {})
    workspace = robot_config.get("safety", {}).get("workspace_limits")
    reasons = []
    if contract.get("enabled") is not True or contract.get("semantics") == "UNVERIFIED":
        reasons.append("the 7-D action contract is not enabled and verified")
    if workspace in (None, "CALIBRATING"):
        reasons.append("workspace limits are still calibrating")
    detail = "; ".join(reasons) or "the actuation path has not been implemented"
    raise RuntimeError(f"Refusing --execute: {detail}. Run shadow evaluation first.")


def main() -> None:
    args = parse_args()
    if args.demonstrations <= 0:
        raise SystemExit("--demonstrations must be positive")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")
    if args.camera_startup_delay < 0:
        raise SystemExit("--camera-startup-delay must be non-negative")
    if args.camera_read_attempts <= 0:
        raise SystemExit("--camera-read-attempts must be positive")
    if args.camera_retry_delay < 0:
        raise SystemExit("--camera-retry-delay must be non-negative")
    if args.camera_min_channel_std < 0:
        raise SystemExit("--camera-min-channel-std must be non-negative")
    if args.camera_max_pair_skew_ms <= 0:
        raise SystemExit("--camera-max-pair-skew-ms must be positive")
    if args.controller_timeout <= 0:
        raise SystemExit("--controller-timeout must be positive")
    config = load_yaml(args.config)
    robot_config = load_yaml(args.robot_config)
    if args.execute_policy:
        args.execute_home = True
        args.execute_clipped_step = True
        if args.max_steps is None:
            args.max_steps = int(config["evaluation"]["max_rollout_steps"])
    if args.camera_backend == "realsense-sdk":
        configured_cameras = robot_config["cameras"]
        args.cam_main_serial = args.cam_main_serial or str(
            configured_cameras["cam_main"]["serial_number"]
        )
        args.cam_wrist_serial = args.cam_wrist_serial or str(
            configured_cameras["cam_wrist"]["serial_number"]
        )
        if args.cam_main_serial == args.cam_wrist_serial:
            raise SystemExit("Main and wrist RealSense serials must be distinct")
    elif not args.cam_main or not args.cam_wrist:
        raise SystemExit(
            "--camera-backend v4l2 requires --cam-main and --cam-wrist"
        )
    assert_shadow_only(robot_config, args.execute)
    if args.execute_clipped_step:
        if not args.execute_home:
            raise SystemExit("--execute-clipped-step requires --execute-home")
        clipped_settings = robot_config["policy_evaluation"]["clipped_rollout"]
        clipped_max_steps = int(clipped_settings["max_steps"])
        if args.max_steps is None or not 1 <= args.max_steps <= clipped_max_steps:
            raise SystemExit(
                "--execute-clipped-step requires --max-steps within "
                f"[1, {clipped_max_steps}]"
            )
        if not args.emergency_stop_ready:
            raise SystemExit(
                "--execute-clipped-step requires --emergency-stop-ready"
            )

    from tcc_real_robot.policy_runtime import (
        load_policy_bundle,
        predict_action,
        resolve_device,
    )
    from tcc_real_robot.tcc_backbone import load_frozen_tcc_backbone

    device = resolve_device(args.device)
    task_names = [str(value) for value in config["dataset"]["tasks"]]
    task_index, task_name = resolve_task(args.task, task_names)
    dataset_rollout_steps = int(config["evaluation"]["max_rollout_steps"])
    max_allowed_steps = (
        int(robot_config["policy_evaluation"]["clipped_rollout"]["max_steps"])
        if args.execute_clipped_step
        else dataset_rollout_steps
    )
    max_steps = args.max_steps or dataset_rollout_steps
    if not 1 <= max_steps <= max_allowed_steps:
        raise ValueError(f"--max-steps must be within [1, {max_allowed_steps}]")
    fps = float(config["observations"]["fps"])
    evaluation_settings = robot_config["policy_evaluation"]
    camera_capture_fps = float(evaluation_settings["camera_capture_fps"])
    if camera_capture_fps <= 0:
        raise ValueError("policy_evaluation.camera_capture_fps must be positive")
    inference_warmup_steps = (
        args.inference_warmup_steps
        if args.inference_warmup_steps is not None
        else int(evaluation_settings["inference_warmup_steps"])
    )
    if inference_warmup_steps < 0:
        raise ValueError("--inference-warmup-steps must be non-negative")
    width, height = [int(value) for value in config["observations"]["resolution"]]
    assets = resolve_model_assets(
        config,
        args.backbone,
        args.demonstrations,
        cache_dir=args.hub_cache_dir,
        local_files_only=args.offline,
    )
    backbone, backbone_metadata = load_frozen_tcc_backbone(
        assets.backbone_path, args.tcc_source_root, device
    )
    bundle = load_policy_bundle(
        assets.policy_path,
        expected_feature_dim=int(backbone_metadata["feature_dim"]),
        device=device,
    )
    checkpoint_policy_config = bundle.config["policy"]
    action_representation = checkpoint_policy_config.get(
        "action_representation", "absolute"
    )
    runtime_execution_delta_gain: float | None = None
    if action_representation == "future_delta":
        checkpoint_lookahead = int(checkpoint_policy_config["lookahead_frames"])
        configured_lookahead = int(config["policy"]["lookahead_frames"])
        if checkpoint_lookahead != configured_lookahead:
            raise RuntimeError(
                "Checkpoint/config lookahead mismatch: "
                f"{checkpoint_lookahead} != {configured_lookahead}"
            )
        runtime_execution_delta_gain = float(
            config["policy"]["execution_delta_gain"]
        )
        if not 0.0 < runtime_execution_delta_gain <= 1.0:
            raise ValueError("Runtime execution_delta_gain must be in (0, 1]")
    policy_uses_proprioception = bundle.model.proprio_dim > 0
    if policy_uses_proprioception and not args.execute_home:
        raise RuntimeError(
            "This policy requires robot state; run with --execute-home or "
            "--execute-policy"
        )
    trained_tasks = [str(value) for value in bundle.config["dataset"]["tasks"]]
    if trained_tasks != task_names:
        raise RuntimeError(
            f"Checkpoint task order {trained_tasks} differs from config {task_names}"
        )
    expected_resolution = (height, width, 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_kind = (
        "policy_clipped_rollout" if args.execute_clipped_step else "policy_shadow"
    )
    output_path = args.output_dir / f"{output_kind}_{args.backbone}_{stamp}.txt"
    started = time.monotonic()
    rollout_started: float | None = None
    completed = 0
    decision = "FAIL"
    exit_ok = False
    failure = ""
    latencies_ms: list[float] = []
    camera_latencies_ms: list[float] = []
    state_latencies_ms: list[float] = []
    policy_latencies_ms: list[float] = []
    command_latencies_ms: list[float] = []
    pair_skews_ms: list[float] = []
    home_session: Any | None = None
    cameras: Any | None = None
    home_reference: list[float] | None = None
    first_arm_delta: float | None = None
    first_gripper_delta: float | None = None
    bounded_steps: list[Any] = []
    final_verification: Any | None = None
    previous_policy_sample: tuple[float, tuple[float, ...]] | None = None
    max_observed_arm_velocity = 0.0
    max_observed_gripper_velocity = 0.0
    dataset_action_min: list[float] | None = None
    dataset_action_max: list[float] | None = None
    if args.execute_clipped_step:
        task_limits = evaluation_settings["clipped_rollout"][
            "dataset_action_limits"
        ].get(task_name)
        if task_limits is None:
            raise ValueError(f"No dataset action limits configured for {task_name}")
        dataset_action_min = [float(value) for value in task_limits["min"]]
        dataset_action_max = [float(value) for value in task_limits["max"]]
        if len(dataset_action_min) != 7 or len(dataset_action_max) != 7:
            raise ValueError("Dataset action limits must contain seven values")

    with output_path.open("w", encoding="utf-8", buffering=1) as report:
        report.write("TCC Real-Robot Policy Evaluation\n")
        report.write("=======================================\n")
        if args.execute_clipped_step:
            report.write(
                "Mode: CLIPPED ROLLOUT "
                "(bounded per-step and by dataset action envelope)\n"
            )
        else:
            report.write("Mode: SHADOW (policy predictions are never actuated)\n")
        report.write(f"Home staging: {'ENABLED' if args.execute_home else 'NOT RUN'}\n")
        report.write(f"Task: {task_name} (index {task_index})\n")
        report.write(f"Backbone: {args.backbone}\n")
        report.write(f"Demonstrations: {args.demonstrations}\n")
        report.write(f"Training step: {bundle.step}\n")
        report.write(
            "Policy proprioception: "
            f"{'ENABLED' if policy_uses_proprioception else 'DISABLED'}\n"
        )
        report.write(
            "Policy action representation: "
            f"{action_representation}\n"
        )
        if runtime_execution_delta_gain is not None:
            report.write(
                "Checkpoint future-delta gain: "
                f"{float(checkpoint_policy_config.get('execution_delta_gain', 0.0)):.6f}\n"
            )
            report.write(
                "Runtime future-delta gain: "
                f"{runtime_execution_delta_gain:.6f}\n"
            )
        report.write(f"Hub repository: {assets.repository}\n")
        report.write(f"Hub revision: {assets.revision}\n")
        report.write(f"Backbone SHA256: {assets.backbone_sha256}\n")
        report.write(f"Policy SHA256: {assets.policy_sha256}\n")
        report.write(f"Device: {device}\n")
        report.write(f"Camera backend: {args.camera_backend}\n")
        report.write(f"Camera read mode: {args.camera_read_mode}\n")
        if args.camera_backend == "realsense-sdk":
            report.write(f"Camera main serial: {args.cam_main_serial}\n")
            report.write(f"Camera wrist serial: {args.cam_wrist_serial}\n")
        else:
            report.write(f"Camera main: {args.cam_main}\n")
            report.write(f"Camera wrist: {args.cam_wrist}\n")
        report.write(
            f"Camera capture rate: {camera_capture_fps:.3f} FPS "
            f"(validated {args.camera_backend} color profile)\n"
        )
        report.write(f"Camera startup delay: {args.camera_startup_delay:.3f} s\n")
        report.write(f"Camera read attempts: {args.camera_read_attempts}\n")
        report.write(f"Camera retry delay: {args.camera_retry_delay:.3f} s\n")
        report.write(
            f"Camera minimum channel std: {args.camera_min_channel_std:.3f}\n"
        )
        report.write(
            f"Camera maximum pair skew: {args.camera_max_pair_skew_ms:.3f} ms\n"
        )
        report.write(f"Policy rollout rate: {fps:.3f} Hz\n")
        if args.execute_clipped_step:
            clipped = evaluation_settings["clipped_rollout"]
            goal_time = float(clipped["min_time_to_move_multiplier"]) / float(
                clipped["control_fps"]
            )
            report.write("Policy command blocking: False (official continuous mode)\n")
            report.write(f"Policy command goal time: {goal_time:.3f} s\n")
            report.write(
                "Policy command shaping: stateful previous-command slew with "
                "measured-position lead cap\n"
            )
            report.write(
                f"Policy maximum command lead: {clipped['max_command_lead']}\n"
            )
        report.write(f"Inference warmup steps: {inference_warmup_steps}\n")
        report.write(f"Maximum steps: {max_steps}\n\n")
        if dataset_action_min is not None and dataset_action_max is not None:
            report.write(f"Dataset action minimum: {dataset_action_min}\n")
            report.write(f"Dataset action maximum: {dataset_action_max}\n\n")
        period = 1.0 / fps
        try:
            if args.execute_home:
                try:
                    import trossen_arm
                except ImportError as exc:
                    raise RuntimeError(
                        "Trossen driver is missing; install with pip install -e "
                        "'.[robot]'"
                    ) from exc
                from tcc_real_robot.policy_home import PolicyHomeSession

                print(
                    "Moving arm and gripper to dataset_collection_home. "
                    + (
                        "A short clipped policy rollout may follow."
                        if args.execute_clipped_step
                        else "Policy actions remain shadow-only."
                    )
                )
                home_session = PolicyHomeSession(
                    trossen_arm, robot_config, args.controller_timeout
                )
                preparation = home_session.prepare()
                home_reference = list(preparation.observed)
                report.write("Home preparation\n")
                report.write(f"Driver: {preparation.driver_version}\n")
                report.write(f"Firmware: {preparation.firmware_version}\n")
                report.write(
                    "Target: ["
                    + ", ".join(f"{value:.7f}" for value in preparation.target)
                    + "]\n"
                )
                report.write(
                    "Observed: ["
                    + ", ".join(f"{value:.7f}" for value in preparation.observed)
                    + "]\n"
                )
                report.write(
                    f"Maximum arm tracking error: "
                    f"{preparation.max_arm_error_rad:.7f} rad\n"
                )
                report.write(
                    f"Gripper tracking error: {preparation.gripper_error_m:.7f} m\n\n"
                )

            report.write("Predicted denormalized absolute actions\n")
            if args.camera_backend == "realsense-sdk":
                try:
                    import pyrealsense2 as rs
                except ImportError as exc:
                    raise RuntimeError(
                        "pyrealsense2 is required for --camera-backend "
                        "realsense-sdk; install the robot extras"
                    ) from exc
                from tcc_real_robot.realsense_cameras import RealSenseColorCameras

                if not camera_capture_fps.is_integer():
                    raise RuntimeError(
                        "RealSense SDK capture FPS must be a whole number"
                    )
                camera_context: Any = RealSenseColorCameras(
                    rs,
                    args.cam_main_serial,
                    args.cam_wrist_serial,
                    width,
                    height,
                    int(camera_capture_fps),
                    timeout_ms=max(1, int(args.controller_timeout * 1000)),
                    read_attempts=args.camera_read_attempts,
                    minimum_channel_std=args.camera_min_channel_std,
                    maximum_pair_skew_ms=args.camera_max_pair_skew_ms,
                )
            else:
                import cv2

                camera_context = SynchronizedCameras(
                    cv2,
                    args.cam_main,
                    args.cam_wrist,
                    width,
                    height,
                    camera_capture_fps,
                    startup_delay_s=args.camera_startup_delay,
                    read_attempts=args.camera_read_attempts,
                    retry_delay_s=args.camera_retry_delay,
                    minimum_channel_std=args.camera_min_channel_std,
                    maximum_pair_skew_ms=args.camera_max_pair_skew_ms,
                )
            if args.camera_read_mode == "latest":
                from tcc_real_robot.camera_buffer import LatestFramePairBuffer

                camera_context = LatestFramePairBuffer(
                    camera_context,
                    timeout_s=2.0,
                )
            with camera_context as cameras:
                report.write(f"Camera main negotiated: {cameras.main_properties}\n")
                report.write(
                    f"Camera wrist negotiated: {cameras.wrist_properties}\n\n"
                )
                for _ in range(args.warmup_frames):
                    cameras.read_rgb_pair()
                for _ in range(inference_warmup_steps):
                    warm_main, warm_wrist = cameras.read_rgb_pair()
                    warm_state = (
                        home_session.read_positions()
                        if policy_uses_proprioception and home_session is not None
                        else None
                    )
                    predict_action(
                        backbone,
                        bundle,
                        warm_main,
                        warm_wrist,
                        task_index,
                        int(backbone_metadata["image_size"]),
                        device,
                        observation_state=warm_state,
                        execution_delta_gain_override=runtime_execution_delta_gain,
                    )
                if home_session is not None:
                    home_reference = home_session.read_positions()
                rollout_started = time.monotonic()
                for step in range(max_steps):
                    deadline = rollout_started + step * period
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    iteration_started = time.monotonic()
                    camera_started = time.monotonic()
                    main_rgb, wrist_rgb = cameras.read_rgb_pair()
                    camera_latencies_ms.append(
                        (time.monotonic() - camera_started) * 1000.0
                    )
                    if cameras.last_pair_skew_ms is None:
                        raise RuntimeError("Camera pair skew was not recorded")
                    pair_skews_ms.append(cameras.last_pair_skew_ms)
                    if main_rgb.shape != expected_resolution:
                        raise RuntimeError(
                            f"cam_main shape {main_rgb.shape} != {expected_resolution}"
                        )
                    if wrist_rgb.shape != expected_resolution:
                        raise RuntimeError(
                            f"cam_wrist shape {wrist_rgb.shape} != {expected_resolution}"
                        )
                    state_started = time.monotonic()
                    policy_state = None
                    if policy_uses_proprioception and home_session is not None:
                        policy_state = home_session.read_positions()
                    state_latencies_ms.append(
                        (time.monotonic() - state_started) * 1000.0
                    )
                    policy_started = time.monotonic()
                    action = predict_action(
                        backbone,
                        bundle,
                        main_rgb,
                        wrist_rgb,
                        task_index,
                        int(backbone_metadata["image_size"]),
                        device,
                        observation_state=policy_state,
                        execution_delta_gain_override=runtime_execution_delta_gain,
                    )
                    policy_latencies_ms.append(
                        (time.monotonic() - policy_started) * 1000.0
                    )
                    completed += 1
                    elapsed = time.monotonic() - rollout_started
                    inference_ms = (time.monotonic() - iteration_started) * 1000.0
                    latencies_ms.append(inference_ms)
                    if step == 0 and home_reference is not None:
                        first_arm_delta = max(
                            abs(predicted - current)
                            for predicted, current in zip(
                                action[:6].tolist(),
                                home_reference[:6],
                                strict=True,
                            )
                        )
                        first_gripper_delta = abs(float(action[6]) - home_reference[6])
                    if args.execute_clipped_step:
                        if home_session is None or home_reference is None:
                            raise RuntimeError(
                                "Clipped actuation requires a prepared home reference"
                            )
                        command_started = time.monotonic()
                        bounded_step = home_session.execute_bounded_policy_step(
                            [float(value) for value in action.tolist()],
                            home_reference,
                            absolute_min=dataset_action_min,
                            absolute_max=dataset_action_max,
                        )
                        command_latencies_ms.append(
                            (time.monotonic() - command_started) * 1000.0
                        )
                        bounded_steps.append(bounded_step)
                        if previous_policy_sample is not None:
                            previous_time, previous_positions = previous_policy_sample
                            sample_period = (
                                bounded_step.sampled_at_monotonic - previous_time
                            )
                            if sample_period <= 0:
                                raise RuntimeError("Policy state timestamps did not advance")
                            observed_arm_velocity = max(
                                abs(current - previous) / sample_period
                                for current, previous in zip(
                                    bounded_step.start[:6],
                                    previous_positions[:6],
                                    strict=True,
                                )
                            )
                            observed_gripper_velocity = (
                                abs(bounded_step.start[6] - previous_positions[6])
                                / sample_period
                            )
                            max_observed_arm_velocity = max(
                                max_observed_arm_velocity, observed_arm_velocity
                            )
                            max_observed_gripper_velocity = max(
                                max_observed_gripper_velocity,
                                observed_gripper_velocity,
                            )
                            if observed_arm_velocity > float(
                                robot_config["safety"]["max_joint_velocity_rad_s"]
                            ):
                                raise RuntimeError(
                                    "Observed arm velocity "
                                    f"{observed_arm_velocity:.6f} rad/s exceeds limit"
                                )
                            if observed_gripper_velocity > float(
                                robot_config["safety"]["max_gripper_velocity_m_s"]
                            ):
                                raise RuntimeError(
                                    "Observed gripper velocity "
                                    f"{observed_gripper_velocity:.6f} m/s exceeds limit"
                                )
                        previous_policy_sample = (
                            bounded_step.sampled_at_monotonic,
                            bounded_step.start,
                        )
                        commanded_values = ", ".join(
                            f"{value:.7f}" for value in bounded_step.commanded
                        )
                        observed_values = ", ".join(
                            f"{value:.7f}" for value in bounded_step.observed
                        )
                        report.write(
                            f"step={step:03d} commanded_clipped=[{commanded_values}]\n"
                            f"step={step:03d} observed_after_command="
                            f"[{observed_values}]\n"
                            f"step={step:03d} commanded_max_arm_delta_rad="
                            f"{bounded_step.max_commanded_arm_delta_rad:.7f}\n"
                            f"step={step:03d} commanded_gripper_delta_m="
                            f"{bounded_step.commanded_gripper_delta_m:.7f}\n"
                            f"step={step:03d} commanded_max_arm_lead_rad="
                            f"{bounded_step.max_arm_command_lead_rad:.7f}\n"
                            f"step={step:03d} commanded_gripper_lead_m="
                            f"{bounded_step.gripper_command_lead_m:.7f}\n"
                            f"step={step:03d} arm_immediate_command_gap_rad="
                            f"{bounded_step.max_arm_command_gap_rad:.7f}\n"
                            f"step={step:03d} gripper_immediate_command_gap_m="
                            f"{bounded_step.gripper_command_gap_m:.7f}\n"
                        )
                    values = ", ".join(f"{value:.7f}" for value in action.tolist())
                    report.write(
                        f"step={step:03d} elapsed_s={elapsed:.6f} "
                        f"inference_ms={inference_ms:.3f} "
                        f"pair_skew_ms={cameras.last_pair_skew_ms:.3f} "
                        f"action=[{values}]\n"
                    )
            if args.execute_clipped_step and bounded_steps:
                if home_session is None:
                    raise RuntimeError("Policy session closed before final verification")
                final_verification = home_session.settle_and_verify_policy_target(
                    list(bounded_steps[-1].commanded)
                )
                report.write("\nFinal settled target verification\n")
                report.write(
                    "Observed: ["
                    + ", ".join(
                        f"{value:.7f}" for value in final_verification.observed
                    )
                    + "]\n"
                )
                report.write(
                    "Maximum arm tracking error: "
                    f"{final_verification.max_arm_tracking_error_rad:.7f} rad\n"
                )
                report.write(
                    "Gripper tracking error: "
                    f"{final_verification.gripper_tracking_error_m:.7f} m\n"
                )
            if home_session is not None:
                home_session.close()
                home_session = None
        except KeyboardInterrupt:
            failure = "Interrupted by operator"
        except Exception as exc:  # noqa: BLE001 - preserve a hardware-side report
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if home_session is not None:
                try:
                    home_session.close()
                except Exception as exc:  # noqa: BLE001 - report cleanup failures
                    cleanup_failure = (
                        f"Home cleanup failed: {type(exc).__name__}: {exc}"
                    )
                    failure = (
                        f"{failure}; {cleanup_failure}" if failure else cleanup_failure
                    )
            timing_origin = rollout_started if rollout_started is not None else started
            elapsed = time.monotonic() - timing_origin
            observed_rate = completed / elapsed if elapsed > 0 else 0.0
            latency_median = (
                float(np.median(latencies_ms)) if latencies_ms else float("nan")
            )
            latency_p95 = (
                float(np.percentile(latencies_ms, 95)) if latencies_ms else float("nan")
            )
            latency_max = max(latencies_ms, default=float("nan"))
            pair_skew_median = (
                float(np.median(pair_skews_ms)) if pair_skews_ms else float("nan")
            )
            pair_skew_max = max(pair_skews_ms, default=float("nan"))
            minimum_rate_hz = float(evaluation_settings["minimum_observed_rate_hz"])
            if args.execute_clipped_step:
                inference_rate = (
                    1000.0 / latency_max
                    if np.isfinite(latency_max) and latency_max > 0
                    else 0.0
                )
                if len(bounded_steps) > 1:
                    command_span = (
                        bounded_steps[-1].sampled_at_monotonic
                        - bounded_steps[0].sampled_at_monotonic
                    )
                    control_rate = (
                        (len(bounded_steps) - 1) / command_span
                        if command_span > 0
                        else 0.0
                    )
                else:
                    control_rate = 0.0
                minimum_rate_met = control_rate >= minimum_rate_hz
            else:
                inference_rate = observed_rate
                control_rate = observed_rate
                minimum_rate_met = observed_rate >= minimum_rate_hz
            checks = {
                "all_steps_completed": completed == max_steps,
                "minimum_rate_met": minimum_rate_met,
                "home_staging_completed": home_reference is not None,
                "camera_pair_skew_safe": len(pair_skews_ms) == completed
                and completed > 0
                and pair_skew_max <= args.camera_max_pair_skew_ms,
                "first_arm_delta_safe": first_arm_delta is not None
                and first_arm_delta
                <= float(robot_config["safety"]["max_joint_delta_rad"]),
                "first_gripper_delta_safe": first_gripper_delta is not None
                and first_gripper_delta
                <= float(robot_config["safety"]["max_gripper_delta_m"]),
            }
            if args.execute_clipped_step:
                clipped = evaluation_settings["clipped_rollout"]
                checks.update(
                    {
                        "clipped_rollout_completed": len(bounded_steps) == completed
                        and completed == max_steps,
                        "commanded_arm_delta_safe": bool(bounded_steps)
                        and max(
                            item.max_commanded_arm_delta_rad for item in bounded_steps
                        )
                        <= max(float(value) for value in clipped["max_action_delta"][:6])
                        + 1e-9,
                        "commanded_gripper_delta_safe": bool(bounded_steps)
                        and max(item.commanded_gripper_delta_m for item in bounded_steps)
                        <= float(clipped["max_action_delta"][6]) + 1e-9,
                        "command_lead_safe": bool(bounded_steps)
                        and all(
                            item.max_arm_command_lead_rad
                            <= max(
                                float(value)
                                for value in clipped["max_command_lead"][:6]
                            )
                            + 1e-9
                            and item.gripper_command_lead_m
                            <= float(clipped["max_command_lead"][6]) + 1e-9
                            for item in bounded_steps
                        ),
                        "official_nonblocking_commands": clipped["command_blocking"]
                        is False,
                        "observed_arm_velocity_safe": max_observed_arm_velocity
                        <= float(
                            robot_config["safety"]["max_joint_velocity_rad_s"]
                        ),
                        "observed_gripper_velocity_safe": max_observed_gripper_velocity
                        <= float(
                            robot_config["safety"]["max_gripper_velocity_m_s"]
                        ),
                        "final_tracking_safe": final_verification is not None,
                    }
                )
            if failure:
                decision = "FAIL"
            elif args.execute_clipped_step:
                execution_checks = (
                    "all_steps_completed",
                    "home_staging_completed",
                    "camera_pair_skew_safe",
                    "minimum_rate_met",
                    "clipped_rollout_completed",
                    "commanded_arm_delta_safe",
                    "commanded_gripper_delta_safe",
                    "command_lead_safe",
                    "official_nonblocking_commands",
                    "observed_arm_velocity_safe",
                    "observed_gripper_velocity_safe",
                    "final_tracking_safe",
                )
                if all(checks[name] for name in execution_checks):
                    decision = "CLIPPED_ROLLOUT_COMPLETE_RAW_POLICY_BLOCKED"
                    exit_ok = True
                else:
                    decision = "BLOCKED"
            elif all(checks.values()):
                decision = "PASS"
                exit_ok = True
            else:
                decision = "BLOCKED"
            report.write("\nSummary\n")
            report.write(f"Completed steps: {completed}/{max_steps}\n")
            if args.execute_clipped_step:
                report.write(f"End-to-end elapsed including motion: {elapsed:.6f} s\n")
                report.write(
                    f"Single-step inference-only rate: {inference_rate:.3f} Hz "
                    "(not a sustained-rate measurement)\n"
                )
                report.write(f"Observed command rate: {control_rate:.3f} Hz\n")
                report.write(
                    "Maximum observed arm velocity: "
                    f"{max_observed_arm_velocity:.6f} rad/s\n"
                )
                report.write(
                    "Maximum observed gripper velocity: "
                    f"{max_observed_gripper_velocity:.6f} m/s\n"
                )
            else:
                report.write(f"Elapsed: {elapsed:.6f} s\n")
                report.write(f"Observed rate: {observed_rate:.3f} Hz\n")
            report.write(f"Latency median: {latency_median:.3f} ms\n")
            report.write(f"Latency p95: {latency_p95:.3f} ms\n")
            report.write(f"Latency maximum: {latency_max:.3f} ms\n")
            for name, values in (
                ("Camera-read", camera_latencies_ms),
                ("Robot-state", state_latencies_ms),
                ("Policy", policy_latencies_ms),
                ("Command", command_latencies_ms),
            ):
                if values:
                    report.write(
                        f"{name} latency median/p95: "
                        f"{np.median(values):.3f} / "
                        f"{np.percentile(values, 95):.3f} ms\n"
                    )
            captured_pairs = getattr(cameras, "captured_pairs", None)
            dropped_pairs = getattr(cameras, "dropped_pairs", None)
            if captured_pairs is not None and dropped_pairs is not None:
                report.write(f"Camera pairs captured: {captured_pairs}\n")
                report.write(f"Camera pairs superseded: {dropped_pairs}\n")
            report.write(f"Camera pair skew median: {pair_skew_median:.3f} ms\n")
            report.write(f"Camera pair skew maximum: {pair_skew_max:.3f} ms\n")
            if first_arm_delta is not None:
                report.write(
                    f"First raw-policy action maximum arm delta: "
                    f"{first_arm_delta:.7f} rad\n"
                )
            if first_gripper_delta is not None:
                report.write(
                    f"First raw-policy action gripper delta: "
                    f"{first_gripper_delta:.7f} m\n"
                )
            if bounded_steps:
                report.write(
                    "Maximum commanded arm lead: "
                    f"{max(item.max_arm_command_lead_rad for item in bounded_steps):.7f} "
                    "rad\n"
                )
                report.write(
                    "Maximum commanded gripper lead: "
                    f"{max(item.gripper_command_lead_m for item in bounded_steps):.7f} "
                    "m\n"
                )
            report.write("Checks:\n")
            for name, passed in checks.items():
                report.write(f"- {name}: {'PASS' if passed else 'FAIL'}\n")
            if failure:
                report.write(f"Failure: {failure}\n")
            if args.execute_clipped_step:
                report.write("Raw continuous policy release: BLOCKED\n")
            report.write(f"Decision: {decision}\n")

    print(output_path.read_text(encoding="utf-8"), end="")
    print(f"report: {output_path}")
    if not exit_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
