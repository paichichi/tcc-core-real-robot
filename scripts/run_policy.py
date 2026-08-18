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
            "Evaluate a pinned backbone-policy pair from two RGB cameras. "
            "Policy predictions remain shadow-only; an explicit option can "
            "move the robot to dataset home before evaluation."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robot.yaml"))
    parser.add_argument("--backbone", default="ours_rn50")
    parser.add_argument("--demonstrations", type=int, default=60)
    parser.add_argument(
        "--task",
        required=True,
        help="Configured task name, or carrot/pineapple/starfruit/strawberry.",
    )
    parser.add_argument(
        "--cam-main",
        required=True,
        help="Main camera index/path, for example 0 or /dev/v4l/by-id/...",
    )
    parser.add_argument(
        "--cam-wrist",
        required=True,
        help="Wrist camera index/path, for example 2 or /dev/v4l/by-id/...",
    )
    parser.add_argument("--tcc-source-root", type=Path, required=True)
    parser.add_argument("--hub-cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--camera-startup-delay", type=float, default=1.0)
    parser.add_argument("--camera-read-attempts", type=int, default=3)
    parser.add_argument("--camera-retry-delay", type=float, default=0.25)
    parser.add_argument("--camera-min-channel-std", type=float, default=2.0)
    parser.add_argument("--camera-max-pair-skew-ms", type=float, default=50.0)
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
    assert_shadow_only(robot_config, args.execute)

    import cv2

    from tcc_real_robot.policy_runtime import (
        load_policy_bundle,
        predict_action,
        resolve_device,
    )
    from tcc_real_robot.tcc_backbone import load_frozen_tcc_backbone

    device = resolve_device(args.device)
    task_names = [str(value) for value in config["dataset"]["tasks"]]
    task_index, task_name = resolve_task(args.task, task_names)
    max_configured_steps = int(config["evaluation"]["max_rollout_steps"])
    max_steps = args.max_steps or max_configured_steps
    if not 1 <= max_steps <= max_configured_steps:
        raise ValueError(f"--max-steps must be within [1, {max_configured_steps}]")
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
    trained_tasks = [str(value) for value in bundle.config["dataset"]["tasks"]]
    if trained_tasks != task_names:
        raise RuntimeError(
            f"Checkpoint task order {trained_tasks} differs from config {task_names}"
        )
    expected_resolution = (height, width, 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"policy_shadow_{args.backbone}_{stamp}.txt"
    started = time.monotonic()
    rollout_started: float | None = None
    completed = 0
    decision = "FAIL"
    failure = ""
    latencies_ms: list[float] = []
    pair_skews_ms: list[float] = []
    home_session: Any | None = None
    home_reference: list[float] | None = None
    first_arm_delta: float | None = None
    first_gripper_delta: float | None = None

    with output_path.open("w", encoding="utf-8", buffering=1) as report:
        report.write("TCC Real-Robot Policy Shadow Evaluation\n")
        report.write("=======================================\n")
        report.write("Mode: SHADOW (policy predictions are never actuated)\n")
        report.write(f"Home staging: {'ENABLED' if args.execute_home else 'NOT RUN'}\n")
        report.write(f"Task: {task_name} (index {task_index})\n")
        report.write(f"Backbone: {args.backbone}\n")
        report.write(f"Demonstrations: {args.demonstrations}\n")
        report.write(f"Training step: {bundle.step}\n")
        report.write(f"Hub repository: {assets.repository}\n")
        report.write(f"Hub revision: {assets.revision}\n")
        report.write(f"Backbone SHA256: {assets.backbone_sha256}\n")
        report.write(f"Policy SHA256: {assets.policy_sha256}\n")
        report.write(f"Device: {device}\n")
        report.write(f"Camera main: {args.cam_main}\n")
        report.write(f"Camera wrist: {args.cam_wrist}\n")
        report.write(
            f"Camera capture rate: {camera_capture_fps:.3f} FPS "
            "(validated V4L2 profile)\n"
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
        report.write(f"Inference warmup steps: {inference_warmup_steps}\n")
        report.write(f"Maximum steps: {max_steps}\n\n")
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
                    "Policy actions remain shadow-only."
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
            with SynchronizedCameras(
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
            ) as cameras:
                report.write(f"Camera main negotiated: {cameras.main_properties}\n")
                report.write(
                    f"Camera wrist negotiated: {cameras.wrist_properties}\n\n"
                )
                for _ in range(args.warmup_frames):
                    cameras.read_rgb_pair()
                for _ in range(inference_warmup_steps):
                    warm_main, warm_wrist = cameras.read_rgb_pair()
                    predict_action(
                        backbone,
                        bundle,
                        warm_main,
                        warm_wrist,
                        task_index,
                        int(backbone_metadata["image_size"]),
                        device,
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
                    main_rgb, wrist_rgb = cameras.read_rgb_pair()
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
                    action = predict_action(
                        backbone,
                        bundle,
                        main_rgb,
                        wrist_rgb,
                        task_index,
                        int(backbone_metadata["image_size"]),
                        device,
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
                    values = ", ".join(f"{value:.7f}" for value in action.tolist())
                    report.write(
                        f"step={step:03d} elapsed_s={elapsed:.6f} "
                        f"inference_ms={inference_ms:.3f} "
                        f"pair_skew_ms={cameras.last_pair_skew_ms:.3f} "
                        f"action=[{values}]\n"
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
            checks = {
                "all_steps_completed": completed == max_steps,
                "minimum_rate_met": observed_rate
                >= float(evaluation_settings["minimum_observed_rate_hz"]),
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
            if failure:
                decision = "FAIL"
            elif all(checks.values()):
                decision = "PASS"
            else:
                decision = "BLOCKED"
            report.write("\nSummary\n")
            report.write(f"Completed steps: {completed}/{max_steps}\n")
            report.write(f"Elapsed: {elapsed:.6f} s\n")
            report.write(f"Observed rate: {observed_rate:.3f} Hz\n")
            report.write(f"Latency median: {latency_median:.3f} ms\n")
            report.write(f"Latency p95: {latency_p95:.3f} ms\n")
            report.write(f"Latency maximum: {latency_max:.3f} ms\n")
            report.write(f"Camera pair skew median: {pair_skew_median:.3f} ms\n")
            report.write(f"Camera pair skew maximum: {pair_skew_max:.3f} ms\n")
            if first_arm_delta is not None:
                report.write(
                    f"First action maximum arm delta: {first_arm_delta:.7f} rad\n"
                )
            if first_gripper_delta is not None:
                report.write(
                    f"First action gripper delta: {first_gripper_delta:.7f} m\n"
                )
            report.write("Checks:\n")
            for name, passed in checks.items():
                report.write(f"- {name}: {'PASS' if passed else 'FAIL'}\n")
            if failure:
                report.write(f"Failure: {failure}\n")
            report.write(f"Decision: {decision}\n")

    print(output_path.read_text(encoding="utf-8"), end="")
    print(f"report: {output_path}")
    if decision != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
