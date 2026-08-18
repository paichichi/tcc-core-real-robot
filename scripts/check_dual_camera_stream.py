#!/usr/bin/env python3
"""Check two V4L2 cameras concurrently without connecting to the robot."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-main", required=True)
    parser.add_argument("--cam-wrist", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--maximum-pair-skew-ms", type=float, default=50.0)
    parser.add_argument("--startup-delay", type=float, default=1.0)
    parser.add_argument("--minimum-channel-std", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def decode_fourcc(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))


def configure(
    capture: Any, width: int, height: int, camera_fps: float
) -> dict[str, object]:
    requested = {
        "width": bool(capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)),
        "height": bool(capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)),
        "fps": bool(capture.set(cv2.CAP_PROP_FPS, camera_fps)),
        "buffer_size": bool(capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)),
    }
    return {
        "set_results": requested,
        "actual_width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "actual_height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "actual_fps": capture.get(cv2.CAP_PROP_FPS),
        "actual_fourcc": decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
    }


def frame_status(
    retrieved: bool, frame: np.ndarray | None, minimum_channel_std: float
) -> tuple[bool, str]:
    if not retrieved or frame is None:
        return False, "RETRIEVE_FAILED"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False, f"UNEXPECTED_SHAPE_{frame.shape}"
    channel_std = np.std(frame, axis=(0, 1))
    valid = float(np.max(channel_std)) >= minimum_channel_std
    status = (
        f"{'VALID' if valid else 'FLAT'} shape={frame.shape} "
        f"channel_std={channel_std.round(3).tolist()}"
    )
    return valid, status


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.camera_fps, args.frames) <= 0:
        raise SystemExit("Width, height, camera FPS, and frames must be positive")
    if args.startup_delay < 0 or args.minimum_channel_std < 0:
        raise SystemExit("Delays and thresholds must be non-negative")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")
    if args.maximum_pair_skew_ms <= 0:
        raise SystemExit("--maximum-pair-skew-ms must be positive")

    main_camera = cv2.VideoCapture(args.cam_main, cv2.CAP_V4L2)
    wrist_camera = cv2.VideoCapture(args.cam_wrist, cv2.CAP_V4L2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.output_dir / f"dual_camera_stream_{stamp}.txt"
    main_valid_count = 0
    wrist_valid_count = 0
    pair_skews_ms: list[float] = []
    passed = False

    try:
        with report_path.open("w", encoding="utf-8", buffering=1) as report:
            report.write("Dual V4L2 Camera Stream Diagnostic\n")
            report.write("==================================\n")
            report.write(f"cam_main: {args.cam_main}\n")
            report.write(f"cam_wrist: {args.cam_wrist}\n")
            report.write(
                f"requested: {args.width}x{args.height} @ {args.camera_fps:.3f} FPS\n"
            )
            report.write(f"main_opened: {main_camera.isOpened()}\n")
            report.write(f"wrist_opened: {wrist_camera.isOpened()}\n")
            if not main_camera.isOpened() or not wrist_camera.isOpened():
                report.write("Decision: FAIL_OPEN\n")
            else:
                main_negotiated = configure(
                    main_camera, args.width, args.height, args.camera_fps
                )
                wrist_negotiated = configure(
                    wrist_camera, args.width, args.height, args.camera_fps
                )
                report.write(f"main_negotiated: {main_negotiated}\n")
                report.write(f"wrist_negotiated: {wrist_negotiated}\n")
                profiles_match = all(
                    abs(float(properties[key]) - expected) <= tolerance
                    for properties in (main_negotiated, wrist_negotiated)
                    for key, expected, tolerance in (
                        ("actual_width", args.width, 0.5),
                        ("actual_height", args.height, 0.5),
                        ("actual_fps", args.camera_fps, 0.1),
                    )
                )
                report.write(f"strict_dataset_profiles_match: {profiles_match}\n")
                if args.startup_delay:
                    time.sleep(args.startup_delay)
                for _ in range(args.warmup_frames):
                    main_grabbed = bool(main_camera.grab())
                    wrist_grabbed = bool(wrist_camera.grab())
                    if main_grabbed:
                        main_camera.retrieve()
                    if wrist_grabbed:
                        wrist_camera.retrieve()
                report.write(f"discarded_warmup_pairs: {args.warmup_frames}\n")
                for index in range(args.frames):
                    main_grabbed = bool(main_camera.grab())
                    main_grabbed_at = time.monotonic()
                    wrist_grabbed = bool(wrist_camera.grab())
                    wrist_grabbed_at = time.monotonic()
                    pair_skew_ms = abs(wrist_grabbed_at - main_grabbed_at) * 1000.0
                    pair_skews_ms.append(pair_skew_ms)
                    main_retrieved, main_frame = (
                        main_camera.retrieve() if main_grabbed else (False, None)
                    )
                    wrist_retrieved, wrist_frame = (
                        wrist_camera.retrieve() if wrist_grabbed else (False, None)
                    )
                    main_valid, main_status = frame_status(
                        main_retrieved, main_frame, args.minimum_channel_std
                    )
                    wrist_valid, wrist_status = frame_status(
                        wrist_retrieved, wrist_frame, args.minimum_channel_std
                    )
                    main_valid_count += int(main_valid)
                    wrist_valid_count += int(wrist_valid)
                    report.write(
                        f"frame={index:03d} "
                        f"main_grab={'PASS' if main_grabbed else 'FAIL'} "
                        f"main={main_status} "
                        f"wrist_grab={'PASS' if wrist_grabbed else 'FAIL'} "
                        f"wrist={wrist_status} pair_skew_ms={pair_skew_ms:.3f}\n"
                    )
                report.write("\nSummary\n")
                report.write(f"main_valid_frames: {main_valid_count}/{args.frames}\n")
                report.write(f"wrist_valid_frames: {wrist_valid_count}/{args.frames}\n")
                report.write(
                    f"pair_skew_max_ms: {max(pair_skews_ms, default=float('nan')):.3f}\n"
                )
                passed = (
                    profiles_match
                    and main_valid_count == args.frames
                    and wrist_valid_count == args.frames
                    and max(pair_skews_ms, default=float("inf"))
                    <= args.maximum_pair_skew_ms
                )
                report.write(f"Decision: {'PASS' if passed else 'FAIL'}\n")
    finally:
        main_camera.release()
        wrist_camera.release()

    print(report_path.read_text(encoding="utf-8"), end="")
    print(f"report: {report_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
