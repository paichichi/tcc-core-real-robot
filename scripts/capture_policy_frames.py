#!/usr/bin/env python3
"""Capture one fresh synchronized RGB pair from the policy cameras."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from tcc_real_robot.realsense_cameras import RealSenseColorCameras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture fresh cam_main and cam_wrist frames without robot access."
    )
    parser.add_argument("--cam-main-serial", required=True)
    parser.add_argument("--cam-wrist-serial", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def save_rgb(path: Path, frame: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to save {path}")


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.fps) <= 0:
        raise SystemExit("Width, height, and FPS must be positive")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "PYREALSENSE2_NOT_INSTALLED: run python -m pip install pyrealsense2"
        ) from exc

    with RealSenseColorCameras(
        rs,
        args.cam_main_serial,
        args.cam_wrist_serial,
        args.width,
        args.height,
        int(args.fps),
    ) as cameras:
        for _ in range(args.warmup_frames):
            cameras.read_rgb_pair()
        main_rgb, wrist_rgb = cameras.read_rgb_pair()
        main_properties = cameras.main_properties
        wrist_properties = cameras.wrist_properties
        pair_skew_ms = cameras.last_pair_skew_ms

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    main_path = args.output_dir / f"live_policy_frame_{stamp}_cam_main.png"
    wrist_path = args.output_dir / f"live_policy_frame_{stamp}_cam_wrist.png"
    report_path = args.output_dir / f"live_policy_frame_{stamp}.txt"
    save_rgb(main_path, main_rgb)
    save_rgb(wrist_path, wrist_rgb)
    with report_path.open("w", encoding="utf-8") as report:
        report.write("Fresh Policy Camera Frames\n")
        report.write("==========================\n")
        report.write("Robot connection: DISABLED\n")
        report.write("Policy inference: DISABLED\n")
        report.write("camera_backend: RealSense SDK color/rgb8\n")
        report.write(f"cam_main: {main_properties}\n")
        report.write(f"cam_wrist: {wrist_properties}\n")
        report.write(f"pair_skew_ms: {pair_skew_ms:.3f}\n")
        report.write(
            "cam_main_rgb_mean: "
            f"{np.mean(main_rgb, axis=(0, 1)).round(3).tolist()}\n"
        )
        report.write(
            "cam_wrist_rgb_mean: "
            f"{np.mean(wrist_rgb, axis=(0, 1)).round(3).tolist()}\n"
        )
        report.write(f"cam_main_image: {main_path}\n")
        report.write(f"cam_wrist_image: {wrist_path}\n")

    print(f"cam_main: {main_path}")
    print(f"cam_wrist: {wrist_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
