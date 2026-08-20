#!/usr/bin/env python3
"""Capture candidate V4L2 streams selected by stable camera serial numbers."""

from __future__ import annotations

import argparse
import glob
from datetime import datetime, timezone
from pathlib import Path

import cv2

from inspect_camera_identity import udev_properties, video_number
from preview_cameras import build_contact_sheet, capture_node, label_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve V4L2 nodes by camera serial and save one labeled frame "
            "from every capture-capable candidate."
        )
    )
    parser.add_argument("--main-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def matches_serial(properties: dict[str, str], serial: str) -> bool:
    return properties.get("ID_SERIAL_SHORT") == serial or serial in properties.get(
        "ID_SERIAL", ""
    )


def main() -> None:
    args = parse_args()
    if args.main_serial == args.wrist_serial:
        raise SystemExit("Main and wrist camera serial numbers must differ")
    if min(args.width, args.height, args.fps) <= 0:
        raise SystemExit("Width, height, and FPS must be positive")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")

    serial_roles = {
        args.main_serial: "cam_main / D435",
        args.wrist_serial: "cam_wrist / D405",
    }
    candidates = []
    for device in sorted(glob.glob("/dev/video*"), key=video_number):
        properties = udev_properties(device)
        for serial, role in serial_roles.items():
            if matches_serial(properties, serial):
                candidates.append((device, serial, role, properties))
                break

    missing = [
        serial
        for serial in serial_roles
        if not any(candidate[1] == serial for candidate in candidates)
    ]
    if missing:
        raise RuntimeError(f"No V4L2 nodes found for serials: {missing}")

    frames = []
    rows = [
        "Camera Serial Stream Probe",
        "==========================",
        "Robot connection: DISABLED",
        "",
    ]
    for device, serial, role, properties in candidates:
        capabilities = properties.get("ID_V4L_CAPABILITIES", "UNAVAILABLE")
        image, status = capture_node(
            device,
            args.width,
            args.height,
            args.fps,
            args.warmup_frames,
        )
        path = properties.get("ID_PATH", "UNAVAILABLE")
        label = f"{role} {serial} {device}"
        frames.append(label_frame(image, label, status))
        rows.extend(
            (
                f"role: {role}",
                f"serial: {serial}",
                f"device: {device}",
                f"physical_path: {path}",
                f"capabilities: {capabilities}",
                f"status: {status}",
                "",
            )
        )
        print(f"{label}: {status}")

    sheet = build_contact_sheet(frames, 2, args.width, args.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    image_path = args.output_dir / f"camera_serial_probe_{stamp}.jpg"
    report_path = args.output_dir / f"camera_serial_probe_{stamp}.txt"
    if not cv2.imwrite(str(image_path), sheet):
        raise RuntimeError(f"Failed to save {image_path}")
    rows.append(f"contact_sheet: {image_path}")
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"contact sheet: {image_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
