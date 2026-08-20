#!/usr/bin/env python3
"""Capture candidate V4L2 streams selected by stable camera serial numbers."""

from __future__ import annotations

import argparse
import glob
from datetime import datetime, timezone
from pathlib import Path

import cv2

from inspect_camera_identity import (
    sysfs_usb_identity,
    udev_properties,
    video_number,
)
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
    parser.add_argument("--main-model", default="D435")
    parser.add_argument("--wrist-model", default="D405")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def matches_serial(
    properties: dict[str, str], usb_identity: dict[str, str], serial: str
) -> bool:
    return (
        properties.get("ID_SERIAL_SHORT") == serial
        or serial in properties.get("ID_SERIAL", "")
        or usb_identity.get("SYSFS_USB_SERIAL") == serial
    )


def main() -> None:
    args = parse_args()
    if args.main_serial == args.wrist_serial:
        raise SystemExit("Main and wrist camera serial numbers must differ")
    if min(args.width, args.height, args.fps) <= 0:
        raise SystemExit("Width, height, and FPS must be positive")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")

    requested = (
        ("cam_main / D435", args.main_serial, args.main_model),
        ("cam_wrist / D405", args.wrist_serial, args.wrist_model),
    )
    inventory = []
    discovered = []
    for device in sorted(glob.glob("/dev/video*"), key=video_number):
        properties = udev_properties(device)
        usb_identity = sysfs_usb_identity(device)
        product = usb_identity.get(
            "SYSFS_USB_PRODUCT", properties.get("ID_V4L_PRODUCT", "UNAVAILABLE")
        )
        inventory.append((device, properties, usb_identity, product))
        discovered.append(
            (
                device,
                properties.get("ID_SERIAL_SHORT", "UNAVAILABLE"),
                usb_identity.get("SYSFS_USB_SERIAL", "UNAVAILABLE"),
                product,
            )
        )

    candidates = []
    missing = []
    for role, serial, model in requested:
        role_matches = [
            row for row in inventory if matches_serial(row[1], row[2], serial)
        ]
        resolution = "serial"
        if not role_matches:
            role_matches = [
                row for row in inventory if model.lower() in row[3].lower()
            ]
            resolution = "unique-model-and-physical-path fallback"
            physical_paths = {
                row[2].get("SYSFS_USB_PATH", "UNAVAILABLE") for row in role_matches
            }
            if len(physical_paths) != 1:
                role_matches = []
        if not role_matches:
            missing.append(f"{role}: serial={serial}, model={model}")
            continue
        for device, properties, usb_identity, product in role_matches:
            candidates.append(
                (
                    device,
                    serial,
                    role,
                    properties,
                    usb_identity,
                    product,
                    resolution,
                )
            )

    if missing:
        details = "; ".join(
            f"{device}: udev={udev_serial}, sysfs={sysfs_serial}, product={product}"
            for device, udev_serial, sysfs_serial, product in discovered
        )
        raise RuntimeError(
            f"No unique V4L2 device found for: {missing}. Discovered: {details}"
        )

    frames = []
    rows = [
        "Camera Serial Stream Probe",
        "==========================",
        "Robot connection: DISABLED",
        "",
    ]
    for (
        device,
        serial,
        role,
        properties,
        usb_identity,
        product,
        resolution,
    ) in candidates:
        capabilities = properties.get("ID_V4L_CAPABILITIES", "UNAVAILABLE")
        image, status = capture_node(
            device,
            args.width,
            args.height,
            args.fps,
            args.warmup_frames,
        )
        path = properties.get("ID_PATH", "UNAVAILABLE")
        usb_path = usb_identity.get("SYSFS_USB_PATH", "UNAVAILABLE")
        observed_serial = usb_identity.get("SYSFS_USB_SERIAL", "UNAVAILABLE")
        label = f"{role} {device} {resolution}"
        frames.append(label_frame(image, label, status))
        rows.extend(
            (
                f"role: {role}",
                f"requested_serial: {serial}",
                f"observed_usb_serial: {observed_serial}",
                f"product: {product}",
                f"resolution_method: {resolution}",
                f"device: {device}",
                f"physical_path: {path}",
                f"sysfs_usb_path: {usb_path}",
                f"capabilities: {capabilities}",
                f"status: {status}",
                "",
            )
        )
        print(
            f"{label}: observed_usb_serial={observed_serial}, status={status}"
        )

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
