#!/usr/bin/env python3
"""List RealSense SDK identities and stream profiles without starting streams."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def optional_info(device: object, rs: object, name: str) -> str:
    field = getattr(rs.camera_info, name, None)
    if field is None or not device.supports(field):
        return "UNAVAILABLE"
    return str(device.get_info(field))


def main() -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "PYREALSENSE2_NOT_INSTALLED: run python -m pip install pyrealsense2"
        ) from exc

    context = rs.context()
    devices = list(context.query_devices())
    if not devices:
        raise SystemExit("NO_REALSENSE_DEVICES_FOUND")

    lines = [
        "RealSense SDK Identity Report",
        "=============================",
        "Streams started: NO",
        f"Devices: {len(devices)}",
        "",
    ]
    info_names = (
        "name",
        "serial_number",
        "asic_serial_number",
        "physical_port",
        "product_id",
        "product_line",
        "firmware_version",
        "usb_type_descriptor",
    )
    for device_index, device in enumerate(devices):
        lines.append(f"[device {device_index}]")
        for name in info_names:
            lines.append(f"{name}: {optional_info(device, rs, name)}")
        lines.append("sensors:")
        for sensor_index, sensor in enumerate(device.query_sensors()):
            sensor_name = optional_info(sensor, rs, "name")
            lines.append(f"  [{sensor_index}] {sensor_name}")
            matching_profiles = []
            for profile in sensor.get_stream_profiles():
                try:
                    video = profile.as_video_stream_profile()
                    width = int(video.width())
                    height = int(video.height())
                except RuntimeError:
                    continue
                stream_name = str(profile.stream_type())
                if (
                    width == 640
                    and height == 480
                    and int(profile.fps()) == 30
                    and any(
                        name in stream_name
                        for name in ("color", "depth", "infrared")
                    )
                ):
                    matching_profiles.append(
                        "    "
                        f"stream={stream_name}, index={profile.stream_index()}, "
                        f"format={profile.format()}, {width}x{height}@{profile.fps()}"
                    )
            lines.extend(matching_profiles or ["    no 640x480@30 video profiles"])
        lines.append("")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"realsense_sdk_identity_{stamp}.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
