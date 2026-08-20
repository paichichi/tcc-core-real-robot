#!/usr/bin/env python3
"""Report stable Linux identities for every V4L2 camera node."""

from __future__ import annotations

import glob
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROPERTY_KEYS = (
    "ID_VENDOR_ID",
    "ID_MODEL_ID",
    "ID_SERIAL",
    "ID_SERIAL_SHORT",
    "ID_PATH",
    "ID_PATH_TAG",
    "ID_V4L_PRODUCT",
    "ID_V4L_CAPABILITIES",
    "DEVLINKS",
)


def video_number(path: str) -> int:
    suffix = Path(path).name.removeprefix("video")
    return int(suffix) if suffix.isdecimal() else 10**9


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return "UNAVAILABLE"


def udev_properties(device: str) -> dict[str, str]:
    if shutil.which("udevadm") is None:
        return {}
    completed = subprocess.run(
        ["udevadm", "info", "--query=property", f"--name={device}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return {"UDEV_ERROR": completed.stderr.strip() or "unknown error"}
    properties = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def sysfs_usb_identity(device: str) -> dict[str, str]:
    """Find USB identity attributes by walking above a video4linux node."""
    node = Path(device).name
    device_path = (Path("/sys/class/video4linux") / node / "device").resolve()
    for candidate in (device_path, *device_path.parents):
        serial = read_text(candidate / "serial")
        vendor = read_text(candidate / "idVendor")
        product_id = read_text(candidate / "idProduct")
        if serial != "UNAVAILABLE" or (
            vendor != "UNAVAILABLE" and product_id != "UNAVAILABLE"
        ):
            return {
                "SYSFS_USB_PATH": str(candidate),
                "SYSFS_USB_SERIAL": serial,
                "SYSFS_USB_VENDOR_ID": vendor,
                "SYSFS_USB_PRODUCT_ID": product_id,
                "SYSFS_USB_PRODUCT": read_text(candidate / "product"),
                "SYSFS_USB_BUSNUM": read_text(candidate / "busnum"),
                "SYSFS_USB_DEVPATH": read_text(candidate / "devpath"),
            }
    return {}


def symlink_rows(directory: Path) -> list[str]:
    if not directory.is_dir():
        return [f"{directory}: UNAVAILABLE"]
    rows = []
    for path in sorted(directory.iterdir()):
        try:
            target = path.resolve(strict=True)
        except (FileNotFoundError, OSError):
            target = Path("BROKEN")
        rows.append(f"{path} -> {target}")
    return rows or [f"{directory}: EMPTY"]


def realsense_summary() -> list[str]:
    executable = shutil.which("rs-enumerate-devices")
    if executable is None:
        return ["rs-enumerate-devices: NOT INSTALLED"]
    completed = subprocess.run(
        [executable, "-s"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = completed.stdout.strip()
    if completed.stderr.strip():
        output = f"{output}\nSTDERR: {completed.stderr.strip()}".strip()
    return output.splitlines() or [f"rs-enumerate-devices exit={completed.returncode}"]


def main() -> None:
    devices = sorted(glob.glob("/dev/video*"), key=video_number)
    if not devices:
        raise SystemExit("No /dev/video* devices found")

    lines = [
        "Linux Camera Identity Report",
        "============================",
        "This command does not open cameras or connect to the robot.",
        "",
    ]
    for device in devices:
        node = Path(device).name
        sysfs_root = Path("/sys/class/video4linux") / node
        properties = udev_properties(device)
        usb_identity = sysfs_usb_identity(device)
        lines.extend(
            (
                f"[{device}]",
                f"sysfs_name: {read_text(sysfs_root / 'name')}",
                f"sysfs_index: {read_text(sysfs_root / 'index')}",
                f"sysfs_device: {(sysfs_root / 'device').resolve()}",
            )
        )
        for key in PROPERTY_KEYS:
            lines.append(f"{key}: {properties.get(key, 'UNAVAILABLE')}")
        for key in (
            "SYSFS_USB_PATH",
            "SYSFS_USB_SERIAL",
            "SYSFS_USB_VENDOR_ID",
            "SYSFS_USB_PRODUCT_ID",
            "SYSFS_USB_PRODUCT",
            "SYSFS_USB_BUSNUM",
            "SYSFS_USB_DEVPATH",
        ):
            lines.append(f"{key}: {usb_identity.get(key, 'UNAVAILABLE')}")
        if "UDEV_ERROR" in properties:
            lines.append(f"UDEV_ERROR: {properties['UDEV_ERROR']}")
        lines.append("")

    lines.append("[/dev/v4l/by-id]")
    lines.extend(symlink_rows(Path("/dev/v4l/by-id")))
    lines.append("")
    lines.append("[/dev/v4l/by-path]")
    lines.extend(symlink_rows(Path("/dev/v4l/by-path")))
    lines.append("")
    lines.append("[RealSense SDK summary]")
    lines.extend(realsense_summary())
    lines.append("")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"camera_identity_{stamp}.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
