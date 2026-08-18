#!/usr/bin/env python3
"""Preview every Linux video node without connecting to the robot."""

from __future__ import annotations

import argparse
import glob
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture and label one frame from every /dev/video* node so the "
            "operator can identify cam_main and cam_wrist."
        )
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Only save the contact sheet; do not open a GUI window.",
    )
    return parser.parse_args()


def natural_video_key(path: str) -> int:
    match = re.search(r"(\d+)$", path)
    return int(match.group(1)) if match else 10**9


def normalize_for_display(frame: np.ndarray) -> np.ndarray:
    """Convert color, grayscale, or 16-bit frames to uint8 BGR."""
    if frame.dtype != np.uint8:
        minimum = float(np.min(frame))
        maximum = float(np.max(frame))
        if maximum > minimum:
            frame = ((frame.astype(np.float32) - minimum) * 255.0) / (maximum - minimum)
        else:
            frame = np.zeros_like(frame, dtype=np.float32)
        frame = frame.astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"Unsupported frame shape {frame.shape}")


def placeholder(width: int, height: int, message: str) -> np.ndarray:
    image = np.full((height, width, 3), 35, dtype=np.uint8)
    cv2.putText(
        image,
        message,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def capture_node(
    path: str,
    width: int,
    height: int,
    fps: float,
    warmup_frames: int,
) -> tuple[np.ndarray, str]:
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            return placeholder(width, height, "OPEN FAILED"), "open failed"
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        frame: np.ndarray | None = None
        for _ in range(max(1, warmup_frames + 1)):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            return placeholder(width, height, "READ FAILED"), "read failed"
        image = normalize_for_display(frame)
        original_shape = "x".join(str(value) for value in frame.shape)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image, f"frame={original_shape} dtype={frame.dtype}"
    except Exception as exc:  # noqa: BLE001 - continue probing other camera nodes
        return (
            placeholder(width, height, type(exc).__name__),
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        capture.release()


def label_frame(image: np.ndarray, path: str, status: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 72), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        path,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        labeled,
        status[:80],
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return labeled


def build_contact_sheet(
    frames: list[np.ndarray], columns: int, width: int, height: int
) -> np.ndarray:
    rows = (len(frames) + columns - 1) // columns
    empty = np.zeros((height, width, 3), dtype=np.uint8)
    padded = frames + [empty] * (rows * columns - len(frames))
    return np.vstack(
        [
            np.hstack(padded[index : index + columns])
            for index in range(0, len(padded), columns)
        ]
    )


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.fps, args.columns) <= 0:
        raise SystemExit("Width, height, FPS, and columns must be positive")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")
    devices = sorted(glob.glob("/dev/video*"), key=natural_video_key)
    if not devices:
        raise SystemExit("No /dev/video* devices were found")

    frames = []
    print("Camera probe results:")
    for device in devices:
        image, status = capture_node(
            device,
            args.width,
            args.height,
            args.fps,
            args.warmup_frames,
        )
        print(f"{device}: {status}")
        frames.append(label_frame(image, device, status))

    sheet = build_contact_sheet(frames, args.columns, args.width, args.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"camera_probe_{stamp}.jpg"
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to save {output_path}")
    print(f"contact sheet: {output_path}")

    if not args.no_display:
        try:
            cv2.namedWindow("Camera probe - press q to close", cv2.WINDOW_NORMAL)
            cv2.imshow("Camera probe - press q to close", sheet)
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key in (ord("q"), 27):
                    break
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"GUI unavailable; inspect the saved JPG instead: {exc}")


if __name__ == "__main__":
    main()
