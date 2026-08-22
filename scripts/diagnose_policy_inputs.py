#!/usr/bin/env python3
"""Compare live policy inputs with recorded demo first frames without actuation."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from run_policy import SynchronizedCameras, resolve_task
from torch.nn import functional as torch_f

from tcc_real_robot.config import load_yaml
from tcc_real_robot.model_assets import resolve_model_assets
from tcc_real_robot.policy_data import build_episode_records
from tcc_real_robot.policy_home import PolicyHomeSession
from tcc_real_robot.policy_runtime import (
    load_policy_bundle,
    predict_action,
    preprocess_rgb_frames,
    resolve_device,
    restore_policy_backbone,
)
from tcc_real_robot.tcc_backbone import (
    IndependentCameraBackbones,
    load_policy_camera_backbones,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage at dataset home, capture the exact live RGB policy inputs, "
            "and compare them with recorded demo first frames. Policy actions "
            "are never sent to the robot."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robot.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--backbone", default="ours_rn50")
    parser.add_argument("--demonstrations", type=int, default=80)
    parser.add_argument("--task", default="carrot")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--camera-backend",
        choices=("realsense-sdk", "v4l2"),
        default="realsense-sdk",
    )
    parser.add_argument("--cam-main-serial")
    parser.add_argument("--cam-wrist-serial")
    parser.add_argument("--cam-main")
    parser.add_argument("--cam-wrist")
    parser.add_argument("--tcc-source-root", type=Path, required=True)
    parser.add_argument("--hub-cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--controller-timeout", type=float, default=20.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--execute-home", action="store_true")
    parser.add_argument("--emergency-stop-ready", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def first_rgb_frame(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"Video contains no frames: {path}")


def resolve_dataset_root(config: dict, override: Path | None) -> Path:
    value = override if override is not None else config["dataset"].get("local_root")
    if value in (None, "TBD"):
        raise ValueError("Set --dataset-root or dataset.local_root")
    root = Path(str(value)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


@torch.inference_mode()
def encode_frames(
    backbone: torch.nn.Module,
    frames: list[np.ndarray],
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    images = preprocess_rgb_frames(frames, image_size).to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        if isinstance(backbone, IndependentCameraBackbones):
            if len(frames) % 2:
                raise ValueError("Independent camera diagnostics require frame pairs")
            main_features, wrist_features = backbone(images[0::2], images[1::2])
            features = torch.empty(
                (len(frames), main_features.shape[1]),
                dtype=main_features.dtype,
                device=main_features.device,
            )
            features[0::2] = main_features
            features[1::2] = wrist_features
            features = features.float()
        else:
            features = backbone(images).float()
    return torch_f.normalize(features, dim=1).cpu()


def action_delta(action: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    difference = np.abs(action - reference)
    return float(difference[:6].max()), float(difference[6])


def save_rgb(path: Path, frame: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to save {path}")


def labeled_tile(frame: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tile = cv2.resize(tile, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (640, 48), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def match_channel_moments(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match per-channel source moments to a reference image for diagnosis."""
    source_float = source.astype(np.float32)
    target_float = target.astype(np.float32)
    source_mean = source_float.mean(axis=(0, 1), keepdims=True)
    source_std = source_float.std(axis=(0, 1), keepdims=True)
    target_mean = target_float.mean(axis=(0, 1), keepdims=True)
    target_std = target_float.std(axis=(0, 1), keepdims=True)
    if np.any(source_std < 1e-6):
        raise ValueError("Cannot color-match a flat source channel")
    matched = (source_float - source_mean) * (target_std / source_std) + target_mean
    return np.clip(np.rint(matched), 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")
    if args.profile_steps <= 0:
        raise SystemExit("--profile-steps must be positive")
    if args.execute_home and not args.emergency_stop_ready:
        raise SystemExit(
            "--execute-home requires --emergency-stop-ready; policy actions remain disabled"
        )

    import pyarrow.parquet as pq

    config = load_yaml(args.config)
    robot_config = load_yaml(args.robot_config)
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
        raise SystemExit("--camera-backend v4l2 requires --cam-main and --cam-wrist")
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    task_names = [str(value) for value in config["dataset"]["tasks"]]
    task_index, task_name = resolve_task(args.task, task_names)
    split = config["split"]
    records = build_episode_records(
        dataset_root,
        task_names,
        int(config["seed"]),
        (
            int(split["train_episodes_per_task"]),
            int(split["validation_episodes_per_task"]),
            int(split["test_episodes_per_task"]),
        ),
    )
    selected = [
        record
        for record in records
        if record.task_index == task_index and record.split == "train"
    ][: args.episodes]
    if len(selected) != args.episodes:
        raise RuntimeError(
            f"Only {len(selected)} training episodes available; requested {args.episodes}"
        )

    device = resolve_device(args.device)
    assets = resolve_model_assets(
        config,
        args.backbone,
        args.demonstrations,
        cache_dir=args.hub_cache_dir,
        local_files_only=args.offline,
    )
    backbone, backbone_metadata = load_policy_camera_backbones(
        assets.backbone_path,
        args.tcc_source_root,
        device,
        config["policy"],
    )
    bundle = load_policy_bundle(
        assets.policy_path,
        expected_feature_dim=int(backbone_metadata["feature_dim"]),
        device=device,
    )
    restored = restore_policy_backbone(backbone, bundle)
    if config.get("backbone", {}).get("frozen") is False and not restored:
        raise RuntimeError(
            "End-to-end checkpoint is missing its embedded backbone_model"
        )
    width, height = [int(value) for value in config["observations"]["resolution"]]
    camera_fps = float(robot_config["policy_evaluation"]["camera_capture_fps"])

    if args.execute_home:
        try:
            import trossen_arm
        except ImportError as exc:
            raise RuntimeError(
                "Install the robot dependencies with pip install -e '.[robot]'"
            ) from exc
        home_session = PolicyHomeSession(
            trossen_arm, robot_config, args.controller_timeout
        )
        try:
            preparation = home_session.prepare()
            home = np.asarray(preparation.observed, dtype=np.float64)
        finally:
            home_session.close()
    else:
        robot = robot_config["robot"]
        home = np.asarray(
            [*robot["home_arm_positions_rad"], robot["home_gripper_position_m"]],
            dtype=np.float64,
        )

    if args.camera_backend == "realsense-sdk":
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for the RealSense SDK backend"
            ) from exc
        from tcc_real_robot.realsense_cameras import RealSenseColorCameras

        if not camera_fps.is_integer():
            raise RuntimeError("RealSense SDK capture FPS must be a whole number")
        camera_context = RealSenseColorCameras(
            rs,
            args.cam_main_serial,
            args.cam_wrist_serial,
            width,
            height,
            int(camera_fps),
        )
    else:
        camera_context = SynchronizedCameras(
            cv2,
            args.cam_main,
            args.cam_wrist,
            width,
            height,
            camera_fps,
        )
    with camera_context as cameras:
        for _ in range(args.warmup_frames):
            cameras.read_rgb_pair()
        live_main, live_wrist = cameras.read_rgb_pair()
        camera_properties = (cameras.main_properties, cameras.wrist_properties)
        pair_skew_ms = cameras.last_pair_skew_ms

    image_size = int(backbone_metadata["image_size"])
    live_prediction = (
        predict_action(
            backbone,
            bundle,
            live_main,
            live_wrist,
            task_index,
            image_size,
            device,
            observation_state=home,
            episode_progress=0.0,
        )
        .numpy()
        .astype(np.float64)
    )
    profile_latencies_ms = []
    for _ in range(args.profile_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile_started = time.perf_counter()
        predict_action(
            backbone,
            bundle,
            live_main,
            live_wrist,
            task_index,
            image_size,
            device,
            observation_state=home,
            episode_progress=0.0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile_latencies_ms.append((time.perf_counter() - profile_started) * 1000.0)
    swapped_prediction = (
        predict_action(
            backbone,
            bundle,
            live_wrist,
            live_main,
            task_index,
            image_size,
            device,
            observation_state=home,
            episode_progress=0.0,
        )
        .numpy()
        .astype(np.float64)
    )

    demo_rows = []
    all_frames = [live_main, live_wrist]
    for record in selected:
        table = pq.read_table(
            record.parquet_path, columns=["action", "observation.state"]
        )
        main = first_rgb_frame(record.video_path("cam_main"))
        wrist = first_rgb_frame(record.video_path("cam_wrist"))
        demo_rows.append(
            {
                "episode": record.episode_index,
                "main": main,
                "wrist": wrist,
                "state": np.asarray(table["observation.state"][0].as_py()),
                "action": np.asarray(table["action"][0].as_py()),
            }
        )
        all_frames.extend((main, wrist))

    features = encode_frames(backbone, all_frames, image_size, device)
    live_main_feature, live_wrist_feature = features[0], features[1]
    for index, row in enumerate(demo_rows):
        demo_main_feature = features[2 + index * 2]
        demo_wrist_feature = features[3 + index * 2]
        row["main_similarity"] = float(
            torch_f.cosine_similarity(live_main_feature, demo_main_feature, dim=0)
        )
        row["wrist_similarity"] = float(
            torch_f.cosine_similarity(live_wrist_feature, demo_wrist_feature, dim=0)
        )
        row["normal_similarity"] = float(
            (row["main_similarity"] + row["wrist_similarity"]) / 2.0
        )
        row["swapped_similarity"] = float(
            (
                torch_f.cosine_similarity(live_wrist_feature, demo_main_feature, dim=0)
                + torch_f.cosine_similarity(
                    live_main_feature, demo_wrist_feature, dim=0
                )
            )
            / 2.0
        )

    nearest_normal = max(demo_rows, key=lambda row: row["normal_similarity"])
    nearest_swapped = max(demo_rows, key=lambda row: row["swapped_similarity"])
    demo_pair_similarities = []
    for left in range(len(demo_rows)):
        for right in range(left + 1, len(demo_rows)):
            demo_pair_similarities.append(
                float(
                    (
                        torch_f.cosine_similarity(
                            features[2 + left * 2],
                            features[2 + right * 2],
                            dim=0,
                        )
                        + torch_f.cosine_similarity(
                            features[3 + left * 2],
                            features[3 + right * 2],
                            dim=0,
                        )
                    )
                    / 2.0
                )
            )
    nearest_prediction = (
        predict_action(
            backbone,
            bundle,
            nearest_normal["main"],
            nearest_normal["wrist"],
            task_index,
            image_size,
            device,
            observation_state=nearest_normal["state"],
            episode_progress=0.0,
        )
        .numpy()
        .astype(np.float64)
    )
    matched_live_main = match_channel_moments(live_main, nearest_normal["main"])
    matched_live_wrist = match_channel_moments(live_wrist, nearest_normal["wrist"])
    matched_both_prediction = (
        predict_action(
            backbone,
            bundle,
            matched_live_main,
            matched_live_wrist,
            task_index,
            image_size,
            device,
            observation_state=home,
            episode_progress=0.0,
        )
        .numpy()
        .astype(np.float64)
    )
    matched_main_prediction = (
        predict_action(
            backbone,
            bundle,
            matched_live_main,
            live_wrist,
            task_index,
            image_size,
            device,
            observation_state=home,
            episode_progress=0.0,
        )
        .numpy()
        .astype(np.float64)
    )
    matched_wrist_prediction = (
        predict_action(
            backbone,
            bundle,
            live_main,
            matched_live_wrist,
            task_index,
            image_size,
            device,
            observation_state=home,
            episode_progress=0.0,
        )
        .numpy()
        .astype(np.float64)
    )
    live_main_demo_wrist_prediction = (
        predict_action(
            backbone,
            bundle,
            live_main,
            nearest_normal["wrist"],
            task_index,
            image_size,
            device,
            observation_state=home,
            episode_progress=0.0,
        )
        .numpy()
        .astype(np.float64)
    )
    demo_main_live_wrist_prediction = (
        predict_action(
            backbone,
            bundle,
            nearest_normal["main"],
            live_wrist,
            task_index,
            image_size,
            device,
            observation_state=home,
            episode_progress=0.0,
        )
        .numpy()
        .astype(np.float64)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"policy_input_diagnostic_{args.backbone}_{task_name}_{stamp}"
    live_main_path = args.output_dir / f"{stem}_live_main.png"
    live_wrist_path = args.output_dir / f"{stem}_live_wrist.png"
    demo_main_path = args.output_dir / f"{stem}_nearest_demo_main.png"
    demo_wrist_path = args.output_dir / f"{stem}_nearest_demo_wrist.png"
    matched_main_path = args.output_dir / f"{stem}_matched_live_main.png"
    matched_wrist_path = args.output_dir / f"{stem}_matched_live_wrist.png"
    contact_path = args.output_dir / f"{stem}_comparison.jpg"
    report_path = args.output_dir / f"{stem}.txt"
    save_rgb(live_main_path, live_main)
    save_rgb(live_wrist_path, live_wrist)
    save_rgb(demo_main_path, nearest_normal["main"])
    save_rgb(demo_wrist_path, nearest_normal["wrist"])
    save_rgb(matched_main_path, matched_live_main)
    save_rgb(matched_wrist_path, matched_live_wrist)
    contact = np.vstack(
        (
            np.hstack(
                (
                    labeled_tile(live_main, "LIVE cam_main"),
                    labeled_tile(live_wrist, "LIVE cam_wrist"),
                )
            ),
            np.hstack(
                (
                    labeled_tile(
                        nearest_normal["main"],
                        f"DEMO {nearest_normal['episode']:06d} cam_main",
                    ),
                    labeled_tile(
                        nearest_normal["wrist"],
                        f"DEMO {nearest_normal['episode']:06d} cam_wrist",
                    ),
                )
            ),
        )
    )
    if not cv2.imwrite(str(contact_path), contact):
        raise RuntimeError(f"Failed to save {contact_path}")

    normal_arm_delta, normal_gripper_delta = action_delta(live_prediction, home)
    swapped_arm_delta, swapped_gripper_delta = action_delta(swapped_prediction, home)
    live_main_demo_wrist_delta = action_delta(live_main_demo_wrist_prediction, home)
    demo_main_live_wrist_delta = action_delta(demo_main_live_wrist_prediction, home)
    matched_both_delta = action_delta(matched_both_prediction, home)
    matched_main_delta = action_delta(matched_main_prediction, home)
    matched_wrist_delta = action_delta(matched_wrist_prediction, home)
    nearest_action_error = np.abs(nearest_prediction - nearest_normal["action"])
    with report_path.open("w", encoding="utf-8") as report:
        report.write("Live Policy Input Diagnostic\n")
        report.write("============================\n")
        report.write("Policy actuation: DISABLED\n")
        report.write(
            "Robot movement: "
            + ("dataset-home staging only\n" if args.execute_home else "DISABLED\n")
        )
        report.write(f"Task: {task_name}\n")
        report.write(f"Backbone: {args.backbone}\n")
        report.write(f"Policy SHA256: {assets.policy_sha256}\n")
        report.write(f"Camera backend: {args.camera_backend}\n")
        report.write(f"Camera main: {camera_properties[0]}\n")
        report.write(f"Camera wrist: {camera_properties[1]}\n")
        report.write(f"Camera pair skew: {pair_skew_ms:.3f} ms\n")
        report.write(
            f"Home {'observed' if args.execute_home else 'configured reference'}: "
            f"{home.tolist()}\n\n"
        )
        report.write(f"Inference profile steps: {args.profile_steps}\n")
        report.write(
            "Inference latency median/p95/max: "
            f"{float(np.median(profile_latencies_ms)):.3f} / "
            f"{float(np.percentile(profile_latencies_ms, 95)):.3f} / "
            f"{max(profile_latencies_ms):.3f} ms\n"
        )
        report.write(
            "Inference rate from median latency: "
            f"{1000.0 / float(np.median(profile_latencies_ms)):.3f} Hz\n\n"
        )
        report.write(f"Live normal prediction: {live_prediction.tolist()}\n")
        report.write(f"Live normal maximum arm delta: {normal_arm_delta:.7f} rad\n")
        report.write(f"Live normal gripper delta: {normal_gripper_delta:.7f} m\n")
        report.write(f"Live swapped prediction: {swapped_prediction.tolist()}\n")
        report.write(f"Live swapped maximum arm delta: {swapped_arm_delta:.7f} rad\n")
        report.write(f"Live swapped gripper delta: {swapped_gripper_delta:.7f} m\n\n")
        report.write(
            "Live-main + demo-wrist prediction: "
            f"{live_main_demo_wrist_prediction.tolist()}\n"
        )
        report.write(
            "Live-main + demo-wrist maximum arm delta: "
            f"{live_main_demo_wrist_delta[0]:.7f} rad\n"
        )
        report.write(
            "Live-main + demo-wrist gripper delta: "
            f"{live_main_demo_wrist_delta[1]:.7f} m\n"
        )
        report.write(
            "Demo-main + live-wrist prediction: "
            f"{demo_main_live_wrist_prediction.tolist()}\n"
        )
        report.write(
            "Demo-main + live-wrist maximum arm delta: "
            f"{demo_main_live_wrist_delta[0]:.7f} rad\n"
        )
        report.write(
            "Demo-main + live-wrist gripper delta: "
            f"{demo_main_live_wrist_delta[1]:.7f} m\n\n"
        )
        report.write(
            f"Color-matched both prediction: {matched_both_prediction.tolist()}\n"
        )
        report.write(
            f"Color-matched both maximum arm delta: {matched_both_delta[0]:.7f} rad\n"
        )
        report.write(
            f"Color-matched both gripper delta: {matched_both_delta[1]:.7f} m\n"
        )
        report.write(
            "Color-matched main-only maximum arm delta: "
            f"{matched_main_delta[0]:.7f} rad\n"
        )
        report.write(
            f"Color-matched main-only gripper delta: {matched_main_delta[1]:.7f} m\n"
        )
        report.write(
            "Color-matched wrist-only maximum arm delta: "
            f"{matched_wrist_delta[0]:.7f} rad\n"
        )
        report.write(
            "Color-matched wrist-only gripper delta: "
            f"{matched_wrist_delta[1]:.7f} m\n\n"
        )
        report.write(
            "Nearest normal-order demo: "
            f"episode {nearest_normal['episode']:06d}, "
            f"similarity={nearest_normal['normal_similarity']:.7f}\n"
        )
        report.write(
            "Nearest normal-order component similarities: "
            f"main={nearest_normal['main_similarity']:.7f}, "
            f"wrist={nearest_normal['wrist_similarity']:.7f}\n"
        )
        report.write(
            "Nearest swapped-order demo: "
            f"episode {nearest_swapped['episode']:06d}, "
            f"similarity={nearest_swapped['swapped_similarity']:.7f}\n"
        )
        report.write(
            "Demo-to-demo pair similarity min/median/max: "
            f"{min(demo_pair_similarities):.7f} / "
            f"{float(np.median(demo_pair_similarities)):.7f} / "
            f"{max(demo_pair_similarities):.7f}\n"
        )
        report.write(f"Nearest demo state: {nearest_normal['state'].tolist()}\n")
        report.write(f"Nearest demo action: {nearest_normal['action'].tolist()}\n")
        report.write(f"Nearest demo prediction: {nearest_prediction.tolist()}\n")
        report.write(
            f"Nearest demo prediction MAE: {float(nearest_action_error.mean()):.7f}\n"
        )
        report.write(
            "Nearest demo prediction maximum error: "
            f"{float(nearest_action_error.max()):.7f}\n\n"
        )
        report.write(f"Live main image: {live_main_path}\n")
        report.write(f"Live wrist image: {live_wrist_path}\n")
        report.write(f"Nearest demo main image: {demo_main_path}\n")
        report.write(f"Nearest demo wrist image: {demo_wrist_path}\n")
        report.write(f"Color-matched live main image: {matched_main_path}\n")
        report.write(f"Color-matched live wrist image: {matched_wrist_path}\n")
        report.write(f"Comparison image: {contact_path}\n")

    print(f"report: {report_path}")
    print(f"comparison: {contact_path}")


if __name__ == "__main__":
    main()
