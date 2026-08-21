#!/usr/bin/env python3
"""Convert the LeRobot MP4 dataset into an HRP-style JPEG observation buffer."""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import yaml

from tcc_real_robot.hrp_action_space import (
    dataset_euler_pose_to_hrp_pose,
    hrp_state,
    measured_hrp_velocity,
)
from tcc_real_robot.policy_data import build_episode_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_v8_hrp_official_single_view_60.yaml"),
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/hrp_image_buffer_carrot_60.sqlite3"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int)
    return parser.parse_args()


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY,
            split TEXT NOT NULL,
            task_index INTEGER NOT NULL,
            episode_index INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            jpeg BLOB NOT NULL,
            state BLOB NOT NULL,
            action BLOB NOT NULL,
            UNIQUE(task_index, episode_index, frame_index)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS samples_split ON samples(split)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else Path(config["dataset"]["local_root"])
    ).expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.overwrite and output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    split = config["split"]
    records = build_episode_records(
        dataset_root,
        list(config["dataset"]["tasks"]),
        int(config["seed"]),
        (
            int(split["train_episodes_per_task"]),
            int(split["validation_episodes_per_task"]),
            int(split["test_episodes_per_task"]),
        ),
    )
    if args.limit_episodes is not None:
        records = records[: args.limit_episodes]

    with sqlite3.connect(output) as connection:
        initialize_database(connection)
        for number, record in enumerate(records, start=1):
            existing = connection.execute(
                "SELECT COUNT(*) FROM samples WHERE task_index = ? AND episode_index = ?",
                (record.task_index, record.episode_index),
            ).fetchone()
            if existing is not None and int(existing[0]) > 0:
                print(f"[{number}/{len(records)}] cached {record.task_name}/{record.episode_index}")
                continue
            table = pq.read_table(
                record.parquet_path,
                columns=[
                    "observation.state",
                    "observation.cartesian_position",
                    "timestamp",
                ],
            )
            joint_states = np.asarray(
                table["observation.state"].to_pylist(), dtype=np.float32
            )
            cartesian = np.asarray(
                table["observation.cartesian_position"].to_pylist(),
                dtype=np.float32,
            )
            timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            if len(cartesian) != len(joint_states):
                raise RuntimeError(
                    f"Pose/state length mismatch in {record}: "
                    f"{len(cartesian)} != {len(joint_states)}"
                )
            states = np.stack(
                [
                    hrp_state(
                        dataset_euler_pose_to_hrp_pose(pose),
                        float(joints[6]),
                    )
                    for pose, joints in zip(cartesian, joint_states)
                ]
            )
            actions = np.stack(
                [
                    measured_hrp_velocity(
                        states[index],
                        states[index + 1],
                        float(timestamps[index + 1] - timestamps[index]),
                    )
                    for index in range(len(states) - 1)
                ]
            )
            rows = []
            with av.open(str(record.video_path("cam_main"))) as container:
                for frame_index, frame in enumerate(
                    container.decode(container.streams.video[0])
                ):
                    buffer = io.BytesIO()
                    frame.to_image().save(buffer, format="JPEG", quality=95)
                    if frame_index >= len(actions):
                        break
                    rows.append(
                        (
                            record.split,
                            record.task_index,
                            record.episode_index,
                            frame_index,
                            buffer.getvalue(),
                            states[frame_index].tobytes(),
                            actions[frame_index].tobytes(),
                        )
                    )
            if len(rows) != len(actions) or len(states) != len(actions) + 1:
                raise RuntimeError(
                    f"Unsynchronized episode {record}: "
                    f"transitions={len(rows)}, actions={len(actions)}, "
                    f"states={len(states)}"
                )
            connection.executemany(
                """
                INSERT INTO samples(
                    split, task_index, episode_index, frame_index, jpeg, state, action
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()
            print(f"[{number}/{len(records)}] wrote {len(rows)} samples")
        manifest = {
            "dataset_revision": config["dataset"]["revision"],
            "config": str(args.config),
            "camera": "cam_main",
            "source_pose_semantics": "dataset_xyz_plus_intrinsic_xyz_roll_pitch_yaw",
            "state_semantics": "trossen_xyz_plus_angle_axis_plus_gripper_position",
            "action_semantics": (
                "trossen_base_frame_vx_vy_vz_wx_wy_wz_plus_gripper_velocity"
            ),
            "jpeg_quality": 95,
            "episodes": len(records),
            "actuation_enabled": False,
        }
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("manifest", json.dumps(manifest, sort_keys=True)),
        )
        connection.commit()
    print(output)


if __name__ == "__main__":
    main()
