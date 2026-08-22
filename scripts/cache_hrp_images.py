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

from tcc_real_robot.hrp_image_data import (
    JOINT_ACTION_SEMANTICS,
    JOINT_STATE_SEMANTICS,
    require_complete_joint_position_buffer,
    validate_joint_position_manifest,
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
        default=Path("runs/hrp_image_buffer_carrot_joint_position.sqlite3"),
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
    connection.execute("CREATE INDEX IF NOT EXISTS samples_split ON samples(split)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    dataset_root = (
        (
            args.dataset_root
            if args.dataset_root is not None
            else Path(config["dataset"]["local_root"])
        )
        .expanduser()
        .resolve()
    )
    output = args.output.expanduser().resolve()
    if args.overwrite and output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = build_episode_records(
        dataset_root,
        list(config["dataset"]["tasks"]),
        int(config["seed"]),
        (int(config["dataset"]["demonstrations_per_task"]), 0, 0),
        shuffle=False,
    )
    if args.limit_episodes is not None:
        records = records[: args.limit_episodes]

    with sqlite3.connect(output) as connection:
        initialize_database(connection)
        existing_samples = int(
            connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        )
        if existing_samples:
            manifest_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'manifest'"
            ).fetchone()
            if manifest_row is None:
                raise RuntimeError(
                    "Existing image buffer has samples but no manifest; rerun with "
                    "--overwrite"
                )
            validate_joint_position_manifest(json.loads(str(manifest_row[0])))
        for number, record in enumerate(records, start=1):
            existing = connection.execute(
                "SELECT COUNT(*) FROM samples WHERE task_index = ? AND episode_index = ?",
                (record.task_index, record.episode_index),
            ).fetchone()
            if existing is not None and int(existing[0]) > 0:
                print(
                    f"[{number}/{len(records)}] cached {record.task_name}/{record.episode_index}"
                )
                continue
            table = pq.read_table(
                record.parquet_path,
                columns=["observation.state", "action"],
            )
            states = np.asarray(
                table["observation.state"].to_pylist(), dtype=np.float32
            )
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if (
                states.shape != actions.shape
                or states.ndim != 2
                or states.shape[1] != 7
            ):
                raise RuntimeError(
                    f"Expected aligned [N, 7] state/action in {record}, got "
                    f"{states.shape} and {actions.shape}"
                )
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise RuntimeError(f"Non-finite state/action in {record}")
            rows = []
            with av.open(str(record.video_path("cam_main"))) as container:
                for frame_index, frame in enumerate(
                    container.decode(container.streams.video[0])
                ):
                    buffer = io.BytesIO()
                    frame.to_image().save(buffer, format="JPEG", quality=95)
                    if frame_index >= len(states):
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
            if len(rows) != len(states):
                raise RuntimeError(
                    f"Unsynchronized episode {record}: "
                    f"video={len(rows)}, states={len(states)}, actions={len(actions)}"
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
            "tasks": list(config["dataset"]["tasks"]),
            "camera": "cam_main",
            "state_semantics": JOINT_STATE_SEMANTICS,
            "action_semantics": JOINT_ACTION_SEMANTICS,
            "action_source": "original_lerobot_action_column",
            "driver_command": "set_all_positions",
            "action_leads_measured_state_frames": int(
                config["dataset"]["action_leads_measured_state_frames"]
            ),
            "jpeg_quality": 95,
            "episodes": len(records),
            "episodes_per_task": int(config["dataset"]["demonstrations_per_task"]),
            "frames_per_episode": int(config["evaluation"]["max_rollout_steps"]),
            "samples": sum(
                int(row[0])
                for row in connection.execute(
                    "SELECT COUNT(*) FROM samples GROUP BY task_index, episode_index"
                )
            ),
            "split_protocol": config["split"],
            "actuation_enabled": False,
        }
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("manifest", json.dumps(manifest, sort_keys=True)),
        )
        connection.commit()
    if args.limit_episodes is None:
        require_complete_joint_position_buffer(
            output,
            dataset_revision=str(config["dataset"]["revision"]),
            tasks=[str(task) for task in config["dataset"]["tasks"]],
            episodes_per_task=int(config["dataset"]["demonstrations_per_task"]),
            frames_per_episode=int(config["evaluation"]["max_rollout_steps"]),
        )
    print(output)


if __name__ == "__main__":
    main()
