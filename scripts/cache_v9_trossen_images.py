#!/usr/bin/env python3
"""Cache synchronized Trossen RGB observations and original joint goals."""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
from contextlib import ExitStack
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import yaml

from tcc_real_robot.policy_data import build_episode_records
from tcc_real_robot.trossen_image_data import (
    JOINT_ACTION_SEMANTICS,
    JOINT_STATE_SEMANTICS,
    require_complete_trossen_buffer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_v9_r3m_robomimic_proprio_100.yaml"),
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v9_trossen_dual_camera.sqlite3")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY,
            split TEXT NOT NULL,
            task_index INTEGER NOT NULL,
            episode_index INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            jpeg_main BLOB NOT NULL,
            jpeg_wrist BLOB NOT NULL,
            state BLOB NOT NULL,
            action BLOB NOT NULL,
            UNIQUE(task_index, episode_index, frame_index)
        )
        """
    )
    connection.execute("CREATE INDEX samples_split ON samples(split)")
    connection.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    dataset_root = Path(
        args.dataset_root or config["dataset"]["local_root"]
    ).expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = build_episode_records(
        dataset_root,
        [str(task) for task in config["dataset"]["tasks"]],
        int(config["seed"]),
        (int(config["dataset"]["demonstrations_per_task"]), 0, 0),
        shuffle=False,
    )
    with sqlite3.connect(output) as connection:
        initialize_database(connection)
        for number, record in enumerate(records, start=1):
            table = pq.read_table(
                record.parquet_path, columns=["observation.state", "action"]
            )
            states = np.asarray(
                table["observation.state"].to_pylist(), dtype=np.float32
            )
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if states.shape != actions.shape or states.shape[1:] != (7,):
                raise RuntimeError(f"Invalid state/action shapes in {record}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise RuntimeError(f"Non-finite state/action in {record}")
            rows = []
            with ExitStack() as stack:
                main_container = stack.enter_context(
                    av.open(str(record.video_path("cam_main")))
                )
                wrist_container = stack.enter_context(
                    av.open(str(record.video_path("cam_wrist")))
                )
                main_frames = main_container.decode(main_container.streams.video[0])
                wrist_frames = wrist_container.decode(wrist_container.streams.video[0])
                for frame_index, (main_frame, wrist_frame) in enumerate(
                    zip(main_frames, wrist_frames, strict=True)
                ):
                    if frame_index >= len(states):
                        raise RuntimeError(f"Videos exceed tabular data in {record}")
                    main_buffer = io.BytesIO()
                    wrist_buffer = io.BytesIO()
                    main_frame.to_image().save(main_buffer, format="JPEG", quality=95)
                    wrist_frame.to_image().save(
                        wrist_buffer, format="JPEG", quality=95
                    )
                    rows.append(
                        (
                            "train",
                            record.task_index,
                            record.episode_index,
                            frame_index,
                            main_buffer.getvalue(),
                            wrist_buffer.getvalue(),
                            states[frame_index].tobytes(),
                            actions[frame_index].tobytes(),
                        )
                    )
            if len(rows) != len(states):
                raise RuntimeError(f"Unsynchronized episode {record}")
            connection.executemany(
                "INSERT INTO samples(split, task_index, episode_index, frame_index, "
                "jpeg_main, jpeg_wrist, state, action) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            connection.commit()
            print(f"[{number}/{len(records)}] wrote {len(rows)} samples")
        manifest = {
            "dataset_revision": config["dataset"]["revision"],
            "config": str(args.config),
            "tasks": list(config["dataset"]["tasks"]),
            "cameras": ["cam_main", "cam_wrist"],
            "state_semantics": JOINT_STATE_SEMANTICS,
            "action_semantics": JOINT_ACTION_SEMANTICS,
            "action_source": "original_lerobot_action_column",
            "driver_command": "set_all_positions",
            "jpeg_quality": 95,
            "episodes": len(records),
            "episodes_per_task": int(config["dataset"]["demonstrations_per_task"]),
            "frames_per_episode": int(config["evaluation"]["max_rollout_steps"]),
            "samples": len(records) * int(config["evaluation"]["max_rollout_steps"]),
            "actuation_enabled": False,
        }
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("manifest", json.dumps(manifest, sort_keys=True)),
        )
        connection.commit()
    require_complete_trossen_buffer(
        output,
        dataset_revision=str(config["dataset"]["revision"]),
        tasks=[str(task) for task in config["dataset"]["tasks"]],
        episodes_per_task=int(config["dataset"]["demonstrations_per_task"]),
        frames_per_episode=int(config["evaluation"]["max_rollout_steps"]),
    )
    print(output)


if __name__ == "__main__":
    main()
