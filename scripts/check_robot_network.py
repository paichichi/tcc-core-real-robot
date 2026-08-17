#!/usr/bin/env python3
"""Read-only TCP reachability check; this script sends no robot commands."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

from tcc_real_robot.config import assert_actuation_disabled, load_yaml


def probe(host: str, port: int, timeout: float) -> str:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "reachable"
    except OSError as exc:
        return f"unreachable ({exc})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/robot.yaml")
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    assert_actuation_disabled(config)
    host = config["robot"]["controller_ip"]
    tcp_port = 50001
    print(f"{host}:{tcp_port} {probe(host, tcp_port, args.timeout)}")


if __name__ == "__main__":
    main()
