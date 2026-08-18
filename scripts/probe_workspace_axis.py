#!/usr/bin/env python3
"""Interactively probe one Cartesian workspace direction in tiny steps."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tcc_real_robot.config import assert_actuation_disabled, load_yaml
from tcc_real_robot.policy_home import PolicyHomeSession
from tcc_real_robot.workspace_probe import (
    format_workspace_probe_report,
    next_probe_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move one Cartesian direction by one tiny step per Enter press. "
            "Type q then Enter to record the boundary and return home."
        )
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument(
        "--direction", choices=("positive", "negative"), default="positive"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/robot.yaml"))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/workspace_probes"))
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Automatically send one bounded step at a time. Press q without "
            "Enter to stop after the current step and return home."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that Enter presses move the real arm.",
    )
    return parser.parse_args()


@contextmanager
def raw_terminal_keys() -> Any:
    """Temporarily enable immediate single-key input and always restore the TTY."""
    if not sys.stdin.isatty():
        raise RuntimeError("--auto requires an interactive terminal")
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def poll_stop_key(timeout_s: float = 0.0) -> bool:
    """Return true when q is available; fail closed if terminal input closes."""
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return False
    key = sys.stdin.read(1)
    if key == "":
        raise EOFError("terminal input closed")
    return key.lower() == "q"


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise SystemExit("Refusing to move the arm without --execute")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    config = load_yaml(args.config)
    assert_actuation_disabled(config)
    settings = config["workspace_probe"]
    step_m = float(settings["step_m"])
    goal_time_s = float(settings["goal_time_s"])
    return_goal_time_s = float(settings["return_goal_time_s"])
    max_tracking_error_m = float(settings["max_tracking_error_m"])
    trajectory_check_samples = int(settings["trajectory_check_samples"])
    limit_key = f"{args.axis}_{args.direction}"
    hard_travel_limit_m = float(settings["hard_travel_limits_m"][limit_key])

    try:
        import trossen_arm
    except ImportError as exc:
        raise SystemExit(
            "Trossen driver is not installed. Run: pip install -e '.[robot]'"
        ) from exc

    session = PolicyHomeSession(trossen_arm, config, args.timeout)
    origin: list[float] = []
    current_target: list[float] = []
    last_safe: list[float] = []
    points: list[dict[str, Any]] = []
    cumulative_travel_m = 0.0
    returned_home = False
    idle_restored = False
    stop_reason = "not_started"
    failure = ""
    normal_stop = False
    try:
        print(f"Moving slowly to {config['robot']['home_name']}...")
        session.prepare()
        origin = session.read_cartesian_positions()
        current_target = origin.copy()
        last_safe = origin.copy()
        trigger = "per automatic step" if args.auto else "per Enter"
        print(
            f"Ready: {args.axis} {args.direction}, "
            f"{step_m * 1000:.1f} mm {trigger}."
        )
        if args.auto:
            print("AUTO starts in 3 seconds. Press q to stop after the current step.")
            for remaining in (3, 2, 1):
                print(f"Starting in {remaining}...")
                time.sleep(1.0)
            terminal_context = raw_terminal_keys()
        else:
            print("Press Enter for ONE step. Type q then Enter to stop and return home.")
            terminal_context = nullcontext()

        with terminal_context:
            while True:
                if args.auto:
                    if poll_stop_key():
                        stop_reason = "operator_q"
                        normal_stop = True
                        break
                else:
                    try:
                        response = input("probe> ").strip().lower()
                    except EOFError:
                        stop_reason = "terminal_input_closed"
                        break
                    if response == "q":
                        stop_reason = "operator_q"
                        normal_stop = True
                        break
                    if response:
                        print(
                            "No motion sent. Press Enter for one step, or q then Enter."
                        )
                        continue
                target, travel_m, reached_limit = next_probe_target(
                    origin,
                    current_target,
                    axis=args.axis,
                    direction=args.direction,
                    step_m=step_m,
                    hard_travel_limit_m=hard_travel_limit_m,
                )
                observed = session.move_cartesian(
                    target,
                    goal_time_s=goal_time_s,
                    trajectory_check_samples=trajectory_check_samples,
                )
                tracking_error = max(
                    abs(actual - expected)
                    for actual, expected in zip(observed[:3], target[:3], strict=True)
                )
                if tracking_error > max_tracking_error_m:
                    raise RuntimeError(
                        f"Tracking error {tracking_error:.6f} m exceeds "
                        f"{max_tracking_error_m:.6f} m"
                    )
                current_target = target
                last_safe = observed
                cumulative_travel_m = travel_m
                points.append(
                    {
                        "step": len(points) + 1,
                        "travel_m": travel_m,
                        "tracking_error_m": tracking_error,
                        "cartesian": observed,
                    }
                )
                print(
                    f"accepted step={len(points)} travel={travel_m:.3f} m "
                    f"position={observed[:3]}"
                )
                if args.auto and poll_stop_key():
                    stop_reason = "operator_q"
                    normal_stop = True
                    break
                if reached_limit:
                    stop_reason = "software_hard_travel_limit"
                    normal_stop = True
                    print("Software hard travel limit reached; returning home.")
                    break
    except KeyboardInterrupt:
        stop_reason = "operator_keyboard_interrupt"
        print("\nInterrupted: no return motion will be sent; restoring Idle.")
    except Exception as exc:  # noqa: BLE001 - preserve hardware failure report
        stop_reason = "exception"
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if normal_stop and origin:
            try:
                session.move_cartesian(
                    origin,
                    goal_time_s=return_goal_time_s,
                    trajectory_check_samples=trajectory_check_samples,
                )
                returned_home = True
            except Exception as exc:  # noqa: BLE001 - continue to Idle cleanup
                failure = f"Return failed: {type(exc).__name__}: {exc}"
        try:
            session.close()
            idle_restored = True
        except Exception as exc:  # noqa: BLE001 - preserve cleanup failure
            cleanup_failure = f"Idle cleanup failed: {type(exc).__name__}: {exc}"
            failure = f"{failure}; {cleanup_failure}" if failure else cleanup_failure

    passed = normal_stop and returned_home and idle_restored and not failure
    report = {
        "passed": passed,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "axis": args.axis,
        "direction": args.direction,
        "step_m": step_m,
        "hard_travel_limit_m": hard_travel_limit_m,
        "stop_reason": stop_reason,
        "returned_home": returned_home,
        "idle_restored": idle_restored,
        "origin_cartesian": origin,
        "last_safe_cartesian": last_safe,
        "cumulative_travel_m": cumulative_travel_m,
        "points": points,
        "failure": failure,
    }
    report_text = format_workspace_probe_report(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / (
        f"workspace_probe_{args.axis}_{args.direction}_{stamp}.txt"
    )
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"report: {output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
