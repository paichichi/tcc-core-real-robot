# TCC-Core Real Robot

Real-robot evaluation workspace for transferring TCC-Core visual backbones to a
Trossen AI Solo follower arm on four pick-and-place tasks.

This repository starts deliberately with **data and environment inspection
only**. It contains no command that actuates the arm. Motion-control code must
be introduced separately after the robot, firmware, driver, workspace limits,
emergency stop, and action convention have been verified on site.

## Experimental scope

- Robot: Trossen AI Solo follower (`trossen_ai_solo`)
- Tasks: carrot, pineapple, starfruit, and strawberry pick-and-place
- Demonstrations: 100 per task
- Cameras: `cam_main` and `cam_wrist`, 640 x 480 RGB at 20 FPS
- Dataset: `UoA-Trossen-Arm/pick_and_place_4_object_diverse`
- Policy protocol: to be locked after the first read-only data audit

The dataset metadata reports 7-D action and state vectors. Their numerical
ranges suggest absolute joint targets plus a gripper value, but this is a
working hypothesis and **must not be used to command the robot until verified**.

## Repository layout

```text
configs/             Versioned experiment and hardware assumptions
docs/                Safety and dataset notes
scripts/             Read-only audit utilities
src/tcc_real_robot/  Reusable Python package
tests/               Configuration and safety-invariant tests
```

## Setup

Ubuntu 22.04 is the target runtime used beside the robot.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The Trossen driver is intentionally an optional dependency:

```bash
python -m pip install -e '.[robot]'
```

## Safe first checks

These commands do not send robot motion commands:

```bash
python scripts/audit_dataset.py --metadata-only
python scripts/check_robot_network.py
python scripts/inspect_robot.py
python scripts/monitor_robot.py --duration 5 --rate 20
python scripts/robot_preflight.py
pytest
```

After a passing preflight and an on-site safety check, the explicitly gated
position-mode hold diagnostic can be run with
`python scripts/test_position_hold.py --execute`. It sends no motion target.

If that diagnostic passes, the next gated test is
`python scripts/test_current_position_hold.py --execute`. It reads the current
seven-joint position and sends that exact value back as the position target;
it does not add an offset.

`inspect_robot.py` follows the vendor's configure/read/cleanup lifecycle. It
does not clear controller faults, change joint modes, or send motion commands.
`monitor_robot.py` applies the same restrictions while sampling state repeatedly.

Before any future actuation, complete every item in [docs/SAFETY.md](docs/SAFETY.md).

## Data policy

Datasets, videos, checkpoints, run directories, credentials, and local hardware
settings are excluded from Git. Record the Hugging Face dataset revision in
`configs/experiment.yaml` so each experiment remains reproducible.
