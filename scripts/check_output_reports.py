#!/usr/bin/env python3
"""Require every committed TXT output report to contain a valid verdict."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

VERDICT = re.compile(
    r"^(?:Overall: (?:PASS|FAIL)|Decision: (?:PASS|FAIL|BLOCKED).*)$",
    re.MULTILINE,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that every output TXT report has a verdict."
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    reports = sorted(output_dir.rglob("*.txt"))
    if not reports:
        raise SystemExit(f"No TXT reports found under {output_dir}")

    failures: list[str] = []
    for report in reports:
        match = VERDICT.search(report.read_text(encoding="utf-8"))
        if match is None:
            failures.append(f"{report}: missing valid Overall/Decision verdict")
        else:
            print(f"{report}: {match.group(0)}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
