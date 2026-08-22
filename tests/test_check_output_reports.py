import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
CHECKER = ROOT / "scripts" / "check_output_reports.py"


def run_checker(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--output-dir", str(output_dir)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_output_report_checker_accepts_an_empty_directory(tmp_path: Path) -> None:
    result = run_checker(tmp_path / "missing")

    assert result.returncode == 0
    assert "nothing to check" in result.stdout


def test_output_report_checker_rejects_a_report_without_verdict(
    tmp_path: Path,
) -> None:
    (tmp_path / "report.txt").write_text("unfinished\n", encoding="utf-8")

    result = run_checker(tmp_path)

    assert result.returncode == 1
    assert "missing valid Overall/Decision verdict" in result.stderr


def test_output_report_checker_accepts_a_valid_verdict(tmp_path: Path) -> None:
    (tmp_path / "report.txt").write_text("Decision: PASS\n", encoding="utf-8")

    result = run_checker(tmp_path)

    assert result.returncode == 0
    assert "Decision: PASS" in result.stdout
