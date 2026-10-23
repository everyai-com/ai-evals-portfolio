"""Re-run the CLIs the teardowns cite and save their output under reports/."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P01 = "projects/p01_trajectory_grader"
P03 = "projects/p03_calibrated_judge"
P07 = "projects/p07_significance_engine"
P11 = "projects/p11_counterfactual_replay"

COMMANDS: dict[str, list[str]] = {
    "01_grade.md": [sys.executable, "-m", f"{P01.replace('/', '.')}.cli", "grade",
                    "--trajectories", f"{P01}/data/trajectories.jsonl", "--golden", f"{P01}/data/golden",
                    "--tools", f"{P01}/data/tools.yaml"],
    "01_bisect.md": [sys.executable, "-m", f"{P11.replace('/', '.')}.cli", "bisect",
                     "--run", f"{P11}/data/corrupted_run.jsonl"],
    "02_kappa.md": [sys.executable, "-m", f"{P03.replace('/', '.')}.cli", "kappa", "--anchors", f"{P03}/data/anchors.jsonl"],
    "02_calibrate_biased.md": [sys.executable, "-m", f"{P03.replace('/', '.')}.cli", "calibrate",
                               "--anchors", f"{P03}/data/anchors.jsonl", "--judge", "biased"],
    "02_calibrate_unbiased.md": [sys.executable, "-m", f"{P03.replace('/', '.')}.cli", "calibrate",
                                 "--anchors", f"{P03}/data/anchors.jsonl", "--judge", "unbiased"],
    "03_compare.md": [sys.executable, "-m", f"{P07.replace('/', '.')}.cli", "compare",
                      "--a", f"{P07}/data/model_a.jsonl", "--b", f"{P07}/data/model_b.jsonl", "--metric", "success", "--n-boot", "10000", "--seed", "0"],
    "03_small.md": [sys.executable, "-m", f"{P07.replace('/', '.')}.cli", "compare",
                    "--a", f"{P07}/data/small_a.jsonl", "--b", f"{P07}/data/small_b.jsonl", "--metric", "success", "--seed", "0"],
    "03_seeds.md": [sys.executable, "-m", f"{P07.replace('/', '.')}.cli", "seeds", "--runs", f"{P07}/data/runs.jsonl"],
}


def main(argv: list[str] | None = None) -> int:
    out_dir = HERE / "reports"
    out_dir.mkdir(exist_ok=True)
    failures = 0
    for name, cmd in COMMANDS.items():
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        body = f"$ {' '.join(cmd[1:])}\n\n{proc.stdout}"
        if proc.returncode not in (0, 1):   # 1 = eval failed, which is a legitimate result
            body += f"\n[exit {proc.returncode}]\n{proc.stderr}"
            failures += 1
        (out_dir / name).write_text("```\n" + body + "\n```\n")
        print(f"{name}: exit {proc.returncode}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
