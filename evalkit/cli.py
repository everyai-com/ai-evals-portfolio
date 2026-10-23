"""`evalkit` umbrella CLI: dispatches to each project's CLI."""
from __future__ import annotations

import importlib
import sys

PROJECTS = {
    "trajectory": "projects.p01_trajectory_grader.cli",
    "shadow": "projects.p02_shadow_router.cli",
    "judge": "projects.p03_calibrated_judge.cli",
    "gate": "projects.p04_ci_regression_gate.cli",
    "rag": "projects.p05_rag_adversarial.cli",
    "dpo": "projects.p06_dpo_flywheel.cli",
    "stats": "projects.p07_significance_engine.cli",
    "redteam": "projects.p08_agent_redteam_fuzzer.cli",
    "drift": "projects.p09_drift_monitor.cli",
    "pareto": "projects.p10_pareto_dashboard.cli",
    "replay": "projects.p11_counterfactual_replay.cli",
    "synth": "projects.p12_synthetic_edgecases.cli",
    "context": "projects.p13_context_eviction.cli",
    "contamination": "projects.p14_contamination_checker.cli",
    "harness": "projects.p16_unified_harness.cli",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: evalkit <project> [args]\n\nprojects: " + ", ".join(PROJECTS))
        return 0
    name, rest = argv[0], argv[1:]
    if name not in PROJECTS:
        print(f"unknown project {name!r}; choose from {', '.join(PROJECTS)}", file=sys.stderr)
        return 2
    mod = importlib.import_module(PROJECTS[name])
    return int(mod.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
