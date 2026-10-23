"""p02 Shadow Routing Comparator: mirror a slice of traffic to a candidate model, diff, report."""
from projects.p02_shadow_router.core import (
    DiffResult, JsonlStore, MemoryStore, ShadowAgentRunner, ShadowProxy, TrajectoryDiff,
    should_shadow,
)
from projects.p02_shadow_router.report import ReportBuilder, load_thresholds, recommend

__all__ = ["DiffResult", "JsonlStore", "MemoryStore", "ShadowAgentRunner", "ShadowProxy",
           "TrajectoryDiff", "should_shadow", "ReportBuilder", "load_thresholds", "recommend"]
