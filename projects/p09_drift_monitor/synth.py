"""Synthetic production traffic: per-day trajectory logs whose quality decays after `drift_start_day`."""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from evalkit.llm import estimate_cost
from evalkit.trajectory import Trajectory
from projects.p09_drift_monitor.core import HALLUCINATION_MARKERS

TENANTS = ["pizza-us", "pizza-uk", "wings-ca"]
MODEL = "mock-system-one"
HOUR_WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 8, 9, 8, 6, 5, 6, 8, 10, 10, 8, 5, 3, 2], float)

GOOD = {"p_empty": 0.03, "p_error": 0.03, "p_halluc": 0.02, "p_bad_args": 0.02, "len_mean": 55.0, "extra_tools": 0.0}
BAD = {"p_empty": 0.14, "p_error": 0.12, "p_halluc": 0.12, "p_bad_args": 0.06, "len_mean": 110.0, "extra_tools": 1.2}

TASKS = [("order_status", ["lookup_order"], "Your order {oid} is {state} and should arrive in {m} minutes."),
         ("menu", ["search_menu"], "We have {n} options matching {q}; the most popular is {item}."),
         ("refund", ["lookup_order", "issue_refund"], "A refund of ${amt:.2f} for order {oid} has been issued."),
         ("hours", ["get_store_hours"], "Store {sid} is open from {o}:00 to {c}:00 today.")]
FILLER = ("Let me know if there is anything else I can help with. " * 20).split(". ")


def day_params(day: int, drift_start_day: int) -> dict[str, float]:
    """Linear ramp from GOOD to BAD over two days starting at drift_start_day."""
    if day < drift_start_day:
        return dict(GOOD)
    f = min(1.0, 0.6 + 0.4 * (day - drift_start_day))  # 60% of the way on day 0, 100% from day 1
    return {k: GOOD[k] + f * (BAD[k] - GOOD[k]) for k in GOOD}


def _traj(rng: np.random.Generator, day_str: str, i: int, params: dict[str, float]) -> dict[str, Any]:
    tenant = TENANTS[int(rng.integers(len(TENANTS)))]
    hour = int(rng.choice(24, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum()))
    task, tools, template = TASKS[int(rng.integers(len(TASKS)))]
    oid, sid = f"O{int(rng.integers(10000, 99999))}", f"S{int(rng.integers(100, 999))}"
    t = Trajectory(task_id=f"{day_str}-{i:05d}", model=MODEL,
                   meta={"ts": f"{day_str}T{hour:02d}:{int(rng.integers(60)):02d}:00Z", "hour": hour,
                         "tenant": tenant, "task_type": task})
    it, ot = int(rng.integers(300, 700)), int(rng.integers(20, 80))
    t.add_llm("plan: " + task, model=MODEL, input_tokens=it, output_tokens=ot,
              latency_ms=float(rng.gamma(4, 60)), cost_usd=estimate_cost(MODEL, it, ot))
    failed = False
    n_extra = int(rng.poisson(params["extra_tools"]))
    for name in tools + ["lookup_order"] * n_extra:
        args = {"lookup_order": {"order_id": oid}, "search_menu": {"query": "veggie"},
                "issue_refund": {"order_id": oid, "amount": 12.5}, "get_store_hours": {"store_id": sid}}[name]
        if rng.random() < params["p_bad_args"]:
            args = {k: v for k, v in list(args.items())[1:]}  # drop the first required arg
            failed = True
        err = None
        if rng.random() < params["p_error"]:
            err, failed = "ToolTimeout: upstream took > 5s", True
        t.add_tool(name, args, result=None if err else {"ok": True, "id": oid}, error=err,
                   latency_ms=float(rng.gamma(3, 80)))
    if rng.random() < params["p_empty"]:
        answer, failed = "", True
    else:
        answer = template.format(oid=oid, state="out for delivery", m=int(rng.integers(10, 45)), n=int(rng.integers(2, 9)),
                                 q="veggie", item="Veggie Supreme", amt=12.5, sid=sid, o=10, c=22)
        n_words = max(8, int(rng.normal(params["len_mean"], params["len_mean"] * 0.25)))
        answer = answer + " " + ". ".join(FILLER[: max(1, (n_words - 12) // 10)]) + "."
        if rng.random() < params["p_halluc"]:
            answer += " " + HALLUCINATION_MARKERS[int(rng.integers(len(HALLUCINATION_MARKERS)))].capitalize() + "."
    it2, ot2 = int(rng.integers(400, 900)), max(5, len(answer) // 4)
    t.add_llm("compose", model=MODEL, input_tokens=it2, output_tokens=ot2,
              latency_ms=float(rng.gamma(4, 70)), cost_usd=estimate_cost(MODEL, it2, ot2))
    t.finish(answer)
    fb_p_down = 0.45 if failed else 0.02
    u = rng.random()
    t.meta["feedback"] = "down" if u < fb_p_down else ("up" if u < fb_p_down + 0.10 else None)
    return t.to_dict()


def date_for(day: int, start: str = "2026-10-01") -> str:
    return (dt.date.fromisoformat(start) + dt.timedelta(days=day)).isoformat()


def make_day(day: int, drift_start_day: int = 10, seed: int = 0, per_day: int = 1000,
             start: str = "2026-10-01") -> list[dict[str, Any]]:
    rng = np.random.default_rng([seed, day])
    params = day_params(day, drift_start_day)
    d = date_for(day, start)
    return [_traj(rng, d, i, params) for i in range(per_day)]


def make_traffic(days: int = 14, drift_start_day: int = 10, seed: int = 0, per_day: int = 1000,
                 start: str = "2026-10-01") -> dict[str, list[dict[str, Any]]]:
    """{date: [trajectory dicts]} for `days` consecutive days; quality decays from drift_start_day."""
    return {date_for(d, start): make_day(d, drift_start_day, seed, per_day, start) for d in range(days)}
