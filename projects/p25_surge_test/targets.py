"""Targets the harness can drive, all behind one protocol: `call(Request) -> Response`.

* `FunctionTarget`  — any Python callable (real threads, wall clock)
* `HttpTarget`      — urllib GET/POST against a real URL
* `SimulatedTarget` — a capacity model (lognormal service time, concurrency limit, bounded queue,
                      error rate rising past the knee, cold start) that runs on a virtual clock so
                      tests take milliseconds and the README can show a realistic curve
* `RateLimiter`     — token buckets (per user + global) wrapped around any target; rejected
                      requests get a 429 and never reach the target

Simulated latency model (all times in ms, `n` = requests in flight when this one arrives,
`c` = concurrency_limit, `q` = queue_limit, `S` = a lognormal service sample with mean service_ms):

    n <  c        : latency = S * (1 + d * (n/c)^2)                   status 200   (d = degradation, 0.1)
    c <= n < c+q  : latency = (n-c+1) * service_ms / c + S             status 500 w.p. error_slope*(n-c+1)/q
    n >= c+q      : latency ~ 1 ms (fast reject)                       status 503
    t < cold_start_window_s : + cold_start_ms

So the knee sits where in-flight reaches `c`; by Little's law that is `c / (service_s * (1+d))`
requests per second, i.e. `nominal_capacity_users(rps_per_user)` users.
"""
from __future__ import annotations

import heapq
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class Request:
    user_id: int
    t: float                 # scheduled start, seconds since scenario start (virtual or wall)
    path: str = "/"
    seq: int = 0


@dataclass
class Response:
    status: int
    latency_ms: float
    bytes: int = 0

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400

    @property
    def server_error(self) -> bool:
        return self.status >= 500

    @property
    def rate_limited(self) -> bool:
        return self.status == 429


class Target(Protocol):
    def call(self, request: Request) -> Response: ...


class FunctionTarget:
    """Wrap any callable. It may return a Response, an int status, (status, bytes) or anything
    else (-> 200). Latency is measured on the wall clock."""

    def __init__(self, fn: Callable[[Request], Any]):
        self.fn = fn

    def call(self, request: Request) -> Response:
        t0 = time.perf_counter()
        try:
            out = self.fn(request)
        except Exception:  # a crashing handler is a 500, not a crashed harness  # noqa: BLE001
            return Response(500, (time.perf_counter() - t0) * 1000, 0)
        ms = (time.perf_counter() - t0) * 1000
        if isinstance(out, Response):
            return Response(out.status, out.latency_ms or ms, out.bytes)
        if isinstance(out, int):
            return Response(out, ms, 0)
        if isinstance(out, tuple) and len(out) == 2:
            return Response(int(out[0]), ms, int(out[1]))
        return Response(200, ms, len(out) if isinstance(out, (bytes, str)) else 0)


class HttpTarget:
    """Real HTTP via urllib. HTTP errors keep their status; connection errors / timeouts are 599."""

    def __init__(self, url: str, timeout: float = 10.0, method: str = "GET", headers: dict[str, str] | None = None):
        self.url, self.timeout, self.method = url, timeout, method
        self.headers = {"User-Agent": "p25-surge-test/0.1", **(headers or {})}

    def call(self, request: Request) -> Response:
        url = self.url if request.path in ("/", "") else self.url.rstrip("/") + "/" + request.path.lstrip("/")
        req = urllib.request.Request(url, method=self.method, headers=self.headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read()
                return Response(r.status, (time.perf_counter() - t0) * 1000, len(body))
        except urllib.error.HTTPError as e:
            return Response(e.code, (time.perf_counter() - t0) * 1000, 0)
        except (urllib.error.URLError, TimeoutError, OSError):
            return Response(599, (time.perf_counter() - t0) * 1000, 0)


class SimulatedTarget:
    """Capacity model on a virtual clock (`request.t`) or the wall clock (`virtual_clock=False`,
    where `call` also sleeps for the modelled latency so ThreadPool runs feel real)."""

    def __init__(self, service_ms: float = 50.0, service_cv: float = 0.3, concurrency_limit: int = 150,
                 queue_limit: int = 150, error_slope: float = 0.5, cold_start_ms: float = 0.0,
                 cold_start_window_s: float = 0.0, response_bytes: int = 2048, degradation: float = 0.1,
                 seed: int = 0, virtual_clock: bool = True):
        self.service_ms = float(service_ms)
        self.service_cv = float(service_cv)
        self.concurrency_limit = int(concurrency_limit)
        self.queue_limit = int(queue_limit)
        self.error_slope = float(error_slope)
        self.cold_start_ms = float(cold_start_ms)
        self.cold_start_window_s = float(cold_start_window_s)
        self.response_bytes = int(response_bytes)
        self.degradation = float(degradation)
        self.seed = seed
        self.virtual_clock = virtual_clock
        self.rng = np.random.default_rng(seed)
        self._heap: list[float] = []          # completion times of in-flight requests
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._svc: np.ndarray = np.zeros(0)
        self._u: np.ndarray = np.zeros(0)
        self._i = 0
        self.calls = 0
        self.peak_in_flight = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], seed: int = 0, virtual_clock: bool = True) -> SimulatedTarget:
        return cls(**{k: v for k, v in cfg.items() if k in cls.__init__.__code__.co_varnames}, seed=seed, virtual_clock=virtual_clock)

    def nominal_capacity_users(self, rps_per_user: float = 1.0) -> float:
        """Users at which in-flight reaches the concurrency limit: c / (service_s * (1 + d)) / rps."""
        return self.concurrency_limit / (self.service_ms / 1000.0 * (1 + self.degradation)) / rps_per_user

    def _refill(self) -> None:
        sigma = math.sqrt(math.log(1 + self.service_cv ** 2)) if self.service_cv > 0 else 0.0
        mu = math.log(self.service_ms) - sigma ** 2 / 2
        self._svc = self.rng.lognormal(mu, sigma, 4096) if sigma else np.full(4096, self.service_ms)
        self._u = self.rng.random(4096)
        self._i = 0

    def _now(self, request: Request) -> float:
        if self.virtual_clock:
            return request.t
        if self._t0 is None:
            self._t0 = time.perf_counter()
        return time.perf_counter() - self._t0

    def in_flight_at(self, t: float) -> int:
        while self._heap and self._heap[0] <= t:
            heapq.heappop(self._heap)
        return len(self._heap)

    def call(self, request: Request) -> Response:
        with self._lock:
            t = self._now(request)
            if self._i >= len(self._svc):
                self._refill()
            svc, u = float(self._svc[self._i]), float(self._u[self._i])
            self._i += 1
            n = self.in_flight_at(t)
            self.calls += 1
            self.peak_in_flight = max(self.peak_in_flight, n + 1)
            c, q = self.concurrency_limit, self.queue_limit
            cold = self.cold_start_ms if t < self.cold_start_window_s else 0.0
            if n >= c + q:
                latency, status = 1.0 + u, 503
            elif n < c:
                rho = n / c
                latency, status = svc * (1 + self.degradation * rho * rho) + cold, 200
            else:
                queued = n - c + 1
                latency = queued * self.service_ms / c + svc + cold
                status = 500 if u < min(0.95, self.error_slope * queued / q) else 200
            if status != 503:
                heapq.heappush(self._heap, t + latency / 1000.0)
        if not self.virtual_clock:
            time.sleep(latency / 1000.0)
        return Response(status, latency, self.response_bytes if status == 200 else 128)


class _Bucket:
    __slots__ = ("last", "tokens")

    def __init__(self, tokens: float, last: float):
        self.tokens, self.last = tokens, last


class RateLimiter:
    """Token buckets around a target. `per_user_rps` with `burst` tokens per user, plus an optional
    `global_rps` admission bucket. Refused requests get 429 in ~0.5 ms and never hit the target.

    Bucket maths: tokens = min(burst, tokens + (now - last) * rate); take one if tokens >= 1."""

    def __init__(self, target: Target, per_user_rps: float = 1.0, burst: float = 2.0,
                 global_rps: float | None = None, global_burst: float | None = None, virtual_clock: bool = True):
        self.target = target
        self.per_user_rps, self.burst = float(per_user_rps), float(burst)
        self.global_rps = float(global_rps) if global_rps else None
        # the admission bucket starts with a small burst (rate/20 ~= 50 ms of traffic): a full bucket
        # would admit 2x the rate in second 0 and push the target straight into its queue
        self.global_burst = float(global_burst) if global_burst else (self.global_rps or 0.0) / 20.0
        self.virtual_clock = virtual_clock
        self._users: dict[int, _Bucket] = {}
        self._global: _Bucket | None = None
        self._lock = threading.Lock()
        self._t0: float | None = None
        self.rejected = 0
        self.rejected_global = 0

    def _now(self, request: Request) -> float:
        if self.virtual_clock:
            return request.t
        if self._t0 is None:
            self._t0 = time.perf_counter()
        return time.perf_counter() - self._t0

    @staticmethod
    def _take(b: _Bucket, now: float, rate: float, burst: float) -> bool:
        b.tokens = min(burst, b.tokens + max(0.0, now - b.last) * rate)
        b.last = now
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True
        return False

    def call(self, request: Request) -> Response:
        with self._lock:
            now = self._now(request)
            b = self._users.get(request.user_id)
            if b is None:
                b = self._users[request.user_id] = _Bucket(self.burst, now)
            if not self._take(b, now, self.per_user_rps, self.burst):
                self.rejected += 1
                return Response(429, 0.5, 64)
            if self.global_rps:
                if self._global is None:
                    self._global = _Bucket(self.global_burst, now)
                if not self._take(self._global, now, self.global_rps, self.global_burst):
                    self.rejected += 1
                    self.rejected_global += 1
                    return Response(429, 0.5, 64)
        return self.target.call(request)
