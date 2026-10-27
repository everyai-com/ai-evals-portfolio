# p25 · Surge and scale test harness

**The mistake it answers.** *"We bolted on scalability too late."* When TELUS demoed the platform
to thousands of people who all joined at once, every weak point showed: "if you're going to blow up
from 1,000 to 10,000 users, account for it early". This harness makes the all-hands demo a test you
run before the demo. The test that encodes the lesson:
`test_spike_errors_without_rate_limiter_and_429s_with_it` — the same spike that turns >5% of
requests into 5xx on the bare target becomes 429s with p95 held down once a token-bucket limiter
sits in front, and `test_simulated_target_knee_near_configured_capacity` proves the harness finds
the knee where the capacity model puts it.

**What.**

| piece | file | what it does |
|---|---|---|
| `Target` protocol, `Request`, `Response(status, latency_ms, bytes)` | `targets.py` | one `call(request)` interface for everything |
| `FunctionTarget` | `targets.py` | any Python callable; exceptions become 500s; wall-clock latency |
| `HttpTarget` | `targets.py` | real HTTP through `urllib`; HTTP errors keep their status, connection errors are 599 |
| `SimulatedTarget` | `targets.py` | capacity model: lognormal service time, concurrency limit, bounded queue, 5xx probability rising past the knee, fast 503 when the queue is full, cold start; runs on a **virtual clock** (`request.t`) so a 285,000-request spike takes about a second |
| `RateLimiter` | `targets.py` | per-user token bucket plus an optional global admission bucket around any target; refusals are 429 and never reach the target |
| `Scenario` (`ramp`, `spike`, `soak`, `step`) | `core.py` | per-second user schedule -> seeded arrivals (`numpy.random.default_rng(seed)`); a spike can add a `join_burst` (page-load requests when a user joins) |
| `run_scenario` | `core.py` | virtual mode (sequential in arrival order) or real concurrency via `ThreadPoolExecutor` at scheduled wall-clock offsets (`time_scale` compresses time) |
| `analyse`, `find_knee`, `littles_law`, `headroom` | `core.py` | knee, max sustainable users, Little's-law check, capacity headroom statement |
| `render_markdown` (ASCII charts), `render_html` (four inline-SVG charts, no scripts, no external URLs), `result_json` | `core.py` | reports |
| demo config | `data/scenarios.yaml` | target with knee ~3,000 users, launch target 10,000, rate limiter, ramp to 5,000 / spike of 10,000 in 10 s / step 500 -> 8,000 / soak |

### The maths

* **Series per second**: `offered` = arrivals in the second; `throughput` = 2xx completed in the
  second (plateaus at capacity); p50/p95/p99 over requests that arrived in the second (429s
  excluded); `error_rate` = 5xx / offered; `in_flight` = time-averaged concurrency, the integral of
  open requests over the second.
* **Capacity model**: with `c` = concurrency limit and service time `S` (mean 50 ms), a request that
  finds `n < c` in flight takes `S(1 + 0.1(n/c)^2)`; in the queue it waits `(n-c+1) * 50/c` ms and
  fails with probability `error_slope * (n-c+1)/q`; past `c + q` it gets a 1 ms 503. By Little's
  law the knee is at `c / (0.05 s x 1.1)` ~= 2,700 req/s, i.e. about 2,700-3,000 users at 1 req/s.
* **Knee**: walk the points in order of increasing users; the first whose p95 >= 2 x baseline p95
  or whose 5xx rate > 1%. Baseline = median p95 over points with users <= 10% of the peak. The
  first 3 seconds (cold-start window + backlog drain) are never baseline or knee candidates.
* **Max sustainable concurrency**: the largest user level below the knee with p95 < 2 x baseline
  and errors <= 1%.
* **Little's law** (Little, 1961): `L = lambda * W`. Per second, `offered x (1 - 429 share) x mean
  latency` vs measured `in_flight`; the median relative error must be <= 25%. A large error means
  the harness mis-measures, or the system is far from steady state.
* **Headroom**: `ratio = launch_users / knee_users`; above 1 is "N x short", below 1 is
  "1/ratio x headroom". With no knee, the only honest claim is "at least tested/launch x headroom".

## How to run

```bash
P=projects/p25_surge_test
python -m projects.p25_surge_test.cli run --scenario spike --users 10000 --window 10 --target simulated [--json out.json] [--html report.html]
python -m projects.p25_surge_test.cli run --scenario spike --users 10000 --window 10 --target simulated --rate-limit
python -m projects.p25_surge_test.cli run --scenario ramp --users 5000 --target simulated
python -m projects.p25_surge_test.cli run --scenario step --target simulated
python -m projects.p25_surge_test.cli run --scenario ramp --users 50 --duration 10 --target http --url http://localhost:8000/health   # real HTTP, real threads
python -m projects.p25_surge_test.cli headroom --scenario spike --users 10000 --target simulated [--launch-users 10000]
python -m projects.p25_surge_test.cli run --scenario soak --target simulated --real-time --time-scale 0.1   # simulated target on the wall clock
evalkit surge headroom --scenario ramp --users 5000
.venv/bin/pytest projects/p25_surge_test -q
```

Exit codes: `run` 0, or 1 when the knee falls below the scenario's user count (the platform fell
over); `headroom` 0 when the knee is at or above the launch target, 1 when short; 2 usage error
(e.g. `--target http` without `--url`).

## Sample output

Real runs on the committed `data/scenarios.yaml`, virtual clock. The `wall s` column was measured on
a heavily loaded machine (load average ~50); on an idle laptop each run takes 0.5-1 s.

### The all-hands demo: 10,000 users join in 10 s (exit 1)

```text
$ python -m projects.p25_surge_test.cli run --scenario spike --users 10000 --window 10 --target simulated --html report.html
## surge test: spike (spike) -> simulated(c=150, service=50ms) [virtual]

| users | seconds | requests | ok | 5xx | 429 | peak rps | p95 max | err max | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 10000 | 30 | 285000 | 66450 | 218550 | 0 | 2443 | 895 ms | 95.0% | 4.90 |

| baseline p95 | knee (users) | knee reason | max sustainable users | Little's law median error | little ok |
|---|---|---|---|---|---|
| 117 ms | 4000 | error rate 66.1% > 1.0% | 0 | 0.2% | yes |

**headroom:** projected 10,000 users at launch demo; harness shows knee at 4,000 -> 2.5x short

### p95 latency per second (ms)

```text
   s=0 |########################################| 894.9 ms
   s=2 |#####                                   | 105.5 ms
   s=4 |#####                                   | 116.3 ms
   s=6 |#####                                   | 113.3 ms
   s=8 |#####                                   | 110.2 ms
  s=10 |#####                                   | 113 ms
  s=12 |#####                                   | 113.9 ms
  s=14 |#####                                   | 112.8 ms
  s=16 |#####                                   | 113.2 ms
  s=18 |#####                                   | 113.3 ms
  s=20 |#####                                   | 114.4 ms
  s=22 |#####                                   | 112.8 ms
  s=24 |#####                                   | 114.2 ms
  s=26 |#####                                   | 113.5 ms
  s=28 |#####                                   | 113.7 ms
```

### 5xx error rate per second

```text
   s=0 |#####################################   | 86.88%
   s=2 |####################################    | 85.85%
   s=4 |##############################          | 70.08%
   s=6 |################################        | 76.17%
   s=8 |##################################      | 80.06%
  s=10 |################################        | 76%
  s=12 |################################        | 75.98%
  s=14 |################################        | 75.71%
  s=16 |################################        | 75.69%
  s=18 |################################        | 76.21%
  s=20 |################################        | 76.31%
  s=22 |################################        | 75.9%
  s=24 |################################        | 75.96%
  s=26 |################################        | 76.34%
  s=28 |################################        | 76.01%
```

### Series (sampled)

| s | users | offered | rps ok | p50 | p95 | p99 | 5xx | 429 | in-flight |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 4000 | 285 | 2 | 895 | 917 | 86.9% | 0.0% | 294.1 |
| 2 | 3000 | 6000 | 865 | 2 | 106 | 125 | 85.9% | 0.0% | 307.1 |
| 4 | 5000 | 8000 | 2388 | 2 | 116 | 133 | 70.1% | 0.0% | 306.9 |
| 6 | 7000 | 10000 | 2386 | 2 | 113 | 130 | 76.2% | 0.0% | 310.1 |
| 8 | 9000 | 12000 | 2396 | 2 | 110 | 130 | 80.1% | 0.0% | 313.2 |
| 10 | 10000 | 10000 | 2397 | 2 | 113 | 130 | 76.0% | 0.0% | 310.1 |
| 12 | 10000 | 10000 | 2396 | 2 | 114 | 133 | 76.0% | 0.0% | 310.1 |
| 14 | 10000 | 10000 | 2434 | 2 | 113 | 132 | 75.7% | 0.0% | 310.0 |
| 16 | 10000 | 10000 | 2427 | 2 | 113 | 131 | 75.7% | 0.0% | 310.1 |
| 18 | 10000 | 10000 | 2380 | 2 | 113 | 133 | 76.2% | 0.0% | 310.1 |
| 20 | 10000 | 10000 | 2377 | 2 | 114 | 131 | 76.3% | 0.0% | 310.1 |
| 22 | 10000 | 10000 | 2399 | 2 | 113 | 131 | 75.9% | 0.0% | 310.1 |
| 24 | 10000 | 10000 | 2410 | 2 | 114 | 133 | 76.0% | 0.0% | 310.1 |
| 26 | 10000 | 10000 | 2351 | 2 | 113 | 133 | 76.3% | 0.0% | 310.1 |
| 28 | 10000 | 10000 | 2401 | 2 | 114 | 130 | 76.0% | 0.0% | 310.1 |
```

The `report.html` from the same run has four inline-SVG charts (offered vs throughput, p50/p95/p99,
5xx/429, in-flight) with the knee marked, and no external URLs.

### The same spike behind the rate limiter (exit 0)

Per-user bucket (1 req/s, burst 1) plus a 2,400 req/s global admission bucket, just under the
knee. 5xx drop from 218,550 to 2,233, all of them inside the 2 s cold-start window; from second 6
the target serves ~2,400 req/s at p95 ~83 ms while 76% of the excess gets a fast 429:

```text
$ python -m projects.p25_surge_test.cli run --scenario spike --users 10000 --window 10 --target simulated --rate-limit
## surge test: spike (spike) -> simulated(c=150, service=50ms) + ratelimit(user=1.0rps/burst1, global=2400) [virtual]

| users | seconds | requests | ok | 5xx | 429 | peak rps | p95 max | err max | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 10000 | 30 | 285000 | 66808 | 2233 | 215959 | 2440 | 884 ms | 22.3% | 6.70 |

| baseline p95 | knee (users) | knee reason | max sustainable users | Little's law median error | little ok |
|---|---|---|---|---|---|
| 84 ms | none | no knee within the tested load | 10000 | 2.9% | yes |

**headroom:** projected 10,000 users at launch demo; harness found no knee up to 10,000 users -> at least 1.0x headroom
...
### Series (sampled)

| s | users | offered | rps ok | p50 | p95 | p99 | 5xx | 429 | in-flight |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 4000 | 284 | 2 | 883 | 906 | 17.2% | 75.5% | 261.9 |
| 2 | 3000 | 6000 | 1549 | 84 | 120 | 141 | 6.1% | 68.6% | 218.2 |
| 4 | 5000 | 8000 | 2368 | 53 | 85 | 100 | 0.3% | 69.9% | 134.9 |
| 6 | 7000 | 10000 | 2417 | 51 | 83 | 99 | 0.0% | 76.0% | 132.7 |
| 8 | 9000 | 12000 | 2397 | 52 | 85 | 105 | 0.0% | 80.0% | 135.0 |
| 10 | 10000 | 10000 | 2397 | 52 | 82 | 99 | 0.0% | 76.0% | 132.7 |
| 12 | 10000 | 10000 | 2399 | 51 | 82 | 101 | 0.0% | 76.0% | 131.7 |
| 14 | 10000 | 10000 | 2401 | 52 | 83 | 102 | 0.0% | 76.0% | 132.8 |
| 16 | 10000 | 10000 | 2400 | 51 | 83 | 99 | 0.0% | 76.0% | 132.0 |
| 18 | 10000 | 10000 | 2406 | 51 | 82 | 99 | 0.0% | 76.0% | 131.3 |
| 20 | 10000 | 10000 | 2398 | 51 | 82 | 100 | 0.0% | 76.0% | 132.0 |
| 22 | 10000 | 10000 | 2390 | 52 | 84 | 104 | 0.0% | 76.0% | 134.2 |
| 24 | 10000 | 10000 | 2399 | 51 | 84 | 103 | 0.0% | 76.0% | 132.2 |
| 26 | 10000 | 10000 | 2405 | 53 | 86 | 105 | 0.0% | 76.0% | 134.9 |
| 28 | 10000 | 10000 | 2401 | 51 | 82 | 98 | 0.0% | 76.0% | 131.6 |
```

### Headroom against the launch target (exit 1)

```text
$ python -m projects.p25_surge_test.cli headroom --scenario spike --users 10000 --target simulated
## headroom

- scenario: spike, 10,000 users over 30 s -> simulated(c=150, service=50ms)
- knee: 4000 (error rate 66.1% > 1.0%); max sustainable: 0 users
- **projected 10,000 users at launch demo; harness shows knee at 4,000 -> 2.5x short**
```

### A ramp to 5,000 users: the precise knee (exit 1)

```text
$ python -m projects.p25_surge_test.cli run --scenario ramp --users 5000 --target simulated
## surge test: ramp (ramp) -> simulated(c=150, service=50ms) [virtual]

| users | seconds | requests | ok | 5xx | 429 | peak rps | p95 max | err max | wall s |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | 60 | 152500 | 113438 | 39062 | 0 | 2842 | 881 ms | 51.7% | 1.05 |

| baseline p95 | knee (users) | knee reason | max sustainable users | Little's law median error | little ok |
|---|---|---|---|---|---|
| 76 ms | 2917 | error rate 1.9% > 1.0% | 2833 | 0.2% | yes |

**headroom:** projected 10,000 users at launch demo; harness shows knee at 2,917 -> 3.4x short

### p95 latency per second (ms)

```text
   s=0 |########################################| 880.8 ms
   s=3 |###                                     | 73.9 ms
   s=6 |####                                    | 79.01 ms
   s=9 |####                                    | 80.28 ms
  s=12 |####                                    | 78.43 ms
  s=15 |####                                    | 79.86 ms
  s=18 |####                                    | 78.45 ms
  s=21 |####                                    | 79.09 ms
  s=24 |####                                    | 81.39 ms
  s=27 |####                                    | 83.27 ms
  s=30 |####                                    | 83.68 ms
  s=33 |####                                    | 80.99 ms
  s=36 |#####                                   | 109.8 ms
  s=39 |######                                  | 123 ms
  s=42 |######                                  | 126.2 ms
  s=45 |######                                  | 123.9 ms
  s=48 |######                                  | 123.5 ms
  s=51 |######                                  | 122.9 ms
  s=54 |#####                                   | 121 ms
  s=57 |######                                  | 122.6 ms
```

### 5xx error rate per second

```text
   s=0 |                                        | 0%
   s=3 |                                        | 0%
   s=6 |                                        | 0%
   s=9 |                                        | 0%
  s=12 |                                        | 0%
  s=15 |                                        | 0%
  s=18 |                                        | 0%
  s=21 |                                        | 0%
  s=24 |                                        | 0%
  s=27 |                                        | 0%
  s=30 |                                        | 0.1936%
  s=33 |#                                       | 0.8472%
  s=36 |########                                | 10.67%
  s=39 |#####################                   | 27.18%
  s=42 |##########################              | 33.24%
  s=45 |############################            | 36.39%
  s=48 |################################        | 41.32%
  s=51 |##################################      | 44.01%
  s=54 |#####################################   | 47.7%
  s=57 |####################################### | 50.26%
```

### Series (sampled)

| s | users | offered | rps ok | p50 | p95 | p99 | 5xx | 429 | in-flight |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 83 | 83 | 13 | 850 | 881 | 886 | 0.0% | 0.0% | 39.1 |
| 4 | 417 | 417 | 399 | 47 | 76 | 92 | 0.0% | 0.0% | 19.9 |
| 8 | 750 | 750 | 740 | 49 | 78 | 100 | 0.0% | 0.0% | 37.7 |
| 12 | 1083 | 1083 | 1071 | 48 | 78 | 92 | 0.0% | 0.0% | 54.6 |
| 16 | 1417 | 1417 | 1407 | 49 | 78 | 92 | 0.0% | 0.0% | 71.9 |
| 20 | 1750 | 1750 | 1746 | 50 | 81 | 101 | 0.0% | 0.0% | 91.4 |
| 24 | 2083 | 2083 | 2076 | 50 | 81 | 98 | 0.0% | 0.0% | 109.5 |
| 28 | 2417 | 2417 | 2412 | 51 | 83 | 102 | 0.0% | 0.0% | 128.7 |
| 32 | 2750 | 2750 | 2718 | 52 | 84 | 99 | 0.9% | 0.0% | 150.0 |
| 36 | 3083 | 3083 | 2693 | 76 | 110 | 128 | 10.7% | 0.0% | 229.0 |
| 40 | 3417 | 3417 | 2438 | 93 | 124 | 140 | 28.7% | 0.0% | 293.6 |
| 44 | 3750 | 3750 | 2421 | 93 | 124 | 141 | 35.7% | 0.0% | 297.7 |
| 48 | 4083 | 4083 | 2400 | 91 | 123 | 141 | 41.3% | 0.0% | 298.8 |
| 52 | 4417 | 4417 | 2397 | 89 | 123 | 142 | 45.7% | 0.0% | 300.2 |
| 56 | 4750 | 4750 | 2361 | 87 | 123 | 141 | 50.4% | 0.0% | 300.9 |
```

**Reading the numbers.** The ramp is the measuring instrument: it adds ~83 users per second and puts
the knee at **2,917 users** (max sustainable 2,833), which matches the configured capacity of about
2,700-3,000. Against the 10,000-user launch that is **3.4x short**. The spike is the realistic
event: users arrive 1,000 per second (plus a 3-request join burst each), so its knee can only be
read at 1,000-user resolution, and the 3-second warm-up skip pushes its first candidate to 4,000.
That is why `headroom` on the spike says **2.5x short**. Both say the same thing: the demo will fall
over. Run the ramp to locate the knee, and the spike to see what the failure looks like.

## Tests (12, ~1-2 s on an idle machine)

| claim | test |
|---|---|
| ramp finds the knee within +/-30% of the configured capacity; cold-start seconds are not the knee; throughput plateaus | `test_simulated_target_knee_near_configured_capacity` |
| spike: >5% 5xx bare; with the limiter 429s replace them, 5xx <= 1%, p95 bounded, ~global_rps still served | `test_spike_errors_without_rate_limiter_and_429s_with_it` |
| token-bucket arithmetic, per-user isolation, global bucket | `test_rate_limiter_bucket_maths` |
| Little's law within tolerance at 50% utilisation; checked by hand on one second | `test_littles_law_check_within_tolerance` |
| "3.1x short" for 10,000 vs 3,200; headroom, no-knee and untested cases | `test_headroom_statement_math` |
| step levels 500 -> 8,000 double; offered load is monotonic; ramp and spike user curves | `test_step_scenario_monotonic_load_and_levels` |
| same seed -> identical series; different seed -> different | `test_seeded_determinism` |
| HTML has 4 `<svg>`, no `http(s)://`, no `<script>` or `<link>` | `test_html_has_svg_and_no_external_urls_and_markdown_charts` |
| ThreadPool path on a tiny `FunctionTarget` (statuses, bytes, per-second buckets, exceptions -> 500) | `test_threadpool_path_on_tiny_function_target` |
| simulated target on the wall clock through threads | `test_simulated_target_on_wall_clock_through_threads` |
| knee edge case with no traffic | `test_find_knee_edge_cases` |
| CLI: spike exit 1 with JSON + HTML, rate-limited spike exit 0, headroom short and ok, usage errors | `test_cli_run_and_headroom` |

## Design notes

* **Why a virtual clock.** Real load tests are slow, flaky and need a real system. The capacity
  model runs in arrival order on simulated time, so tests take milliseconds and are deterministic,
  and the same harness drives a real URL (`HttpTarget`) or callable (`FunctionTarget`) through a
  real `ThreadPoolExecutor` when you have one.
* **Why the spike goes wrong so fast.** Arrivals past `c + q` in flight get an instant 503 that never
  occupies a slot, so the target keeps serving its ~2,400 req/s while everyone else fails. That is
  the "every weak point showed" moment: the service is not down, it is drowning.
* **Why the limiter has two buckets.** A per-user bucket alone does not save a 10,000-user spike:
  10,000 well-behaved users at 1 req/s are still 3x the knee. The per-user bucket removes the join
  burst; the global admission bucket, sized just under the knee and starting with only 1/20 s of
  burst, turns the rest into cheap 429s. Starting it full would admit two seconds' worth at once and
  push the target into its queue.
* **What the harness does not model.** Retries with backoff (a retry storm makes the 429 picture
  worse), autoscaling (add capacity mid-run by swapping the target), or downstream dependencies.
  Wrap a `FunctionTarget` around any of these to test them.
* **evalkit gap.** None needed beyond `evalkit.report.md_table` / `pct`. Percentile and Little's-law
  helpers are local; a shared `evalkit.stats.percentiles(values, qs)` would serve p10/p20 too.
