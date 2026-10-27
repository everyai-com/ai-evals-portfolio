# p17 Independent status page

**Mistake it answers:** "it's fine" — for five weeks the Gizmos platform had no observability, users
reported outages on Slack, the status page arrived *after* the first outage, and then it
false-alarmed because the checks ran from far-away regions. **Lesson encoded:** ship the status
page on day one, on its own infrastructure (own worker, cron every 5 minutes), probing from the
regions the users are in, with a quorum rule so one distant region cannot page anyone.
**Test that proves it:** `test_single_far_region_never_opens_an_incident` (ap-southeast fails on
20 consecutive ticks, zero incidents) next to `test_home_region_failure_opens_incident_after_debounce`
and `test_two_of_three_regions_opens_incident`.

## What
* `Probe` definitions from `data/probes.yaml`: name, kind (`http` / `tcp` / `function`), target,
  timeout, expected status, body regex, regions, `home_region`, `quorum`, `debounce_ticks`.
* A `Prober` protocol with `UrllibProber` (real HTTP via `urllib`, TCP via `socket`, `function`
  via `module:callable`; the opener is injectable) and `FakeProber` (scripted per-region outages
  and latencies from `data/scenario.yaml`).
* `StatusStore`: JSONL state (`results.jsonl` append-only, `incidents.jsonl`, `meta.json` with the
  last tick and the per-probe failure streaks, so debounce survives a restart).
* `Monitor.tick(now)` runs every probe in every region, applies the rules below, opens / closes
  incidents. `Scheduler.run(interval_s, ticks)` with injected `clock` and `sleep` (`SimClock`
  advances time instead of waiting; tests never sleep).
* `compute_state` → uptime per probe per 24h / 7d / 30d, p50 / p95 latency per region (24h),
  30 daily uptime buckets, open incidents, history, and the **monitor-the-monitor** staleness flag.
* `render_html(state)`: one self-contained page (inline CSS variables, light/dark via
  `prefers-color-scheme`, inline SVG uptime bars, no external URLs, no scripts). `render_json`.

| rule | definition |
|---|---|
| regional quorum | DOWN on a tick iff the probe fails in `home_region` **or** in ≥ `quorum` regions; fewer remote-only failures are `degraded-remote` (shown, never an incident) |
| debounce | an incident opens after `debounce_ticks` consecutive DOWN ticks; closes on the first non-DOWN tick |
| uptime | `# ticks not DOWN / # ticks` in the window `(now - W, now]`, using the raw quorum verdict per tick (a single bad tick costs its share even if it never became an incident) |
| latency | `numpy.percentile` p50 / p95 over successful results per (probe, region) in the last 24h |
| staleness | `now - last_tick > 2 × interval_s` → banner "monitor stale", exit code 1 on `render` |

## Why
A status page that lives inside the platform dies with it. One that probes from Sydney while the
users are in Toronto pages on-call for an undersea cable. The two rules above are the whole fix:
independence (own state, own scheduler, own renderer) and a quorum anchored on the home region.

## How to run
```bash
# 12 simulated 5-minute ticks on the committed scenario (ap-southeast flaps alone, then
# us-east + eu-west fail together on agent-api, then recover). No real sleeping, no network.
python -m projects.p17_status_page.cli run --fake --ticks 12 --interval 300 --state state/ --reset
python -m projects.p17_status_page.cli render --state state/ --out status.html --json status.json
python -m projects.p17_status_page.cli tick --state state/ --fake-scenario projects/p17_status_page/data/scenario.yaml
python -m projects.p17_status_page.cli check --url https://example.internal/healthz --regions us-east eu-west   # real probe, one shot
# cron-style (real clock, real probes): python -m projects.p17_status_page.cli run --interval 300
.venv/bin/pytest projects/p17_status_page -q
```
Exit codes: `run` / `tick` return 1 while any probe is DOWN or an incident is open; `render` returns 1
when the monitor is stale (the page itself cannot be trusted) or an incident is open.

## Sample output
Real run on `data/probes.yaml` (4 probes × 3 regions, home us-east, quorum 2, debounce 2) and
`data/scenario.yaml`.

```
$ python -m projects.p17_status_page.cli run --fake --ticks 12 --interval 300 --state state/ --reset
2026-10-24 00:00:00 UTC  gateway=up  agent-api=up  model-proxy=up  tool-sandbox=up
2026-10-24 00:05:00 UTC  gateway=up  agent-api=up  model-proxy=up  tool-sandbox=up
2026-10-24 00:10:00 UTC  gateway=degraded-remote  agent-api=degraded-remote  model-proxy=degraded-remote  tool-sandbox=up
2026-10-24 00:15:00 UTC  gateway=up  agent-api=up  model-proxy=up  tool-sandbox=up
2026-10-24 00:20:00 UTC  gateway=degraded-remote  agent-api=degraded-remote  model-proxy=degraded-remote  tool-sandbox=up
2026-10-24 00:25:00 UTC  gateway=up  agent-api=up  model-proxy=up  tool-sandbox=up
2026-10-24 00:30:00 UTC  gateway=degraded-remote  agent-api=up  model-proxy=up  tool-sandbox=up
2026-10-24 00:35:00 UTC  gateway=up  agent-api=up  model-proxy=up  tool-sandbox=up
2026-10-24 00:40:00 UTC  gateway=up  agent-api=down  model-proxy=up  tool-sandbox=up
2026-10-24 00:45:00 UTC  gateway=up  agent-api=down  model-proxy=up  tool-sandbox=up  OPENED inc-1792802700-agent-api (eu-west, us-east: us-east: HTTP 503; eu-west: HTTP 503)
2026-10-24 00:50:00 UTC  gateway=up  agent-api=down  model-proxy=up  tool-sandbox=up
2026-10-24 00:55:00 UTC  gateway=up  agent-api=up  model-proxy=up  tool-sandbox=up  CLOSED inc-1792802700-agent-api

12 ticks; overall: all systems operational; open incidents: 0; history: 1
| probe | status | 24h | 7d | 30d |
|---|---|---|---|---|
| gateway | up | 100.00% | 100.00% | 100.00% |
| agent-api | up | 75.00% | 75.00% | 75.00% |
| model-proxy | up | 100.00% | 100.00% | 100.00% |
| tool-sandbox | up | 100.00% | 100.00% | 100.00% |
(exit 0)
$ python -m projects.p17_status_page.cli render --state state/ --out status.html --now 2026-10-24T00:57:00Z
wrote status.html (19227 bytes); overall: all systems operational; stale: False; last tick: 2026-10-24 00:55:00 UTC
(exit 0)
$ python -m projects.p17_status_page.cli render --state state/ --out status.html --now 2026-10-24T02:00:00Z
wrote status.html (19401 bytes); overall: monitor stale; stale: True; last tick: 2026-10-24 00:55:00 UTC
(exit 1)
$ python -m projects.p17_status_page.cli check --url https://127.0.0.1:9/healthz --regions local --timeout 1
| region | ok | status | latency ms | error |
|---|---|---|---|---|
| local | False | - | 37 | URLError: <urlopen error [Errno 61] Connection refused> |
verdict: down
(exit 1)
```
Reading the run: ap-southeast failing alone on ticks 2, 4 and 6 (three probes, then one) is
`degraded-remote` every time and never becomes an incident; the agent-api failure in us-east +
eu-west at 00:40 is DOWN immediately but the incident opens one tick later (debounce 2) and
closes at 00:55 on the first healthy tick. agent-api's uptime is 9/12 = 75%: the three DOWN ticks
count, the debounce does not hide them.

## Design notes
* **Own infrastructure.** `core.py` has no dependency on anything it monitors; the state is three
  files and the page is static HTML, so it can be served from a bucket while the platform is down.
* **Quorum anchored on home.** `quorum` counts regions, but `home_region` short-circuits: if the
  users' region fails, it is an outage regardless of what Frankfurt and Singapore see. Conversely
  a remote region can only contribute to an outage together with another region.
* **Debounce and uptime are separate.** Debounce protects on-call from a one-tick blip; uptime is
  what the users experienced, so it uses the per-tick quorum verdict without debounce.
* **Monitor the monitor.** `meta.json` records `last_tick`; `compute_state` compares it to
  `2 × interval_s`. A stale monitor renders a grey banner that says so, instead of a green page
  nobody is updating — the failure mode of "it's fine".
* **FakeProber tick index** is the per-(probe, region) call count, which equals the monitor tick
  because every tick probes every pair exactly once. That keeps the scenario file independent of
  wall-clock time.
* **Real probes** go through `UrllibProber`; one process cannot be in three regions, so in
  production one worker per region writes to the shared store with its region label and the
  quorum rule works unchanged.
