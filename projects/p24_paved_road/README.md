# p24 · Paved road for builders

**The mistake it answers.** *"Draw the rest of the owl."* "Upload a zip, get a subdomain" sounded
easy to the platform team. First-time builders got stuck on the zip, and their AI tools (a
prototyping studio, a chat assistant) produced apps with their own Express server that the platform
could not deploy. TELUS's fix: publish an `llms.txt` and a skill so any agent knows how to build
for the platform, add a hosted studio, and **watch someone try it before calling it easy**. The
tests that encode it: `test_validator_flags_own_server_with_file_line_and_builder_fix` (the
studio-exported Express app is rejected with the exact line and a fix an agent can apply) and
`test_first_run_summary_finds_the_zip_blocker_and_is_not_ready` (8 first runs, 5 stuck on the zip,
`ready_to_call_easy: false`).

**What.** One machine-readable contract (`data/platform.yaml`) rendered three ways, a precheck that
speaks builder language, and a usability log with a verdict:

| piece | what it does |
|---|---|
| `PlatformSpec` | loads `platform.yaml`: what deploys (static + serverless handlers, no own listening server), runtimes/versions, size caps, manifest fields, forbidden deps with *why*, entrypoint shape, secrets rule, egress allowlist, deploy command and URL pattern |
| `render_llms_txt(spec)` | concise agent-readable page: what works, what fails and why, minimal manifest + handler, limits, deploy steps, checklist, studio link |
| `render_skill(spec)` | `{"SKILL.md", "reference.md"}` — frontmatter `name`/`description`, a 7-step procedure, the checklist; `reference.md` holds every rule as a table |
| `render_agents_md_snippet(spec)` | a fenced `<!-- gizmos:start/end -->` block for a project's AGENTS.md |
| `CompatValidator.validate(dir)` | deploy-time precheck -> `list[Finding(code, severity, file, line, message, fix)]`: own server (`app.listen(`, `http.createServer(`, `uvicorn.run(`, `app.run(` ... with the port), forbidden deps from `package.json` / `requirements.txt` / `pyproject.toml` (tomllib), manifest missing / fields / runtime, wrong entrypoint export, oversized files and bundles, committed secrets (`.env`, `*.pem`, `id_rsa`), outbound URLs off the allowlist |
| `FirstRunStudy` | `data/first_runs.jsonl` rows (`participant, role, tool, step_reached, finished, unaided, minutes, blocker`) -> `summarise_first_runs()`: stuck count per step, median minutes to first deploy, blocker themes by keyword clustering, `ready_to_call_easy` only when >= 3 non-engineers finished unaided |

All text is deterministic — regenerating yields byte-identical files (tested) so the outputs can
be committed and diffed when the spec changes.

## How to run

```bash
P=projects/p24_paved_road
python -m projects.p24_paved_road.cli generate --spec $P/data/platform.yaml --out out/
python -m projects.p24_paved_road.cli validate --dir $P/data/sample_app_express --spec $P/data/platform.yaml [--json out.json]
python -m projects.p24_paved_road.cli validate --dir $P/data/sample_app_ok
python -m projects.p24_paved_road.cli first-runs --log $P/data/first_runs.jsonl [--min-unaided 3]
evalkit pavedroad validate --dir $P/data/sample_app_ok
.venv/bin/pytest projects/p24_paved_road -q
```

Exit codes: `0` deployable / ready to call easy / generated, `1` blocking findings or not ready,
`2` usage error.

## Sample output

The app a prototyping studio exports (own Express server, `express` dep, no `owner`, a committed
`.env`, a weather API off the allowlist). Exit 1:

```text
$ python -m projects.p24_paved_road.cli validate --dir projects/p24_paved_road/data/sample_app_express --spec projects/p24_paved_road/data/platform.yaml
## gizmo validate ./my-app — `projects/p24_paved_road/data/sample_app_express`

**not deployable**: 6 blocking, 0 warnings

| severity | code | where | what the platform saw |
|---|---|---|---|
| blocking | entrypoint | server.js | `server.js` does not export `handler`; the platform calls `handler(request)` and nothing else |
| blocking | forbidden_dependency | package.json:7 | `express` is listed in package.json; the platform already routes requests; export a handler instead of app.listen() |
| blocking | manifest_field | gizmo.yaml | `gizmo.yaml` is missing required field `owner` |
| blocking | outbound_domain | server.js:11 | `api.randomweather.io` (server.js:11) is not on the egress allowlist; egress is allowlisted; a request to any other domain fails at runtime with 403 |
| blocking | own_server | server.js:16 | this app starts its own server on port 3000 (`app.listen(` at server.js:16); the platform already routes requests and nothing may listen on a port |
| blocking | secret_file | .env | `.env` looks like a secrets file committed into the bundle; deployed bundles are readable by the platform team |

### Fixes

1. **entrypoint** (server.js): add `export async function handler(request) -> Response` to `server.js` and export it
2. **forbidden_dependency** (package.json:7): remove `express` from package.json and move its routes into the exported `handler` (see llms.txt 'Entrypoint')
3. **manifest_field** (gizmo.yaml): add `owner: <owner>` to `gizmo.yaml` (e.g. `owner: jane.doe@example.com`)
4. **outbound_domain** (server.js:11): either call an allowlisted domain (api.anthropic.com, api.internal.example.com, api.openai.com, gizmos.example.com) or request `api.randomweather.io` be added before deploying
5. **own_server** (server.js:16): delete the `app.listen(...)` call at server.js:16 and export a handler instead: node: `export async function handler(request) -> Response`; python: `def handler(request) -> Response`
6. **secret_file** (.env): delete `.env` from the bundle; declare env names under manifest.env; set values with `gizmo secrets set NAME`; never commit .env files
```

The same app shaped for the platform (`handler.js` exporting `handler`, no deps, full manifest).
Exit 0:

```text
$ python -m projects.p24_paved_road.cli validate --dir projects/p24_paved_road/data/sample_app_ok
## gizmo validate ./my-app — `projects/p24_paved_road/data/sample_app_ok`

**deployable**: 0 blocking, 0 warnings

No findings. Next: `gizmo deploy ./my-app` -> `https://{name}.gizmos.example.com`
```

Eight first runs, five stuck on "where do I get a zip". Exit 1:

```text
$ python -m projects.p24_paved_road.cli first-runs --log projects/p24_paved_road/data/first_runs.jsonl
## first-run study

| participants | finished | non-engineers unaided | median min to deploy | ready_to_call_easy |
|---|---|---|---|---|
| 8 | 2 | 1 | 20 | false |

### Where people stop

| step | stuck |
|---|---|
| make_zip | 5 |
| fix_precheck | 1 |

### Blocker themes

| theme | n | participants |
|---|---|---|
| zip | 5 | p01, p02, p04, p06, p08 |
| own server | 1 | p05 |

**verdict:** only 1 non-engineer(s) finished unaided (need 3); 5 of 8 stopped at `make_zip`: not easy yet — fix the top blocker and watch three more people try it
```

Generation (`out/` is git-ignored; the files are deterministic):

```text
$ python -m projects.p24_paved_road.cli generate --spec projects/p24_paved_road/data/platform.yaml --out out/
## generate (Gizmos)

- out/llms.txt (108 lines)
- out/AGENTS.md.snippet (12 lines)
- out/SKILL.md (56 lines)
- out/reference.md (70 lines)
```

The first lines of the generated `llms.txt`:

```text
# Gizmos

> upload a bundle, get a subdomain. Gizmos deploys static files plus serverless handlers to `https://{name}.gizmos.example.com`.
> This file is for coding agents: read it before generating an app for Gizmos.

## What works

- static assets (HTML/CSS/JS/images) served from `public/`
- serverless request handlers exported from the entrypoint; the platform owns the port and the router
- environment variables declared in the manifest and set with the secrets command

## What fails, and why

- apps that start their own HTTP server (express, fastify, koa, flask, fastapi, uvicorn, gunicorn): nothing may listen on a port
- long-running processes, background workers, websocket servers, cron daemons
- Docker images or anything that needs a system package

Forbidden dependencies (the deploy precheck rejects them):
- `express`: the platform already routes requests; export a handler instead of app.listen()
- `fastapi`: the platform already routes requests; export def handler(request) instead of running an ASGI server
- `fastify`: the platform already routes requests; export a handler instead of fastify.listen()
- `flask`: the platform already routes requests; export def handler(request) instead of app.run()
- `gunicorn`: nothing may listen on a port; the platform calls your handler directly
- `koa`: the platform already routes requests; export a handler instead of app.listen()
- `socket.io`: websocket servers are not supported; use request/response handlers
- `uvicorn`: nothing may listen on a port; the platform calls your handler directly

## Runtimes
...
```

## Tests (15, < 1 s)

| claim | test |
|---|---|
| llms.txt names every forbidden dep with its reason, the deploy + precheck commands, manifest fields, runtimes, a handler example | `test_llms_txt_mentions_every_forbidden_dep_and_the_deploy_command` |
| SKILL.md frontmatter, reference.md rules table, AGENTS.md fenced snippet | `test_skill_files_and_agents_snippet` |
| regenerating twice (in memory and on disk via the CLI) is byte-identical | `test_regeneration_is_idempotent` |
| own server flagged at `server.js:16` with "port 3000" and a fix naming the handler signature | `test_validator_flags_own_server_with_file_line_and_builder_fix` |
| `express` found at `package.json:7`; `cors` is fine | `test_validator_flags_forbidden_dep_from_package_json` |
| the Express sample yields exactly the six finding codes, all blocking, CLI exit 1, `--json` | `test_express_sample_has_the_full_set_of_findings_and_exit_1` |
| the ok app has no findings, exit 0 | `test_ok_app_passes_and_exit_0` |
| missing `owner` (with the example value in the fix); missing manifest; not-a-directory | `test_manifest_field_missing_and_missing_manifest` |
| `api.randomweather.io` flagged at `server.js:11`; `www.w3.org`, allowlisted and sub-domains are not | `test_outbound_domain_outside_allowlist_and_ignored_domains` |
| Python app: `python3.9` rejected, `Flask` from requirements.txt and `uvicorn` from pyproject, `app.run(... port=5000)`, missing `def handler`, `id_rsa` | `test_python_app_own_server_runtime_and_entrypoint_checks` |
| per-file, bundle-size and file-count caps | `test_file_too_large` |
| zip is the top blocker (5/8), median 20 min, 1 non-engineer unaided -> not ready | `test_first_run_summary_finds_the_zip_blocker_and_is_not_ready` |
| 3 non-engineers unaided -> ready; engineers and aided finishes never count | `test_first_run_ready_when_three_non_engineers_finish_unaided` |
| blocker clustering falls back to the most frequent word when no theme keyword matches | `test_blocker_clustering_falls_back_to_frequent_word` |
| CLI exit codes for first-runs, missing log, bad subcommand | `test_cli_first_runs_exit_codes_and_json` |

## Design notes

* **Findings are written for the person who did not know what a port is.** Participant p05 in the
  log stopped at `fix_precheck` because "the app starts its own express server" meant nothing to
  them. Every finding therefore carries the platform's reason ("the platform already routes
  requests") and a `fix` concrete enough for their agent to apply (`delete the app.listen(...) call
  at server.js:16 and export a handler instead: node: ...`).
* **One own-server finding per file.** The point is to explain the shape problem, not to list
  every `listen` call; the fix text already names the replacement.
* **Egress findings are blocking, not warnings.** An app that cannot reach its API is broken at
  runtime; it is kinder to fail at precheck. Schema/namespace domains (`www.w3.org`, `schema.org`)
  and localhost are ignored via `outbound.ignored_domains`.
* **`ready_to_call_easy` counts only unaided non-engineers.** Engineers are the people who wrote
  the platform's mental model; an aided finish measures the helper. Three is the minimum the TELUS
  team used before trusting their own "it's easy".
* **Why the zip fixture matters.** Five of eight participants stopped before the platform ever saw
  their app. No precheck can help there — only a hosted studio or a tool that uploads for them,
  which is why `llms.txt` ends with the studio link.
* **The committed `.env` fixture** holds placeholder values only; the monorepo `.gitignore`
  excludes `.env`, so `data/sample_app_express/.gitignore` re-includes it with `!.env`.
* **evalkit gap.** Nothing new needed; `evalkit.io.read_jsonl` / `write_jsonl` and
  `evalkit.report.md_table` cover the log and the tables.
