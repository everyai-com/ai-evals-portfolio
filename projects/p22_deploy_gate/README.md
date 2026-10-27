# p22 — Deploy gate: secret scanning and automated review

**The mistake (from the Q&A on vibe-coded apps):** letting an app built in an afternoon by someone
who has never heard of a secret store go straight to production. The platform's answer: every push
is secret-scanned and run through a model asking "is anything here obviously bad?"; findings block
the deploy and go back to the builder's agent as tasks, which fixes and redeploys — a loop. Plus:
every app on its own domain, one locked-down request path, no servers of your own.

The tests that prove the lesson is encoded: `test_each_planted_secret_detected_with_right_rule_and_masked`
(every planted secret is caught and the raw value appears in **no** output: markdown, JSON,
findings.md or the model digest), `test_placeholder_allowlisted_and_fixture_values_not_flagged`,
`test_sample_blocked_clean_passes_and_exit_codes`, `test_findings_md_has_one_task_per_finding`.

## What

| piece | does |
|---|---|
| `SecretScanner` | regex rules from `data/secret_rules.yaml` (AWS keys, GitHub / Slack / Stripe tokens, private-key blocks, JWTs, URLs with credentials, generic `api_key=` assignments) + a Shannon-entropy detector; skips test-fixture paths, placeholder values (`sk_test_`, `xxxx`, `EXAMPLE` ...), `.deploygateignore` path globs and `allow-value:` entries, and lines marked `# deploygate:allow`; every finding is `file:line`, rule, severity and a **masked** match |
| `PolicyChecks` | forbidden files (`.env` / `.env.*`, `*.pem`, `node_modules/**`, blobs > 25 MB), required files, forbidden patterns (`eval(`, `child_process.exec` + concat, `subprocess(..., shell=True)` with an f-string, internal hostnames, binding `0.0.0.0` when the platform forbids own servers), manifest validation against `data/manifest_schema.yaml` |
| `ModelReview(client)` | sends a bounded digest (file tree + the suspicious lines, with secret-bearing lines withheld) to an `LLMClient`, asks for JSON `[{severity, file, why, fix}]`, parses strictly and drops malformed rows |
| `Gate.run(dir) -> Verdict` | `blocked` when any secret is critical/high or any forbidden/required/manifest policy fires; everything else warns |
| `findings_md` | the fix-loop packet: one task per finding with file, line, what and what to change |

## Why

Secret scanning that prints the secret has just leaked it a second time; scanning that flags every
`sk_test_xxx` gets switched off within a week. The useful gate is strict on real credentials,
quiet on placeholders and fixtures, explains each finding as a task, and never blocks on the
model's opinion alone — the model review warns, the deterministic checks block.

## How to run

```bash
D=projects/p22_deploy_gate/data
.venv/bin/python -m projects.p22_deploy_gate.cli scan [--json [out.json]] [--findings findings.md] [--no-review]   # builds + scans the planted demo; exit 1
.venv/bin/python -m projects.p22_deploy_gate.cli demo --out sample_artifact                                         # or write it out to inspect
.venv/bin/python -m projects.p22_deploy_gate.cli scan --dir sample_artifact                                         # exit 1
.venv/bin/python -m projects.p22_deploy_gate.cli scan --dir $D/clean_artifact                                                             # exit 0
.venv/bin/python -m projects.p22_deploy_gate.cli rules
.venv/bin/pytest projects/p22_deploy_gate -q
```

**Why the planted artifact is generated, not committed.** It contains fake Slack, Stripe, GitHub and AWS tokens in their real
formats, and GitHub push protection correctly refuses any commit holding them, fake or not. So the committed
`data/sample_artifact_template/` holds `{{FAKE:name}}` placeholders, and `fixtures.py` keeps each value split across two
string literals and fills the placeholders at runtime. The rebuilt artifact is byte-identical to the original fixture. This
is the same lesson the gate teaches: a scanner cannot tell a convincing fake from a leak, so keep both out of the repo.

`EVALKIT_LLM=anthropic` swaps the deterministic mock reviewer for a real model with no code change
(`core.default_review_client`). `evalkit deploygate ...` is the alias.

Library use:

```python
from projects.p22_deploy_gate import core
from projects.p22_deploy_gate.fixtures import FAKE   # fabricated tokens, assembled at runtime
verdict = core.Gate().run("path/to/artifact")
verdict.blocked, verdict.summary, [f.to_dict() for f in verdict.findings]
core.SecretScanner().scan_line(f'key = "{FAKE['aws_1']}"', "a.py", 1)[0].masked   # 'AKIA************ (20 chars)'
core.ModelReview(my_client).review(artifact_dir, prior_findings)                     # ReviewResult(findings, dropped, raw_rows, model)
```

## Sample output

`demo --out sample_artifact` then `scan --dir sample_artifact --findings findings.md` (exit 1):

```
# Deploy gate: BLOCKED for `sample_artifact`

11 files scanned; 21 findings (9 blocking, 12 warnings); secrets 8, policy 9, review 4 (model mock, 2 malformed rows dropped).

| severity | gate | kind | rule | where | finding |
|---|---|---|---|---|---|
| critical | block | secret | aws_access_key | app.py:9 | AWS access key id in source [AKIA************ (20 chars)] |
| critical | block | policy | forbidden_file | certs/server.pem | certificate / key material in the artifact |
| critical | block | secret | private_key_block | certs/server.pem:1 | Private key block in source [----************ (27 chars)] |
| high | block | policy | forbidden_file | .env.local | environment file shipped in the artifact |
| high | block | secret | slack_token | .env.local:1 | Slack token in source [xoxb************ (56 chars)] |
| high | block | secret | url_credentials | .env.local:2 | URL with embedded credentials in source [post************ (51 chars)] |
| high | block | secret | github_token | config.js:2 | GitHub token in source [ghp_************ (43 chars)] |
| high | block | secret | url_credentials | config.js:3 | URL with embedded credentials in source [post************ (51 chars)] |
| high | block | policy | forbidden_file | node_modules/left-pad/index.js | vendored node_modules: install at build time |
| high | warn | review | model_review | app.py:19 | shell=True with interpolated input lets a path like '; rm -rf /' run |
| high | warn | policy | subprocess_shell_fstring | app.py:19 | subprocess with shell=True and an f-string command (shell injection) |
| high | warn | policy | eval_call | config.js:8 | eval() on dynamic input |
| high | warn | review | model_review | config.js:8 | eval over a template string is remote code execution if the template is user-supplied |
| high | warn | policy | child_process_exec_concat | config.js:12 | child_process.exec with string concatenation (shell injection) |
| high | warn | review | model_review | config.js:12 | exec builds a shell command from its argument |
| medium | warn | policy | internal_hostname | .env.local:2 | hard-coded internal hostname |
| medium | warn | secret | high_entropy_string | app.py:14 | high-entropy string (5.09 bits/char) [Qm7v************ (34 chars)] |
| medium | warn | policy | bind_all_interfaces | app.py:28 | app binds its own server on 0.0.0.0; the platform forbids own servers |
| medium | warn | review | model_review | app.py:28 | the app starts its own server; on the platform every app is reached through one locked-down request path |
| medium | warn | policy | internal_hostname | config.js:3 | hard-coded internal hostname |
| medium | warn | secret | jwt | config.js:4 | JSON Web Token in source [eyJh************ (124 chars)] |

fix-loop packet written to findings.md (21 tasks)
```

Not in that table, on purpose: `app.py:10` (`sk_test_xxxx…` placeholder), `app.py:11`
(`# deploygate:allow`), `app.py:12` (a base64 content hash listed under `allow-value:` in
`.deploygateignore`), `app.py:13` (a UUID: hex never clears the entropy bar), `docs/notes.md` (a
fake AWS key, path ignored by `docs/**`), `tests/fixtures/sample_keys.txt` (a `sk_live_` key in a
fixture path) and `tests/helpers.py` (an `eval(` in test code).

`findings.md` (first tasks):

```
# Fix tasks for `sample_artifact`: BLOCKED

9 blocking task(s) must be fixed before redeploy; 12 warning(s) should be. Resolve each task, then push again: the gate re-runs.

## Task 1: app.py:9 - aws_access_key (critical, blocking)
- what: AWS access key id in source (match `AKIA************ (20 chars)`)
- change: remove the literal, read it from the platform secret store, and rotate the key in IAM

## Task 2: certs/server.pem - forbidden_file (critical, blocking)
- what: certificate / key material in the artifact
- change: remove it from the artifact (and from version control)

## Task 3: certs/server.pem:1 - private_key_block (critical, blocking)
- what: Private key block in source (match `----************ (27 chars)`)
- change: delete the key material from the artifact and provision certificates through the platform

## Task 4: .env.local - forbidden_file (high, blocking)
- what: environment file shipped in the artifact
- change: remove it from the artifact (and from version control)
...
```

`scan --dir data/clean_artifact` (exit 0):

```
# Deploy gate: PASS for `projects/p22_deploy_gate/data/clean_artifact`

4 files scanned; 0 findings (0 blocking, 0 warnings); secrets 0, policy 0, review 0 (model mock, 2 malformed rows dropped).

No findings.
```

`rules` lists the eight regex rules (with whether they block), the entropy bar, and every policy
check and pattern with its severity and file extensions.

## Design notes

- **Masking.** `mask()` keeps the first 4 characters and stars the rest (capped at 12) plus the
  length; the raw value is never stored on a `Finding`. The digest sent to the model withholds any
  line that holds a secret finding entirely (`[line withheld: slack_token, match xoxb****...]`) — a
  quote-based redaction was tried first and leaked an unquoted `.env.local` value, which is exactly what
  the "raw key absent from all output" test is for.
- **Rule precedence.** Rules are tried in file order and the first hit owns the line, so `jwt`
  beats `generic_api_key` on `SESSION_JWT = "eyJ..."`. The entropy detector only runs on lines no
  rule matched. A rule may name a `(?P<value>...)` group to say which part is the credential.
- **Entropy.** `H = -Σ p(c) log2 p(c)` over `[A-Za-z0-9+/=_-]{20,}` tokens with at least one letter
  and one digit; hex/UUID-shaped tokens are skipped outright (they top out at 4.0 bits/char), and
  the bar is 4.3. A random 34-char token scores 5.09.
- **Blocking.** Secrets block at critical/high; `forbidden_file`, `required_file_missing`,
  `manifest_invalid` and `blob_too_large` always block; code patterns and model-review findings
  warn, however severe, because a regex on `eval(` and a model's opinion are both evidence, not
  proof.
- **Strict review parsing.** A row is kept only if it is a dict, `severity` is one of
  `critical|high|medium|low|info`, `file` is a path that exists in the artifact (no hallucinated
  files), `why` is a non-empty string and `fix` is a string. The mock reviewer appends two
  malformed rows on every call so the drop path runs in the demo.
- **Fixture note.** The planted env file is `.env.local`, not `.env`, because the repo's root
  `.gitignore` ignores `.env` and the fixture would vanish from commits; the `.env.*` glob catches it.
- **Test paths** are skipped by directory marker (`tests/`, `fixtures/`, `__tests__/` ...) or file
  name (`test_*.py`, `*.spec.js`); forbidden-file and size checks still apply there.
- `evalkit` reuse: `MockLLM` / `get_client` / `LLMResponse.json`, `md_table`.
