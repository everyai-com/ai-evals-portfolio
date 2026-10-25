"""Tests prove the README claims: every planted secret is caught by the right rule and masked (the
raw value never appears in any output), placeholders / allowlisted values / fixtures are not flagged,
entropy catches random tokens and not hashes, policy checks catch .env and shell=True, the manifest is
validated, model-review rows are parsed strictly, the clean artifact passes, findings.md has one task
per finding."""

import atexit
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from evalkit.llm import MockLLM
from projects.p22_deploy_gate import core
from projects.p22_deploy_gate.cli import main
from projects.p22_deploy_gate.fixtures import FAKE, materialize_sample

DATA = Path(__file__).resolve().parents[1] / "data"
# The demo artifact is built from a placeholder template at import time (see fixtures.py), so no
# secret-shaped string is ever committed. One build is shared by every test in this module.
_SAMPLE_ROOT = Path(tempfile.mkdtemp(prefix="p22-sample-"))
atexit.register(shutil.rmtree, _SAMPLE_ROOT, True)
SAMPLE = materialize_sample(_SAMPLE_ROOT / "sample_artifact")
CLEAN = DATA / "clean_artifact"

PLANTED = {  # rule -> (file, raw secret that must never be printed)
    "aws_access_key": ("app.py", FAKE["aws_1"]),
    "github_token": ("config.js", FAKE["github_1"]),
    "slack_token": (".env.local", FAKE["slack_1"]),
    "url_credentials": ("config.js", "S3cretPassw0rd"),
    "jwt": ("config.js", "sflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"),
    "private_key_block": ("certs/server.pem", "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDfakekeyfakekey"),
    "high_entropy_string": ("app.py", "Qm7vX2pLk9ZrT4wNc8Yb3HsJ6dF1gA5eUo"),
}


@pytest.fixture(scope="module")
def verdict():
    return core.Gate().run(SAMPLE)


@pytest.fixture(scope="module")
def all_output(verdict, tmp_path_factory):
    tmp_path_factory.mktemp("out")
    md = verdict.to_markdown()
    fm = core.findings_md(verdict)
    js = json.dumps(verdict.to_dict())
    digest = core.build_digest(SAMPLE, verdict.findings)
    return md + fm + js + digest


# ---------------------------------------------------------------- secrets


def test_each_planted_secret_detected_with_right_rule_and_masked(verdict, all_output):
    by_rule = {}
    for f in verdict.findings:
        if f.kind == "secret":
            by_rule.setdefault(f.rule, []).append(f)
    for rule, (file, raw) in PLANTED.items():
        hits = [f for f in by_rule.get(rule, []) if f.file == file]
        assert hits, f"{rule} not found in {file}"
        assert hits[0].masked and "*" in hits[0].masked and raw not in hits[0].masked
        assert raw not in all_output, f"raw secret for {rule} leaked into output"
    assert any(f.rule == "stripe_live_key" for f in verdict.findings) is False  # only in the skipped fixture
    assert FAKE["stripe_live_1"] not in all_output


def test_placeholder_allowlisted_and_fixture_values_not_flagged(verdict):
    lines = {(f.file, f.line) for f in verdict.findings if f.kind == "secret"}
    assert ("app.py", 10) not in lines  # sk_test_xxxx placeholder
    assert ("app.py", 11) not in lines  # inline deploygate:allow
    assert ("app.py", 12) not in lines  # allow-value in .deploygateignore
    assert ("app.py", 13) not in lines  # UUID
    assert ("app.py", 15) not in lines  # os.environ.get
    assert not any(f.file.startswith("tests/") for f in verdict.findings if f.kind == "secret")
    assert not any(f.file.startswith("docs/") for f in verdict.findings)
    assert not any(f.rule == "eval_call" and f.file.startswith("tests/") for f in verdict.findings)


def test_scan_line_rules_and_placeholders():
    sc = core.SecretScanner()
    assert sc.scan_line('key = "AKIAIOSFODNN7EXAMPLE"', "a.py", 1) == []  # AWS doc example value
    f = sc.scan_line('key = "AKI' 'A4Q7ZP2M9XCV1BTR8"', "a.py", 1)
    assert [x.rule for x in f] == ["aws_access_key"] and f[0].blocking and f[0].masked.startswith("AKIA****")
    f = sc.scan_line('password = "Tr0ub4dor3xyzxyzxyzQ"', "a.py", 2)
    assert f and f[0].rule == "generic_api_key" and f[0].masked.startswith("Tr0u")
    assert sc.scan_line('api_key = os.environ["API_KEY"]', "a.py", 3) == []
    assert sc.scan_line('token = "AKI' 'A4Q7ZP2M9XCV1BTR8"  # deploygate:allow', "a.py", 4) == []
    assert sc.scan_line('u = "https://me:hunter2hunter2@db.example.org/x"', "a.py", 5) == []  # example host = placeholder
    assert sc.scan_line('u = "https://me:hunter2hunter2@db.prod.org/x"', "a.py", 6)[0].rule == "url_credentials"


def test_entropy_catches_random_token_and_ignores_hashes_and_allowed_values():
    sc = core.SecretScanner()
    assert core.shannon_entropy("aaaa") == 0.0
    assert core.shannon_entropy("0123456789abcdef") == pytest.approx(4.0)
    random_tok = 'nonce = "Qm7vX2pLk9ZrT4wNc8Yb3HsJ6dF1gA5eUo"'
    f = sc.scan_line(random_tok, "a.py", 1)
    assert f and f[0].rule == "high_entropy_string" and "Qm7vX2pLk9ZrT4wNc8Yb3HsJ6dF1gA5eUo" not in f[0].masked
    assert sc.scan_line('id = "550e8400-e29b-41d4-a716-446655440000"', "a.py", 2) == []
    assert sc.scan_line('h = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"', "a.py", 3) == []
    b64_hash = 'integrity = "sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="'
    assert sc.scan_line(b64_hash, "a.py", 4)[0].rule == "high_entropy_string"  # would fire ...
    allow = core.Allowlist(values={"47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="})
    assert sc.scan_line(b64_hash, "a.py", 4, allow) == []  # ... unless listed as allowed


def test_test_paths_and_globs():
    assert core.is_test_path("tests/fixtures/keys.txt", ["tests", "fixtures"])
    assert core.is_test_path("src/test_app.py", ["tests"]) and core.is_test_path("src/app.spec.js", ["tests"])
    assert not core.is_test_path("src/app.py", ["tests"])
    assert core._glob_match("docs/a/b.md", "docs/**") and core._glob_match("x/.env", ".env")
    assert core._glob_match("node_modules/a/b.js", "node_modules/**") and not core._glob_match("src/a.js", "node_modules/**")


# ---------------------------------------------------------------- policy


def test_policy_checks_catch_env_pem_node_modules_and_shell_true(verdict):
    pol = {(f.rule, f.file) for f in verdict.findings if f.kind == "policy"}
    assert ("forbidden_file", ".env.local") in pol
    assert ("forbidden_file", "certs/server.pem") in pol
    assert ("forbidden_file", "node_modules/left-pad/index.js") in pol
    assert ("subprocess_shell_fstring", "app.py") in pol
    assert ("eval_call", "config.js") in pol
    assert ("child_process_exec_concat", "config.js") in pol
    assert ("bind_all_interfaces", "app.py") in pol
    assert ("internal_hostname", "config.js") in pol
    blocking = {f.rule for f in verdict.findings if f.blocking}
    assert "forbidden_file" in blocking and "subprocess_shell_fstring" not in blocking


def test_policy_patterns_on_hand_built_lines(tmp_path):
    (tmp_path / "manifest.yaml").write_text((CLEAN / "manifest.yaml").read_text())
    (tmp_path / "a.py").write_text(
        'subprocess.run(["ls", path])\n'
        'subprocess.run(f"ls {p}", shell=True)\n'
        'subprocess.run("ls", shell=True)\n'  # shell=True without f-string: not the planted pattern
        'app.run(host="127.0.0.1")\n'
    )
    findings = core.PolicyChecks().run(tmp_path)
    assert [(f.rule, f.line) for f in findings] == [("subprocess_shell_fstring", 2)]
    pol = core.Policy.load()
    pol.allow_own_server = True
    (tmp_path / "b.py").write_text('app.run(host="0.0.0.0")\n')
    assert not any(f.rule == "bind_all_interfaces" for f in core.PolicyChecks(pol).run(tmp_path))
    assert any(f.rule == "bind_all_interfaces" for f in core.PolicyChecks().run(tmp_path))


def test_blob_size_limit(tmp_path):
    shutil.copytree(CLEAN, tmp_path / "art")
    pol = core.Policy.load()
    pol.max_blob_mb = 0.001  # 1 KB
    (tmp_path / "art" / "big.bin").write_bytes(b"\0" * 4096)
    f = [x for x in core.PolicyChecks(pol).run(tmp_path / "art") if x.rule == "blob_too_large"]
    assert f and f[0].file == "big.bin" and f[0].blocking


def test_manifest_validation(tmp_path):
    v = core.ManifestValidator()
    good = {"name": "team-digest", "owner": "a@b.co", "risk_tier": "low", "data_classes": ["internal"], "outbound_domains": ["api.example.com"]}
    assert v.validate(good) == []
    bad = {"name": "Bad Name!", "owner": "nobody", "risk_tier": "extreme", "data_classes": ["secret"], "outbound_domains": ["*"], "extra": 1}
    errs = v.validate(bad)
    assert len(errs) == 6 and any("risk_tier" in e for e in errs) and any("unknown field" in e for e in errs)
    assert v.validate("nope") == ["manifest is not a mapping"]
    assert v.validate({}) and all(e.startswith("missing required") for e in v.validate({}))
    shutil.copytree(CLEAN, tmp_path / "art")
    (tmp_path / "art" / "manifest.yaml").write_text("name: ok-app\nowner: a@b.co\n")
    f = core.PolicyChecks().run(tmp_path / "art")
    assert {x.rule for x in f} == {"manifest_invalid"} and all(x.blocking for x in f)
    (tmp_path / "art" / "manifest.yaml").unlink()
    assert [x.rule for x in core.PolicyChecks().run(tmp_path / "art")] == ["required_file_missing"]


# ---------------------------------------------------------------- model review


def test_model_review_parses_strictly_and_drops_malformed_rows(verdict):
    rows = [
        {"severity": "high", "file": "app.py", "why": "bad", "fix": "fix it", "line": 19},
        {"severity": "urgent", "file": "app.py", "why": "unknown severity"},
        {"severity": "high", "file": "ghost.py", "why": "hallucinated file"},
        {"severity": "low", "file": "app.py", "why": ""},
        {"severity": "low", "file": "app.py"},
        "just a string",
        {"severity": "medium", "file": "config.js", "why": "ok", "fix": 3},
    ]
    client = MockLLM(handlers=[(r"Review this artifact", json.dumps(rows))])
    rr = core.ModelReview(client).review(SAMPLE, verdict.findings)
    assert rr.raw_rows == 7 and rr.dropped == 6 and len(rr.findings) == 1
    assert rr.findings[0].kind == "review" and rr.findings[0].line == 19 and not rr.findings[0].blocking
    garbage = MockLLM(handlers=[(r"Review this artifact", "I cannot help with that")])
    assert core.ModelReview(garbage).review(SAMPLE, []).findings == []
    not_list = MockLLM(handlers=[(r"Review this artifact", '{"severity": "high"}')])
    assert core.ModelReview(not_list).review(SAMPLE, []).dropped == 1
    # the default mock reviewer flags the planted cases and its two malformed rows are dropped
    assert verdict.summary["review_dropped"] == 2
    assert {f.file for f in verdict.findings if f.kind == "review"} == {"app.py", "config.js"}


def test_digest_is_bounded_and_masks_secrets(verdict):
    d = core.build_digest(SAMPLE, verdict.findings, max_chars=400)
    assert len(d) <= 400 + len("\n[digest truncated]") and d.endswith("[digest truncated]")
    full = core.build_digest(SAMPLE, verdict.findings)
    assert "Artifact tree:" in full and "app.py:19" in full
    for _, raw in PLANTED.values():
        assert raw not in full


# ---------------------------------------------------------------- gate / CLI


def test_sample_blocked_clean_passes_and_exit_codes(tmp_path):
    assert core.Gate().run(CLEAN).blocked is False
    assert main(["scan", "--dir", str(CLEAN)]) == 0
    assert main(["scan", "--dir", str(SAMPLE), "--json", str(tmp_path / "v.json")]) == 1
    payload = json.loads((tmp_path / "v.json").read_text())
    assert payload["blocked"] and payload["summary"]["n_blocking"] >= 5
    assert main(["scan", "--dir", str(tmp_path / "nope")]) == 2
    assert main(["rules"]) == 0
    assert main(["scan", "--dir", str(SAMPLE), "--no-review"]) == 1


def test_findings_md_has_one_task_per_finding(tmp_path, verdict):
    out = tmp_path / "findings.md"
    assert main(["scan", "--dir", str(SAMPLE), "--findings", str(out)]) == 1
    text = out.read_text()
    assert text.count("## Task ") == len(verdict.findings) > 0
    for f in verdict.findings:
        assert f"{f.file}:{f.line}" in text if f.line else f.file in text
    assert "BLOCKED" in text and "- change:" in text
    for _, raw in PLANTED.values():
        assert raw not in text
    clean = core.findings_md(core.Gate().run(CLEAN))
    assert "Nothing to fix" in clean and "## Task" not in clean
