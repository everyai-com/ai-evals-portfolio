"""Deploy gate: secret scanning, artifact policy checks and a model review, before anything ships.

Pipeline (Gate.run)
-------------------
1. SecretScanner   regex rules from secret_rules.yaml + a Shannon-entropy detector, with
                   false-positive controls: test-fixture paths are skipped, placeholder values
                   (sk_test_, xxxx, EXAMPLE ...) are ignored, `.deploygateignore` path globs and
                   `allow-value:` lines are honoured, and an inline `# deploygate:allow` marker
                   silences one line. Every finding carries a masked match; the raw secret is
                   never stored or printed.
2. PolicyChecks    forbidden files (.env, *.pem, node_modules, blobs over max_blob_mb), required
                   files, forbidden code patterns (eval(, child_process.exec + concat,
                   subprocess(shell=True) with an f-string, internal hostnames, binding 0.0.0.0
                   when the platform forbids own servers), and manifest schema validation.
3. ModelReview     a bounded digest (file tree + suspicious snippets with secrets masked) goes to
                   an LLMClient asking for JSON findings [{severity, file, why, fix}]; rows are
                   parsed strictly (known severity, file that exists in the artifact, string why)
                   and malformed rows are dropped and counted.
4. Verdict         blocked when any secret finding is critical/high or any forbidden/required/
                   manifest policy finding fires; everything else is a warning. `findings_md`
                   renders one task per finding (file, line, what to change) so the builder's
                   agent can fix and redeploy: the loop.

Entropy
-------
H(s) = -sum_c p(c) log2 p(c) over the characters of a candidate token (length >= min_length,
charset [A-Za-z0-9+/=_-]). Hex strings (hashes, UUIDs) cannot exceed 4.0 bits/char, random
base64 strings of 30+ chars sit at 4.5-5.0, so the default bar of 4.3 separates them.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evalkit.llm import LLMClient
from evalkit.report import md_table

HERE = Path(__file__).parent
SEVERITIES = ("critical", "high", "medium", "low", "info")
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
BLOCKING_SECRET = {"critical", "high"}
BLOCKING_POLICY_RULES = {"forbidden_file", "required_file_missing", "manifest_invalid", "blob_too_large"}
ALLOW_MARKER = "deploygate:allow"
TEXT_EXTENSIONS = {".py", ".js", ".ts", ".mjs", ".cjs", ".go", ".rb", ".java", ".kt", ".cs", ".php", ".sh",
                   ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".env", ".txt", ".md", ".html", ".css",
                   ".xml", ".sql", ".pem", ".key", ".crt", ".conf", ".properties", ""}


# ---------------------------------------------------------------- findings


@dataclass
class Finding:
    kind: str  # secret | policy | review
    rule: str
    severity: str
    file: str
    line: int | None
    message: str
    fix: str = ""
    masked: str = ""
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mask(value: str, keep: int = 4) -> str:
    """Keep the first `keep` chars, star the rest (capped), append the length. Never the raw value."""
    if not value:
        return ""
    head = value[:keep]
    stars = "*" * min(max(len(value) - keep, 4), 12)
    return f"{head}{stars} ({len(value)} chars)"


def shannon_entropy(s: str) -> float:
    """Bits per character: -sum p(c) log2 p(c)."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


# ---------------------------------------------------------------- allowlist


_HEXISH = re.compile(r"^(?:sha\d+-)?[0-9a-fA-F-]+$")
_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")


@dataclass
class Allowlist:
    path_globs: list[str] = field(default_factory=list)
    values: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, artifact_dir: Path, name: str = ".deploygateignore") -> Allowlist:
        p = artifact_dir / name
        al = cls()
        if not p.exists():
            return al
        for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("allow-value:"):
                al.values.add(line.split(":", 1)[1].strip())
            else:
                al.path_globs.append(line)
        return al

    def path_allowed(self, rel: str) -> bool:
        return any(_glob_match(rel, g) for g in self.path_globs)

    def value_allowed(self, value: str) -> bool:
        return any(v and v in value for v in self.values)


def _glob_match(rel: str, pattern: str) -> bool:
    """fnmatch where `**` also matches across directories and a bare name matches at any depth."""
    rel = rel.replace("\\", "/")
    if "/" not in pattern:
        return fnmatch.fnmatch(Path(rel).name, pattern)
    if pattern.endswith("/**"):
        base = pattern[:-3]
        return rel == base or rel.startswith(base + "/")
    return fnmatch.fnmatch(rel, pattern)


def is_test_path(rel: str, markers: Sequence[str]) -> bool:
    parts = [p.lower() for p in Path(rel).parts]
    name = parts[-1] if parts else ""
    if any(p in markers for p in parts[:-1]):
        return True
    return bool(re.match(r"^(test_.*|.*_test\.\w+|.*\.(test|spec)\.\w+)$", name))


def is_placeholder(value: str, markers: Sequence[str]) -> bool:
    v = value.lower()
    return any(m.lower() in v for m in markers)


# ---------------------------------------------------------------- secret scanner


@dataclass
class SecretRule:
    id: str
    name: str
    pattern: re.Pattern
    severity: str
    fix: str


class SecretScanner:
    def __init__(self, rules_path: str | Path | None = None, policy: Policy | None = None):
        doc = yaml.safe_load(Path(rules_path or HERE / "data" / "secret_rules.yaml").read_text(encoding="utf-8"))
        self.rules = [SecretRule(r["id"], r.get("name", r["id"]), re.compile(r["pattern"]), r.get("severity", "medium"),
                                 r.get("fix", "")) for r in doc["rules"]]
        for r in self.rules:
            if r.severity not in SEVERITIES:
                raise ValueError(f"rule {r.id}: unknown severity {r.severity}")
        ent = doc.get("entropy", {})
        self.entropy_min_len = int(ent.get("min_length", 20))
        self.entropy_bits = float(ent.get("min_bits_per_char", 4.3))
        self.entropy_severity = ent.get("severity", "medium")
        self.entropy_fix = ent.get("fix", "")
        self.policy = policy or Policy()

    # -- single line
    def scan_line(self, line: str, rel: str, lineno: int, allow: Allowlist | None = None) -> list[Finding]:
        allow = allow or Allowlist()
        if ALLOW_MARKER in line:
            return []
        out: list[Finding] = []
        for rule in self.rules:
            for m in rule.pattern.finditer(line):
                value = m.groupdict().get("value") or m.group(0)
                if is_placeholder(value, self.policy.placeholder_markers) or allow.value_allowed(value):
                    continue
                out.append(Finding("secret", rule.id, rule.severity, rel, lineno,
                                   f"{rule.name} in source", rule.fix, mask(value), rule.severity in BLOCKING_SECRET))
            if out:
                return out[:1]  # first (most specific) rule wins for the line
        for m in _TOKEN.finditer(line):
            tok = m.group(0)
            if len(tok) < self.entropy_min_len or _HEXISH.match(tok):
                continue
            if not (re.search(r"[0-9]", tok) and re.search(r"[A-Za-z]", tok)):
                continue
            if is_placeholder(tok, self.policy.placeholder_markers) or allow.value_allowed(tok):
                continue
            h = shannon_entropy(tok)
            if h >= self.entropy_bits:
                out.append(Finding("secret", "high_entropy_string", self.entropy_severity, rel, lineno,
                                   f"high-entropy string ({h:.2f} bits/char)", self.entropy_fix, mask(tok),
                                   self.entropy_severity in BLOCKING_SECRET))
                return out
        return out

    def scan_text(self, text: str, rel: str = "<text>", allow: Allowlist | None = None) -> list[Finding]:
        out = []
        for i, line in enumerate(text.splitlines(), 1):
            out.extend(self.scan_line(line, rel, i, allow))
        return out

    def scan_dir(self, artifact_dir: str | Path) -> list[Finding]:
        root = Path(artifact_dir)
        allow = Allowlist.load(root)
        out: list[Finding] = []
        for p in iter_files(root):
            rel = p.relative_to(root).as_posix()
            if allow.path_allowed(rel) or is_test_path(rel, self.policy.test_path_markers):
                continue
            if not is_text_file(p):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            out.extend(self.scan_text(text, rel, allow))
        return out


def iter_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)


def is_text_file(p: Path) -> bool:
    """Scan by extension; dotenv-style names (.env, .env.local) are always text."""
    return p.name.startswith(".env") or p.suffix.lower() in TEXT_EXTENSIONS


# ---------------------------------------------------------------- policy checks


@dataclass
class PolicyPattern:
    id: str
    regex: re.Pattern
    extensions: set[str]
    severity: str
    message: str
    fix: str
    requires: str | None = None


@dataclass
class Policy:
    required_files: list[str] = field(default_factory=lambda: ["manifest.yaml"])
    max_blob_mb: float = 25.0
    allow_own_server: bool = False
    forbidden_files: list[dict[str, str]] = field(default_factory=list)
    test_path_markers: list[str] = field(default_factory=lambda: ["test", "tests", "fixtures", "fixture", "__tests__", "testdata", "spec"])
    placeholder_markers: list[str] = field(default_factory=lambda: ["xxxx", "example", "placeholder", "changeme", "your_", "<", ">", "dummy", "fake", "redacted", "sk_test_", "pk_test_"])
    patterns: list[PolicyPattern] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Policy:
        doc = yaml.safe_load(Path(path or HERE / "data" / "policy.yaml").read_text(encoding="utf-8")) or {}
        pats = [PolicyPattern(p["id"], re.compile(p["regex"]), {e.lower() for e in p.get("extensions", [])},
                              p.get("severity", "medium"), p.get("message", p["id"]), p.get("fix", ""), p.get("requires"))
                for p in doc.get("patterns", [])]
        return cls(
            required_files=list(doc.get("required_files", ["manifest.yaml"])),
            max_blob_mb=float(doc.get("max_blob_mb", 25)),
            allow_own_server=bool(doc.get("allow_own_server", False)),
            forbidden_files=[f if isinstance(f, dict) else {"glob": f} for f in doc.get("forbidden_files", [])],
            test_path_markers=[str(m).lower() for m in doc.get("test_path_markers", cls().test_path_markers)],
            placeholder_markers=[str(m) for m in doc.get("placeholder_markers", cls().placeholder_markers)],
            patterns=pats,
        )


class ManifestValidator:
    def __init__(self, schema_path: str | Path | None = None):
        self.schema = yaml.safe_load(Path(schema_path or HERE / "data" / "manifest_schema.yaml").read_text(encoding="utf-8"))

    def validate(self, manifest: Any) -> list[str]:
        """Return a list of human-readable problems (empty = valid)."""
        errs: list[str] = []
        if not isinstance(manifest, dict):
            return ["manifest is not a mapping"]
        for key in self.schema.get("required", []):
            if key not in manifest:
                errs.append(f"missing required field '{key}'")
        for key, spec in self.schema.get("fields", {}).items():
            if key not in manifest:
                continue
            v = manifest[key]
            t = spec.get("type", "string")
            if t == "string":
                if not isinstance(v, str):
                    errs.append(f"'{key}' must be a string")
                    continue
                if "pattern" in spec and not re.match(spec["pattern"], v):
                    errs.append(f"'{key}' = {v!r} does not match {spec['pattern']}")
                if "enum" in spec and v not in spec["enum"]:
                    errs.append(f"'{key}' = {v!r} not one of {spec['enum']}")
            elif t == "list":
                if not isinstance(v, list):
                    errs.append(f"'{key}' must be a list")
                    continue
                if "max_items" in spec and len(v) > spec["max_items"]:
                    errs.append(f"'{key}' has {len(v)} items (max {spec['max_items']})")
                for item in v:
                    if "item_enum" in spec and item not in spec["item_enum"]:
                        errs.append(f"'{key}' item {item!r} not one of {spec['item_enum']}")
                    if "item_pattern" in spec and not (isinstance(item, str) and re.match(spec["item_pattern"], item)):
                        errs.append(f"'{key}' item {item!r} does not match {spec['item_pattern']}")
            elif t == "int" and not isinstance(v, int):
                errs.append(f"'{key}' must be an integer")
        for key in manifest:
            if key not in self.schema.get("fields", {}):
                errs.append(f"unknown field '{key}'")
        return errs


class PolicyChecks:
    def __init__(self, policy: Policy | None = None, validator: ManifestValidator | None = None):
        self.policy = policy or Policy.load()
        self.validator = validator or ManifestValidator()

    def run(self, artifact_dir: str | Path) -> list[Finding]:
        root = Path(artifact_dir)
        pol = self.policy
        out: list[Finding] = []
        files = iter_files(root)
        rels = [p.relative_to(root).as_posix() for p in files]
        for req in pol.required_files:
            if req not in rels:
                out.append(Finding("policy", "required_file_missing", "critical", req, None,
                                   f"required file {req} is missing", f"add {req} to the artifact root", blocking=True))
            elif req.endswith((".yaml", ".yml")):
                try:
                    manifest = yaml.safe_load((root / req).read_text(encoding="utf-8"))
                    problems = self.validator.validate(manifest)
                except yaml.YAMLError as e:
                    problems = [f"not valid YAML: {e}"]
                for pr in problems:
                    out.append(Finding("policy", "manifest_invalid", "high", req, None, pr,
                                       "fix the manifest to match manifest_schema.yaml", blocking=True))
        limit = pol.max_blob_mb * 1024 * 1024
        for p, rel in zip(files, rels):
            for fb in pol.forbidden_files:
                if _glob_match(rel, fb["glob"]):
                    out.append(Finding("policy", "forbidden_file", fb.get("severity", "high"), rel, None,
                                       fb.get("message", f"forbidden file {fb['glob']}"),
                                       "remove it from the artifact (and from version control)", blocking=True))
                    break
            size = p.stat().st_size
            if size > limit:
                out.append(Finding("policy", "blob_too_large", "high", rel, None,
                                   f"{size / 1024 / 1024:.1f} MB exceeds {pol.max_blob_mb:.0f} MB",
                                   "move large assets to object storage", blocking=True))
            if is_test_path(rel, pol.test_path_markers):
                continue
            ext = ".env" if p.name.startswith(".env") else p.suffix.lower()
            active = [pt for pt in pol.patterns if (not pt.extensions or ext in pt.extensions)
                      and not (pt.requires == "own_server_forbidden" and pol.allow_own_server)]
            if not active or not is_text_file(p):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if ALLOW_MARKER in line:
                    continue
                for pt in active:
                    if pt.regex.search(line):
                        out.append(Finding("policy", pt.id, pt.severity, rel, i, pt.message, pt.fix))
        return out


# ---------------------------------------------------------------- model review


REVIEW_SYSTEM = (
    "You review application artifacts before deploy. Answer ONLY with a JSON array of findings, each "
    '{"severity": "critical|high|medium|low|info", "file": "<path in the tree>", "why": "<one sentence>", '
    '"fix": "<what to change>"}. Return [] when nothing is obviously bad.'
)


def build_digest(artifact_dir: str | Path, findings: Sequence[Finding], max_chars: int = 6000,
                 max_snippet_lines: int = 40) -> str:
    """File tree + suspicious lines (the ones existing findings point at), bounded. Lines that hold a
    secret finding are withheld entirely (rule + masked match only), so no credential reaches the model."""
    root = Path(artifact_dir)
    files = iter_files(root)
    tree = "\n".join(f"  {p.relative_to(root).as_posix()} ({p.stat().st_size} B)" for p in files)
    snippets: list[str] = []
    seen: set[tuple[str, int]] = set()
    secret_lines = {(f.file, f.line): f for f in findings if f.kind == "secret" and f.line is not None}
    for f in sorted(findings, key=lambda f: (f.file, f.line or 0)):
        if f.line is None or (f.file, f.line) in seen or len(snippets) >= max_snippet_lines:
            continue
        seen.add((f.file, f.line))
        sec = secret_lines.get((f.file, f.line))
        if sec is not None:
            # never send the line itself: the reviewer only needs to know a credential sits here
            snippets.append(f"  {f.file}:{f.line}: [line withheld: {sec.rule}, match {sec.masked}]")
            continue
        try:
            line = (root / f.file).read_text(encoding="utf-8", errors="replace").splitlines()[f.line - 1]
        except (OSError, IndexError):
            continue
        snippets.append(f"  {f.file}:{f.line}: {line.strip()[:160]}")
    text = "Artifact tree:\n" + tree + "\n\nSuspicious lines:\n" + ("\n".join(snippets) or "  (none)")
    return text[:max_chars] + ("\n[digest truncated]" if len(text) > max_chars else "")


@dataclass
class ReviewResult:
    findings: list[Finding]
    dropped: int
    raw_rows: int
    model: str


class ModelReview:
    def __init__(self, client: LLMClient, max_chars: int = 6000):
        self.client = client
        self.max_chars = max_chars

    def review(self, artifact_dir: str | Path, prior: Sequence[Finding]) -> ReviewResult:
        root = Path(artifact_dir)
        digest = build_digest(root, prior, self.max_chars)
        resp = self.client.complete("Review this artifact.\n\n" + digest, system=REVIEW_SYSTEM, max_tokens=1024)
        try:
            rows = resp.json()
        except (ValueError, json.JSONDecodeError):
            return ReviewResult([], 0, 0, self.client.model)
        if not isinstance(rows, list):
            return ReviewResult([], 1, 1, self.client.model)
        known = {p.relative_to(root).as_posix() for p in iter_files(root)}
        out, dropped = [], 0
        for row in rows:
            f = parse_review_row(row, known)
            if f is None:
                dropped += 1
            else:
                out.append(f)
        return ReviewResult(out, dropped, len(rows), self.client.model)


def parse_review_row(row: Any, known_files: set[str]) -> Finding | None:
    """Strict: dict, severity in SEVERITIES, file exists in the artifact, non-empty string why."""
    if not isinstance(row, dict):
        return None
    sev, file, why, fix = row.get("severity"), row.get("file"), row.get("why"), row.get("fix", "")
    if sev not in SEVERITIES or not isinstance(file, str) or file not in known_files:
        return None
    if not isinstance(why, str) or not why.strip() or not isinstance(fix, str):
        return None
    line = row.get("line")
    line = int(line) if isinstance(line, int) and line > 0 else None
    return Finding("review", "model_review", sev, file, line, why.strip(), fix.strip())


def mock_review_handler(system: str, prompt: str) -> str:
    """Deterministic stand-in for the reviewer: flags the planted patterns it can see in the digest
    and always appends two malformed rows (unknown severity; a bare string) that the strict parser
    must drop."""
    rows: list[Any] = []
    for file_line in re.findall(r"^  (\S+):(\d+): (.*)$", prompt, re.MULTILINE):
        file, line, text = file_line
        if "shell=True" in text:
            rows.append({"severity": "high", "file": file, "line": int(line),
                         "why": "shell=True with interpolated input lets a path like '; rm -rf /' run",
                         "fix": "subprocess.run(['ls', '-la', path], capture_output=True)"})
        elif "0.0.0.0" in text:
            rows.append({"severity": "medium", "file": file, "line": int(line),
                         "why": "the app starts its own server; on the platform every app is reached through one locked-down request path",
                         "fix": "delete the app.run block and export the Flask app for the platform runner"})
        elif "eval(" in text:
            rows.append({"severity": "high", "file": file, "line": int(line),
                         "why": "eval over a template string is remote code execution if the template is user-supplied",
                         "fix": "use a template engine with autoescaping"})
        elif "exec(" in text:
            rows.append({"severity": "high", "file": file, "line": int(line),
                         "why": "exec builds a shell command from its argument",
                         "fix": "execFile('ls', [cmd], cb)"})
    rows.append({"severity": "whatever", "file": "app.py", "why": "malformed severity"})
    rows.append("not a finding")
    return json.dumps(rows)


def default_review_client() -> LLMClient:
    from evalkit.llm import get_client

    return get_client(handlers=[(r"Review this artifact", mock_review_handler)])


# ---------------------------------------------------------------- gate


@dataclass
class Verdict:
    artifact: str
    blocked: bool
    findings: list[Finding]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact, "blocked": self.blocked, "summary": self.summary,
                "findings": [f.to_dict() for f in self.findings]}

    def to_markdown(self) -> str:
        s = self.summary
        head = (f"# Deploy gate: {'BLOCKED' if self.blocked else 'PASS'} for `{self.artifact}`\n\n"
                f"{s['n_files']} files scanned; {s['n_findings']} findings "
                f"({s['n_blocking']} blocking, {s['n_warnings']} warnings); "
                f"secrets {s['by_kind'].get('secret', 0)}, policy {s['by_kind'].get('policy', 0)}, "
                f"review {s['by_kind'].get('review', 0)} (model {s['review_model']}, "
                f"{s['review_dropped']} malformed rows dropped).")
        if not self.findings:
            return head + "\n\nNo findings."
        rows = [[f.severity, "block" if f.blocking else "warn", f.kind, f.rule,
                 f"{f.file}:{f.line}" if f.line else f.file, f.message + (f" [{f.masked}]" if f.masked else "")]
                for f in self.findings]
        return head + "\n\n" + md_table(["severity", "gate", "kind", "rule", "where", "finding"], rows)


class Gate:
    def __init__(self, scanner: SecretScanner | None = None, checks: PolicyChecks | None = None,
                 review: ModelReview | None = None, use_review: bool = True):
        self.checks = checks or PolicyChecks()
        self.scanner = scanner or SecretScanner(policy=self.checks.policy)
        self.review = review if review is not None else (ModelReview(default_review_client()) if use_review else None)

    def run(self, artifact_dir: str | Path) -> Verdict:
        root = Path(artifact_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")
        findings = self.scanner.scan_dir(root) + self.checks.run(root)
        review_model, dropped = "-", 0
        if self.review is not None:
            rr = self.review.review(root, findings)
            findings += rr.findings
            review_model, dropped = rr.model, rr.dropped
        findings.sort(key=lambda f: (not f.blocking, SEV_RANK[f.severity], f.file, f.line or 0, f.rule))
        by_kind = Counter(f.kind for f in findings)
        by_sev = Counter(f.severity for f in findings)
        n_block = sum(f.blocking for f in findings)
        summary = {
            "n_files": len(iter_files(root)), "n_findings": len(findings), "n_blocking": n_block,
            "n_warnings": len(findings) - n_block, "by_kind": dict(by_kind), "by_severity": dict(by_sev),
            "review_model": review_model, "review_dropped": dropped,
        }
        return Verdict(str(root), n_block > 0, findings, summary)


def findings_md(verdict: Verdict) -> str:
    """The fix-loop packet: one task per finding, phrased for the builder's agent."""
    lines = [f"# Fix tasks for `{verdict.artifact}`: {'BLOCKED' if verdict.blocked else 'PASS'}",
             "", (f"{verdict.summary['n_blocking']} blocking task(s) must be fixed before redeploy; "
             f"{verdict.summary['n_warnings']} warning(s) should be. Resolve each task, then push again: the gate re-runs."), ""]
    for i, f in enumerate(verdict.findings, 1):
        where = f"{f.file}:{f.line}" if f.line else f.file
        lines += [f"## Task {i}: {where} - {f.rule} ({f.severity}, {'blocking' if f.blocking else 'warning'})",
                  f"- what: {f.message}" + (f" (match `{f.masked}`)" if f.masked else ""),
                  f"- change: {f.fix or 'remove or rewrite the flagged line'}", ""]
    if not verdict.findings:
        lines.append("Nothing to fix.")
    return "\n".join(lines)
