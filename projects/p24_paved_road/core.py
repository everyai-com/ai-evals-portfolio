"""Paved road: PlatformSpec, deterministic generators (llms.txt, skill, AGENTS.md snippet),
CompatValidator (deploy-time precheck in builder language), FirstRunStudy (usability log).

Nothing here calls a model: the paved road exists so the *builder's* agent does not have to guess.
"""
from __future__ import annotations

import fnmatch
import json
import re
import statistics
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evalkit.io import read_jsonl

HERE = Path(__file__).parent
DEFAULT_SPEC = HERE / "data" / "platform.yaml"
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build", ".next"}
SOURCE_EXT = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".py", ".html", ".json", ".yaml", ".yml", ".toml", ".txt", ".md"}
CODE_EXT = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".py"}


# ---------------------------------------------------------------- spec


@dataclass
class PlatformSpec:
    name: str
    tagline: str
    studio_url: str
    docs_url: str
    deploys: list[str]
    does_not_deploy: list[str]
    runtimes: dict[str, list[str]]                 # runtime -> allowed versions
    limits: dict[str, int]
    manifest: dict[str, Any]                       # file, required_fields, optional_fields, example
    entrypoint: dict[str, dict[str, str]]          # runtime -> {file, export, signature}
    forbidden_dependencies: dict[str, str]         # dep -> why
    server_patterns: list[str]
    secrets: dict[str, Any]                        # forbidden_files, rule
    outbound: dict[str, Any]                       # allowed_domains, ignored_domains, rule
    deploy: dict[str, Any]                         # command, precheck_command, url_pattern, steps
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SPEC) -> PlatformSpec:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlatformSpec:
        return cls(
            name=d["name"], tagline=d.get("tagline", ""), studio_url=d.get("studio_url", ""),
            docs_url=d.get("docs_url", ""), deploys=list(d.get("deploys", [])),
            does_not_deploy=list(d.get("does_not_deploy", [])),
            runtimes={k: [str(v) for v in vs] for k, vs in d.get("runtimes", {}).items()},
            limits=dict(d.get("limits", {})), manifest=dict(d.get("manifest", {})),
            entrypoint={k: dict(v) for k, v in d.get("entrypoint", {}).items()},
            forbidden_dependencies=dict(d.get("forbidden_dependencies", {})),
            server_patterns=list(d.get("server_patterns", [])), secrets=dict(d.get("secrets", {})),
            outbound=dict(d.get("outbound", {})), deploy=dict(d.get("deploy", {})), raw=d)

    @property
    def manifest_file(self) -> str:
        return self.manifest.get("file", "manifest.yaml")

    @property
    def required_fields(self) -> list[str]:
        return list(self.manifest.get("required_fields", []))

    @property
    def deploy_command(self) -> str:
        return self.deploy.get("command", "deploy")

    @property
    def url_pattern(self) -> str:
        return self.deploy.get("url_pattern", "")

    def runtime_names(self) -> list[str]:
        """`node18`, `node20`, `python3.11` ... as a builder writes them in the manifest."""
        return [f"{rt}{v}" for rt, vs in self.runtimes.items() for v in vs]

    def split_runtime(self, value: str) -> tuple[str, str] | None:
        """`node20` -> ("node", "20"); `python3.12` -> ("python", "3.12"); unknown -> None."""
        m = re.match(r"^([a-z]+)[-_ ]?([0-9][0-9.]*)?$", str(value).strip().lower())
        if not m:
            return None
        rt, ver = m.group(1), (m.group(2) or "")
        if rt not in self.runtimes:
            return None
        return rt, ver


# ---------------------------------------------------------------- generators (deterministic text)


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def _example_manifest(spec: PlatformSpec) -> str:
    ex = spec.manifest.get("example") or {f: f"<{f}>" for f in spec.required_fields}
    return yaml.safe_dump(ex, sort_keys=False).strip()


def _example_handler(spec: PlatformSpec) -> str:
    node = spec.entrypoint.get("node", {})
    py = spec.entrypoint.get("python", {})
    parts = []
    if node:
        parts.append(f"// {node.get('file', 'handler.js')}  ({node.get('signature', '')})\n"
                     f"export async function {node.get('export', 'handler')}(request) {{\n"
                     "  const url = new URL(request.url);\n"
                     "  if (url.pathname === '/api/hello') return Response.json({ hello: 'world' });\n"
                     "  return new Response('not found', { status: 404 });  // static files in public/ are served for you\n"
                     "}")
    if py:
        parts.append(f"# {py.get('file', 'handler.py')}  ({py.get('signature', '')})\n"
                     f"def {py.get('export', 'handler')}(request):\n"
                     "    if request.path == '/api/hello':\n"
                     "        return {'status': 200, 'json': {'hello': 'world'}}\n"
                     "    return {'status': 404, 'body': 'not found'}")
    return "\n\n".join(parts)


def checklist(spec: PlatformSpec) -> list[str]:
    items = [
        f"`{spec.manifest_file}` exists with {', '.join(spec.required_fields)}",
        f"`runtime` is one of {', '.join(spec.runtime_names())}",
        "the entrypoint exports `" + "` / `".join(sorted({e.get('export', 'handler') for e in spec.entrypoint.values()})) + "` and nothing calls listen()/run()",
        "no forbidden dependency: " + ", ".join(sorted(spec.forbidden_dependencies)),
        f"no file over {spec.limits.get('max_file_bytes', 0) // (1024 * 1024)} MiB; bundle under {spec.limits.get('max_bundle_bytes', 0) // (1024 * 1024)} MiB",
        "no committed secrets (" + ", ".join(spec.secrets.get("forbidden_files", [])) + "); env names declared under `env`",
        "every outbound URL is on the allowlist: " + ", ".join(spec.outbound.get("allowed_domains", [])),
        f"`{spec.deploy.get('precheck_command', 'validate')}` prints no blocking findings",
    ]
    return items


def render_llms_txt(spec: PlatformSpec) -> str:
    """Concise, agent-readable contract. Deterministic: same spec -> byte-identical text."""
    forbidden = "\n".join(f"- `{dep}`: {why}" for dep, why in sorted(spec.forbidden_dependencies.items()))
    runtimes = "\n".join(f"- {rt}: {', '.join(vs)}" for rt, vs in spec.runtimes.items())
    entry = "\n".join(f"- {rt}: `{e.get('file')}` must export `{e.get('export')}` — `{e.get('signature')}`"
                      for rt, e in spec.entrypoint.items())
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(spec.deploy.get("steps", []), 1))
    return f"""# {spec.name}

> {spec.tagline}. {spec.name} deploys static files plus serverless handlers to `{spec.url_pattern}`.
> This file is for coding agents: read it before generating an app for {spec.name}.

## What works

{_bullets(spec.deploys)}

## What fails, and why

{_bullets(spec.does_not_deploy)}

Forbidden dependencies (the deploy precheck rejects them):
{forbidden}

## Runtimes

{runtimes}
Write the runtime in the manifest as `<runtime><version>`, e.g. `{spec.runtime_names()[0]}`.

## Manifest: `{spec.manifest_file}`

Required: {', '.join(spec.required_fields)}. Optional: {', '.join(spec.manifest.get('optional_fields', []))}.

```yaml
{_example_manifest(spec)}
```

## Entrypoint

{entry}

```
{_example_handler(spec)}
```

## Limits

- max file size: {spec.limits.get('max_file_bytes')} bytes
- max bundle size: {spec.limits.get('max_bundle_bytes')} bytes
- max files: {spec.limits.get('max_files')}

## Secrets

{spec.secrets.get('rule', '')}
Never commit: {', '.join(spec.secrets.get('forbidden_files', []))}.

## Outbound network

{spec.outbound.get('rule', '')}
Allowed: {', '.join(spec.outbound.get('allowed_domains', []))}.

## Deploy

{steps}

```
{spec.deploy.get('precheck_command', '')}
{spec.deploy_command}
# -> {spec.url_pattern}
```

## Checklist before you say "done"

{_bullets(checklist(spec))}

Hosted studio (no zip, no local setup): {spec.studio_url}
Docs: {spec.docs_url}
"""


def render_reference_md(spec: PlatformSpec) -> str:
    rows = [f"| `{dep}` | {why} |" for dep, why in sorted(spec.forbidden_dependencies.items())]
    pats = "\n".join(f"- `{p}`" for p in spec.server_patterns)
    return f"""# {spec.name} rules reference

## Deployable shape

{_bullets(spec.deploys)}

## Not deployable

{_bullets(spec.does_not_deploy)}

## Forbidden dependencies

| dependency | why it fails |
|---|---|
{chr(10).join(rows)}

## Own-server patterns the precheck rejects

{pats}

## Manifest `{spec.manifest_file}`

- required: {', '.join(spec.required_fields)}
- optional: {', '.join(spec.manifest.get('optional_fields', []))}
- runtime values: {', '.join(spec.runtime_names())}

## Entrypoint

{chr(10).join(f"- {rt}: `{e.get('file')}` exports `{e.get('export')}` — `{e.get('signature')}`" for rt, e in spec.entrypoint.items())}

## Limits

- max_file_bytes: {spec.limits.get('max_file_bytes')}
- max_bundle_bytes: {spec.limits.get('max_bundle_bytes')}
- max_files: {spec.limits.get('max_files')}

## Secrets

- {spec.secrets.get('rule', '')}
- forbidden files: {', '.join(spec.secrets.get('forbidden_files', []))}

## Outbound

- {spec.outbound.get('rule', '')}
- allowed: {', '.join(spec.outbound.get('allowed_domains', []))}

## Deploy

- precheck: `{spec.deploy.get('precheck_command', '')}`
- deploy: `{spec.deploy_command}`
- url: `{spec.url_pattern}`
"""


def render_skill(spec: PlatformSpec) -> dict[str, str]:
    """Files of an agent skill: SKILL.md (frontmatter + procedure) and reference.md (the rules)."""
    name = re.sub(r"[^a-z0-9]+", "-", spec.name.lower()).strip("-") + "-app"
    skill = f"""---
name: {name}
description: Build and deploy an app for {spec.name} ({spec.tagline}). Use when asked to create, port, fix or deploy an app for {spec.name}, or when a deploy precheck reports findings.
---

# Build for {spec.name}

{spec.name} deploys static files plus serverless handlers. It owns the port and the router: an app
that starts its own server cannot be deployed. Read `reference.md` for the exact rules.

## Procedure

1. Write `{spec.manifest_file}` with {', '.join(spec.required_fields)}; runtime is one of {', '.join(spec.runtime_names())}.
2. Put static files in `public/`; export `handler` from the entrypoint ({', '.join(f"`{e.get('file')}`" for e in spec.entrypoint.values())}). Never call listen()/run().
3. Do not add {', '.join(f'`{d}`' for d in sorted(spec.forbidden_dependencies))}. If a generator (prototyping studio, scaffold) added one, remove it and move the routes into the handler.
4. Declare env names under `env`; never write a `.env` file into the bundle.
5. Only call allowlisted domains: {', '.join(spec.outbound.get('allowed_domains', []))}.
6. Run `{spec.deploy.get('precheck_command', '')}`; apply every `fix` until no blocking finding remains.
7. Run `{spec.deploy_command}`; the app is at `{spec.url_pattern}`.

## Checklist

{_bullets(checklist(spec))}

## Minimal example

```yaml
{_example_manifest(spec)}
```

```
{_example_handler(spec)}
```
"""
    return {"SKILL.md": skill, "reference.md": render_reference_md(spec)}


def render_agents_md_snippet(spec: PlatformSpec) -> str:
    return f"""<!-- {spec.name.lower()}:start -->
## Deploying to {spec.name}

This app targets {spec.name} ({spec.tagline}). Rules: `llms.txt` (or {spec.docs_url}/llms.txt).

- The platform owns the port and router: export `handler` from the entrypoint, never call listen()/run().
- Forbidden dependencies: {', '.join(sorted(spec.forbidden_dependencies))}.
- Manifest `{spec.manifest_file}` needs {', '.join(spec.required_fields)}; runtime in {', '.join(spec.runtime_names())}.
- Secrets: {spec.secrets.get('rule', '')}
- Outbound allowlist: {', '.join(spec.outbound.get('allowed_domains', []))}.
- Precheck `{spec.deploy.get('precheck_command', '')}` then `{spec.deploy_command}` -> `{spec.url_pattern}`.
<!-- {spec.name.lower()}:end -->
"""


def generate_all(spec: PlatformSpec) -> dict[str, str]:
    files = {"llms.txt": render_llms_txt(spec), "AGENTS.md.snippet": render_agents_md_snippet(spec)}
    files.update(render_skill(spec))
    return files


# ---------------------------------------------------------------- validator


@dataclass
class Finding:
    code: str            # own_server | forbidden_dependency | manifest_missing | manifest_field | runtime | entrypoint | file_too_large | secret_file | outbound_domain | bundle
    severity: str        # blocking | warning
    file: str | None
    line: int | None
    message: str         # builder language: what is wrong and why the platform cares
    fix: str             # what to change, concrete enough for an agent to apply

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_URL = re.compile(r"https?://([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?::\d+)?")
_PORT = re.compile(r"(?:listen|run)\(\s*(?:host\s*=\s*['\"][^'\"]*['\"]\s*,\s*)?(?:port\s*=\s*)?(\d{2,5})")
_PORT_ENV = re.compile(r"process\.env\.PORT\s*\|\|\s*(\d{2,5})")


def _iter_files(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            out.append(p)
    return out


def _read(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _dep_names_package_json(text: str) -> dict[str, int]:
    """dependency name -> line number (1-based) in package.json."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    names = set()
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        names.update((data.get(key) or {}).keys())
    lines = text.splitlines()
    out = {}
    for n in names:
        out[n] = next((i for i, ln in enumerate(lines, 1) if f'"{n}"' in ln), 0)
    return out


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _dep_names_requirements(text: str) -> dict[str, int]:
    out = {}
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith(("#", "-")):
            continue
        m = _REQ_NAME.match(s)
        if m:
            out[m.group(1).lower()] = i
    return out


def _dep_names_pyproject(text: str) -> dict[str, int]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}
    deps = list((data.get("project") or {}).get("dependencies") or [])
    for extra in ((data.get("project") or {}).get("optional-dependencies") or {}).values():
        deps.extend(extra)
    poetry = ((data.get("tool") or {}).get("poetry") or {}).get("dependencies") or {}
    deps.extend(k for k in poetry if k != "python")
    lines = text.splitlines()
    out = {}
    for d in deps:
        m = _REQ_NAME.match(d)
        if m:
            name = m.group(1).lower()
            out[name] = next((i for i, ln in enumerate(lines, 1) if name in ln.lower()), 0)
    return out


class CompatValidator:
    """Deploy-time precheck. Every finding says what the platform saw, why it matters and what to
    change — in the builder's language, with the exact file and line where possible."""

    def __init__(self, spec: PlatformSpec):
        self.spec = spec

    def validate(self, artifact_dir: str | Path) -> list[Finding]:
        root = Path(artifact_dir)
        if not root.is_dir():
            return [Finding("bundle", "blocking", None, None, f"{root} is not a directory",
                            "point the validator at the unpacked app folder (the one holding the manifest)")]
        files = _iter_files(root)
        findings: list[Finding] = []
        manifest, runtime = self._check_manifest(root, findings)
        self._check_sizes(root, files, findings)
        self._check_secrets(root, files, findings)
        self._check_dependencies(root, findings)
        self._check_own_server(root, files, findings)
        self._check_entrypoint(root, manifest, runtime, findings)
        self._check_outbound(root, files, findings)
        order = {"blocking": 0, "warning": 1}
        findings.sort(key=lambda f: (order[f.severity], f.code, f.file or "", f.line or 0))
        return findings

    # ---- checks
    def _check_manifest(self, root: Path, out: list[Finding]) -> tuple[dict[str, Any], str | None]:
        spec = self.spec
        path = root / spec.manifest_file
        if not path.exists():
            out.append(Finding("manifest_missing", "blocking", spec.manifest_file, None,
                               f"no `{spec.manifest_file}` in the bundle; the platform cannot tell what to run",
                               f"create `{spec.manifest_file}` with {', '.join(spec.required_fields)}:\n{_example_manifest(spec)}"))
            return {}, None
        try:
            manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            out.append(Finding("manifest_missing", "blocking", spec.manifest_file, None,
                               f"`{spec.manifest_file}` is not valid YAML: {e}", "fix the YAML syntax; keys are plain `name: value` lines"))
            return {}, None
        if not isinstance(manifest, dict):
            manifest = {}
        for f in spec.required_fields:
            if f not in manifest or manifest[f] in (None, ""):
                out.append(Finding("manifest_field", "blocking", spec.manifest_file, None,
                                   f"`{spec.manifest_file}` is missing required field `{f}`",
                                   f"add `{f}: <{f}>` to `{spec.manifest_file}`" + (
                                       f" (e.g. `{f}: {spec.manifest.get('example', {}).get(f)}`)" if spec.manifest.get("example", {}).get(f) else "")))
        runtime = None
        rt_value = manifest.get("runtime")
        if rt_value:
            split = spec.split_runtime(str(rt_value))
            if split is None or (split[1] and split[1] not in spec.runtimes[split[0]]):
                out.append(Finding("runtime", "blocking", spec.manifest_file, None,
                                   f"runtime `{rt_value}` is not available on {spec.name}",
                                   f"set `runtime` to one of {', '.join(spec.runtime_names())}"))
            else:
                runtime = split[0]
        return manifest, runtime

    def _check_sizes(self, root: Path, files: list[Path], out: list[Finding]) -> None:
        max_file = int(self.spec.limits.get("max_file_bytes", 0) or 0)
        max_bundle = int(self.spec.limits.get("max_bundle_bytes", 0) or 0)
        max_files = int(self.spec.limits.get("max_files", 0) or 0)
        total = 0
        for p in files:
            size = p.stat().st_size
            total += size
            if max_file and size > max_file:
                out.append(Finding("file_too_large", "blocking", p.relative_to(root).as_posix(), None,
                                   f"{p.name} is {size} bytes; the per-file limit is {max_file} bytes",
                                   "move large media to object storage and reference it by URL, or compress it; bundles hold code and small assets"))
        if max_bundle and total > max_bundle:
            out.append(Finding("bundle", "blocking", None, None, f"bundle is {total} bytes; the limit is {max_bundle} bytes",
                               "exclude node_modules/, build caches and media from the bundle"))
        if max_files and len(files) > max_files:
            out.append(Finding("bundle", "blocking", None, None, f"bundle has {len(files)} files; the limit is {max_files}",
                               "exclude node_modules/ and generated folders from the bundle"))

    def _check_secrets(self, root: Path, files: list[Path], out: list[Finding]) -> None:
        pats = self.spec.secrets.get("forbidden_files", [])
        for p in files:
            rel = p.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(p.name, pat) for pat in pats):
                out.append(Finding("secret_file", "blocking", rel, None,
                                   f"`{rel}` looks like a secrets file committed into the bundle; deployed bundles are readable by the platform team",
                                   f"delete `{rel}` from the bundle; {self.spec.secrets.get('rule', '')}"))

    def _check_dependencies(self, root: Path, out: list[Finding]) -> None:
        forbidden = {k.lower(): v for k, v in self.spec.forbidden_dependencies.items()}
        sources = [("package.json", _dep_names_package_json), ("requirements.txt", _dep_names_requirements),
                   ("pyproject.toml", _dep_names_pyproject)]
        for fname, parser in sources:
            p = root / fname
            if not p.exists():
                continue
            text = _read(p) or ""
            for dep, line in parser(text).items():
                if dep.lower() in forbidden:
                    out.append(Finding("forbidden_dependency", "blocking", fname, line or None,
                                       f"`{dep}` is listed in {fname}; {forbidden[dep.lower()]}",
                                       f"remove `{dep}` from {fname} and move its routes into the exported `handler` (see llms.txt 'Entrypoint')"))

    def _check_own_server(self, root: Path, files: list[Path], out: list[Finding]) -> None:
        pats = self.spec.server_patterns
        for p in files:
            if p.suffix not in CODE_EXT:
                continue
            text = _read(p)
            if text is None:
                continue
            rel = p.relative_to(root).as_posix()
            for i, ln in enumerate(text.splitlines(), 1):
                s = ln.strip()
                if s.startswith(("//", "#")):
                    continue
                hit = next((pat for pat in pats if pat in ln), None)
                if not hit:
                    continue
                m = _PORT_ENV.search(ln) or _PORT.search(ln)
                port = f" on port {m.group(1)}" if m else ""
                out.append(Finding("own_server", "blocking", rel, i,
                                   f"this app starts its own server{port} (`{hit.strip()}` at {rel}:{i}); the platform already routes requests and nothing may listen on a port",
                                   f"delete the `{hit.strip()}...)` call at {rel}:{i} and export a handler instead: "
                                   + "; ".join(f"{rt}: `{e.get('signature')}`" for rt, e in self.spec.entrypoint.items())))
                break  # one finding per file is enough to explain the problem

    def _check_entrypoint(self, root: Path, manifest: dict[str, Any], runtime: str | None, out: list[Finding]) -> None:
        spec = self.spec
        if not manifest:
            return
        entry = manifest.get("entrypoint")
        if not entry:
            return  # already reported as a missing field
        p = root / str(entry)
        rt = runtime or ("python" if str(entry).endswith(".py") else "node")
        shape = spec.entrypoint.get(rt, {})
        export = shape.get("export", "handler")
        if not p.exists():
            out.append(Finding("entrypoint", "blocking", str(entry), None,
                               f"manifest entrypoint `{entry}` does not exist in the bundle",
                               f"create `{shape.get('file', entry)}` exporting `{export}` — `{shape.get('signature', '')}` — and point `entrypoint` at it"))
            return
        text = _read(p) or ""
        if rt == "python":
            ok = re.search(rf"^\s*(async\s+)?def\s+{export}\s*\(", text, re.MULTILINE) is not None
        else:
            ok = re.search(rf"export\s+(async\s+)?function\s+{export}\b|export\s+(const|let|var)\s+{export}\b"
                           rf"|exports\.{export}\s*=|module\.exports\s*=\s*\{{[^}}]*\b{export}\b|export\s*\{{[^}}]*\b{export}\b", text) is not None
        if not ok:
            out.append(Finding("entrypoint", "blocking", str(entry), None,
                               f"`{entry}` does not export `{export}`; the platform calls `{export}(request)` and nothing else",
                               f"add `{shape.get('signature', f'{export}(request)')}` to `{entry}` and export it"))

    def _check_outbound(self, root: Path, files: list[Path], out: list[Finding]) -> None:
        allowed = set(self.spec.outbound.get("allowed_domains", []))
        ignored = set(self.spec.outbound.get("ignored_domains", []))
        if not allowed:
            return
        seen: set[tuple[str, str]] = set()
        for p in files:
            if p.suffix not in SOURCE_EXT or p.name in ("package-lock.json",) or p.suffix == ".md":
                continue
            text = _read(p)
            if text is None:
                continue
            rel = p.relative_to(root).as_posix()
            for i, ln in enumerate(text.splitlines(), 1):
                for m in _URL.finditer(ln):
                    dom = m.group(1).lower()
                    if dom in allowed or dom in ignored or any(dom.endswith("." + a) for a in allowed | ignored):
                        continue
                    if (rel, dom) in seen:
                        continue
                    seen.add((rel, dom))
                    out.append(Finding("outbound_domain", "blocking", rel, i,
                                       f"`{dom}` ({rel}:{i}) is not on the egress allowlist; {self.spec.outbound.get('rule', '')}",
                                       f"either call an allowlisted domain ({', '.join(sorted(allowed))}) or request `{dom}` be added before deploying"))
        return


def render_findings(findings: list[Finding], spec: PlatformSpec, artifact: str) -> str:
    from evalkit.report import md_table
    blocking = [f for f in findings if f.blocking]
    head = [f"## {spec.deploy.get('precheck_command', 'validate')} — `{artifact}`", "",
            f"**{'deployable' if not blocking else 'not deployable'}**: {len(blocking)} blocking, {len(findings) - len(blocking)} warnings", ""]
    if not findings:
        head.append(f"No findings. Next: `{spec.deploy_command}` -> `{spec.url_pattern}`")
        return "\n".join(head)
    rows = [[f.severity, f.code, (f"{f.file}:{f.line}" if f.line else f.file or "-"), f.message.replace("|", "\\|")] for f in findings]
    head.append(md_table(["severity", "code", "where", "what the platform saw"], rows))
    head += ["", "### Fixes", ""]
    for i, f in enumerate(findings, 1):
        head.append(f"{i}. **{f.code}** ({f.file or 'bundle'}{':' + str(f.line) if f.line else ''}): {f.fix}")
    return "\n".join(head)


# ---------------------------------------------------------------- first-run study

STEPS = ["open_studio", "build_app", "make_zip", "upload_zip", "fix_precheck", "deploy", "open_url"]
ENGINEER_ROLES = ("engineer", "developer", "swe", "sre", "devops", "programmer")
_STOP_WORDS = {"the", "a", "an", "to", "of", "and", "in", "on", "for", "it", "is", "my", "i", "was", "do", "did",
               "not", "no", "how", "what", "where", "with", "this", "that", "me", "get", "got", "from", "at",
               "be", "so", "but", "could", "couldn", "didn", "t", "s", "then", "had", "have", "has", "after",
               "or", "by", "up", "out", "its", "into", "which", "one", "just"}
THEMES = {  # keyword -> theme label; first match wins, so order matters
    "zip": "zip", "archive": "zip", "compress": "zip", "bundle": "zip",
    "express": "own server", "server": "own server", "listen": "own server", "port": "own server", "localhost": "own server",
    "manifest": "manifest", "gizmo.yaml": "manifest", "yaml": "manifest", "field": "manifest", "owner": "manifest",
    "secret": "secrets", "env": "secrets", "key": "secrets", "token": "secrets",
    "runtime": "runtime", "version": "runtime", "node": "runtime", "python": "runtime",
    "url": "url", "subdomain": "url", "404": "url", "domain": "url",
    "cli": "cli", "install": "cli", "command": "cli", "terminal": "cli",
}


@dataclass
class FirstRun:
    participant: str
    role: str
    step_reached: str
    finished: bool
    unaided: bool
    minutes: float
    blocker: str = ""
    tool: str = ""

    @property
    def engineer(self) -> bool:
        r = self.role.lower()
        return any(k in r for k in ENGINEER_ROLES)


@dataclass
class FirstRunSummary:
    n: int
    stuck_by_step: dict[str, int]          # step -> participants who stopped there (not finished)
    top_stuck_step: str | None
    finished: int
    finished_unaided_non_engineers: int
    median_minutes_to_deploy: float | None  # over finished participants
    blocker_themes: dict[str, list[str]]    # theme -> participants
    ready_to_call_easy: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_first_runs(path: str | Path) -> list[FirstRun]:
    rows = read_jsonl(path)
    return [FirstRun(r["participant"], r["role"], r["step_reached"], bool(r.get("finished", False)),
                     bool(r.get("unaided", False)), float(r.get("minutes", 0)), r.get("blocker", "") or "",
                     r.get("tool", "") or "") for r in rows]


def _keywords(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9.]+", text.lower())
    return [t for t in toks if t not in _STOP_WORDS and len(t) > 1]


def cluster_blockers(runs: list[FirstRun]) -> dict[str, list[str]]:
    """Keyword clustering: a blocker joins the first theme whose keyword it contains; blockers with
    no theme keyword are grouped by their most frequent non-stop word across all blockers."""
    themes: dict[str, list[str]] = defaultdict(list)
    freq = Counter(k for r in runs if r.blocker for k in set(_keywords(r.blocker)))
    for r in runs:
        if not r.blocker:
            continue
        kws = _keywords(r.blocker)
        label = next((THEMES[k] for k in kws if k in THEMES), None)
        if label is None:
            label = max(kws, key=lambda k: (freq[k], -kws.index(k)), default="other")
        themes[label].append(r.participant)
    return dict(sorted(themes.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def summarise_first_runs(runs: list[FirstRun], min_unaided_non_engineers: int = 3) -> FirstRunSummary:
    """`ready_to_call_easy` only when >= 3 non-engineers finished unaided: "watch someone try it
    before calling it easy". Median time to first deploy is over finished participants only."""
    stuck = Counter(r.step_reached for r in runs if not r.finished)
    stuck_by_step = {s: stuck.get(s, 0) for s in STEPS if stuck.get(s, 0)} | {s: n for s, n in stuck.items() if s not in STEPS}
    finished = [r for r in runs if r.finished]
    good = [r for r in finished if r.unaided and not r.engineer]
    top = max(stuck_by_step, key=lambda s: (stuck_by_step[s], -STEPS.index(s) if s in STEPS else 0)) if stuck_by_step else None
    median = statistics.median(r.minutes for r in finished) if finished else None
    ready = len(good) >= min_unaided_non_engineers
    if ready:
        verdict = (f"{len(good)} non-engineers finished unaided (median {median:.0f} min): you may call it easy"
                   if median is not None else "ready")
    else:
        where = f"; {stuck_by_step[top]} of {len(runs)} stopped at `{top}`" if top else ""
        verdict = (f"only {len(good)} non-engineer(s) finished unaided (need {min_unaided_non_engineers}){where}: "
                   "not easy yet — fix the top blocker and watch three more people try it")
    return FirstRunSummary(len(runs), stuck_by_step, top, len(finished), len(good), median,
                           cluster_blockers(runs), ready, verdict)


def render_first_runs(s: FirstRunSummary) -> str:
    from evalkit.report import md_table
    out = ["## first-run study", "",
           md_table(["participants", "finished", "non-engineers unaided", "median min to deploy", "ready_to_call_easy"],
                    [[s.n, s.finished, s.finished_unaided_non_engineers,
                      f"{s.median_minutes_to_deploy:.0f}" if s.median_minutes_to_deploy is not None else "-",
                      "true" if s.ready_to_call_easy else "false"]]), ""]
    if s.stuck_by_step:
        out += ["### Where people stop", "", md_table(["step", "stuck"], [[k, v] for k, v in s.stuck_by_step.items()]), ""]
    if s.blocker_themes:
        out += ["### Blocker themes", "", md_table(["theme", "n", "participants"], [[k, len(v), ", ".join(v)] for k, v in s.blocker_themes.items()]), ""]
    out.append(f"**verdict:** {s.verdict}")
    return "\n".join(out)
