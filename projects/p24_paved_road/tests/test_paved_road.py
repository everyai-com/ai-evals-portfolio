"""Generators, validator, first-run study, CLI."""
from __future__ import annotations

import json
from pathlib import Path

from evalkit.io import write_jsonl
from projects.p24_paved_road.cli import main
from projects.p24_paved_road.core import (
    CompatValidator,
    FirstRun,
    PlatformSpec,
    cluster_blockers,
    generate_all,
    load_first_runs,
    render_agents_md_snippet,
    render_findings,
    render_llms_txt,
    render_skill,
    summarise_first_runs,
)

DATA = Path(__file__).resolve().parent.parent / "data"
SPEC = PlatformSpec.load(DATA / "platform.yaml")


# ---------------------------------------------------------------- generators

def test_llms_txt_mentions_every_forbidden_dep_and_the_deploy_command():
    txt = render_llms_txt(SPEC)
    for dep, why in SPEC.forbidden_dependencies.items():
        assert f"`{dep}`" in txt and why in txt
    assert SPEC.deploy_command in txt and SPEC.deploy["precheck_command"] in txt
    assert SPEC.url_pattern in txt and SPEC.manifest_file in txt
    for f in SPEC.required_fields:
        assert f in txt
    for rt in SPEC.runtime_names():
        assert rt in txt or rt.split(".")[0] in txt
    assert "## Checklist" in txt and "## What fails" in txt and "export async function handler" in txt


def test_skill_files_and_agents_snippet():
    files = render_skill(SPEC)
    assert set(files) == {"SKILL.md", "reference.md"}
    skill = files["SKILL.md"]
    assert skill.startswith("---\nname: gizmos-app\ndescription:") and "\n---\n" in skill
    assert "reference.md" in skill and SPEC.deploy_command in skill
    ref = files["reference.md"]
    assert all(f"`{d}`" in ref for d in SPEC.forbidden_dependencies) and "app.listen(" in ref
    snippet = render_agents_md_snippet(SPEC)
    assert snippet.startswith("<!-- gizmos:start -->") and snippet.rstrip().endswith("<!-- gizmos:end -->")
    assert "express" in snippet and SPEC.deploy_command in snippet


def test_regeneration_is_idempotent(tmp_path):
    a = generate_all(SPEC)
    b = generate_all(PlatformSpec.load(DATA / "platform.yaml"))
    assert a == b
    assert main(["generate", "--spec", str(DATA / "platform.yaml"), "--out", str(tmp_path)]) == 0
    first = {p.name: p.read_text() for p in tmp_path.iterdir()}
    assert main(["generate", "--spec", str(DATA / "platform.yaml"), "--out", str(tmp_path)]) == 0
    second = {p.name: p.read_text() for p in tmp_path.iterdir()}
    assert first == second and set(first) == {"llms.txt", "SKILL.md", "reference.md", "AGENTS.md.snippet"}


# ---------------------------------------------------------------- validator

def test_validator_flags_own_server_with_file_line_and_builder_fix():
    findings = CompatValidator(SPEC).validate(DATA / "sample_app_express")
    own = [f for f in findings if f.code == "own_server"]
    assert len(own) == 1
    f = own[0]
    assert f.blocking and f.file == "server.js" and f.line == 16
    assert "starts its own server on port 3000" in f.message and "the platform already routes requests" in f.message
    assert "export a handler" in f.fix and "server.js:16" in f.fix and "handler(request)" in f.fix


def test_validator_flags_forbidden_dep_from_package_json():
    findings = CompatValidator(SPEC).validate(DATA / "sample_app_express")
    dep = [f for f in findings if f.code == "forbidden_dependency"]
    assert [d.file for d in dep] == ["package.json"] and dep[0].line == 7
    assert "`express`" in dep[0].message and "export a handler" in dep[0].message and "remove `express`" in dep[0].fix
    assert not any("cors" in f.message for f in dep)


def test_express_sample_has_the_full_set_of_findings_and_exit_1(tmp_path, capsys):
    findings = CompatValidator(SPEC).validate(DATA / "sample_app_express")
    codes = sorted(f.code for f in findings)
    assert codes == ["entrypoint", "forbidden_dependency", "manifest_field", "outbound_domain", "own_server", "secret_file"]
    assert all(f.blocking for f in findings)
    out = tmp_path / "v.json"
    assert main(["validate", "--dir", str(DATA / "sample_app_express"), "--json", str(out)]) == 1
    text = capsys.readouterr().out
    assert "not deployable" in text and "### Fixes" in text
    assert json.loads(out.read_text())["deployable"] is False


def test_ok_app_passes_and_exit_0(capsys):
    assert CompatValidator(SPEC).validate(DATA / "sample_app_ok") == []
    assert main(["validate", "--dir", str(DATA / "sample_app_ok")]) == 0
    assert "deployable" in capsys.readouterr().out
    assert "No findings" in render_findings([], SPEC, "x")


def test_manifest_field_missing_and_missing_manifest(tmp_path):
    findings = CompatValidator(SPEC).validate(DATA / "sample_app_express")
    mf = [f for f in findings if f.code == "manifest_field"]
    assert len(mf) == 1 and "`owner`" in mf[0].message and "owner: jane.doe@example.com" in mf[0].fix
    (tmp_path / "handler.js").write_text("export async function handler(r) { return new Response('hi'); }")
    f = CompatValidator(SPEC).validate(tmp_path)
    assert [x.code for x in f] == ["manifest_missing"] and "gizmo.yaml" in f[0].fix
    assert CompatValidator(SPEC).validate(tmp_path / "nope")[0].code == "bundle"


def test_outbound_domain_outside_allowlist_and_ignored_domains(tmp_path):
    findings = CompatValidator(SPEC).validate(DATA / "sample_app_express")
    ob = [f for f in findings if f.code == "outbound_domain"]
    assert len(ob) == 1 and ob[0].file == "server.js" and ob[0].line == 11 and "api.randomweather.io" in ob[0].message
    assert "api.openai.com" in ob[0].fix
    (tmp_path / "gizmo.yaml").write_text("name: x\nruntime: python3.12\nentrypoint: handler.py\nowner: o@example.com\n")
    (tmp_path / "handler.py").write_text(
        "SVG = 'http://www.w3.org/2000/svg'\nAPI = 'https://api.openai.com/v1'\nSUB = 'https://eu.api.internal.example.com/x'\n"
        "def handler(request):\n    return {'status': 200}\n")
    assert CompatValidator(SPEC).validate(tmp_path) == []


def test_python_app_own_server_runtime_and_entrypoint_checks(tmp_path):
    (tmp_path / "gizmo.yaml").write_text("name: x\nruntime: python3.9\nentrypoint: app.py\nowner: o@example.com\n")
    (tmp_path / "requirements.txt").write_text("Flask==3.0.0\nrequests>=2\n# gunicorn\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["uvicorn>=0.30", "httpx"]\n')
    (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n\n# app.run(port=9)\nif __name__ == '__main__':\n    app.run(host='0.0.0.0', port=5000)\n")
    (tmp_path / "id_rsa").write_text("not a key\n")
    f = CompatValidator(SPEC).validate(tmp_path)
    by = {}
    for x in f:
        by.setdefault(x.code, []).append(x)
    assert "python3.9" in by["runtime"][0].message and "python3.11, python3.12" in by["runtime"][0].fix
    assert sorted(d.message.split("`")[1] for d in by["forbidden_dependency"]) == ["flask", "uvicorn"]
    assert by["own_server"][0].line == 6 and "port 5000" in by["own_server"][0].message
    assert "does not export `handler`" in by["entrypoint"][0].message and "def handler(request)" in by["entrypoint"][0].fix
    assert by["secret_file"][0].file == "id_rsa"


def test_file_too_large(tmp_path):
    small = PlatformSpec.from_dict({**SPEC.raw, "limits": {"max_file_bytes": 10, "max_bundle_bytes": 100, "max_files": 1}})
    (tmp_path / "gizmo.yaml").write_text("name: x\nruntime: node20\nentrypoint: handler.js\nowner: o@example.com\n")
    (tmp_path / "handler.js").write_text("export const handler = async (r) => new Response('x'.repeat(100));\n")
    codes = [f.code for f in CompatValidator(small).validate(tmp_path)]
    assert codes.count("file_too_large") == 2 and codes.count("bundle") == 2


# ---------------------------------------------------------------- first-run study

def test_first_run_summary_finds_the_zip_blocker_and_is_not_ready():
    runs = load_first_runs(DATA / "first_runs.jsonl")
    s = summarise_first_runs(runs)
    assert s.n == 8 and s.top_stuck_step == "make_zip" and s.stuck_by_step["make_zip"] == 5
    assert s.finished == 2 and s.finished_unaided_non_engineers == 1
    assert s.median_minutes_to_deploy == 20.0
    assert next(iter(s.blocker_themes)) == "zip" and len(s.blocker_themes["zip"]) == 5
    assert "own server" in s.blocker_themes and s.blocker_themes["own server"] == ["p05"]
    assert s.ready_to_call_easy is False and "make_zip" in s.verdict and "not easy yet" in s.verdict


def test_first_run_ready_when_three_non_engineers_finish_unaided(tmp_path):
    rows = [
        {"participant": "a", "role": "designer", "step_reached": "open_url", "finished": True, "unaided": True, "minutes": 12},
        {"participant": "b", "role": "product manager", "step_reached": "open_url", "finished": True, "unaided": True, "minutes": 20},
        {"participant": "c", "role": "recruiter", "step_reached": "open_url", "finished": True, "unaided": True, "minutes": 31},
        {"participant": "d", "role": "software engineer", "step_reached": "open_url", "finished": True, "unaided": True, "minutes": 5},
        {"participant": "e", "role": "analyst", "step_reached": "open_url", "finished": True, "unaided": False, "minutes": 9, "blocker": "needed help with the manifest owner field"},
    ]
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, rows)
    s = summarise_first_runs(load_first_runs(path))
    assert s.finished_unaided_non_engineers == 3 and s.ready_to_call_easy is True
    assert s.median_minutes_to_deploy == 12.0 and s.stuck_by_step == {} and s.top_stuck_step is None
    assert s.blocker_themes == {"manifest": ["e"]}
    assert main(["first-runs", "--log", str(path)]) == 0
    # engineers and aided finishes do not count
    s2 = summarise_first_runs([FirstRun("x", "SRE", "open_url", True, True, 3)] * 5)
    assert s2.ready_to_call_easy is False
    s3 = summarise_first_runs([FirstRun(str(i), "pm", "open_url", True, False, 3) for i in range(5)])
    assert s3.ready_to_call_easy is False


def test_blocker_clustering_falls_back_to_frequent_word():
    runs = [FirstRun("1", "pm", "build_app", False, True, 5, "assistant produced broken code"),
            FirstRun("2", "pm", "build_app", False, True, 5, "the assistant wrote wrong code"),
            FirstRun("3", "pm", "deploy", False, True, 5, "")]
    themes = cluster_blockers(runs)
    assert list(themes) in (["assistant"], ["code"]) and themes[next(iter(themes))] == ["1", "2"]


def test_cli_first_runs_exit_codes_and_json(tmp_path, capsys):
    out = tmp_path / "fr.json"
    assert main(["first-runs", "--log", str(DATA / "first_runs.jsonl"), "--json", str(out)]) == 1
    assert "make_zip" in capsys.readouterr().out
    assert json.loads(out.read_text())["ready_to_call_easy"] is False
    assert main(["first-runs", "--log", str(tmp_path / "missing.jsonl")]) == 2
    assert main(["nope"]) == 2
