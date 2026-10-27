"""The teardowns cite real paths and symbols; this test fails when a citation rots."""
import importlib
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DOCS = sorted(HERE.glob("0*_*.md"))

PATH_RX = re.compile(r"`((?:projects|evalkit)/[A-Za-z0-9_./-]+)(?::([A-Za-z_][A-Za-z0-9_.]*))?`")
MOD_RX = re.compile(r"`((?:projects|evalkit)(?:\.[a-z0-9_]+)+):([A-Za-z_][A-Za-z0-9_]*)`")


def _cites():
    for doc in DOCS:
        text = doc.read_text()
        for m in PATH_RX.finditer(text):
            yield doc.name, "path", m.group(1), m.group(2)
        for m in MOD_RX.finditer(text):
            yield doc.name, "module", m.group(1), m.group(2)


CITES = list(_cites())


def test_three_teardowns_exist():
    assert len(DOCS) == 3
    for d in DOCS:
        assert len(d.read_text().split()) > 600, f"{d.name} is too short to be a deep-dive"


@pytest.mark.parametrize("doc,kind,target,symbol", CITES, ids=[f"{c[0]}:{c[2]}" for c in CITES])
def test_citation_resolves(doc, kind, target, symbol):
    if kind == "path":
        path = ROOT / target
        assert path.exists(), f"{doc} cites missing path {target}"
        if symbol and path.suffix == ".py":
            assert re.search(rf"^\s*(def|class)\s+{re.escape(symbol.split('.')[0])}\b|^{re.escape(symbol)}\s*[:=]",
                             path.read_text(), re.MULTILINE), f"{doc}: {symbol} not defined in {target}"
    else:
        mod = importlib.import_module(target)
        assert hasattr(mod, symbol), f"{doc} cites {target}:{symbol} which does not exist"


def test_headline_format_on_cited_data():
    """Teardown 03 quotes the headline format; check it on the real cited files."""
    from evalkit.io import read_jsonl
    from evalkit.stats import paired_bootstrap
    p07 = ROOT / "projects/p07_significance_engine/data"
    a = {r["id"]: r for r in read_jsonl(p07 / "model_a.jsonl")}
    b = {r["id"]: r for r in read_jsonl(p07 / "model_b.jsonl")}
    ids = sorted(set(a) & set(b))
    key = "success" if "success" in a[ids[0]] else "score"
    rep = paired_bootstrap([float(a[i][key]) for i in ids], [float(b[i][key]) for i in ids],
                           metric="task_success", n_boot=4000, seed=0)
    assert re.fullmatch(r"Model B wins on task_success by \d+\.\d% ± \d+\.\d% \(p<0\.05, n=\d+\)", rep.headline())
