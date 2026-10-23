"""Tests prove the README claims: every planted tier is caught by the right scanner, clean rows stay clean."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from evalkit.io import read_jsonl
from projects.p14_contamination_checker import core
from projects.p14_contamination_checker.cli import main

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def corpus():
    return read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "eval.jsonl")


@pytest.fixture(scope="module")
def report(corpus):
    train, ev = corpus
    return core.scan(ev, train)


def _by_planted(report, ev):
    out = {}
    for item, row in zip(report.items, ev):
        out.setdefault(row["planted"], []).append(item)
    return out


def test_normalisation_and_hashing():
    assert core.normalize("  Hello,  WORLD!! it's 9pm. ") == "hello world it s 9pm"
    assert core.ngram_hashes(["a", "b", "c"], 2) == [core.hash64("a b"), core.hash64("b c")]
    assert core.ngram_hashes(["a"], 2) == []
    assert core.hash64("x") < 2**64


def test_all_planted_exact_leaks_flagged_exact(report, corpus):
    train, ev = corpus
    exact = _by_planted(report, ev)["exact"]
    assert len(exact) == 10 and all(i.tier == "exact" for i in exact)
    assert all(i.exact_match_id and i.exact_match_id.startswith("train-") for i in exact)
    # re-cased / re-punctuated copies are still exact after normalisation
    recased = [r for r in ev[:10] if r["text"] != r["text"].lower() and "!" in r["text"]]
    assert recased, "generator should have planted re-cased copies"


def test_paraphrases_caught_by_embedding_not_13gram(report, corpus):
    train, ev = corpus
    para = _by_planted(report, ev)["paraphrase"]
    assert len(para) == 10
    assert all(i.cosine >= 0.9 for i in para), [i.cosine for i in para]
    assert not any(i.exact_ngram_hit for i in para)
    assert all(i.tier == "semantic_match" for i in para)
    assert all(i.nearest_train_id and i.nearest_train_id.startswith("train-") for i in para)
    # n-gram-only scan would NOT catch them as near-duplicates
    ng = core.scan(ev, train, use_embeddings=False, use_minhash=False)
    assert all(i.tier in ("clean", "partial") for i in _by_planted(ng, ev)["paraphrase"])


def test_partial_spans_caught_by_ngram_tier(report, corpus):
    train, ev = corpus
    part = _by_planted(report, ev)["partial"]
    assert len(part) == 10 and all(i.tier == "partial" for i in part)
    assert all(i.exact_ngram_hit and i.longest_span >= 13 for i in part)
    assert all(i.cosine < 0.9 for i in part)
    assert all("[[" in i.evidence and "]]" in i.evidence for i in part)
    ng = core.scan(ev, train, use_embeddings=False, use_minhash=False)
    assert all(i.tier == "partial" for i in _by_planted(ng, ev)["partial"])


def test_clean_rows_false_positive_rate_below_2_percent(report, corpus):
    train, ev = corpus
    clean = _by_planted(report, ev)[None]
    fpr = sum(i.tier != "clean" for i in clean) / len(clean)
    assert len(clean) == 170 and fpr < 0.02
    assert max(i.cosine for i in clean) < 0.9 and max(i.ngram_overlap for i in clean) < 0.3


def test_report_shape(report):
    assert report.rate == pytest.approx(30 / 200)
    assert report.by_tier() == {
        "exact": 10,
        "near_duplicate": 0,
        "semantic_match": 10,
        "partial": 10,
        "clean": 170,
    }
    bs = report.by_strata()
    assert (
        set(bs) == {"qa", "code", "math", "summarize"}
        and sum(v["contaminated"] for v in bs.values()) == 30
    )
    ev = report.evidence()
    assert len(ev) == 30 and all(t.startswith("train-") for _, t, _ in ev)
    md = report.to_markdown()
    assert "Contamination rate: 15.0%" in md and "[[" in md
    d = report.to_dict()
    assert d["by_tier"]["exact"] == 10 and len(d["items"]) == 200


def test_near_duplicate_tier_and_index_details():
    train = [
        {
            "id": "t1",
            "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi",
        },
        {"id": "t2", "text": "one two three four five six seven eight nine ten"},
    ]
    idx = core.NgramIndex(n=4, exact_n=6).build(train)
    h = idx.lookup("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu omega")
    assert (
        h.overlap >= 0.8 and h.longest_span == 13 and h.span_train_id == "t1" and h.exact_ngram_hit
    )
    assert (
        core.classify(h, None, {"near_duplicate": 0.8, "partial": 0.3, "cosine": 0.9})
        == "near_duplicate"
    )
    assert idx.lookup("completely unrelated words here").overlap == 0.0
    assert idx.lookup("One, two; THREE four five six seven eight nine ten!").exact_match_id == "t2"
    assert idx.n_rows == 2 and idx.memory_bytes > 0 and idx.n_tokens == 24
    same = core.NgramIndex(n=4, exact_n=4).build(train)
    assert same.lookup("one two three four").exact_ngram_hit


def test_minhash_jaccard_estimate_within_0_1_of_truth():
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(400)]
    base = [words[i] for i in rng.permutation(400)[:120]]
    edited = base[:90] + [f"x{i}" for i in range(30)]  # replace a quarter of the shingles
    a, b = " ".join(base), " ".join(edited)
    true_j = core.jaccard(core.shingles(a), core.shingles(b))
    lsh = core.MinHashLSH(num_perm=128, seed=1)
    est = lsh.estimate_jaccard(lsh.signature(a), lsh.signature(b))
    assert abs(est - true_j) < 0.1, (est, true_j)
    assert lsh.estimate_jaccard(lsh.signature(a), lsh.signature(a)) == 1.0
    assert (
        lsh.estimate_jaccard(lsh.signature(a), lsh.signature("totally different content entirely"))
        < 0.1
    )


def test_lsh_recall_on_planted_near_dups(corpus):
    train, ev = corpus
    lsh = core.MinHashLSH().build(train)
    planted = [r for r in ev if r["planted"] in ("exact", "paraphrase")]
    src = {r["id"]: r for r in train}
    hits = 0
    for r in planted:
        cands = lsh.query(r["text"], top_k=3)
        if not cands:
            continue
        tid, est = cands[0]
        true_j = core.jaccard(core.shingles(r["text"]), core.shingles(src[tid]["text"]))
        hits += est >= 0.3 and abs(est - true_j) < 0.1
    assert hits / len(planted) >= 0.9
    clean_hits = sum(
        1 for r in ev if r["planted"] is None for c in lsh.query(r["text"], top_k=1) if c[1] >= 0.5
    )
    assert clean_hits == 0


def test_embedding_scanner_top_k_and_backend():
    train = [
        {"id": f"t{i}", "text": t}
        for i, t in enumerate(
            [
                "the quick brown fox jumps over the lazy dog",
                "a slow green turtle crawls under the busy road",
                "the quick brown fox leaps over the lazy dog",
                "completely unrelated sentence about compilers",
            ]
        )
    ]
    sc = core.EmbeddingScanner(core.HashedCharNgramBackend(dim=1024), threshold=0.9, chunk=2).fit(
        train
    )
    res = sc.scan(["the quick brown fox jumps over the lazy dog"], top_k=2)[0]
    assert res[0] == ("t0", pytest.approx(1.0, abs=1e-5)) and res[1][0] == "t2" and res[1][1] > 0.8
    assert core.get_backend("hashed").name == "hashed-char-ngram-tfidf"
    assert core.get_backend("st").name in ("hashed-char-ngram-tfidf",) or core.get_backend(
        "st"
    ).name.startswith("sentence-transformers")
    with pytest.raises(ValueError):
        core.get_backend("nope")


def test_clean_split_row_counts(report, corpus, tmp_path):
    train, ev = corpus
    out = core.clean_split(report, ev, tmp_path / "clean.jsonl", tmp_path / "removed.jsonl")
    assert out == {
        "kept": 170,
        "removed": 30,
        "clean_path": str(tmp_path / "clean.jsonl"),
        "removed_path": str(tmp_path / "removed.jsonl"),
    }
    clean = read_jsonl(tmp_path / "clean.jsonl")
    removed = read_jsonl(tmp_path / "removed.jsonl")
    assert len(clean) == 170 and all(r["planted"] is None for r in clean)
    assert Counter(r["contamination_tier"] for r in removed) == {
        "exact": 10,
        "semantic_match": 10,
        "partial": 10,
    }
    only_exact = core.clean_split(report, ev, tmp_path / "c2.jsonl", remove_tiers=("exact",))
    assert only_exact["kept"] == 190 and only_exact["removed"] == 10


def test_contamination_aware_metrics(report):
    results = read_jsonl(DATA / "results.jsonl")
    note = core.contamination_aware_metrics(report, results)
    assert note["metric"] == "success" and note["n_full"] == 200 and note["n_clean"] == 170
    assert note["contaminated"] > note["clean"] and note["full"] > note["clean"]
    assert "Contamination-aware reporting" in report.to_markdown()


def test_cli_scan_and_fingerprint(tmp_path):
    rc = main(
        [
            "scan",
            "--eval",
            str(DATA / "eval.jsonl"),
            "--train",
            str(DATA / "train.jsonl"),
            "--results",
            str(DATA / "results.jsonl"),
            "--clean-out",
            str(tmp_path / "clean.jsonl"),
            "--json",
            str(tmp_path / "r.json"),
        ]
    )
    assert (
        rc == 1
        and len(read_jsonl(tmp_path / "clean.jsonl")) == 170
        and (tmp_path / "r.json").exists()
    )
    assert main(["scan", "--max-rate", "0.2", "--no-embed", "--no-minhash"]) == 0
    assert main(["scan", "--field", "nope"]) == 2
    assert main(["fingerprint", "--dataset", str(DATA / "eval.jsonl")]) == 0
    assert main(["fingerprint", "--dataset", str(tmp_path / "missing.jsonl")]) == 2
    assert main(["bogus"]) == 2
