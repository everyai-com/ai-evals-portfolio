"""FeedbackStore (dedup, PII, quality filter), PairBuilder strategies, split leakage, golden set."""
from __future__ import annotations

from pathlib import Path

from evalkit.io import read_jsonl
from evalkit.llm import MockLLM
from projects.p06_dpo_flywheel.core import (
    EMAIL_RX,
    LONG_DIGITS_RX,
    PHONE_RX,
    FeedbackStore,
    FilterConfig,
    PairBuilder,
    append_golden,
    contains_pii,
    heuristic_judge,
    scrub_pii,
    split_pairs,
)

DATA = Path(__file__).resolve().parent.parent / "data"
EVENTS = read_jsonl(DATA / "feedback.jsonl")


def ev(i, prompt, response, rating="down", conv=None, **kw):
    return {"event_id": f"e{i}", "ts": f"2026-10-01T00:{i:02d}:00Z", "conversation_id": conv or f"c{i}",
            "user_id": f"u{i}", "prompt": prompt, "response": response, "rating": rating, "tags": ["t"],
            "tenant": "acme", **kw}


def test_dedup_by_event_id(tmp_path):
    store = FeedbackStore(tmp_path)
    s1 = store.ingest(EVENTS)
    assert s1.duplicates == 20 and s1.accepted + sum(s1.rejected.values()) == 400
    s2 = store.ingest(EVENTS)                       # replay the same file: everything is a duplicate
    assert s2.duplicates == 420 and s2.accepted == 0
    assert len(store.events()) == s1.accepted
    assert FeedbackStore(tmp_path).watermark == s1.watermark   # persisted


def test_pii_is_scrubbed(tmp_path):
    raw = "Mail me at jane.doe+x@example.co.uk or call (415) 555-2671, card 4111111111111111, order 40012"
    clean, counts = scrub_pii(raw)
    assert counts == {"email": 1, "phone": 1, "number": 1}
    assert "<EMAIL>" in clean and "<PHONE>" in clean and "<NUMBER>" in clean and "order 40012" in clean
    assert not contains_pii(clean)
    store = FeedbackStore(tmp_path)
    stats = store.ingest(EVENTS)
    assert stats.pii_scrubbed["email"] > 0 and stats.pii_scrubbed["phone"] > 0
    assert any(contains_pii(e["prompt"]) for e in EVENTS)
    for e in store.events():
        for f in ("prompt", "response", "correction"):
            txt = e.get(f) or ""
            assert not EMAIL_RX.search(txt) and not PHONE_RX.search(txt) and not LONG_DIGITS_RX.search(txt)


def test_quality_filter_and_user_cap(tmp_path):
    cfg = FilterConfig(max_per_user=2)
    store = FeedbackStore(tmp_path, cfg)
    events = [ev(1, "hi", "x"), ev(2, "Where is my order 40001?", ""), ev(3, "Buy followers now click here", "ok"),
              ev(4, "Where is my order 40001?", "soon", rating="meh")]
    events += [{**ev(10 + k, "Where is my order 40001? waiting", "soon"), "user_id": "heavy"} for k in range(4)]
    st = store.ingest(events)
    assert st.rejected == {"short_prompt": 1, "no_response": 1, "spam": 1, "bad_rating": 1, "user_cap": 2}
    assert st.accepted == 2


def test_correction_strategy_wins_over_thumbs_up():
    down = ev(1, "How do I reset my password?", "Go to settings.", correction="Tap Account > Security > Reset password.")
    up = ev(2, "How do I reset my password?", "Open Account, then Security, then Reset password link.", rating="up")
    pairs = PairBuilder(client=None).build([down, up])
    assert len(pairs) == 1 and pairs[0]["source"] == "correction"
    assert pairs[0]["chosen"] == down["correction"] and pairs[0]["rejected"] == down["response"]
    assert pairs[0]["confidence"] == 0.9


def test_thumbs_up_pairing_by_near_identical_prompt():
    down = ev(1, "Where is my order 40001? It's been 5 days.", "It will arrive soon.")
    up = ev(2, "where is my order 40001 its been 5 days", "Order 40001 is with the courier; tracking updates every 15 minutes.", rating="up")
    unrelated = ev(3, "Can I change the sauce on the Margherita?", "Yes, tap Customise.", rating="up")
    pairs = PairBuilder(client=None).build([down, up, unrelated])
    assert len(pairs) == 1 and pairs[0]["source"] == "thumbs_up" and pairs[0]["chosen"] == up["response"]
    # not similar enough -> no pair without a client
    down2 = ev(4, "Why was I charged twice for order 40002?", "Please contact support.")
    assert PairBuilder(client=None).build([down2, up, unrelated]) == []


def test_regenerated_candidate_rejected_when_judge_says_worse():
    down = ev(1, "Why was I charged twice for order 40002?", "The second line is a temporary authorisation hold that your bank releases in 3-5 days.")
    bad_client = MockLLM(handlers=[(r".", "I can't help with that.")])
    b = PairBuilder(client=bad_client)
    assert b.build([down]) == [] and b.stats["regen_rejected"] == 1 and len(bad_client.calls) == 1
    good_client = MockLLM(handlers=[(r".", ("Sorry about the duplicate charge on order 40002: one line is a pending "
                                           "authorisation hold that your bank releases within 3-5 business days, so you are billed once."))])
    pairs = PairBuilder(client=good_client).build([down])
    assert len(pairs) == 1 and pairs[0]["source"] == "regenerated" and 0 < pairs[0]["confidence"] <= 0.6
    assert heuristic_judge(down["prompt"], pairs[0]["chosen"]) > heuristic_judge(down["prompt"], down["response"])


def test_split_has_no_conversation_leakage(tmp_path):
    store = FeedbackStore(tmp_path)
    store.ingest(EVENTS)
    pairs = PairBuilder(client=None).build(store.events())
    assert len(pairs) >= 60
    train, val = split_pairs(pairs, val_frac=0.25, seed=1)
    assert len(train) + len(val) == len(pairs) and val
    assert not ({p["conversation_id"] for p in train} & {p["conversation_id"] for p in val})
    assert split_pairs(pairs, 0.25, 1) == (train, val)          # deterministic


def test_golden_additions_append_and_changelog(tmp_path):
    down = ev(1, "How do I reset my password?", "Go to settings.", correction="Tap Account > Security > Reset password.")
    pairs = PairBuilder(client=None).build([down])
    g, c = tmp_path / "golden.jsonl", tmp_path / "CHANGELOG.md"
    assert append_golden(pairs, g, c, today="2026-10-23") == 1
    assert append_golden(pairs, g, c, today="2026-10-24") == 0           # idempotent
    rows = read_jsonl(g)
    assert rows[0] == {"id": f"g_{pairs[0]['pair_id']}", "input": down["prompt"], "expected_behavior": down["correction"],
                       "rejected_behavior": down["response"], "tags": ["t"], "source": "correction"}
    assert "- 2026-10-23: +1 golden cases" in c.read_text() and "2026-10-24" not in c.read_text()
