"""Feedback ingestion, PII scrubbing, quality filtering, preference-pair building, splitting and
golden-set emission. Pure logic except FeedbackStore, which owns its `state/` directory.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.io import read_jsonl, stable_hash, write_jsonl
from evalkit.llm import LLMClient

# ----------------------------------------------------------------------------- PII scrubbing

EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RX = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
LONG_DIGITS_RX = re.compile(r"(?<!\d)\d{7,}(?!\d)")


def scrub_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, phone numbers and long digit runs (>= 7 digits: card/account numbers) with
    placeholders. Order matters: emails first (they may contain digits), then phones, then digit runs.
    Returns (scrubbed_text, counts)."""
    counts: dict[str, int] = {}
    text, counts["email"] = EMAIL_RX.subn("<EMAIL>", text)
    text, counts["phone"] = PHONE_RX.subn("<PHONE>", text)
    text, counts["number"] = LONG_DIGITS_RX.subn("<NUMBER>", text)
    return text, counts


def contains_pii(text: str) -> bool:
    return bool(EMAIL_RX.search(text) or PHONE_RX.search(text) or LONG_DIGITS_RX.search(text))


# ----------------------------------------------------------------------------- quality filter

SPAM_KEYWORDS = ["buy followers", "crypto giveaway", "click here", "free money", "casino bonus",
                 "work from home $$$", "http://bit.ly"]
ABUSE_KEYWORDS = ["stupid bot", "you idiot", "shut up"]


@dataclass
class FilterConfig:
    min_prompt_chars: int = 12
    max_per_user: int = 25
    spam_keywords: list[str] = field(default_factory=lambda: list(SPAM_KEYWORDS))
    abuse_keywords: list[str] = field(default_factory=lambda: list(ABUSE_KEYWORDS))


def quality_reason(ev: dict[str, Any], cfg: FilterConfig) -> str | None:
    """Return a rejection reason or None when the event is usable."""
    prompt = (ev.get("prompt") or "").strip()
    if len(prompt) < cfg.min_prompt_chars:
        return "short_prompt"
    if not (ev.get("response") or "").strip():
        return "no_response"
    if ev.get("rating") not in {"up", "down"}:
        return "bad_rating"
    low = (prompt + " " + (ev.get("response") or "") + " " + (ev.get("correction") or "")).lower()
    if any(k in low for k in cfg.spam_keywords):
        return "spam"
    if any(k in low for k in cfg.abuse_keywords):
        return "abuse"
    return None


# ----------------------------------------------------------------------------- feedback store

@dataclass
class IngestStats:
    received: int = 0
    duplicates: int = 0
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    pii_scrubbed: dict[str, int] = field(default_factory=dict)
    watermark: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackStore:
    """Append-only store of accepted, scrubbed feedback events under `state_dir`.

    Files: `events.jsonl` (accepted events), `seen_ids.json` (every event_id ever received, for
    dedup), `watermark.json` (max `ts` ingested so far)."""

    def __init__(self, state_dir: str | Path, filter_cfg: FilterConfig | None = None):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = filter_cfg or FilterConfig()
        self.events_path = self.dir / "events.jsonl"
        self.seen_path = self.dir / "seen_ids.json"
        self.wm_path = self.dir / "watermark.json"
        self.seen: set[str] = set(json.loads(self.seen_path.read_text())) if self.seen_path.exists() else set()
        self.watermark: str | None = (json.loads(self.wm_path.read_text()).get("ts")
                                      if self.wm_path.exists() else None)

    def events(self) -> list[dict[str, Any]]:
        return read_jsonl(self.events_path) if self.events_path.exists() else []

    def ingest(self, raw_events: list[dict[str, Any]]) -> IngestStats:
        stats = IngestStats(received=len(raw_events), watermark=self.watermark)
        per_user: Counter[str] = Counter(e.get("user_id", "?") for e in self.events())
        accepted: list[dict[str, Any]] = []
        for ev in sorted(raw_events, key=lambda e: (str(e.get("ts", "")), str(e.get("event_id", "")))):
            eid = str(ev.get("event_id", ""))
            if not eid or eid in self.seen:
                stats.duplicates += 1
                continue
            self.seen.add(eid)
            reason = quality_reason(ev, self.cfg)
            if reason is None and per_user[ev.get("user_id", "?")] >= self.cfg.max_per_user:
                reason = "user_cap"
            if reason:
                stats.rejected[reason] = stats.rejected.get(reason, 0) + 1
                continue
            clean = dict(ev)
            for fld in ("prompt", "response", "correction"):
                if clean.get(fld):
                    clean[fld], c = scrub_pii(clean[fld])
                    for k, v in c.items():
                        if v:
                            stats.pii_scrubbed[k] = stats.pii_scrubbed.get(k, 0) + v
            clean.setdefault("tags", [])
            per_user[ev.get("user_id", "?")] += 1
            accepted.append(clean)
            if ev.get("ts") and (stats.watermark is None or str(ev["ts"]) > stats.watermark):
                stats.watermark = str(ev["ts"])
        stats.accepted = len(accepted)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in accepted)
        self.watermark = stats.watermark
        self.seen_path.write_text(json.dumps(sorted(self.seen)))
        self.wm_path.write_text(json.dumps({"ts": self.watermark}))
        return stats


# ----------------------------------------------------------------------------- pair building

def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(normalise(a).split()), set(normalise(b).split())
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


UNHELPFUL = ["i can't help", "i cannot help", "contact support", "not sure", "i don't know",
             "[mock:", "unable to assist", "try again later"]


def heuristic_judge(prompt: str, response: str) -> float:
    """Cheap offline judge in [0, 1]: rewards length up to ~150 chars and lexical overlap with the
    prompt, penalises unhelpful boilerplate. Swap for an LLM judge in production."""
    r = (response or "").strip()
    if not r:
        return 0.0
    score = 0.3 + 0.3 * min(1.0, len(r) / 150)
    pt = {t for t in normalise(prompt).split() if len(t) > 3}
    rt = set(normalise(r).split())
    if pt:
        score += 0.3 * len(pt & rt) / len(pt)
    if any(u in r.lower() for u in UNHELPFUL):
        score -= 0.4
    return max(0.0, min(1.0, score))


Judge = Callable[[str, str], float]
REGEN_SYSTEM = ("You are a helpful customer-support assistant. The previous answer was rated unhelpful. "
                "Write a better, specific and polite answer to the user's message.")


@dataclass
class PairBuilder:
    """Turn thumbs-down events into DPO pairs {prompt, chosen, rejected, source, confidence}.

    Strategies, in priority order:
      1. `correction`  the user supplied a correction            -> chosen = correction      (conf 0.9)
      2. `thumbs_up`   another response to a near-identical prompt was rated up -> chosen = it (conf 0.7)
      3. `regenerated` ask the LLM for a candidate; keep it only if judge(candidate) > judge(rejected)
                       (conf = judge margin, capped at 0.6). Skipped when no client is given.
    """
    client: LLMClient | None = None
    judge: Judge = heuristic_judge
    similarity: float = 0.8
    stats: Counter = field(default_factory=Counter)

    def build(self, events: list[dict[str, Any]], exclude_event_ids: set[str] | None = None) -> list[dict[str, Any]]:
        exclude = exclude_event_ids or set()
        ups = [e for e in events if e.get("rating") == "up"]
        by_norm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in ups:
            by_norm[normalise(e["prompt"])].append(e)
        pairs: list[dict[str, Any]] = []
        for ev in sorted(events, key=lambda e: (str(e.get("ts", "")), e["event_id"])):
            if ev.get("rating") != "down" or ev["event_id"] in exclude:
                continue
            self.stats["downs"] += 1
            chosen, source, conf = None, None, 0.0
            if (ev.get("correction") or "").strip() and ev["correction"].strip() != ev["response"].strip():
                chosen, source, conf = ev["correction"].strip(), "correction", 0.9
            else:
                match = self._find_thumbs_up(ev, ups, by_norm)
                if match is not None:
                    chosen, source, conf = match["response"], "thumbs_up", 0.7
                elif self.client is not None:
                    cand = self.client.complete(ev["prompt"], system=REGEN_SYSTEM, seed=0).text.strip()
                    jc, jr = self.judge(ev["prompt"], cand), self.judge(ev["prompt"], ev["response"])
                    if cand and jc > jr:
                        chosen, source, conf = cand, "regenerated", min(0.6, round(jc - jr, 3))
                    else:
                        self.stats["regen_rejected"] += 1
            if chosen is None or chosen == ev["response"]:
                self.stats["unpaired"] += 1
                continue
            self.stats[source] += 1
            pairs.append({"pair_id": stable_hash([ev["event_id"], source]), "prompt": ev["prompt"],
                          "chosen": chosen, "rejected": ev["response"], "source": source,
                          "confidence": conf, "tags": list(ev.get("tags", [])),
                          "conversation_id": ev.get("conversation_id"), "tenant": ev.get("tenant"),
                          "event_id": ev["event_id"]})
        return pairs

    def _find_thumbs_up(self, ev, ups, by_norm):
        key = normalise(ev["prompt"])
        exact = [u for u in by_norm.get(key, []) if u["event_id"] != ev["event_id"]]
        if exact:
            return max(exact, key=lambda u: (self.judge(ev["prompt"], u["response"]), u["event_id"]))
        best, best_sim = None, 0.0
        for u in ups:
            sim = token_jaccard(ev["prompt"], u["prompt"])
            if sim >= self.similarity and (sim, u["event_id"]) > (best_sim, best["event_id"] if best else ""):
                best, best_sim = u, sim
        return best


def to_trl_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """trl.DPOTrainer wants `prompt`, `chosen`, `rejected`; extra columns are kept for provenance."""
    return [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"], "source": p["source"],
             "confidence": p["confidence"], "pair_id": p["pair_id"], "conversation_id": p["conversation_id"],
             "tags": p["tags"], "tenant": p["tenant"]} for p in pairs]


def split_pairs(pairs: list[dict[str, Any]], val_frac: float = 0.2, seed: int = 0
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Grouped + stratified split. Every conversation_id lands in exactly one split (no leakage);
    conversations are bucketed by their majority tag and ~val_frac of each bucket goes to val."""
    convs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        convs[str(p.get("conversation_id"))].append(p)
    buckets: dict[str, list[str]] = defaultdict(list)
    for cid, ps in convs.items():
        tags = Counter(t for p in ps for t in (p.get("tags") or ["untagged"]))
        buckets[tags.most_common(1)[0][0]].append(cid)
    rng = np.random.default_rng(seed)
    val_convs: set[str] = set()
    for tag in sorted(buckets):
        cids = sorted(buckets[tag])
        rng.shuffle(cids)
        n_val = int(round(val_frac * len(cids))) if len(cids) > 1 else 0
        val_convs.update(cids[:n_val])
    train = [p for p in pairs if str(p.get("conversation_id")) not in val_convs]
    val = [p for p in pairs if str(p.get("conversation_id")) in val_convs]
    return train, val


# ----------------------------------------------------------------------------- golden set

def golden_case(pair: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"g_{pair['pair_id']}", "input": pair["prompt"], "expected_behavior": pair["chosen"],
            "rejected_behavior": pair["rejected"], "tags": list(pair.get("tags", [])), "source": pair["source"]}


def append_golden(pairs: list[dict[str, Any]], golden_path: str | Path, changelog_path: str | Path,
                  today: str | None = None) -> int:
    """Append one golden eval case per new pair (dedup by id) and a CHANGELOG line. Returns #added."""
    golden_path, changelog_path = Path(golden_path), Path(changelog_path)
    existing = {c["id"] for c in read_jsonl(golden_path)} if golden_path.exists() else set()
    new = [golden_case(p) for p in pairs if f"g_{p['pair_id']}" not in existing]
    if not new:
        return 0
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in new)
    today = today or _date.today().isoformat()
    srcs = Counter(c["source"] for c in new)
    line = f"- {today}: +{len(new)} golden cases from DPO pairs (" + ", ".join(f"{k}={v}" for k, v in sorted(srcs.items())) + f"); total {len(existing) + len(new)}\n"
    if not changelog_path.exists():
        changelog_path.write_text("# Golden additions changelog\n\n")
    with open(changelog_path, "a", encoding="utf-8") as f:
        f.write(line)
    return len(new)


def write_pairs(pairs: list[dict[str, Any]], out_dir: str | Path, val_frac: float = 0.2, seed: int = 0) -> dict[str, int]:
    out = Path(out_dir)
    train, val = split_pairs(pairs, val_frac, seed)
    write_jsonl(out / "all.jsonl", to_trl_rows(pairs))
    write_jsonl(out / "train.jsonl", to_trl_rows(train))
    write_jsonl(out / "val.jsonl", to_trl_rows(val))
    return {"all": len(pairs), "train": len(train), "val": len(val)}
