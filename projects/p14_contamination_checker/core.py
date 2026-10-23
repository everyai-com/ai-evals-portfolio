"""Dataset contamination checker: n-gram, MinHash-LSH and embedding tiers.

Tiers (highest wins)
--------------------
exact           normalised eval text is identical to a train row
near_duplicate  n-gram overlap >= 0.8 (default 8-grams)
semantic_match  cosine similarity of hashed char-n-gram TF-IDF (or sentence-transformers)
                embedding >= 0.9: catches paraphrases that n-grams miss
partial         n-gram overlap >= 0.3, or any 13-gram hit (the GPT-3/GPT-4 "exact" criterion)
clean           none of the above

Formulas
--------
normalise(text)     lowercase, replace non-alphanumerics with spaces, collapse whitespace.
n-gram overlap      |{eval n-grams} ∩ {train n-grams}| / |{eval n-grams}| using 64-bit
                    blake2b hashes of token tuples; train hashes live in a sorted uint64 array
                    (8 bytes each -> ~100k rows x 60 grams x 2 sizes ≈ 100 MB) looked up with
                    searchsorted.
longest span        longest run of consecutive eval n-grams that hit the index; span tokens =
                    run + n - 1.
MinHash             sig_i(doc) = min over shingles s of (a_i * h(s) + b_i) mod p, p = 2^31 - 1,
                    h(s) a 31-bit blake2b hash of the token 3-gram, a_i in [1, p), b_i in [0, p);
                    a_i * h(s) < 2^62 fits uint64 and wraps p ~2^31 times, so each function
                    behaves like a random permutation. 128 permutations. P(sig_i(A) == sig_i(B)) = J(A, B), so
                    J_hat = mean_i [sig_i(A) == sig_i(B)].
LSH                 split the 128-row signature into b = 32 bands of r = 4 rows; docs sharing
                    any band are candidates. P(candidate) = 1 - (1 - J^r)^b, ~0.5 at J = 0.42.
embedding           hashed char 3/4/5-gram counts (dim 4096) with sublinear tf
                    (1 + log tf) x idf (log((N + 1) / (df + 1)) + 1), L2-normalised; cosine
                    is the dot product. Train rows are vectorised in chunks so memory is
                    bounded by the chunk, not the corpus.
contamination-aware metric   metric on the full eval set vs on the clean subset; the delta is
                    how much leakage inflated the number.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.io import write_jsonl
from evalkit.metrics import wilson_ci
from evalkit.report import md_table, pct

TIERS = ("exact", "near_duplicate", "semantic_match", "partial", "clean")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MERSENNE31 = (1 << 31) - 1  # prime modulus for the universal hashes


# ---------------------------------------------------------------- text


def normalize(text: str) -> str:
    """lowercase, strip punctuation (to spaces), collapse whitespace."""
    return " ".join(_NON_ALNUM.sub(" ", text.lower()).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def ngram_hashes(toks: Sequence[str], n: int) -> list[int]:
    """64-bit blake2b hashes of every contiguous n-token tuple (in order)."""
    if len(toks) < n:
        return []
    return [hash64(" ".join(toks[i : i + n])) for i in range(len(toks) - n + 1)]


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """(length, start) of the longest run of True."""
    best = cur = 0
    best_start = start = 0
    for i, m in enumerate(mask):
        if m:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


# ---------------------------------------------------------------- n-gram index


class _HashSet:
    """Sorted uint64 hash array + parallel int32 train-row index (first row that had it)."""

    def __init__(self, hashes: list[int], owners: list[int]):
        if hashes:
            h = np.array(hashes, dtype=np.uint64)
            o = np.array(owners, dtype=np.int32)
            self.h, first = np.unique(h, return_index=True)
            self.o = o[first]
        else:
            self.h, self.o = np.zeros(0, np.uint64), np.zeros(0, np.int32)

    def __len__(self) -> int:
        return int(self.h.size)

    def lookup(self, hashes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """(hit mask, owner index or -1) for each query hash."""
        if not hashes or self.h.size == 0:
            return np.zeros(len(hashes), bool), np.full(len(hashes), -1, np.int32)
        q = np.array(hashes, dtype=np.uint64)
        pos = np.searchsorted(self.h, q)
        pos[pos >= self.h.size] = 0
        hit = self.h[pos] == q
        owner = np.where(hit, self.o[pos], -1).astype(np.int32)
        return hit, owner


@dataclass
class NgramHit:
    overlap: float
    longest_span: int  # tokens
    span_start: int  # token offset in the eval text
    span_train_id: str | None
    exact_ngram_hit: bool  # any exact_n-gram (default 13) present in train
    exact_ngram_train_id: str | None
    exact_match_id: str | None  # normalised text identical to this train row
    n_grams: int


class NgramIndex:
    """Hashed n-gram index of a training corpus with an n-gram tier and a 13-gram tier."""

    def __init__(self, n: int = 8, exact_n: int = 13):
        if n < 1 or exact_n < 1:
            raise ValueError("n-gram sizes must be >= 1")
        self.n, self.exact_n = n, exact_n
        self.ids: list[str] = []
        self._text_hash: dict[int, str] = {}
        self._grams = _HashSet([], [])
        self._exact = _HashSet([], [])
        self.n_tokens = 0

    def build(
        self, rows: Iterable[dict], text_field: str = "text", id_field: str = "id"
    ) -> NgramIndex:
        hs: list[int] = []
        ho: list[int] = []
        es: list[int] = []
        eo: list[int] = []
        for i, r in enumerate(rows):
            tid = str(r[id_field])
            self.ids.append(tid)
            toks = tokens(str(r[text_field]))
            self.n_tokens += len(toks)
            self._text_hash.setdefault(hash64(" ".join(toks)), tid)
            g = ngram_hashes(toks, self.n)
            hs.extend(g)
            ho.extend([i] * len(g))
            if self.exact_n != self.n:
                g2 = ngram_hashes(toks, self.exact_n)
                es.extend(g2)
                eo.extend([i] * len(g2))
        self._grams = _HashSet(hs, ho)
        self._exact = self._grams if self.exact_n == self.n else _HashSet(es, eo)
        return self

    @property
    def n_rows(self) -> int:
        return len(self.ids)

    @property
    def memory_bytes(self) -> int:
        return int(
            self._grams.h.nbytes
            + self._grams.o.nbytes
            + (0 if self._exact is self._grams else self._exact.h.nbytes + self._exact.o.nbytes)
        )

    def lookup(self, text: str) -> NgramHit:
        toks = tokens(text)
        exact_id = self._text_hash.get(hash64(" ".join(toks)))
        g = ngram_hashes(toks, self.n)
        hit, owner = self._grams.lookup(g)
        overlap = float(hit.mean()) if len(g) else (1.0 if exact_id else 0.0)
        run, start = _longest_true_run(hit)
        span = run + self.n - 1 if run else 0
        span_id = self.ids[int(owner[start])] if run else None
        g2 = ngram_hashes(toks, self.exact_n)
        hit2, owner2 = self._exact.lookup(g2)
        ex = bool(hit2.any())
        ex_id = self.ids[int(owner2[int(np.argmax(hit2))])] if ex else None
        return NgramHit(
            overlap=overlap,
            longest_span=span,
            span_start=start,
            span_train_id=span_id,
            exact_ngram_hit=ex,
            exact_ngram_train_id=ex_id,
            exact_match_id=exact_id,
            n_grams=len(g),
        )


# ---------------------------------------------------------------- MinHash + LSH


def shingles(text: str, k: int = 3) -> set[int]:
    toks = tokens(text)
    if len(toks) < k:
        return {hash64(" ".join(toks)) & 0x7FFFFFFF} if toks else set()
    return {hash64(" ".join(toks[i : i + k])) & 0x7FFFFFFF for i in range(len(toks) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


class MinHashLSH:
    """MinHash signatures (num_perm universal hashes mod 2^31-1) + banded LSH candidate search."""

    def __init__(self, num_perm: int = 128, bands: int = 32, shingle: int = 3, seed: int = 0):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm, self.bands, self.rows, self.shingle = (
            num_perm,
            bands,
            num_perm // bands,
            shingle,
        )
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, _MERSENNE31, size=num_perm, dtype=np.uint64)
        self._b = rng.integers(0, _MERSENNE31, size=num_perm, dtype=np.uint64)
        self.ids: list[str] = []
        self.sigs: np.ndarray = np.zeros((0, num_perm), np.uint64)
        self._buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]

    def signature(self, text: str) -> np.ndarray:
        sh = shingles(text, self.shingle)
        if not sh:
            return np.full(self.num_perm, np.iinfo(np.uint64).max, np.uint64)
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))
        # a, x < 2^31 -> a*x < 2^62, + b < 2^31 -> no uint64 overflow; mod p wraps ~2^31 times
        vals = (self._a[:, None] * x[None, :] + self._b[:, None]) % np.uint64(_MERSENNE31)
        return vals.min(axis=1)

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        return [sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def build(
        self, rows: Iterable[dict], text_field: str = "text", id_field: str = "id"
    ) -> MinHashLSH:
        sigs = []
        for r in rows:
            sig = self.signature(str(r[text_field]))
            idx = len(self.ids)
            self.ids.append(str(r[id_field]))
            sigs.append(sig)
            for band, key in enumerate(self._band_keys(sig)):
                self._buckets[band][key].append(idx)
        self.sigs = np.vstack(sigs) if sigs else self.sigs
        return self

    @staticmethod
    def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))

    def query(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Candidate train rows sharing >= 1 LSH band, with estimated Jaccard, best first."""
        sig = self.signature(text)
        cands: set[int] = set()
        for band, key in enumerate(self._band_keys(sig)):
            cands.update(self._buckets[band].get(key, ()))
        if not cands:
            return []
        idx = np.fromiter(cands, dtype=np.int64, count=len(cands))
        est = (self.sigs[idx] == sig[None, :]).mean(axis=1)
        order = np.argsort(-est)[:top_k]
        return [(self.ids[int(idx[o])], float(est[o])) for o in order]


# ---------------------------------------------------------------- embeddings


class HashedCharNgramBackend:
    """Deterministic hashed char-n-gram TF-IDF embedding (numpy only)."""

    name = "hashed-char-ngram-tfidf"

    def __init__(self, dim: int = 4096, ngram_sizes: Sequence[int] = (3, 4, 5)):
        if dim & (dim - 1):
            raise ValueError("dim must be a power of two")
        self.dim, self.ngram_sizes = dim, tuple(ngram_sizes)
        self._shift = np.uint64(64 - int(math.log2(dim)))
        self.idf: np.ndarray | None = None
        self.n_docs = 0

    def _indices(self, text: str) -> np.ndarray:
        b = np.frombuffer((" " + normalize(text) + " ").encode("utf-8"), dtype=np.uint8).astype(
            np.uint64
        )
        out = []
        for n in self.ngram_sizes:
            if b.size < n:
                continue
            h = np.zeros(b.size - n + 1, np.uint64)
            for j in range(n):
                h = h * np.uint64(257) + b[j : b.size - n + 1 + j]
            out.append((h * np.uint64(0x9E3779B97F4A7C15)) >> self._shift)
        return np.concatenate(out) if out else np.zeros(0, np.uint64)

    def _counts(self, text: str) -> np.ndarray:
        idx = self._indices(text)
        return np.bincount(idx.astype(np.int64), minlength=self.dim).astype(np.float32)

    def fit(self, texts: Iterable[str]) -> HashedCharNgramBackend:
        df = np.zeros(self.dim, np.float64)
        n = 0
        for t in texts:
            df += self._counts(t) > 0
            n += 1
        self.n_docs = n
        self.idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("fit() first")
        M = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            c = self._counts(t)
            nz = c > 0
            c[nz] = 1 + np.log(c[nz])
            M[i] = c * self.idf
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return M / norms


class SentenceTransformerBackend:
    """Optional dense backend; `pip install sentence-transformers`."""

    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._m = SentenceTransformer(model)
        self.name = f"sentence-transformers:{model}"

    def fit(self, texts: Iterable[str]) -> SentenceTransformerBackend:
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._m.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def get_backend(name: str = "hashed") -> Any:
    if name in ("hashed", "tfidf"):
        return HashedCharNgramBackend()
    if name in ("st", "sentence-transformers"):
        try:
            return SentenceTransformerBackend()
        except ImportError:
            return HashedCharNgramBackend()
    raise ValueError(f"unknown backend {name!r}")


class EmbeddingScanner:
    """Top-k nearest train row per eval row by cosine; flags cosine >= threshold."""

    def __init__(self, backend: Any | None = None, threshold: float = 0.9, chunk: int = 2048):
        self.backend = backend or HashedCharNgramBackend()
        self.threshold, self.chunk = threshold, chunk
        self.ids: list[str] = []
        self._texts: list[str] = []

    def fit(
        self, rows: Iterable[dict], text_field: str = "text", id_field: str = "id"
    ) -> EmbeddingScanner:
        for r in rows:
            self.ids.append(str(r[id_field]))
            self._texts.append(str(r[text_field]))
        self.backend.fit(self._texts)
        return self

    def scan(self, texts: Sequence[str], top_k: int = 1) -> list[list[tuple[str, float]]]:
        if not self.ids:
            return [[] for _ in texts]
        E = self.backend.encode(list(texts))  # (m, d)
        best_s = np.full((len(texts), top_k), -np.inf, np.float32)
        best_i = np.full((len(texts), top_k), -1, np.int64)
        for start in range(0, len(self.ids), self.chunk):  # train is vectorised chunk by chunk
            T = self.backend.encode(self._texts[start : start + self.chunk])
            S = E @ T.T  # (m, chunk)
            cand_s = np.concatenate([best_s, S], axis=1)
            cand_i = np.concatenate(
                [best_i, np.arange(start, start + T.shape[0])[None, :].repeat(len(texts), 0)],
                axis=1,
            )
            order = np.argsort(-cand_s, axis=1)[:, :top_k]
            best_s = np.take_along_axis(cand_s, order, 1)
            best_i = np.take_along_axis(cand_i, order, 1)
        return [
            [(self.ids[int(i)], float(s)) for s, i in zip(best_s[r], best_i[r]) if i >= 0]
            for r in range(len(texts))
        ]


# ---------------------------------------------------------------- report


@dataclass
class ItemResult:
    eval_id: str
    strata: str | None
    tier: str
    ngram_overlap: float
    longest_span: int
    exact_ngram_hit: bool
    ngram_train_id: str | None
    exact_match_id: str | None
    cosine: float | None
    nearest_train_id: str | None
    minhash_jaccard: float | None
    minhash_train_id: str | None
    evidence: str | None

    @property
    def train_id(self) -> str | None:
        return (
            self.exact_match_id
            or self.ngram_train_id
            or self.nearest_train_id
            or self.minhash_train_id
        )

    def to_dict(self) -> dict:
        return {**self.__dict__, "train_id": self.train_id}


@dataclass
class ContaminationReport:
    items: list[ItemResult]
    n_train: int
    thresholds: dict
    backend: str
    results_note: dict | None = None

    @property
    def contaminated(self) -> list[ItemResult]:
        return [i for i in self.items if i.tier != "clean"]

    @property
    def rate(self) -> float:
        return len(self.contaminated) / len(self.items) if self.items else 0.0

    def by_tier(self) -> dict[str, int]:
        c = {t: 0 for t in TIERS}
        for i in self.items:
            c[i.tier] += 1
        return c

    def by_strata(self) -> dict[str, dict]:
        g: dict[str, list[ItemResult]] = defaultdict(list)
        for i in self.items:
            g[i.strata or "untagged"].append(i)
        return {
            s: {
                "n": len(v),
                "contaminated": sum(i.tier != "clean" for i in v),
                "rate": sum(i.tier != "clean" for i in v) / len(v),
            }
            for s, v in sorted(g.items())
        }

    def evidence(self) -> list[tuple[str, str, str]]:
        return [(i.eval_id, i.train_id or "?", i.evidence or "") for i in self.contaminated]

    def to_dict(self) -> dict:
        lo, hi = wilson_ci(len(self.contaminated), len(self.items))
        return {
            "n_eval": len(self.items),
            "n_train": self.n_train,
            "backend": self.backend,
            "thresholds": self.thresholds,
            "contamination_rate": self.rate,
            "rate_ci95": [lo, hi],
            "by_tier": self.by_tier(),
            "by_strata": self.by_strata(),
            "evidence": [
                {"eval_id": e, "train_id": t, "snippet": s} for e, t, s in self.evidence()
            ],
            "results_note": self.results_note,
            "items": [i.to_dict() for i in self.items],
        }

    def to_markdown(self, max_evidence: int = 12) -> str:
        lo, hi = wilson_ci(len(self.contaminated), len(self.items))
        th = self.thresholds
        out = [
            f"# Contamination report: {len(self.items)} eval rows vs {self.n_train} train rows",
            "",
            f"**Contamination rate: {pct(self.rate)}** ({len(self.contaminated)}/{len(self.items)}, "
            f"Wilson 95% CI [{pct(lo)}, {pct(hi)}]). Tiers: exact, near_duplicate ({th['ngram']}-gram overlap >= "
            f"{th['near_duplicate']}), semantic_match (cosine >= {th['cosine']}, {self.backend}), partial "
            f"(overlap >= {th['partial']} or a {th['exact_ngram']}-gram hit).",
            "",
            md_table(
                ["tier", "count", "share"],
                [
                    [t, c, pct(c / len(self.items)) if self.items else "-"]
                    for t, c in self.by_tier().items()
                ],
            ),
            "",
            "## By strata",
            "",
            md_table(
                ["strata", "n", "contaminated", "rate"],
                [
                    [s, v["n"], v["contaminated"], pct(v["rate"])]
                    for s, v in self.by_strata().items()
                ],
            ),
            "",
            "## Evidence (eval_id -> train_id; overlapping span in [[ ]])",
            "",
        ]
        ev = self.evidence()
        for e, t, s in ev[:max_evidence]:
            tier = next(i.tier for i in self.items if i.eval_id == e)
            out.append(f"- `{e}` -> `{t}` ({tier}): {s}")
        if len(ev) > max_evidence:
            out.append(f"- ... {len(ev) - max_evidence} more (see --json)")
        if self.results_note:
            r = self.results_note
            out += [
                "",
                "## Contamination-aware reporting",
                "",
                f"`{r['metric']}` on the full eval set: **{r['full']:.3f}** (n={r['n_full']}); on the clean subset: "
                f"**{r['clean']:.3f}** (n={r['n_clean']}); on contaminated rows: {r['contaminated']:.3f} "
                f"(n={r['n_contaminated']}). Leakage inflates the headline metric by "
                f"{(r['full'] - r['clean']) * 100:+.1f} points; report the clean number.",
            ]
        return "\n".join(out)


def _highlight(text: str, start: int, length: int) -> str:
    toks = tokens(text)
    if length <= 0:
        return " ".join(toks[:40]) + (" ..." if len(toks) > 40 else "")
    lo, hi = max(0, start - 4), min(len(toks), start + length + 4)
    body = (
        toks[lo:start] + ["[["] + toks[start : start + length] + ["]]"] + toks[start + length : hi]
    )
    return ("... " if lo else "") + " ".join(body) + (" ..." if hi < len(toks) else "")


def classify(hit: NgramHit, cosine: float | None, thresholds: dict) -> str:
    if hit.exact_match_id:
        return "exact"
    if hit.overlap >= thresholds["near_duplicate"]:
        return "near_duplicate"
    if cosine is not None and cosine >= thresholds["cosine"]:
        return "semantic_match"
    if hit.overlap >= thresholds["partial"] or hit.exact_ngram_hit:
        return "partial"
    return "clean"


def scan(
    eval_rows: Sequence[dict],
    train_rows: Sequence[dict],
    *,
    ngram: int = 8,
    exact_ngram: int = 13,
    cosine: float = 0.9,
    near_duplicate: float = 0.8,
    partial: float = 0.3,
    text_field: str = "text",
    id_field: str = "id",
    strata_field: str = "strata",
    use_embeddings: bool = True,
    use_minhash: bool = True,
    backend: Any | None = None,
    index: NgramIndex | None = None,
    embedder: EmbeddingScanner | None = None,
    lsh: MinHashLSH | None = None,
) -> ContaminationReport:
    """Run every tier over the eval rows and assemble the report. Pre-built index/embedder/lsh
    can be passed to amortise the train-side work across many eval sets."""
    thresholds = {
        "ngram": ngram,
        "exact_ngram": exact_ngram,
        "cosine": cosine,
        "near_duplicate": near_duplicate,
        "partial": partial,
    }
    index = index or NgramIndex(ngram, exact_ngram).build(train_rows, text_field, id_field)
    texts = [str(r[text_field]) for r in eval_rows]
    train_text = {str(r[id_field]): str(r[text_field]) for r in train_rows}
    if use_embeddings:
        embedder = embedder or EmbeddingScanner(backend, cosine).fit(
            train_rows, text_field, id_field
        )
        nearest = embedder.scan(texts, top_k=1)
        backend_name = embedder.backend.name
    else:
        nearest, backend_name = [[] for _ in texts], "none"
    if use_minhash:
        lsh = lsh or MinHashLSH().build(train_rows, text_field, id_field)
        mh = [lsh.query(t, top_k=1) for t in texts]
    else:
        mh = [[] for _ in texts]
    items = []
    for r, text, nn, m in zip(eval_rows, texts, nearest, mh):
        hit = index.lookup(text)
        cos, nid = (nn[0][1], nn[0][0]) if nn else (None, None)
        mj, mid = (m[0][1], m[0][0]) if m else (None, None)
        tier = classify(hit, cos, thresholds)
        evidence = None
        if tier == "exact":
            evidence = f"identical after normalisation: [[ {' '.join(tokens(text)[:20])} ... ]]"
        elif tier in ("near_duplicate", "partial") or (
            tier == "semantic_match" and hit.longest_span >= ngram
        ):
            evidence = (
                _highlight(text, hit.span_start, hit.longest_span)
                + f" (span {hit.longest_span} tokens, overlap {hit.overlap:.2f})"
            )
        elif tier == "semantic_match":
            evidence = (
                f'cosine {cos:.3f} to `{nid}`: "{" ".join(tokens(train_text.get(nid, ""))[:18])} ..." vs '
                f'"{" ".join(tokens(text)[:18])} ..."'
            )
        items.append(
            ItemResult(
                eval_id=str(r[id_field]),
                strata=(str(r[strata_field]) if r.get(strata_field) is not None else None),
                tier=tier,
                ngram_overlap=hit.overlap,
                longest_span=hit.longest_span,
                exact_ngram_hit=hit.exact_ngram_hit,
                ngram_train_id=hit.span_train_id or hit.exact_ngram_train_id,
                exact_match_id=hit.exact_match_id,
                cosine=cos,
                nearest_train_id=nid,
                minhash_jaccard=mj,
                minhash_train_id=mid,
                evidence=evidence,
            )
        )
    return ContaminationReport(
        items=items, n_train=len(train_rows), thresholds=thresholds, backend=backend_name
    )


def contamination_aware_metrics(
    report: ContaminationReport, results: Sequence[dict], metric: str | None = None
) -> dict:
    """Metric on full vs clean vs contaminated eval rows, joined on id."""
    if not results:
        raise ValueError("empty results")
    metric = metric or next((k for k in ("success", "score", "correct") if k in results[0]), None)
    if metric is None:
        raise ValueError("results need a success/score/correct field")
    val = {str(r["id"]): float(r[metric]) for r in results}
    tiers = {i.eval_id: i.tier for i in report.items}
    full = [val[i] for i in tiers if i in val]
    clean = [val[i] for i, t in tiers.items() if t == "clean" and i in val]
    dirty = [val[i] for i, t in tiers.items() if t != "clean" and i in val]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    note = {
        "metric": metric,
        "full": mean(full),
        "n_full": len(full),
        "clean": mean(clean),
        "n_clean": len(clean),
        "contaminated": mean(dirty),
        "n_contaminated": len(dirty),
        "inflation": mean(full) - mean(clean) if clean else 0.0,
    }
    report.results_note = note
    return note


def clean_split(
    report: ContaminationReport,
    eval_rows: Sequence[dict],
    clean_out: str | Path,
    removed_out: str | Path | None = None,
    remove_tiers: Sequence[str] = ("exact", "near_duplicate", "semantic_match", "partial"),
    id_field: str = "id",
) -> dict:
    """Write the decontaminated eval set and the removed rows (annotated with tier + train id)."""
    tiers = {i.eval_id: i for i in report.items}
    keep, drop = [], []
    for r in eval_rows:
        it = tiers.get(str(r[id_field]))
        if it and it.tier in remove_tiers:
            drop.append(
                {
                    **r,
                    "contamination_tier": it.tier,
                    "contamination_train_id": it.train_id,
                    "contamination_evidence": it.evidence,
                }
            )
        else:
            keep.append(r)
    n_keep = write_jsonl(clean_out, keep)
    n_drop = write_jsonl(removed_out, drop) if removed_out else len(drop)
    return {
        "kept": n_keep,
        "removed": n_drop,
        "clean_path": str(clean_out),
        "removed_path": str(removed_out) if removed_out else None,
    }
