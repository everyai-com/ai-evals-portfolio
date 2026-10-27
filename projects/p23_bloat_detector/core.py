"""Bloat analysis engine: per-file line metrics, diary comments, duplicate / near-duplicate
function bodies, wrapper chains, concept redundancy, dead-code candidates, file-size outliers.

Pure functions over source text; `Analyzer.scan(repo_dir)` walks a directory. Python files are
parsed with `tokenize` + `ast`; JS/TS files get the language-agnostic subset (line classes,
diary comments, file size) via regexes.

Definitions (also quoted in the README):
* code line        — a line holding at least one non-comment token that is not part of a docstring
* comment line     — a line whose only token is a comment (inline comments count as code lines but
                     their text still feeds the diary detector)
* docstring line   — lines of a string expression that opens a module / class / function body
* comment ratio    — comment_lines / max(1, code_lines)
* duplicate lines  — for each exact-duplicate cluster, the lines of every member except the one kept
* near-duplicate   — Jaccard(5-gram token shingles) >= 0.6 between two functions of similar size
* wrapper          — a function whose whole body is `return g(<its own parameters, same order>)`
"""
from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PY_EXT = {".py"}
JS_EXT = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
             "dist", "build", ".mypy_cache", "out", ".eval-cache"}

# ---------------------------------------------------------------- data classes


@dataclass
class DiaryFinding:
    file: str
    line: int
    end_line: int
    kind: str          # long_block | narration | first_person
    text: str

    @property
    def n_lines(self) -> int:
        return self.end_line - self.line + 1


@dataclass
class FunctionInfo:
    file: str
    name: str
    qualname: str
    lineno: int
    end_lineno: int
    n_params: int
    body_hash: str | None
    n_body_statements: int
    tokens: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    wrapper_target: str | None = None
    wrapper_target_is_attr: bool = False   # `return obj.get(x)`: only resolved within the same file
    is_top_level: bool = True
    in_test_file: bool = False
    has_docstring: bool = False
    ref_count: int = 0           # references anywhere in the scanned tree (filled by Analyzer.scan)

    @property
    def n_lines(self) -> int:
        return self.end_lineno - self.lineno + 1

    @property
    def ref(self) -> str:
        return f"{self.file}:{self.qualname}"


@dataclass
class FileMetrics:
    path: str
    language: str
    total_lines: int
    code_lines: int
    comment_lines: int
    docstring_lines: int
    blank_lines: int
    diary: list[DiaryFinding] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    parse_error: str | None = None

    @property
    def comment_ratio(self) -> float:
        return self.comment_lines / max(1, self.code_lines)

    @property
    def diary_lines(self) -> int:
        return len(diary_line_set(self.diary))


@dataclass
class DuplicateCluster:
    body_hash: str
    members: list[str]           # function refs
    lines_each: list[int]
    duplicated_lines: int        # sum(lines) - max(lines): what a merge removes
    keep: str


@dataclass
class NearDuplicate:
    a: str
    b: str
    jaccard: float
    lines_a: int
    lines_b: int


@dataclass
class WrapperChain:
    chain: list[str]             # refs, outermost wrapper first, final target last
    depth: int                   # number of wrapper hops


@dataclass
class ConceptCluster:
    stem: str                    # e.g. "parse:date"
    members: list[str]
    lines_each: list[int]


@dataclass
class DeadCodeCandidate:
    ref: str
    file: str
    name: str
    lineno: int
    n_lines: int


@dataclass
class FileOutlier:
    path: str
    code_lines: int
    reason: str


@dataclass
class RepoMetrics:
    root: str
    files: list[FileMetrics]
    duplicate_clusters: list[DuplicateCluster]
    near_duplicates: list[NearDuplicate]
    wrapper_chains: list[WrapperChain]
    concept_clusters: list[ConceptCluster]
    dead_code: list[DeadCodeCandidate]
    file_outliers: list[FileOutlier]

    # ---- totals
    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def total_lines(self) -> int:
        return sum(f.total_lines for f in self.files)

    @property
    def code_lines(self) -> int:
        return sum(f.code_lines for f in self.files)

    @property
    def comment_lines(self) -> int:
        return sum(f.comment_lines for f in self.files)

    @property
    def docstring_lines(self) -> int:
        return sum(f.docstring_lines for f in self.files)

    @property
    def blank_lines(self) -> int:
        return sum(f.blank_lines for f in self.files)

    @property
    def source_lines(self) -> int:
        """Non-blank lines: code + comment + docstring."""
        return self.total_lines - self.blank_lines

    @property
    def comment_ratio(self) -> float:
        return self.comment_lines / max(1, self.code_lines)

    @property
    def diary_lines(self) -> int:
        return sum(f.diary_lines for f in self.files)

    @property
    def duplicated_lines(self) -> int:
        return sum(c.duplicated_lines for c in self.duplicate_clusters)

    @property
    def duplicate_pct(self) -> float:
        return self.duplicated_lines / max(1, self.code_lines)

    @property
    def max_wrapper_depth(self) -> int:
        return max((w.depth for w in self.wrapper_chains), default=0)

    @property
    def max_concept_cluster(self) -> int:
        return max((len(c.members) for c in self.concept_clusters), default=0)

    def functions(self) -> list[FunctionInfo]:
        return [fn for f in self.files for fn in f.functions]

    def summary(self) -> dict[str, Any]:
        return {
            "root": self.root, "files": self.n_files, "total_lines": self.total_lines,
            "code_lines": self.code_lines, "comment_lines": self.comment_lines,
            "docstring_lines": self.docstring_lines, "blank_lines": self.blank_lines,
            "comment_ratio": round(self.comment_ratio, 4), "diary_lines": self.diary_lines,
            "duplicated_lines": self.duplicated_lines, "duplicate_pct": round(self.duplicate_pct, 4),
            "duplicate_clusters": len(self.duplicate_clusters), "near_duplicates": len(self.near_duplicates),
            "wrapper_chains": len(self.wrapper_chains), "max_wrapper_depth": self.max_wrapper_depth,
            "concept_clusters": len(self.concept_clusters), "max_concept_cluster": self.max_concept_cluster,
            "dead_code_candidates": len(self.dead_code), "file_outliers": len(self.file_outliers),
        }

    def to_dict(self) -> dict[str, Any]:
        d = {"summary": self.summary(),
             "files": [{**asdict(f), "comment_ratio": round(f.comment_ratio, 4), "diary_lines": f.diary_lines,
                        "functions": [fn.qualname for fn in f.functions]} for f in self.files],
             "duplicate_clusters": [asdict(c) for c in self.duplicate_clusters],
             "near_duplicates": [asdict(n) for n in self.near_duplicates],
             "wrapper_chains": [asdict(w) for w in self.wrapper_chains],
             "concept_clusters": [asdict(c) for c in self.concept_clusters],
             "dead_code": [asdict(d) for d in self.dead_code],
             "file_outliers": [asdict(o) for o in self.file_outliers]}
        for f in d["files"]:
            f.pop("parse_error", None)
        return d


# ---------------------------------------------------------------- line classification (python)


def _docstring_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.append((body[0].lineno, body[0].end_lineno or body[0].lineno))
    return out


def classify_python_lines(source: str, tree: ast.AST | None = None) -> dict[str, Any]:
    """Return code/comment/docstring/blank line sets plus standalone + inline comment texts.

    Uses `tokenize` so strings containing '#' are never mistaken for comments.
    """
    lines = source.splitlines()
    n = len(lines)
    code: set[int] = set()
    comment_only: set[int] = set()
    comments: dict[int, str] = {}          # line -> comment text (without '#')
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        toks = []
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            comments[tok.start[0]] = tok.string.lstrip("#").strip()
        elif tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                          tokenize.ENDMARKER, tokenize.ENCODING):
            continue
        else:
            for ln in range(tok.start[0], tok.end[0] + 1):
                code.add(ln)
    doc: set[int] = set()
    if tree is not None:
        for a, b in _docstring_ranges(tree):
            doc.update(range(a, b + 1))
    code -= doc
    for ln in comments:
        if ln not in code and ln not in doc:
            comment_only.add(ln)
    blank = {ln for ln in range(1, n + 1) if ln not in code and ln not in comment_only and ln not in doc}
    return {"n": n, "code": code, "comment": comment_only, "doc": doc, "blank": blank,
            "comments": comments, "lines": lines}


_JS_LINE_COMMENT = re.compile(r"^\s*//(.*)$")
_JS_BLOCK_START = re.compile(r"^\s*/\*(.*)$")


def classify_js_lines(source: str) -> dict[str, Any]:
    """Regex subset for JS/TS: `//` lines and `/* ... */` blocks are comments. Strings holding
    `//` (e.g. URLs) are not standalone comment lines because the line starts with code."""
    lines = source.splitlines()
    code: set[int] = set()
    comment_only: set[int] = set()
    comments: dict[int, str] = {}
    in_block = False
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if in_block:
            comments[i] = s.strip("*/ ").strip()
            comment_only.add(i)
            if "*/" in s:
                in_block = False
                rest = s.split("*/", 1)[1].strip()
                if rest:
                    comment_only.discard(i)
                    code.add(i)
            continue
        if not s:
            continue
        m = _JS_LINE_COMMENT.match(raw)
        if m:
            comments[i] = m.group(1).strip()
            comment_only.add(i)
            continue
        m = _JS_BLOCK_START.match(raw)
        if m:
            comments[i] = m.group(1).replace("*/", "").strip("* ").strip()
            if "*/" not in s:
                in_block = True
            comment_only.add(i)
            continue
        code.add(i)
        idx = raw.find("//")
        if idx > 0 and "://" not in raw[max(0, idx - 1):idx + 3]:
            comments[i] = raw[idx + 2:].strip()
    blank = {ln for ln in range(1, len(lines) + 1) if ln not in code and ln not in comment_only}
    return {"n": len(lines), "code": code, "comment": comment_only, "doc": set(), "blank": blank,
            "comments": comments, "lines": lines}


# ---------------------------------------------------------------- diary detector

FIRST_PERSON = re.compile(
    r"(?:\bI\b(?!/O)|\bI'm\b|\bI've\b|\bI'll\b|\bI think\b|\bwe\b|\bwe're\b|\bwe'll\b|\bwe've\b"
    r"|\blet's\b|\blets\b|\bnow we\b|\bour\b|\bmy\b)", re.IGNORECASE)
# "I" alone is only first person when capitalised; re-check case for the bare pronoun.
_BARE_I = re.compile(r"\bI\b(?!/O)")
NARRATION_VERBS = {"increment", "decrement", "set", "sets", "return", "returns", "call", "calls", "loop",
                   "loops", "iterate", "iterates", "add", "adds", "append", "appends", "check", "checks",
                   "assign", "assigns", "initialize", "initialise", "initializes", "init", "import",
                   "imports", "define", "defines", "create", "creates", "get", "gets", "print", "prints",
                   "open", "opens", "close", "closes", "update", "updates", "compute", "computes",
                   "store", "stores", "save", "saves", "convert", "converts", "declare", "declares",
                   "raise", "raises", "yield", "yields", "break", "continue", "pass"}
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_STOP = {"the", "a", "an", "to", "of", "and", "in", "on", "for", "it", "is", "this", "that", "by", "with",
         "as", "now", "then", "here", "value", "values", "variable", "var"}


def _is_first_person(text: str) -> bool:
    m = FIRST_PERSON.search(text)
    if not m:
        return False
    tok = m.group(0)
    if tok == "I" or tok == "i":
        return bool(_BARE_I.search(text))
    return True


def _is_narration(comment: str, next_code_line: str | None, max_words: int = 6) -> bool:
    """`# increment i` above `i += 1`: comment starts with a narration verb and names an
    identifier that appears on the next code line."""
    if next_code_line is None:
        return False
    words = [w.lower() for w in _WORD.findall(comment)]
    if not words or words[0] not in NARRATION_VERBS or len(words) > max_words:
        return False  # long sentences starting with a verb are contracts ("Raises X when..."), not narration
    idents = {w.lower() for w in _WORD.findall(next_code_line)}
    if words[0] in {"return", "break", "continue", "pass", "raise", "yield"} and words[0] in idents:
        return True
    return any(w in idents for w in words[1:] if w not in _STOP)


def detect_diary(path: str, cls: dict[str, Any], long_block: int = 8) -> list[DiaryFinding]:
    """Three diary signals over comment text (docstrings are never scanned):
    * long_block   — a run of >= `long_block` consecutive standalone comment lines
    * narration    — a comment that narrates the next code line (`# increment i` / `i += 1`)
    * first_person — "I", "we", "let's", "now we", "our" ... in a comment
    """
    out: list[DiaryFinding] = []
    comments: dict[int, str] = cls["comments"]
    code: set[int] = cls["code"]
    lines: list[str] = cls["lines"]
    standalone = sorted(cls["comment"])
    # long blocks
    run: list[int] = []
    for ln in standalone + [None]:
        if run and (ln is None or ln != run[-1] + 1):
            if len(run) >= long_block:
                out.append(DiaryFinding(path, run[0], run[-1], "long_block",
                                        comments.get(run[0], "")[:60]))
            run = []
        if ln is not None:
            run.append(ln)
    # per-comment signals
    for ln in sorted(comments):
        text = comments[ln]
        if not text:
            continue
        if _is_first_person(text):
            out.append(DiaryFinding(path, ln, ln, "first_person", text[:80]))
            continue
        nxt = None
        if ln in code:
            nxt = lines[ln - 1].split("#", 1)[0]
        else:
            for k in range(ln + 1, min(ln + 4, len(lines) + 1)):
                if k in code:
                    nxt = lines[k - 1]
                    break
        if _is_narration(text, nxt):
            out.append(DiaryFinding(path, ln, ln, "narration", text[:80]))
    return out


def diary_line_set(findings: Iterable[DiaryFinding]) -> set[int]:
    s: set[int] = set()
    for f in findings:
        s.update(range(f.line, f.end_line + 1))
    return s


# ---------------------------------------------------------------- functions (python ast)


class _Normaliser(ast.NodeTransformer):
    """Rename every local identifier (Name ids, arg names) to `_v<k>` in first-appearance order so
    `def a(x): y = x*2; return y` and `def b(q): r = q*2; return r` hash identically. Attribute names
    and constants are kept — `.strip()` vs `.lower()` is a real difference."""

    def __init__(self) -> None:
        self.map: dict[str, str] = {}

    def _ph(self, name: str) -> str:
        if name not in self.map:
            self.map[name] = f"_v{len(self.map)}"
        return self.map[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._ph(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg=self._ph(node.arg), annotation=None), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # nested defs
        node.name = self._ph(node.name)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


def _body_without_docstring(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def normalised_body_hash(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """sha1 of `ast.dump` of the identifier-normalised body (docstring removed)."""
    norm = _Normaliser()
    # normalise params first so their placeholders are stable regardless of body order
    for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        norm._ph(a.arg)
    if fn.args.vararg:
        norm._ph(fn.args.vararg.arg)
    if fn.args.kwarg:
        norm._ph(fn.args.kwarg.arg)
    body = [norm.visit(ast.parse(ast.unparse(s)).body[0]) for s in _body_without_docstring(fn)]
    dumped = "\n".join(ast.dump(s, include_attributes=False) for s in body)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()[:16]


def function_tokens(source_lines: list[str], fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Raw token strings (names, ops, numbers, strings) of the function source, comments dropped."""
    seg = "\n".join(source_lines[fn.lineno - 1: fn.end_lineno])
    seg = _dedent(seg)
    out: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(seg).readline):
            if tok.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING):
                out.append(tok.string)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass
    return out


def _dedent(seg: str) -> str:
    lines = seg.splitlines()
    first = lines[0]
    indent = len(first) - len(first.lstrip())
    return "\n".join(ln[indent:] if ln[:indent].strip() == "" else ln for ln in lines)


def _call_target(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def wrapper_target(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, bool] | None:
    """(name, is_attribute_call) of the function this one merely forwards to, or None.

    A wrapper's body (docstring aside) is exactly `return g(p1, p2, ..., *args, **kwargs)` (or the
    bare call, or `await g(...)`) where the positional arguments are the wrapper's own parameters in
    order and keyword arguments forward same-named parameters."""
    body = _body_without_docstring(fn)
    if len(body) != 1:
        return None
    stmt = body[0]
    val = stmt.value if isinstance(stmt, (ast.Return, ast.Expr)) else None
    if isinstance(val, ast.Await):
        val = val.value
    if not isinstance(val, ast.Call):
        return None
    params = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    kwonly = [a.arg for a in fn.args.kwonlyargs]
    pos: list[str] = []
    for a in val.args:
        if isinstance(a, ast.Name):
            pos.append(a.id)
        elif isinstance(a, ast.Starred) and isinstance(a.value, ast.Name) and fn.args.vararg \
                and a.value.id == fn.args.vararg.arg:
            continue
        else:
            return None
    if pos != params[:len(pos)]:
        return None
    remaining = set(params[len(pos):])
    forwarded_kw = {k.arg for k in val.keywords if k.arg}
    if remaining - forwarded_kw:
        return None  # a parameter was dropped or re-ordered: not a pure wrapper
    for k in val.keywords:
        if k.arg is None:
            if not (isinstance(k.value, ast.Name) and fn.args.kwarg and k.value.id == fn.args.kwarg.arg):
                return None
        elif not (isinstance(k.value, ast.Name) and k.value.id == k.arg and k.arg in params + kwonly):
            return None
    target = _call_target(val)
    if target is None or target == fn.name:
        return None
    return target, isinstance(val.func, ast.Attribute)


def _decorator_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    out = []
    for d in fn.decorator_list:
        node = d.func if isinstance(d, ast.Call) else d
        out.append(ast.unparse(node))
    return out


def extract_functions(path: str, source: str, tree: ast.Module, in_test_file: bool) -> list[FunctionInfo]:
    lines = source.splitlines()
    out: list[FunctionInfo] = []

    def visit(node: ast.AST, prefix: str, top: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{child.name}"
                body = _body_without_docstring(child)
                wt = wrapper_target(child)
                out.append(FunctionInfo(
                    file=path, name=child.name, qualname=qual, lineno=child.lineno,
                    end_lineno=child.end_lineno or child.lineno,
                    n_params=len(child.args.posonlyargs + child.args.args + child.args.kwonlyargs),
                    body_hash=normalised_body_hash(child), n_body_statements=len(body),
                    tokens=function_tokens(lines, child), decorators=_decorator_names(child),
                    wrapper_target=wt[0] if wt else None, wrapper_target_is_attr=bool(wt and wt[1]),
                    is_top_level=top, in_test_file=in_test_file,
                    has_docstring=len(body) != len(child.body)))
                visit(child, qual + ".", False)
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.", False)
            else:
                visit(child, prefix, top)

    visit(tree, "", True)
    return out


# ---------------------------------------------------------------- repo-level detectors


def shingles(tokens: list[str], n: int = 5) -> set[str]:
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_duplicate_clusters(functions: list[FunctionInfo], min_lines: int = 3, min_tokens: int = 20,
                            refs: dict[str, int] | None = None) -> list[DuplicateCluster]:
    """Group by normalised body hash. Trivial bodies (`return None`) are skipped: a member needs
    >= `min_lines` lines or >= `min_tokens` tokens."""
    groups: dict[str, list[FunctionInfo]] = defaultdict(list)
    for fn in functions:
        if fn.body_hash and fn.n_body_statements >= 1 and not fn.in_test_file \
                and (fn.n_lines >= min_lines or len(fn.tokens) >= min_tokens):
            groups[fn.body_hash].append(fn)
    out: list[DuplicateCluster] = []
    for h, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda f: (-(refs or {}).get(f.name, 0), f.ref))
        lines = [m.n_lines for m in members]
        out.append(DuplicateCluster(h, [m.ref for m in members], lines, sum(lines) - max(lines), members[0].ref))
    out.sort(key=lambda c: (-c.duplicated_lines, c.members[0]))
    return out


def find_near_duplicates(functions: list[FunctionInfo], threshold: float = 0.6, min_tokens: int = 20,
                         size_ratio: float = 0.5, n: int = 5) -> list[NearDuplicate]:
    """Token 5-gram Jaccard between every pair of similarly sized functions (idea borrowed from
    p14's n-gram contamination index, implemented locally and without hashing)."""
    cands = [f for f in functions if len(f.tokens) >= min_tokens and not f.in_test_file]
    sh = {f.ref: shingles(f.tokens, n) for f in cands}
    out: list[NearDuplicate] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if a.body_hash == b.body_hash:
                continue  # exact duplicates are reported as clusters, not as near-duplicates
            if a.file == b.file and (a.qualname.startswith(b.qualname + ".") or b.qualname.startswith(a.qualname + ".")):
                continue  # a nested function shares its parent's tokens by construction
            la, lb = len(a.tokens), len(b.tokens)
            if min(la, lb) / max(la, lb) < size_ratio:
                continue
            j = jaccard(sh[a.ref], sh[b.ref])
            if j >= threshold and (a.ref, b.ref) not in seen_pairs:
                seen_pairs.add((a.ref, b.ref))
                out.append(NearDuplicate(a.ref, b.ref, round(j, 3), a.n_lines, b.n_lines))
    out.sort(key=lambda d: (-d.jaccard, d.a))
    return out


def find_wrapper_chains(functions: list[FunctionInfo]) -> list[WrapperChain]:
    """Follow `wrapper_target` links. A bare-name call resolves to any function with that name
    (same file preferred); an attribute call (`obj.get(...)`) only resolves within the same file, so
    `_DB.get` never links to some unrelated `Cache.get`. A function forwarding to something unknown
    (a dict, a library) is the real implementation, not a wrapper. Chains are reported from each
    outermost wrapper; depth = number of wrapper hops before the real implementation."""
    by_name: dict[str, list[FunctionInfo]] = defaultdict(list)
    for f in functions:
        by_name[f.name].append(f)

    def resolve(f: FunctionInfo) -> FunctionInfo | None:
        if not f.wrapper_target:
            return None
        cands = [g for g in by_name.get(f.wrapper_target, []) if g is not f]
        same = [g for g in cands if g.file == f.file]
        if f.wrapper_target_is_attr:
            cands = same
        return (same or cands or [None])[0]

    wrappers = {f.ref: tgt for f in functions if (tgt := resolve(f)) is not None}
    inner = {tgt.ref for tgt in wrappers.values()}
    refs = {f.ref: f for f in functions}
    out: list[WrapperChain] = []
    for ref in sorted(wrappers):
        if ref in inner:
            continue  # not outermost
        chain, seen = [ref], {ref}
        cur = refs[ref]
        while cur.ref in wrappers and wrappers[cur.ref].ref not in seen:
            cur = wrappers[cur.ref]
            seen.add(cur.ref)
            chain.append(cur.ref)
        out.append(WrapperChain(chain, len(chain) - 1))
    out.sort(key=lambda w: (-w.depth, w.chain[0]))
    return out


# concept redundancy: split names into tokens, map verbs to a synonym class, key = class:nouns
_GENERIC = {"get", "do", "make", "new", "impl", "internal", "old", "legacy", "helper", "util", "utils",
            "fn", "func", "function", "v1", "v2", "v3", "v4", "my", "the", "a", "an", "safe", "fast",
            "simple", "quick", "better", "real", "actual", "core", "base", "main", "handler", "wrapper"}
_VERB_CLASS = {
    "parse": {"parse", "to", "as", "into", "convert", "load", "read", "decode", "deserialize",
              "deserialise", "coerce", "cast", "interpret"},
    "format": {"format", "render", "dump", "serialize", "serialise", "encode", "stringify", "str", "write",
               "print", "show", "display"},
    "fetch": {"fetch", "retrieve", "lookup", "find", "query", "request", "pull", "select", "search"},
    "validate": {"validate", "check", "verify", "assert", "ensure", "is", "has", "can", "should", "test"},
    "normalize": {"normalize", "normalise", "clean", "sanitize", "sanitise", "strip", "canonicalize",
                  "canonicalise", "tidy", "fix"},
    "compute": {"compute", "calc", "calculate", "count", "sum", "measure", "score", "estimate", "evaluate"},
    "build": {"build", "create", "construct", "generate", "init", "setup", "prepare", "assemble", "produce", "from"},
    "send": {"send", "post", "emit", "publish", "dispatch", "push", "notify", "deliver"},
    "update": {"update", "set", "put", "save", "store", "persist", "upsert", "patch", "apply"},
    "delete": {"delete", "remove", "drop", "clear", "purge", "destroy"},
}
_VERB_TO_CLASS = {v: k for k, vs in _VERB_CLASS.items() for v in vs}
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def name_tokens(name: str) -> list[str]:
    name = name.strip("_")
    name = _CAMEL.sub("_", name)
    return [t for t in name.lower().split("_") if t]


def concept_stem(name: str) -> str | None:
    """`parse_date`, `parseDate`, `date_parse`, `to_date` -> "parse:date"; `format_date` -> "format:date".
    Returns None when nothing noun-like survives (e.g. `main`, `run`)."""
    toks = name_tokens(name)
    verbs = [t for t in toks if t in _VERB_TO_CLASS]
    nouns = [t for t in toks if t not in _VERB_TO_CLASS and t not in _GENERIC and not t.isdigit()]
    if not nouns:
        return None
    cls = _VERB_TO_CLASS[verbs[0]] if verbs else "-"
    return f"{cls}:{'_'.join(sorted(set(nouns)))}"


def find_concept_clusters(functions: list[FunctionInfo], min_members: int = 2) -> list[ConceptCluster]:
    groups: dict[str, list[FunctionInfo]] = defaultdict(list)
    for f in functions:
        # methods sharing a name across classes (`to_dict` on every dataclass) are a protocol, not
        # three ways of doing one thing: only module-level functions are clustered
        if f.in_test_file or not f.is_top_level or f.name.startswith("__"):
            continue
        stem = concept_stem(f.name)
        if stem:
            groups[stem].append(f)
    out = []
    for stem, members in groups.items():
        names = {m.name for m in members}
        if len(members) >= min_members and len(names) >= 2:  # distinct spellings of one concept
            members.sort(key=lambda m: m.ref)
            out.append(ConceptCluster(stem, [m.ref for m in members], [m.n_lines for m in members]))
    out.sort(key=lambda c: (-len(c.members), c.stem))
    return out


ROUTE_DECORATOR = re.compile(r"(route|\.get|\.post|\.put|\.delete|\.patch|\.api|command|register|listens_for"
                             r"|fixture|hookimpl|task|job|subscribe|on_event|receiver|handler|endpoint"
                             r"|\.command|\.callback|app\.|router\.|bp\.|cli\.|pytest\.)", re.IGNORECASE)
_TEST_FILE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")


def is_test_file(path: str) -> bool:
    return bool(_TEST_FILE.search(path.replace("\\", "/")))


def collect_references(tree: ast.Module) -> dict[str, list[int]]:
    """name -> lines where it is referenced (Name loads, Attribute attrs, strings inside __all__)."""
    refs: dict[str, list[int]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            refs[node.id].append(node.lineno)
        elif isinstance(node, ast.Attribute):
            refs[node.attr].append(node.lineno)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            for elt in getattr(node.value, "elts", []):
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    refs[elt.value].append(0)
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            for alias in node.names:
                refs[alias.name.split(".")[-1]].append(node.lineno)
    return refs


def find_dead_code(functions: list[FunctionInfo], refs_by_file: dict[str, dict[str, list[int]]]) -> list[DeadCodeCandidate]:
    """Top-level functions never referenced outside their own body. Excluded: `main`, dunder names,
    `pytest_*` hooks, test files / `test_*`, route-ish decorators, names exported via `__all__`
    (counted as a ref). Scan the *whole* tree: a library's public API looks dead on its own."""
    out = []
    for f in functions:
        if not f.is_top_level or f.in_test_file or f.name == "main" or f.name.startswith("__") \
                or f.name.startswith("test_") or f.name.startswith("pytest_"):
            continue  # entry points and framework hooks are called by name from outside
        if any(ROUTE_DECORATOR.search(d) for d in f.decorators):
            continue
        referenced = False
        for path, refs in refs_by_file.items():
            for ln in refs.get(f.name, []):
                if path != f.file or not (f.lineno <= ln <= f.end_lineno):
                    referenced = True
                    break
            if referenced:
                break
        if not referenced:
            out.append(DeadCodeCandidate(f.ref, f.file, f.name, f.lineno, f.n_lines))
    out.sort(key=lambda d: (-d.n_lines, d.ref))
    return out


def find_file_outliers(files: list[FileMetrics], abs_lines: int = 400, median_factor: float = 3.0,
                       min_lines: int = 150) -> list[FileOutlier]:
    sizes = sorted(f.code_lines for f in files)
    median = sizes[len(sizes) // 2] if sizes else 0
    out = []
    for f in files:
        if f.code_lines >= abs_lines:
            out.append(FileOutlier(f.path, f.code_lines, f"{f.code_lines} code lines >= {abs_lines}"))
        elif len(files) >= 4 and f.code_lines >= min_lines and f.code_lines > median_factor * median:
            out.append(FileOutlier(f.path, f.code_lines, f"{f.code_lines} code lines > {median_factor:g}x median ({median})"))
    out.sort(key=lambda o: -o.code_lines)
    return out


# ---------------------------------------------------------------- analyzer


class Analyzer:
    """`Analyzer().scan(repo_dir)` -> RepoMetrics. All thresholds that shape *detection* live here;
    thresholds that shape the *verdict* live in `budget.RefactorBudget`."""

    def __init__(self, long_block: int = 8, near_dup_jaccard: float = 0.6, dup_min_lines: int = 3,
                 outlier_lines: int = 400, exts: Iterable[str] | None = None):
        self.long_block = long_block
        self.near_dup_jaccard = near_dup_jaccard
        self.dup_min_lines = dup_min_lines
        self.outlier_lines = outlier_lines
        self.exts = set(exts) if exts else PY_EXT | JS_EXT

    def iter_files(self, root: Path) -> list[Path]:
        out = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in self.exts and not any(part in SKIP_DIRS for part in p.relative_to(root).parts):
                out.append(p)
        return out

    def analyse_file(self, rel: str, source: str) -> tuple[FileMetrics, dict[str, list[int]]]:
        ext = Path(rel).suffix
        refs: dict[str, list[int]] = {}
        if ext in PY_EXT:
            tree: ast.Module | None = None
            err = None
            try:
                tree = ast.parse(source)
            except SyntaxError as e:  # still count lines
                err = f"{e.__class__.__name__}: {e.msg} (line {e.lineno})"
            cls = classify_python_lines(source, tree)
            fns = extract_functions(rel, source, tree, is_test_file(rel)) if tree is not None else []
            refs = collect_references(tree) if tree is not None else {}
            fm = FileMetrics(rel, "python", cls["n"], len(cls["code"]), len(cls["comment"]), len(cls["doc"]),
                             len(cls["blank"]), functions=fns, parse_error=err)
        else:
            cls = classify_js_lines(source)
            fm = FileMetrics(rel, "javascript", cls["n"], len(cls["code"]), len(cls["comment"]), 0, len(cls["blank"]))
        fm.diary = detect_diary(rel, cls, self.long_block)
        return fm, refs

    def scan(self, repo_dir: str | Path) -> RepoMetrics:
        root = Path(repo_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"not a directory: {root}")
        files: list[FileMetrics] = []
        refs_by_file: dict[str, dict[str, list[int]]] = {}
        for p in self.iter_files(root):
            rel = p.relative_to(root).as_posix()
            try:
                src = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            fm, refs = self.analyse_file(rel, src)
            files.append(fm)
            refs_by_file[rel] = refs
        fns = [fn for f in files for fn in f.functions]
        ref_counts = {name: sum(len(v) for v in (r.get(name, []) for r in refs_by_file.values()))
                      for name in {fn.name for fn in fns}}
        for fn in fns:
            fn.ref_count = ref_counts.get(fn.name, 0)
        dups = find_duplicate_clusters(fns, self.dup_min_lines, refs=ref_counts)
        near = find_near_duplicates(fns, self.near_dup_jaccard)
        return RepoMetrics(
            root=str(root), files=files, duplicate_clusters=dups, near_duplicates=near,
            wrapper_chains=find_wrapper_chains(fns), concept_clusters=find_concept_clusters(fns),
            dead_code=find_dead_code(fns, refs_by_file),
            file_outliers=find_file_outliers(files, self.outlier_lines))
