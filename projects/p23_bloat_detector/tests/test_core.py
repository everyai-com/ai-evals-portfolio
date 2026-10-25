"""Detectors on hand-built sources: duplicates, near-duplicates, wrappers, concepts, diary, dead code."""
from __future__ import annotations

import textwrap
from pathlib import Path

from projects.p23_bloat_detector.core import (Analyzer, classify_js_lines, concept_stem, detect_diary,
                                              find_concept_clusters, find_dead_code, find_duplicate_clusters,
                                              find_near_duplicates, find_wrapper_chains, name_tokens)

DATA = Path(__file__).resolve().parent.parent / "data"


def _scan_sources(tmp_path: Path, files: dict[str, str]):
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    return Analyzer().scan(tmp_path)


def test_duplicate_bodies_found_regardless_of_identifier_names(tmp_path):
    m = _scan_sources(tmp_path, {"a.py": '''
        def total_price(items, tax):
            subtotal = 0
            for item in items:
                subtotal += item.price * item.qty
            return subtotal * (1 + tax)
        ''', "b.py": '''
        def computeTotal(rows, vat_rate):
            """Docstrings do not change the body hash."""
            acc = 0
            for r in rows:
                acc += r.price * r.qty
            return acc * (1 + vat_rate)

        def different(rows, vat_rate):
            acc = 0
            for r in rows:
                acc += r.price * r.qty
            return acc * (1 - vat_rate)
        '''})
    assert len(m.duplicate_clusters) == 1
    c = m.duplicate_clusters[0]
    assert set(c.members) == {"a.py:total_price", "b.py:computeTotal"}
    assert c.duplicated_lines == min(c.lines_each)
    assert m.duplicate_pct > 0


def test_near_duplicate_found_but_not_exact_duplicate():
    m = Analyzer().scan(DATA / "bloated_repo")
    pairs = {(d.a.split(":")[1], d.b.split(":")[1]) for d in m.near_duplicates}
    assert ("load_orders", "load_customers") in pairs
    nd = m.near_duplicates[0]
    assert 0.6 <= nd.jaccard < 1.0
    # the exact-duplicate pair is reported as a cluster, not as a near-duplicate
    assert not any({d.a.split(":")[1], d.b.split(":")[1]} == {"normalize_sku", "clean_sku"} for d in m.near_duplicates)


def test_wrapper_chain_depth(tmp_path):
    m = _scan_sources(tmp_path, {"w.py": '''
        def _impl(x, y, *, flag=False):
            return [x, y, flag]

        def layer1(x, y, *, flag=False):
            return _impl(x, y, flag=flag)

        def layer2(x, y, *, flag=False):
            """Just forwards."""
            return layer1(x, y, flag=flag)

        def layer3(x, y, *, flag=False):
            return layer2(x, y, flag=flag)

        def not_a_wrapper(x, y, *, flag=False):
            return layer3(y, x, flag=flag)   # arguments swapped: real logic

        def drops_arg(x, y):
            return _impl(x)
        '''})
    chains = {w.chain[0].split(":")[1]: w for w in m.wrapper_chains}
    assert "layer3" in chains and chains["layer3"].depth == 3
    assert [c.split(":")[1] for c in chains["layer3"].chain] == ["layer3", "layer2", "layer1", "_impl"]
    assert "not_a_wrapper" not in chains and "drops_arg" not in chains
    assert m.max_wrapper_depth == 3


def test_concept_cluster_for_date_parsers():
    assert name_tokens("parseDate") == ["parse", "date"]
    assert concept_stem("parse_date") == concept_stem("parseDate") == concept_stem("date_parse") == concept_stem("to_date") == "parse:date"
    assert concept_stem("format_date") == "format:date"
    assert concept_stem("main") is None
    m = Analyzer().scan(DATA / "bloated_repo")
    by_stem = {c.stem: c for c in m.concept_clusters}
    assert "parse:date" in by_stem
    assert {r.split(":")[1] for r in by_stem["parse:date"].members} == {"parse_date", "parseDate", "date_parse", "to_date"}
    assert m.max_concept_cluster == 4
    assert "format:date" not in by_stem  # a single format_date is not a cluster


def test_diary_detection_hand_built_cases_and_docstrings_not_flagged(tmp_path):
    src = '''
    """Module docstring. I wrote this and we like it; let's keep it. Not a comment, never flagged."""
    # 1 this block is nine lines long
    # 2 and narrates the history of the file
    # 3 rather than documenting any contract
    # 4 line four
    # 5 line five
    # 6 line six
    # 7 line seven
    # 8 line eight
    # 9 line nine
    import os


    def f(items):
        """Return the count of items.

        I think docstrings may say "we" and "I" freely: they are documentation, not diary.
        """
        # increment i
        i = 0
        i += 1
        # loop over the items
        for item in items:
            pass
        # now we return the result, I think this works
        return i  # return i


    def g(path):
        # Uses the I/O layer; not first person.
        # Raises FileNotFoundError when the path is missing.  (contract, not narration)
        return os.path.exists(path)
    '''
    m = _scan_sources(tmp_path, {"d.py": src})
    f = m.files[0]
    kinds = {(d.line, d.kind) for d in f.diary}
    assert (3, "long_block") in kinds and any(d.kind == "long_block" and d.n_lines == 9 for d in f.diary)
    assert (20, "narration") in kinds       # "# increment i" above "i = 0"
    assert (23, "narration") in kinds       # "# loop over the items" above "for item in items"
    assert (26, "first_person") in kinds    # "now we return ... I think"
    assert (27, "first_person") not in kinds and (27, "narration") in kinds  # inline "# return i"
    assert not any(d.line in (31, 32) for d in f.diary), "I/O and contract comments are clean"
    assert f.docstring_lines >= 5 and f.diary_lines == 9 + 4
    # docstrings never appear in diary findings
    assert all(not (2 <= d.line <= 2) for d in f.diary)
    assert not any(15 <= d.line <= 18 for d in f.diary)


def test_normal_docstrings_only_file_has_no_diary():
    m = Analyzer().scan(DATA / "clean_repo")
    assert m.diary_lines == 0
    assert m.docstring_lines > 0 and m.comment_lines == 0


def test_dead_code_candidate_and_exclusions(tmp_path):
    m = _scan_sources(tmp_path, {"pkg/mod.py": '''
        from flask import Flask
        app = Flask(__name__)

        def used_helper(x):
            return x + 1

        def never_called(x):
            y = used_helper(x)
            return y * 2

        def recursive_only(n):
            return recursive_only(n - 1) if n else 0

        @app.route("/")
        def index():
            return "ok"

        def main():
            return used_helper(1)

        def __dunder__():
            return 1
        ''', "pkg/other.py": '''
        from pkg.mod import used_helper
        print(used_helper(2))
        ''', "tests/test_mod.py": '''
        def test_x():
            pass

        def helper_in_tests():
            pass
        '''})
    names = {d.name for d in m.dead_code}
    assert names == {"never_called", "recursive_only"}
    bl = Analyzer().scan(DATA / "bloated_repo")
    assert [d.name for d in bl.dead_code] == ["convert_old_format"]


def test_line_classification_python_and_js(tmp_path):
    m = _scan_sources(tmp_path, {"a.py": '''
        """doc"""
        x = "# not a comment"  # inline comment

        # standalone
        y = 1
        ''', "b.js": '''
        // line comment
        const url = "https://example.com/x"; // inline
        /* block
           comment */
        function f() { return 1; }

        '''})
    py = next(f for f in m.files if f.path == "a.py")
    assert (py.code_lines, py.comment_lines, py.docstring_lines, py.blank_lines) == (2, 1, 1, 2)  # leading newline from the heredoc is blank
    js = next(f for f in m.files if f.path == "b.js")
    assert js.language == "javascript"
    assert (js.code_lines, js.comment_lines, js.blank_lines) == (2, 3, 2)
    cls = classify_js_lines("const u = 'http://x';\n")
    assert cls["comment"] == set() and 1 not in cls["comments"]


def test_file_outliers(tmp_path):
    big = "\n".join(f"v{i} = {i}" for i in range(450))
    m = _scan_sources(tmp_path, {"big.py": big, "s1.py": "a = 1", "s2.py": "b = 2", "s3.py": "c = 3"})
    assert [o.path for o in m.file_outliers] == ["big.py"]
    assert "450 code lines >= 400" in m.file_outliers[0].reason


def test_unit_detectors_on_empty_input():
    assert find_duplicate_clusters([]) == [] and find_near_duplicates([]) == []
    assert find_wrapper_chains([]) == [] and find_concept_clusters([]) == [] and find_dead_code([], {}) == []
    assert detect_diary("x", {"comments": {}, "code": set(), "comment": set(), "lines": []}) == []


def test_attribute_call_wrapper_does_not_cross_files_and_unknown_target_ends_chain(tmp_path):
    m = _scan_sources(tmp_path, {"a.py": """
        _DB = {}

        def db_get(key, default=None):
            return _DB.get(key, default)      # forwards to dict.get: real implementation, not a wrapper

        def fetch(key, default=None):
            return db_get(key, default)
        """, "b.py": """
        class Cache:
            def get(self, key, default=None):
                return self.store.get(key, default)
        """})
    assert [(w.chain, w.depth) for w in m.wrapper_chains] == [(["a.py:fetch", "a.py:db_get"], 1)]


def test_nested_function_is_not_a_near_duplicate_of_its_parent_and_tiny_exact_dups_cluster(tmp_path):
    m = _scan_sources(tmp_path, {"n.py": """
        def outer(items, threshold, factor):
            def inner(items, threshold, factor):
                return [x * factor for x in items if x > threshold and x < threshold * factor + 1]
            return [x * factor for x in items if x > threshold and x < threshold * factor + 1] + inner(items, threshold, factor)
        """, "p.py": """
        def exact_match(a, b):
            return a.strip().lower().replace("  ", " ") == b.strip().lower().replace("  ", " ")
        """, "q.py": """
        def exact_match(a, b):
            return a.strip().lower().replace("  ", " ") == b.strip().lower().replace("  ", " ")
        """})
    assert m.near_duplicates == []
    assert len(m.duplicate_clusters) == 1 and set(m.duplicate_clusters[0].members) == {"p.py:exact_match", "q.py:exact_match"}
