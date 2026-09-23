"""Differential tests against the golden vectors in `fixtures.json`, run
through the actual compiled Polars plugin surface (`tag_address` /
`parse_address`) rather than the plain Rust API that `tests/parity.rs`
exercises.

This closes a coverage gap: `tests/parity.rs` calls `Parser::tag`/`parse`
directly, which never goes through `src/expressions.rs`'s `LABELS`-driven
struct-building code. Two real bugs lived there and had zero test coverage
before this file:

* the tokenizer silently dropped a leading Unicode vulgar fraction ('½') as
  its own token (regex crate's `\\w` is narrower than Python's) -- covered
  in `tests/parity.rs` too, but only at the `Parser` level;
* the `LABELS` constant was built off upstream's `usaddress.LABELS`, which
  omits three labels the model can actually emit (`CountryName`,
  `ZipPlus4`, `StreetNamePostModifier`) -- `Parser::tag`'s `HashMap` always
  had these right; only the Polars struct schema in `expressions.rs` (and
  its Python mirror, `LABELS` in `__init__.py`) dropped them on the floor.
  Nothing in `tests/parity.rs` can catch that regression, because it
  doesn't touch `expressions.rs` at all.

Requires the extension to be built first: `maturin develop -r` (see the
`test` target in `Makefile`).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import polars_usaddress as plua

FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text())["fixtures"]


def _case_id(case: dict) -> str:
    return case["input"] or "<empty>"


@pytest.mark.parametrize("case", FIXTURES, ids=_case_id)
def test_tag_address_matches_fixture(case: dict) -> None:
    df = pl.DataFrame({"address": [case["input"]]})
    row = df.select(plua.tag_address("address").alias("tagged")).unnest("tagged").row(0, named=True)

    if case["repeated_label_error"]:
        assert row["address_type"] is None
        assert all(v is None for k, v in row.items() if k != "address_type")
        return

    got = {k: v for k, v in row.items() if k != "address_type" and v is not None}
    assert got == case["tag"], f"components for {case['input']!r}"
    assert row["address_type"] == case["address_type"], f"address_type for {case['input']!r}"


@pytest.mark.parametrize("case", FIXTURES, ids=_case_id)
def test_parse_address_matches_fixture(case: dict) -> None:
    df = pl.DataFrame({"address": [case["input"]]})
    rows = df.select(plua.parse_address("address").alias("parsed")).to_series()[0]
    got = [] if rows is None else [(r["token"], r["label"]) for r in rows]
    want = [(tl["token"], tl["label"]) for tl in case["parse"]]
    assert got == want, f"parse for {case['input']!r}"


CORPUS = json.loads((Path(__file__).parent / "corpus_fixtures.json").read_text())["fixtures"]


def test_corpus_tag_address_matches_upstream() -> None:
    """All 20k corpus addresses in one column, so this also exercises the
    multi-threaded path and row ordering, not just one-row frames."""
    df = pl.DataFrame({"address": [c["input"] for c in CORPUS]})
    rows = df.select(plua.tag_address("address").alias("t")).unnest("t").to_dicts()
    bad = []
    for case, row in zip(CORPUS, rows, strict=True):
        got = {k: v for k, v in row.items() if k != "address_type" and v is not None}
        if case["tag"] is None:
            ok = row["address_type"] is None and not got
        else:
            ok = got == case["tag"] and row["address_type"] == case["address_type"]
        if not ok:
            bad.append(case["input"])
    assert not bad, f"{len(bad)} rows diverge, e.g. {bad[:5]!r}"


def test_corpus_parse_address_matches_upstream() -> None:
    df = pl.DataFrame({"address": [c["input"] for c in CORPUS]})
    parsed = df.select(plua.parse_address("address")).to_series().to_list()
    bad = [
        case["input"]
        for case, rows in zip(CORPUS, parsed, strict=True)
        if [[r["token"], r["label"]] for r in (rows or [])] != case["parse"]
    ]
    assert not bad, f"{len(bad)} rows diverge, e.g. {bad[:5]!r}"


# The two regressions this file exists for, spelled out explicitly rather
# than only via the fixture sweep above -- if these ever fail, it's one of
# the two bugs described in the module docstring, not a fixture drift.


def test_leading_fraction_is_not_dropped() -> None:
    df = pl.DataFrame({"address": ["½ Main St"]})
    row = df.select(plua.tag_address("address").alias("t")).unnest("t").row(0, named=True)
    assert row["AddressNumber"] == "½"
    assert row["StreetName"] == "Main"
    assert row["StreetNamePostType"] == "St"


@pytest.mark.parametrize(
    ("address", "field", "expected"),
    [
        ("123 Main St, Chicago IL 60601 1234", "ZipPlus4", "1234"),
        ("123 Main St, Chicago IL 60601-1234 USA", "CountryName", "USA"),
        ("123 Main St Ext Chicago IL", "StreetNamePostModifier", "Ext"),
        (
            "Main St and Oak St Ext Chicago IL",
            "SecondStreetNamePostModifier",
            "Ext",
        ),
    ],
)
def test_labels_missing_from_upstream_usaddress_LABELS_are_not_dropped(
    address: str, field: str, expected: str
) -> None:
    df = pl.DataFrame({"address": [address]})
    row = df.select(plua.tag_address("address").alias("t")).unnest("t").row(0, named=True)
    assert row[field] == expected
