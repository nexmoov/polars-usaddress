# polars-usaddress

US address parsing for Polars — a Rust port of [DataMade's usaddress][usaddress],
exposed as a native Polars plugin.

The trained CRF model is the **unmodified** `usaddr.crfsuite` from the upstream
Python wheel, embedded at compile time. Only the tokenizer and feature
extraction are reimplemented; inference runs through [`crfs`][crfs], a pure-Rust
implementation of the CRFsuite format.

Pinned to **usaddress 0.5.16**.

## Usage

```python
import polars as pl
import polars_usaddress as ua

df = pl.DataFrame({"address": ["123 Main St. Suite 100 Chicago, IL 60601"]})

df.with_columns(parsed=ua.tag_address("address")).unnest("parsed")
```

`tag_address` returns a struct with one nullable string field per component
(see `ua.LABELS`) plus `address_type`. `parse_address` returns the raw
`list[struct[{token, label}]]` labelling when you need order and repetition.

## Parity with upstream

The port's correctness rests on matching two things exactly: the tokenizer, and
the attribute encoding `python-crfsuite` applies between usaddress's nested
feature dicts and the CRF. The latter is **not documented in usaddress** and was
recovered empirically:

| feature dict            | encoded attribute        | weight |
| ----------------------- | ------------------------ | ------ |
| `{"digits": "all_digits"}` | `digits:all_digits`   | 1.0    |
| `{"trailing.zeros": ""}`   | `trailing.zeros:`     | 1.0    |
| `{"abbrev": True}`         | `abbrev`              | 1.0    |
| `{"abbrev": False}`        | `abbrev`              | **0.0**|
| `{"next": {"word": "main"}}` | `next:word:main`    | 1.0    |

False booleans are **emitted with weight 0.0**, not omitted. Nested dicts
prefix-join with `:` and are never recursive — there is no `next:next:`.
`previous:address.start` occurs only at index 1; `next:address.end` only at
index *n*−2.

Getting any of this wrong does not raise — it silently degrades parses. So the
encoder was differential-tested against the reference implementation over 4,034
addresses before any Rust was written (`tools/verify_model.py`, 0 mismatches),
and `tests/parity.rs` replays golden vectors captured from it.

## Regenerating after an upstream bump

```bash
pip install --upgrade usaddress
python tools/verify_model.py     # confirm the encoding still holds
python tools/gen_fixtures.py     # refresh golden vectors
cp "$(python -c 'import usaddress,os;print(usaddress.MODEL_PATH)')" models/
```

The model file and `src/lexicon.rs` are versioned **together** — the feature
code and the weights must agree. Bump `UPSTREAM_VERSION` in `src/lib.rs` and
`python/polars_usaddress/__init__.py` to match.

## Build

```bash
maturin develop --release    # --release matters; debug builds are ~20x slower
cargo test --no-default-features   # parity suite, no Python needed
```

Requires Rust 1.98+ (edition 2024).

## License

MIT, matching upstream. `models/LICENSE.usaddress` covers the vendored model.

[usaddress]: https://github.com/datamade/usaddress
[crfs]: https://crates.io/crates/crfs
