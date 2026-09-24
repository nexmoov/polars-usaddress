# polars-usaddress

US address parsing for Polars — a Rust port of [DataMade's usaddress][usaddress],
exposed as a native Polars plugin.

The trained CRF model is the **unmodified** `usaddr.crfsuite` from the upstream
Python wheel, embedded at compile time. Only the tokenizer and feature
extraction are reimplemented; inference runs through [`crfs`][crfs], a pure-Rust
implementation of the CRFsuite format.

The CRF inference engine (vendor/crfs) is vendored from [vinvomero/fastaddress](https://github.com/vinvomero/fastaddress) 's fork of crfs-rs — 
its pre-decoded model tables, id-based tag_ids tagging path, and forward-backward marginals are what make this port 
fast and confidence-aware, and the parse_with_confidence/tag_with_confidence API here mirrors fastaddress's own.
Credit to the fastaddress author for the performance work; this mostly adds the Polars plugin layer.

## How this was built

This was built with extensive AI assistance (Claude), by someone fluent in
Python and Polars, but novice to Rust. The parts in
`src/` (tokenizing, feature extraction, the Polars glue) are things flansbreezy can
walk through and explain. The CRF math itself — Viterbi decoding, forward-backward marginals,
in `vendor/crfs` — is vendored, not authored, and honestly opaque to the devs the
way most people's dependencies are: I don't understand crfsuite's internals,

The outputs are tested by:
- Differential tests against real Python `usaddress`: golden vectors captured by `tools/gen_fixtures.py`
  (hand-picked edge cases plus a 20k-address synthetic corpus), replayed through the Rust API in
  `tests/parity.rs` and through the Polars plugin in `tests/test_plugin.py`
- An exhaustive equivalence test between this crate's two feature-extraction paths (`id_features_match_string_features_exactly` in `src/features.rs`)

If something looks wrong, issues and contributions are very welcome.

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

`tag_address_with_confidence` and `parse_address_with_confidence` return the
same, plus the CRF's confidence (`sequence_confidence` per address, or
`confidence` per token). They cost an extra forward-backward pass per row.

The plugin parses rows in parallel on its own thread pool, sized like Polars'
own: `POLARS_MAX_THREADS` if set, otherwise one thread per core.

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
encoder was differential-tested against the reference implementation before any
Rust was written, and `tests/parity.rs` replays golden vectors captured from it
(`tools/gen_fixtures.py`: 50 hand-picked addresses covering ordinary addresses,
intersections, PO boxes, punctuation/tokenizer edge cases, unicode, and
`RepeatedLabelError` cases, plus upstream's outputs on a 20k-address synthetic corpus).

## Regenerating after an upstream bump

```bash
uv lock --upgrade-package usaddress && uv sync
uv run python tools/gen_lexicon.py    # regenerate src/lexicon.rs
uv run python tools/gen_fixtures.py   # refresh golden vectors; confirms the encoding still holds
cp "$(uv run python -c 'import usaddress;print(usaddress.MODEL_PATH)')" models/
```

The model file and `src/lexicon.rs` are versioned **together** — the feature
code and the weights must agree. Bump `UPSTREAM_VERSION` in `src/lib.rs` and
`python/polars_usaddress/__init__.py` to match.

## Build

```bash
uv run maturin develop --release   # --release matters; debug builds are ~20x slower
cargo test --no-default-features   # Rust unit + parity tests, no Python needed
make test                          # the above, plus the plugin tests and vendor/crfs's own
make check-rust                    # clippy + rustfmt check
make bench                         # release builds, then per-core throughput vs. Python
make format                        # rustfmt + ruff
```

Requires Rust 1.98+ (edition 2024).

## License

MIT, matching upstream. `models/LICENSE.usaddress` covers the vendored model.

[usaddress]: https://github.com/datamade/usaddress
[crfs]: https://crates.io/crates/crfs
