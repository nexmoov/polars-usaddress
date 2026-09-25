# Vendored: crfs 0.4.1

This directory is [vinvomero/fastaddress](https://github.com/vinvomero/fastaddress)'s fork of
[messense/crfs-rs](https://github.com/messense/crfs-rs) (crates.io `crfs` 0.4.1, MIT — see
LICENSE in this directory), plus a few changes of our own (see "Provenance in this repo" below).

## What fastaddress changed

From fastaddress's own notes (it vendored crfs 0.4.1 on 2026-08-09; each change a candidate
for an upstream PR):

- Pre-decoded feature/label tables at `Model::new` (removes per-access buffer parsing in the
  scoring loop)
- `tag_ids` API with reusable scratch buffers (removes per-call allocations and per-attribute
  cqdb string lookups for callers that cache attribute ids)
- 26-label arm in the unrolled Viterbi match, for fastaddress's own 26-label model. (The
  usaddress 0.5.16 model embedded here has 29 labels, so it takes the generic path.)
- Forward-backward marginals: `MarginalState`, `Context::forward_backward`, `Context::score`, and
  the `Tagger::tag_ids_with_marginals` entry point. Upstream crfs never ported CRFsuite's
  `crf1dc_alpha_score` / `crf1dc_beta_score`, so the `MARGINALS` flag and its context fields were
  inert. Scaled (not log-space) forward-backward, matching CRFsuite, with one deviation: the
  per-position state-score maximum is subtracted before `exp` and added back into `log_norm`.
  Marginals are exactly invariant under that shift, and it keeps `exp` away from overflow.
  Strictly additive — `tag`, `tag_ids` and their buffers are untouched.

fastaddress states all of these preserve arithmetic order, and verified them with its own
parity suites, a brute-force check of the marginals on a 2-label toy model
(`tagger::tests::marginals_match_brute_force_enumeration`), and a comparison against
`pycrfsuite.Tagger` (its `benchmark/compare_marginals.py`). Neither the toy model
(`tests/model.crfsuite`) nor that benchmark was vendored, so those tests, and the `model::tests`
that also read the toy model, can't run here; only `context::tests` do (`make test`). In this
repo, correctness rests on our own suites: `tests/parity.rs`, `tests/test_plugin.py`, and the
id-vs-string equivalence test in `src/features.rs`.

## Provenance in this repo (polars-usaddress)

This copy was pulled into `polars-usaddress/vendor/crfs` verbatim from
[vinvomero/fastaddress](https://github.com/vinvomero/fastaddress)'s `crates/crf` on
2026-09-12, as a drop-in replacement for the crates.io `crfs = "0.4"` dependency. How our
code came to use it, in order:

- Step 1 (this vendoring itself): swap in the fork's pre-decoded feature/label tables,
  still calling the plain string-based `Attribute`/`Tagger::tag` API unchanged.
- Step 2 (`src/attr_cache.rs`): build our own attribute-name -> id table from the embedded
  model's fixed vocabulary, resolve attribute names to ids ourselves, and switch `Parser`
  (`src/lib.rs`) to the fork's `tag_ids` fast path instead of `tag` — `tag_ids` does no
  string work at all.
- Step 3 (`src/bounded_ids.rs`): precompute ids for every feature whose attribute name is a
  small enumerable set (not just resolve them faster), so `src/features.rs` skips
  `format!()`/hashing for most attributes and only falls back to `attr_cache::resolve_one`
  for the handful of free-text feature values (`word`, `endsinpunc`).

- Step 4: `Tagger::tag_ids_with_marginals` backs the `*_with_confidence` API
  (`Parser::parse_with_confidence`/`tag_with_confidence` and their plugin expressions).

Local changes made in this repo, on top of the fastaddress copy:

- Generic-path Viterbi (`Context::viterbi`, used for the usaddress model's 29 labels) finds
  each step's best predecessor in two passes, a max reduction then a first-index search,
  instead of one running argmax. Same sums and same tie-break (first index wins), about 7%
  faster end to end. A dedicated `29` unrolled arm was tried and measured slower.
- `Cargo.toml` allows `dead_code`, so the unused training-side API doesn't warn on every build.

fastaddress's own `benchmark/` and `training/` directories, its `crates/core` and
`crates/python`, and its bundled model file are NOT part of this vendoring — only
`crates/crf` (this directory's contents) was taken.

## CRFsuite attribution

The alpha/beta scaling scheme and decoder structure are ports of CRFsuite's
`crf1d` C implementation (Naoaki Okazaki, BSD 3-clause). The full license and
copyright notice travel with this crate in `LICENSE-CRFSUITE`, alongside the
MIT license of the `crfs` Rust crate this vendoring started from.
