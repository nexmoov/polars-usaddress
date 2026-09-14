# Vendored: crfs 0.4.1

Vendored unmodified from [messense/crfs-rs](https://github.com/messense/crfs-rs) (crates.io
`crfs` 0.4.1, MIT — see LICENSE in this directory) on 2026-08-09, then extended for this
project's inference-speed requirements. Local changes (each a candidate for an upstream PR):

- Pre-decoded feature/label tables at `Model::new` (removes per-access buffer parsing in the
  scoring loop)
- `tag_ids` API with reusable scratch buffers (removes per-call allocations and per-attribute
  cqdb string lookups for callers that cache attribute ids)
- 26-label arm in the unrolled Viterbi match (the usaddress model's label count previously fell
  to the generic path)
- Forward-backward marginals: `MarginalState`, `Context::forward_backward`, `Context::score`, and
  the `Tagger::tag_ids_with_marginals` entry point. Upstream crfs never ported CRFsuite's
  `crf1dc_alpha_score` / `crf1dc_beta_score`, so the `MARGINALS` flag and its context fields were
  inert. Scaled (not log-space) forward-backward, matching CRFsuite, with one deviation: the
  per-position state-score maximum is subtracted before `exp` and added back into `log_norm`.
  Marginals are exactly invariant under that shift, and it keeps `exp` away from overflow.
  Strictly additive — `tag`, `tag_ids` and their buffers are untouched.

All changes preserve arithmetic order; correctness is enforced by this repo's four-layer oracle
parity gate and the full-corpus ID-vs-string equivalence test. The marginals are additionally
checked against brute-force enumeration of all label sequences on the bundled 2-label toy model
(`tagger::tests::marginals_match_brute_force_enumeration`) and against `pycrfsuite.Tagger`
(`benchmark/compare_marginals.py`).

## Provenance in this repo (polars-usaddress)

This copy was pulled into `polars-usaddress/vendor/crfs` verbatim from
[vinvomero/fastaddress](https://github.com/vinvomero/fastaddress)'s `crates/crf` on
2026-09-12, as a drop-in replacement for the crates.io `crfs = "0.4"` dependency, then
built on in two more steps, all in the same session:

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

`tag_ids_with_marginals` (confidence/marginal scoring) remains unused — we only ever want
the single best label sequence, not per-position probabilities, so there's been no reason
to reach for it.

fastaddress's own `benchmark/` and `training/` directories, its `crates/core` and
`crates/python`, and its bundled model file are NOT part of this vendoring — only
`crates/crf` (this directory's contents) was taken.

## CRFsuite attribution

The alpha/beta scaling scheme and decoder structure are ports of CRFsuite's
`crf1d` C implementation (Naoaki Okazaki, BSD 3-clause). The full license and
copyright notice travel with this crate in `LICENSE-CRFSUITE`, alongside the
MIT license of the `crfs` Rust crate this vendoring started from.
