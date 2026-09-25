# Reading Guide: polars-usaddress

This is a tour of the codebase for someone new to Rust. It assumes no Rust
background beyond "I've heard of it" and walks through the files in the order
that makes each one easiest to understand, explaining the language features
as they come up rather than all at once up front.

The short version of what this crate does: it's a Rust reimplementation of
DataMade's Python `usaddress` library (a CRF-based US address parser),
exposed as a native Polars plugin so it runs fast and in parallel across
DataFrame columns. The trained model is unchanged from upstream -- only the
tokenizer and feature extraction are reimplemented; inference runs through
a vendored pure-Rust CRFsuite reader (`vendor/crfs`).

## 0. Two things worth knowing before you start

**Comments here answer "why", not "what".** The code was deliberately kept
light on comments that just restate what a line does -- if you can't tell
what `s.chars().all(|c| c.is_ascii_digit())` does from reading it, that's a
Rust-syntax question (covered below), not a missing comment. The comments
that remain exist because the code alone can't tell you the thing that
matters: a subtle mismatch with the Python original, a correctness
invariant, or a reason something is built the way it is instead of some
simpler-looking way. Treat this guide as the place the broader "why" now
lives, since a lot of it used to be duplicated across several files as
comments.

**This is a port, not an original design.** Almost every function in
`src/features.rs` and `src/lib.rs` mirrors a specific Python function in
upstream `usaddress`, bug-for-bug where upstream has quirks. When something
in the Rust looks oddly specific, the likely reason is "Python did it that
way and we need bit-for-bit identical output."

## 1. Rust concepts you'll need, introduced where they first matter

You don't need to learn these up front -- skip to section 2 and come back
here when something doesn't parse.

- **`Option<T>`**: either `Some(value)` or `None`. Rust has no `null`;
  "this might not have a value" is always spelled out in the type. You'll
  see `Option<String>`, `Option<u32>`, even `Option<Option<u32>>` (two
  independent "maybe"s stacked -- see `bounded_ids.rs`).
- **`Result<T, E>`**: either `Ok(value)` or `Err(error)`. Rust has no
  exceptions; a function that can fail returns a `Result` and the caller
  must handle both cases. The `?` operator at the end of an expression means
  "if this is `Err`, return it immediately from the current function;
  otherwise unwrap the `Ok` value and keep going."
- **`match`**: like a `switch`, but the compiler checks you've covered every
  case. You'll see it constantly, often destructuring an `Option` or
  `Result` in the same expression that inspects it.
- **Closures**: `|x| x + 1` is an anonymous function, same idea as a Python
  lambda or JS arrow function. `.map(|x| ...)`, `.filter(|x| ...)` chain
  them over iterators, much like Python's `map`/`filter` but resolved at
  compile time with no per-call overhead.
- **Ownership and `&`**: every value has exactly one owner; `&value` borrows
  it temporarily without taking ownership. You mostly don't need to reason
  about this deeply to *read* this codebase -- just know that a `&str`
  parameter means "give me a peek at a string you still own" and a `String`
  parameter means "I'm taking this."
- **Traits**: Rust's interfaces. `#[derive(Debug, Clone)]` on a struct
  auto-generates an implementation of the `Debug` (pretty-printable) and
  `Clone` (duplicatable) traits. You'll see this a lot and can mostly treat
  it as boilerplate.
- **`LazyLock`**: a value that's computed once, the first time it's
  touched, and cached forever after. Used here for anything expensive to
  build that should only happen once per process: the loaded CRF model, the
  attribute-id lookup tables, compiled regexes.
- **`#[cfg(...)]`**: compile-time conditional inclusion. `#[cfg(test)]`
  means "only compile this when running `cargo test`"; `#[cfg(feature =
  "bench-timing")]` means "only compile this when that Cargo feature is
  turned on." Code behind a `cfg` you're not building simply doesn't exist
  in the binary.
- **Modules (`mod`)**: `mod features;` in `lib.rs` pulls in `features.rs` as
  a submodule. `pub`, `pub(crate)`, and no modifier control visibility:
  public to everyone, public within this crate only, or private to the
  module, respectively.

## 2. Suggested reading order

### `Cargo.toml`

Start here just to see the shape of the project: what depends on what, and
the two Cargo *features* (`polars-plugin`, `bench-timing`) that turn parts
of the code on or off at compile time. Note `crfs = { path = "vendor/crfs" }`
-- this crate's CRF inference engine is vendored source, not a crates.io
dependency (more on why in section 3).

### `src/tokenize.rs`

A gentle warm-up. It splits an address string into tokens with one regex
(plus a small helper regex that strips a match's leading junk), and
normalizes HTML ampersand entities with another. The long module doc
explains why the regex can't be a literal copy of Python's.
Notice the module-level tests at the bottom (`#[cfg(test)] mod tests`) --
every file in this crate keeps its tests alongside the code they test,
which is the normal Rust convention (as opposed to a separate `tests/`
folder, which this project also has, for a different purpose -- see
section 2's note on `tests/`).

### `src/lexicon.rs` (617 lines, almost all data)

Two big generated sets of strings (street-type abbreviations, directionals)
used by feature extraction. `phf::phf_set!` builds a *perfect hash function*
at compile time -- a lookup table with no runtime construction cost and no
collisions, generated by a proc macro. You don't need to read the data, just
know it exists and is regenerated from upstream, never hand-edited (see the
file's header).

### `src/features.rs` (the conceptual core, ~500 lines)

This is where "given a token, what does the CRF model actually see" is
decided. Read the module doc at the top first -- it explains the attribute
encoding scheme (string values, booleans, nested prefixes) that the rest of
the file implements. Then read top-to-bottom:

- `token_features` builds the nine raw features for one token (is it an
  abbreviation, how many digits does it have, does it end in punctuation,
  etc.) -- a straight port of the Python original, including its quirks
  (see the comments on `is_digit_str` and `has_vowels` for two examples of
  "this looks like a bug but it's intentional fidelity to upstream").
- `tokens_to_features` combines every token's own features with its
  neighbors' (`previous:`/`next:` prefixed) into the final per-token
  attribute list the CRF model scores against. It's `#[cfg(test)]`: the
  readable reference, not what runs in production.
- Everything below the `// id-based path` marker is the faster
  implementation of the same thing that actually runs, which section 3
  explains.

### `src/attr_cache.rs` and `src/bounded_ids.rs` (an optimization pair)

Skip these on a first pass if you just want to understand *what* this crate
does -- come back once `features.rs` makes sense. Together they're the
answer to "how do we avoid the CPU cost of converting attribute names to
strings and hashing them, for every token, on every parse." `attr_cache.rs`
is the general fallback (hash every possible attribute name once, at
startup); `bounded_ids.rs` is the specialization (most attributes come from
a small fixed set of possible names, so precompute a lookup table indexed
by an enum/integer instead of a string). This two-tier structure is the
result of iterating on performance in stages; `vendor/crfs/VENDORED.md`'s
"Provenance in this repo" section has the blow-by-blow of why it happened
in two steps instead of one.

### `vendor/crfs/`

This is a vendored (copied into the repo, not pulled from crates.io) fork of
a pure-Rust CRFsuite-format reader. You don't need to read all of it, but
it's worth understanding the two concepts it implements, since everything
upstream of it exists to feed this:

- **Conditional Random Field (CRF)**: a statistical model that labels a
  *sequence* of tokens jointly, rather than one token at a time in
  isolation -- so "is this token a `StreetName`" can depend on what the
  neighboring tokens were labeled, not just the token itself.
- **Viterbi decoding** (`Tagger::tag_ids`): the algorithm that finds the
  single most likely label sequence for a token sequence, efficiently
  (without literally trying every possible sequence).
- **Forward-backward / marginals** (`Tagger::tag_ids_with_marginals`): a
  related algorithm that additionally computes *confidence* -- how much
  more likely the winning sequence was than the alternatives -- which is
  what powers `tag_address_with_confidence`/`parse_address_with_confidence`.

`vendor/crfs/VENDORED.md` documents exactly what was changed relative to the
upstream `crfs` crate and why -- read it once you're comfortable with the
two concepts above.

### `src/lib.rs`

The crate's public API and the glue between tokenizing, feature extraction,
and CRF tagging. Read `Parser` and its methods (`parse`, `tag`,
`parse_with_confidence`, `tag_with_confidence`) -- `parse` gives you one
label per token; `tag` additionally collapses consecutive same-labeled
tokens into a `HashMap<String, String>` (e.g. all the `StreetName` tokens
joined into one string) and classifies the address type. `tag_components`
is the same with a fixed array indexed like `LABELS` instead of a map --
what the plugin uses, since it avoids hashing per row. The `_confidence`
variants are the same, plus a probability from the CRF's marginals. Then
`collapse` (private, at the bottom) is the actual token-merging logic `tag`
delegates to -- worth reading closely since it's the one function with real
business logic (intersections, PO boxes) rather than mechanical translation.

### `src/expressions.rs`

The Polars plugin surface: four `#[polars_expr]`-annotated functions (one
per public Python-facing function) that each take a column of address
strings and return a column of parsed results. If you're not familiar with
Polars internals this will read as more foreign than the rest of the crate
-- the short version is: build one `Parser` per parallel worker (via
Rayon's `map_init`), run every row through it, then assemble the results
into the specific columnar shapes (`Struct` for `tag_address`, `List<Struct>`
for `parse_address`) Polars expects back.

### `tests/`, `examples/` and `tools/`

`tests/fixtures.json` (hand-picked edge cases, with upstream's attributes)
and `tests/corpus_fixtures.json` (upstream's outputs on a 20k-address
synthetic corpus) are golden vectors generated by `tools/gen_fixtures.py`;
several files' tests load them via `include_str!`. `tests/parity.rs`
checks the Rust API against them, `tests/test_plugin.py` checks the Polars
plugin against them. `examples/bench_split.rs` measures where time goes at
runtime, `examples/dump_outputs.rs` dumps every output bit-for-bit so a
performance change can be shown to be output-neutral, and
`tools/bench_per_core.py` is the per-core throughput comparison against
Python. Each one's doc comment explains how to run it.

## 3. Why there's an "id-based path" next to a "string-based path"

You'll notice `features.rs` has two ways of doing almost the same thing
(`tokens_to_features` vs `tokens_to_id_features`), and `lib.rs`'s `Parser`
only ever calls the id-based one. This is the biggest structural quirk in
the codebase, so it's worth understanding once rather than re-deriving it
from scattered comments:

1. The CRF model scores *attribute ids* (small integers), not strings. The
   straightforward way to get from a token to an id is: build the
   attribute's name as a string (e.g. `"digits:all_digits"`), then look
   that string up in the model's vocabulary.
2. Building and hashing a string for every one of a token's ~9 attributes,
   for every token, on every parse, is measurable overhead at scale.
3. `bounded_ids.rs` sidesteps it: most attribute names come from a small,
   fixed, enumerable set (a handful of prefixes times a handful of values),
   so their ids can be precomputed into a small array *once*, at startup,
   and looked up by array index instead of built and hashed per call.
   `attr_cache.rs` is the fallback for the few attributes that are genuinely
   free text (the token itself, for `word`/`endsinpunc`).
4. `tokens_to_features` (the string-based path) didn't go away: it's kept
   as the "obviously correct, easy to reason about" reference
   implementation, and a test (`id_features_match_string_features_exactly`
   in `features.rs`) checks the fast path produces bit-for-bit identical
   output against it over the whole fixture corpus, every time you run
   `cargo test`. That test existing is *why* the fast path can be trusted:
   if you ever touch `bounded_ids.rs` or `encode_ids_into`, that's the test
   that will catch a mistake.

## 4. Running things

```sh
# Run every Rust test (unit tests in src/ plus tests/parity.rs).
# --no-default-features skips the Python/Polars plugin layer.
cargo test --no-default-features

# Run the parity suite against upstream Python's output specifically
cargo test --no-default-features --test parity

# Everything, including the plugin tests (builds the extension first)
make test

# Measure feature-extraction vs. CRF-tagging time (see the file's own doc
# comment for corpus options)
cargo run --release --no-default-features --features bench-timing --example bench_split

# Build the Python extension in-place for local testing (needs maturin)
uv run maturin develop --release
```

## 5. If you get stuck

The fastest way to understand any one function here is to find its test (or
its upstream Python equivalent, in the `usaddress` PyPI package) and read
inputs/outputs side by side with the implementation -- this crate leans
much more on "the tests demonstrate the behavior" than on prose, by design.
