//! Port of `usaddress.tokenize`.
//!
//! Upstream's pattern is:
//!
//! ```text
//! \(*\b[^\s,;#&()]+[.,;)\n]*   |   [#&]
//! ```
//!
//! That does *not* map directly onto the `regex` crate, despite having no
//! lookaround or backreferences. Python's `\b`/`\w` are Unicode-aware via
//! `str.isalnum()`, which is true for any character with a Unicode numeric
//! value -- including vulgar fractions like '½' (category `No`, "Number,
//! other"). The `regex` crate's Unicode `\w` is narrower: `\p{Alphabetic} +
//! \p{M} + \p{Nd} + \p{Pc} + \p{Join_Control}`, using `Nd` ("Decimal number")
//! only. So a literal port of `\b[^\s,;#&()]+` never sees a boundary in front
//! of a leading '½' -- the character is silently skipped, never becomes a
//! token, and everything after it in the address shifts by one position
//! through feature extraction and the CRF.
//!
//! The fix below drops `\b` and spells out the same "skip leading junk,
//! then start the token at the first word-ish character" behaviour using an
//! explicit character class that matches Python's broader definition:
//! `\p{L} + \p{N} + \p{M} + \p{Pc}` (using `\p{N}`, the general Number
//! category covering `Nd`/`Nl`/`No` together, not just `Nd`). Because the
//! `regex` crate has no lookaround, the "junk to skip" and "token to keep"
//! have to be two different parts of the same match: a non-capturing prefix
//! that consumes leading junk (dropped), followed by a capturing group for
//! the real token (kept). `[#&]` still needs its own capture group so the
//! two alternatives can be told apart after the fact.

use regex::Regex;
use std::borrow::Cow;
use std::sync::LazyLock;

static RE_TOKENS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[^\s,;#&()\p{L}\p{N}\p{M}\p{Pc}]*(\(*[\p{L}\p{N}\p{M}\p{Pc}][^\s,;#&()]*[.,;)\n]*)|([#&])")
        .expect("tokenizer regex is valid")
});

/// The leading-junk prefix of an `RE_TOKENS` match: exactly its first,
/// un-grouped part.
static RE_JUNK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[^\s,;#&()\p{L}\p{N}\p{M}\p{Pc}]+").expect("junk regex is valid")
});

/// Group 1 or 2 of an `RE_TOKENS` match, recovered from the whole match
/// without asking the regex for capture groups (which forces its slower
/// engine). The junk prefix can't contain `(`, `#`, `&` or a word character,
/// so stripping it lands exactly where the kept group starts; a first byte
/// that is one of those means there's no junk at all.
fn strip_junk(m: &str) -> &str {
    let first = m.as_bytes()[0];
    if first.is_ascii_alphanumeric() || matches!(first, b'_' | b'(' | b'#' | b'&') {
        return m;
    }
    RE_JUNK.find(m).map_or(m, |junk| &m[junk.end()..])
}

static RE_AMP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(&#38;)|(&amp;)").expect("ampersand regex is valid"));

/// Normalise HTML ampersand entities, exactly as upstream does before tokenising.
/// Borrows when there is nothing to rewrite, which is the common case.
pub fn normalize(address: &str) -> Cow<'_, str> {
    RE_AMP.replace_all(address, "&")
}

/// Split an (already [`normalize`]d) address string into tokens.
///
/// Group 1 is the real token (leading junk consumed by the un-grouped
/// prefix is discarded); group 2 is the standalone `#`/`&` alternative.
/// Exactly one of the two participates in any given match; see
/// [`strip_junk`] for how it's recovered from the whole match.
pub fn tokenize(address: &str) -> Vec<&str> {
    RE_TOKENS
        .find_iter(address)
        .map(|m| strip_junk(m.as_str()))
        .collect()
}

/// The original capture-group implementation, kept as the reference the
/// fast path is tested against.
#[cfg(test)]
fn tokenize_with_captures(address: &str) -> Vec<&str> {
    RE_TOKENS
        .captures_iter(address)
        .filter_map(|caps| caps.get(1).or_else(|| caps.get(2)))
        .map(|m| m.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        let n = normalize(s);
        tokenize(&n).into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn basic() {
        assert_eq!(
            toks("123 Main St. Suite 100 Chicago, IL 60601"),
            [
                "123", "Main", "St.", "Suite", "100", "Chicago,", "IL", "60601"
            ]
        );
    }

    #[test]
    fn parens_are_kept() {
        assert_eq!(toks("(123) Main St"), ["(123)", "Main", "St"]);
    }

    #[test]
    fn html_entities_become_ampersand() {
        assert_eq!(
            toks("123 Main St &amp; Oak Ave"),
            ["123", "Main", "St", "&", "Oak", "Ave"]
        );
        assert_eq!(
            toks("123 Main St &#38; Oak Ave"),
            ["123", "Main", "St", "&", "Oak", "Ave"]
        );
    }

    #[test]
    fn normalize_borrows_when_no_entity() {
        assert!(matches!(normalize("123 Main St"), Cow::Borrowed(_)));
    }

    #[test]
    fn degenerate_inputs_yield_nothing() {
        for s in ["", "   ", ",,,"] {
            assert!(toks(s).is_empty(), "expected no tokens for {s:?}");
        }
    }

    #[test]
    fn bare_symbols_are_tokens() {
        assert_eq!(toks("#"), ["#"]);
        assert_eq!(toks("&"), ["&"]);
    }

    /// Regression test: a leading vulgar fraction (Unicode category `No`) is
    /// a word character to Python's `\b`/`\w`, but not to the `regex`
    /// crate's narrower Unicode `\w`. A literal port of upstream's
    /// `\b[^\s,;#&()]+` silently drops it as the first token.
    #[test]
    fn leading_fraction_is_not_dropped() {
        assert_eq!(toks("½ Main St"), ["½", "Main", "St"]);
        assert_eq!(toks("¼ Main St"), ["¼", "Main", "St"]);
        assert_eq!(toks("½ Ave.."), ["½", "Ave.."]);
    }

    /// A fraction fused onto a preceding digit (no standalone-token edge
    /// case) already worked before the fix; keep it covered too.
    #[test]
    fn fraction_fused_to_digit() {
        assert_eq!(toks("123\u{bd} Main St"), ["123½", "Main", "St"]);
    }

    /// `find_iter` + `strip_junk` must give exactly what the capture-group
    /// version gives, on awkward prefixes (non-ASCII junk, ASCII junk,
    /// combining marks, parens after junk) and on the whole parity corpus.
    #[test]
    fn find_iter_matches_captures() {
        let awkward = [
            "°abc",
            "(°abc)",
            "°(abc)",
            "--12 Main",
            "'O'Hare",
            "½ Main",
            "\u{301}123 Main",
            "＿123 Main",
            "#4 & 5",
            "***",
            "°",
            "..(a",
            "123\u{1f}Main St",
            "",
        ];
        for s in awkward {
            assert_eq!(tokenize(s), tokenize_with_captures(s), "for {s:?}");
        }

        #[derive(serde::Deserialize)]
        struct Case {
            input: String,
        }
        #[derive(serde::Deserialize)]
        struct Corpus {
            fixtures: Vec<Case>,
        }
        let corpus: Corpus =
            serde_json::from_str(include_str!("../tests/corpus_fixtures.json")).unwrap();
        for case in &corpus.fixtures {
            let n = normalize(&case.input);
            assert_eq!(
                tokenize(&n),
                tokenize_with_captures(&n),
                "for {:?}",
                case.input
            );
        }
    }

    /// Leading junk that isn't a fraction (arbitrary symbols) should still
    /// be skipped, same as upstream.
    #[test]
    fn leading_junk_symbols_are_skipped() {
        assert_eq!(toks("°abc"), ["abc"]);
        assert_eq!(toks("(°abc)"), ["abc)"]);
        assert_eq!(toks("°(abc)"), ["(abc)"]);
    }
}
