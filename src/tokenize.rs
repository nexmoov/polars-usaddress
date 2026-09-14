//! Port of `usaddress.tokenize`.
//!
//! The upstream pattern uses no lookaround or backreferences, so it maps
//! directly onto the `regex` crate:
//!
//! ```text
//! \(*\b[^\s,;#&()]+[.,;)\n]*   |   [#&]
//! ```

use regex::Regex;
use std::borrow::Cow;
use std::sync::LazyLock;

static RE_TOKENS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(*\b[^\s,;#&()]+[.,;)\n]*|[#&]").expect("tokenizer regex is valid")
});

static RE_AMP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(&#38;)|(&amp;)").expect("ampersand regex is valid"));

/// Normalise HTML ampersand entities, exactly as upstream does before tokenising.
/// Borrows when there is nothing to rewrite, which is the common case.
pub fn normalize(address: &str) -> Cow<'_, str> {
    RE_AMP.replace_all(address, "&")
}

/// Split an (already [`normalize`]d) address string into tokens.
pub fn tokenize(address: &str) -> Vec<&str> {
    RE_TOKENS.find_iter(address).map(|m| m.as_str()).collect()
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
}
