//! Port of `usaddress.tokenFeatures` / `tokens2features`, plus the attribute
//! encoding `python-crfsuite`'s `ItemSequence` applies on the way into the
//! tagger (recovered from the reference implementation, not documented
//! upstream):
//!
//! * string value  -> attribute named `key:value`, weight 1.0
//! * bool value    -> attribute named `key`, weight 1.0 if true, **0.0 if false**
//!                    (emitted either way -- not omitted)
//! * nested dict   -> every child attribute prefixed with `parent:`
//!
//! An empty string value still produces the colon, e.g. `trailing.zeros:`.
//! Getting any of this subtly wrong does not error; it silently degrades the
//! parse, which is what the `tests/parity.rs` fixtures guard against.
//!
//! [`tokens_to_features`] builds the upstream-shaped `Vec<Attribute>`.
//! [`tokens_to_id_features`] is the fast path `Parser::parse` actually
//! calls -- same features, resolved straight to `(attr_id, weight)` pairs via
//! `bounded_ids` -- and is checked against `tokens_to_features` for exact
//! equivalence in this module's tests.

use crfs::Attribute;

use crate::bounded_ids;
use crate::lexicon::{DIRECTIONS, STREET_NAMES};

const SKIP_ZERO_WEIGHT_ATTRS: bool = false;

/// Which of the three enumerable digit classes a token falls into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DigitsClass {
    AllDigits,
    SomeDigits,
    NoDigits,
}

impl DigitsClass {
    fn as_str(self) -> &'static str {
        match self {
            DigitsClass::AllDigits => "all_digits",
            DigitsClass::SomeDigits => "some_digits",
            DigitsClass::NoDigits => "no_digits",
        }
    }
}

/// The nine per-token features, before `next:`/`previous:` nesting.
///
/// String-valued features are `Option<&str>`/`String` where `None` means the
/// upstream dict held `False` rather than a string.
#[derive(Debug, Clone)]
pub struct TokenFeatures {
    abbrev: bool,
    digits: DigitsClass,
    /// `None` when the token is all digits (upstream stores `False`).
    word: Option<String>,
    /// `None` when the token is not all digits (upstream stores `False`).
    trailing_zeros: Option<String>,
    /// `(is_word, char count)`. Upstream renders this as `"w:<n>"`/`"d:<n>"`;
    /// kept as data so `encode_ids_into` can index `bounded_ids` directly.
    length: (bool, usize),
    /// `None` when the token has no interior punctuation; otherwise the token's
    /// final character.
    endsinpunc: Option<String>,
    directional: bool,
    street_name: bool,
    has_vowels: bool,
}

/// Python's `\w` under `re.UNICODE`: alphanumerics plus underscore.
#[inline]
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Python's `str.isdigit()`. Diverges from Python only on non-ASCII digits
/// (e.g. Arabic-Indic numerals), which don't occur in US address data.
#[inline]
fn is_digit_str(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Port of `usaddress.digits`. Note that upstream's `some_digits` branch tests
/// against `string.digits`, i.e. ASCII only, even though the `all_digits`
/// branch uses the Unicode-aware `isdigit`.
fn digits_class(clean: &str) -> DigitsClass {
    if is_digit_str(clean) {
        DigitsClass::AllDigits
    } else if clean.chars().any(|c| c.is_ascii_digit()) {
        DigitsClass::SomeDigits
    } else {
        DigitsClass::NoDigits
    }
}

/// Port of the `re.sub(r"(^[\W]*)|([^.\w]*$)", "", token)` cleanup: strip
/// leading non-word characters, then trailing characters that are neither a dot
/// nor a word character.
fn clean_token(token: &str) -> &str {
    if matches!(token, "&" | "#" | "½") {
        return token;
    }
    token
        .trim_start_matches(|c: char| !is_word_char(c))
        .trim_end_matches(|c: char| c != '.' && !is_word_char(c))
}

/// Port of `usaddress.tokenFeatures`.
pub fn token_features(token: &str) -> TokenFeatures {
    let clean = clean_token(token);
    let abbrev_str: String = clean.to_lowercase().replace('.', "");

    let numeric = is_digit_str(&abbrev_str);

    let (word, trailing_zeros) = if numeric {
        let trimmed = abbrev_str.trim_end_matches('0');
        let zeros = abbrev_str[trimmed.len()..].to_owned();
        (None, Some(zeros))
    } else {
        (Some(abbrev_str.clone()), None)
    };

    let length = (!numeric, abbrev_str.chars().count());

    // Upstream: `token[-1] if re.match(r".+[^.\w]", token) else False`. The
    // `[^.\w]` class also matches `\n`, so this is: does any character at
    // index >= 1 fail to be a dot or word character?
    let endsinpunc = token
        .chars()
        .skip(1)
        .any(|c| c != '.' && !is_word_char(c))
        .then(|| {
            token
                .chars()
                .next_back()
                .map(|c| c.to_string())
                .unwrap_or_default()
        });

    // `has.vowels` deliberately skips the first character, matching
    // `set(token_abbrev[1:]) & set("aeiou")`.
    let has_vowels = abbrev_str
        .chars()
        .skip(1)
        .any(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u'));

    TokenFeatures {
        abbrev: clean.ends_with('.'),
        digits: digits_class(clean),
        word,
        trailing_zeros,
        length,
        endsinpunc,
        directional: DIRECTIONS.contains(abbrev_str.as_str()),
        street_name: STREET_NAMES.contains(abbrev_str.as_str()),
        has_vowels,
    }
}

#[inline]
fn push_bool(out: &mut Vec<Attribute>, prefix: &str, key: &str, value: bool) {
    if !value && SKIP_ZERO_WEIGHT_ATTRS {
        return;
    }
    out.push(Attribute::new(
        format!("{prefix}{key}"),
        if value { 1.0 } else { 0.0 },
    ));
}

#[inline]
fn push_str(out: &mut Vec<Attribute>, prefix: &str, key: &str, value: &str) {
    out.push(Attribute::new(format!("{prefix}{key}:{value}"), 1.0));
}

impl TokenFeatures {
    /// Emit this token's nine features with the given prefix (`""`, `"next:"`
    /// or `"previous:"`), in upstream's dict-insertion order.
    fn encode_into(&self, out: &mut Vec<Attribute>, prefix: &str) {
        push_bool(out, prefix, "abbrev", self.abbrev);
        push_str(out, prefix, "digits", self.digits.as_str());
        match &self.word {
            Some(w) => push_str(out, prefix, "word", w),
            None => push_bool(out, prefix, "word", false),
        }
        match &self.trailing_zeros {
            Some(z) => push_str(out, prefix, "trailing.zeros", z),
            None => push_bool(out, prefix, "trailing.zeros", false),
        }
        let (is_word, count) = self.length;
        let length_str = format!("{}:{count}", if is_word { 'w' } else { 'd' });
        push_str(out, prefix, "length", &length_str);
        match &self.endsinpunc {
            Some(c) => push_str(out, prefix, "endsinpunc", c),
            None => push_bool(out, prefix, "endsinpunc", false),
        }
        push_bool(out, prefix, "directional", self.directional);
        push_bool(out, prefix, "street_name", self.street_name);
        push_bool(out, prefix, "has.vowels", self.has_vowels);
    }
}

/// Port of `usaddress.tokens2features`, flattened.
///
/// Upstream builds nested dicts with shallow copies, so `next:` and `previous:`
/// carry only the nine base features -- never a recursive `next:next:`. The two
/// positional flags land in exactly one place each:
///
/// * `previous:address.start` appears only at index 1
/// * `next:address.end` appears only at index `n - 2`
pub fn tokens_to_features(tokens: &[&str]) -> Vec<Vec<Attribute>> {
    let base: Vec<TokenFeatures> = tokens.iter().map(|t| token_features(t)).collect();
    let n = base.len();

    (0..n)
        .map(|i| {
            let mut attrs = Vec::with_capacity(29); // 9 own + 9 next + 9 previous + up to 2 flags

            base[i].encode_into(&mut attrs, "");
            if i == 0 {
                attrs.push(Attribute::new("address.start".to_string(), 1.0));
            }
            if i + 1 == n {
                attrs.push(Attribute::new("address.end".to_string(), 1.0));
            }
            if i > 0 {
                base[i - 1].encode_into(&mut attrs, "previous:");
                if i == 1 {
                    attrs.push(Attribute::new("previous:address.start".to_string(), 1.0));
                }
            }
            if i + 1 < n {
                base[i + 1].encode_into(&mut attrs, "next:");
                if i + 2 == n {
                    attrs.push(Attribute::new("next:address.end".to_string(), 1.0));
                }
            }
            attrs
        })
        .collect()
}

// ------------------------------------------------------------- id-based path

#[inline]
fn push_fixed(out: &mut Vec<(u32, f64)>, id: Option<u32>, value: bool) {
    if !value && SKIP_ZERO_WEIGHT_ATTRS {
        return;
    }
    if let Some(id) = id {
        out.push((id, if value { 1.0 } else { 0.0 }));
    }
}

#[inline]
fn push_present(out: &mut Vec<(u32, f64)>, id: Option<u32>) {
    if let Some(id) = id {
        out.push((id, 1.0));
    }
}

/// Fallback: format the name exactly as `push_str` would, then resolve it
/// through `attr_cache`'s full-vocabulary hash map.
#[inline]
fn push_slow(out: &mut Vec<(u32, f64)>, prefix: &str, key: &str, value: &str) {
    if let Some(id) = crate::attr_cache::resolve_one(&format!("{prefix}{key}:{value}")) {
        out.push((id, 1.0));
    }
}

impl TokenFeatures {
    /// Same nine features, same order, as [`TokenFeatures::encode_into`], but
    /// resolved straight to `(attr_id, weight)` pairs via `bounded_ids`.
    /// `prefix_idx` indexes `bounded_ids::PREFIXES` (0 = own, 1 = `previous:`,
    /// 2 = `next:`) and must agree with the string `prefix` used below.
    fn encode_ids_into(&self, out: &mut Vec<(u32, f64)>, prefix_idx: usize) {
        let t = &bounded_ids::TABLES;
        let prefix = bounded_ids::PREFIXES[prefix_idx];

        push_fixed(out, t.abbrev(prefix_idx), self.abbrev);

        let class_idx = self.digits as usize;
        push_present(out, t.digits_id(prefix_idx, class_idx));

        match &self.word {
            Some(w) => push_slow(out, prefix, "word", w),
            None => push_fixed(out, t.word_false(prefix_idx), false),
        }

        match &self.trailing_zeros {
            Some(z) => match t.trailing_zeros_some_id(prefix_idx, z.len()) {
                Some(cached) => push_present(out, cached),
                None => push_slow(out, prefix, "trailing.zeros", z),
            },
            None => push_fixed(out, t.trailing_zeros_false(prefix_idx), false),
        }

        let (is_word, count) = self.length;
        match t.length_id(prefix_idx, is_word, count) {
            Some(cached) => push_present(out, cached),
            None => {
                let length_str = format!("{}:{count}", if is_word { 'w' } else { 'd' });
                push_slow(out, prefix, "length", &length_str);
            }
        }

        match &self.endsinpunc {
            Some(c) => push_slow(out, prefix, "endsinpunc", c),
            None => push_fixed(out, t.endsinpunc_false(prefix_idx), false),
        }

        push_fixed(out, t.directional(prefix_idx), self.directional);
        push_fixed(out, t.street_name(prefix_idx), self.street_name);
        push_fixed(out, t.has_vowels(prefix_idx), self.has_vowels);
    }
}

/// Id-based equivalent of [`tokens_to_features`]; what `Parser::parse`
/// actually calls.
pub fn tokens_to_id_features(tokens: &[&str]) -> Vec<Vec<(u32, f64)>> {
    let base: Vec<TokenFeatures> = tokens.iter().map(|t| token_features(t)).collect();
    let n = base.len();
    let t = &bounded_ids::TABLES;

    (0..n)
        .map(|i| {
            let mut ids = Vec::with_capacity(29);

            base[i].encode_ids_into(&mut ids, 0);
            if i == 0 {
                push_present(&mut ids, t.address_start);
            }
            if i + 1 == n {
                push_present(&mut ids, t.address_end);
            }
            if i > 0 {
                base[i - 1].encode_ids_into(&mut ids, 1);
                if i == 1 {
                    push_present(&mut ids, t.previous_address_start);
                }
            }
            if i + 1 < n {
                base[i + 1].encode_ids_into(&mut ids, 2);
                if i + 2 == n {
                    push_present(&mut ids, t.next_address_end);
                }
            }
            ids
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(attrs: &[Attribute]) -> Vec<String> {
        let mut v: Vec<String> = attrs
            .iter()
            .map(|a| format!("{}={}", a.name, a.value))
            .collect();
        v.sort();
        v
    }

    #[test]
    fn empty_string_value_still_gets_a_colon() {
        let f = token_features("123");
        let mut out = Vec::new();
        f.encode_into(&mut out, "");
        assert!(names(&out).contains(&"trailing.zeros:=1".to_string()));
    }

    #[test]
    fn false_booleans_are_emitted_with_zero_weight() {
        let f = token_features("123");
        let mut out = Vec::new();
        f.encode_into(&mut out, "");
        assert!(names(&out).contains(&"word=0".to_string()));
    }

    #[test]
    fn trailing_zeros_are_captured() {
        let f = token_features("1200");
        assert_eq!(f.trailing_zeros.as_deref(), Some("00"));
        let f = token_features("123");
        assert_eq!(f.trailing_zeros.as_deref(), Some(""));
    }

    #[test]
    fn has_vowels_skips_the_first_character() {
        assert!(token_features("Ave").has_vowels);
        assert!(!token_features("A").has_vowels);
    }

    #[test]
    fn endsinpunc_matches_newlines() {
        let f = token_features("St.\n");
        assert_eq!(f.endsinpunc.as_deref(), Some("\n"));
    }

    #[test]
    fn positional_flags_land_in_one_place_each() {
        let tokens = ["123", "Main", "St", "Chicago"];
        let seq = tokens_to_features(&tokens);

        let has = |i: usize, name: &str| seq[i].iter().any(|a| a.name == name);

        assert!(has(0, "address.start"));
        assert!(has(1, "previous:address.start"));
        assert!(!has(2, "previous:address.start"));

        assert!(has(3, "address.end"));
        assert!(has(2, "next:address.end"));
        assert!(!has(1, "next:address.end"));
    }

    #[test]
    fn nesting_is_not_recursive() {
        let seq = tokens_to_features(&["123", "Main", "St"]);
        assert!(!seq
            .iter()
            .flatten()
            .any(|a| a.name.contains("next:next:") || a.name.contains("previous:previous:")));
    }

    /// `tokens_to_id_features` must match `tokens_to_features` resolved
    /// through `attr_cache::resolve_one` -- same ids, weights, and order.
    /// Order matters: state scores are f64 sums, and float addition isn't
    /// strictly associative, so a reordering could change a Viterbi
    /// tie-break. Checked against every address in the parity fixtures.
    #[test]
    fn id_features_match_string_features_exactly() {
        use crate::attr_cache;

        #[derive(serde::Deserialize)]
        struct Case {
            input: String,
        }
        #[derive(serde::Deserialize)]
        struct Fixtures {
            fixtures: Vec<Case>,
        }
        let fixtures: Fixtures =
            serde_json::from_str(include_str!("../tests/fixtures.json")).expect("fixtures.json parses");

        let mut checked_any = false;
        for case in &fixtures.fixtures {
            let normalised = crate::tokenize::normalize(&case.input);
            let tokens = crate::tokenize::tokenize(&normalised);
            if tokens.is_empty() {
                continue;
            }
            checked_any = true;

            let string_features = tokens_to_features(&tokens);
            let expected: Vec<Vec<(u32, f64)>> = string_features
                .iter()
                .map(|attrs| {
                    attrs
                        .iter()
                        .filter_map(|a| attr_cache::resolve_one(&a.name).map(|id| (id, a.value)))
                        .collect()
                })
                .collect();

            let got = tokens_to_id_features(&tokens);

            assert_eq!(
                got, expected,
                "id-based and string-based feature resolution diverged for input {:?}",
                case.input
            );
        }
        assert!(checked_any, "fixtures.json produced no non-empty token sequences");
    }
}
