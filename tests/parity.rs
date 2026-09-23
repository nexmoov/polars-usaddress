//! Differential tests against golden vectors captured from the reference
//! Python implementation (usaddress 0.5.16).
//!
//! Regenerate with `python tools/gen_fixtures.py` after any upstream bump.
//! A failure here means the port has drifted from upstream -- which otherwise
//! shows up only as quietly wrong parses.

use std::collections::HashMap;

use polars_usaddress::{AddressType, Parser, parse, tag};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    usaddress_version: String,
    fixtures: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    input: String,
    tokens: Vec<String>,
    parse: Vec<TokenLabel>,
    tag: Option<HashMap<String, String>>,
    address_type: Option<String>,
    repeated_label_error: bool,
}

#[derive(Deserialize)]
struct TokenLabel {
    token: String,
    label: String,
}

fn load() -> Fixtures {
    serde_json::from_str(include_str!("fixtures.json")).expect("fixtures.json parses")
}

#[test]
fn fixtures_match_the_pinned_upstream_version() {
    assert_eq!(load().usaddress_version, polars_usaddress::UPSTREAM_VERSION);
}

#[test]
fn tokenization_matches_upstream() {
    let mut parser = Parser::new().unwrap();
    for case in load().fixtures {
        let got: Vec<String> = parser
            .parse(&case.input)
            .unwrap()
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert_eq!(got, case.tokens, "tokenizing {:?}", case.input);
    }
}

#[test]
fn labels_match_upstream() {
    let mut parser = Parser::new().unwrap();
    for case in load().fixtures {
        let got = parser.parse(&case.input).unwrap();
        let want: Vec<(String, String)> = case
            .parse
            .into_iter()
            .map(|tl| (tl.token, tl.label))
            .collect();
        assert_eq!(got, want, "labelling {:?}", case.input);
    }
}

#[test]
fn tag_matches_upstream() {
    let mut parser = Parser::new().unwrap();
    for case in load().fixtures {
        match parser.tag(&case.input) {
            Ok((components, addr_type)) => {
                assert!(
                    !case.repeated_label_error,
                    "expected RepeatedLabelError for {:?}",
                    case.input
                );
                assert_eq!(
                    components,
                    case.tag.expect("non-error case has a tag"),
                    "components for {:?}",
                    case.input
                );
                assert_eq!(
                    addr_type.as_str(),
                    case.address_type.expect("non-error case has a type"),
                    "address type for {:?}",
                    case.input
                );
            }
            Err(polars_usaddress::Error::RepeatedLabel { .. }) => {
                assert!(
                    case.repeated_label_error,
                    "unexpected RepeatedLabelError for {:?}",
                    case.input
                );
            }
            Err(e) => panic!("unexpected error for {:?}: {e}", case.input),
        }
    }
}

/// `tests/corpus_fixtures.json`: upstream's outputs for every address in the
/// notebook's seeded 20k synthetic corpus. Outputs only (no attributes), so it
/// stays small enough to check on every `cargo test`.
#[derive(Deserialize)]
struct Corpus {
    usaddress_version: String,
    fixtures: Vec<CorpusCase>,
}

#[derive(Deserialize)]
struct CorpusCase {
    input: String,
    parse: Vec<(String, String)>,
    /// `None` exactly when upstream raised `RepeatedLabelError`.
    tag: Option<HashMap<String, String>>,
    address_type: Option<String>,
}

#[test]
fn corpus_matches_upstream() {
    let corpus: Corpus = serde_json::from_str(include_str!("corpus_fixtures.json"))
        .expect("corpus_fixtures.json parses");
    assert_eq!(corpus.usaddress_version, polars_usaddress::UPSTREAM_VERSION);
    assert!(corpus.fixtures.len() >= 10_000, "corpus unexpectedly small");

    let mut parser = Parser::new().unwrap();
    let mut failures = Vec::new();
    for case in &corpus.fixtures {
        let parsed = parser.parse(&case.input).unwrap();
        if parsed != case.parse {
            failures.push(format!("parse {:?}: got {parsed:?}", case.input));
            continue;
        }
        let got = match parser.tag(&case.input) {
            Ok((c, t)) => Some((c, t.as_str().to_string())),
            Err(polars_usaddress::Error::RepeatedLabel { .. }) => None,
            Err(e) => panic!("unexpected error for {:?}: {e}", case.input),
        };
        let want = case.tag.clone().zip(case.address_type.clone());
        if got != want {
            failures.push(format!("tag {:?}: got {got:?}, want {want:?}", case.input));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} corpus addresses diverge from upstream; first few:\n{}",
        failures.len(),
        corpus.fixtures.len(),
        failures[..failures.len().min(10)].join("\n")
    );
}

#[test]
fn empty_input_is_ambiguous_not_an_error() {
    for s in ["", "   ", ",,,"] {
        let (components, t) = tag(s).unwrap();
        assert!(components.is_empty());
        assert_eq!(t, AddressType::Ambiguous);
        assert!(parse(s).unwrap().is_empty());
    }
}
