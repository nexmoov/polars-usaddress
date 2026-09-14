//! Differential tests against golden vectors captured from the reference
//! Python implementation (usaddress 0.5.16).
//!
//! Regenerate with `python tools/gen_fixtures.py` after any upstream bump.
//! A failure here means the port has drifted from upstream -- which otherwise
//! shows up only as quietly wrong parses.

use std::collections::HashMap;

use polars_usaddress::{parse, tag, AddressType, Parser};
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

#[test]
fn empty_input_is_ambiguous_not_an_error() {
    for s in ["", "   ", ",,,"] {
        let (components, t) = tag(s).unwrap();
        assert!(components.is_empty());
        assert_eq!(t, AddressType::Ambiguous);
        assert!(parse(s).unwrap().is_empty());
    }
}
