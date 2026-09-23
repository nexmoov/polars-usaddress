//! Rust port of the DataMade `usaddress` US address parser, exposed as a
//! Polars plugin.
//!
//! The trained CRF model is the *unmodified* `usaddr.crfsuite` file shipped in
//! the upstream Python wheel, embedded at compile time. We reimplement only the
//! tokenizer and the feature extraction around it; inference is handled by
//! `crfs`, a pure-Rust reimplementation of the CRFsuite format.
//!
//! Because the feature code and the model must agree exactly, the model file
//! and `src/lexicon.rs` are versioned together and regenerated as a pair.

mod attr_cache;
mod bounded_ids;
mod features;
mod lexicon;
mod tokenize;

#[cfg(feature = "polars-plugin")]
mod expressions;

use std::collections::HashMap;
use std::sync::LazyLock;

use crfs::Model;

/// The trained model, byte-identical to the one in usaddress 0.5.16.
pub const MODEL_BYTES: &[u8] = include_bytes!("../models/usaddr.crfsuite");

/// usaddress version the embedded model and lexicons were taken from.
pub const UPSTREAM_VERSION: &str = "0.5.16";

static MODEL: LazyLock<Model<'static>> =
    LazyLock::new(|| Model::new(MODEL_BYTES).expect("embedded CRF model is valid"));

/// Diagnostic-only timing split between feature extraction and CRF tagging,
/// used by `examples/bench_split.rs` to find out which one is actually the
/// bottleneck relative to Python. No effect on the real build: only compiled
/// in with `--features bench-timing`, off by default.
#[cfg(feature = "bench-timing")]
pub mod bench_timing {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static FEATURE_NS: AtomicU64 = AtomicU64::new(0);
    pub static TAG_NS: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        FEATURE_NS.store(0, Ordering::Relaxed);
        TAG_NS.store(0, Ordering::Relaxed);
    }

    /// `(feature_extraction_ns, crf_tagging_ns)` accumulated since the last [`reset`].
    pub fn snapshot() -> (u64, u64) {
        (
            FEATURE_NS.load(Ordering::Relaxed),
            TAG_NS.load(Ordering::Relaxed),
        )
    }
}

/// The 29 labels the embedded CRF model can actually emit, plus the 7
/// `Second*` variants that [`tag`] can synthesise for the far side of an
/// intersection (one per model label whose name contains `"StreetName"` --
/// see the substring test in [`collapse`]).
///
/// This is the full set of keys [`tag`] can ever produce, and therefore the
/// struct schema the Polars plugin exposes.
///
/// Note this is **not** the same as the `usaddress.LABELS` constant in the
/// upstream Python package: that constant only lists 26 labels and omits
/// three the model can genuinely produce -- `CountryName`, `ZipPlus4`, and
/// `StreetNamePostModifier` (confirmed by dumping the label alphabet
/// straight from the pinned `usaddr.crfsuite` model via
/// `pycrfsuite.Tagger().labels()`, which returns 29, not 26). A schema built
/// from `usaddress.LABELS` silently drops any row the model tags with one of
/// those three -- e.g. `usaddress.tag("O'Hare.")` produces
/// `{"CountryName": "O'Hare."}` upstream, which would have nowhere to go.
pub const LABELS: [&str; 36] = [
    "AddressNumberPrefix",
    "AddressNumber",
    "AddressNumberSuffix",
    "StreetNamePreModifier",
    "StreetNamePreDirectional",
    "StreetNamePreType",
    "StreetName",
    "StreetNamePostType",
    "StreetNamePostDirectional",
    "StreetNamePostModifier",
    "SubaddressType",
    "SubaddressIdentifier",
    "BuildingName",
    "OccupancyType",
    "OccupancyIdentifier",
    "CornerOf",
    "LandmarkName",
    "PlaceName",
    "StateName",
    "ZipCode",
    "ZipPlus4",
    "CountryName",
    "USPSBoxType",
    "USPSBoxID",
    "USPSBoxGroupType",
    "USPSBoxGroupID",
    "IntersectionSeparator",
    "Recipient",
    "NotAddress",
    // Synthesised by `tag` once an IntersectionSeparator has been seen, for
    // every base label above whose name contains "StreetName".
    "SecondStreetNamePreModifier",
    "SecondStreetNamePreDirectional",
    "SecondStreetNamePreType",
    "SecondStreetName",
    "SecondStreetNamePostType",
    "SecondStreetNamePostDirectional",
    "SecondStreetNamePostModifier",
];

/// What kind of address `tag` decided it was looking at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressType {
    StreetAddress,
    Intersection,
    PoBox,
    Ambiguous,
}

impl AddressType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StreetAddress => "Street Address",
            Self::Intersection => "Intersection",
            Self::PoBox => "PO Box",
            Self::Ambiguous => "Ambiguous",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CRF tagging failed: {0}")]
    Crf(#[from] std::io::Error),

    /// Upstream raises `RepeatedLabelError` here. A label reappeared after a
    /// different label intervened, so the components cannot be collapsed into a
    /// flat mapping without losing information.
    #[error("label {label:?} appeared more than once, non-consecutively")]
    RepeatedLabel {
        label: String,
        parsed: Vec<(String, String)>,
    },
}

/// One slot per entry of [`LABELS`], in the same order: the collapsed
/// component text for that label, or `None` if the address had none.
pub type Components = [Option<String>; LABELS.len()];

/// Model label id -> position in [`LABELS`], precomputed once so collapsing
/// a row is array indexing rather than string compares and hashing.
struct LabelIndex {
    /// `LABELS` index of each model label.
    base: Vec<usize>,
    /// `LABELS` index of its `Second*` variant, for labels whose name contains
    /// "StreetName" (upstream's substring test, which also catches
    /// `StreetNamePreType` and friends).
    second: Vec<Option<usize>>,
    address_number: u32,
    usps_box_id: u32,
    intersection_separator: u32,
}

static LABEL_INDEX: LazyLock<LabelIndex> = LazyLock::new(|| {
    let names: Vec<&str> = (0..MODEL.num_labels())
        .map(|id| MODEL.label_name(id))
        .collect();
    let in_labels = |name: &str| {
        LABELS
            .iter()
            .position(|l| *l == name)
            .unwrap_or_else(|| panic!("label {name:?} missing from LABELS"))
    };
    let model_id = |name: &str| {
        MODEL
            .to_label_id(name)
            .unwrap_or_else(|| panic!("model has no {name:?} label"))
    };
    LabelIndex {
        base: names.iter().map(|n| in_labels(n)).collect(),
        second: names
            .iter()
            .map(|n| {
                n.contains("StreetName")
                    .then(|| in_labels(&format!("Second{n}")))
            })
            .collect(),
        address_number: model_id("AddressNumber"),
        usps_box_id: model_id("USPSBoxID"),
        intersection_separator: model_id("IntersectionSeparator"),
    }
});

/// A reusable tagger. Constructing one allocates working buffers, so create it
/// **once per chunk/thread** and reuse it across rows -- doing it per row gives
/// back a large share of the speedup this crate exists for.
///
/// Not `Sync`; give each Rayon worker its own.
pub struct Parser {
    tagger: crfs::Tagger<'static>,
}

impl Parser {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            tagger: MODEL.tagger()?,
        })
    }

    /// Tokenise an already-normalised address and label it: tokens borrow
    /// from `normalised`, labels are model label ids.
    fn label<'a>(&mut self, normalised: &'a str) -> (Vec<&'a str>, Vec<u32>) {
        let tokens = tokenize::tokenize(normalised);
        if tokens.is_empty() {
            return (tokens, Vec::new());
        }

        #[cfg(feature = "bench-timing")]
        let t0 = std::time::Instant::now();
        let id_seq = features::tokens_to_id_features(&tokens);
        #[cfg(feature = "bench-timing")]
        bench_timing::FEATURE_NS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        #[cfg(feature = "bench-timing")]
        let t1 = std::time::Instant::now();
        let label_ids = self.tagger.tag_ids(&id_seq);
        #[cfg(feature = "bench-timing")]
        bench_timing::TAG_NS.fetch_add(
            t1.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        (tokens, label_ids)
    }

    /// Low-level parse: one label per token, in order. Mirrors `usaddress.parse`.
    pub fn parse(&mut self, address: &str) -> Result<Vec<(String, String)>, Error> {
        let normalised = tokenize::normalize(address);
        let (tokens, label_ids) = self.label(&normalised);
        Ok(labelled_pairs(&tokens, &label_ids))
    }

    /// Collapse consecutive same-labelled tokens into one component per label,
    /// and classify the address. Mirrors `usaddress.tag`.
    pub fn tag(&mut self, address: &str) -> Result<(HashMap<String, String>, AddressType), Error> {
        let (components, address_type) = self.tag_components(address)?;
        Ok((to_map(components), address_type))
    }

    /// Same result as [`Parser::tag`], as a fixed array indexed like
    /// [`LABELS`] instead of a `HashMap`. What the Polars plugin uses: no
    /// hashing or label-string allocation per row.
    pub fn tag_components(&mut self, address: &str) -> Result<(Components, AddressType), Error> {
        let normalised = tokenize::normalize(address);
        let (tokens, label_ids) = self.label(&normalised);
        collapse(&tokens, &label_ids)
    }

    /// Like [`Parser::parse`], but each token also carries the CRF's marginal
    /// probability for the label it was actually given. Runs a
    /// forward-backward pass in addition to Viterbi decoding, so it costs
    /// more; reach for `parse` when you don't need the confidence.
    pub fn parse_with_confidence(
        &mut self,
        address: &str,
    ) -> Result<Vec<(String, String, f64)>, Error> {
        let normalised = tokenize::normalize(address);
        let tokens = tokenize::tokenize(&normalised);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let id_seq = features::tokens_to_id_features(&tokens);
        let marginals = self.tagger.tag_ids_with_marginals(&id_seq);

        Ok(tokens
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let label_id = marginals.labels[i];
                let confidence = marginals.marginal(i, label_id);
                (
                    (*t).to_string(),
                    MODEL.label_name(label_id).to_string(),
                    confidence,
                )
            })
            .collect())
    }

    /// Like [`Parser::tag`], but also returns the CRF's confidence in the
    /// *whole* label sequence it produced: `exp(sequence_score - log Z)`, one
    /// number for the address rather than per component (see
    /// [`Parser::parse_with_confidence`] for per-token confidence). Empty
    /// input returns confidence `1.0`, matching `tag`'s own empty-input
    /// handling. Costs more than `tag`: runs a forward-backward pass in
    /// addition to Viterbi decoding.
    pub fn tag_with_confidence(
        &mut self,
        address: &str,
    ) -> Result<(HashMap<String, String>, AddressType, f64), Error> {
        let (components, address_type, confidence) =
            self.tag_components_with_confidence(address)?;
        Ok((to_map(components), address_type, confidence))
    }

    /// [`Parser::tag_with_confidence`], returning [`Components`] like
    /// [`Parser::tag_components`].
    pub fn tag_components_with_confidence(
        &mut self,
        address: &str,
    ) -> Result<(Components, AddressType, f64), Error> {
        let normalised = tokenize::normalize(address);
        let tokens = tokenize::tokenize(&normalised);
        if tokens.is_empty() {
            let (components, address_type) = collapse(&[], &[])?;
            return Ok((components, address_type, 1.0));
        }

        let id_seq = features::tokens_to_id_features(&tokens);
        let marginals = self.tagger.tag_ids_with_marginals(&id_seq);
        let (components, address_type) = collapse(&tokens, &marginals.labels)?;
        Ok((components, address_type, marginals.sequence_probability()))
    }
}

/// Convenience wrapper that builds a throwaway [`Parser`].
///
/// Fine for one-off calls; use [`Parser`] directly in a loop.
pub fn parse(address: &str) -> Result<Vec<(String, String)>, Error> {
    Parser::new()?.parse(address)
}

/// Convenience wrapper that builds a throwaway [`Parser`].
pub fn tag(address: &str) -> Result<(HashMap<String, String>, AddressType), Error> {
    Parser::new()?.tag(address)
}

fn labelled_pairs(tokens: &[&str], label_ids: &[u32]) -> Vec<(String, String)> {
    tokens
        .iter()
        .zip(label_ids)
        .map(|(t, &lid)| ((*t).to_string(), MODEL.label_name(lid).to_string()))
        .collect()
}

fn to_map(components: Components) -> HashMap<String, String> {
    LABELS
        .iter()
        .zip(components)
        .filter_map(|(label, text)| text.map(|text| ((*label).to_string(), text)))
        .collect()
}

/// Collapse a token/label-id sequence into components. Port of the body of
/// `usaddress.tag`.
fn collapse(tokens: &[&str], label_ids: &[u32]) -> Result<(Components, AddressType), Error> {
    let ix = &*LABEL_INDEX;
    let mut components: Components = std::array::from_fn(|_| None);
    let mut last: Option<usize> = None;
    let mut is_intersection = false;
    let mut saw_address_number = false;
    let mut saw_usps_box_id = false;

    for (&token, &lid) in tokens.iter().zip(label_ids) {
        if lid == ix.intersection_separator {
            is_intersection = true;
        }
        // Everything after the separator that is street-name-ish belongs to
        // the second street.
        let idx = match ix.second[lid as usize] {
            Some(second) if is_intersection => second,
            _ => ix.base[lid as usize],
        };
        // Upstream checks these on the labels before any tag mapping; neither
        // can be Second-prefixed, so the raw model label is the same thing.
        saw_address_number |= lid == ix.address_number;
        saw_usps_box_id |= lid == ix.usps_box_id;

        match &mut components[idx] {
            Some(text) if last == Some(idx) => {
                text.push(' ');
                text.push_str(token);
            }
            slot @ None => *slot = Some(token.to_string()),
            Some(_) => {
                return Err(Error::RepeatedLabel {
                    label: LABELS[idx].to_string(),
                    parsed: labelled_pairs(tokens, label_ids),
                });
            }
        }
        last = Some(idx);
    }

    // Upstream joins with " " and then strips " ,;" from both ends.
    for text in components.iter_mut().flatten() {
        let trimmed = text.trim_matches([' ', ',', ';']);
        if trimmed.len() != text.len() {
            *text = trimmed.to_string();
        }
    }

    let address_type = if saw_address_number && !is_intersection {
        AddressType::StreetAddress
    } else if is_intersection && !saw_address_number {
        AddressType::Intersection
    } else if saw_usps_box_id {
        AddressType::PoBox
    } else {
        AddressType::Ambiguous
    };

    Ok((components, address_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse` and `parse_with_confidence` must decode the same Viterbi path,
    /// and every confidence value must be a legitimate probability.
    #[test]
    fn parse_with_confidence_matches_parse_and_stays_in_bounds() {
        let addresses = [
            "123 Main St",
            "123 N Main St Apt 4B Springfield IL 62704",
            "PO Box 123",
        ];
        for addr in addresses {
            let plain = Parser::new()
                .expect("embedded model loads")
                .parse(addr)
                .expect("parse succeeds");
            let confident = Parser::new()
                .expect("embedded model loads")
                .parse_with_confidence(addr)
                .expect("parse_with_confidence succeeds");

            assert_eq!(
                plain.len(),
                confident.len(),
                "token count differs for {addr:?}"
            );
            for ((token, label), (c_token, c_label, confidence)) in
                plain.iter().zip(confident.iter())
            {
                assert_eq!(token, c_token);
                assert_eq!(label, c_label);
                assert!(
                    (0.0..=1.0).contains(confidence),
                    "confidence {confidence} for {c_token:?}/{c_label:?} out of [0, 1]"
                );
            }
        }
    }

    /// `tag_with_confidence` must agree with `tag` on components and address
    /// type (confidence is additive, not a different decoding), and stay in
    /// `[0, 1]`.
    #[test]
    fn tag_with_confidence_matches_tag_and_stays_in_bounds() {
        let addresses = [
            "123 Main St",
            "123 N Main St Apt 4B Springfield IL 62704",
            "PO Box 123",
        ];
        for addr in addresses {
            let (plain_tagged, plain_type) = Parser::new()
                .expect("embedded model loads")
                .tag(addr)
                .expect("tag succeeds");
            let (confident_tagged, confident_type, confidence) = Parser::new()
                .expect("embedded model loads")
                .tag_with_confidence(addr)
                .expect("tag_with_confidence succeeds");

            assert_eq!(
                plain_tagged, confident_tagged,
                "components differ for {addr:?}"
            );
            assert_eq!(
                plain_type, confident_type,
                "address type differs for {addr:?}"
            );
            assert!(
                (0.0..=1.0).contains(&confidence),
                "sequence confidence {confidence} for {addr:?} out of [0, 1]"
            );
        }
    }

    #[test]
    fn tag_with_confidence_on_empty_input_is_ambiguous_and_fully_confident() {
        let (tagged, address_type, confidence) = Parser::new()
            .expect("embedded model loads")
            .tag_with_confidence("")
            .expect("empty input is not an error");
        assert!(tagged.is_empty());
        assert_eq!(address_type, AddressType::Ambiguous);
        assert_eq!(confidence, 1.0);
    }
}
