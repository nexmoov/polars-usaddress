//! Attribute-name -> id cache, built once from the model's fixed vocabulary
//! (see the crate root and `vendor/crfs/VENDORED.md` for why this exists).

use std::sync::LazyLock;

use rustc_hash::FxHashMap;

use crate::MODEL;
use crate::bounded_ids::PREFIXES;

/// Every attribute string the embedded model knows, mapped to its id.
static ATTR_IDS: LazyLock<FxHashMap<String, u32>> = LazyLock::new(|| {
    let n = MODEL.num_attrs();
    let mut map: FxHashMap<String, u32> =
        FxHashMap::with_capacity_and_hasher(n as usize, Default::default());
    for id in 0..n {
        if let Some(name) = MODEL.to_attr(id) {
            map.insert(name.to_string(), id);
        }
    }
    map
});

/// One attribute's model id from each viewpoint, indexed like
/// `bounded_ids::PREFIXES`: `[own, previous:, next:]`.
pub(crate) type ViewIds = [Option<u32>; 3];

/// For a free-text feature `key`, every value the model saw, mapped to the ids
/// of `{prefix}{key}:{value}` for each of the three prefixes.
fn ids_by_value(key: &str) -> FxHashMap<Box<str>, ViewIds> {
    let mut map: FxHashMap<Box<str>, ViewIds> = FxHashMap::default();
    for id in 0..MODEL.num_attrs() {
        let Some(name) = MODEL.to_attr(id) else {
            continue;
        };
        for (p, prefix) in PREFIXES.iter().enumerate() {
            let value = name
                .strip_prefix(prefix)
                .and_then(|rest| rest.strip_prefix(key))
                .and_then(|rest| rest.strip_prefix(':'));
            if let Some(value) = value {
                map.entry(value.into()).or_insert([None; 3])[p] = Some(id);
            }
        }
    }
    map
}

static WORD_IDS: LazyLock<FxHashMap<Box<str>, ViewIds>> = LazyLock::new(|| ids_by_value("word"));
static ENDSINPUNC_IDS: LazyLock<FxHashMap<Box<str>, ViewIds>> =
    LazyLock::new(|| ids_by_value("endsinpunc"));

/// Ids of `word:{word}` from all three viewpoints, in one lookup.
pub(crate) fn word_ids(word: &str) -> ViewIds {
    WORD_IDS.get(word).copied().unwrap_or([None; 3])
}

/// Ids of `endsinpunc:{c}` from all three viewpoints, in one lookup.
pub(crate) fn endsinpunc_ids(c: &str) -> ViewIds {
    ENDSINPUNC_IDS.get(c).copied().unwrap_or([None; 3])
}

/// Resolve one attribute name to its model id, or `None` if the model never
/// saw it during training. Fallback path for out-of-range `length` and
/// `trailing.zeros` counts.
pub(crate) fn resolve_one(name: &str) -> Option<u32> {
    ATTR_IDS.get(name).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The by-value tables must give exactly the id a full-name lookup gives.
    #[test]
    fn value_tables_match_full_name_lookup() {
        for word in ["main", "st", "chicago", "suite", "not-a-word-the-model-saw"] {
            for (p, prefix) in PREFIXES.iter().enumerate() {
                assert_eq!(
                    word_ids(word)[p],
                    resolve_one(&format!("{prefix}word:{word}"))
                );
            }
        }
        for c in [",", ".", ")", "\n", ";"] {
            for (p, prefix) in PREFIXES.iter().enumerate() {
                assert_eq!(
                    endsinpunc_ids(c)[p],
                    resolve_one(&format!("{prefix}endsinpunc:{c}"))
                );
            }
        }
    }

    /// This cache must resolve every attribute name the same way the
    /// model's own `to_attr_id` would.
    #[test]
    fn matches_model_to_attr_id_for_real_feature_output() {
        let tokens = ["123", "Main", "St", "Chicago", "IL", "60614"];
        let seq = crate::features::tokens_to_features(&tokens);
        for token_attrs in &seq {
            for attr in token_attrs {
                let expected = MODEL.to_attr_id(&attr.name);
                let got = ATTR_IDS.get(&attr.name).copied();
                assert_eq!(
                    got, expected,
                    "attribute {:?}: cache said {:?}, model said {:?}",
                    attr.name, got, expected
                );
            }
        }
    }
}
