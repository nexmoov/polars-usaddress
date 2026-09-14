//! Attribute-name -> id cache, built once from the model's fixed vocabulary
//! (see the crate root and `vendor/crfs/VENDORED.md` for why this exists).

use std::sync::LazyLock;

use rustc_hash::FxHashMap;

use crate::MODEL;

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

/// Resolve one attribute name to its model id, or `None` if the model never
/// saw it during training. Fallback path for `bounded_ids`'s free-text
/// features and out-of-range counts.
pub(crate) fn resolve_one(name: &str) -> Option<u32> {
    ATTR_IDS.get(name).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

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
