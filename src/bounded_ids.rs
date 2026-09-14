//! Precomputed attribute ids for every feature whose name is fully
//! determined by a small, enumerable domain (a prefix times a handful of
//! values), so `features::encode_ids_into` can index a small array instead
//! of building and hashing a name string. `word`/`endsinpunc`'s free-text
//! values and out-of-range `length`/`trailing.zeros` counts fall back to
//! `attr_cache::resolve_one` instead -- see the crate root for the bigger
//! picture.

use std::sync::LazyLock;

use crate::MODEL;

/// "own" (index 0), "previous:" (1), "next:" (2) -- the three viewpoints
/// `TokenFeatures::encode_into`/`encode_ids_into` emit from.
pub(crate) const PREFIXES: [&str; 3] = ["", "previous:", "next:"];

/// Longest token length this table covers; longer tokens still get a
/// correct answer via the slow path.
const MAX_LENGTH: usize = 48;

/// Longest run of trailing zeros the table covers, same reasoning.
const MAX_TRAILING_ZEROS: usize = 24;

fn id(name: String) -> Option<u32> {
    MODEL.to_attr_id(&name)
}

pub(crate) struct BoundedIds {
    abbrev: [Option<u32>; 3],
    directional: [Option<u32>; 3],
    street_name: [Option<u32>; 3],
    has_vowels: [Option<u32>; 3],
    /// `word`'s `None` (all-digits token) branch only; `Some(word)` is
    /// arbitrary text and stays on the slow path.
    word_false: [Option<u32>; 3],
    /// Same shape, for `trailing.zeros`'s `None` branch.
    trailing_zeros_false: [Option<u32>; 3],
    /// Same shape, for `endsinpunc`'s `None` branch.
    endsinpunc_false: [Option<u32>; 3],
    /// `[prefix][digits_class]` (0=all_digits, 1=some_digits, 2=no_digits).
    digits: [[Option<u32>; 3]; 3],
    /// `[prefix][is_word as usize][count]`. Outer `Option` (via `.get`)
    /// means "out of range, use the slow path"; inner `Option` means "in
    /// range, but never seen in training" -- see `length_id`.
    length: [[Vec<Option<u32>>; 2]; 3],
    /// `[prefix][zero_count]`. Same nested meaning as `length`.
    trailing_zeros_some: [Vec<Option<u32>>; 3],
    pub(crate) address_start: Option<u32>,
    pub(crate) previous_address_start: Option<u32>,
    pub(crate) address_end: Option<u32>,
    pub(crate) next_address_end: Option<u32>,
}

pub(crate) static TABLES: LazyLock<BoundedIds> = LazyLock::new(|| {
    let fixed = |key: &str| -> [Option<u32>; 3] {
        std::array::from_fn(|p| id(format!("{}{key}", PREFIXES[p])))
    };
    const CLASSES: [&str; 3] = ["all_digits", "some_digits", "no_digits"];
    let digits: [[Option<u32>; 3]; 3] = std::array::from_fn(|p| {
        std::array::from_fn(|c| id(format!("{}digits:{}", PREFIXES[p], CLASSES[c])))
    });
    let length: [[Vec<Option<u32>>; 2]; 3] = std::array::from_fn(|p| {
        let for_type = |type_char: char| -> Vec<Option<u32>> {
            (0..=MAX_LENGTH)
                .map(|n| id(format!("{}length:{type_char}:{n}", PREFIXES[p])))
                .collect()
        };
        [for_type('d'), for_type('w')]
    });
    let trailing_zeros_some: [Vec<Option<u32>>; 3] = std::array::from_fn(|p| {
        (0..=MAX_TRAILING_ZEROS)
            .map(|n| id(format!("{}trailing.zeros:{}", PREFIXES[p], "0".repeat(n))))
            .collect()
    });
    BoundedIds {
        abbrev: fixed("abbrev"),
        directional: fixed("directional"),
        street_name: fixed("street_name"),
        has_vowels: fixed("has.vowels"),
        word_false: fixed("word"),
        trailing_zeros_false: fixed("trailing.zeros"),
        endsinpunc_false: fixed("endsinpunc"),
        digits,
        length,
        trailing_zeros_some,
        address_start: id("address.start".to_string()),
        previous_address_start: id("previous:address.start".to_string()),
        address_end: id("address.end".to_string()),
        next_address_end: id("next:address.end".to_string()),
    }
});

impl BoundedIds {
    #[inline]
    pub(crate) fn abbrev(&self, prefix: usize) -> Option<u32> {
        self.abbrev[prefix]
    }
    #[inline]
    pub(crate) fn directional(&self, prefix: usize) -> Option<u32> {
        self.directional[prefix]
    }
    #[inline]
    pub(crate) fn street_name(&self, prefix: usize) -> Option<u32> {
        self.street_name[prefix]
    }
    #[inline]
    pub(crate) fn has_vowels(&self, prefix: usize) -> Option<u32> {
        self.has_vowels[prefix]
    }
    #[inline]
    pub(crate) fn word_false(&self, prefix: usize) -> Option<u32> {
        self.word_false[prefix]
    }
    #[inline]
    pub(crate) fn trailing_zeros_false(&self, prefix: usize) -> Option<u32> {
        self.trailing_zeros_false[prefix]
    }
    #[inline]
    pub(crate) fn endsinpunc_false(&self, prefix: usize) -> Option<u32> {
        self.endsinpunc_false[prefix]
    }

    /// Digits classes are fully enumerable (always exactly one of three), so
    /// unlike `length_id`/`trailing_zeros_some_id` there is no "out of
    /// range" case.
    #[inline]
    pub(crate) fn digits_id(&self, prefix: usize, class_idx: usize) -> Option<u32> {
        self.digits[prefix][class_idx]
    }

    /// `None` -> out of range, caller must fall back to the slow path.
    /// `Some(inner)` -> in range; `inner` is the (possibly absent) id.
    #[inline]
    pub(crate) fn length_id(
        &self,
        prefix: usize,
        is_word: bool,
        count: usize,
    ) -> Option<Option<u32>> {
        self.length[prefix][is_word as usize].get(count).copied()
    }

    /// Same in-range/out-of-range shape as `length_id`.
    #[inline]
    pub(crate) fn trailing_zeros_some_id(
        &self,
        prefix: usize,
        zero_count: usize,
    ) -> Option<Option<u32>> {
        self.trailing_zeros_some[prefix].get(zero_count).copied()
    }
}
