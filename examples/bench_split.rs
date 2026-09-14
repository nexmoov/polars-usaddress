//! Diagnostic: how much of `Parser::tag`'s time is feature extraction vs. CRF
//! tagging? Companion to `tools/benchmark.ipynb`'s Python-side split of the
//! same question.
//!
//! ```text
//! cargo run --release --no-default-features --features bench-timing --example bench_split
//! ```
//!
//! By default this cycles the ~40 addresses in `tests/fixtures.json` out to a
//! benchmark-sized corpus -- fine for a quick check, but not the same corpus
//! `tools/benchmark.ipynb` uses, so totals aren't directly comparable to the
//! notebook's numbers. Pass a JSON array of addresses as an argument (e.g.
//! exported from the notebook) to compare apples-to-apples:
//!
//! ```text
//! cargo run --release --no-default-features --features bench-timing --example bench_split -- /path/to/corpus.json
//! ```

use std::time::Instant;

use polars_usaddress::Parser;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    input: String,
}

#[derive(Deserialize)]
struct Fixtures {
    fixtures: Vec<Case>,
}

/// Cycle the ~40 fixture addresses out to a benchmark-sized corpus. Only used
/// when no corpus file is given on the command line.
fn fallback_corpus(n: usize) -> Vec<String> {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("../tests/fixtures.json")).expect("fixtures.json parses");
    let addresses: Vec<String> = fixtures
        .fixtures
        .into_iter()
        .map(|c| c.input)
        .filter(|s| !s.trim().is_empty())
        .collect();
    assert!(
        !addresses.is_empty(),
        "no non-empty fixture addresses to cycle"
    );
    (0..n)
        .map(|i| addresses[i % addresses.len()].clone())
        .collect()
}

fn main() {
    let arg = std::env::args().nth(1);
    let (corpus, source): (Vec<String>, &str) = match &arg {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("couldn't read corpus file {path:?}: {e}"));
            let corpus: Vec<String> = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{path:?} isn't a JSON array of strings: {e}"));
            (corpus, "given corpus file")
        }
        None => (
            fallback_corpus(20_000),
            "fixtures.json, cycled (pass a corpus file for an apples-to-apples comparison -- see this file's doc comment)",
        ),
    };
    let n = corpus.len();
    let corpus: Vec<&str> = corpus.iter().map(String::as_str).collect();

    let mut parser = Parser::new().expect("embedded model loads");

    // Warm-up: don't let one-time costs land in the timed region.
    let _ = parser.tag(corpus[0]);
    #[cfg(feature = "bench-timing")]
    polars_usaddress::bench_timing::reset();

    let start = Instant::now();
    for addr in &corpus {
        let _ = parser.tag(addr);
    }
    let total = start.elapsed();

    println!("n = {n} ({source})");
    println!(
        "total:               {:>8.3} ms  ({:>10.0} addrs/s)",
        total.as_secs_f64() * 1e3,
        n as f64 / total.as_secs_f64()
    );

    #[cfg(feature = "bench-timing")]
    {
        let (feat_ns, tag_ns) = polars_usaddress::bench_timing::snapshot();
        let split_total = (feat_ns + tag_ns) as f64;
        println!(
            "  feature extraction: {:>8.3} ms  ({:>5.1}% of tokenize+features+tag)",
            feat_ns as f64 / 1e6,
            100.0 * feat_ns as f64 / split_total
        );
        println!(
            "  crf tagging:        {:>8.3} ms  ({:>5.1}% of tokenize+features+tag)",
            tag_ns as f64 / 1e6,
            100.0 * tag_ns as f64 / split_total
        );
    }
    #[cfg(not(feature = "bench-timing"))]
    println!("(run with --features bench-timing for the feature/tagging split)");
}
