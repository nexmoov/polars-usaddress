//! Dump every output bit the Parser API produces for a corpus, one line per
//! address: labels, per-token confidences and sequence confidence, with f64s
//! written as raw bits so two dumps can be `diff`ed for *exact* equality.
//!
//! Used to show a performance change is bit-for-bit neutral:
//!
//! ```text
//! cargo run --release --no-default-features --example dump_outputs -- corpus.json > before.txt
//! # ...apply the change...
//! cargo run --release --no-default-features --example dump_outputs -- corpus.json > after.txt
//! diff before.txt after.txt
//! ```

use std::io::Write;

use polars_usaddress::Parser;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_outputs <corpus.json>");
    let text = std::fs::read_to_string(&path).expect("corpus file readable");
    let corpus: Vec<String> = serde_json::from_str(&text).expect("JSON array of strings");

    let mut parser = Parser::new().expect("embedded model loads");
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for addr in &corpus {
        let tokens = parser.parse_with_confidence(addr).expect("parse succeeds");
        let seq = match parser.tag_with_confidence(addr) {
            Ok((_, t, c)) => format!("{}:{:016x}", t.as_str(), c.to_bits()),
            Err(_) => "repeated-label".to_string(),
        };
        let toks: Vec<String> = tokens
            .iter()
            .map(|(_, label, c)| format!("{label}:{:016x}", c.to_bits()))
            .collect();
        writeln!(out, "{seq} | {}", toks.join(" ")).unwrap();
    }
}
