//! Polars expression entry points.
//!
//! Two output shapes, because they suit different downstream work:
//!
//! * [`parse_address`] -> `List[Struct{token, label}]`, faithful to the raw
//!   per-token labelling. Nothing is merged or dropped.
//! * [`tag_address`]   -> `Struct{<32 label fields>, address_type}`, the
//!   ergonomic one: `.struct.field("ZipCode")` and you are done.

use std::collections::HashMap;

use polars::prelude::*;
use polars_core::chunked_array::builder::list::AnonymousOwnedListBuilder;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;

use crate::{Error, LABELS, Parser};

fn new_worker_parser() -> Parser {
    Parser::new().expect("embedded CRF model already validated to load")
}

// ---------------------------------------------------------------- tag_address

fn tag_output(_: &[Field]) -> PolarsResult<Field> {
    let mut fields: Vec<Field> = LABELS
        .iter()
        .map(|name| Field::new((*name).into(), DataType::String))
        .collect();
    fields.push(Field::new("address_type".into(), DataType::String));
    Ok(Field::new("address".into(), DataType::Struct(fields)))
}

/// Parse into one nullable string field per address component.
///
/// Rows that fail -- empty input, or upstream's `RepeatedLabelError` -- come
/// back as all-null components. `address_type` still reports `"Ambiguous"` for
/// empty input, matching upstream, but is null for a genuine error so the two
/// cases stay distinguishable.
#[polars_expr(output_type_func=tag_output)]
fn tag_address(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let len = ca.len();

    // Surface a real load failure here, once, rather than inside `map_init`
    // on some worker thread.
    Parser::new().map_err(|e| polars_err!(ComputeError: "failed to load address model: {e}"))?;

    // One `Parser` per Rayon task
    // reused across every address that task is handed. `None` means "no
    // components for this row": a null input, or a repeated-label address
    // upstream would also have rejected.
    let tagged: Vec<Option<(HashMap<String, String>, &'static str)>> = ca
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(
            new_worker_parser,
            |parser, opt| -> PolarsResult<Option<(HashMap<String, String>, &'static str)>> {
                match opt {
                    None => Ok(None),
                    Some(s) => match parser.tag(s) {
                        Ok((components, t)) => Ok(Some((components, t.as_str()))),
                        Err(Error::RepeatedLabel { .. }) => Ok(None),
                        Err(e) => Err(polars_err!(ComputeError: "address tagging failed: {e}")),
                    },
                }
            },
        )
        .collect::<PolarsResult<Vec<_>>>()?;

    let mut builders: Vec<Vec<Option<String>>> =
        LABELS.iter().map(|_| Vec::with_capacity(len)).collect();
    let mut types: Vec<Option<&'static str>> = Vec::with_capacity(len);

    for row in tagged {
        match row {
            Some((mut components, addr_type)) => {
                for (i, label) in LABELS.iter().enumerate() {
                    builders[i].push(components.remove(*label));
                }
                types.push(Some(addr_type));
            }
            None => {
                for builder in &mut builders {
                    builder.push(None);
                }
                types.push(None);
            }
        }
    }

    let mut fields: Vec<Series> = builders
        .into_iter()
        .zip(LABELS.iter())
        .map(|(vals, name)| {
            StringChunked::from_iter_options((*name).into(), vals.into_iter()).into_series()
        })
        .collect();
    fields.push(
        StringChunked::from_iter_options("address_type".into(), types.into_iter()).into_series(),
    );

    StructChunked::from_series("address".into(), len, fields.iter()).map(|ca| ca.into_series())
}

// ------------------------------------------------------- tag_address_with_confidence

fn tag_confidence_output(_: &[Field]) -> PolarsResult<Field> {
    let mut fields: Vec<Field> = LABELS
        .iter()
        .map(|name| Field::new((*name).into(), DataType::String))
        .collect();
    fields.push(Field::new("address_type".into(), DataType::String));
    fields.push(Field::new("sequence_confidence".into(), DataType::Float64));
    Ok(Field::new("address".into(), DataType::Struct(fields)))
}

/// One row's result from `tag_with_confidence`: components, address type,
/// sequence confidence.
type ConfidentTagRow = (HashMap<String, String>, &'static str, f64);

/// Like [`tag_address`], with one extra `sequence_confidence` field: the
/// CRF's confidence in the whole label sequence for that row, not per
/// component (see [`parse_address_with_confidence`] for per-token
/// confidence). Null rows behave exactly like `tag_address`. Costs more:
/// runs a forward-backward pass per row in addition to Viterbi decoding.
#[polars_expr(output_type_func=tag_confidence_output)]
fn tag_address_with_confidence(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let len = ca.len();

    Parser::new().map_err(|e| polars_err!(ComputeError: "failed to load address model: {e}"))?;

    let tagged: Vec<Option<ConfidentTagRow>> = ca
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(
            new_worker_parser,
            |parser, opt| -> PolarsResult<Option<ConfidentTagRow>> {
                match opt {
                    None => Ok(None),
                    Some(s) => match parser.tag_with_confidence(s) {
                        Ok((components, t, confidence)) => {
                            Ok(Some((components, t.as_str(), confidence)))
                        }
                        Err(Error::RepeatedLabel { .. }) => Ok(None),
                        Err(e) => Err(polars_err!(ComputeError: "address tagging failed: {e}")),
                    },
                }
            },
        )
        .collect::<PolarsResult<Vec<_>>>()?;

    let mut builders: Vec<Vec<Option<String>>> =
        LABELS.iter().map(|_| Vec::with_capacity(len)).collect();
    let mut types: Vec<Option<&'static str>> = Vec::with_capacity(len);
    let mut confidences: Vec<Option<f64>> = Vec::with_capacity(len);

    for row in tagged {
        match row {
            Some((mut components, addr_type, confidence)) => {
                for (i, label) in LABELS.iter().enumerate() {
                    builders[i].push(components.remove(*label));
                }
                types.push(Some(addr_type));
                confidences.push(Some(confidence));
            }
            None => {
                for builder in &mut builders {
                    builder.push(None);
                }
                types.push(None);
                confidences.push(None);
            }
        }
    }

    let mut fields: Vec<Series> = builders
        .into_iter()
        .zip(LABELS.iter())
        .map(|(vals, name)| {
            StringChunked::from_iter_options((*name).into(), vals.into_iter()).into_series()
        })
        .collect();
    fields.push(
        StringChunked::from_iter_options("address_type".into(), types.into_iter()).into_series(),
    );
    fields.push(
        Float64Chunked::from_iter_options("sequence_confidence".into(), confidences.into_iter())
            .into_series(),
    );

    StructChunked::from_series("address".into(), len, fields.iter()).map(|ca| ca.into_series())
}

// -------------------------------------------------------------- parse_address

fn parse_output(_: &[Field]) -> PolarsResult<Field> {
    let inner = DataType::Struct(vec![
        Field::new("token".into(), DataType::String),
        Field::new("label".into(), DataType::String),
    ]);
    Ok(Field::new(
        "address".into(),
        DataType::List(Box::new(inner)),
    ))
}

/// Raw per-token labelling, preserving order and repetition.
#[polars_expr(output_type_func=parse_output)]
fn parse_address(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    Parser::new().map_err(|e| polars_err!(ComputeError: "failed to load address model: {e}"))?;

    let parsed_rows: Vec<Option<Vec<(String, String)>>> = ca
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(
            new_worker_parser,
            |parser, opt| -> PolarsResult<Option<Vec<(String, String)>>> {
                match opt {
                    None => Ok(None),
                    Some(s) => parser
                        .parse(s)
                        .map(Some)
                        .map_err(|e| polars_err!(ComputeError: "address parsing failed: {e}")),
                }
            },
        )
        .collect::<PolarsResult<Vec<_>>>()?;

    // Flat token/label buffers plus per-row offsets, assembled into a
    // ListChunked at the end -- avoids building one Series per row.
    let mut tokens: Vec<String> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut offsets: Vec<i64> = Vec::with_capacity(ca.len() + 1);
    let mut validity: Vec<bool> = Vec::with_capacity(ca.len());
    offsets.push(0);

    for row in parsed_rows {
        match row {
            None => validity.push(false),
            Some(parsed) => {
                for (t, l) in parsed {
                    tokens.push(t);
                    labels.push(l);
                }
                validity.push(true);
            }
        }
        offsets.push(tokens.len() as i64);
    }

    let inner = StructChunked::from_series(
        "".into(),
        tokens.len(),
        [
            StringChunked::from_iter_values("token".into(), tokens.into_iter()).into_series(),
            StringChunked::from_iter_values("label".into(), labels.into_iter()).into_series(),
        ]
        .iter(),
    )?
    .into_series();

    // polars-core >=0.55.2 dropped `ListChunked::from_iter_and_offsets`, so a
    // list column is built via a builder that takes one Series per row --
    // `inner` stays one flat struct Series for the whole column, sliced here.
    let mut builder =
        AnonymousOwnedListBuilder::new("address".into(), ca.len(), Some(inner.dtype().clone()));
    for (i, valid) in validity.iter().enumerate() {
        if *valid {
            let start = offsets[i];
            let len = (offsets[i + 1] - start) as usize;
            builder.append_series(&inner.slice(start, len))?;
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish().into_series())
}

// ------------------------------------------------------- parse_address_with_confidence

fn parse_confidence_output(_: &[Field]) -> PolarsResult<Field> {
    let inner = DataType::Struct(vec![
        Field::new("token".into(), DataType::String),
        Field::new("label".into(), DataType::String),
        Field::new("confidence".into(), DataType::Float64),
    ]);
    Ok(Field::new(
        "address".into(),
        DataType::List(Box::new(inner)),
    ))
}

/// Like [`parse_address`], but each token also carries the CRF's marginal
/// probability for the label it was actually given. Costs more: also runs a
/// forward-backward pass per row.
#[polars_expr(output_type_func=parse_confidence_output)]
fn parse_address_with_confidence(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    Parser::new().map_err(|e| polars_err!(ComputeError: "failed to load address model: {e}"))?;

    let parsed_rows: Vec<Option<Vec<(String, String, f64)>>> = ca
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(
            new_worker_parser,
            |parser, opt| -> PolarsResult<Option<Vec<(String, String, f64)>>> {
                match opt {
                    None => Ok(None),
                    Some(s) => parser
                        .parse_with_confidence(s)
                        .map(Some)
                        .map_err(|e| polars_err!(ComputeError: "address parsing failed: {e}")),
                }
            },
        )
        .collect::<PolarsResult<Vec<_>>>()?;

    // Same flat-buffers-plus-offsets shape as `parse_address`.
    let mut tokens: Vec<String> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut confidences: Vec<f64> = Vec::new();
    let mut offsets: Vec<i64> = Vec::with_capacity(ca.len() + 1);
    let mut validity: Vec<bool> = Vec::with_capacity(ca.len());
    offsets.push(0);

    for row in parsed_rows {
        match row {
            None => validity.push(false),
            Some(parsed) => {
                for (t, l, c) in parsed {
                    tokens.push(t);
                    labels.push(l);
                    confidences.push(c);
                }
                validity.push(true);
            }
        }
        offsets.push(tokens.len() as i64);
    }

    let inner = StructChunked::from_series(
        "".into(),
        tokens.len(),
        [
            StringChunked::from_iter_values("token".into(), tokens.into_iter()).into_series(),
            StringChunked::from_iter_values("label".into(), labels.into_iter()).into_series(),
            Float64Chunked::from_iter_values("confidence".into(), confidences.into_iter())
                .into_series(),
        ]
        .iter(),
    )?
    .into_series();

    let mut builder =
        AnonymousOwnedListBuilder::new("address".into(), ca.len(), Some(inner.dtype().clone()));
    for (i, valid) in validity.iter().enumerate() {
        if *valid {
            let start = offsets[i];
            let len = (offsets[i + 1] - start) as usize;
            builder.append_series(&inner.slice(start, len))?;
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish().into_series())
}
