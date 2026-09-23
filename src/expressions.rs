//! Polars expression entry points.
//!
//! Two output shapes, because they suit different downstream work:
//!
//! * [`parse_address`] -> `List[Struct{token, label}]`, faithful to the raw
//!   per-token labelling. Nothing is merged or dropped.
//! * [`tag_address`]   -> `Struct{<36 label fields>, address_type}`, the
//!   ergonomic one: `.struct.field("ZipCode")` and you are done.

use std::sync::LazyLock;

use polars::prelude::*;
use polars_arrow::array::ListArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::OffsetsBuffer;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;

use crate::{Components, Error, LABELS, Parser};

fn new_worker_parser() -> Parser {
    Parser::new().expect("embedded CRF model already validated to load")
}

/// Run `f` over every non-null row in parallel, one [`Parser`] per Rayon
/// task, reused across every address that task is handed. Null input rows
/// come back as `None` without calling `f`.
fn par_map_rows<T, F>(ca: &StringChunked, f: F) -> PolarsResult<Vec<Option<T>>>
where
    T: Send,
    F: Fn(&mut Parser, &str) -> PolarsResult<Option<T>> + Sync + Send,
{
    // Surface a real load failure here, once, rather than inside `map_init`
    // on some worker thread.
    Parser::new().map_err(|e| polars_err!(ComputeError: "failed to load address model: {e}"))?;

    POOL.install(|| {
        ca.iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map_init(new_worker_parser, |parser, opt| match opt {
                None => Ok(None),
                Some(s) => f(parser, s),
            })
            .collect()
    })
}

/// The plugin's own Rayon pool. This library carries its own copy of Rayon,
/// separate from the Polars it's loaded into, so Rayon's default global pool
/// would ignore Polars' `POLARS_MAX_THREADS` cap. Size it from that instead.
static POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let threads = crate::pool_size(std::env::var("POLARS_MAX_THREADS").ok().as_deref());
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("polars-usaddress-{i}"))
        .build()
        .expect("plugin thread pool builds")
});

/// One `String` column per entry of [`LABELS`], moving each component out of
/// its row rather than cloning it. `None` rows are null in every column.
fn label_columns<R>(
    rows: &mut [Option<R>],
    components: impl Fn(&mut R) -> &mut Components,
) -> Vec<Series> {
    LABELS
        .iter()
        .enumerate()
        .map(|(i, name)| {
            StringChunked::from_iter_options(
                (*name).into(),
                rows.iter_mut()
                    .map(|row| row.as_mut().and_then(|r| components(r)[i].take())),
            )
            .into_series()
        })
        .collect()
}

/// Wrap one flat `inner` column into a list column, row `i` being
/// `inner[offsets[i]..offsets[i + 1]]`, or null where `validity[i]` is false.
/// Zero-copy over `inner`: one Arrow `ListArray<i64>` over its single chunk.
fn list_column(inner: Series, offsets: Vec<i64>, validity: Vec<bool>) -> PolarsResult<Series> {
    let dtype = DataType::List(Box::new(inner.dtype().clone()));
    let inner = inner.rechunk();
    let values = inner.chunks()[0].clone();
    let arrow_dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
    let validity = (!validity.iter().all(|v| *v)).then(|| Bitmap::from_iter(validity));
    let array = ListArray::<i64>::new(
        arrow_dtype,
        OffsetsBuffer::try_from(offsets)?,
        values,
        validity,
    );
    // SAFETY: `values` is `inner`'s own physical chunk and `dtype` is `inner`'s
    // own dtype wrapped in `List`, so the array and the declared dtype agree.
    let ca = unsafe {
        ListChunked::from_chunks_and_dtype("address".into(), vec![Box::new(array)], dtype)
    };
    Ok(ca.into_series())
}

fn label_fields() -> Vec<Field> {
    LABELS
        .iter()
        .map(|name| Field::new((*name).into(), DataType::String))
        .collect()
}

// ---------------------------------------------------------------- tag_address

fn tag_output(_: &[Field]) -> PolarsResult<Field> {
    let mut fields = label_fields();
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

    // `None` means "no components for this row": a null input, or a
    // repeated-label address upstream would also have rejected.
    let mut rows = par_map_rows(ca, |parser, s| match parser.tag_components(s) {
        Ok((components, t)) => Ok(Some((components, t.as_str()))),
        Err(Error::RepeatedLabel { .. }) => Ok(None),
        Err(e) => Err(polars_err!(ComputeError: "address tagging failed: {e}")),
    })?;

    let mut fields = label_columns(&mut rows, |(components, _)| components);
    fields.push(
        StringChunked::from_iter_options(
            "address_type".into(),
            rows.iter().map(|row| row.as_ref().map(|(_, t)| *t)),
        )
        .into_series(),
    );

    StructChunked::from_series("address".into(), ca.len(), fields.iter()).map(|ca| ca.into_series())
}

// ------------------------------------------------------- tag_address_with_confidence

fn tag_confidence_output(_: &[Field]) -> PolarsResult<Field> {
    let mut fields = label_fields();
    fields.push(Field::new("address_type".into(), DataType::String));
    fields.push(Field::new("sequence_confidence".into(), DataType::Float64));
    Ok(Field::new("address".into(), DataType::Struct(fields)))
}

/// Like [`tag_address`], with one extra `sequence_confidence` field: the
/// CRF's confidence in the whole label sequence for that row, not per
/// component (see [`parse_address_with_confidence`] for per-token
/// confidence). Null rows behave exactly like `tag_address`. Costs more:
/// runs a forward-backward pass per row in addition to Viterbi decoding.
#[polars_expr(output_type_func=tag_confidence_output)]
fn tag_address_with_confidence(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    let mut rows = par_map_rows(ca, |parser, s| {
        match parser.tag_components_with_confidence(s) {
            Ok((components, t, confidence)) => Ok(Some((components, t.as_str(), confidence))),
            Err(Error::RepeatedLabel { .. }) => Ok(None),
            Err(e) => Err(polars_err!(ComputeError: "address tagging failed: {e}")),
        }
    })?;

    let mut fields = label_columns(&mut rows, |(components, _, _)| components);
    fields.push(
        StringChunked::from_iter_options(
            "address_type".into(),
            rows.iter().map(|row| row.as_ref().map(|(_, t, _)| *t)),
        )
        .into_series(),
    );
    fields.push(
        Float64Chunked::from_iter_options(
            "sequence_confidence".into(),
            rows.iter().map(|row| row.as_ref().map(|(_, _, c)| *c)),
        )
        .into_series(),
    );

    StructChunked::from_series("address".into(), ca.len(), fields.iter()).map(|ca| ca.into_series())
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

    let parsed_rows = par_map_rows(ca, |parser, s| {
        parser
            .parse_static(s)
            .map(Some)
            .map_err(|e| polars_err!(ComputeError: "address parsing failed: {e}"))
    })?;

    // Flat token/label buffers plus per-row offsets, assembled into a
    // ListChunked at the end -- no Series per row.
    let mut tokens: Vec<String> = Vec::new();
    let mut labels: Vec<&'static str> = Vec::new();
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

    list_column(inner, offsets, validity)
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

    let parsed_rows = par_map_rows(ca, |parser, s| {
        parser
            .parse_with_confidence_static(s)
            .map(Some)
            .map_err(|e| polars_err!(ComputeError: "address parsing failed: {e}"))
    })?;

    // Same flat-buffers-plus-offsets shape as `parse_address`.
    let mut tokens: Vec<String> = Vec::new();
    let mut labels: Vec<&'static str> = Vec::new();
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

    list_column(inner, offsets, validity)
}
