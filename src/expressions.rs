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

/// Every token of every non-null row, in order: the flat values a list
/// column's offsets index into.
fn all_tokens<T>(rows: &[Option<Vec<T>>]) -> impl Iterator<Item = &T> {
    rows.iter().flatten().flatten()
}

/// Wrap one flat `inner` column (built from [`all_tokens`]) into a list
/// column with one entry per row of `rows`: that row's tokens, or null for a
/// `None` row. Zero-copy over `inner`: one Arrow `ListArray<i64>` over its
/// single chunk.
fn list_column<T>(inner: Series, rows: &[Option<Vec<T>>]) -> PolarsResult<Series> {
    let offsets: Vec<i64> = std::iter::once(0)
        .chain(rows.iter().scan(0i64, |end, row| {
            *end += row.as_ref().map_or(0, Vec::len) as i64;
            Some(*end)
        }))
        .collect();
    polars_ensure!(
        offsets.last() == Some(&(inner.len() as i64)),
        ComputeError: "list offsets don't cover the inner column"
    );

    let dtype = DataType::List(Box::new(inner.dtype().clone()));
    let inner = inner.rechunk();
    let values = inner.chunks()[0].clone();
    let arrow_dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
    let validity = rows
        .iter()
        .any(Option::is_none)
        .then(|| Bitmap::from_iter(rows.iter().map(Option::is_some)));
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

// ---------------------------------------------------------------- tag_address

/// The struct fields both `tag_address*` expressions share: one per label,
/// then `address_type`.
fn tag_fields() -> Vec<Field> {
    LABELS
        .iter()
        .map(|name| Field::new((*name).into(), DataType::String))
        .chain([Field::new("address_type".into(), DataType::String)])
        .collect()
}

/// The columns for [`tag_fields`], moving each row's components out.
/// `split` gives a row's components and address type.
fn tag_columns<R>(
    rows: &mut [Option<R>],
    split: impl Fn(&mut R) -> (&mut Components, &'static str),
) -> Vec<Series> {
    let mut columns = label_columns(rows, |r| split(r).0);
    columns.push(
        StringChunked::from_iter_options(
            "address_type".into(),
            rows.iter_mut().map(|row| row.as_mut().map(|r| split(r).1)),
        )
        .into_series(),
    );
    columns
}

fn tag_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new("address".into(), DataType::Struct(tag_fields())))
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

    let fields = tag_columns(&mut rows, |(components, t)| (components, *t));
    StructChunked::from_series("address".into(), ca.len(), fields.iter()).map(|ca| ca.into_series())
}

// ------------------------------------------------------- tag_address_with_confidence

fn tag_confidence_output(_: &[Field]) -> PolarsResult<Field> {
    let mut fields = tag_fields();
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

    let mut fields = tag_columns(&mut rows, |(components, t, _)| (components, *t));
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

    // One flat column per struct field across all rows, then offsets over
    // it -- no Series per row.
    let tokens = || all_tokens(&parsed_rows);
    let inner = StructChunked::from_series(
        "".into(),
        tokens().count(),
        [
            StringChunked::from_iter_values("token".into(), tokens().map(|(t, _)| t.as_str()))
                .into_series(),
            StringChunked::from_iter_values("label".into(), tokens().map(|(_, l)| *l))
                .into_series(),
        ]
        .iter(),
    )?
    .into_series();

    list_column(inner, &parsed_rows)
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

    // Same shape as `parse_address`, plus the confidence column.
    let tokens = || all_tokens(&parsed_rows);
    let inner = StructChunked::from_series(
        "".into(),
        tokens().count(),
        [
            StringChunked::from_iter_values("token".into(), tokens().map(|(t, _, _)| t.as_str()))
                .into_series(),
            StringChunked::from_iter_values("label".into(), tokens().map(|(_, l, _)| *l))
                .into_series(),
            Float64Chunked::from_iter_values("confidence".into(), tokens().map(|(_, _, c)| *c))
                .into_series(),
        ]
        .iter(),
    )?
    .into_series();

    list_column(inner, &parsed_rows)
}
