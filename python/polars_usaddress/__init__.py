"""US address parsing for Polars, ported from DataMade's usaddress."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExprColumn

__all__ = [
    "tag_address",
    "tag_address_with_confidence",
    "parse_address",
    "parse_address_with_confidence",
    "LABELS",
    "UPSTREAM_VERSION",
]

LIB = Path(__file__).parent

UPSTREAM_VERSION = "0.5.16"

#: Every field `tag_address` can produce: the 29 labels the embedded CRF
#: model can actually emit, plus the 7 `Second*` variants that appear only
#: on the far side of an intersection. Kept in sync by hand with the Rust
#: `LABELS` constant in `src/lib.rs` -- update both together.
#:
#: Deliberately *not* the same as upstream's own `usaddress.LABELS`, which
#: only lists 26 labels and omits three the model can genuinely produce:
#: `CountryName`, `ZipPlus4`, `StreetNamePostModifier` (confirmed by dumping
#: the label alphabet straight from the pinned `usaddr.crfsuite` model via
#: `pycrfsuite.Tagger().labels()`, which returns 29). A schema built from
#: `usaddress.LABELS` would silently drop any row the model tags with one of
#: those three -- e.g. `usaddress.tag("O'Hare.")` produces
#: `{"CountryName": "O'Hare."}` upstream, which would have nowhere to go.
LABELS = (
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
    "SecondStreetNamePreModifier",
    "SecondStreetNamePreDirectional",
    "SecondStreetNamePreType",
    "SecondStreetName",
    "SecondStreetNamePostType",
    "SecondStreetNamePostDirectional",
    "SecondStreetNamePostModifier",
)


def tag_address(expr: IntoExprColumn) -> pl.Expr:
    """Parse addresses into a struct with one field per component.

    Returns a struct with a nullable string field for each entry in
    :data:`LABELS`, plus ``address_type`` (``"Street Address"``,
    ``"Intersection"``, ``"PO Box"`` or ``"Ambiguous"``).

    Addresses that cannot be collapsed into one component per label -- what
    upstream raises ``RepeatedLabelError`` for, typically two addresses in one
    string -- come back with all fields null rather than raising.

    >>> df.with_columns(parsed=tag_address("address")).unnest("parsed")
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="tag_address",
        is_elementwise=True,
    )


def tag_address_with_confidence(expr: IntoExprColumn) -> pl.Expr:
    """Like :func:`tag_address`, with one extra ``sequence_confidence`` field:
    the CRF's confidence in the whole label sequence for that row, not per
    component. Mirrors fastaddress's ``tag_with_confidence``, minus its
    per-component confidences -- see :func:`parse_address_with_confidence`
    for per-token confidence instead.

    Null rows behave exactly like :func:`tag_address`: a row that raises
    upstream's ``RepeatedLabelError`` comes back with every field null,
    ``sequence_confidence`` included, so ``address_type.is_null()`` stays the
    one signal for "this row didn't collapse." Empty input is not an error:
    it gets ``address_type = "Ambiguous"`` and ``sequence_confidence = 1.0``,
    same as :func:`tag_address`'s own empty-input handling.

    Costs more than :func:`tag_address`: runs a forward-backward pass per row
    in addition to Viterbi decoding. Reach for :func:`tag_address` when you
    don't need the confidence.

    >>> df.with_columns(parsed=tag_address_with_confidence("address")).unnest("parsed")
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="tag_address_with_confidence",
        is_elementwise=True,
    )


def parse_address(expr: IntoExprColumn) -> pl.Expr:
    """Label each token of an address, preserving order and repetition.

    Returns ``list[struct[{token: str, label: str}]]``. Use this when you need
    the raw labelling -- repeated labels, token order, or addresses that
    :func:`tag_address` would null out.
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="parse_address",
        is_elementwise=True,
    )


def parse_address_with_confidence(expr: IntoExprColumn) -> pl.Expr:
    """Like :func:`parse_address`, but each token also carries the CRF's
    confidence in the label it was actually given.

    Returns ``list[struct[{token: str, label: str, confidence: f64}]]``.
    ``confidence`` is the model's marginal probability for that token's label,
    in ``[0, 1]`` -- how sure it was of *that* choice, not a measure of
    whether the choice was right. Mirrors fastaddress's
    ``parse_with_confidence``.

    Costs more than :func:`parse_address`: runs a forward-backward pass in
    addition to Viterbi decoding for every row. Reach for :func:`parse_address`
    when you don't need the confidence.

    >>> df.with_columns(parsed=parse_address_with_confidence("address")).explode("parsed").unnest("parsed")
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="parse_address_with_confidence",
        is_elementwise=True,
    )
