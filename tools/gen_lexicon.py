"""Generate src/lexicon.rs's phf sets from the installed usaddress package.

Run after bumping the `usaddress` dev dependency (and `UPSTREAM_VERSION` in
src/lib.rs to match):

    uv run python tools/gen_lexicon.py

Then diff and review before committing. A no-op upgrade should produce a
no-op diff, since both sets are sorted before being written out.
"""
from importlib.metadata import version
from pathlib import Path

import usaddress

LEXICON_PATH = Path(__file__).resolve().parent.parent / "src" / "lexicon.rs"

HEADER = """//! Street-type and directional lexicons.
//!
//! GENERATED from usaddress {version} -- do not hand-edit.
//! Regenerate with `python tools/gen_lexicon.py` after upgrading usaddress.
//!
//! Compile-time perfect hashing via `phf`, so lookups are a single probe with
//! no runtime construction and no allocation.
"""


def phf_set(name: str, values: set[str]) -> str:
    entries = "\n".join(f'    "{v}",' for v in sorted(values))
    return f"pub static {name}: phf::Set<&'static str> = phf::phf_set! {{\n{entries}\n}};\n"


usaddress_version = version("usaddress")
sections = [
    phf_set("DIRECTIONS", usaddress.DIRECTIONS),
    phf_set("STREET_NAMES", usaddress.STREET_NAMES),
]
LEXICON_PATH.write_text(HEADER.format(version=usaddress_version) + "\n" + "\n".join(sections))

print(
    f"wrote {LEXICON_PATH.relative_to(LEXICON_PATH.resolve().parent.parent)} "
    f"({len(usaddress.DIRECTIONS)} directions, {len(usaddress.STREET_NAMES)} street names) "
    f"from usaddress {usaddress_version}"
)
