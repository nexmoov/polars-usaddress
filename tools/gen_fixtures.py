"""Generate golden parity fixtures from the reference Python usaddress."""
import json
from importlib.metadata import version
import usaddress
import pycrfsuite

ADDRESSES = [
    # --- ordinary street addresses ---
    "123 Main St. Suite 100 Chicago, IL 60601",
    "1600 Pennsylvania Ave NW Washington DC 20500",
    "3222 Clifton Ave, Cincinnati, Ohio 45220",
    "1 Infinite Loop, Cupertino, CA 95014",
    # --- directionals, pre/post types ---
    "2020 N Milwaukee Ave, Chicago IL",
    "500 South East Blvd Apt 3B",
    "W123 N456 Main Street",
    # --- intersections (exercise SecondStreetName logic) ---
    "170th St and Broadway Ave New York, NY 10033",
    "Clark St & Addison St, Chicago IL",
    "Corner of 5th and Main",
    # --- PO boxes ---
    "PO Box 1234, Springfield, IL 62701",
    "P.O. Box 52, RR 2, Butler, PA",
    "HC 68 Box 23A, Grand Canyon AZ",
    # --- occupancy / subaddress ---
    "123 Main St #4",
    "123 Main St Apt. 4B",
    "1 Hacker Way, Bldg 17, Menlo Park CA",
    # --- punctuation / tokenizer edge cases ---
    "123 Main St.,",
    "(123) Main St",
    "123 Main St &amp; Oak Ave",
    "123 Main St &#38; Oak Ave",
    # --- number edge cases: trailing zeros, fractions, suffixes ---
    "100 Main St",
    "1000 Main St",
    "10200 Main St",
    "123 1/2 Main St",
    "123\u00bd Main St",
    "123A Main St",
    # --- recipient / landmark / ambiguous ---
    "Attn: Jane Doe, 123 Main St, Chicago IL",
    "The White House",
    "Sears Tower",
    # --- unicode / non-ascii ---
    "123 Caf\u00e9 Ave, San Jos\u00e9, CA",
    "123 \u00d1andu St",
    # --- degenerate inputs ---
    "",
    "   ",
    ",,,",
    "#",
    "&",
    "123",
    # --- RepeatedLabelError cases (Python raises; plugin must have a policy) ---
    "123 Main St, 456 Oak Ave, Chicago IL",
    "123 Main St Chicago IL 123 Main St Chicago IL",
    "Main St and Oak Ave and Elm St",
]


def encode(feats):
    """Flat attribute encoding, exactly as pycrfsuite.ItemSequence produces it."""
    return [
        {k: v for k, v in sorted(item.items())}
        for item in pycrfsuite.ItemSequence(feats).items()
    ]


fixtures = []
for addr in ADDRESSES:
    entry = {"input": addr}
    tokens = usaddress.tokenize(addr)
    entry["tokens"] = tokens

    if tokens:
        feats = usaddress.tokens2features(tokens)
        entry["attributes"] = encode(feats)
        entry["parse"] = [
            {"token": t, "label": l} for t, l in usaddress.parse(addr)
        ]
        try:
            tagged, addr_type = usaddress.tag(addr)
            entry["tag"] = dict(tagged)
            entry["address_type"] = addr_type
            entry["repeated_label_error"] = False
        except usaddress.RepeatedLabelError:
            entry["tag"] = None
            entry["address_type"] = None
            entry["repeated_label_error"] = True
    else:
        entry["attributes"] = []
        entry["parse"] = []
        entry["tag"] = {}
        entry["address_type"] = "Ambiguous"
        entry["repeated_label_error"] = False

    fixtures.append(entry)

out = {
    "usaddress_version": version("usaddress"),
    "note": "Golden vectors from reference Python usaddress. Do not hand-edit.",
    "fixtures": fixtures,
}
with open("fixtures.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

n_attrs = sum(len(a) for e in fixtures for a in e["attributes"])
print(f"{len(fixtures)} addresses, "
      f"{sum(len(e['tokens']) for e in fixtures)} tokens, "
      f"{n_attrs} attributes")
