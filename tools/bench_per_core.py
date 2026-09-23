"""Per-core and all-core throughput: polars_usaddress vs. Python usaddress.

The go/no-go number for this crate is the *per-core* ratio: single-threaded
Rust against single-core Python. The all-core number is reported too, but it
mostly measures how many cores the machine has.

    uv run maturin develop -r
    cargo build --release --no-default-features --example bench_split
    uv run python tools/bench_per_core.py

Each plugin measurement runs in a fresh subprocess, because Rayon sizes its
global pool once, at first use, from RAYON_NUM_THREADS.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "corpus_fixtures.json"
FALLBACK = ROOT / "tools" / "bench_corpus.json"
REPEATS = 5


def corpus() -> list[str]:
    if FIXTURES.exists():
        return [c["input"] for c in json.loads(FIXTURES.read_text())["fixtures"]]
    return json.loads(FALLBACK.read_text())


def best_of(fn, repeats: int = REPEATS) -> float:
    """Median wall time of `repeats` runs, after one warm-up."""
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def python_usaddress(addresses: list[str]) -> float:
    import usaddress

    def run():
        for a in addresses:
            try:
                usaddress.tag(a)
            except usaddress.RepeatedLabelError:
                pass

    return best_of(run, repeats=3)


def plugin_child(fn_name: str) -> None:
    """Runs inside the subprocess: time one plugin function on the corpus."""
    import polars as pl

    import polars_usaddress as plua

    df = pl.DataFrame({"address": corpus()})
    expr = getattr(plua, fn_name)("address")
    print(best_of(lambda: df.select(expr)))


def plugin(fn_name: str, threads: int | None) -> float:
    env = dict(os.environ)
    if threads is not None:
        env["RAYON_NUM_THREADS"] = str(threads)
        env["POLARS_MAX_THREADS"] = str(threads)
    out = subprocess.run(
        [sys.executable, __file__, "--child", fn_name],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


def rust_parser_loop(path: Path) -> float | None:
    """Plain `Parser::tag` loop, single thread, no Polars. Needs bench_split built."""
    exe = ROOT / "target" / "release" / "examples" / "bench_split"
    if not exe.exists():
        return None
    runs = []
    for _ in range(REPEATS):
        out = subprocess.run([exe, path], check=True, capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("total:"))
        runs.append(float(line.split()[1]) / 1e3)
    return statistics.median(runs)


def main() -> None:
    addresses = corpus()
    n = len(addresses)
    tmp = ROOT / "target" / "bench_corpus_inputs.json"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(addresses))

    py = python_usaddress(addresses)
    rows = [("python usaddress.tag, 1 core", py)]
    rust = rust_parser_loop(tmp)
    if rust is not None:
        rows.append(("rust Parser::tag loop, 1 thread", rust))
    for fn_name in ("tag_address", "parse_address"):
        rows.append((f"plugin {fn_name}, 1 thread", plugin(fn_name, 1)))
        rows.append((f"plugin {fn_name}, all threads", plugin(fn_name, None)))

    print(f"n = {n} addresses, {os.cpu_count()} cores, median of {REPEATS}\n")
    print(f"{'engine':<36}{'seconds':>10}{'addrs/s':>14}{'vs python':>11}")
    for name, t in rows:
        print(f"{name:<36}{t:>10.4f}{n / t:>14,.0f}{py / t:>10.1f}x")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        plugin_child(sys.argv[2])
    else:
        main()
