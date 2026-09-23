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

import importlib.util
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
    if threads is None:
        env.pop("RAYON_NUM_THREADS", None)
        env.pop("POLARS_MAX_THREADS", None)
    else:
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


BENCH_SPLIT = ROOT / "target" / "release" / "examples" / "bench_split"


def plugin_library() -> Path | None:
    """The compiled plugin the child processes will load, found without importing it."""
    spec = importlib.util.find_spec("polars_usaddress")
    if spec is None or spec.origin is None:
        return None
    return next(iter(sorted(Path(spec.origin).parent.glob("_internal*"))), None)


def newest_source_mtime() -> float:
    sources = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    sources += (ROOT / "src").rglob("*.rs")
    sources += (ROOT / "vendor" / "crfs" / "src").rglob("*.rs")
    return max(p.stat().st_mtime for p in sources if p.exists())


def sanity_warnings(rust: float | None, plugin_1t: float, artifacts: dict[str, Path | None]) -> list[str]:
    """Catch the two ways this benchmark has produced wrong numbers before:
    a debug build of the plugin, and a build older than the source."""
    warnings = []
    # One thread, same parser: the plugin should be close to the plain Rust loop.
    if rust is not None and plugin_1t > 3 * rust:
        warnings.append(
            f"plugin tag_address on 1 thread is {plugin_1t / rust:.1f}x slower than the Rust "
            "loop running the same parser. It is almost certainly a debug build: rebuild with "
            "`uv run maturin develop -r`."
        )
    newest = newest_source_mtime()
    for name, path in artifacts.items():
        if path is not None and path.stat().st_mtime < newest:
            warnings.append(f"{name} ({path.relative_to(ROOT)}) is older than the Rust source; rebuild it.")
    return warnings


def rust_parser_loop(path: Path) -> float | None:
    """Plain `Parser::tag` loop, single thread, no Polars. Needs bench_split built."""
    exe = BENCH_SPLIT
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

    plugin_1t = dict(rows)["plugin tag_address, 1 thread"]
    artifacts = {
        "plugin": plugin_library(),
        "bench_split": BENCH_SPLIT if BENCH_SPLIT.exists() else None,
    }
    warnings = sanity_warnings(rust, plugin_1t, artifacts)
    if rust is None:
        warnings.append(
            "no Rust loop row: build it with "
            "`cargo build --release --no-default-features --example bench_split`."
        )
    for w in warnings:
        print(f"\n*** WARNING: {w}", file=sys.stderr)
    if warnings:
        print("\n*** These numbers are probably not trustworthy.", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        plugin_child(sys.argv[2])
    else:
        main()
