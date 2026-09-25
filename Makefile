.PHONY: test
test:
	cargo test --no-default-features && uv run maturin develop -r && uv run pytest tests
	@# vendor/crfs is a path dependency, so the root `cargo test` doesn't run its
	@# tests. Only `context::` is runnable: the others need upstream's
	@# tests/model.crfsuite, which wasn't vendored.
	cargo test --manifest-path vendor/crfs/Cargo.toml --release context::

.PHONY: bench
bench: ## Per-core throughput vs. Python usaddress (tools/bench_per_core.py)
	@# Build both things the script measures, in release, from the current source:
	@# the plugin, and the plain-Rust loop. The loop is built *without*
	@# bench-timing, whose per-row timers would inflate its total. --no-sync so uv
	@# doesn't reinstall the project over the plugin maturin just built.
	uv run maturin develop --release
	cargo build --release --no-default-features --example bench_split
	uv run --no-sync python tools/bench_per_core.py

.PHONY: format
format: ## Format the code
	$(info --- Rust format ---)
	cargo fmt
	$(info --- Python format ---)
	uv run ruff check . --fix
	uv run ruff format .


.PHONY: check-rust
check-rust: ## Run check on Rust
	$(info --- Check Rust clippy ---)
	@# --all-targets so tests and examples are linted too, not just the library.
	@# vendor/crfs's dead-code warnings are silenced in its own Cargo.toml.
	cargo clippy --all-targets --all-features
	$(info --- Check Rust format ---)
	cargo fmt -- --check
