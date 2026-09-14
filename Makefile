.PHONY: test
test:
	cargo test && poetry run maturin develop -r && poetry run pytest tests

.PHONY: check-stubs
check-stubs:
	poetry run mypy --strict geohash_polygon/__init__.pyi


.PHONY: format
format: ## Format the code
	$(info --- Rust format ---)
	cargo fmt
	$(info --- Python format ---)
	poetry run ruff check . --fix
	poetry run ruff format .


.PHONY: check-rust
check-rust: ## Run check on Rust
	$(info --- Check Rust clippy ---)
	@# --all-targets so functions only reachable from #[cfg(test)] (the string-based
	@# reference path our tests hold the fast path accountable to) don't get flagged as
	@# dead code by a plain, tests-blind `cargo clippy`.
	@#
	@# vendor/crfs is vendored upstream source we deliberately don't hand-edit (see
	@# vendor/crfs/VENDORED.md); it carries training-side API our port never calls, so
	@# its warnings are expected. Filtered from the printed output, but the real cargo
	@# clippy exit status is still what this target returns -- a genuine failure in our
	@# own code still fails the build.
	@log=$$(mktemp); \
	cargo clippy --all-targets --all-features >"$$log" 2>&1; \
	status=$$?; \
	grep -v '/vendor/crfs/' "$$log" || true; \
	rm -f "$$log"; \
	exit $$status
	$(info --- Check Rust format ---)
	cargo fmt -- --check
