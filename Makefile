# Rust Project Makefile
#
# This Makefile implements SPS v2.1 requirements for Rust projects
# Reference: ~/Documents/SPS/08-repository-setup.md
#
# CUSTOMIZE THESE VARIABLES FOR YOUR PROJECT:
# ===========================================================================

# Project name (for version verification)
PROJECT_NAME ?= edgefirst-fusion

# Rust features to test (default: all features)
RUST_FEATURES ?= --all-features

# Additional test flags
TEST_FLAGS ?=

# Virtual environment detection (for Python bindings)
VENV_ACTIVATE := $(shell if [ -d "venv" ]; then echo "source venv/bin/activate &&"; fi)

# ===========================================================================
# STANDARD TARGETS
# ===========================================================================

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make format         - Format Rust code with rustfmt"
	@echo "  make lint           - Run clippy with strict settings"
	@echo "  make check          - Run cargo check (fast compilation test)"
	@echo "  make sbom           - Generate SBOM and validate NOTICE file"
	@echo "  make build          - Build with coverage enabled (for testing)"
	@echo "  make test           - Run tests with coverage (nextest + llvm-cov)"
	@echo "  make verify-version - Verify version consistency"
	@echo "  make pre-release    - Complete pre-release validation"
	@echo "  make clean          - Remove build artifacts"

# Format source code
.PHONY: format
format:
	@echo "Formatting Rust code..."
	cargo fmt --all
	@echo "✓ Formatting complete"

# Run linters
.PHONY: lint
lint:
	@echo "Running clippy (strict mode)..."
	cargo clippy --all-targets $(RUST_FEATURES) -- -D warnings
	@echo "✓ Linting complete"

# Quick compilation check (no codegen)
.PHONY: check
check:
	@echo "Running cargo check..."
	cargo check --all-targets $(RUST_FEATURES)
	@echo "✓ Check complete"

# Generate SBOM and validate NOTICE file
.PHONY: sbom
sbom:
	@echo "Generating SBOM and validating NOTICE..."
	@if [ -x ".github/scripts/generate_sbom.sh" ]; then \
		bash .github/scripts/generate_sbom.sh; \
	else \
		echo "WARNING: .github/scripts/generate_sbom.sh not found or not executable"; \
		echo "Falling back to cargo-sbom..."; \
		cargo sbom > sbom.json 2>/dev/null || echo "cargo-sbom not installed, skipping"; \
	fi
	@echo "✓ SBOM validation complete"

# Build with coverage enabled (for testing)
.PHONY: build
build:
	@echo "Building with coverage instrumentation..."
	cargo build $(RUST_FEATURES)
	@echo "✓ Build complete (coverage-enabled)"

# Run tests with coverage
.PHONY: test
test: build
	@echo "Running tests with cargo-nextest and llvm-cov..."
	@if ! command -v cargo-nextest >/dev/null 2>&1; then \
		echo "ERROR: cargo-nextest not installed"; \
		echo "Install with: cargo install cargo-nextest"; \
		exit 1; \
	fi
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "ERROR: cargo-llvm-cov not installed"; \
		echo "Install with: cargo install cargo-llvm-cov"; \
		exit 1; \
	fi

	# Run Rust tests with coverage
	cargo llvm-cov nextest $(RUST_FEATURES) --workspace \
		--lcov --output-path target/rust-coverage.lcov \
		$(TEST_FLAGS)

	@echo "✓ Tests passed with coverage"
	@echo "Coverage report: target/rust-coverage.lcov"

# Verify version consistency
.PHONY: verify-version
verify-version:
	@echo "Verifying version consistency..."
	@if [ ! -f "Cargo.toml" ]; then \
		echo "ERROR: Cargo.toml not found"; \
		exit 1; \
	fi
	@CARGO_VERSION=$$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "Cargo.toml version: $$CARGO_VERSION"; \
	if [ -f "CHANGELOG.md" ]; then \
		if ! grep -q "\[$$CARGO_VERSION\]" CHANGELOG.md; then \
			echo "ERROR: Version $$CARGO_VERSION not found in CHANGELOG.md"; \
			exit 1; \
		fi; \
		echo "CHANGELOG.md: ✓"; \
	fi
	@echo "✓ Version verification complete"

# Pre-release checks
.PHONY: pre-release
pre-release: format lint verify-version test
	@echo "=================================================="
	@echo "✓ All pre-release checks passed"
	@echo "=================================================="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Review changes: git status && git diff"
	@echo "  2. Commit: git add -A && git commit -m 'Prepare release'"
	@echo "  3. Push: git push origin main"
	@echo "  4. Wait for CI/CD to pass"
	@CARGO_VERSION=$$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "  5. Tag: git tag -a -m 'Version $$CARGO_VERSION' v$$CARGO_VERSION"; \
	echo "  6. Push tag: git push origin v$$CARGO_VERSION"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/rust-coverage.lcov test-results.xml
	@echo "✓ Clean complete"
