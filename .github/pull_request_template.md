<!--
Thank you for contributing to EdgeFirst Fusion!
Please fill out this template to help us review your pull request.
-->

## Description

<!-- Provide a clear and concise summary of the changes in this PR -->

## Related Issues

<!--
Link to related issues using GitHub keywords:
- Fixes #123 (closes the issue when PR is merged)
- Closes #456 (closes the issue when PR is merged)
- Related to #789 (references without closing)

For JIRA tickets, use format: EDGEAI-123
-->

Fixes #
JIRA: EDGEAI-

## Type of Change

<!-- Please check the type(s) that apply to this PR -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)
- [ ] CI/CD or build changes

## Testing

<!-- Describe the testing you've performed to verify your changes -->

**Test environment:**

- Platform: <!-- e.g., NXP i.MX8M Plus, x86_64 desktop -->
- Sensors: <!-- e.g., Radar: DRVEGRD-171, Camera: MIPI CSI-2 -->
- OS: <!-- e.g., Yocto Kirkstone, Ubuntu 22.04 -->

**Automated tests:**

- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo test --lib` passes (unit tests)
- [ ] `cargo build --release` succeeds

**Platform testing:**

- [ ] Tested on x86_64 (development)
- [ ] Tested on ARM64 (target hardware)

**Manual testing:**

<!-- Describe manual testing performed, including commands and observed results -->

```bash
# Example: Commands run and their results
edgefirst-fusion --model model.rtm --track
# Expected: Fusion runs with tracking enabled
# Actual: ...
```

## Documentation

<!-- Check all that apply to your changes -->

- [ ] Code is self-documenting and/or includes inline comments
- [ ] Public API has rustdoc documentation (/// comments with examples)
- [ ] README.md updated (if user-facing changes)
- [ ] ARCHITECTURE.md updated (if design changes)
- [ ] CHANGELOG.md updated (if notable changes)

## License Compliance

<!-- Required if dependencies were modified -->

- [ ] No new GPL/AGPL dependencies added
- [ ] SBOM generation succeeds (`bash .github/scripts/generate_sbom.sh`)
- [ ] License policy check passes (`python3 .github/scripts/check_license_policy.py sbom.json`)
- [ ] NOTICE file updated (if new dependencies with special licenses)

## Checklist

<!-- Please review all items before requesting review -->

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] My branch name follows the pattern `<type>/EDGEAI-###[-desc]`
- [ ] My commits follow the pattern `EDGEAI-###: Brief description`
- [ ] My code follows the project's style guidelines (`cargo fmt`)
- [ ] I have performed a self-review of my own code
- [ ] I have commented complex or non-obvious code
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing unit tests pass locally
- [ ] No new compiler warnings introduced
- [ ] Breaking changes are documented and justified

## Performance Impact

<!-- Required for changes affecting runtime performance -->

- [ ] No performance impact expected
- [ ] Performance improved (see benchmarks below)
- [ ] Performance may be affected (justification provided)

**Benchmarks:**

<!-- If applicable, paste cargo bench output or manual timing results -->

```
# cargo bench output or manual performance measurements
```

## Breaking Changes

<!--
If this PR introduces breaking changes, describe them here.
Provide migration instructions for users upgrading from previous versions.
-->

## Additional Context

<!-- Any other information that would help reviewers understand this PR -->

---

## Reviewer Checklist

<!-- For reviewers: Verify the following before approving -->

- [ ] Code quality meets project standards
- [ ] Test coverage is adequate
- [ ] Documentation is complete and accurate
- [ ] No security concerns or hardcoded credentials
- [ ] Performance impact is acceptable
- [ ] License compliance verified (if dependencies changed)
