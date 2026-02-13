---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Run command: `edgefirst-fusion --model model.rtm ...`
2. Observe behavior: ...
3. See error: ...

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

What actually happened (including error messages).

## Environment

**Hardware:**
- Platform: [e.g., NXP i.MX8M Plus, x86_64 desktop]
- Sensors: [e.g., Radar: DRVEGRD-171, LiDAR: Ouster OS1, Camera: MIPI CSI-2]

**Software:**
- OS: [e.g., Linux 5.15, Yocto Kirkstone]
- Rust version: [e.g., 1.90.0]
- edgefirst-fusion version: [e.g., 0.1.0, commit SHA]

**Configuration:**
```bash
# Paste your command-line arguments or configuration
edgefirst-fusion --model model.rtm --track --engine npu
```

## Logs

<details>
<summary>Logs (click to expand)</summary>

```
# Paste journalctl output or stderr logs
journalctl -u edgefirst-fusion -n 100
```

</details>

## Additional Context

Any other context about the problem (e.g., happens only with specific sensor combinations, works with radar but not LiDAR).

## Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have tested with the latest version
- [ ] I have included all relevant logs and environment details
- [ ] I have provided steps to reproduce the issue
