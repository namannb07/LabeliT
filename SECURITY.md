# Security Policy

## Supported Versions

Only the latest minor release on `main` receives fixes. Older versions are not supported.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security problems.

Email **security@example.com** with:

- A description of the issue and its impact
- Steps to reproduce
- The affected version (commit SHA or release tag)
- Any suggested mitigation if you have one

You can expect:

- An acknowledgement within **3 business days**
- A status update within **14 days**
- A fix or written explanation within **90 days** of the initial report

We follow a 90-day coordinated-disclosure window. After a fix lands (or after 90 days, whichever comes first), we are happy to credit reporters in the release notes if desired.

## Scope

This project runs locally on Jetson devices and does not ship a network service. Reports most relevant to us:

- Path-traversal or arbitrary-file-write via the import/export paths
- Vulnerabilities in the installer (`scripts/install.sh`) that could affect a user's system beyond the install directory
- Malicious-engine handling in `detector.py` and the parsers

Bugs in upstream dependencies (TensorRT, pycuda, opencv, pillow, numpy) should be reported to those projects directly.
