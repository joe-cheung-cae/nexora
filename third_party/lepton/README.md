# Lepton scalar host subset

This is the Lepton expression library from OpenMM tag **8.6.1**, commit
`b399af4725573963b46d6c1083fdcf7a37615857`. OpenMM's version identifies
this snapshot; Lepton does not declare an independent 8.6.1 package version.

The five compiled translation units are `Parser.cpp`, `ParsedExpression.cpp`,
`ExpressionTreeNode.cpp`, `ExpressionProgram.cpp`, and `Operation.cpp`.
Only host scalar expression parsing, symbolic differentiation, optimization and
evaluation are built. There is no JIT, vector evaluator, assembly generator,
thread pool, device execution or OpenMM runtime dependency. `nexora_lepton`
is an internal static target, has no install/export rule, uses C++20, and carries
upstream's static-library macros for both its compilation and local consumers.
It does not select a Nexora execution or numerical-storage backend.

Driver APIs: `Lepton::Parser::parse`, `ParsedExpression::differentiate`,
`optimize`, `evaluate`, and `getRootNode`. The tree API allows callers to
validate variable names before evaluation. Missing variables and malformed
expressions throw upstream `Lepton::Exception`. Application-level domain,
finite-result, ownership and time validation remain the caller's responsibility.

## Provenance and reproduction

Origin: <https://github.com/openmm/openmm/tree/b399af4725573963b46d6c1083fdcf7a37615857/libraries/lepton>.
Verified on 2026-11-26: `git ls-remote https://github.com/openmm/openmm.git refs/tags/8.6.1`
returned the locked commit, matching the official 8.6.1 release page. GitHub's
signature display was observed; the signature was not independently verified.

Fixed source archive:
<https://codeload.github.com/openmm/openmm/tar.gz/b399af4725573963b46d6c1083fdcf7a37615857>.
SHA-256: `ed6496c2a9a351529e573fbb568483266588066695249eb01a2cf74ecc79ea7a`.
The downloaded archive remains outside tracked source in
`.cache/dependencies/lepton-8.6.1/openmm.tar.gz`; builds never fetch it.

To reproduce, download that fixed URL explicitly, verify its SHA-256, extract
`openmm-b399af4725573963b46d6c1083fdcf7a37615857/libraries/lepton/`, and copy
only the paths listed in `UPSTREAM-SHA256.txt`. That manifest hashes the
original bytes before patching. Apply `git -c core.autocrlf=false apply scalar-only.patch` from the copied
subset root (place the patch there first); disabling automatic line-ending
conversion reproduces the checked vendored bytes. Copy `docs-source/licenses/Licenses.txt` from the same archive as
`UPSTREAM-LICENSES.txt`. This includes upstream notices for components not
included here; those components are not linked by this subset. Preserve all
individual source copyright and MIT notices in redistribution.

## Local patches

`scalar-only.patch` is the complete upstream source patch:

- Remove both compiled-expression factories (declarations, definitions and
  includes), and the aggregate compiled header include. Neither compiled scalar
  nor compiled vector implementations/headers are vendored. The expression tree
  and scalar expression-program interpreter remain unchanged.
- In `Parser::parsePrecedence`, require successful numeric-token extraction
  and complete token consumption. Upstream silently accepted malformed or
  out-of-range numeric tokens; `'.'`, `'1e+'`, `'1e2.3'` and `'1e999'`
  now throw using the existing parse-error path.

`MSVC_erfc.h` is retained unchanged because `Operation.cpp` includes it;
its pre-VC2013 fallback is inactive under the required C++20 toolchains.
The original upstream `tests/TestParser.cpp` was inspected as historical
coverage context; it is not vendored and its test results are not claimed here.

## Validation scope

At `2026-11-26T21:47:50+08:00`, all 15 original upstream file digests
matched the manifest. Applying the recorded patch with automatic line-ending
conversion disabled reproduced all 15 vendored files byte-for-byte. The preserved
upstream license collection SHA-256 is
`437b7168cc997abea3b5f2a9e0fb6894f96de77b9c69be428ccfcfe9bed58293`.
Nexora's Windows MSVC 19.38.33133 Release builds compiled this target and ran
the displacement and velocity expression pistons in both float and double.
The [motion-driver validation](../../validation/sph/velocity-driven-wall-serial.md)
records the measured errors, pass criteria and remaining piston regression.
