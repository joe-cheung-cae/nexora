# Third-party dependency policy

## Selected foundation and current state

Kokkos is the selected sole execution/numerical-storage foundation and a **required CPU dependency**. GPU backends and MPI are optional. The former requirement for a third-party-free CPU build is superseded by [ADR-003](../docs/adr/ADR-003-kokkos-only-foundation.md). Default operation must remain offline using locally provisioned inputs.

The limited CPU slice consumes a locally installed exact Serial-only Kokkos package. The repository does not vendor the archive or binaries. See [build and validation](../docs/architecture/build-and-validation.md) for provisioning and [slice validation](../docs/architecture/kokkos-cpu-slice-validation.md) for execution evidence. Full Phase 1 remains unapproved.

## Kokkos provenance record

| Field | Recorded target |
| --- | --- |
| Upstream | [kokkos/kokkos](https://github.com/kokkos/kokkos) |
| Exact release | [5.2.2](https://github.com/kokkos/kokkos/releases/tag/5.2.2); release date shown as 2026-09-10, exact seconds/timezone not obtained from the summary |
| Exact commit | [`8e15454876298832f013ebcaf08d53825b5bda40`](https://github.com/kokkos/kokkos/commit/8e15454876298832f013ebcaf08d53825b5bda40) |
| License | [Apache License 2.0 with LLVM exceptions](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/LICENSE), SPDX `Apache-2.0 WITH LLVM-exception` |
| Role | Required build/link dependency for numerical data and execution, including CPU; runtime deployment depends on static/shared linkage and selected backend/runtime |
| Local patches | None proposed |
| Artifact digest | Official release `kokkos-5.2.2.tar.gz`: SHA-256 `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f`, matched against the official release SHA-256 manifest |
| Provisioning | Verified local source cache → configuration-specific installed prefix → exact installed-package discovery; no configure/build download |
| Verification time | `2026-11-06T16:24:07+08:00`: official archive checksum matched and annotated tag peeled to the locked commit; signature not independently verified |

The official release publishes source distributions, a SHA-256 manifest, signature and provenance. Preserve the actual bytes, digest, revision, and applicable license/notices before offline use. The acquired release includes desul atomics at `83d26f34db50b4fefe30ed4db38b10e04c7a9384` (BSD-3-Clause source notices), mdspan at `5d4eb209c77f4744980c0b0c2af44636cc81b08b` (Apache-2.0 WITH LLVM-exception notices), and gtest with its own license (tests disabled in the local build). Preserve these notices when redistributing. GitHub-generated archives need not match the uploaded release archive.

## Rules for additions and upgrades

Record origin, revision/version, artifact digest, retrieval/verification method, license and attribution, patches, build/runtime role, toolchain and configuration. Preserve license/notice material and modification records with redistributed artifacts. Do not commit binaries or large archives here. A locally installed package needs a manifest just as a vendored source dependency does.

An upgrade changes the lock only through a reviewed dependency update with source/license review, applicable compiler/CMake changes, offline installation and downstream consumption checks, and the required numerical/backend acceptance. Do not track moving branches, use version ranges as locks, silently download missing packages, or replace missing Kokkos with native execution. Additional libraries remain separate opt-in decisions; Kokkos Kernels and external solver packages are not implicitly required by selecting Kokkos Core.

Nexora's original code and documentation use the [BSD 3-Clause License](../LICENSE). Dependencies retain their own licenses. Confirm reuse rights before importing legacy material, as described in [ADR-014](../docs/adr/ADR-014-license-and-provenance.md).

## Current local provisioning

The temporary Kokkos prefix from the initial validation is no longer present. At `2026-11-11T10:49:31+08:00`, the same checksum-verified official 5.2.2 source was rebuilt without patches as a C++20 Serial-only Release static package under the Git-ignored `.cache/dependencies/kokkos-5.2.2/install-release`. See the [field migration review](../docs/architecture/field-migration-review.md) for the toolchain, new binary digest and four-profile execution evidence. Configuration and builds continue to consume a locally provisioned package; they do not download dependencies.

## GoogleTest provenance record

GoogleTest is a **test-only** dependency when `BUILD_TESTING=ON`. Production builds
and installed Nexora consumers do not require it. It does not provide numerical
storage or execution. Nexora consumes an exact, locally installed package; missing
or incompatible packages fail configuration without downloading sources.

| Field | Recorded value |
| --- | --- |
| Upstream | [google/googletest](https://github.com/google/googletest) |
| Exact release | [v1.17.0](https://github.com/google/googletest/releases/tag/v1.17.0) |
| Exact commit | [`52eb8108c5bdec04579160ae17225d66034bd723`](https://github.com/google/googletest/commit/52eb8108c5bdec04579160ae17225d66034bd723) |
| Artifact | [GitHub tag source archive](https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0), cached locally as `googletest-1.17.0.tar.gz` |
| SHA-256 | `65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c` |
| License | [BSD-3-Clause](https://github.com/google/googletest/blob/52eb8108c5bdec04579160ae17225d66034bd723/LICENSE); Copyright 2008 Google Inc.; preserve the complete LICENSE with source/binary redistribution |
| Patches | None |
| Build profile | C++20, Ninja, static libraries, `BUILD_GMOCK=OFF`, `INSTALL_GTEST=ON`, upstream tests/samples disabled; `gtest_force_shared_crt=ON` matches MSVC's default dynamic CRT |
| Toolchain record | Exact C/C++ compiler path, binary digest/version output, CMake/Ninja versions, platform, build configuration and relevant environment recorded by the provisioner |
| Verification | `2026-11-19T13:36:05+08:00`: official tag/release commit checked, archive retrieved from GitHub codeload over HTTPS and locally SHA-256 hashed; LICENSE inspected. This is a repository artifact lock, not an upstream-published checksum/signature verification |

`tools/validate_serial.py provision` verifies both archives before their extraction/build, installs
into a fresh configuration/compiler-specific prefix, preserves LICENSE in
`share/nexora-gtest-provenance`, and records installed file digests in
`nexora-gtest-provision.json`. For example, with a previously acquired archive:

```sh
python3 tools/validate_serial.py --cc clang --cxx clang++ --config Debug provision \
  --archive /path/to/kokkos-5.2.2.tar.gz \
  --prefix /path/to/kokkos-debug --logs /path/to/logs
```

Only an explicit `--download` permits acquisition. Configure tests with
`-DGTest_DIR=/path/to/gtest-1.17.0-debug/lib/cmake/GTest`.
The default GoogleTest archive is `googletest-1.17.0.tar.gz` beside the Kokkos
archive; its prefix defaults to sibling `gtest-1.17.0-<debug|release>` of the
Kokkos prefix. Both can be overridden explicitly. The Serial validator provisions
both dependencies using `provision --archive <kokkos-archive> --gtest-archive
<googletest-archive> --prefix <kokkos-prefix> --gtest-prefix <gtest-prefix>` and
uses the same prefix default for `verify`. Existing prefixes are reused only after
matching the complete runner profile and checking all installed file digests. A
failed GoogleTest acquisition/build can therefore be retried without rebuilding
an already verified Kokkos installation; incompatible or modified prefixes fail
rather than being overwritten. Its combined cache identity includes both
source locks and compiler/configuration metadata. CI acquires dependencies before
the Linux network-isolated configure/build/test/install-consumer phase, which
checks both installation manifests. Do not reuse a Release/MSVC runtime prefix
for a different configuration or compiler.
