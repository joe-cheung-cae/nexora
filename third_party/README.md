# Third-party dependency policy

## Selected foundation and current state

Kokkos is the selected sole execution/numerical-storage foundation and a **required CPU dependency**. GPU backends and MPI are optional. The former requirement for a third-party-free CPU build is superseded by [ADR-003](../docs/adr/ADR-003-kokkos-only-foundation.md). Default operation must remain offline using locally provisioned inputs.

This phase changes documentation only. No dependency source, archive, binary, package integration, or build option has been added. The target provisioning and package-consumption procedure is in [build and validation](../docs/architecture/build-and-validation.md), pending [phase-1 approval](../docs/architecture/phase-1-design-review.md).

## Proposed Kokkos provenance record

| Field | Recorded target |
| --- | --- |
| Upstream | [kokkos/kokkos](https://github.com/kokkos/kokkos) |
| Exact release | [5.2.2](https://github.com/kokkos/kokkos/releases/tag/5.2.2); release date shown as 2026-09-10, exact seconds/timezone not obtained from the summary |
| Exact commit | [`8e15454876298832f013ebcaf08d53825b5bda40`](https://github.com/kokkos/kokkos/commit/8e15454876298832f013ebcaf08d53825b5bda40) |
| License | [Apache License 2.0 with LLVM exceptions](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/LICENSE), SPDX `Apache-2.0 WITH LLVM-exception` |
| Role | Required build/link dependency for numerical data and execution, including CPU; runtime deployment depends on static/shared linkage and selected backend/runtime |
| Local patches | None proposed |
| Artifact digest | **Pending acquisition and verification**; no archive imported or checksum claimed |
| Provisioning | Verified local source cache → configuration-specific installed prefix → exact installed-package discovery; no configure/build download |
| Verification time | `2026-11-06T14:49:39+08:00`, upstream release/source review only |

The official release publishes source distributions, a SHA-256 manifest, signature and provenance. Acquisition must verify and preserve the actual bytes, digest, revision, and applicable license/notices before offline use. Do not assume a GitHub-generated source archive and an uploaded release archive have identical bytes or hashes. Inspect any included third-party components and notices in the exact distribution.

## Rules for additions and upgrades

Record origin, revision/version, artifact digest, retrieval/verification method, license and attribution, patches, build/runtime role, toolchain and configuration. Preserve license/notice material and modification records with redistributed artifacts. Do not commit binaries or large archives here. A locally installed package needs a manifest just as a vendored source dependency does.

An upgrade changes the lock only through a reviewed dependency update with source/license review, applicable compiler/CMake changes, offline installation and downstream consumption checks, and the required numerical/backend acceptance. Do not track moving branches, use version ranges as locks, silently download missing packages, or replace missing Kokkos with native execution. Additional libraries remain separate opt-in decisions; Kokkos Kernels and external solver packages are not implicitly required by selecting Kokkos Core.

Nexora's project license and legacy reuse rights remain pending under [ADR-014](../docs/adr/ADR-014-license-and-provenance.md). Kokkos's license does not resolve either question.
