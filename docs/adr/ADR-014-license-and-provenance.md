# ADR-014: Project license and provenance

- Status: Accepted
- Originally recorded at `2026-11-06T11:01:56+08:00`
- BSD 3-Clause License adopted at `2026-11-11T16:19:53+08:00`

## Decision

Nexora adopts the [BSD 3-Clause License](../../LICENSE) (SPDX `BSD-3-Clause`) for its original code and documentation. The license permits commercial use, modification, and redistribution. Retain the copyright notice, license conditions, and disclaimer in source redistributions and in the documentation or materials accompanying binary redistributions. Using the copyright holder's or contributors' names to endorse or promote derived products requires specific prior written permission.

Before importing legacy or third-party material, record its origin, authorship or rights holder where known, license terms, and intended reuse scope. Verify that those terms permit the intended use.

Kokkos is a required dependency under [ADR-003](ADR-003-kokkos-only-foundation.md). The selected 5.2.2 release uses **Apache License 2.0 with LLVM exceptions** (`Apache-2.0 WITH LLVM-exception`), as stated in the [upstream LICENSE](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/LICENSE).

## Provenance and distribution requirements

Follow [the dependency record](../../third_party/README.md): exact upstream revision, source artifact digest, license and applicable notices, local patches, build/runtime role, and toolchain/configuration must accompany provisioned dependencies. Preserve Kokkos's license text, applicable attribution/notices, and modification records in redistributed artifacts according to its terms. Inspect the exact source distribution for included components and notices rather than assuming every bundled file has identical terms.

Nexora's CMake install includes the project `LICENSE` under `CMAKE_INSTALL_DOCDIR`. The source repository uses a separately provisioned Kokkos installation. Check the license and notice requirements for all dependencies included in a distribution, including statically linked components.
