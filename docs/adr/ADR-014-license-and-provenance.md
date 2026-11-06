# ADR-014: License and provenance remain pending

- Status: Nexora license pending maintainer decision; Kokkos provenance recorded for the proposed dependency
- Originally recorded at `2026-11-06T11:01:56+08:00`
- Updated at `2026-11-06T14:49:39+08:00`

## Decision

No Nexora project license has been selected. `LICENSE` is a neutral status and reservation notice, not an OSI license or a grant. No legacy source is presumed reusable. Record origin, authorship/rights holder where known, license terms, and intended reuse scope before import. Maintainers must resolve project licensing before describing Nexora as open source or accepting contributions under assumed terms.

Kokkos is the required target dependency under [ADR-003](ADR-003-kokkos-only-foundation.md). Its selected 5.2.2 source is licensed under **Apache License 2.0 with LLVM exceptions** (`Apache-2.0 WITH LLVM-exception`), as stated in the [upstream LICENSE](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/LICENSE). This is a distinct third-party license; it neither selects Nexora's license nor grants rights to legacy Nexora/CFDX material.

## Provenance and distribution requirements

Follow [the dependency record](../../third_party/README.md): exact upstream revision, source artifact digest, license and applicable notices, local patches, build/runtime role, and toolchain/configuration must accompany provisioned dependencies. Preserve Kokkos's license text, applicable attribution/notices, and modification records in redistributed artifacts according to its terms. Inspect the exact source distribution for included components and notices rather than assuming every bundled file has identical terms.

No Kokkos source, binary, or archive is imported in this documentation phase. Source acquisition verification, dependency installation, and distribution packaging remain implementation work after the design gate. Nexora redistribution and contribution terms remain unresolved.
