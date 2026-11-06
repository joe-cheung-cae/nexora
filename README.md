# Nexora

*Different Methods. One Physical World.*

Nexora is a modular C++ foundation for multiphysics and HPC simulation. Its selected architecture uses **Kokkos as the sole execution and numerical-storage foundation**, including the default CPU build. GPU backends and MPI are optional.

## Start here

The [architecture overview](docs/architecture/overview.md) is the authoritative entry point for the target design, current implementation, migration, and review gate. The [Kokkos design](docs/architecture/kokkos-design.md) defines the foundation contracts; [build and validation](docs/architecture/build-and-validation.md) defines the proposed dependency lock, offline workflow, configurations, and evidence requirements.

**This is a design-only phase.** The repository still contains the initialization bootstrap: C++17, CMake 3.24+, native CPU execution, host storage, and optional CUDA/MPI diagnostic probes. Source, CMake, presets, CI, and tests have not been migrated. Those existing artifacts do not implement the Kokkos design. The [phase-1 design review](docs/architecture/phase-1-design-review.md) must be approved before implementation starts.

The proposed implementation baseline is Kokkos **5.2.2**, pinned to commit `8e15454876298832f013ebcaf08d53825b5bda40`, with C++20 and CMake 3.25.2+. Kokkos is a required CPU dependency. Dependencies must be supplied locally; default configure, build, test, install, and package consumption must work without network access. See [ADR-003](docs/adr/ADR-003-kokkos-only-foundation.md).

## Current bootstrap checks

The existing `cpu-debug`, `cpu-release`, and `mpi-debug` presets exercise the bootstrap only; `mpi-debug` is a diagnostic probe. Their usage is retained in [CONTRIBUTING.md](CONTRIBUTING.md). Current `nexora inspect-build` metadata and the dependency checker describe that implementation, not future Kokkos capabilities. A CUDA probe, a successful compile, or a skipped device test does not establish solver or GPU support.

New target build and consumption examples are explicitly marked **proposed, pending design approval and implementation** in [build and validation](docs/architecture/build-and-validation.md). Do not use the old `nexora::exec_cpu` export or native CUDA option as the target architecture's integration contract.

See [AGENTS.md](AGENTS.md) before changing the project. The [archived v1.2 plan](docs/archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md) is historical context, not implementation guidance.

## License status

No Nexora project license has been selected. [LICENSE](LICENSE) and [ADR-014](docs/adr/ADR-014-license-and-provenance.md) record this unresolved decision. Kokkos has its own license; selecting it does not license Nexora or grant rights to legacy material.
