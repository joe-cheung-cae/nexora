# Nexora

*Different Methods. One Physical World.*

Nexora is a modular C++ foundation for multiphysics and HPC simulation. Its selected architecture uses **Kokkos as the sole execution and numerical-storage foundation**, including the default CPU build. GPU backends and MPI are optional.

## Start here

The [architecture overview](docs/architecture/overview.md) is the authoritative entry point for the target design, current implementation, migration, and review gate. The [Kokkos design](docs/architecture/kokkos-design.md) defines the foundation contracts; [build and validation](docs/architecture/build-and-validation.md) defines the dependency lock, current offline workflow, proposed later profiles, and evidence requirements.

**The first limited CPU slice is implemented.** It requires locally installed Kokkos **5.2.2**, locked to commit `8e15454876298832f013ebcaf08d53825b5bda40`, with Serial only, C++20 and CMake 3.25.2+. A typed Kokkos View and execution test and an installed consumer through `nexora::data` exercise the new boundary. The old exec layer and core numerical wrappers have been removed; domain and control contracts remain. See [core data structures](docs/architecture/core-data-structures.md) for migration and native Kokkos interfaces. This does not complete or approve [Phase 1](docs/architecture/phase-1-design-review.md).

**The minimal reference state lifecycle is also implemented.** An internal reference participant uses Kokkos `x/q` arrays, stable IDs and independent accepted/trial storage through the existing Simulation. CPU tests cover commit, restore, numerical failure, retry, replay and retained read-only snapshots; see the [implementation and validation record](docs/architecture/reference-lifecycle-validation.md). It is an infrastructure fixture, with internal scope and no installed reference API.

`nexora::real` is a configurable scalar alias exposed by `nexora/core/Real.hpp` and `Types.hpp`: it defaults to `float`, and `-DNEXORA_ENABLE_DOUBLE_PRECISION=ON` selects `double`. The installed package fixes this choice; changing it requires a clean build and install prefix and rebuilding consumers. Existing reference fields, parameters, operators and local transport now use `real`; time values remain `double`. Field descriptors default to the corresponding float32/float64 dtype. See the [numerical validation](docs/architecture/real-numerics-validation.md) for both precisions. See [core data structures](docs/architecture/core-data-structures.md).

The managed Serial data layer exports `nexora::data_schema` for field semantics and `nexora::data` for native Kokkos View bindings with read/write leases. See the [field binding validation record](docs/architecture/field-binding-validation.md).

Configure, build, test and install use locally provisioned dependencies without implicit downloads. See [CONTRIBUTING.md](CONTRIBUTING.md) for commands, [dependency provenance](third_party/README.md), and the [CPU slice validation record](docs/architecture/kokkos-cpu-slice-validation.md) for actual results. `NEXORA_GPU_BACKEND` is rejected with any value; the old native CUDA source is dormant. OpenMP and GPU profiles are not implemented. The optional existing MPI smoke probe is not distributed numerical support and remains unvalidated for this slice. Existing CI lacks Kokkos provisioning and is not evidence for it.

See [AGENTS.md](AGENTS.md) before changing the project. The [archived v1.2 plan](docs/archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md) is historical context, not implementation guidance.

## License status

No Nexora project license has been selected. [LICENSE](LICENSE) and [ADR-014](docs/adr/ADR-014-license-and-provenance.md) record this unresolved decision. Kokkos has its own license; selecting it does not license Nexora or grant rights to legacy material.
