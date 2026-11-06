# Repository guidance

## Task execution

- Prioritize completing and verifying the requested task. Keep planning proportional to complexity, make the smallest complete change, and add abstractions, dependencies, or refactoring only when the task requires them.
- Reuse existing algorithms, data structures, APIs, and utilities when they meet requirements and preserve correctness. Extend them when appropriate.
- For work with separable subtasks, define their scope and dependencies, then delegate them to subagents using different models matched to complexity and required capabilities. Run independent subtasks in parallel. The lead agent integrates and verifies the result.
- Determine scope and authorization from the current user instructions. Apply the current technical contracts; historical phase gates and approval notes do not override explicit current authorization. Keep progress, pending tasks, and approval records in project documents rather than fixing a development stage in this file.

## Project constraints

- Use **Nexora** for the product and CMake project/package, `nexora` for the CLI and C++ namespace, `nexora::` for exported targets, and `NEXORA_*` for Nexora build options and configuration macros. Preserve upstream `Kokkos_*` option names when configuring Kokkos.
- Start with the [architecture overview](docs/architecture/overview.md). Follow its dependency rules and the [Kokkos foundation contracts](docs/architecture/kokkos-design.md). Superseded ADRs, archived plans, and transitional bootstrap artifacts are historical evidence, not requirements for new work.
- Kokkos is the sole execution and numerical-storage foundation for Nexora-owned numerical work and is required for CPU builds. Use its storage, execution, copy, and synchronization facilities directly at the designated boundaries. Do not introduce or extend a native CPU/CUDA/HIP numerical route, generic operator provider, DSL runtime, or duplicate allocator/execution/dispatch framework. Optional CUDA/HIP execution means Kokkos execution spaces; MPI is a separate optional communication integration.
- An explicitly approved complete external engine may retain its internal runtime; Warp is permitted only as such an engine's internal dependency. Application, service, worker, or process wrappers must not turn that runtime into a Nexora generic operator provider or a second computing foundation. Use the [current task map](docs/architecture/follow-on-task-plan.md) for the applicable integration task's scope and prerequisites.
- Use Kokkos types at application lifecycle, numerical data/kernel, and private interoperability boundaries; core, schema, and high-level control contracts follow the architecture's dependency rules. Keep raw CUDA/HIP runtime types private to optional interoperability implementations, and MPI runtime types private to communication implementations or MPI diagnostics. Do not expose these native runtime types in public physics contracts.
- When changing state, account for ownership, host/device accessibility, synchronization, communication, and checkpointing. Document any gaps.
- Change numerical formulas, defaults, tolerances, or physical conventions only on explicit request and with correctness evidence. Reject unsupported capabilities explicitly.

## Build and validation

- Use the language standard, CMake minimum, generator, and exact Kokkos lock defined in [build and validation](docs/architecture/build-and-validation.md). Default CPU operation uses Kokkos Serial; OpenMP, CUDA/HIP, and MPI require explicit configurations and separate validation evidence. Consult that document for implemented and validated profiles; a planned profile is not an implemented capability.
- CUDA work follows the [nvcc_wrapper/CXX contract](docs/architecture/build-and-validation.md#cuda-build-contract), including matched installed consumers. Reject `NEXORA_GPU_BACKEND` with any value; derive execution-space profiles from the verified Kokkos package. Do not claim runtime validation from configuration or compilation.
- Offline means locally provisioned, version-verified dependencies and no implicit configure/build/test/install downloads. It does not mean zero third-party dependencies. Never substitute a native CPU path when Kokkos is absent.
- Update the dependency checker and installed-package tests alongside target, header-boundary, or export changes. Passing checks establishes only the behavior and configurations actually covered; retained legacy tests do not validate new Kokkos paths.
- Use [CONTRIBUTING.md](CONTRIBUTING.md) for current checks and the target validation process. Substantiate runtime support with executed results, relevant hardware/toolchain details, and timestamps in ISO 8601 with seconds and timezone. Distinguish configured, compiled, executed, and numerically validated states.
- Record source, revision, artifact digest, license/notices, patches, and toolchain for dependencies. Consult [LICENSE](LICENSE) and the [provenance policy](third_party/README.md) for project licensing and reuse requirements; a dependency's license neither selects Nexora's license nor grants rights to legacy material.
