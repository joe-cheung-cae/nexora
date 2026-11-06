# Repository guidance

## Task execution

- Prioritize completing and verifying the requested task. Keep planning proportional to complexity, make the smallest complete change, and add abstractions, dependencies, or refactoring only when the task requires them.
- Reuse existing algorithms, data structures, APIs, and utilities when they meet requirements and preserve correctness. Extend them when appropriate.
- For work with separable subtasks, define their scope and dependencies, then delegate them to subagents using different models matched to complexity and required capabilities. Run independent subtasks in parallel. The lead agent integrates and verifies the result.
- This phase is documentation and design review only. Do not migrate source, CMake, presets, CI, or test code until the [phase-1 review gate](docs/architecture/phase-1-design-review.md) is approved. Existing bootstrap code is evidence of current behavior, not permission to continue its architecture.

## Project constraints

- Use **Nexora** for the product and CMake project/package, `nexora` for the CLI and C++ namespace, `nexora::` for exported targets, and `NEXORA_*` for Nexora build options and configuration macros. Preserve upstream `Kokkos_*` option names when configuring Kokkos.
- Start with the [architecture overview](docs/architecture/overview.md). Follow its dependency rules and the [Kokkos foundation contracts](docs/architecture/kokkos-design.md).
- Kokkos is the sole execution and numerical-storage foundation and is required for CPU builds. Do not introduce or extend native CPU/CUDA/HIP parallel implementations, provider selection, a DSL runtime, or a second allocation/dispatch framework. Optional CUDA/HIP execution means Kokkos execution spaces; MPI is a separate optional communication integration.
- Warp is permitted only as an internal dependency of an explicitly approved complete external engine. Wrapping it in an application, service, worker, or process must not create a Nexora generic operator provider or a second computing foundation. Future M/I tasks require separate design and approval under the [current task map](docs/architecture/follow-on-task-plan.md); archive text supplies provenance, not missing implementation requirements.
- Kokkos types are permitted at the designated foundation/data/kernel boundaries. Keep raw CUDA/HIP and MPI runtime types private to optional interoperability implementations; do not leak them into public physics contracts.
- When changing state, account for ownership, host/device accessibility, synchronization, communication, and checkpointing. Document any gaps.
- Change numerical formulas, defaults, tolerances, or physical conventions only on explicit request and with correctness evidence. Reject unsupported capabilities explicitly.

## Build and validation

- The proposed migration baseline is C++20, CMake 3.25.2+, Ninja, and the exact Kokkos lock in [build and validation](docs/architecture/build-and-validation.md). Default CPU operation uses Kokkos Serial; OpenMP, CUDA/HIP, and MPI are explicit configurations with separate evidence requirements.
- The initial CUDA target uses the [nvcc_wrapper/CXX contract](docs/architecture/build-and-validation.md#cuda-build-contract), including matched installed consumers. Retire and reject `NEXORA_GPU_BACKEND` at cutover; derive allowed profiles from the verified Kokkos package. Do not claim runtime validation from configuration or compilation.
- Offline means locally provisioned, version-verified dependencies and no implicit configure/build/test/install downloads. It does not mean zero third-party dependencies. Never substitute a native CPU path when Kokkos is absent.
- Until migration approval and implementation, the existing C++17 / CMake 3.24 bootstrap presets and `tools/check_dependencies.py` only validate the old target graph. Passing them does not validate Kokkos boundaries. Update the checker and package tests together with the approved implementation.
- Use [CONTRIBUTING.md](CONTRIBUTING.md) for current checks and the target validation process. Substantiate runtime support with executed results, relevant hardware/toolchain details, and timestamps in ISO 8601 with seconds and timezone. Distinguish configured, compiled, executed, and numerically validated states.
- Record source, revision, artifact digest, license/notices, patches, and toolchain for dependencies. Nexora's license and legacy reuse rights remain unresolved; do not infer authorization from Kokkos's license.
