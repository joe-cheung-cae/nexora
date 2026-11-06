# Repository guidance

## Task execution

- Prioritize completing and verifying the requested task. Keep planning proportional to complexity, make the smallest complete change, and add abstractions, dependencies, or refactoring only when the task requires them.
- Reuse existing algorithms, data structures, APIs, and utilities when they meet requirements and preserve correctness. Extend them when appropriate.
- For work with separable subtasks, define their scope and dependencies, then delegate them to subagents using different models matched to complexity and required capabilities. Run independent subtasks in parallel. The lead agent integrates and verifies the result.

## Project constraints

- Use **Nexora** for the product and CMake project/package, `nexora` for the CLI and C++ namespace, `nexora::` for exported targets, and `NEXORA_*` for build options and configuration macros.
- Follow the [architecture dependency rules](docs/architecture/overview.md) and check target links and includes with `tools/check_dependencies.py`. Keep CUDA and MPI types private to optional implementation targets.
- When changing state, account for ownership, communication, and checkpointing. Document any gaps.
- Change numerical formulas, defaults, tolerances, or physical conventions only on explicit request and with correctness evidence. Reject unsupported capabilities explicitly.

## Build and validation

- Keep the default CPU-only C++17 / CMake 3.24+ / Ninja build offline and free of third-party dependencies. Keep optional dependencies opt-in and documented.
- Use [CONTRIBUTING.md](CONTRIBUTING.md) for presets, tests, benchmarks, and dependency provenance. Run the relevant checks for the change and substantiate runtime support with execution results and relevant hardware details.
