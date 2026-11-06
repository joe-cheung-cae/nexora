# ADR-001: Modular targets and dependency direction

- Status: accepted for the initialization baseline
- Recorded at `2026-11-06T11:01:56+08:00`

## Decision

Use separate CMake targets for core, execution API, CPU execution, data, participant API, and simulation. Consumers depend on lower-level contracts. The command-line application and tests are leaves. Optional CUDA and MPI diagnostic paths are built only when explicitly enabled and do not create a false claim that full CUDA execution or distributed solvers exist.

The direct target links and public-header include boundaries are machine-checked in CI. A target may not reach into another module's implementation headers.

## Consequences

The graph gives each module a small compile and API boundary. New modules require an explicit dependency policy entry and tests. Existing legacy code is not presumed to fit the graph until its provenance and behavior are reviewed.
