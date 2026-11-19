# Nexora

Nexora is a C++ foundation for high-performance multiphysics simulation. It uses Kokkos for numerical storage and execution. The current Serial CPU implementation includes field management, simulation state and time-step control, a Reference case CLI, and an internal one-dimensional SPH density calculation.

## Quick start

Use a 64-bit C/C++ toolchain with C++20 support, CMake 3.25.2 or newer, and Ninja. The Debug preset requires locally installed Kokkos 5.2.2 (Serial only) and GoogleTest 1.17.0. See the [dependency provisioning commands](docs/architecture/serial-ci-validation.md#重跑) and [CMake package paths](CONTRIBUTING.md#current-cpu-checks).

Build and test from the repository root, then run the Reference example.

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
./build/cpu-debug/apps/nexora/nexora run --fixture cases/reference/cli-nonbinary-tail.case
```

The example reports `accepted_steps=3`. See the [Reference case guide](docs/architecture/reference-case.md) for the input format and the `validate`, `plan`, and `run` commands.

## Documentation

- [Build and validation](docs/architecture/build-and-validation.md)
- [Architecture and roadmap](docs/architecture/overview.md)
- [Serial SPH validation](validation/sph/serial.md)

## Contributing

See the [contribution guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md). Use [GitHub Issues](https://github.com/joe-cheung-cae/nexora/issues) for bugs and feature requests. See the [security policy](SECURITY.md) to report a vulnerability privately.

## License

Nexora is licensed under the [BSD 3-Clause License](LICENSE). Dependency licenses and provenance are listed in the [third-party documentation](third_party/README.md).
