#include <nexora/core/Version.hpp>
#include <Kokkos_Core.hpp>

#include "BuildInfo.hpp"

#include <iostream>
#include <string_view>

#if defined(_MSVC_LANG)
constexpr long cxx_language = _MSVC_LANG;
#else
constexpr long cxx_language = __cplusplus;
#endif

#if NEXORA_HAS_CUDA
int cuda_smoke();
#endif
#if NEXORA_HAS_MPI
int mpi_smoke(int argc, char** argv);
#endif

namespace {

void usage(std::ostream& out) {
    out << "Usage: nexora <command>\n"
        << "  --version      Print the project version\n"
        << "  inspect-build  Print compiled capabilities\n"
        << "  smoke-cuda     Run the optional CUDA diagnostic\n"
        << "  smoke-mpi      Run the optional MPI diagnostic with two ranks\n";
}

void inspect_build() {
    using namespace nexora::build_info;
    std::cout << "project=Nexora\n"
              << "version=" << nexora::version << '\n'
              << "platform=" << platform << '\n'
              << "architecture=" << architecture << '\n'
              << "compiler=" << compiler << '\n'
              << "cxx_language=" << cxx_language << '\n'
              << "cmake_version=" << cmake_version << '\n'
              << "configuration=" << NEXORA_CONFIGURATION << '\n'
              << "cpu_execution=kokkos_serial\n"
              << "kokkos_version=" << KOKKOS_VERSION_MAJOR << '.' << KOKKOS_VERSION_MINOR
              << '.' << KOKKOS_VERSION_PATCH << '\n'
              << "kokkos_execution_space=" << Kokkos::DefaultExecutionSpace::name() << '\n'
              << "cuda_smoke=" << (NEXORA_HAS_CUDA ? "on" : "off") << '\n'
              << "mpi_smoke=" << (NEXORA_HAS_MPI ? "on" : "off") << '\n';
#if NEXORA_HAS_CUDA
    std::cout << "cuda_architectures=" << cuda_architectures << '\n';
#endif
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        usage(std::cerr);
        return 2;
    }

    const std::string_view command{argv[1]};
    if (command == "--help") {
        usage(std::cout);
    } else if (command == "--version") {
        std::cout << "Nexora " << nexora::version << '\n';
    } else if (command == "inspect-build") {
        inspect_build();
    } else if (command == "smoke-cuda") {
#if NEXORA_HAS_CUDA
        return cuda_smoke();
#else
        std::cerr << "CUDA smoke probe was disabled at build time.\n";
        return 2;
#endif
    } else if (command == "smoke-mpi") {
#if NEXORA_HAS_MPI
        return mpi_smoke(argc, argv);
#else
        std::cerr << "MPI smoke probe was disabled at build time.\n";
        return 2;
#endif
    } else {
        std::cerr << "Unknown command: " << command << '\n';
        usage(std::cerr);
        return 2;
    }
    return 0;
}
