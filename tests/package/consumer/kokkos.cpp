#include "view_probe.hpp"

int main(int argc, char** argv) {
    Kokkos::ScopeGuard runtime(argc, argv);
    return verify_native_views(std::cerr);
}
