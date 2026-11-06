#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard runtime(argc, argv);
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    constexpr int count = 32;
    const ExecutionSpace execution_space;
    const Kokkos::View<int*, ExecutionSpace> values("nexora_values", count);
    const Kokkos::RangePolicy<ExecutionSpace> range(execution_space, 0, count);
    Kokkos::parallel_for(
        "nexora_integer_probe", range, KOKKOS_LAMBDA(int index) { values(index) = 3 * index - 7; });
    execution_space.fence("nexora integer probe complete");

    const auto host_values = Kokkos::create_mirror(values);
    Kokkos::deep_copy(host_values, values);
    int mismatches = 0;
    for (int index = 0; index < count; ++index) {
        const int expected = 3 * index - 7;
        if (host_values(index) != expected) {
            std::cerr << "Element " << index << " expected " << expected << ", got "
                      << host_values(index) << '\n';
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        return 1;
    }

    std::cout << "Verified " << count << " Kokkos integer results on " << ExecutionSpace::name()
              << ". Final value " << host_values(count - 1) << '\n';
    return 0;
}
