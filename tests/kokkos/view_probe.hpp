#pragma once

#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <nexora/core/Real.hpp>
#include <type_traits>
#include <utility>

inline int verify_native_views(std::ostream& errors) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using Values = Kokkos::View<int*, ExecutionSpace>;
    using RealValues = Kokkos::View<nexora::real*, ExecutionSpace>;
    using Components = Kokkos::View<int*[3], Kokkos::LayoutRight, ExecutionSpace>;
    using Matrices = Kokkos::View<int*[2][2], Kokkos::LayoutRight, ExecutionSpace>;
    static_assert(Components::rank() == 2 && Matrices::rank() == 3);
    static_assert(!std::is_assignable_v<decltype(std::declval<Values::const_type>()(0)), int>);
    constexpr int count = 32;
    const ExecutionSpace execution_space;
    Values::const_type values;
    Components::const_type components;
    Matrices::const_type matrices;
    {
        const Values source_values("nexora_values", count);
        const Components source_components("nexora_components", count);
        const Matrices source_matrices("nexora_matrices", count);
        const Kokkos::RangePolicy<ExecutionSpace> range(execution_space, 0, count);
        Kokkos::parallel_for(
            "nexora_integer_probe", range, KOKKOS_LAMBDA(int index) {
                source_values(index) = 3 * index - 7;
                const Kokkos::Array<int, 3> local{index, index + 1, index + 2};
                for (int component = 0; component < 3; ++component)
                    source_components(index, component) = local[component];
                for (int row = 0; row < 2; ++row)
                    for (int column = 0; column < 2; ++column)
                        source_matrices(index, row, column) = 100 * index + 10 * row + column;
            });
        execution_space.fence("nexora integer probe complete");
        values = source_values;
        components = source_components;
        matrices = source_matrices;
    }

    const RealValues real_values("nexora_real_values", count);
    Kokkos::parallel_for(
        "nexora_real_probe", Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, count),
        KOKKOS_LAMBDA(int index) {
            const Kokkos::Array<nexora::real, 2> coefficients{nexora::real{0.5}, nexora::real{0.25}};
            real_values(index) = coefficients[0] * static_cast<nexora::real>(index) + coefficients[1];
        });

    // Managed const Views retain the allocations after every writable handle is gone.
    const auto host_values = Kokkos::create_mirror(values);
    const auto host_components = Kokkos::create_mirror(components);
    const auto host_matrices = Kokkos::create_mirror(matrices);
    const auto column = Kokkos::subview(components, Kokkos::ALL, 1);
    const auto host_column = Kokkos::create_mirror(column);
    const auto host_real_values = Kokkos::create_mirror(real_values);
    Kokkos::deep_copy(execution_space, host_values, values);
    Kokkos::deep_copy(execution_space, host_components, components);
    Kokkos::deep_copy(execution_space, host_matrices, matrices);
    Kokkos::deep_copy(execution_space, host_column, column);
    Kokkos::deep_copy(execution_space, host_real_values, real_values);
    execution_space.fence("nexora host results ready");
    int mismatches = 0;
    if (components.extent(1) != 3 || matrices.extent(1) != 2 || matrices.extent(2) != 2 ||
        column.stride(0) != 3) {
        errors << "Unexpected native View extents or subview stride\n";
        ++mismatches;
    }
    for (int index = 0; index < count; ++index) {
        const int expected = 3 * index - 7;
        if (host_values(index) != expected) {
            errors << "Element " << index << " expected " << expected << ", got "
                      << host_values(index) << '\n';
            ++mismatches;
        }
        for (int component = 0; component < 3; ++component)
            if (host_components(index, component) != index + component)
                ++mismatches;
        for (int row = 0; row < 2; ++row)
            for (int col = 0; col < 2; ++col)
                if (host_matrices(index, row, col) != 100 * index + 10 * row + col)
                    ++mismatches;
        if (host_column(index) != index + 1)
            ++mismatches;
        // Every result is exactly representable in both supported precisions.
        const nexora::real expected_real = static_cast<nexora::real>(2 * index + 1) / nexora::real{4};
        if (host_real_values(index) != expected_real) {
            errors << "Real element " << index << " expected " << expected_real << ", got "
                      << host_real_values(index) << '\n';
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        errors << "Native View checks failed: " << mismatches << '\n';
        return 1;
    }

    std::cout << "Verified " << count << " Kokkos scalar/vector/matrix results, managed const "
              << "ownership, and strided subviews on " << ExecutionSpace::name()
              << ". Final scalar value " << host_values(count - 1) << "; real="
              << (std::is_same_v<nexora::real, double> ? "double" : "float")
              << ", final real value " << host_real_values(count - 1) << '\n';
    return 0;
}
