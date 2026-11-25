#include <fixed_csr/Density.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

namespace nexora::sph_fixture {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;

// One-dimensional normalized M4 spline with the frozen smoothing length h = 1 m.
constexpr real normalization = real{2} / real{3};

KOKKOS_INLINE_FUNCTION real contribution(real x, real y, real mass) {
    const real q = Kokkos::abs(x - y);
    if (q <= real{1}) [[likely]]
        return mass * (normalization * (real{1} + q * q * (real{0.75} * q - real{1.5})));
    const real gap = real{2} - q;
    return mass * ((normalization * real{0.25}) * gap * gap * gap);
}

} // namespace

Result<real> evaluate_density(const Kokkos::Serial& exec, const DensityInput& input,
                              const Values& output) {
    const auto count = output.extent(0);
    if (input.self_columns.extent(0) != count || input.offsets.extent(0) != count + 1 ||
        input.positions.extent(0) != input.masses.extent(0))
        return Status{StatusCode::invalid_argument, "SPH density field and CSR extents disagree"};

    const Policy rows(exec, 0, count);
    // Store each row in real and accumulate its diagnostic in double in one pass.
    double sum{};
    Kokkos::parallel_reduce(
        "sph.density.gather_and_owned_sum", rows,
        KOKKOS_LAMBDA(std::size_t row, double& total) {
            const auto self = input.self_columns(row);
            const real x = input.positions(self);
            real density = input.masses(self) * normalization;
            for (Offset edge = input.offsets(row); edge < input.offsets(row + 1); ++edge) {
                const auto neighbor = input.indices(edge);
                density += contribution(x, input.positions(neighbor), input.masses(neighbor));
            }
            output(row) = density;
            total += density;
        }, sum);
    if (!std::isfinite(sum) || sum > std::numeric_limits<real>::max())
        return Status{StatusCode::numerical_failure, "SPH density or owned sum exceeds real range"};
    return static_cast<real>(sum);
}
} // namespace nexora::sph_fixture
