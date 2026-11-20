#include "Density3D.hpp"

#include <limits>
#include <new>
#include <numbers>

namespace nexora::sph {
double density_at_slot_3d(const CellGrid& grid, LocalIndex self) {
    const auto& mass = grid.fields().mass;
    const double h = grid.smoothing_length();
    int exponent;
    const double fraction = Kokkos::frexp(h, &exponent);
    const auto contribution = [&](double m, double weight) {
        if (h == 1)
            return m * weight;
        int mass_exponent;
        const double mass_fraction = Kokkos::frexp(m, &mass_exponent);
        return Kokkos::ldexp(mass_fraction * weight / (fraction * fraction * fraction),
                             mass_exponent - 3 * exponent);
    };
    double sum = contribution(mass(self), std::numbers::inv_pi);
    grid.for_each_neighbor_of_slot(self, [&](LocalIndex slot, double distance_squared) {
        sum += contribution(mass(slot), m4_weight_3d(Kokkos::sqrt(distance_squared) / h));
    });
    return sum;
}

Result<Values> evaluate_density_3d(const Kokkos::Serial& exec, const CellGrid& grid) {
    try {
        Values density(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.density"),
                       grid.owned_slots().extent(0));
        if (auto status = evaluate_density_3d(exec, grid, density); !status.ok())
            return status;
        return density;
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate 3D SPH density"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot construct 3D SPH density"};
    }
}

Status evaluate_density_3d(const Kokkos::Serial& exec, const CellGrid& grid,
                           const Values& density) {
    const auto rows = grid.owned_slots();
    if (rows.extent(0) != density.extent(0))
        return {StatusCode::invalid_argument, "3D SPH density must have one value per owned row"};
    using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
    try {
        std::size_t invalid = 0;
        Kokkos::parallel_reduce(
            "sph.grid.density_3d", Policy(exec, 0, rows.extent(0)),
            KOKKOS_LAMBDA(std::size_t row, std::size_t& failures) {
                const double sum = density_at_slot_3d(grid, rows(row));
                if (Kokkos::isfinite(sum) && sum <= std::numeric_limits<real>::max())
                    density(row) = static_cast<real>(sum);
                else
                    ++failures;
            },
            invalid);
        if (invalid != 0)
            return Status{StatusCode::numerical_failure, "3D SPH density exceeds real range"};
        return {};
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate 3D SPH density"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot evaluate 3D SPH density"};
    }
}

} // namespace nexora::sph
