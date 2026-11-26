#include "Density3D.hpp"

#include <limits>
#include <new>
#include <numbers>

namespace nexora::sph {
real density_at_slot_3d(const NeighborView3D& grid, LocalIndex self) {
    const auto& mass = grid.fields().mass;
    const real h = grid.smoothing_length();
    real sum = mass(self) * std::numbers::inv_pi_v<real>;
    grid.for_each_neighbor_of_slot(self, [&](LocalIndex slot, real distance_squared) {
        sum += mass(slot) * m4_weight_3d(Kokkos::sqrt(distance_squared) / h);
    });
    return sum / (h * h * h);
}

Status evaluate_density_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
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
                const real sum = density_at_slot_3d(grid, rows(row));
                if (Kokkos::isfinite(sum))
                    density(row) = sum;
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
