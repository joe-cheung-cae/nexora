#include "AdamiWallPressure3D.hpp"

namespace nexora::sph {

real wall_pressure(const NeighborView3D& grid, LocalIndex wall, const NeighborView3D::Indices& row_of_slot,
                     const Values::const_type& density, const Values::const_type& pressure,
                     const std::array<real, 3>& wall_acceleration,
                     const std::array<real, 3>& gravity) {
    real numerator = 0;
    real weights = 0;
    const auto& fields = grid.fields();
    const real h = grid.smoothing_length();
    grid.for_each_neighbor_of_slot(wall, [&](LocalIndex fluid, real d2) {
        if (fields.ghost(fluid))
            return;
        // The common h^-3 normalization cancels in the weighted wall average.
        const real weight = m4_weight_3d(Kokkos::sqrt(d2) / h);
        const auto row = row_of_slot(fluid);
        real head = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const real dx = fields.position(wall, axis) - fields.position(fluid, axis);
            head += (gravity[axis] - wall_acceleration[axis]) * dx;
        }
        numerator += (pressure(row) + density(row) * head) * weight;
        weights += weight;
    });
    return numerator / weights;
}

std::optional<real> wall_coefficient(const NeighborView3D& grid, LocalIndex wall,
                                       const NeighborView3D::Indices& row_of_slot,
                                       const Values::const_type& density,
                                       const Values::const_type& pressure,
                                       const std::array<real, 3>& wall_acceleration,
                                       const std::array<real, 3>& gravity, const TaitEos& eos) {
    const real p = wall_pressure(grid, wall, row_of_slot, density, pressure, wall_acceleration, gravity);
    const real rho = eos.density_from_pressure(p);
    if (!(rho > 0 && Kokkos::isfinite(rho)))
        return std::nullopt;
    return p / (rho * rho);
}

} // namespace nexora::sph
