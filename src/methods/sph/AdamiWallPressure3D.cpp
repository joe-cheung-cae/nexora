#include "AdamiWallPressure3D.hpp"

#include <limits>

namespace nexora::sph {

double wall_pressure(const NeighborView3D& grid, LocalIndex wall, const NeighborView3D::Indices& row_of_slot,
                     const Values::const_type& density, const Values::const_type& pressure,
                     const std::array<double, 3>& wall_acceleration,
                     const std::array<double, 3>& gravity) {
    double numerator = 0;
    double weights = 0;
    const auto& fields = grid.fields();
    const double h = grid.smoothing_length();
    grid.for_each_neighbor_of_slot(wall, [&](LocalIndex fluid, double d2) {
        if (fields.ghost(fluid))
            return;
        // The common h^-3 normalization cancels in the weighted wall average.
        const double weight = m4_weight_3d(Kokkos::sqrt(d2) / h);
        const auto row = row_of_slot(fluid);
        double head = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const double dx = double(fields.position(wall, axis)) - fields.position(fluid, axis);
            head += (gravity[axis] - wall_acceleration[axis]) * dx;
        }
        numerator += (pressure(row) + density(row) * head) * weight;
        weights += weight;
    });
    return numerator / weights;
}

std::optional<double> wall_coefficient(const NeighborView3D& grid, LocalIndex wall,
                                       const NeighborView3D::Indices& row_of_slot,
                                       const Values::const_type& density,
                                       const Values::const_type& pressure,
                                       const std::array<double, 3>& wall_acceleration,
                                       const std::array<double, 3>& gravity, const TaitEos& eos) {
    const double raw = wall_pressure(grid, wall, row_of_slot, density, pressure, wall_acceleration,
                                     gravity);
    constexpr double maximum = std::numeric_limits<real>::max();
    if (!(Kokkos::abs(raw) <= maximum))
        return std::nullopt;
    const real p = static_cast<real>(raw);
    const double rho = eos.density_from_pressure(p);
    if (!(Kokkos::abs(rho) <= maximum) || static_cast<real>(rho) <= 0)
        return std::nullopt;
    const double wall_density = static_cast<real>(rho);
    return double(p) / (wall_density * wall_density);
}

} // namespace nexora::sph
