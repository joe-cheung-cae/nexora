#pragma once
#include "Verlet3D.hpp"

#include <functional>
#include <type_traits>

namespace nexora::sph {

using DensityRate3D = Kokkos::View<double*, Kokkos::HostSpace>;

// Velocity follows sorted source slots, including wall slots.
// Output rates follow owned rows.
Status continuity_rate_3d(const Kokkos::Serial&, const NeighborView3D&,
                          const Velocity3D::const_type& velocity, const DensityRate3D& rate);

// The WCSPH workspace provides disjoint, correctly sized views. Its checked trapezoidal
// correction consumes every rate before any density is published.
void stage_continuity_rate_3d(const Kokkos::Serial&, const NeighborView3D&,
                              const Velocity3D::const_type&, const DensityRate3D&);

using StageVelocityReady3D = Kokkos::View<std::uint8_t*, Kokkos::HostSpace>;
using WriteWallVelocity3D = std::function<bool(LocalIndex)>;
Status stage_wall_continuity_rate_3d(const Kokkos::Serial&, const NeighborView3D&, const Velocity3D&,
                                    const DensityRate3D&, const StageVelocityReady3D&,
                                    const WriteWallVelocity3D&);

struct ContinuityPredictor3D {
    const Velocity3D::const_type& velocity;
    const Acceleration3D::const_type& acceleration;
    double dt;
    const std::array<double, 3>& gravity;

    template <bool observed>
    bool write(std::size_t row, LocalIndex slot, const Velocity3D& output) const {
        for (int axis = 0; axis < 3; ++axis) {
            const double value = Kokkos::fma(dt, acceleration(row, axis) + gravity[axis],
                                             double(velocity(row, axis)));
            if constexpr (observed || std::is_same_v<real, float>) {
                if (!(Kokkos::abs(value) <= std::numeric_limits<real>::max()))
                    return false;
            }
            // A consumed nonfinite double component poisons the pair's dot
            // product. Density correction rejects it before publication.
            output(slot, axis) = static_cast<real>(value);
        }
        return true;
    }
};

// Stage scratch and readiness follow next sorted source slots. Each scratch
// value is produced only upon its first nonzero geometric contribution.
Status stage_predicted_continuity_rate_3d(const Kokkos::Serial&, const NeighborView3D&,
    const Velocity3D&, const DensityRate3D&, const StageVelocityReady3D&,
    const ContinuityPredictor3D&, const NeighborView3D::Indices& new_to_old_slot,
    const NeighborView3D::Indices& old_row, const WriteWallVelocity3D& write_wall);
// With no walls, the correction buffer has one entry per sorted slot. This
// overload uses it for readiness until the subsequent correction overwrites it.
Status stage_predicted_continuity_rate_3d(const Kokkos::Serial&, const NeighborView3D&,
    const Velocity3D&, const DensityRate3D&, const Values& correction_scratch,
    const ContinuityPredictor3D&, const NeighborView3D::Indices& new_to_old_slot,
    const NeighborView3D::Indices& old_row);

// The new grid input followed the old sorted slots. Both old fields have owned-row order.
// old_row maps old owned slots to their respective rows.
Status correct_density_3d(const Kokkos::Serial&, const NeighborView3D::Indices& new_owned,
                          const NeighborView3D::Indices& new_to_old_slot,
                          const NeighborView3D::Indices& old_row, const Values::const_type& density,
                          const DensityRate3D& rate_n, const DensityRate3D& rate_star,
                          double dt, const Values& corrected);

} // namespace nexora::sph
