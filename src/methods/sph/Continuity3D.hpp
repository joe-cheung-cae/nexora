#pragma once
#include "Verlet3D.hpp"

namespace nexora::sph {

using DensityRate3D = Kokkos::View<double*, Kokkos::HostSpace>;

// Velocity follows sorted source slots, including wall slots.
// Output rates follow owned rows.
Status continuity_rate_3d(const Kokkos::Serial&, const NeighborView3D&,
                          const Velocity3D::const_type& velocity, const DensityRate3D& rate);

// The new grid input followed the old sorted slots. Both old fields have owned-row order.
// old_row maps old owned slots to their respective rows.
void correct_density_3d(const Kokkos::Serial&, const NeighborView3D::Indices& new_owned,
                        const NeighborView3D::Indices& new_to_old_slot,
                        const NeighborView3D::Indices& old_row, const Values::const_type& density,
                        const DensityRate3D& rate_n, const DensityRate3D& rate_star,
                        double dt, const Values& corrected);

} // namespace nexora::sph
