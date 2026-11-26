#pragma once

#include "Density3D.hpp"
#include "TaitEos.hpp"

#include <array>
#include <optional>

namespace nexora::sph {

// All fields describe the same stage. The wall slot must have a fluid neighbor,
// and each owned source slot must map to its density and pressure row.
real wall_pressure(const NeighborView3D&, LocalIndex wall, const NeighborView3D::Indices& row_of_slot,
                     const Values::const_type& density, const Values::const_type& pressure,
                     const std::array<real, 3>& wall_acceleration,
                     const std::array<real, 3>& gravity);

std::optional<real> wall_coefficient(const NeighborView3D&, LocalIndex wall,
                                       const NeighborView3D::Indices& row_of_slot,
                                       const Values::const_type& density,
                                       const Values::const_type& pressure,
                                       const std::array<real, 3>& wall_acceleration,
                                       const std::array<real, 3>& gravity, const TaitEos& eos);

} // namespace nexora::sph
