#pragma once

#include "Density3D.hpp"
#include "TaitEos.hpp"

#include <array>
#include <optional>

namespace nexora::sph {

// All fields describe the same stage. The wall slot must have a fluid neighbor,
// and each owned source slot must map to its density and pressure row.
double wall_pressure(const CellGrid&, LocalIndex wall, const CellGrid::Indices& row_of_slot,
                     const Values::const_type& density, const Values::const_type& pressure,
                     const std::array<double, 3>& wall_acceleration,
                     const std::array<double, 3>& gravity);

std::optional<double> wall_coefficient(const CellGrid&, LocalIndex wall,
                                       const CellGrid::Indices& row_of_slot,
                                       const Values::const_type& density,
                                       const Values::const_type& pressure,
                                       const std::array<double, 3>& wall_acceleration,
                                       const std::array<double, 3>& gravity, const TaitEos& eos);

} // namespace nexora::sph
