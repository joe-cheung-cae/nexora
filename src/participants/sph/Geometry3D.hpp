#pragma once

#include <memory>
#include <methods/sph/neighbors/CellGrid.hpp>
#include <participants/ResourceBudget.hpp>

namespace nexora::sph {

struct Geometry3D {
    CellGrid grid;
    TopologyVersion topology{1};
    GeometryVersion geometry{1};
    LayoutVersion layout{1};
};

Result<std::shared_ptr<const Geometry3D>>
prepare_geometry_3d(const Kokkos::Serial&, const ParticleFields3D&, std::size_t max_cells,
                    const detail::ResourceBudget&, double smoothing_length,
                    const Geometry3D* previous = nullptr);

} // namespace nexora::sph
