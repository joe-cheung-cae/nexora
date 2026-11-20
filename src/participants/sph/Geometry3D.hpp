#pragma once

#include <memory>
#include <methods/sph/neighbors/CellGrid.hpp>
#include <participants/ResourceBudget.hpp>

namespace nexora::sph {

struct StagedFields3D {
    detail::ResourceBudget::Ticket reservation;
    ParticleFields3D fields;
};

struct Geometry3D {
    // Copies of a ticket follow exactly the Views that share that allocation group.
    detail::ResourceBudget::Ticket values, cells, permutations;
    CellGrid grid;
    TopologyVersion topology{1};
    GeometryVersion geometry{1};
    LayoutVersion layout{1};
};

Result<StagedFields3D> copy_stage_3d(const Kokkos::Serial&, const ParticleFields3D&,
                                     const detail::ResourceBudget&);
Result<std::shared_ptr<const Geometry3D>>
prepare_geometry_3d(const Kokkos::Serial&, const ParticleFields3D&, std::size_t max_cells,
                    const detail::ResourceBudget&, const Geometry3D* previous = nullptr);

} // namespace nexora::sph
