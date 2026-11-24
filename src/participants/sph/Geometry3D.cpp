#include "Geometry3D.hpp"
#include "GeometryVersions3D.hpp"

#include <exception>
#include <new>
#include <utility>

namespace nexora::sph {
namespace {

GeometryLayout3D layout_of(const CellGrid& grid) {
    return {grid.fields(), grid.owned_slots(), grid.sorted_to_input(), grid.input_to_sorted()};
}
} // namespace

Result<std::shared_ptr<const Geometry3D>>
prepare_geometry_3d(const Kokkos::Serial& exec, const ParticleFields3D& input,
                    std::size_t max_cells, const detail::ResourceBudget& budget,
                    double smoothing_length, const Geometry3D* previous) {
    try {
        const auto admit = [&](CellGrid::StorageGroup, std::size_t bytes) {
            return budget.reserve(bytes);
        };
        auto grid = previous ? previous->grid.update(exec, input, max_cells, admit)
                             : CellGrid::build(exec, input, max_cells, admit, smoothing_length);
        if (!grid.ok())
            return grid.status();
        Geometry3D candidate{std::move(grid).value()};
        if (previous) {
            candidate.topology = previous->topology;
            candidate.geometry = previous->geometry;
            candidate.layout = previous->layout;
            auto s = update_geometry_versions(exec, layout_of(previous->grid), layout_of(candidate.grid),
                                              budget, candidate.topology, candidate.geometry, candidate.layout);
            if (!s.ok())
                return s;
        }
        return std::make_shared<const Geometry3D>(std::move(candidate));
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate 3D SPH geometry"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot prepare 3D SPH geometry"};
    }
}

} // namespace nexora::sph
