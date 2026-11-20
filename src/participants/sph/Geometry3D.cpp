#include "Geometry3D.hpp"
#include "GeometryVersions3D.hpp"

#include <data/EntityCapacity.hpp>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace nexora::sph {
namespace {

Status reserve_array(const detail::ResourceBudget& budget, std::size_t count, std::size_t stride,
                     detail::ResourceBudget::Ticket& ticket) {
    const auto size = detail::checked_array_bytes(count, stride);
    if (!size.ok())
        return size.status();
    auto reservation = budget.reserve(size.value());
    if (!reservation.ok())
        return reservation.status();
    ticket = std::move(reservation).value();
    return {};
}

} // namespace

Result<std::shared_ptr<const Geometry3D>>
prepare_geometry_3d(const Kokkos::Serial& exec, const ParticleFields3D& input,
                    std::size_t max_cells, const detail::ResourceBudget& budget,
                    double smoothing_length, const Geometry3D* previous) {
    try {
        detail::ResourceBudget::Ticket values, cells, permutations;
        const auto n = input.ids.extent(0);
        const auto admit = [&](CellGrid::StorageChange change, std::size_t owned,
                               std::size_t cell_count) -> Status {
            if (auto s = reserve_array(budget, n, 4 * sizeof(real), values); !s.ok())
                return s;
            if (change == CellGrid::StorageChange::reuse)
                permutations = previous->permutations;
            else if (auto s = reserve_array(budget, n, 2 * sizeof(LocalIndex), permutations);
                     !s.ok())
                return s;
            if (change != CellGrid::StorageChange::rebuild) {
                cells = previous->cells;
                return {};
            }
            // IDs, ghost flags, owned slots, and the cell table have one shared lifetime.
            const auto ids =
                detail::checked_array_bytes(n, sizeof(std::uint64_t) + sizeof(std::uint8_t));
            const auto rows = detail::checked_array_bytes(owned, sizeof(LocalIndex));
            const auto offsets = detail::checked_array_bytes(cell_count + 1, sizeof(LocalIndex));
            if (!ids.ok() || !rows.ok() || !offsets.ok() ||
                ids.value() > std::numeric_limits<std::size_t>::max() - rows.value() ||
                ids.value() + rows.value() >
                    std::numeric_limits<std::size_t>::max() - offsets.value())
                return {StatusCode::resource_exhausted, "3D SPH cell storage exceeds capacity"};
            auto reservation = budget.reserve(ids.value() + rows.value() + offsets.value());
            if (!reservation.ok())
                return reservation.status();
            cells = std::move(reservation).value();
            return {};
        };
        auto grid = previous ? previous->grid.update(exec, input, max_cells, admit)
                             : CellGrid::build(exec, input, max_cells, admit, smoothing_length);
        if (!grid.ok())
            return grid.status();
        Geometry3D candidate{std::move(values), std::move(cells), std::move(permutations),
                             std::move(grid).value()};
        if (previous) {
            candidate.topology = previous->topology;
            candidate.geometry = previous->geometry;
            candidate.layout = previous->layout;
            auto s =
                update_geometry_versions(exec, previous->grid, candidate.grid, candidate.topology,
                                         candidate.geometry, candidate.layout);
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
