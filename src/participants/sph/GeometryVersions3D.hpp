#pragma once

#include <Kokkos_Sort.hpp>
#include <cstddef>
#include <cstdint>
#include <data/EntityCapacity.hpp>
#include <methods/sph/neighbors/CellGrid.hpp>

namespace nexora::sph {

inline Status update_geometry_versions(const Kokkos::Serial& exec, const CellGrid& before,
                                       const CellGrid& after, TopologyVersion& topology,
                                       GeometryVersion& geometry, LayoutVersion& layout) {
    using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
    const auto& old_fields = before.fields();
    const auto& new_fields = after.fields();
    const auto n = new_fields.ids.extent(0);
    const bool resized = old_fields.ids.extent(0) != n;
    // Topology describes the particle domain and ownership. Neighbor edges can
    // change with geometry even when the domain and ownership stay fixed.
    bool topology_changed = resized;
    bool geometry_changed = resized;
    bool layout_changed = resized;
    if (!resized) {
        const Policy slots(exec, 0, n);
        const auto old_rows = before.owned_slots();
        const auto new_rows = after.owned_slots();
        const auto old_input = before.sorted_to_input();
        const auto new_input = after.sorted_to_input();
        bool slots_changed = false;
        layout_changed = old_rows.extent(0) != new_rows.extent(0);
        if (old_fields.ids.data() != new_fields.ids.data() ||
            old_input.data() != new_input.data() || old_rows.data() != new_rows.data()) {
            Kokkos::parallel_for("sph.grid.compare_layout", slots, [&](std::size_t i) {
                slots_changed |= old_fields.ids(i) != new_fields.ids(i);
                layout_changed |= old_input(i) != new_input(i);
                if (i < old_rows.extent(0) && i < new_rows.extent(0))
                    layout_changed |= old_rows(i) != new_rows(i);
            });
        }
        layout_changed |= slots_changed;

        const auto compare_particle = [&](std::size_t old_slot, std::size_t new_slot) {
            topology_changed |= old_fields.ghost(old_slot) != new_fields.ghost(new_slot);
            for (int axis = 0; axis < 3; ++axis)
                geometry_changed |=
                    old_fields.position(old_slot, axis) != new_fields.position(new_slot, axis);
        };
        if (!slots_changed) {
            if (old_fields.position.data() != new_fields.position.data() ||
                old_fields.ghost.data() != new_fields.ghost.data())
                Kokkos::parallel_for("sph.grid.compare_geometry", slots,
                                     [&](std::size_t i) { compare_particle(i, i); });
        } else {
            // A rebuilt grid can put the same IDs into unrelated sorted slots.
            // First try aligning by input order, which often remains stable.
            const auto old_slots = before.input_to_sorted();
            const auto new_slots = after.input_to_sorted();
            bool input_ids_changed = false;
            Kokkos::parallel_for("sph.grid.compare_input_ids", slots, [&](std::size_t i) {
                input_ids_changed |= old_fields.ids(old_slots(i)) != new_fields.ids(new_slots(i));
            });
            if (!input_ids_changed) {
                Kokkos::parallel_for(
                    "sph.grid.compare_geometry_by_input", slots,
                    [&](std::size_t i) { compare_particle(old_slots(i), new_slots(i)); });
            } else {
                struct ParticleSlot {
                    std::uint64_t id;
                    LocalIndex slot;
                };
                using ById = Kokkos::View<ParticleSlot*, Kokkos::HostSpace>;
                const ById old_by_id(
                    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.old_by_id"), n);
                const ById new_by_id(
                    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.new_by_id"), n);
                Kokkos::parallel_for("sph.grid.map_geometry_ids", slots, [&](std::size_t i) {
                    const auto slot = static_cast<LocalIndex>(i);
                    old_by_id(i) = {old_fields.ids(i), slot};
                    new_by_id(i) = {new_fields.ids(i), slot};
                });
                const auto by_id = [](const ParticleSlot& a, const ParticleSlot& b) {
                    return a.id < b.id;
                };
                Kokkos::sort(exec, old_by_id, by_id);
                Kokkos::sort(exec, new_by_id, by_id);
                Kokkos::parallel_for("sph.grid.compare_geometry_by_id", slots, [&](std::size_t i) {
                    const auto old_particle = old_by_id(i);
                    const auto new_particle = new_by_id(i);
                    if (old_particle.id == new_particle.id) {
                        compare_particle(old_particle.slot, new_particle.slot);
                    } else {
                        topology_changed = true;
                        geometry_changed = true;
                    }
                });
            }
        }
    }

    // Advance copies first, so an overflow leaves every output untouched.
    const auto next_topology = topology_changed ? detail::checked_next_version(topology)
                                                : Result<TopologyVersion>{topology};
    if (!next_topology.ok())
        return next_topology.status();
    const auto next_geometry = geometry_changed ? detail::checked_next_version(geometry)
                                                : Result<GeometryVersion>{geometry};
    if (!next_geometry.ok())
        return next_geometry.status();
    const auto next_layout =
        layout_changed ? detail::checked_next_version(layout) : Result<LayoutVersion>{layout};
    if (!next_layout.ok())
        return next_layout.status();
    topology = next_topology.value();
    geometry = next_geometry.value();
    layout = next_layout.value();
    return {};
}

} // namespace nexora::sph
