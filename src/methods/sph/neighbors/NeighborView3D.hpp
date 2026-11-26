#pragma once

#include <methods/sph/ParticleFields3D.hpp>

#include <array>
#include <memory>
#include <utility>

namespace nexora::sph {

// A sorted stage query retains only the arrays used in its stencils.
class NeighborView3D {
  public:
    using Indices = Kokkos::View<const LocalIndex*, Kokkos::HostSpace>;
    NeighborView3D(const NeighborView3D&) = default;
    NeighborView3D(NeighborView3D&&) = default;
    // The displaced arrays and their reservations leave together in replacement.
    NeighborView3D& operator=(NeighborView3D replacement) {
        swap(replacement);
        return *this;
    }

    real smoothing_length() const { return smoothing_length_; }
    real support_radius() const { return 2 * smoothing_length_; }

    const ParticleFields3D& fields() const {
        return fields_;
    }
    const Indices& owned_slots() const {
        return owned_slots_;
    }
    // row indexes owned_slots(); callback receives a sorted source slot and squared distance.
    template <class Callback> void for_each_neighbor(std::size_t row, Callback&& visit) const {
        for_each_neighbor_of_slot(owned_slots_(row), std::forward<Callback>(visit));
    }

    // Source-centered stencils are also needed for locally reconstructed ghost density.
    template <class Callback>
    void for_each_neighbor_of_slot(LocalIndex self, Callback&& visit) const {
        for_each_neighbor_impl(self,
            [&](LocalIndex j, real d2, real, real, real) { visit(j, d2); });
    }

    // The displacement is position(neighbor) - position(center), using the
    // same real differences already evaluated by the neighbor cutoff.
    template <class Callback>
    void for_each_neighbor_with_displacement(std::size_t row, Callback&& visit) const {
        for_each_neighbor_impl(owned_slots_(row), std::forward<Callback>(visit));
    }

  private:
    template <class Callback>
    void for_each_neighbor_impl(LocalIndex self, Callback&& visit) const {
        std::array<real, 3> center{};
        std::array<std::size_t, 3> lo{}, hi{};
        for (int axis = 0; axis < 3; ++axis) {
            center[axis] = fields_.position(self, axis);
            const auto cell = relative_bin(absolute_bin(center[axis]), origin_[axis]);
            lo[axis] = cell == 0 ? cell : cell - 1;
            hi[axis] = cell + 1 < shape_[axis] ? cell + 1 : cell;
        }
        // Contiguous x cells share a single slot range for each (y,z) pair.
        for (auto z = lo[2]; z <= hi[2]; ++z) {
            for (auto y = lo[1]; y <= hi[1]; ++y) {
                const auto base = shape_[0] * (y + shape_[1] * z);
                const auto end = cell_begin_(base + hi[0] + 1);
                for (auto j = cell_begin_(base + lo[0]); j < end; ++j) {
                    if (j == self)
                        continue;
                    const real dx = fields_.position(j, 0) - center[0];
                    const real dy = fields_.position(j, 1) - center[1];
                    const real dz = fields_.position(j, 2) - center[2];
                    // Fix the rounding order for near-cutoff diagonals in every build.
                    const real d2 = Kokkos::fma(dz, dz, Kokkos::fma(dy, dy, dx * dx));
                    if (d2 <= radius_squared_)
                        visit(j, d2, dx, dy, dz);
                }
            }
        }
    }

    friend class CellGrid;
    NeighborView3D() = default;
    void swap(NeighborView3D& other) {
        using std::swap;
        swap(smoothing_length_, other.smoothing_length_);
        swap(radius_squared_, other.radius_squared_);
        swap(field_storage_, other.field_storage_);
        swap(cells_ticket_, other.cells_ticket_);
        swap(fields_, other.fields_);
        swap(owned_slots_, other.owned_slots_);
        swap(cell_begin_, other.cell_begin_);
        swap(origin_, other.origin_);
        swap(shape_, other.shape_);
    }
    std::int64_t absolute_bin(real x) const;
    static std::size_t relative_bin(std::int64_t bin, std::int64_t origin) {
        return static_cast<std::size_t>(bin - origin);
    }

    real smoothing_length_ = 1;
    real radius_squared_ = 4;
    std::array<std::shared_ptr<const void>, 3> field_storage_; // position, mass, IDs/ghost
    std::shared_ptr<const void> cells_ticket_;
    ParticleFields3D fields_;
    Indices owned_slots_, cell_begin_;
    std::array<std::int64_t, 3> origin_{};
    std::array<std::size_t, 3> shape_{};
};
} // namespace nexora::sph
