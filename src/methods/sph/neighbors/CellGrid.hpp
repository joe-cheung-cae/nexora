#pragma once

#include <Kokkos_Core.hpp>
#include <nexora/core/Result.hpp>
#include <nexora/core/Types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

namespace nexora::sph {

struct ParticleFields3D {
    Kokkos::View<const real* [3], Kokkos::LayoutLeft, Kokkos::HostSpace> position;
    Kokkos::View<const real*, Kokkos::HostSpace> mass;
    Kokkos::View<const std::uint64_t*, Kokkos::HostSpace> ids;
    Kokkos::View<const std::uint8_t*, Kokkos::HostSpace> ghost;
};

// Immutable sorted snapshot of one nonperiodic geometry stage, with h=1 and R=2.
class CellGrid {
  public:
    using Indices = Kokkos::View<const LocalIndex*, Kokkos::HostSpace>;
    enum class StorageChange { rebuild, remap, reuse };
    // Called synchronously for the selected route before its retained storage
    // is allocated or published. A rejected route ends the operation.
    using AdmitStorage = std::function<Status(StorageChange, std::size_t owned, std::size_t cells)>;
    static Result<CellGrid> build(const Kokkos::Serial&, const ParticleFields3D&,
                                  std::size_t max_cells, const AdmitStorage& admit = {});
    // Return a fresh field snapshot. Share cell_begin when each ID retains its cell
    // and ownership. Input rows may be reordered between calls.
    Result<CellGrid> update(const Kokkos::Serial&, const ParticleFields3D&, std::size_t max_cells,
                            const AdmitStorage& admit = {}) const;

    const ParticleFields3D& fields() const {
        return fields_;
    }
    const Indices& owned_slots() const {
        return owned_slots_;
    }
    const Indices& cell_begin() const {
        return cell_begin_;
    }
    const Indices& sorted_to_input() const {
        return sorted_to_input_;
    }
    const Indices& input_to_sorted() const {
        return input_to_sorted_;
    }

    // row indexes owned_slots(); callback receives a sorted source slot and squared distance.
    template <class Callback> void for_each_neighbor(std::size_t row, Callback&& visit) const {
        for_each_neighbor_of_slot(owned_slots_(row), std::forward<Callback>(visit));
    }

    // Source-centered stencils are also needed for locally reconstructed ghost density.
    template <class Callback>
    void for_each_neighbor_of_slot(LocalIndex self, Callback&& visit) const {
        std::array<double, 3> center{};
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
                    const double dx = double(fields_.position(j, 0)) - center[0];
                    const double dy = double(fields_.position(j, 1)) - center[1];
                    const double dz = double(fields_.position(j, 2)) - center[2];
                    // Fix the rounding order for near-cutoff diagonals in every build.
                    const double d2 = Kokkos::fma(dz, dz, Kokkos::fma(dy, dy, dx * dx));
                    // A component just outside R can round to R on subtraction.
                    // Only a squared-distance tie can admit that candidate.
                    if (d2 < 4.0 || (d2 == 4.0 && components_within_support(self, j)))
                        visit(j, d2);
                }
            }
        }
    }

  private:
    CellGrid() = default;
    static Result<CellGrid> prepare(const Kokkos::Serial&, const ParticleFields3D&,
                                    std::size_t max_cells, const CellGrid* previous,
                                    const AdmitStorage&);
    Result<bool> try_reuse(const Kokkos::Serial&, const ParticleFields3D&, const CellGrid& previous,
                           const AdmitStorage&);
    std::size_t cell_of(const ParticleFields3D&, std::size_t i) const;
    bool components_within_support(LocalIndex self, LocalIndex neighbor) const;
    static std::int64_t absolute_bin(double x) {
        // Halving the smallest negative subnormal can round to negative zero.
        return x < 0 && x > -2 ? -1 : static_cast<std::int64_t>(Kokkos::floor(x * .5));
    }
    static std::size_t relative_bin(std::int64_t bin, std::int64_t origin) {
        // The nonnegative difference may exceed INT64_MAX across a signed origin.
        return static_cast<std::uint64_t>(bin) - static_cast<std::uint64_t>(origin);
    }

    ParticleFields3D fields_;
    Indices owned_slots_, cell_begin_, sorted_to_input_, input_to_sorted_;
    std::array<std::int64_t, 3> origin_{};
    std::array<std::size_t, 3> shape_{};
};
} // namespace nexora::sph
