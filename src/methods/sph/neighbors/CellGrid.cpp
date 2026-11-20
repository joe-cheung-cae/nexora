#include "CellGrid.hpp"

#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
using Slots = CellGrid::Indices::non_const_type;

double floor_cell_index(double position, double width) {
    // The quotient can round up at a nonbinary cell boundary, such as 1 / 0.2.
    const double index = std::floor(position / width);
    return index - (std::fma(-index, width, position) < 0);
}

struct SortEntry {
    std::size_t cell;
    std::uint64_t id;
    LocalIndex input;
};
struct PackedValues {
    decltype(ParticleFields3D::position)::non_const_type position;
    decltype(ParticleFields3D::mass)::non_const_type mass;

    PackedValues(const Kokkos::Serial& exec, std::size_t n)
        : position(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.position"), n),
          mass(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.mass"), n) {}

    void gather(std::size_t slot, const ParticleFields3D& input, LocalIndex i) const {
        for (int axis = 0; axis < 3; ++axis)
            position(slot, axis) = input.position(i, axis);
        mass(slot) = input.mass(i);
    }
};
} // namespace

bool CellGrid::components_within_support(LocalIndex self, LocalIndex neighbor) const {
    for (int axis = 0; axis < 3; ++axis) {
        const double x = fields_.position(self, axis);
        const double y = fields_.position(neighbor, axis);
        const double hi = std::max(x, y), neg_lo = -std::min(x, y);
        const double distance = hi + neg_lo;
        if (distance > support_radius())
            return false;
        if (distance == support_radius()) {
            // At a rounded tie recover the exact sign of the subtraction residual.
            const double residual = (std::max(hi, neg_lo) - distance) + std::min(hi, neg_lo);
            if (residual > 0)
                return false;
        }
    }
    return true;
}

Result<CellGrid> CellGrid::build(const Kokkos::Serial& exec, const ParticleFields3D& input,
                                 std::size_t max_cells, const AdmitStorage& admit,
                                 double smoothing_length) {
    return prepare(exec, input, max_cells, nullptr, admit, smoothing_length);
}

Result<CellGrid> CellGrid::update(const Kokkos::Serial& exec, const ParticleFields3D& input,
                                  std::size_t max_cells, const AdmitStorage& admit) const {
    return prepare(exec, input, max_cells, this, admit, smoothing_length_);
}

std::int64_t CellGrid::absolute_bin(double x) const {
    return static_cast<std::int64_t>(floor_cell_index(x, support_radius()));
}

std::size_t CellGrid::cell_of(const ParticleFields3D& fields, std::size_t i) const {
    std::array<std::size_t, 3> bin;
    for (int axis = 0; axis < 3; ++axis)
        bin[axis] = relative_bin(absolute_bin(fields.position(i, axis)), origin_[axis]);
    return bin[0] + shape_[0] * (bin[1] + shape_[1] * bin[2]);
}

// The new bounding box matches the old one. Verify membership by ID before sharing offsets.
Result<bool> CellGrid::try_reuse(const Kokkos::Serial& exec, const ParticleFields3D& input,
                                 const CellGrid& old, const AdmitStorage& admit) {
    const auto n = input.ids.extent(0);
    const Policy particles(exec, 0, n);
    bool reordered = false, reusable = true;
    Kokkos::parallel_for("sph.grid.check_membership", particles, [&](std::size_t i) {
        const auto slot = old.input_to_sorted_(i);
        if (old.fields_.ids(slot) != input.ids(i)) {
            reordered = true;
            return;
        }
        const auto c = cell_of(input, i);
        if (slot < old.cell_begin_(c) || slot >= old.cell_begin_(c + 1) ||
            old.fields_.ghost(slot) != input.ghost(i))
            reusable = false;
    });
    if (!reusable)
        return false;

    Indices to_input = old.sorted_to_input_, from_input = old.input_to_sorted_;
    if (reordered) {
        // Cell-local IDs are sorted and unique. n is a valid missing marker because
        // particle slots range from zero to n-1, even at full LocalIndex capacity.
        Slots sorted(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.to_input"), n);
        Slots inverse(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.from_input"),
                      n);
        Kokkos::deep_copy(exec, sorted, static_cast<LocalIndex>(n));
        Kokkos::parallel_for("sph.grid.remap_input", particles, [&](std::size_t i) {
            const auto c = cell_of(input, i);
            auto first = old.cell_begin_(c);
            const auto end = old.cell_begin_(c + 1);
            auto last = end;
            while (first < last) {
                const auto mid = first + (last - first) / 2;
                if (old.fields_.ids(mid) < input.ids(i))
                    first = mid + 1;
                else
                    last = mid;
            }
            if (first == end || old.fields_.ids(first) != input.ids(i) ||
                old.fields_.ghost(first) != input.ghost(i) || sorted(first) != n) {
                reusable = false;
                return;
            }
            sorted(first) = static_cast<LocalIndex>(i);
            inverse(i) = first;
        });
        if (!reusable)
            return false;
        to_input = sorted;
        from_input = inverse;
    }

    if (admit) {
        const auto route = reordered ? StorageChange::remap : StorageChange::reuse;
        const auto status = admit(route, old.owned_slots_.extent(0), old.cell_begin_.extent(0) - 1);
        if (!status.ok())
            return status;
    }

    const PackedValues values(exec, n);
    Kokkos::parallel_for("sph.grid.refresh_values", particles,
                         [&](std::size_t s) { values.gather(s, input, to_input(s)); });
    exec.fence("sph.grid.updated_snapshot_ready");
    fields_ = {values.position, values.mass, old.fields_.ids, old.fields_.ghost};
    cell_begin_ = old.cell_begin_;
    owned_slots_ = old.owned_slots_;
    sorted_to_input_ = to_input;
    input_to_sorted_ = from_input;
    return true;
}

Result<CellGrid> CellGrid::prepare(const Kokkos::Serial& exec, const ParticleFields3D& input,
                                   std::size_t max_cells, const CellGrid* previous,
                                   const AdmitStorage& admit, double smoothing_length) {
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
        return Status{StatusCode::invalid_state, "3D SPH grid requires active Kokkos"};
    const auto n = input.position.extent(0);
    if (input.mass.extent(0) != n || input.ids.extent(0) != n || input.ghost.extent(0) != n)
        return Status{StatusCode::invalid_argument, "3D SPH particle field extents disagree"};
    if (n > std::numeric_limits<LocalIndex>::max() ||
        n > std::numeric_limits<std::size_t>::max() / std::max(sizeof(SortEntry), 3 * sizeof(real)))
        return Status{StatusCode::resource_exhausted,
                      "3D SPH particle count exceeds index capacity"};
    if (n &&
        (!input.position.data() || !input.mass.data() || !input.ids.data() || !input.ghost.data()))
        return Status{StatusCode::invalid_argument, "3D SPH input has a null particle field"};

    CellGrid grid;
    grid.smoothing_length_ = smoothing_length;
    const double radius = grid.support_radius();
    grid.radius_squared_ = radius * radius;
    if (!(smoothing_length > 0) || !(grid.radius_squared_ > 0) ||
        !std::isfinite(grid.radius_squared_))
        return Status{StatusCode::invalid_argument,
                      "3D SPH requires positive smoothing length with finite positive squared support"};
    grid.origin_.fill(std::numeric_limits<std::int64_t>::max());
    std::array<std::int64_t, 3> largest;
    largest.fill(std::numeric_limits<std::int64_t>::min());
    std::size_t owned = 0;
    bool invalid_field = false, unsupported_bin = false;
    const Policy particles(exec, 0, n);
    // Serial preflight computes the bounding box before allocating the dense table.
    Kokkos::parallel_for("sph.grid.validate", particles, [&](std::size_t i) {
        if (!std::isfinite(input.mass(i)) || input.mass(i) <= 0 || input.ghost(i) > 1)
            invalid_field = true;
        owned += input.ghost(i) == 0;
        for (int axis = 0; axis < 3; ++axis) {
            const double x = input.position(i, axis);
            if (!std::isfinite(x)) {
                invalid_field = true;
            } else {
                const double bin = floor_cell_index(x, radius);
                if (bin < -0x1p63 || bin >= 0x1p63) {
                    unsupported_bin = true;
                } else {
                    const auto key = static_cast<std::int64_t>(bin);
                    grid.origin_[axis] = std::min(grid.origin_[axis], key);
                    largest[axis] = std::max(largest[axis], key);
                }
            }
        }
    });
    if (invalid_field)
        return Status{StatusCode::invalid_argument, "invalid 3D SPH position, mass or ghost flag"};
    if (unsupported_bin)
        return Status{StatusCode::unsupported,
                      "3D SPH absolute cell coordinate exceeds signed 64-bit range"};

    const auto limit =
        std::min(max_cells, std::numeric_limits<std::size_t>::max() / sizeof(LocalIndex) - 1);
    std::size_t cells = n ? 1 : 0;
    for (int axis = 0; n && axis < 3; ++axis) {
        const auto span = static_cast<std::uint64_t>(largest[axis]) -
                          static_cast<std::uint64_t>(grid.origin_[axis]);
        if (span >= limit / cells)
            return Status{StatusCode::resource_exhausted,
                          "3D SPH dense cell count exceeds capacity"};
        grid.shape_[axis] = static_cast<std::size_t>(span) + 1;
        cells *= grid.shape_[axis];
    }

    try {
        if (previous && previous->fields_.ids.extent(0) == n && previous->origin_ == grid.origin_ &&
            previous->shape_ == grid.shape_ && previous->owned_slots_.extent(0) == owned &&
            previous->cell_begin_.extent(0) == cells + 1) {
            auto reuse = grid.try_reuse(exec, input, *previous, admit);
            if (!reuse.ok())
                return reuse.status();
            if (reuse.value())
                return grid;
        }
        if (admit) {
            const auto status = admit(StorageChange::rebuild, owned, cells);
            if (!status.ok())
                return status;
        }

        Kokkos::View<SortEntry*, Kokkos::HostSpace> keys(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.keys"), n);
        Kokkos::parallel_for("sph.grid.keys", particles, [&](std::size_t i) {
            keys(i) = {grid.cell_of(input, i), input.ids(i), static_cast<LocalIndex>(i)};
        });
        if (n > 1) {
            Kokkos::sort(exec, keys,
                         [](const SortEntry& a, const SortEntry& b) { return a.id < b.id; });
            bool duplicate = false;
            Kokkos::parallel_for("sph.grid.unique_ids", Policy(exec, 1, n), [&](std::size_t i) {
                if (keys(i - 1).id == keys(i).id)
                    duplicate = true;
            });
            if (duplicate)
                return Status{StatusCode::invalid_argument, "3D SPH particle IDs must be unique"};
            Kokkos::sort(exec, keys, [](const SortEntry& a, const SortEntry& b) {
                return a.cell == b.cell ? a.id < b.id : a.cell < b.cell;
            });
        }

        const PackedValues values(exec, n);
        decltype(input.ids)::non_const_type ids(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.ids"), n);
        decltype(input.ghost)::non_const_type ghost(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.ghost"), n);
        Slots sorted(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.to_input"), n);
        Slots inverse(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.from_input"),
                      n);
        Slots rows(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.grid.owned_slots"),
                   owned);
        Slots begin(Kokkos::view_alloc(exec, "sph.grid.cell_begin"), cells + 1);
        Kokkos::parallel_for("sph.grid.gather", particles, [&](std::size_t s) {
            const auto key = keys(s);
            sorted(s) = key.input;
            inverse(key.input) = static_cast<LocalIndex>(s);
            values.gather(s, input, key.input);
            ids(s) = key.id;
            ghost(s) = input.ghost(key.input);
            ++begin(key.cell);
        });
        Kokkos::parallel_scan("sph.grid.cell_offsets", Policy(exec, 0, cells + 1),
                              [&](std::size_t c, LocalIndex& sum, bool final) {
                                  const auto count = begin(c);
                                  if (final)
                                      begin(c) = sum;
                                  sum += count;
                              });
        Kokkos::parallel_scan("sph.grid.owned_rows", particles,
                              [&](std::size_t s, LocalIndex& sum, bool final) {
                                  if (!ghost(s)) {
                                      if (final)
                                          rows(sum) = static_cast<LocalIndex>(s);
                                      ++sum;
                                  }
                              });
        exec.fence("sph.grid.snapshot_ready");
        grid.fields_ = {values.position, values.mass, ids, ghost};
        grid.cell_begin_ = begin;
        grid.sorted_to_input_ = sorted;
        grid.input_to_sorted_ = inverse;
        grid.owned_slots_ = rows;
        return grid;
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate 3D SPH cell grid"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot build 3D SPH cell grid"};
    }
}
} // namespace nexora::sph
