#pragma once

#include "NeighborView3D.hpp"
#include <nexora/core/Result.hpp>

#include <functional>

namespace nexora::sph {

// Immutable sorted snapshot of one nonperiodic geometry stage, with a fixed per-case support radius R=2h.
class CellGrid {
  public:
    using Indices = NeighborView3D::Indices;
    enum class StorageGroup { position, mass, identity, cells, permutations, sort_keys };
    using Ticket = std::shared_ptr<const void>;
    using AdmitStorage = std::function<Result<Ticket>(StorageGroup, std::size_t bytes)>;

    CellGrid(const CellGrid&) = default;
    CellGrid(CellGrid&&) = default;
    CellGrid& operator=(CellGrid replacement) {
        using std::swap;
        query_.swap(replacement.query_);
        swap(permutations_ticket_, replacement.permutations_ticket_);
        swap(sorted_to_input_, replacement.sorted_to_input_);
        swap(input_to_sorted_, replacement.input_to_sorted_);
        swap(input_order_, replacement.input_order_);
        swap(positions_changed_, replacement.positions_changed_);
        return *this;
    }

    static Result<CellGrid> build(const Kokkos::Serial&, const ParticleFields3D&,
                                  std::size_t max_cells, const AdmitStorage& admit = {},
                                  real smoothing_length = 1);
    // Return a fresh field snapshot. Share cell_begin when each ID retains its cell
    // and ownership. Input rows may be reordered between calls.
    Result<CellGrid> update(const Kokkos::Serial&, const ParticleFields3D&, std::size_t max_cells,
                            const AdmitStorage& admit = {}) const;
    // Positions follow this grid's sorted slots. Mass, IDs and ownership come
    // from the existing immutable grid.
    Result<CellGrid> update_positions(const Kokkos::Serial&,
                                     const decltype(ParticleFields3D::position)&,
                                     std::size_t max_cells, const AdmitStorage& admit = {}) const;

    const NeighborView3D& neighbors() const { return query_; }
    real smoothing_length() const { return query_.smoothing_length(); }
    real support_radius() const { return query_.support_radius(); }
    const std::array<Ticket, 3>& field_storage() const { return query_.field_storage_; }
    const ParticleFields3D& fields() const { return query_.fields(); }
    const Indices& owned_slots() const { return query_.owned_slots(); }
    const Indices& cell_begin() const { return query_.cell_begin_; }
    const Indices& sorted_to_input() const {
        return sorted_to_input_;
    }
    const Indices& input_to_sorted() const {
        return input_to_sorted_;
    }
  private:
    friend class WcsphStep3D;
    enum class InputKind { external, positions, initial, committed };
    enum class InputOrder { unknown, identity, nonidentity };
    // The solver's initial/position/rebuild entries establish this metadata.
    // General external build/update paths do not consume or compute it.
    bool input_is_sorted() const { return input_order_ == InputOrder::identity; }
    // Only the step owner can supply assembled ownership or immutable accepted
    // geometry. Initial assembly validates all masses and the combined ID set.
    // General build/update callers still validate every external field.
    static Result<CellGrid> build_initial(const Kokkos::Serial&, const ParticleFields3D&,
                                         std::size_t fluid_count, std::size_t max_cells,
                                         const AdmitStorage&, real smoothing_length);
    static Result<CellGrid> rebuild_committed(const Kokkos::Serial&, const ParticleFields3D&,
                                             std::size_t owned_count, std::size_t max_cells,
                                             const AdmitStorage&, real smoothing_length);
    CellGrid() = default;
    static Result<CellGrid> prepare(const Kokkos::Serial&, const ParticleFields3D&,
                                    std::size_t max_cells, const CellGrid* previous,
                                    const AdmitStorage&, real smoothing_length,
                                    InputKind = InputKind::external, std::size_t owned_count = 0);
    template <bool PositionsOnly>
    Result<bool> try_reuse(const Kokkos::Serial&, const ParticleFields3D&, const CellGrid& previous,
                           const AdmitStorage&);
    std::size_t cell_of(const ParticleFields3D&, std::size_t i) const;
    NeighborView3D query_;
    Ticket permutations_ticket_;
    Indices sorted_to_input_, input_to_sorted_;
    InputOrder input_order_{InputOrder::unknown};
    bool positions_changed_{}; // Computed by update_positions in the previous grid's slot order.
};
} // namespace nexora::sph
