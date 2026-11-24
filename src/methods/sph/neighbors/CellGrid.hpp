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
        return *this;
    }

    static Result<CellGrid> build(const Kokkos::Serial&, const ParticleFields3D&,
                                  std::size_t max_cells, const AdmitStorage& admit = {},
                                  double smoothing_length = 1.0);
    // Return a fresh field snapshot. Share cell_begin when each ID retains its cell
    // and ownership. Input rows may be reordered between calls.
    Result<CellGrid> update(const Kokkos::Serial&, const ParticleFields3D&, std::size_t max_cells,
                            const AdmitStorage& admit = {}) const;

    const NeighborView3D& neighbors() const { return query_; }
    double smoothing_length() const { return query_.smoothing_length(); }
    double support_radius() const { return query_.support_radius(); }
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
    CellGrid() = default;
    static Result<CellGrid> prepare(const Kokkos::Serial&, const ParticleFields3D&,
                                    std::size_t max_cells, const CellGrid* previous,
                                    const AdmitStorage&, double smoothing_length);
    Result<bool> try_reuse(const Kokkos::Serial&, const ParticleFields3D&, const CellGrid& previous,
                           const AdmitStorage&);
    std::size_t cell_of(const ParticleFields3D&, std::size_t i) const;
    NeighborView3D query_;
    Ticket permutations_ticket_;
    Indices sorted_to_input_, input_to_sorted_;
};
} // namespace nexora::sph
