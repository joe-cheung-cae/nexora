#pragma once
#include "Permutation.hpp"

namespace nexora::detail {
struct ReorderedCsr;

// Duplicate edges, self edges, and row-local ordering are consumer semantics.
class CsrAdjacency {
  public:
    using offset_view_type = Kokkos::View<const Offset*, Kokkos::HostSpace>;
    using index_view_type = Kokkos::View<const LocalIndex*, Kokkos::HostSpace>;
    static Result<CsrAdjacency> create(const Kokkos::Serial&, const EntityOrder& rows,
                                       const EntityOrder& columns, TopologyVersion,
                                       std::span<const Offset>, std::span<const LocalIndex>);
    Result<ReorderedCsr> reorder(const Kokkos::Serial&, const EntityOrder& rows,
                                 const EntityOrder& columns) const;
    offset_view_type offsets() const {
        return offsets_;
    }
    index_view_type indices() const {
        return indices_;
    }
    Offset offset_at(std::size_t row) const {
        return offsets_(row);
    }
    LocalIndex index_at(std::size_t edge) const {
        return indices_(edge);
    }
    const EntityOrder& rows() const {
        return rows_;
    }
    const EntityOrder& columns() const {
        return columns_;
    }
    TopologyVersion topology() const {
        return topology_;
    }

  private:
    CsrAdjacency(EntityOrder rows, EntityOrder columns, TopologyVersion topology,
                 offset_view_type offsets, index_view_type indices)
        : rows_(std::move(rows)), columns_(std::move(columns)), topology_(topology),
          offsets_(std::move(offsets)), indices_(std::move(indices)) {}
    EntityOrder rows_, columns_;
    TopologyVersion topology_;
    offset_view_type offsets_;
    index_view_type indices_;
};

struct ReorderedCsr {
    CsrAdjacency adjacency;
    Permutation rows, columns;
};
} // namespace nexora::detail
