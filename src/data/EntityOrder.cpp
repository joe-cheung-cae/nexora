#include "CsrAdjacency.hpp"
#include "Permutation.hpp"
#include <unordered_set>

namespace nexora::detail {
Status EntityNamespace::validate() const {
    if (entity_set.empty())
        return {StatusCode::invalid_argument, "entity namespace has an empty entity set"};
    if (static_cast<unsigned>(association) > static_cast<unsigned>(Association::modal_coordinate))
        return {StatusCode::invalid_argument, "entity namespace has an invalid association"};
    return {};
}

Status EntityOrder::validate_input(const EntityNamespace& entity_namespace, std::size_t count) {
    if (!Kokkos::is_initialized())
        return {StatusCode::invalid_state, "entity order requires active Kokkos runtime"};
    if (auto status = entity_namespace.validate(); !status.ok())
        return status;
    const auto capacity = checked_entity_count(count);
    if (!capacity.ok())
        return capacity.status();
    return checked_array_bytes(count, sizeof(std::uint64_t)).status();
}

Status EntityOrder::validate_unique(const id_view_type& ids) {
    std::unordered_set<std::uint64_t> unique;
    unique.reserve(ids.extent(0));
    for (std::size_t i = 0; i < ids.extent(0); ++i)
        if (!unique.insert(ids(i)).second)
            return {StatusCode::invalid_argument, "entity IDs must be unique within their namespace"};
    return {};
}

Result<EntityOrder> EntityOrder::create(const Kokkos::Serial& exec, EntityNamespace entity_namespace,
                                       LayoutVersion layout, std::span<const GlobalEntityId> ids) {
    if (auto status = validate_input(entity_namespace, ids.size()); !status.ok())
        return status;
    try {
        Kokkos::View<std::uint64_t*, Kokkos::HostSpace> copy(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "entity_order.ids"), ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i)
            copy(i) = ids[i].value();
        if (auto status = validate_unique(copy); !status.ok())
            return status;
        return EntityOrder(std::move(entity_namespace), layout, copy);
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate entity order"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot create entity order"};
    }
}

bool EntityOrder::same_order(const EntityOrder& other) const {
    if (namespace_ != other.namespace_ || layout_ != other.layout_ || size() != other.size())
        return false;
    if (ids_.data() == other.ids_.data())
        return true;
    for (std::size_t i = 0; i < size(); ++i)
        if (ids_(i) != other.ids_(i))
            return false;
    return true;
}

Result<Permutation> Permutation::between(const Kokkos::Serial& exec, const EntityOrder& from,
                                         const EntityOrder& to) {
    auto mapping = match_entity_ids(exec, from.entity_namespace(), from.ids(),
                                    to.entity_namespace(), to.ids());
    if (!mapping.ok())
        return mapping.status();
    if (from.layout() == to.layout() && !from.same_order(to))
        return Status{StatusCode::invalid_state, "changed entity order requires a different layout version"};
    return Permutation(std::move(mapping).value());
}

Result<CsrAdjacency> CsrAdjacency::create(const Kokkos::Serial& exec, const EntityOrder& rows,
                                          const EntityOrder& columns, TopologyVersion topology,
                                          std::span<const Offset> offsets,
                                          std::span<const LocalIndex> indices) {
    if (!Kokkos::is_initialized())
        return Status{StatusCode::invalid_state, "CSR construction requires active Kokkos runtime"};
    const auto capacity = checked_csr_bytes(rows.size(), indices.size());
    if (!capacity.ok())
        return capacity.status();
    if (offsets.size() != rows.size() + 1 || offsets.front() != 0 || offsets.back() != indices.size())
        return Status{StatusCode::invalid_argument, "CSR offsets shape, origin, or terminal offset is invalid"};
    for (std::size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] < offsets[i - 1])
            return Status{StatusCode::invalid_argument, "CSR offsets are not bounded and nondecreasing"};
    for (const auto index : indices)
        if (index >= columns.size())
            return Status{StatusCode::invalid_argument, "CSR column index is outside the column entity order"};
    try {
        Kokkos::View<Offset*, Kokkos::HostSpace> offset_copy(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "csr.offsets"), offsets.size());
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> index_copy(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "csr.indices"), indices.size());
        for (std::size_t i = 0; i < offsets.size(); ++i)
            offset_copy(i) = offsets[i];
        for (std::size_t i = 0; i < indices.size(); ++i)
            index_copy(i) = indices[i];
        return CsrAdjacency(rows, columns, topology, offset_copy, index_copy);
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate CSR adjacency"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot create CSR adjacency"};
    }
}

Result<EntityOrder> EntityOrder::clone(const Kokkos::Serial& exec) const {
    try {
        Kokkos::View<std::uint64_t*, Kokkos::HostSpace> copy(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "entity_order.ids"), size());
        Kokkos::deep_copy(exec, copy, ids_);
        exec.fence("entity order clone ready");
        return EntityOrder(namespace_, layout_, copy);
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate entity order"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot snapshot entity order"};
    }
}

Result<ReorderedCsr> CsrAdjacency::reorder(const Kokkos::Serial& exec,
                                           const EntityOrder& rows, const EntityOrder& columns) const {
    try {
        auto row_map = Permutation::between(exec, rows_, rows);
        if (!row_map.ok())
            return row_map.status();
        auto column_map = (&rows == &columns && rows_.ids().data() == columns_.ids().data() &&
                           rows_.layout() == columns_.layout() &&
                           rows_.entity_namespace() == columns_.entity_namespace())
                              ? row_map : Permutation::between(exec, columns_, columns);
        if (!column_map.ok())
            return column_map.status();

        const auto forward = row_map.value().new_to_old();
        const auto inverse = column_map.value().old_to_new();
        const auto old_offsets = offsets_;
        const auto old_indices = indices_;
        Kokkos::View<Offset*, Kokkos::HostSpace> offsets(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "csr.offsets"), offsets_.extent(0));
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> indices(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "csr.indices"), indices_.extent(0));
        using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("reorder CSR offsets", Policy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            offsets(0) = 0;
            for (std::size_t i = 0; i < forward.extent(0); ++i) {
                const auto row = forward(i);
                offsets(i + 1) = offsets(i) + (old_offsets(row + 1) - old_offsets(row));
            }
        });
        Kokkos::parallel_for("reorder CSR edges", Policy(exec, 0, rows.size()), KOKKOS_LAMBDA(std::size_t i) {
            const auto row = forward(i);
            for (Offset j = old_offsets(row); j < old_offsets(row + 1); ++j)
                indices(offsets(i) + (j - old_offsets(row))) = inverse(old_indices(j));
        });
        exec.fence("reordered CSR ready");
        return ReorderedCsr{CsrAdjacency(rows, columns, topology_, offsets, indices),
                            std::move(row_map).value(), std::move(column_map).value()};
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate reordered CSR"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot reorder CSR"};
    }
}
} // namespace nexora::detail
