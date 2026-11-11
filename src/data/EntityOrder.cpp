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
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
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

Status EntityOrder::validate_for(const EntityNamespace& entity_namespace, LayoutVersion layout) const {
    if (namespace_ != entity_namespace || layout_ != layout)
        return {StatusCode::invalid_state, "entity order namespace or layout is stale"};
    return {};
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
    try {
        return Permutation(from, to, std::move(mapping).value());
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot retain permutation orders"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot retain permutation orders"};
    }
}

Result<Permutation> Permutation::create(const Kokkos::Serial& exec, const EntityOrder& from,
                                        const EntityOrder& to, std::span<const LocalIndex> new_to_old,
                                        std::span<const LocalIndex> old_to_new) {
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
        return Status{StatusCode::invalid_state, "permutation construction requires active Kokkos runtime"};
    if (from.entity_namespace() != to.entity_namespace() || from.size() != to.size() ||
        new_to_old.size() != to.size() || old_to_new.size() != from.size())
        return Status{StatusCode::invalid_argument, "permutation namespaces or cardinalities differ"};
    if (from.layout() == to.layout() && !from.same_order(to))
        return Status{StatusCode::invalid_state, "changed entity order requires a different layout version"};
    const auto bytes = checked_array_bytes(from.size(), 2 * sizeof(LocalIndex));
    if (!bytes.ok())
        return bytes.status();
    for (std::size_t i = 0; i < new_to_old.size(); ++i)
        if (!checked_local_index(new_to_old[i], from.size()).ok() ||
            !checked_local_index(old_to_new[i], to.size()).ok())
            return Status{StatusCode::invalid_argument, "permutation index is outside its entity order"};
    const auto from_ids = from.ids();
    const auto to_ids = to.ids();
    for (std::size_t i = 0; i < new_to_old.size(); ++i) {
        if (old_to_new[new_to_old[i]] != i || new_to_old[old_to_new[i]] != i)
            return Status{StatusCode::invalid_argument, "permutation maps are not mutual inverses"};
        if (to_ids(i) != from_ids(new_to_old[i]))
            return Status{StatusCode::invalid_argument, "permutation does not preserve stable entity IDs"};
    }
    try {
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> forward(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "permutation.new_to_old"), from.size());
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> inverse(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "permutation.old_to_new"), from.size());
        for (std::size_t i = 0; i < new_to_old.size(); ++i) {
            forward(i) = new_to_old[i];
            inverse(i) = old_to_new[i];
        }
        return Permutation(from, to, EntityMapping{forward, inverse});
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate permutation"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot create permutation"};
    }
}

Status Permutation::validate_for(const EntityOrder& from, const EntityOrder& to) const {
    if (!from_.same_order(from) || !to_.same_order(to))
        return {StatusCode::invalid_state, "permutation source or target entity order is stale"};
    return {};
}

Result<CsrAdjacency> CsrAdjacency::create(const Kokkos::Serial& exec, const EntityOrder& rows,
                                          const EntityOrder& columns, TopologyVersion topology,
                                          std::span<const Offset> offsets,
                                          std::span<const LocalIndex> indices) {
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
        return Status{StatusCode::invalid_state, "CSR construction requires active Kokkos runtime"};
    const auto capacity = checked_csr_capacity(rows.size(), columns.size(), indices.size());
    if (!capacity.ok())
        return capacity.status();
    if (offsets.size() != rows.size() + 1 || offsets.front() != 0 || offsets.back() != indices.size())
        return Status{StatusCode::invalid_argument, "CSR offsets shape, origin, or terminal offset is invalid"};
    for (std::size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] < offsets[i - 1] || offsets[i] > indices.size())
            return Status{StatusCode::invalid_argument, "CSR offsets are not bounded and nondecreasing"};
    for (const auto index : indices)
        if (!checked_local_index(index, columns.size()).ok())
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

Status CsrAdjacency::validate_for(const EntityOrder& rows, const EntityOrder& columns,
                                 TopologyVersion topology) const {
    if (topology_ != topology || !rows_.same_order(rows) || !columns_.same_order(columns))
        return {StatusCode::invalid_state, "CSR topology or row/column order is stale"};
    return {};
}
} // namespace nexora::detail
