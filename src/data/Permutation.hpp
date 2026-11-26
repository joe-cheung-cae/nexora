#pragma once
#include "EntityOrder.hpp"
#include <type_traits>
#include <unordered_map>

namespace nexora::detail {
struct EntityMapping {
    using view_type = Kokkos::View<const LocalIndex*, Kokkos::HostSpace>;
    view_type new_to_old;
    view_type old_to_new;
};

// Shared preflight for order construction and receive. No destination writes or
// writable aliases escape; the views must be host-accessible rank-1 uint64 IDs.
template <class FromView, class ToView>
Result<EntityMapping> match_entity_ids(const Kokkos::Serial& exec,
                                       const EntityNamespace& from_namespace, const FromView& from,
                                       const EntityNamespace& to_namespace, const ToView& to) {
    static_assert(FromView::rank() == 1 && ToView::rank() == 1);
    static_assert(std::is_same_v<typename FromView::non_const_value_type, std::uint64_t> &&
                  std::is_same_v<typename ToView::non_const_value_type, std::uint64_t>);
    static_assert(Kokkos::SpaceAccessibility<Kokkos::Serial, typename FromView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<Kokkos::Serial, typename ToView::memory_space>::accessible);
    if (!Kokkos::is_initialized())
        return Status{StatusCode::invalid_state, "ID matching requires active Kokkos runtime"};
    if (auto status = from_namespace.validate(); !status.ok())
        return status;
    const bool same_namespace = from_namespace == to_namespace;
    if (!same_namespace) {
        if (auto status = to_namespace.validate(); !status.ok())
            return status;
    }
    if (!same_namespace || from.extent(0) != to.extent(0))
        return Status{StatusCode::invalid_argument, "entity namespaces or cardinalities differ"};
    const auto count = checked_entity_count(from.extent(0));
    if (!count.ok())
        return count.status();
    if (count.value() == 0)
        return EntityMapping{};
    try {
        exec.fence("entity matching source ready");
        std::unordered_map<std::uint64_t, LocalIndex> source_indices;
        source_indices.reserve(count.value());
        for (std::size_t i = 0; i < count.value(); ++i) {
            if (!source_indices.emplace(from(i), static_cast<LocalIndex>(i)).second)
                return Status{StatusCode::invalid_argument, "source entity IDs are not unique"};
        }
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> forward(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "permutation.new_to_old"), count.value());
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> inverse(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "permutation.old_to_new"), count.value());
        for (std::size_t i = 0; i < count.value(); ++i) {
            const auto found = source_indices.find(to(i));
            if (found == source_indices.end())
                return Status{StatusCode::invalid_argument, "target entity IDs are missing or repeated"};
            forward(i) = found->second;
            inverse(found->second) = static_cast<LocalIndex>(i);
            source_indices.erase(found);
        }
        return EntityMapping{forward, inverse};
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate entity ID matching"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot match entity IDs"};
    }
}

class Permutation {
  public:
    static Result<Permutation> between(const Kokkos::Serial&, const EntityOrder&, const EntityOrder&);
    EntityMapping::view_type new_to_old() const {
        return mapping_.new_to_old;
    }
    EntityMapping::view_type old_to_new() const {
        return mapping_.old_to_new;
    }
  private:
    explicit Permutation(EntityMapping mapping) : mapping_(std::move(mapping)) {}
    EntityMapping mapping_;
};
} // namespace nexora::detail
