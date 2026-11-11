#pragma once
#include "EntityCapacity.hpp"
#include <Kokkos_Core.hpp>
#include <exception>
#include <new>
#include <nexora/data/FieldDescriptor.hpp>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace nexora::detail {
struct EntityNamespace {
    DomainId domain_id{};
    Association association{Association::cell};
    std::string entity_set;
    Status validate() const;
    friend bool operator==(const EntityNamespace&, const EntityNamespace&) = default;
};

// A snapshot owns its allocation; its only exported handle has const payload.
// ID retirement/allocation belongs to the owning state, not this snapshot.
class EntityOrder {
  public:
    using id_view_type = Kokkos::View<const std::uint64_t*, Kokkos::HostSpace>;
    static Result<EntityOrder> create(const Kokkos::Serial&, EntityNamespace, LayoutVersion,
                                      std::span<const GlobalEntityId>);
    template <class IdView>
    static Result<EntityOrder> copy_from(const Kokkos::Serial& exec, EntityNamespace entity_namespace,
                                         LayoutVersion layout, const IdView& ids) {
        if constexpr (IdView::rank() != 1 ||
                      !std::is_same_v<typename IdView::non_const_value_type, std::uint64_t>) {
            return Status{StatusCode::invalid_argument, "entity IDs require rank-1 uint64 storage"};
        } else if constexpr (!Kokkos::SpaceAccessibility<Kokkos::Serial,
                                                        typename IdView::memory_space>::accessible) {
            return Status{StatusCode::unsupported, "entity IDs are inaccessible to Serial"};
        } else {
            if (auto status = validate_input(entity_namespace, ids.extent(0)); !status.ok())
                return status;
            try {
                // Complete this instance before inspecting any producer input on the host.
                exec.fence("entity order source ready");
                Kokkos::View<std::uint64_t*, Kokkos::HostSpace> copy(
                    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "entity_order.ids"),
                    ids.extent(0));
                Kokkos::deep_copy(exec, copy, ids);
                exec.fence("entity order snapshot ready");
                if (auto status = validate_unique(copy); !status.ok())
                    return status;
                return EntityOrder(std::move(entity_namespace), layout, copy);
            } catch (const std::bad_alloc&) {
                return Status{StatusCode::resource_exhausted, "cannot allocate entity order"};
            } catch (const std::exception&) {
                return Status{StatusCode::execution_failure, "cannot snapshot entity order"};
            }
        }
    }
    id_view_type ids() const {
        return ids_;
    }
    std::size_t size() const {
        return ids_.extent(0);
    }
    const EntityNamespace& entity_namespace() const {
        return namespace_;
    }
    LayoutVersion layout() const {
        return layout_;
    }
    Status validate_for(const EntityNamespace&, LayoutVersion) const;
    bool same_order(const EntityOrder&) const;

  private:
    EntityOrder(EntityNamespace entity_namespace, LayoutVersion layout, id_view_type ids)
        : namespace_(std::move(entity_namespace)), layout_(layout), ids_(std::move(ids)) {}
    static Status validate_input(const EntityNamespace&, std::size_t);
    static Status validate_unique(const id_view_type&);
    EntityNamespace namespace_;
    LayoutVersion layout_;
    id_view_type ids_;
};
} // namespace nexora::detail
