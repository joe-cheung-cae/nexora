#pragma once
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <nexora/core/Result.hpp>
#include <nexora/data/FieldBinding.hpp>
#include <nexora/data/TransportView.hpp>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace nexora::detail {
// Both sources are read under lease through the pack fence. A caller whose
// consumer needs only the independent packed Views may release them there.
struct TransportBuilder {
    enum class SourceLifetime { until_complete, until_packed };

    // Counts the requested packed array payload, excluding Kokkos allocation metadata.
    template <class T>
    static Result<std::size_t> packed_payload_bytes(std::size_t entities, std::size_t components) {
        constexpr auto max_size = std::numeric_limits<std::size_t>::max();
        constexpr auto id_bytes = sizeof(std::uint64_t);
        if (components == 0 || components > (max_size - id_bytes) / sizeof(T))
            return Status{StatusCode::invalid_argument, "transport component count exceeds size capacity"};
        const auto bytes_per_entity = id_bytes + components * sizeof(T);
        if (entities > max_size / bytes_per_entity)
            return Status{StatusCode::invalid_argument, "transport payload exceeds size capacity"};
        return entities * bytes_per_entity;
    }

    template <class ValueView, class IdView>
    static auto prepare(const Kokkos::Serial& exec, TransportPurpose purpose,
                        ReadField<ValueView> values, ReadField<IdView> ids,
                        SourceLifetime lifetime = SourceLifetime::until_complete,
                        std::shared_ptr<const void> retention = {})
        -> Result<TransportView<std::remove_const_t<typename ValueView::value_type>>> {
        if (!Kokkos::is_initialized() || !values.lease_ || !ids.lease_)
            return Status{StatusCode::invalid_state, "transport requires active CPU read grants"};
        if (static_cast<unsigned>(purpose) > static_cast<unsigned>(TransportPurpose::visualization))
            return Status{StatusCode::invalid_argument, "invalid transport purpose"};

        using Id = std::remove_const_t<typename IdView::value_type>;
        using ValueMemory = typename ValueView::memory_space;
        using IdMemory = typename IdView::memory_space;
        if constexpr (IdView::rank() != 1 || !std::is_same_v<Id, std::uint64_t>) {
            return Status{StatusCode::invalid_argument, "stable IDs must be a rank-1 uint64 field"};
        } else if constexpr (ValueView::rank() != 1 && ValueView::rank() != 2) {
            return Status{StatusCode::unsupported, "CPU transport accepts rank-1 or rank-2 values"};
        } else if constexpr (!Kokkos::SpaceAccessibility<Kokkos::Serial, ValueMemory>::accessible ||
                             !Kokkos::SpaceAccessibility<Kokkos::Serial, IdMemory>::accessible) {
            return Status{StatusCode::unsupported, "transport storage is inaccessible to Serial"};
        } else {
            return pack(exec, purpose, values, ids, lifetime, std::move(retention));
        }
    }

  private:
    template <class ValueView, class IdView>
    static auto pack(const Kokkos::Serial& exec, TransportPurpose purpose,
                     ReadField<ValueView>& values, ReadField<IdView>& ids, SourceLifetime lifetime,
                     std::shared_ptr<const void> retention)
        -> Result<TransportView<std::remove_const_t<typename ValueView::value_type>>> {
        using T = std::remove_const_t<typename ValueView::value_type>;
        using Snapshot = TransportView<T>;
        const auto& value_descriptor = values.descriptor();
        const auto& id_descriptor = ids.descriptor();

        if (ids.component_axes_[0] || id_descriptor.units != "1")
            return Status{StatusCode::invalid_argument, "stable IDs must be unitless entity scalars"};
        if (value_descriptor.entity_set.empty() ||
            value_descriptor.entity_set != id_descriptor.entity_set ||
            value_descriptor.domain_id != id_descriptor.domain_id ||
            value_descriptor.association != id_descriptor.association)
            return Status{StatusCode::invalid_argument, "transport and stable ID namespaces differ"};
        auto id_stamp = ids.stamp();
        id_stamp.storage_incarnation = values.stamp().storage_incarnation;
        if (id_stamp != values.stamp())
            return Status{StatusCode::invalid_state, "transport field and stable ID stamps differ"};

        std::size_t entity_axis = 0;
        if constexpr (ValueView::rank() == 1) {
            if (values.component_axes_[0])
                return Status{StatusCode::invalid_argument, "rank-1 transport needs an entity axis"};
        } else {
            if (values.component_axes_[0] == values.component_axes_[1])
                return Status{StatusCode::invalid_argument,
                              "rank-2 transport requires one entity and one component axis"};
            entity_axis = values.component_axes_[0] ? 1 : 0;
        }

        const auto source = values.view();
        const auto source_ids = ids.view();
        const std::size_t entities = source.extent(entity_axis);
        const std::size_t components = value_descriptor.components;
        if (source_ids.extent(0) != entities)
            return Status{StatusCode::invalid_argument, "transport entity and ID counts differ"};
        if (auto bytes = packed_payload_bytes<T>(entities, components); !bytes.ok())
            return bytes.status();

        try {
            exec.fence("transport stable IDs ready for validation");
            std::unordered_set<std::uint64_t> unique_ids;
            for (std::size_t i = 0; i < entities; ++i) {
                if (!unique_ids.insert(source_ids(i)).second)
                    return Status{StatusCode::invalid_argument, "transport stable IDs are not unique"};
            }

            Kokkos::View<std::uint64_t*, Kokkos::HostSpace> packed_ids(
                Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "transport.ids"), entities);
            Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace> packed_values(
                Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "transport.values"),
                entities, components);
            using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
            Kokkos::parallel_for(
                "transport.pack", Policy(exec, 0, entities), KOKKOS_LAMBDA(const std::size_t i) {
                    packed_ids(i) = source_ids(i);
                    if constexpr (ValueView::rank() == 1) {
                        packed_values(i, 0) = source(i);
                    } else {
                        for (std::size_t component = 0; component < components; ++component)
                            packed_values(i, component) =
                                entity_axis == 0 ? source(i, component) : source(component, i);
                    }
                });
            exec.fence("transport payload ready");

            std::optional<typename Snapshot::SourceGrants> grants;
            if (lifetime == SourceLifetime::until_complete)
                grants.emplace(std::move(values.lease_), std::move(ids.lease_));
            return Snapshot(std::move(values.descriptor_), values.stamp(), purpose,
                            {std::move(retention), std::move(grants), packed_ids, packed_values});
        } catch (const std::bad_alloc&) {
            return Status{StatusCode::resource_exhausted, "cannot allocate transport snapshot"};
        } catch (const std::exception&) {
            return Status{StatusCode::execution_failure, "cannot prepare transport snapshot"};
        }
    }
};
} // namespace nexora::detail
