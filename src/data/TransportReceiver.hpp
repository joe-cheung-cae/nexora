#pragma once
#include "Permutation.hpp"
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <nexora/data/FieldBinding.hpp>
#include <nexora/data/TransportView.hpp>
#include <type_traits>

namespace nexora::detail {
// Local overwrite into the exact region named by target_ids. The caller owns
// all three grants and may correct validation errors and retry with them.
struct TransportReceiver {
    template <class T, class Space, class ValueView, class IdView>
    static Status unpack(const Kokkos::Serial& exec, const TransportView<T, Space>& source,
                         const WriteField<ValueView>& target, const ReadField<IdView>& target_ids,
                         const FieldStamp& expected_source) {
        if (!Kokkos::is_initialized() || Kokkos::is_finalized() || !source || !target.lease_ ||
            !target_ids.lease_)
            return {StatusCode::invalid_state, "receive requires active CPU transport and field grants"};

        using Id = std::remove_const_t<typename IdView::value_type>;
        using PayloadMemory = typename TransportView<T, Space>::value_view_type::memory_space;
        using ValueMemory = typename ValueView::memory_space;
        using IdMemory = typename IdView::memory_space;
        if constexpr (!std::is_same_v<T, typename ValueView::value_type>) {
            return {StatusCode::invalid_argument, "receive scalar types differ"};
        } else if constexpr (IdView::rank() != 1 || !std::is_same_v<Id, std::uint64_t>) {
            return {StatusCode::invalid_argument, "receive IDs must be a rank-1 uint64 field"};
        } else if constexpr (ValueView::rank() != 1 && ValueView::rank() != 2) {
            return {StatusCode::unsupported, "CPU receive requires rank-1 or rank-2 values"};
        } else if constexpr (!Kokkos::SpaceAccessibility<Kokkos::Serial, PayloadMemory>::accessible ||
                             !Kokkos::SpaceAccessibility<Kokkos::Serial, ValueMemory>::accessible ||
                             !Kokkos::SpaceAccessibility<Kokkos::Serial, IdMemory>::accessible) {
            return {StatusCode::unsupported, "receive storage is inaccessible to Serial"};
        } else {
            return validate_and_unpack(exec, source, target, target_ids, expected_source);
        }
    }

  private:
    static bool same_namespace(const FieldDescriptor& left, const FieldDescriptor& right) {
        return !left.entity_set.empty() && left.entity_set == right.entity_set &&
               left.domain_id == right.domain_id && left.association == right.association;
    }

    static bool same_field(const FieldDescriptor& left, const FieldDescriptor& right) {
        return same_namespace(left, right) && left.id == right.id &&
               left.scalar_type == right.scalar_type && left.components == right.components &&
               left.units == right.units && left.tensor_interpretation == right.tensor_interpretation &&
               left.reference_frame == right.reference_frame &&
               left.pressure_convention == right.pressure_convention && left.measure == right.measure &&
               left.time_level == right.time_level && left.stage == right.stage;
    }

    static bool same_epoch(const FieldStamp& source, const FieldStamp& target) {
        return source.topology == target.topology && source.geometry == target.geometry &&
               source.accepted_version == target.accepted_version && source.trial == target.trial &&
               source.time == target.time;
    }

    template <class T, class Space, class ValueView, class IdView>
    static Status validate_and_unpack(const Kokkos::Serial& exec,
                                      const TransportView<T, Space>& source,
                                      const WriteField<ValueView>& target,
                                      const ReadField<IdView>& target_ids,
                                      const FieldStamp& expected_source) {
        const auto& descriptor = target.descriptor();
        const auto& id_descriptor = target_ids.descriptor();
        const auto& stamp = target.stamp();
        const bool ghost = descriptor.ownership == FieldOwnership::ghost;

        if (source.purpose() != TransportPurpose::halo && source.purpose() != TransportPurpose::coupling)
            return {StatusCode::invalid_argument, "transport purpose is unsupported for local receive"};
        if (descriptor.access != FieldAccess::overwrite || stamp.accepted ||
            (source.purpose() == TransportPurpose::halo && !ghost) ||
            (!ghost && stamp.trial == TrialEpoch{}))
            return {StatusCode::invalid_state, "receive needs overwrite access to a trial or ghost region"};

        auto id_stamp = target_ids.stamp();
        id_stamp.storage_incarnation = stamp.storage_incarnation;
        if (source.stamp() != expected_source || id_stamp != stamp || !same_epoch(source.stamp(), stamp))
            return {StatusCode::invalid_state, "receive field stamps are stale or incompatible"};

        if (!same_field(source.descriptor(), descriptor) || !same_namespace(descriptor, id_descriptor))
            return {StatusCode::invalid_argument, "receive field semantics or ID namespace differ"};
        if (target_ids.component_axes_[0] || id_descriptor.components != 1 || id_descriptor.units != "1")
            return {StatusCode::invalid_argument, "receive IDs must be unitless entity scalars"};

        std::size_t entity_axis = 0;
        if constexpr (ValueView::rank() == 1) {
            if (target.component_axes_[0])
                return {StatusCode::invalid_argument, "rank-1 receive needs an entity axis"};
        } else {
            if (target.component_axes_[0] == target.component_axes_[1])
                return {StatusCode::invalid_argument, "rank-2 receive needs one entity and one component axis"};
            entity_axis = target.component_axes_[0] ? 1 : 0;
        }

        const auto payload = source.values();
        const auto source_ids = source.stable_ids();
        const auto destination = target.view();
        const auto destination_ids = target_ids.view();
        const std::size_t entities = destination.extent(entity_axis);
        const std::size_t components = descriptor.components;
        if (payload.extent(0) != entities || source_ids.extent(0) != entities ||
            destination_ids.extent(0) != entities || payload.extent(1) != components)
            return {StatusCode::invalid_argument, "receive field or ID shape mismatch"};
        try {
            const EntityNamespace source_namespace{source.descriptor().domain_id,
                                                   source.descriptor().association,
                                                   source.descriptor().entity_set};
            const EntityNamespace destination_namespace{descriptor.domain_id, descriptor.association,
                                                        descriptor.entity_set};
            auto mapping = match_entity_ids(exec, source_namespace, source_ids,
                                            destination_namespace, destination_ids);
            if (!mapping.ok())
                return mapping.status();
            if (entities == 0)
                return {};

            // The mapping is now a bijection onto the leased destination region.
            // All metadata and ID checks finish before the first destination write.
            const auto destination_index = mapping.value().old_to_new;
            using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
            Kokkos::parallel_for(
                "transport.unpack", Policy(exec, 0, entities), KOKKOS_LAMBDA(const std::size_t i) {
                    const auto row = destination_index(i);
                    if constexpr (ValueView::rank() == 1) {
                        destination(row) = payload(i, 0);
                    } else {
                        for (std::size_t component = 0; component < components; ++component) {
                            if (entity_axis == 0)
                                destination(row, component) = payload(i, component);
                            else
                                destination(component, row) = payload(i, component);
                        }
                    }
                });
            exec.fence("transport receive ready");
            return {};
        } catch (const std::bad_alloc&) {
            return {StatusCode::resource_exhausted, "cannot allocate receive ID mapping"};
        } catch (const std::exception&) {
            return {StatusCode::execution_failure, "cannot unpack local transport"};
        }
    }
};
} // namespace nexora::detail
