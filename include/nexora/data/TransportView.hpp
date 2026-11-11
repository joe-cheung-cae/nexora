#pragma once
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <nexora/data/FieldAccess.hpp>
#include <nexora/data/FieldDescriptor.hpp>
#include <nexora/data/FieldStamp.hpp>
#include <optional>
#include <stdexcept>
#include <utility>

namespace nexora {
namespace detail {
struct TransportBuilder;
}

enum class TransportPurpose { halo, coupling, checkpoint, visualization };

// A local snapshot for one consumer. Move it into that consumer's operation
// record before handoff, and keep the record until its last use has completed.
// Native View copies do not retain any source grants held by the record.
template <class T, class Space = Kokkos::HostSpace> class TransportView {
  public:
    using id_view_type = Kokkos::View<const std::uint64_t*, Space>;
    using value_view_type = Kokkos::View<const T**, Kokkos::LayoutRight, Space>;

    TransportView(TransportView&& other) noexcept
        : descriptor_(std::move(other.descriptor_)), stamp_(other.stamp_), purpose_(other.purpose_),
          payload_(std::exchange(other.payload_, std::nullopt)) {}

    TransportView& operator=(TransportView&& other) noexcept {
        if (this != &other) {
            descriptor_ = std::move(other.descriptor_);
            stamp_ = other.stamp_;
            purpose_ = other.purpose_;
            payload_ = std::exchange(other.payload_, std::nullopt);
        }
        return *this;
    }
    TransportView(const TransportView&) = delete;
    TransportView& operator=(const TransportView&) = delete;

    explicit operator bool() const noexcept {
        return payload_.has_value();
    }

    const FieldDescriptor& descriptor() const noexcept {
        return descriptor_;
    }
    const FieldStamp& stamp() const noexcept {
        return stamp_;
    }
    TransportPurpose purpose() const noexcept {
        return purpose_;
    }

    id_view_type stable_ids() const & {
        return active_payload().ids;
    }
    id_view_type stable_ids() const && = delete;

    value_view_type values() const & {
        return active_payload().values;
    }
    value_view_type values() const && = delete;

    // Call only once the consumer has stopped using the payload. Destruction
    // also releases this snapshot and signals that the operation is complete.
    void complete() noexcept {
        payload_.reset();
    }

  private:
    friend struct detail::TransportBuilder;

    struct SourceGrants {
        ReadLease values;
        ReadLease ids;
    };

    struct Payload {
        std::optional<SourceGrants> source;
        id_view_type ids;
        value_view_type values;
    };

    TransportView(FieldDescriptor descriptor, FieldStamp stamp, TransportPurpose purpose,
                  Payload payload)
        : descriptor_(std::move(descriptor)), stamp_(stamp), purpose_(purpose),
          payload_(std::move(payload)) {}

    const Payload& active_payload() const {
        if (!*this)
            throw std::logic_error("transport snapshot has no active consumer");
        return *payload_;
    }

    FieldDescriptor descriptor_;
    FieldStamp stamp_;
    TransportPurpose purpose_;
    std::optional<Payload> payload_;
};
} // namespace nexora
