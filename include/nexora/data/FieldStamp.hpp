#pragma once

#include <cmath>
#include <cstdint>
#include <nexora/core/Status.hpp>
#include <nexora/core/Types.hpp>

namespace nexora {

// Dynamic snapshot metadata. The registry only stores FieldDescriptor values.
struct FieldStamp {
    TopologyVersion topology{};
    GeometryVersion geometry{};
    LayoutVersion layout{};
    StateVersion accepted_version{};
    TrialEpoch trial{};
    TimeStamp time{};
    double valid_from{};
    double valid_until{};
    std::uint64_t storage_incarnation{}; // Zero until assigned to a storage binding.
    bool accepted{};

    // Validate caller-supplied time at ingress. Bindings authenticate against
    // the issuer's immutable stamp, including its storage incarnation.
    Status validate() const {
        if (!std::isfinite(time.physical_time) || !std::isfinite(valid_from) ||
            !std::isfinite(valid_until) || valid_until < valid_from) {
            return {StatusCode::invalid_argument, "invalid field time or validity interval"};
        }
        if (time.physical_time < valid_from || time.physical_time > valid_until) {
            return {StatusCode::invalid_argument, "field time is outside its validity interval"};
        }
        return {};
    }

    friend bool operator==(const FieldStamp&, const FieldStamp&) = default;
};

} // namespace nexora
