#pragma once
#include <cstdint>
#include <functional>
#include <nexora/core/Real.hpp>
namespace nexora {
template <class Tag, class T = std::uint64_t> class StrongValue {
  public:
    using value_type = T;
    constexpr StrongValue() noexcept = default;
    explicit constexpr StrongValue(T value) noexcept : value_(value) {}
    constexpr T value() const noexcept {
        return value_;
    }
    friend constexpr bool operator==(StrongValue a, StrongValue b) noexcept {
        return a.value_ == b.value_;
    }
    friend constexpr bool operator!=(StrongValue a, StrongValue b) noexcept {
        return !(a == b);
    }
    friend constexpr bool operator<(StrongValue a, StrongValue b) noexcept {
        return a.value_ < b.value_;
    }

  private:
    T value_{};
};
struct ParticipantIdTag;
struct PortIdTag;
struct SupportIdTag;
struct FieldIdTag;
struct DomainIdTag;
struct GlobalEntityIdTag;
struct PhysicalObjectIdTag;
struct TopologyVersionTag;
struct LayoutVersionTag;
struct GeometryVersionTag;
struct StateVersionTag;
using ParticipantId = StrongValue<ParticipantIdTag>;
using PortId = StrongValue<PortIdTag>;
using SupportId = StrongValue<SupportIdTag>;
using FieldId = StrongValue<FieldIdTag>;
using DomainId = StrongValue<DomainIdTag>;
using GlobalEntityId = StrongValue<GlobalEntityIdTag>;
using PhysicalObjectId = StrongValue<PhysicalObjectIdTag>;
using LocalIndex = std::uint32_t;
using Offset = std::uint64_t;
using TopologyVersion = StrongValue<TopologyVersionTag>;
using LayoutVersion = StrongValue<LayoutVersionTag>;
using GeometryVersion = StrongValue<GeometryVersionTag>;
using StateVersion = StrongValue<StateVersionTag>;
struct TrialEpoch {
    std::uint64_t window{}, attempt{}, iteration{}, stage{};
    friend constexpr bool operator==(const TrialEpoch&, const TrialEpoch&) = default;
};
struct TimeStamp {
    double physical_time{};
    std::uint64_t window{}, substep{}, stage{};
    friend constexpr bool operator==(const TimeStamp&, const TimeStamp&) = default;
};
} // namespace nexora
namespace std {
template <class Tag, class T> struct hash<nexora::StrongValue<Tag, T>> {
    size_t operator()(nexora::StrongValue<Tag, T> v) const noexcept {
        return hash<T>{}(v.value());
    }
};
} // namespace std
