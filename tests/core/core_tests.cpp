#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>
#include <nexora/core/CheckedCast.hpp>
#include <nexora/core/Time.hpp>
#include <nexora/core/Types.hpp>
#include <type_traits>
int main() {
    using namespace nexora;
#if NEXORA_ENABLE_DOUBLE_PRECISION
    static_assert(std::is_same_v<real, double>);
#else
    static_assert(std::is_same_v<real, float>);
#endif
    static_assert(std::is_same_v<decltype(TimeInterval::begin), double>);
    static_assert(std::is_same_v<decltype(TimeInterval::end), double>);
    static_assert(std::is_same_v<decltype(TimeInterval::requested_step), double>);
    static_assert(std::is_same_v<decltype(TimeStamp::physical_time), double>);
    static_assert(!std::is_convertible<FieldId, DomainId>::value, "IDs must not mix");
    assert(checked_integer_cast<LocalIndex>(42).value() == 42);
    assert(!checked_integer_cast<LocalIndex>(-1).ok());
    assert(!checked_integer_cast<LocalIndex>(std::numeric_limits<std::uint64_t>::max()).ok());
    assert(!checked_integer_cast<std::int8_t>(128).ok());
    assert(!checked_integer_cast<std::int8_t>(-129).ok());
    const TimeInterval interval{0x1p52, 0x1p52 + 1.0, 0.75};
    assert(interval.end - interval.begin == 1.0);
    assert(interval.duration() == 0.75);
    Status s;
    assert(s.ok());
    Result<int> result(Status{StatusCode::invalid_argument, "bad"});
    assert(!result.ok());
    assert(result.status().code() == StatusCode::invalid_argument);
    Result<int> success(42);
    assert(success.ok() && success.status().ok() && success.value() == 42);
    bool threw = false;
    try {
        Result<int> invalid(Status{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
