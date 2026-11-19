#include <nexora/core/CheckedCast.hpp>
#include <nexora/core/Time.hpp>
#include <nexora/core/Types.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace {
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
static_assert(!std::is_convertible_v<FieldId, DomainId>, "IDs must not mix");

TEST(CheckedIntegerCast, AcceptsRepresentableValuesAndRejectsOverflow) {
    const auto index = checked_integer_cast<LocalIndex>(42);
    ASSERT_TRUE(index.ok());
    EXPECT_EQ(index.value(), 42);
    EXPECT_FALSE(checked_integer_cast<LocalIndex>(-1).ok());
    EXPECT_FALSE(checked_integer_cast<LocalIndex>(std::numeric_limits<std::uint64_t>::max()).ok());
    EXPECT_FALSE(checked_integer_cast<std::int8_t>(128).ok());
    EXPECT_FALSE(checked_integer_cast<std::int8_t>(-129).ok());
}

TEST(TimeInterval, PreservesRequestedDurationAcrossEndpointRounding) {
    const TimeInterval interval{0x1p52, 0x1p52 + 1.0, 0.75};
    EXPECT_EQ(interval.end - interval.begin, 1.0);
    EXPECT_EQ(interval.duration(), 0.75);
}

TEST(Result, DistinguishesSuccessAndErrorStatus) {
    EXPECT_TRUE(Status{}.ok());
    Result<int> result(Status{StatusCode::invalid_argument, "bad"});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::invalid_argument);
    Result<int> success(42);
    ASSERT_TRUE(success.ok());
    EXPECT_TRUE(success.status().ok());
    EXPECT_EQ(success.value(), 42);
    EXPECT_THROW((void)Result<int>(Status{}), std::invalid_argument);
}
} // namespace
