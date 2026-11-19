#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <participants/reference/Participant.hpp>

#include <cstddef>
#include <cstdint>

namespace {

using namespace nexora;
using reference::Participant;

TEST(ReferenceUninitialized, RejectUninitializedRuntime) {
    ASSERT_FALSE(Kokkos::is_initialized());
    static std::size_t allocations = 0;
    Kokkos::Tools::Experimental::set_allocate_data_callback(
        [](Kokkos::Tools::SpaceHandle, const char*, const void*, std::uint64_t) { ++allocations; });
    Participant participant;
    const auto status = participant.initialize(0.0);
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);

    ASSERT_EQ(status.code(), StatusCode::invalid_state);
    ASSERT_EQ(allocations, 0);
    ASSERT_FALSE(Kokkos::is_initialized());
    ASSERT_EQ(participant.snapshot().status().code(), StatusCode::invalid_state);
    ASSERT_TRUE(participant.finalize().ok());
    ASSERT_EQ(participant.finalize().code(), StatusCode::invalid_state);
}

} // namespace
