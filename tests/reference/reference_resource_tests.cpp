#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <nexora/simulation/Simulation.hpp>
#include <participants/reference/Participant.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace nexora;
using reference::Config;
using reference::Participant;
using reference::Snapshot;
constexpr double step = 0.125;
std::size_t kokkos_allocations = 0;
void count_kokkos_allocation(Kokkos::Tools::SpaceHandle, const char*, const void*, std::uint64_t) {
    ++kokkos_allocations;
}
struct AllocationCounter {
    AllocationCounter() {
        kokkos_allocations = 0;
        Kokkos::Tools::Experimental::set_allocate_data_callback(count_kokkos_allocation);
    }
    ~AllocationCounter() {
        Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    }
};
Snapshot snapshot(const Participant& participant) {
    auto result = participant.snapshot();
    if (!result.ok())
        throw std::runtime_error(result.status().message());
    return std::move(result).value();
}
Config fixture() {
    return {{17, 3, 42}, {real{1}, real{2}, real{-4}}, {real{0}, real{1}, real{5}}, real{2}};
}
template <class View, class T> void check_values(const View& actual, const std::vector<T>& expected) {
    ASSERT_EQ(actual.extent(0), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        ASSERT_EQ(actual(i), expected[i]) << "index " << i;
}
void check_same(const Snapshot& expected, const Config& values, const Snapshot& actual) {
    ASSERT_EQ(actual.stamp(), expected.stamp());
    ASSERT_EQ(actual.ids.data(), expected.ids.data());
    ASSERT_EQ(actual.x.data(), expected.x.data());
    ASSERT_EQ(actual.q.data(), expected.q.data());
    ASSERT_EQ(actual.history.data(), expected.history.data());
    ASSERT_EQ(actual.control, expected.control);
    ASSERT_EQ(actual.diagnostic, expected.diagnostic);
    ASSERT_NO_FATAL_FAILURE(check_values(actual.ids, values.ids));
    ASSERT_NO_FATAL_FAILURE(check_values(actual.x, values.x));
    ASSERT_NO_FATAL_FAILURE(check_values(actual.q, values.q));
    ASSERT_EQ(actual.history.extent(0), 1);
    ASSERT_EQ(actual.history(0), values.history_seed);
}

TEST(ReferenceResource, HistoryAndControlPublishWithNumericalFields) {
    auto input = fixture();
    input.history_seed = 7;
    input.control_seed.sequence = 4;
    Participant participant(input);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto initial = snapshot(participant);
    ASSERT_TRUE(simulation.run(step, 2).ok());
    const auto accepted = snapshot(participant);
    ASSERT_NE(accepted.history.data(), initial.history.data());
    ASSERT_EQ(initial.history(0), 7);
    ASSERT_EQ(initial.control.sequence, 4);
    ASSERT_EQ(accepted.history(0), 7794);
    ASSERT_EQ(accepted.control.sequence, 6);
    ASSERT_EQ(accepted.diagnostic, 7800);
    ASSERT_EQ(accepted.x(0), real{0.5625});
    ASSERT_EQ(accepted.q(0), real{0.4375});
    ASSERT_EQ(accepted.version, StateVersion{2});
    ASSERT_EQ(accepted.time, 2 * step);
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(ReferenceResource, FailedInitializationReleasesReservations) {
    Participant participant(fixture(), ParticipantId{1}, {1, 1});
    ASSERT_EQ(participant.initialize(0).code(), StatusCode::resource_exhausted);
    ASSERT_FALSE(participant.snapshot().ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, 0);
    ASSERT_EQ(participant.resource_usage().snapshot_records, 0);
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, SnapshotRecordBudgetReleasesOnLastHolder) {
    Participant participant(fixture(), ParticipantId{1}, {1, 1'000'000});
    ASSERT_TRUE(participant.initialize(0).ok());
    auto first = snapshot(participant);
    auto copy = first;
    ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    ASSERT_EQ(participant.resource_usage().known_external_owner_bytes, 0);
    ASSERT_EQ(participant.snapshot().status().code(), StatusCode::resource_exhausted);
    first = {};
    ASSERT_EQ(participant.snapshot().status().code(), StatusCode::resource_exhausted);
    copy = {};
    ASSERT_TRUE(participant.snapshot().ok());
    ASSERT_EQ(participant.resource_usage().snapshot_records, 0);
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, RetainedBankBudgetRejectsBeforeAllocation) {
    constexpr std::size_t count = 3;
    constexpr std::size_t bank_bytes = 2 * count * sizeof(real) + sizeof(std::uint64_t);
    constexpr std::size_t initial_bytes = count * sizeof(std::uint64_t) + bank_bytes;
    Participant participant(fixture(), ParticipantId{1}, {4, initial_bytes + bank_bytes});
    ASSERT_TRUE(participant.initialize(0).ok());
    std::optional<Snapshot> old{snapshot(participant)};
    std::optional<Snapshot> copy{*old};
    ASSERT_TRUE(participant.step({0, step, step}).ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + bank_bytes);
    const auto accepted = snapshot(participant);
    auto values = fixture();
    values.x = {real{0.75}, real{1.5}, real{-3}};
    values.q = {real{0.25}, real{1.5}, real{4}};
    values.history_seed = 1;
    for (int holder = 0; holder < 2; ++holder) {
        const AllocationCounter counter;
        ASSERT_EQ(participant.step({step, 2 * step, step}).code(), StatusCode::resource_exhausted);
        ASSERT_EQ(kokkos_allocations, 0);
        ASSERT_NO_FATAL_FAILURE(check_same(accepted, values, snapshot(participant)));
        old.reset();
    }
    copy.reset();
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);
    ASSERT_TRUE(participant.step({step, 2 * step, step}).ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + bank_bytes);
    ASSERT_EQ(accepted.x(0), real{0.75});
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, NumericalFailureReleasesTemporaryBank) {
    const real large = std::numeric_limits<real>::max();
    const Config input{{12}, {large}, {large / real{2}}, real{1}};
    Participant participant(input);
    ASSERT_TRUE(participant.initialize(0).ok());
    const auto before = snapshot(participant);
    const auto usage = participant.resource_usage();
    ASSERT_EQ(participant.step({0, 1, 1}).code(), StatusCode::numerical_failure);
    ASSERT_EQ(participant.resource_usage().controlled_bytes, usage.controlled_bytes);
    ASSERT_EQ(participant.resource_usage().snapshot_records, usage.snapshot_records);
    ASSERT_NO_FATAL_FAILURE(check_same(before, input, snapshot(participant)));
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, ControlCounterExhaustionPrecedesAllocation) {
    auto input = fixture();
    input.control_seed.sequence = std::numeric_limits<std::uint64_t>::max();
    Participant participant(input);
    ASSERT_TRUE(participant.initialize(0).ok());
    const auto before = snapshot(participant);
    {
        const AllocationCounter counter;
        ASSERT_EQ(participant.step({0, step, step}).code(), StatusCode::resource_exhausted);
        ASSERT_EQ(kokkos_allocations, 0);
    }
    ASSERT_NO_FATAL_FAILURE(check_same(before, input, snapshot(participant)));
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, SnapshotOutlivesParticipantAndKeepsBudget) {
    constexpr std::size_t count = 3;
    constexpr std::size_t initial_bytes = count * sizeof(std::uint64_t) +
                                         2 * count * sizeof(real) + sizeof(std::uint64_t);
    std::optional<Snapshot> retained;
    {
        Participant participant(fixture());
        ASSERT_TRUE(participant.initialize(0).ok());
        retained = snapshot(participant);
        ASSERT_TRUE(participant.step({0, step, step}).ok());
        ASSERT_TRUE(participant.finalize().ok());
        ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);
        ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    }
    ASSERT_EQ(retained->x(0), real{1});
    ASSERT_EQ(retained->q(1), real{1});
    ASSERT_EQ(retained->history(0), 0);
    ASSERT_EQ(retained->version, StateVersion{0});
    const auto copy = *retained;
    retained.reset();
    ASSERT_EQ(copy.x(2), real{-4});
}
} // namespace
