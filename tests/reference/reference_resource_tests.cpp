#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <nexora/simulation/Simulation.hpp>
#include <participants/reference/Participant.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

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
    if (!result.ok()) {
        throw std::runtime_error(result.status().message());
    }
    return std::move(result).value();
}

Config fixture() {
    return {{17, 3, 42}, {real{1}, real{2}, real{-4}}, {real{0}, real{1}, real{5}}, real{2}};
}

void check_payload(const Snapshot& s, const std::array<real, 3>& x, const std::array<real, 3>& q,
                   double time, std::uint64_t version, std::uint64_t attempt,
                   std::uint64_t accepted_steps, std::optional<std::uint64_t> expected_history = {},
                   std::optional<std::uint64_t> expected_sequence = {}) {
    ASSERT_EQ(s.time, time);
    ASSERT_EQ(s.version, StateVersion{version});
    ASSERT_EQ(s.attempt, attempt);
    ASSERT_EQ(s.ids.extent(0), 3);
    ASSERT_EQ(s.x.extent(0), 3);
    ASSERT_EQ(s.q.extent(0), 3);
    constexpr std::array<std::uint64_t, 3> ids{17, 3, 42};
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(s.ids(i), ids[i]);
        ASSERT_EQ(s.x(i), x[i]);
        ASSERT_EQ(s.q(i), q[i]);
    }
    ASSERT_EQ(s.history.extent(0), 1);
    const auto sequence = expected_sequence.value_or(accepted_steps);
    ASSERT_EQ(s.control.sequence, sequence);
    const auto history =
        expected_history.value_or(accepted_steps == 0 ? 0 : (accepted_steps == 1 ? 1 : 35));
    ASSERT_EQ(s.history(0), history);
    ASSERT_EQ(s.diagnostic, history + sequence);
}

struct AcceptedExpectation {
    FieldStamp stamp;
    StateVersion version;
    double time = 0;
    std::uint64_t attempt = 0;
    reference::Control control;
    std::uint64_t diagnostic = 0;
    std::array<std::uint64_t, 3> ids{};
    std::array<real, 3> x{}, q{};
    std::uint64_t history = 0;
    const void* ids_storage = nullptr;
    const void* x_storage = nullptr;
    const void* q_storage = nullptr;
    const void* history_storage = nullptr;
};

AcceptedExpectation capture_accepted(const Snapshot& s) {
    if (s.ids.extent(0) != 3 || s.x.extent(0) != 3 || s.q.extent(0) != 3 ||
        s.history.extent(0) != 1) {
        throw std::runtime_error("accepted snapshot has unexpected extents");
    }
    AcceptedExpectation result{
        s.stamp(), s.version,    s.time,       s.attempt,  s.control,  s.diagnostic,    {}, {},
        {},        s.history(0), s.ids.data(), s.x.data(), s.q.data(), s.history.data()};
    for (std::size_t i = 0; i < result.ids.size(); ++i) {
        result.ids[i] = s.ids(i);
        result.x[i] = s.x(i);
        result.q[i] = s.q(i);
    }
    return result;
}

void check_accepted(const AcceptedExpectation& expected, const Snapshot& actual) {
    ASSERT_EQ(expected.stamp, actual.stamp());
    ASSERT_EQ(expected.version, actual.version);
    ASSERT_EQ(expected.time, actual.time);
    ASSERT_EQ(expected.attempt, actual.attempt);
    ASSERT_EQ(expected.control, actual.control);
    ASSERT_EQ(expected.diagnostic, actual.diagnostic);
    ASSERT_EQ(expected.ids_storage, actual.ids.data());
    ASSERT_EQ(expected.x_storage, actual.x.data());
    ASSERT_EQ(expected.q_storage, actual.q.data());
    ASSERT_EQ(expected.history_storage, actual.history.data());
    for (std::size_t i = 0; i < expected.ids.size(); ++i) {
        ASSERT_EQ(expected.ids[i], actual.ids(i));
        ASSERT_EQ(expected.x[i], actual.x(i));
        ASSERT_EQ(expected.q[i], actual.q(i));
    }
    ASSERT_EQ(expected.history, actual.history(0));
}

TEST(ReferenceResource, HistoryReplayMatchesUninterrupted) {
    auto replay_input = fixture();
    replay_input.history_seed = 7;
    replay_input.control_seed.sequence = 4;
    auto oracle_input = replay_input;
    Participant replay(std::move(replay_input));
    Participant oracle(std::move(oracle_input));
    Simulation replay_sim(replay), oracle_sim(oracle);
    ASSERT_TRUE(replay_sim.initialize().ok());
    ASSERT_TRUE(replay_sim.prepare().ok());
    ASSERT_TRUE(oracle_sim.initialize().ok());
    ASSERT_TRUE(oracle_sim.prepare().ok());

    const auto initial = snapshot(replay);
    ASSERT_NO_FATAL_FAILURE(check_payload(initial, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0, 7, 4));

    ASSERT_TRUE(replay_sim.begin_window(step).ok());
    ASSERT_TRUE(replay_sim.advance_trial().ok());
    ASSERT_TRUE(replay_sim.restore_window().ok());
    ASSERT_NO_FATAL_FAILURE(
        check_payload(snapshot(replay), {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0, 7, 4));
    ASSERT_EQ(replay_sim.time().accepted_time, 0.0);
    ASSERT_EQ(replay_sim.time().accepted_steps, 0);

    ASSERT_TRUE(replay_sim.advance(step).ok());
    ASSERT_TRUE(oracle_sim.advance(step).ok());
    ASSERT_TRUE(replay_sim.advance(step).ok());
    ASSERT_TRUE(oracle_sim.advance(step).ok());
    const auto replayed = snapshot(replay);
    const auto uninterrupted = snapshot(oracle);
    ASSERT_EQ(replayed.time, uninterrupted.time);
    ASSERT_EQ(replayed.version, uninterrupted.version);
    ASSERT_EQ(replayed.control.sequence, uninterrupted.control.sequence);
    ASSERT_EQ(replayed.history(0), uninterrupted.history(0));
    ASSERT_NE(replayed.history.data(), initial.history.data());
    ASSERT_EQ(replayed.diagnostic, uninterrupted.diagnostic);
    ASSERT_EQ(replayed.history(0), 7794);
    ASSERT_EQ(replayed.control.sequence, 6);
    ASSERT_EQ(replayed.diagnostic, 7800);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(replayed.ids(i), uninterrupted.ids(i));
        ASSERT_EQ(replayed.x(i), uninterrupted.x(i));
        ASSERT_EQ(replayed.q(i), uninterrupted.q(i));
    }
}

TEST(ReferenceResource, FaultsPreserveAcceptedState) {
    const auto exercise = [](Participant::Fault fault) {
        Participant participant(fixture());
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        ASSERT_TRUE(simulation.advance(step).ok());
        auto baseline = snapshot(participant);
        const auto expected = capture_accepted(baseline);
        const auto baseline_time = simulation.time();
        const auto baseline_usage = participant.resource_usage();

        participant.fail_next(fault);
        if (fault == Participant::Fault::begin) {
            ASSERT_EQ(simulation.begin_window(step).code(), StatusCode::execution_failure);
        } else if (fault == Participant::Fault::advance) {
            ASSERT_TRUE(simulation.begin_window(step).ok());
            ASSERT_EQ(simulation.advance_trial().code(), StatusCode::execution_failure);
        } else {
            ASSERT_TRUE(simulation.begin_window(step).ok());
            ASSERT_TRUE(simulation.advance_trial().ok());
            ASSERT_EQ(simulation.commit_window().code(), StatusCode::execution_failure);
        }
        ASSERT_EQ(simulation.state(), SimulationState::prepared);
        ASSERT_EQ(simulation.time().accepted_time, baseline_time.accepted_time);
        ASSERT_EQ(simulation.time().accepted_steps, baseline_time.accepted_steps);
        ASSERT_NO_FATAL_FAILURE(check_accepted(expected, baseline));
        ASSERT_NO_FATAL_FAILURE(check_accepted(expected, snapshot(participant)));
        ASSERT_EQ(participant.resource_usage().controlled_bytes, baseline_usage.controlled_bytes);
        ASSERT_EQ(participant.resource_usage().snapshot_records, baseline_usage.snapshot_records);
        ASSERT_TRUE(simulation.advance(step).ok());
        ASSERT_EQ(simulation.time().accepted_steps, baseline_time.accepted_steps + 1);
        ASSERT_TRUE(simulation.finalize().ok());
    };
    ASSERT_NO_FATAL_FAILURE(exercise(Participant::Fault::begin));
    ASSERT_NO_FATAL_FAILURE(exercise(Participant::Fault::advance));
    ASSERT_NO_FATAL_FAILURE(exercise(Participant::Fault::prepare));
}

TEST(ReferenceResource, ReceiveFaultUsesDistinctSourcePayload) {
    auto remote_config = fixture();
    remote_config.x = {real{30}, real{-10}, real{7}};
    remote_config.q = {real{5}, real{8}, real{-3}};
    Participant receiver(fixture()), sender(std::move(remote_config));
    Simulation receiver_sim(receiver), sender_sim(sender);
    ASSERT_TRUE(receiver_sim.initialize().ok());
    ASSERT_TRUE(receiver_sim.prepare().ok());
    ASSERT_TRUE(sender_sim.initialize().ok());
    ASSERT_TRUE(sender_sim.prepare().ok());
    ASSERT_TRUE(receiver_sim.advance(step).ok());
    ASSERT_TRUE(sender_sim.advance(step).ok());
    auto baseline = snapshot(receiver);
    const auto expected = capture_accepted(baseline);
    const auto baseline_time = receiver_sim.time();
    const auto baseline_usage = receiver.resource_usage();

    ASSERT_TRUE(receiver_sim.begin_window(step).ok());
    ASSERT_TRUE(receiver_sim.advance_trial().ok());
    ASSERT_TRUE(sender_sim.begin_window(step).ok());
    ASSERT_TRUE(sender_sim.advance_trial().ok());
    auto incoming = sender.send_trial(reference::TrialField::x);
    ASSERT_TRUE(incoming.ok());
    auto receiver_trial = receiver.send_trial(reference::TrialField::x);
    ASSERT_TRUE(receiver_trial.ok());
    ASSERT_NE(incoming.value().values()(0, 0), receiver_trial.value().values()(0, 0));
    receiver_trial.value().complete();
    receiver.fail_next(Participant::Fault::receive);
    ASSERT_EQ(
        receiver.receive_trial(reference::TrialField::x, incoming.value(), incoming.value().stamp())
            .code(),
        StatusCode::execution_failure);
    ASSERT_FALSE(receiver_sim.commit_window().ok());
    ASSERT_EQ(receiver_sim.state(), SimulationState::prepared);
    ASSERT_EQ(receiver_sim.time().accepted_time, baseline_time.accepted_time);
    ASSERT_EQ(receiver_sim.time().accepted_steps, baseline_time.accepted_steps);
    ASSERT_NO_FATAL_FAILURE(check_accepted(expected, baseline));
    ASSERT_NO_FATAL_FAILURE(check_accepted(expected, snapshot(receiver)));
    ASSERT_EQ(receiver.resource_usage().controlled_bytes, baseline_usage.controlled_bytes);
    ASSERT_EQ(receiver.resource_usage().snapshot_records, baseline_usage.snapshot_records);
    ASSERT_TRUE(sender_sim.restore_window().ok());
    ASSERT_TRUE(receiver_sim.advance(step).ok());
    ASSERT_TRUE(receiver_sim.finalize().ok());
    ASSERT_TRUE(sender_sim.finalize().ok());
}

TEST(ReferenceResource, FailedRestoreFailsSimulationWithoutPublishing) {
    Participant participant(fixture());
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(simulation.advance(step).ok());
    auto baseline = snapshot(participant);
    const auto expected = capture_accepted(baseline);
    const auto baseline_time = simulation.time();
    const auto baseline_usage = participant.resource_usage();
    participant.fail_next(Participant::Fault::advance);
    participant.fail_next(Participant::Fault::restore);
    ASSERT_TRUE(simulation.begin_window(step).ok());
    ASSERT_EQ(simulation.advance_trial().code(), StatusCode::execution_failure);
    ASSERT_EQ(simulation.state(), SimulationState::failed);
    ASSERT_EQ(simulation.time().accepted_time, baseline_time.accepted_time);
    ASSERT_EQ(simulation.time().accepted_steps, baseline_time.accepted_steps);
    ASSERT_NO_FATAL_FAILURE(check_accepted(expected, baseline));
    ASSERT_NO_FATAL_FAILURE(check_accepted(expected, snapshot(participant)));
    ASSERT_FALSE(simulation.begin_window(step).ok());
    ASSERT_FALSE(simulation.commit_window().ok());
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_accepted(expected, baseline));
    ASSERT_EQ(participant.resource_usage().controlled_bytes, baseline_usage.controlled_bytes);
    ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    baseline = Snapshot{};
    ASSERT_EQ(participant.resource_usage().controlled_bytes, 0);
    ASSERT_EQ(participant.resource_usage().snapshot_records, 0);
}

TEST(ReferenceResource, PartialInitializationFailureAndClose) {
    Participant participant(fixture());
    participant.fail_next(Participant::Fault::initialize);
    ASSERT_EQ(participant.initialize(0.0).code(), StatusCode::execution_failure);
    ASSERT_FALSE(participant.snapshot().ok());
    ASSERT_EQ(participant.resource_usage().snapshot_records, 0);
    ASSERT_EQ(participant.resource_usage().controlled_bytes, 0);
    ASSERT_TRUE(participant.finalize().ok());

    Participant ready(fixture());
    ASSERT_TRUE(ready.initialize(0.0).ok());
    ASSERT_TRUE(ready.prepare().ok());
    auto retained = snapshot(ready);
    ASSERT_TRUE(ready.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_payload(retained, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0));
}

TEST(ReferenceResource, BudgetSnapshotBackpressureAndRetry) {
    Participant participant(fixture(), ParticipantId{1},
                            reference::ResourceBudget::Limits{1, 1'000'000});
    ASSERT_TRUE(participant.initialize(0.0).ok());
    ASSERT_TRUE(participant.prepare().ok());
    auto first = snapshot(participant);
    const auto initial_usage = participant.resource_usage();
    ASSERT_EQ(initial_usage.snapshot_records, 1);
    ASSERT_EQ(initial_usage.known_external_owner_bytes, 0);
    auto exhausted = participant.snapshot();
    ASSERT_EQ(exhausted.status().code(), StatusCode::resource_exhausted);
    first = Snapshot{};
    auto retry = participant.snapshot();
    ASSERT_TRUE(retry.ok());
    ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, ExactSharedAndPackedAllocationAccounting) {
    constexpr std::size_t count = 3;
    const std::size_t initial_bytes =
        count * sizeof(std::uint64_t) + 2 * count * sizeof(real) + sizeof(std::uint64_t);
    const std::size_t bank_bytes = 2 * count * sizeof(real) + sizeof(std::uint64_t);
    const std::size_t packed_bytes = count * (sizeof(real) + sizeof(std::uint64_t));
    Participant participant(
        fixture(), ParticipantId{1},
        reference::ResourceBudget::Limits{2, initial_bytes + bank_bytes + packed_bytes});
    ASSERT_TRUE(participant.initialize(0.0).ok());
    ASSERT_TRUE(participant.prepare().ok());
    auto usage = participant.resource_usage();
    ASSERT_EQ(usage.controlled_bytes, initial_bytes);
    ASSERT_EQ(usage.snapshot_records, 0);
    auto old_snapshot = snapshot(participant);
    ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);

    const TimeInterval interval{0.0, step, step};
    ASSERT_TRUE(participant.begin_window(interval).ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + bank_bytes);
    ASSERT_TRUE(participant.advance_trial(interval).ok());
    std::optional<TransportView<real>> packed;
    {
        auto result = participant.send_trial(reference::TrialField::x);
        ASSERT_TRUE(result.ok());
        packed.emplace(std::move(result).value());
    }
    usage = participant.resource_usage();
    ASSERT_EQ(usage.snapshot_records, 2);
    ASSERT_EQ(usage.controlled_bytes, initial_bytes + bank_bytes + packed_bytes);
    ASSERT_EQ(participant.send_trial(reference::TrialField::x).status().code(),
              StatusCode::resource_exhausted);
    packed.reset();
    ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + bank_bytes);
    {
        auto retry = participant.send_trial(reference::TrialField::x);
        ASSERT_TRUE(retry.ok());
        ASSERT_EQ(participant.resource_usage().snapshot_records, 2);
    }
    ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    old_snapshot = Snapshot{};
    ASSERT_EQ(participant.resource_usage().snapshot_records, 0);
    ASSERT_TRUE(participant.restore_window().ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, DiagnosticCacheKeysRejectOldAttempt) {
    Participant participant(fixture());
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(simulation.begin_window(step).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto first_key = participant.trial_diagnostic_key();
    ASSERT_TRUE(first_key.ok());
    ASSERT_TRUE(participant.trial_diagnostic(first_key.value()).ok());
    auto wrong_layout = first_key.value();
    wrong_layout.stamp.layout = LayoutVersion{first_key.value().stamp.layout.value() + 1};
    ASSERT_EQ(participant.trial_diagnostic(wrong_layout).status().code(),
              StatusCode::invalid_argument);
    auto wrong_storage = first_key.value();
    ++wrong_storage.stamp.storage_incarnation;
    ASSERT_EQ(participant.trial_diagnostic(wrong_storage).status().code(),
              StatusCode::invalid_argument);
    ASSERT_TRUE(simulation.restore_window().ok());
    ASSERT_TRUE(simulation.begin_window(step).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_EQ(participant.trial_diagnostic(first_key.value()).status().code(),
              StatusCode::invalid_argument);
    const auto retry_key = participant.trial_diagnostic_key();
    ASSERT_TRUE(retry_key.ok());
    ASSERT_TRUE(participant.trial_diagnostic(retry_key.value()).ok());
    ASSERT_TRUE(simulation.restore_window().ok());
}

TEST(ReferenceResource, ControlCounterExhaustionPrecedesCandidateAllocation) {
    auto input = fixture();
    input.control_seed.sequence = ~std::uint64_t{0};
    Participant participant(std::move(input));
    ASSERT_TRUE(participant.initialize(0.0).ok());
    ASSERT_TRUE(participant.prepare().ok());
    const auto before = participant.resource_usage();
    ASSERT_EQ(participant.begin_window(TimeInterval{0.0, step, step}).code(),
              StatusCode::resource_exhausted);
    const auto after = participant.resource_usage();
    ASSERT_EQ(after.controlled_bytes, before.controlled_bytes);
    ASSERT_EQ(after.snapshot_records, before.snapshot_records);
    const auto accepted = snapshot(participant);
    ASSERT_EQ(accepted.history(0), 0);
    ASSERT_EQ(accepted.control.sequence, ~std::uint64_t{0});
    ASSERT_EQ(accepted.diagnostic, ~std::uint64_t{0});
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, TwoBankBudgetWaitsForLastSnapshotHolder) {
    constexpr std::size_t count = 3;
    const std::size_t initial_bytes =
        count * sizeof(std::uint64_t) + 2 * count * sizeof(real) + sizeof(std::uint64_t);
    const std::size_t bank_bytes = 2 * count * sizeof(real) + sizeof(std::uint64_t);
    Participant participant(fixture(), ParticipantId{1},
                            reference::ResourceBudget::Limits{4, initial_bytes + bank_bytes});
    ASSERT_TRUE(participant.initialize(0.0).ok());
    ASSERT_TRUE(participant.prepare().ok());
    std::optional<Snapshot> old_snapshot{snapshot(participant)};
    std::optional<Snapshot> old_snapshot_copy{*old_snapshot};
    const TimeInterval interval{0.0, step, step};
    ASSERT_TRUE(participant.begin_window(interval).ok());
    ASSERT_TRUE(participant.advance_trial(interval).ok());
    ASSERT_TRUE(participant.prepare_commit(interval).ok());
    ASSERT_TRUE(participant.commit_window(interval).ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + bank_bytes);
    const auto accepted = capture_accepted(snapshot(participant));

    const TimeInterval next_interval{step, 2 * step, step};
    {
        const AllocationCounter counter;
        ASSERT_EQ(participant.begin_window(next_interval).code(), StatusCode::resource_exhausted);
        ASSERT_EQ(kokkos_allocations, 0);
    }
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + bank_bytes);
    ASSERT_NO_FATAL_FAILURE(check_accepted(accepted, snapshot(participant)));

    old_snapshot.reset();
    {
        const AllocationCounter counter;
        ASSERT_EQ(participant.begin_window(next_interval).code(), StatusCode::resource_exhausted);
        ASSERT_EQ(kokkos_allocations, 0);
    }
    ASSERT_NO_FATAL_FAILURE(check_accepted(accepted, snapshot(participant)));

    old_snapshot_copy.reset();
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);
    ASSERT_TRUE(participant.begin_window(next_interval).ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + bank_bytes);
    ASSERT_TRUE(participant.restore_window().ok());
    ASSERT_NO_FATAL_FAILURE(check_accepted(accepted, snapshot(participant)));
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(ReferenceResource, FinalizeWithSnapshotAndPackedConsumerOutlivingOwner) {
    constexpr std::size_t count = 3;
    const std::size_t initial_bytes =
        count * sizeof(std::uint64_t) + 2 * count * sizeof(real) + sizeof(std::uint64_t);
    const std::size_t packed_bytes = count * (sizeof(real) + sizeof(std::uint64_t));
    Participant participant(fixture());
    ASSERT_TRUE(participant.initialize(0.0).ok());
    ASSERT_TRUE(participant.prepare().ok());
    std::optional<Snapshot> retained{snapshot(participant)};
    ASSERT_TRUE(participant.begin_window(TimeInterval{0.0, step, step}).ok());
    ASSERT_TRUE(participant.advance_trial(TimeInterval{0.0, step, step}).ok());
    std::optional<TransportView<real>> consumer;
    {
        auto packed = participant.send_trial(reference::TrialField::x);
        ASSERT_TRUE(packed.ok());
        consumer.emplace(std::move(packed).value());
    }
    ASSERT_TRUE(participant.finalize().ok());
    auto usage = participant.resource_usage();
    ASSERT_EQ(usage.snapshot_records, 2);
    ASSERT_EQ(usage.controlled_bytes, initial_bytes + packed_bytes);
    retained.reset();
    usage = participant.resource_usage();
    ASSERT_EQ(usage.snapshot_records, 1);
    ASSERT_EQ(usage.controlled_bytes, packed_bytes);
    consumer->complete();
    consumer.reset();
    usage = participant.resource_usage();
    ASSERT_EQ(usage.snapshot_records, 0);
    ASSERT_EQ(usage.controlled_bytes, 0);
}

TEST(ReferenceResource, SnapshotLifetimeAfterParticipantClose) {
    std::optional<Snapshot> old;
    {
        Participant participant(fixture());
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        old.emplace(snapshot(participant));
        ASSERT_TRUE(simulation.advance(step).ok());
        const auto current = snapshot(participant);
        ASSERT_NE(current.x.data(), old->x.data());
        ASSERT_TRUE(simulation.finalize().ok());
    }
    ASSERT_NO_FATAL_FAILURE(check_payload(*old, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0));
    const auto consumer_a = *old;
    const auto consumer_b = *old;
    old.reset();
    ASSERT_NO_FATAL_FAILURE(check_payload(consumer_a, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0));
    ASSERT_NO_FATAL_FAILURE(check_payload(consumer_b, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0));
}

} // namespace
