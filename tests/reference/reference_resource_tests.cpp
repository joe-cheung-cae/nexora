#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Kokkos_Core.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <participants/reference/Participant.hpp>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
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

Snapshot snapshot(const Participant& participant) {
    auto result = participant.snapshot();
    if (!result.ok()) {
        throw std::runtime_error(result.status().message());
    }
    return std::move(result).value();
}

Config fixture() {
    return {{17, 3, 42}, {real{1}, real{2}, real{-4}},
            {real{0}, real{1}, real{5}}, real{2}};
}

void check_payload(const Snapshot& s, const std::array<real, 3>& x,
                   const std::array<real, 3>& q, double time,
                   std::uint64_t version, std::uint64_t attempt,
                   std::uint64_t accepted_steps,
                   std::optional<std::uint64_t> expected_history = {},
                   std::optional<std::uint64_t> expected_sequence = {}) {
    assert(s.time == time && s.version == StateVersion{version} && s.attempt == attempt);
    assert(s.ids.extent(0) == 3 && s.x.extent(0) == 3 && s.q.extent(0) == 3);
    constexpr std::array<std::uint64_t, 3> ids{17, 3, 42};
    for (std::size_t i = 0; i < 3; ++i) {
        assert(s.ids(i) == ids[i]);
        assert(s.x(i) == x[i] && s.q(i) == q[i]);
    }
    assert(s.history.extent(0) == 1);
    const auto sequence = expected_sequence.value_or(accepted_steps);
    assert(s.control.sequence == sequence);
    const auto history = expected_history.value_or(accepted_steps == 0 ? 0 :
                                                    (accepted_steps == 1 ? 1 : 35));
    assert(s.history(0) == history);
    assert(s.diagnostic == history + sequence);
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
    AcceptedExpectation result{s.stamp(), s.version, s.time, s.attempt, s.control,
                               s.diagnostic, {}, {}, {}, s.history(0), s.ids.data(),
                               s.x.data(), s.q.data(), s.history.data()};
    assert(s.ids.extent(0) == result.ids.size() && s.x.extent(0) == result.x.size() &&
           s.q.extent(0) == result.q.size());
    for (std::size_t i = 0; i < result.ids.size(); ++i) {
        result.ids[i] = s.ids(i);
        result.x[i] = s.x(i);
        result.q[i] = s.q(i);
    }
    return result;
}

void check_accepted(const AcceptedExpectation& expected, const Snapshot& actual) {
    assert(expected.stamp == actual.stamp() && expected.version == actual.version &&
           expected.time == actual.time && expected.attempt == actual.attempt);
    assert(expected.control == actual.control && expected.diagnostic == actual.diagnostic);
    assert(expected.ids_storage == actual.ids.data() && expected.x_storage == actual.x.data() &&
           expected.q_storage == actual.q.data() &&
           expected.history_storage == actual.history.data());
    for (std::size_t i = 0; i < expected.ids.size(); ++i) {
        assert(expected.ids[i] == actual.ids(i) && expected.x[i] == actual.x(i) &&
               expected.q[i] == actual.q(i));
    }
    assert(expected.history == actual.history(0));
}

void test_history_replay_matches_uninterrupted() {
    auto replay_input = fixture();
    replay_input.history_seed = 7;
    replay_input.control_seed.sequence = 4;
    auto oracle_input = replay_input;
    Participant replay(std::move(replay_input));
    Participant oracle(std::move(oracle_input));
    Simulation replay_sim(replay), oracle_sim(oracle);
    assert(replay_sim.initialize().ok() && replay_sim.prepare().ok());
    assert(oracle_sim.initialize().ok() && oracle_sim.prepare().ok());

    const auto initial = snapshot(replay);
    check_payload(initial, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0, 7, 4);

    assert(replay_sim.begin_window(step).ok());
    assert(replay_sim.advance_trial().ok());
    assert(replay_sim.restore_window().ok());
    check_payload(snapshot(replay), {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0, 7, 4);
    assert(replay_sim.time().accepted_time == 0.0 && replay_sim.time().accepted_steps == 0);

    assert(replay_sim.advance(step).ok());
    assert(oracle_sim.advance(step).ok());
    assert(replay_sim.advance(step).ok());
    assert(oracle_sim.advance(step).ok());
    const auto replayed = snapshot(replay);
    const auto uninterrupted = snapshot(oracle);
    assert(replayed.time == uninterrupted.time && replayed.version == uninterrupted.version);
    assert(replayed.control.sequence == uninterrupted.control.sequence);
    assert(replayed.history(0) == uninterrupted.history(0));
    assert(replayed.history.data() != initial.history.data());
    assert(replayed.diagnostic == uninterrupted.diagnostic);
    assert(replayed.history(0) == 7794 && replayed.control.sequence == 6 &&
           replayed.diagnostic == 7800);
    for (std::size_t i = 0; i < 3; ++i) {
        assert(replayed.ids(i) == uninterrupted.ids(i));
        assert(replayed.x(i) == uninterrupted.x(i) && replayed.q(i) == uninterrupted.q(i));
    }
}

void test_faults_preserve_accepted_state() {
    const auto exercise = [](Participant::Fault fault) {
        Participant participant(fixture());
        Simulation simulation(participant);
        assert(simulation.initialize().ok() && simulation.prepare().ok());
        assert(simulation.advance(step).ok());
        auto baseline = snapshot(participant);
        const auto expected = capture_accepted(baseline);
        const auto baseline_time = simulation.time();
        const auto baseline_usage = participant.resource_usage();

        participant.fail_next(fault);
        if (fault == Participant::Fault::begin) {
            assert(simulation.begin_window(step).code() == StatusCode::execution_failure);
        } else if (fault == Participant::Fault::advance) {
            assert(simulation.begin_window(step).ok());
            assert(simulation.advance_trial().code() == StatusCode::execution_failure);
        } else {
            assert(simulation.begin_window(step).ok() && simulation.advance_trial().ok());
            assert(simulation.commit_window().code() == StatusCode::execution_failure);
        }
        assert(simulation.state() == SimulationState::prepared);
        assert(simulation.time().accepted_time == baseline_time.accepted_time &&
               simulation.time().accepted_steps == baseline_time.accepted_steps);
        check_accepted(expected, baseline);
        check_accepted(expected, snapshot(participant));
        assert(participant.resource_usage().controlled_bytes == baseline_usage.controlled_bytes);
        assert(participant.resource_usage().snapshot_records == baseline_usage.snapshot_records);
        assert(simulation.advance(step).ok());
        assert(simulation.time().accepted_steps == baseline_time.accepted_steps + 1);
        assert(simulation.finalize().ok());
    };
    exercise(Participant::Fault::begin);
    exercise(Participant::Fault::advance);
    exercise(Participant::Fault::prepare);
}

void test_receive_fault_uses_distinct_source_payload() {
    auto remote_config = fixture();
    remote_config.x = {real{30}, real{-10}, real{7}};
    remote_config.q = {real{5}, real{8}, real{-3}};
    Participant receiver(fixture()), sender(std::move(remote_config));
    Simulation receiver_sim(receiver), sender_sim(sender);
    assert(receiver_sim.initialize().ok() && receiver_sim.prepare().ok());
    assert(sender_sim.initialize().ok() && sender_sim.prepare().ok());
    assert(receiver_sim.advance(step).ok() && sender_sim.advance(step).ok());
    auto baseline = snapshot(receiver);
    const auto expected = capture_accepted(baseline);
    const auto baseline_time = receiver_sim.time();
    const auto baseline_usage = receiver.resource_usage();

    assert(receiver_sim.begin_window(step).ok() && receiver_sim.advance_trial().ok());
    assert(sender_sim.begin_window(step).ok() && sender_sim.advance_trial().ok());
    auto incoming = sender.send_trial(reference::TrialField::x);
    assert(incoming.ok());
    auto receiver_trial = receiver.send_trial(reference::TrialField::x);
    assert(receiver_trial.ok());
    assert(incoming.value().values()(0, 0) != receiver_trial.value().values()(0, 0));
    receiver_trial.value().complete();
    receiver.fail_next(Participant::Fault::receive);
    assert(receiver.receive_trial(reference::TrialField::x, incoming.value(), incoming.value().stamp())
               .code() == StatusCode::execution_failure);
    assert(!receiver_sim.commit_window().ok());
    assert(receiver_sim.state() == SimulationState::prepared);
    assert(receiver_sim.time().accepted_time == baseline_time.accepted_time &&
           receiver_sim.time().accepted_steps == baseline_time.accepted_steps);
    check_accepted(expected, baseline);
    check_accepted(expected, snapshot(receiver));
    assert(receiver.resource_usage().controlled_bytes == baseline_usage.controlled_bytes);
    assert(receiver.resource_usage().snapshot_records == baseline_usage.snapshot_records);
    assert(sender_sim.restore_window().ok());
    assert(receiver_sim.advance(step).ok());
    assert(receiver_sim.finalize().ok() && sender_sim.finalize().ok());
}

void test_failed_restore_fails_simulation_without_publishing() {
    Participant participant(fixture());
    Simulation simulation(participant);
    assert(simulation.initialize().ok() && simulation.prepare().ok());
    assert(simulation.advance(step).ok());
    auto baseline = snapshot(participant);
    const auto expected = capture_accepted(baseline);
    const auto baseline_time = simulation.time();
    const auto baseline_usage = participant.resource_usage();
    participant.fail_next(Participant::Fault::advance);
    participant.fail_next(Participant::Fault::restore);
    assert(simulation.begin_window(step).ok());
    assert(simulation.advance_trial().code() == StatusCode::execution_failure);
    assert(simulation.state() == SimulationState::failed);
    assert(simulation.time().accepted_time == baseline_time.accepted_time &&
           simulation.time().accepted_steps == baseline_time.accepted_steps);
    check_accepted(expected, baseline);
    check_accepted(expected, snapshot(participant));
    assert(!simulation.begin_window(step).ok() && !simulation.commit_window().ok());
    assert(simulation.finalize().ok());
    check_accepted(expected, baseline);
    assert(participant.resource_usage().controlled_bytes == baseline_usage.controlled_bytes);
    assert(participant.resource_usage().snapshot_records == 1);
    baseline = Snapshot{};
    assert(participant.resource_usage().controlled_bytes == 0);
    assert(participant.resource_usage().snapshot_records == 0);
}

void test_partial_initialization_failure_and_close() {
    Participant participant(fixture());
    participant.fail_next(Participant::Fault::initialize);
    assert(participant.initialize(0.0).code() == StatusCode::execution_failure);
    assert(!participant.snapshot().ok());
    assert(participant.resource_usage().snapshot_records == 0);
    assert(participant.resource_usage().controlled_bytes == 0);
    assert(participant.finalize().ok());

    Participant ready(fixture());
    assert(ready.initialize(0.0).ok() && ready.prepare().ok());
    auto retained = snapshot(ready);
    assert(ready.finalize().ok());
    check_payload(retained, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0);
}

void test_budget_snapshot_backpressure_and_retry() {
    Participant participant(fixture(), ParticipantId{1},
                             reference::ResourceBudget::Limits{1, 1'000'000});
    assert(participant.initialize(0.0).ok() && participant.prepare().ok());
    auto first = snapshot(participant);
    const auto initial_usage = participant.resource_usage();
    assert(initial_usage.snapshot_records == 1);
    assert(initial_usage.known_external_owner_bytes == 0);
    auto exhausted = participant.snapshot();
    assert(exhausted.status().code() == StatusCode::resource_exhausted);
    first = Snapshot{};
    auto retry = participant.snapshot();
    assert(retry.ok());
    assert(participant.resource_usage().snapshot_records == 1);
    assert(participant.finalize().ok());
}

void test_exact_shared_and_packed_allocation_accounting() {
    constexpr std::size_t count = 3;
    const std::size_t initial_bytes = count * sizeof(std::uint64_t) +
                                      2 * count * sizeof(real) + sizeof(std::uint64_t);
    const std::size_t bank_bytes = 2 * count * sizeof(real) + sizeof(std::uint64_t);
    const std::size_t packed_bytes = count * (sizeof(real) + sizeof(std::uint64_t));
    Participant participant(fixture(), ParticipantId{1},
                             reference::ResourceBudget::Limits{2, initial_bytes + bank_bytes + packed_bytes});
    assert(participant.initialize(0.0).ok() && participant.prepare().ok());
    auto usage = participant.resource_usage();
    assert(usage.controlled_bytes == initial_bytes);
    assert(usage.snapshot_records == 0);
    auto old_snapshot = snapshot(participant);
    assert(participant.resource_usage().snapshot_records == 1);
    assert(participant.resource_usage().controlled_bytes == initial_bytes);

    const TimeInterval interval{0.0, step, step};
    assert(participant.begin_window(interval).ok());
    assert(participant.resource_usage().controlled_bytes == initial_bytes + bank_bytes);
    assert(participant.advance_trial(interval).ok());
    std::optional<TransportView<real>> packed;
    {
        auto result = participant.send_trial(reference::TrialField::x);
        assert(result.ok());
        packed.emplace(std::move(result).value());
    }
    usage = participant.resource_usage();
    assert(usage.snapshot_records == 2);
    assert(usage.controlled_bytes == initial_bytes + bank_bytes + packed_bytes);
    assert(participant.send_trial(reference::TrialField::x).status().code() ==
           StatusCode::resource_exhausted);
    packed.reset();
    assert(participant.resource_usage().snapshot_records == 1);
    assert(participant.resource_usage().controlled_bytes == initial_bytes + bank_bytes);
    {
        auto retry = participant.send_trial(reference::TrialField::x);
        assert(retry.ok());
        assert(participant.resource_usage().snapshot_records == 2);
    }
    assert(participant.resource_usage().snapshot_records == 1);
    old_snapshot = Snapshot{};
    assert(participant.resource_usage().snapshot_records == 0);
    assert(participant.restore_window().ok());
    assert(participant.resource_usage().controlled_bytes == initial_bytes);
    assert(participant.finalize().ok());
}

void test_diagnostic_cache_keys_reject_old_attempt() {
    Participant participant(fixture());
    Simulation simulation(participant);
    assert(simulation.initialize().ok() && simulation.prepare().ok());
    assert(simulation.begin_window(step).ok() && simulation.advance_trial().ok());
    const auto first_key = participant.trial_diagnostic_key();
    assert(first_key.ok());
    assert(participant.trial_diagnostic(first_key.value()).ok());
    auto wrong_layout = first_key.value();
    wrong_layout.stamp.layout = LayoutVersion{first_key.value().stamp.layout.value() + 1};
    assert(participant.trial_diagnostic(wrong_layout).status().code() ==
           StatusCode::invalid_argument);
    auto wrong_storage = first_key.value();
    ++wrong_storage.stamp.storage_incarnation;
    assert(participant.trial_diagnostic(wrong_storage).status().code() ==
           StatusCode::invalid_argument);
    assert(simulation.restore_window().ok());
    assert(simulation.begin_window(step).ok() && simulation.advance_trial().ok());
    assert(participant.trial_diagnostic(first_key.value()).status().code() ==
           StatusCode::invalid_argument);
    const auto retry_key = participant.trial_diagnostic_key();
    assert(retry_key.ok());
    assert(participant.trial_diagnostic(retry_key.value()).ok());
    assert(simulation.restore_window().ok());
}

void test_control_counter_exhaustion_precedes_candidate_allocation() {
    auto input = fixture();
    input.control_seed.sequence = ~std::uint64_t{0};
    Participant participant(std::move(input));
    assert(participant.initialize(0.0).ok() && participant.prepare().ok());
    const auto before = participant.resource_usage();
    assert(participant.begin_window(TimeInterval{0.0, step, step}).code() ==
           StatusCode::resource_exhausted);
    const auto after = participant.resource_usage();
    assert(after.controlled_bytes == before.controlled_bytes);
    assert(after.snapshot_records == before.snapshot_records);
    const auto accepted = snapshot(participant);
    assert(accepted.history(0) == 0 && accepted.control.sequence == ~std::uint64_t{0});
    assert(accepted.diagnostic == ~std::uint64_t{0});
    assert(participant.finalize().ok());
}

void test_two_bank_budget_waits_for_last_snapshot_holder() {
    constexpr std::size_t count = 3;
    const std::size_t initial_bytes = count * sizeof(std::uint64_t) +
                                      2 * count * sizeof(real) + sizeof(std::uint64_t);
    const std::size_t bank_bytes = 2 * count * sizeof(real) + sizeof(std::uint64_t);
    Participant participant(fixture(), ParticipantId{1},
                             reference::ResourceBudget::Limits{4, initial_bytes + bank_bytes});
    assert(participant.initialize(0.0).ok() && participant.prepare().ok());
    std::optional<Snapshot> old_snapshot{snapshot(participant)};
    std::optional<Snapshot> old_snapshot_copy{*old_snapshot};
    const TimeInterval interval{0.0, step, step};
    assert(participant.begin_window(interval).ok());
    assert(participant.advance_trial(interval).ok());
    assert(participant.prepare_commit(interval).ok());
    assert(participant.commit_window(interval).ok());
    assert(participant.resource_usage().controlled_bytes == initial_bytes + bank_bytes);
    const auto accepted = capture_accepted(snapshot(participant));

    Kokkos::Tools::Experimental::set_allocate_data_callback(count_kokkos_allocation);
    kokkos_allocations = 0;
    const auto exhausted_with_two_holders = participant.begin_window(TimeInterval{step, 2 * step, step});
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    assert(exhausted_with_two_holders.code() == StatusCode::resource_exhausted);
    assert(kokkos_allocations == 0);
    assert(participant.resource_usage().controlled_bytes == initial_bytes + bank_bytes);
    check_accepted(accepted, snapshot(participant));
    old_snapshot.reset();
    Kokkos::Tools::Experimental::set_allocate_data_callback(count_kokkos_allocation);
    kokkos_allocations = 0;
    const auto exhausted_with_one_holder = participant.begin_window(TimeInterval{step, 2 * step, step});
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    assert(exhausted_with_one_holder.code() == StatusCode::resource_exhausted);
    assert(kokkos_allocations == 0);
    check_accepted(accepted, snapshot(participant));

    old_snapshot_copy.reset();
    assert(participant.resource_usage().controlled_bytes == initial_bytes);
    assert(participant.begin_window(TimeInterval{step, 2 * step, step}).ok());
    assert(participant.resource_usage().controlled_bytes == initial_bytes + bank_bytes);
    assert(participant.restore_window().ok());
    check_accepted(accepted, snapshot(participant));
    assert(participant.finalize().ok());
}

void test_finalize_with_snapshot_and_packed_consumer_outliving_owner() {
    constexpr std::size_t count = 3;
    const std::size_t initial_bytes = count * sizeof(std::uint64_t) +
                                      2 * count * sizeof(real) + sizeof(std::uint64_t);
    const std::size_t packed_bytes = count * (sizeof(real) + sizeof(std::uint64_t));
    Participant participant(fixture());
    assert(participant.initialize(0.0).ok() && participant.prepare().ok());
    std::optional<Snapshot> retained{snapshot(participant)};
    assert(participant.begin_window(TimeInterval{0.0, step, step}).ok());
    assert(participant.advance_trial(TimeInterval{0.0, step, step}).ok());
    std::optional<TransportView<real>> consumer;
    {
        auto packed = participant.send_trial(reference::TrialField::x);
        assert(packed.ok());
        consumer.emplace(std::move(packed).value());
    }
    assert(participant.finalize().ok());
    auto usage = participant.resource_usage();
    assert(usage.snapshot_records == 2);
    assert(usage.controlled_bytes == initial_bytes + packed_bytes);
    retained.reset();
    usage = participant.resource_usage();
    assert(usage.snapshot_records == 1 && usage.controlled_bytes == packed_bytes);
    consumer->complete();
    consumer.reset();
    usage = participant.resource_usage();
    assert(usage.snapshot_records == 0 && usage.controlled_bytes == 0);
}

void test_snapshot_lifetime_after_participant_close() {
    std::optional<Snapshot> old;
    {
        Participant participant(fixture());
        Simulation simulation(participant);
        assert(simulation.initialize().ok() && simulation.prepare().ok());
        old.emplace(snapshot(participant));
        assert(simulation.advance(step).ok());
        const auto current = snapshot(participant);
        assert(current.x.data() != old->x.data());
        assert(simulation.finalize().ok());
    }
    check_payload(*old, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0);
    const auto consumer_a = *old;
    const auto consumer_b = *old;
    old.reset();
    check_payload(consumer_a, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0);
    check_payload(consumer_b, {1, 2, -4}, {0, 1, 5}, 0.0, 0, 0, 0);
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    test_history_replay_matches_uninterrupted();
    test_faults_preserve_accepted_state();
    test_receive_fault_uses_distinct_source_payload();
    test_failed_restore_fails_simulation_without_publishing();
    test_partial_initialization_failure_and_close();
    test_budget_snapshot_backpressure_and_retry();
    test_exact_shared_and_packed_allocation_accounting();
    test_diagnostic_cache_keys_reject_old_attempt();
    test_control_counter_exhaustion_precedes_candidate_allocation();
    test_two_bank_budget_waits_for_last_snapshot_holder();
    test_finalize_with_snapshot_and_packed_consumer_outliving_owner();
    test_snapshot_lifetime_after_participant_close();
    std::cout << "reference resource lifecycle tests passed\n";
}
