#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Kokkos_Core.hpp>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <data/FieldIssuer.hpp>
#include <data/TransportBuilder.hpp>
#include <iostream>
#include <limits>
#include <nexora/simulation/Simulation.hpp>
#include <optional>
#include <participants/reference/Participant.hpp>
#include <source_location>
#include <utility>

namespace {
using namespace nexora;
using detail::FieldAxis;
using detail::FieldIssuer;
using detail::TransportBuilder;
using reference::Participant;
using reference::Snapshot;
using reference::TrialField;
using Id = std::uint64_t;
constexpr std::array<Id, 3> ids{17, 3, 42};
constexpr std::array<Id, 3> shuffled_ids{42, 17, 3};
constexpr std::array<real, 3> initial_x{1.0, 2.0, -4.0};
constexpr std::array<real, 3> initial_q{0.0, 1.0, 5.0};
constexpr std::array<real, 3> incoming_x{8.0, 4.0, -2.0};
constexpr std::array<real, 3> incoming_q{-8.0, 1.0, 6.0};
constexpr std::array<real, 3> received_x{4.0, -2.0, 8.0};
constexpr std::array<real, 3> received_q{1.0, 6.0, -8.0};
constexpr double step = 0.125;

template <class T>
T take(Result<T> result, const std::source_location source = std::source_location::current()) {
    if (!result.ok()) {
        std::cerr << source.file_name() << ':' << source.line() << ": " << result.status().message()
                  << '\n';
        std::abort();
    }
    return std::move(result).value();
}

FieldDescriptor input_descriptor(TrialField field) {
    FieldDescriptor descriptor;
    descriptor.id = FieldId{field == TrialField::x ? 1U : 2U};
    descriptor.name = field == TrialField::x ? "x" : "q";
    descriptor.association = Association::node;
    descriptor.entity_set = "reference_samples";
    descriptor.units = "1";
    descriptor.time_level = "trial";
    return descriptor;
}

TransportView<real> pack_input(const Kokkos::Serial& exec, TrialField field, FieldStamp stamp,
                                 const std::array<real, 3>& data,
                                 const std::array<Id, 3>& row_ids = shuffled_ids) {
    Kokkos::View<real*> values("reference input values", data.size());
    Kokkos::View<Id*> input_ids("reference input IDs", row_ids.size());
    for (std::size_t row = 0; row < data.size(); ++row) {
        values(row) = data[row];
        input_ids(row) = row_ids[row];
    }

    const std::array axes{FieldAxis{data.size()}};
    auto descriptor = input_descriptor(field);
    auto id_descriptor = descriptor;
    id_descriptor.id = FieldId{3};
    id_descriptor.name = "stable IDs";
    id_descriptor.scalar_type = ScalarType::uint64;

    auto value_issuer = take(FieldIssuer::create(values, stamp));
    auto id_issuer = take(FieldIssuer::create(input_ids, stamp));
    auto read_values = take(value_issuer.read(values, descriptor, value_issuer.stamp(),
                                              {Association::node, "1", axes}));
    auto read_ids = take(id_issuer.read(input_ids, id_descriptor, id_issuer.stamp(),
                                       {Association::node, "1", axes}));
    return take(TransportBuilder::prepare(exec, TransportPurpose::coupling,
                                          std::move(read_values), std::move(read_ids)));
}

void check_snapshot(const Snapshot& snapshot, const std::array<real, 3>& x,
                    const std::array<real, 3>& q, double time, std::uint64_t version,
                    std::uint64_t attempt, const std::array<Id, 3>& expected_ids = ids) {
    assert(snapshot.time == time && snapshot.version == StateVersion{version});
    assert(snapshot.attempt == attempt);
    assert(snapshot.ids.extent(0) == expected_ids.size() && snapshot.x.extent(0) == x.size() &&
           snapshot.q.extent(0) == q.size());
    for (std::size_t row = 0; row < expected_ids.size(); ++row) {
        assert(snapshot.ids(row) == expected_ids[row] && snapshot.x(row) == x[row] &&
               snapshot.q(row) == q[row]);
    }
}

void check_original(const Participant& participant, const Snapshot& original) {
    const auto snapshot = take(participant.snapshot());
    check_snapshot(snapshot, initial_x, initial_q, 0.0, 0, 0);
    assert(snapshot.ids.data() == original.ids.data() && snapshot.x.data() == original.x.data() &&
           snapshot.q.data() == original.q.data());
}

reference::Config config() {
    return {{ids.begin(), ids.end()}, {initial_x.begin(), initial_x.end()},
            {initial_q.begin(), initial_q.end()}, 2.0};
}

void test_rollback_retry_and_retained_snapshots(const Kokkos::Serial& exec) {
    std::optional<Snapshot> original;
    std::optional<Snapshot> imported;
    {
        Participant participant(config());
        Simulation simulation(participant);
        assert(simulation.initialize().ok() && simulation.prepare().ok());
        original.emplace(take(participant.snapshot()));
        assert(!participant.trial_input_stamp().ok());

        auto advance_trial = [&] {
            assert(simulation.begin_window(step).ok());
            assert(!participant.trial_input_stamp().ok());
            assert(simulation.advance_trial().ok());
            return take(participant.trial_input_stamp());
        };
        auto check_rolled_back = [&] {
            check_original(participant, *original);
            assert(simulation.state() == SimulationState::prepared);
            assert(simulation.time().accepted_time == 0.0 && simulation.time().accepted_steps == 0);
            assert(!participant.trial_input_stamp().ok());
        };

        const auto first_stamp = advance_trial();
        const auto old_input = pack_input(exec, TrialField::x, first_stamp, incoming_x);
        assert(participant.receive_trial(TrialField::x, old_input, old_input.stamp()).ok());
        check_original(participant, *original);
        assert(simulation.restore_window().ok());
        check_rolled_back();

        const auto retry_stamp = advance_trial();
        assert(retry_stamp.time == first_stamp.time &&
               retry_stamp.trial.window == first_stamp.trial.window);
        assert(retry_stamp.trial.attempt > first_stamp.trial.attempt);
        assert(participant.receive_trial(TrialField::x, old_input, old_input.stamp()).code() ==
               StatusCode::invalid_state);
        assert(!simulation.commit_window().ok());
        check_rolled_back();

        const auto missing_ids_stamp = advance_trial();
        const auto bad_ids = pack_input(exec, TrialField::x, missing_ids_stamp, incoming_x,
                                        {42, 17, 999});
        assert(participant.receive_trial(TrialField::x, bad_ids, bad_ids.stamp()).code() ==
               StatusCode::invalid_argument);
        assert(!simulation.commit_window().ok());
        check_rolled_back();

        const auto nonfinite_stamp = advance_trial();
        const auto valid_x = pack_input(exec, TrialField::x, nonfinite_stamp, incoming_x);
        assert(participant.receive_trial(TrialField::x, valid_x, valid_x.stamp()).ok());
        const auto nonfinite = pack_input(exec, TrialField::q, nonfinite_stamp,
                                          {1.0, std::numeric_limits<real>::infinity(), 2.0});
        assert(participant.receive_trial(TrialField::q, nonfinite, nonfinite.stamp()).code() ==
               StatusCode::numerical_failure);
        assert(!simulation.commit_window().ok());
        check_rolled_back();

        const auto commit_stamp = advance_trial();
        const auto x = pack_input(exec, TrialField::x, commit_stamp, incoming_x);
        const auto q = pack_input(exec, TrialField::q, commit_stamp, incoming_q);
        assert(participant.receive_trial(TrialField::x, x, x.stamp()).ok());
        assert(participant.receive_trial(TrialField::q, q, q.stamp()).ok());
        check_original(participant, *original);
        assert(simulation.commit_window().ok());
        imported.emplace(take(participant.snapshot()));
        check_snapshot(*imported, received_x, received_q, step, 1, commit_stamp.trial.attempt);
        assert(imported->x.data() != original->x.data() && imported->q.data() != original->q.data());
        assert(simulation.time().accepted_time == step && simulation.time().accepted_steps == 1);
        assert(simulation.time().trial_attempts == 5);
        assert(!participant.trial_input_stamp().ok());
        assert(!participant.receive_trial(TrialField::x, x, x.stamp()).ok());

        assert(simulation.advance(step).ok());
        check_snapshot(take(participant.snapshot()), {3.0, -1.5, 6.0}, {2.0, 5.5, -6.0},
                       2 * step, 2, commit_stamp.trial.attempt + 1);
        assert(simulation.finalize().ok());
        assert(!participant.snapshot().ok());
    }

    check_snapshot(*original, initial_x, initial_q, 0.0, 0, 0);
    check_snapshot(*imported, received_x, received_q, step, 1, 5);
    const auto retained_view = original->x;
    auto stamp = original->stamp();
    stamp.storage_incarnation = 0;
    {
        const auto copy = *original;
        original.reset();
        assert(FieldIssuer::create(retained_view, stamp).status().code() ==
               StatusCode::already_exists);
        check_snapshot(copy, initial_x, initial_q, 0.0, 0, 0);
    }
    assert(FieldIssuer::create(retained_view, stamp).ok());
}

void test_precommit_publication(const Kokkos::Serial& exec) {
    Participant participant(config());
    assert(participant.initialize(0.0).ok() && participant.prepare().ok());
    const auto original = take(participant.snapshot());
    const TimeInterval interval{0.0, step, step};
    assert(participant.begin_window(interval).ok() && participant.advance_trial(interval).ok());
    const auto stamp = take(participant.trial_input_stamp());
    const auto input = pack_input(exec, TrialField::x, stamp, incoming_x);
    assert(participant.receive_trial(TrialField::x, input, input.stamp()).ok());
    assert(participant.prepare_commit(interval).ok());
    assert(!participant.trial_input_stamp().ok());
    assert(participant.send_trial(TrialField::x).status().code() == StatusCode::invalid_state);
    assert(participant.receive_trial(TrialField::x, input, input.stamp()).code() ==
           StatusCode::invalid_state);
    check_original(participant, original);
    assert(participant.commit_window(interval).ok());
    check_snapshot(take(participant.snapshot()), received_x, {0.25, 1.5, 4.0}, step, 1,
                   stamp.trial.attempt);
    assert(participant.finalize().ok());
}

reference::Config sender_config() {
    return {{shuffled_ids.begin(), shuffled_ids.end()}, {incoming_x.begin(), incoming_x.end()},
            {incoming_q.begin(), incoming_q.end()}, 2.0};
}

void advance_window(Simulation& simulation) {
    assert(simulation.begin_window(step).ok() && simulation.advance_trial().ok());
}

void check_sent(const TransportView<real>& payload, TrialField field,
                const std::array<real, 3>& values, double time, std::uint64_t attempt) {
    assert(payload && payload.purpose() == TransportPurpose::coupling);
    assert(payload.descriptor().id == input_descriptor(field).id);
    assert(payload.stamp().time.physical_time == time && payload.stamp().trial.attempt == attempt);
    assert(payload.stable_ids().extent(0) == shuffled_ids.size());
    assert(payload.values().extent(0) == values.size() && payload.values().extent(1) == 1);
    for (std::size_t row = 0; row < values.size(); ++row) {
        assert(payload.stable_ids()(row) == shuffled_ids[row]);
        assert(payload.values()(row, 0) == values[row]);
    }
}

void test_two_reference_commit_and_sender_snapshot_lifetime() {
    constexpr std::array<real, 3> sender_x{6.0, 3.0, -1.5};
    constexpr std::array<real, 3> sender_q{-6.0, 2.0, 5.5};
    constexpr std::array<real, 3> later_x{4.5, 2.25, -1.125};
    std::optional<TransportView<real>> committed_payload;
    std::optional<TransportView<real>> rolled_back_payload;
    {
        Participant sender(sender_config(), ParticipantId{1});
        Participant receiver(config(), ParticipantId{2});
        Simulation sender_simulation(sender);
        Simulation receiver_simulation(receiver);
        assert(sender_simulation.initialize().ok() && sender_simulation.prepare().ok());
        assert(receiver_simulation.initialize().ok() && receiver_simulation.prepare().ok());
        const auto receiver_original = take(receiver.snapshot());

        assert(sender_simulation.begin_window(step).ok());
        assert(sender.send_trial(TrialField::x).status().code() == StatusCode::invalid_state);
        assert(sender_simulation.advance_trial().ok());
        advance_window(receiver_simulation);
        auto x = take(sender.send_trial(TrialField::x));
        auto q = take(sender.send_trial(TrialField::q));
        check_sent(x, TrialField::x, sender_x, step, 1);
        check_sent(q, TrialField::q, sender_q, step, 1);

        assert(receiver.receive_trial(TrialField::x, x, x.stamp()).ok());
        assert(receiver.receive_trial(TrialField::q, q, q.stamp()).ok());
        check_original(receiver, receiver_original);
        check_snapshot(take(sender.snapshot()), incoming_x, incoming_q, 0.0, 0, 0, shuffled_ids);

        committed_payload.emplace(std::move(x));
        assert(!x);
        assert(sender_simulation.commit_window().ok());
        assert(sender.send_trial(TrialField::x).status().code() == StatusCode::invalid_state);
        check_sent(*committed_payload, TrialField::x, sender_x, step, 1);
        check_sent(q, TrialField::q, sender_q, step, 1);
        check_original(receiver, receiver_original);
        assert(receiver_simulation.commit_window().ok());
        check_snapshot(take(sender.snapshot()), sender_x, sender_q, step, 1, 1, shuffled_ids);
        check_snapshot(take(receiver.snapshot()), {3.0, -1.5, 6.0}, {2.0, 5.5, -6.0}, step, 1, 1);

        advance_window(sender_simulation);
        auto later = take(sender.send_trial(TrialField::x));
        q = std::move(later);
        assert(!later);
        check_sent(q, TrialField::x, later_x, 2 * step, 2);
        rolled_back_payload.emplace(std::move(q));
        assert(!q);
        assert(sender_simulation.restore_window().ok());
        check_snapshot(take(sender.snapshot()), sender_x, sender_q, step, 1, 1, shuffled_ids);
        assert(sender_simulation.finalize().ok() && receiver_simulation.finalize().ok());
        assert(sender.send_trial(TrialField::x).status().code() == StatusCode::invalid_state);
    }

    check_sent(*committed_payload, TrialField::x, sender_x, step, 1);
    check_sent(*rolled_back_payload, TrialField::x, later_x, 2 * step, 2);
    committed_payload->complete();
    rolled_back_payload->complete();
    assert(!*committed_payload && !*rolled_back_payload);
}

void test_two_reference_receive_failure_and_retry() {
    Participant sender(sender_config(), ParticipantId{1});
    Participant receiver(config(), ParticipantId{2});
    Simulation sender_simulation(sender);
    Simulation receiver_simulation(receiver);
    assert(sender_simulation.initialize().ok() && sender_simulation.prepare().ok());
    assert(receiver_simulation.initialize().ok() && receiver_simulation.prepare().ok());
    const auto receiver_original = take(receiver.snapshot());

    auto rollback_both = [&] {
        assert(receiver_simulation.commit_window().code() == StatusCode::invalid_state);
        assert(sender_simulation.restore_window().ok());
        check_original(receiver, receiver_original);
        check_snapshot(take(sender.snapshot()), incoming_x, incoming_q, 0.0, 0, 0, shuffled_ids);
        for (const auto* simulation : {&sender_simulation, &receiver_simulation}) {
            assert(simulation->state() == SimulationState::prepared);
            assert(simulation->time().accepted_time == 0.0 && simulation->time().accepted_steps == 0);
        }
    };

    advance_window(sender_simulation);
    advance_window(receiver_simulation);
    const auto first = take(sender.send_trial(TrialField::x));
    assert(receiver.receive_trial(TrialField::x, first, first.stamp()).ok());
    assert(receiver.receive_trial(TrialField::q, first, first.stamp()).code() ==
           StatusCode::invalid_argument);
    rollback_both();

    advance_window(sender_simulation);
    advance_window(receiver_simulation);
    const auto retry_stamp = take(receiver.trial_input_stamp());
    assert(retry_stamp.time == first.stamp().time);
    assert(retry_stamp.trial.attempt > first.stamp().trial.attempt);
    assert(receiver.receive_trial(TrialField::x, first, first.stamp()).code() ==
           StatusCode::invalid_state);
    rollback_both();

    advance_window(sender_simulation);
    advance_window(receiver_simulation);
    const auto current = take(sender.send_trial(TrialField::x));
    assert(current.stamp().trial.attempt == 3);
    assert(receiver.receive_trial(TrialField::x, current, current.stamp()).ok());
    assert(sender_simulation.commit_window().ok() && receiver_simulation.commit_window().ok());
    check_snapshot(take(sender.snapshot()), {6.0, 3.0, -1.5}, {-6.0, 2.0, 5.5}, step, 1, 3,
                   shuffled_ids);
    check_snapshot(take(receiver.snapshot()), {3.0, -1.5, 6.0}, {0.25, 1.5, 4.0}, step, 1, 3);
    for (const auto* simulation : {&sender_simulation, &receiver_simulation}) {
        assert(simulation->time().accepted_time == step && simulation->time().accepted_steps == 1);
        assert(simulation->time().trial_attempts == 3);
    }
    assert(sender_simulation.finalize().ok() && receiver_simulation.finalize().ok());
}
} // namespace

int main(int argc, char* argv[]) {
    const Kokkos::ScopeGuard runtime(argc, argv);
    const Kokkos::Serial exec;
    test_rollback_retry_and_retained_snapshots(exec);
    test_precommit_publication(exec);
    test_two_reference_commit_and_sender_snapshot_lifetime();
    test_two_reference_receive_failure_and_retry();
}
