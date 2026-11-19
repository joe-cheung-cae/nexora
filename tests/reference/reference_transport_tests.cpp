#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <data/FieldIssuer.hpp>
#include <data/TransportBuilder.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <nexora/simulation/Simulation.hpp>
#include <optional>
#include <participants/reference/Participant.hpp>
#include <stdexcept>
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

template <class T> T take(Result<T> result) {
    if (!result.ok()) {
        throw std::runtime_error(result.status().message());
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
    return take(TransportBuilder::prepare(exec, TransportPurpose::coupling, std::move(read_values),
                                          std::move(read_ids)));
}

void check_snapshot(const Snapshot& snapshot, const std::array<real, 3>& x,
                    const std::array<real, 3>& q, double time, std::uint64_t version,
                    std::uint64_t attempt, const std::array<Id, 3>& expected_ids = ids) {
    ASSERT_EQ(snapshot.time, time);
    ASSERT_EQ(snapshot.version, StateVersion{version});
    ASSERT_EQ(snapshot.attempt, attempt);
    ASSERT_EQ(snapshot.ids.extent(0), expected_ids.size());
    ASSERT_EQ(snapshot.x.extent(0), x.size());
    ASSERT_EQ(snapshot.q.extent(0), q.size());
    for (std::size_t row = 0; row < expected_ids.size(); ++row) {
        ASSERT_EQ(snapshot.ids(row), expected_ids[row]);
        ASSERT_EQ(snapshot.x(row), x[row]);
        ASSERT_EQ(snapshot.q(row), q[row]);
    }
}

void check_original(const Participant& participant, const Snapshot& original) {
    const auto snapshot = take(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(snapshot, initial_x, initial_q, 0.0, 0, 0));
    ASSERT_EQ(snapshot.ids.data(), original.ids.data());
    ASSERT_EQ(snapshot.x.data(), original.x.data());
    ASSERT_EQ(snapshot.q.data(), original.q.data());
}

reference::Config config() {
    return {{ids.begin(), ids.end()},
            {initial_x.begin(), initial_x.end()},
            {initial_q.begin(), initial_q.end()},
            2.0};
}

TEST(ReferenceTransport, RollbackRetryAndRetainedSnapshots) {
    const Kokkos::Serial exec;
    std::optional<Snapshot> original;
    std::optional<Snapshot> imported;
    {
        Participant participant(config());
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        original.emplace(take(participant.snapshot()));
        ASSERT_FALSE(participant.trial_input_stamp().ok());

        auto advance_trial = [&](FieldStamp& stamp) {
            ASSERT_TRUE(simulation.begin_window(step).ok());
            ASSERT_FALSE(participant.trial_input_stamp().ok());
            ASSERT_TRUE(simulation.advance_trial().ok());
            stamp = take(participant.trial_input_stamp());
        };
        auto check_rolled_back = [&] {
            ASSERT_NO_FATAL_FAILURE(check_original(participant, *original));
            ASSERT_EQ(simulation.state(), SimulationState::prepared);
            ASSERT_EQ(simulation.time().accepted_time, 0.0);
            ASSERT_EQ(simulation.time().accepted_steps, 0);
            ASSERT_FALSE(participant.trial_input_stamp().ok());
        };

        FieldStamp first_stamp;
        ASSERT_NO_FATAL_FAILURE(advance_trial(first_stamp));
        const auto old_input = pack_input(exec, TrialField::x, first_stamp, incoming_x);
        ASSERT_TRUE(participant.receive_trial(TrialField::x, old_input, old_input.stamp()).ok());
        ASSERT_NO_FATAL_FAILURE(check_original(participant, *original));
        ASSERT_TRUE(simulation.restore_window().ok());
        ASSERT_NO_FATAL_FAILURE(check_rolled_back());

        FieldStamp retry_stamp;
        ASSERT_NO_FATAL_FAILURE(advance_trial(retry_stamp));
        ASSERT_EQ(retry_stamp.time, first_stamp.time);
        ASSERT_EQ(retry_stamp.trial.window, first_stamp.trial.window);
        ASSERT_GT(retry_stamp.trial.attempt, first_stamp.trial.attempt);
        ASSERT_EQ(participant.receive_trial(TrialField::x, old_input, old_input.stamp()).code(),
                  StatusCode::invalid_state);
        ASSERT_FALSE(simulation.commit_window().ok());
        ASSERT_NO_FATAL_FAILURE(check_rolled_back());

        FieldStamp missing_ids_stamp;
        ASSERT_NO_FATAL_FAILURE(advance_trial(missing_ids_stamp));
        const auto bad_ids =
            pack_input(exec, TrialField::x, missing_ids_stamp, incoming_x, {42, 17, 999});
        ASSERT_EQ(participant.receive_trial(TrialField::x, bad_ids, bad_ids.stamp()).code(),
                  StatusCode::invalid_argument);
        ASSERT_FALSE(simulation.commit_window().ok());
        ASSERT_NO_FATAL_FAILURE(check_rolled_back());

        FieldStamp nonfinite_stamp;
        ASSERT_NO_FATAL_FAILURE(advance_trial(nonfinite_stamp));
        const auto valid_x = pack_input(exec, TrialField::x, nonfinite_stamp, incoming_x);
        ASSERT_TRUE(participant.receive_trial(TrialField::x, valid_x, valid_x.stamp()).ok());
        const auto nonfinite = pack_input(exec, TrialField::q, nonfinite_stamp,
                                          {1.0, std::numeric_limits<real>::infinity(), 2.0});
        ASSERT_EQ(participant.receive_trial(TrialField::q, nonfinite, nonfinite.stamp()).code(),
                  StatusCode::numerical_failure);
        ASSERT_FALSE(simulation.commit_window().ok());
        ASSERT_NO_FATAL_FAILURE(check_rolled_back());

        FieldStamp commit_stamp;
        ASSERT_NO_FATAL_FAILURE(advance_trial(commit_stamp));
        const auto x = pack_input(exec, TrialField::x, commit_stamp, incoming_x);
        const auto q = pack_input(exec, TrialField::q, commit_stamp, incoming_q);
        ASSERT_TRUE(participant.receive_trial(TrialField::x, x, x.stamp()).ok());
        ASSERT_TRUE(participant.receive_trial(TrialField::q, q, q.stamp()).ok());
        ASSERT_NO_FATAL_FAILURE(check_original(participant, *original));
        ASSERT_TRUE(simulation.commit_window().ok());
        imported.emplace(take(participant.snapshot()));
        ASSERT_NO_FATAL_FAILURE(
            check_snapshot(*imported, received_x, received_q, step, 1, commit_stamp.trial.attempt));
        ASSERT_NE(imported->x.data(), original->x.data());
        ASSERT_NE(imported->q.data(), original->q.data());
        ASSERT_EQ(simulation.time().accepted_time, step);
        ASSERT_EQ(simulation.time().accepted_steps, 1);
        ASSERT_EQ(simulation.time().trial_attempts, 5);
        ASSERT_FALSE(participant.trial_input_stamp().ok());
        ASSERT_FALSE(participant.receive_trial(TrialField::x, x, x.stamp()).ok());

        ASSERT_TRUE(simulation.advance(step).ok());
        ASSERT_NO_FATAL_FAILURE(check_snapshot(take(participant.snapshot()), {3.0, -1.5, 6.0},
                                               {2.0, 5.5, -6.0}, 2 * step, 2,
                                               commit_stamp.trial.attempt + 1));
        ASSERT_TRUE(simulation.finalize().ok());
        ASSERT_FALSE(participant.snapshot().ok());
    }

    ASSERT_NO_FATAL_FAILURE(check_snapshot(*original, initial_x, initial_q, 0.0, 0, 0));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(*imported, received_x, received_q, step, 1, 5));
    const auto retained_view = original->x;
    auto stamp = original->stamp();
    stamp.storage_incarnation = 0;
    {
        const auto copy = *original;
        original.reset();
        ASSERT_EQ(FieldIssuer::create(retained_view, stamp).status().code(),
                  StatusCode::already_exists);
        ASSERT_NO_FATAL_FAILURE(check_snapshot(copy, initial_x, initial_q, 0.0, 0, 0));
    }
    ASSERT_TRUE(FieldIssuer::create(retained_view, stamp).ok());
}

TEST(ReferenceTransport, PrecommitPublication) {
    const Kokkos::Serial exec;
    Participant participant(config());
    ASSERT_TRUE(participant.initialize(0.0).ok());
    ASSERT_TRUE(participant.prepare().ok());
    const auto original = take(participant.snapshot());
    const TimeInterval interval{0.0, step, step};
    ASSERT_TRUE(participant.begin_window(interval).ok());
    ASSERT_TRUE(participant.advance_trial(interval).ok());
    const auto stamp = take(participant.trial_input_stamp());
    const auto input = pack_input(exec, TrialField::x, stamp, incoming_x);
    ASSERT_TRUE(participant.receive_trial(TrialField::x, input, input.stamp()).ok());
    ASSERT_TRUE(participant.prepare_commit(interval).ok());
    ASSERT_FALSE(participant.trial_input_stamp().ok());
    ASSERT_EQ(participant.send_trial(TrialField::x).status().code(), StatusCode::invalid_state);
    ASSERT_EQ(participant.receive_trial(TrialField::x, input, input.stamp()).code(),
              StatusCode::invalid_state);
    ASSERT_NO_FATAL_FAILURE(check_original(participant, original));
    ASSERT_TRUE(participant.commit_window(interval).ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(take(participant.snapshot()), received_x,
                                           {0.25, 1.5, 4.0}, step, 1, stamp.trial.attempt));
    ASSERT_TRUE(participant.finalize().ok());
}

reference::Config sender_config() {
    return {{shuffled_ids.begin(), shuffled_ids.end()},
            {incoming_x.begin(), incoming_x.end()},
            {incoming_q.begin(), incoming_q.end()},
            2.0};
}

void advance_window(Simulation& simulation) {
    ASSERT_TRUE(simulation.begin_window(step).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
}

void check_sent(const TransportView<real>& payload, TrialField field,
                const std::array<real, 3>& values, double time, std::uint64_t attempt) {
    ASSERT_TRUE(payload);
    ASSERT_EQ(payload.purpose(), TransportPurpose::coupling);
    ASSERT_EQ(payload.descriptor().id, input_descriptor(field).id);
    ASSERT_EQ(payload.stamp().time.physical_time, time);
    ASSERT_EQ(payload.stamp().trial.attempt, attempt);
    ASSERT_EQ(payload.stable_ids().extent(0), shuffled_ids.size());
    ASSERT_EQ(payload.values().extent(0), values.size());
    ASSERT_EQ(payload.values().extent(1), 1);
    for (std::size_t row = 0; row < values.size(); ++row) {
        ASSERT_EQ(payload.stable_ids()(row), shuffled_ids[row]);
        ASSERT_EQ(payload.values()(row, 0), values[row]);
    }
}

TEST(ReferenceTransport, TwoReferenceCommitAndSenderSnapshotLifetime) {
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
        ASSERT_TRUE(sender_simulation.initialize().ok());
        ASSERT_TRUE(sender_simulation.prepare().ok());
        ASSERT_TRUE(receiver_simulation.initialize().ok());
        ASSERT_TRUE(receiver_simulation.prepare().ok());
        const auto receiver_original = take(receiver.snapshot());

        ASSERT_TRUE(sender_simulation.begin_window(step).ok());
        ASSERT_EQ(sender.send_trial(TrialField::x).status().code(), StatusCode::invalid_state);
        ASSERT_TRUE(sender_simulation.advance_trial().ok());
        ASSERT_NO_FATAL_FAILURE(advance_window(receiver_simulation));
        auto x = take(sender.send_trial(TrialField::x));
        auto q = take(sender.send_trial(TrialField::q));
        ASSERT_NO_FATAL_FAILURE(check_sent(x, TrialField::x, sender_x, step, 1));
        ASSERT_NO_FATAL_FAILURE(check_sent(q, TrialField::q, sender_q, step, 1));

        ASSERT_TRUE(receiver.receive_trial(TrialField::x, x, x.stamp()).ok());
        ASSERT_TRUE(receiver.receive_trial(TrialField::q, q, q.stamp()).ok());
        ASSERT_NO_FATAL_FAILURE(check_original(receiver, receiver_original));
        ASSERT_NO_FATAL_FAILURE(check_snapshot(take(sender.snapshot()), incoming_x, incoming_q, 0.0,
                                               0, 0, shuffled_ids));

        committed_payload.emplace(std::move(x));
        ASSERT_FALSE(x);
        ASSERT_TRUE(sender_simulation.commit_window().ok());
        ASSERT_EQ(sender.send_trial(TrialField::x).status().code(), StatusCode::invalid_state);
        ASSERT_NO_FATAL_FAILURE(check_sent(*committed_payload, TrialField::x, sender_x, step, 1));
        ASSERT_NO_FATAL_FAILURE(check_sent(q, TrialField::q, sender_q, step, 1));
        ASSERT_NO_FATAL_FAILURE(check_original(receiver, receiver_original));
        ASSERT_TRUE(receiver_simulation.commit_window().ok());
        ASSERT_NO_FATAL_FAILURE(
            check_snapshot(take(sender.snapshot()), sender_x, sender_q, step, 1, 1, shuffled_ids));
        ASSERT_NO_FATAL_FAILURE(check_snapshot(take(receiver.snapshot()), {3.0, -1.5, 6.0},
                                               {2.0, 5.5, -6.0}, step, 1, 1));

        ASSERT_NO_FATAL_FAILURE(advance_window(sender_simulation));
        auto later = take(sender.send_trial(TrialField::x));
        q = std::move(later);
        ASSERT_FALSE(later);
        ASSERT_NO_FATAL_FAILURE(check_sent(q, TrialField::x, later_x, 2 * step, 2));
        rolled_back_payload.emplace(std::move(q));
        ASSERT_FALSE(q);
        ASSERT_TRUE(sender_simulation.restore_window().ok());
        ASSERT_NO_FATAL_FAILURE(
            check_snapshot(take(sender.snapshot()), sender_x, sender_q, step, 1, 1, shuffled_ids));
        ASSERT_TRUE(sender_simulation.finalize().ok());
        ASSERT_TRUE(receiver_simulation.finalize().ok());
        ASSERT_EQ(sender.send_trial(TrialField::x).status().code(), StatusCode::invalid_state);
    }

    ASSERT_NO_FATAL_FAILURE(check_sent(*committed_payload, TrialField::x, sender_x, step, 1));
    ASSERT_NO_FATAL_FAILURE(check_sent(*rolled_back_payload, TrialField::x, later_x, 2 * step, 2));
    committed_payload->complete();
    rolled_back_payload->complete();
    ASSERT_FALSE(*committed_payload);
    ASSERT_FALSE(*rolled_back_payload);
}

TEST(ReferenceTransport, TwoReferenceReceiveFailureAndRetry) {
    Participant sender(sender_config(), ParticipantId{1});
    Participant receiver(config(), ParticipantId{2});
    Simulation sender_simulation(sender);
    Simulation receiver_simulation(receiver);
    ASSERT_TRUE(sender_simulation.initialize().ok());
    ASSERT_TRUE(sender_simulation.prepare().ok());
    ASSERT_TRUE(receiver_simulation.initialize().ok());
    ASSERT_TRUE(receiver_simulation.prepare().ok());
    const auto receiver_original = take(receiver.snapshot());

    auto rollback_both = [&] {
        ASSERT_EQ(receiver_simulation.commit_window().code(), StatusCode::invalid_state);
        ASSERT_TRUE(sender_simulation.restore_window().ok());
        ASSERT_NO_FATAL_FAILURE(check_original(receiver, receiver_original));
        ASSERT_NO_FATAL_FAILURE(check_snapshot(take(sender.snapshot()), incoming_x, incoming_q, 0.0,
                                               0, 0, shuffled_ids));
        for (const auto* simulation : {&sender_simulation, &receiver_simulation}) {
            ASSERT_EQ(simulation->state(), SimulationState::prepared);
            ASSERT_EQ(simulation->time().accepted_time, 0.0);
            ASSERT_EQ(simulation->time().accepted_steps, 0);
        }
    };

    ASSERT_NO_FATAL_FAILURE(advance_window(sender_simulation));
    ASSERT_NO_FATAL_FAILURE(advance_window(receiver_simulation));
    const auto first = take(sender.send_trial(TrialField::x));
    ASSERT_TRUE(receiver.receive_trial(TrialField::x, first, first.stamp()).ok());
    ASSERT_EQ(receiver.receive_trial(TrialField::q, first, first.stamp()).code(),
              StatusCode::invalid_argument);
    ASSERT_NO_FATAL_FAILURE(rollback_both());

    ASSERT_NO_FATAL_FAILURE(advance_window(sender_simulation));
    ASSERT_NO_FATAL_FAILURE(advance_window(receiver_simulation));
    const auto retry_stamp = take(receiver.trial_input_stamp());
    ASSERT_EQ(retry_stamp.time, first.stamp().time);
    ASSERT_GT(retry_stamp.trial.attempt, first.stamp().trial.attempt);
    ASSERT_EQ(receiver.receive_trial(TrialField::x, first, first.stamp()).code(),
              StatusCode::invalid_state);
    ASSERT_NO_FATAL_FAILURE(rollback_both());

    ASSERT_NO_FATAL_FAILURE(advance_window(sender_simulation));
    ASSERT_NO_FATAL_FAILURE(advance_window(receiver_simulation));
    const auto current = take(sender.send_trial(TrialField::x));
    ASSERT_EQ(current.stamp().trial.attempt, 3);
    ASSERT_TRUE(receiver.receive_trial(TrialField::x, current, current.stamp()).ok());
    ASSERT_TRUE(sender_simulation.commit_window().ok());
    ASSERT_TRUE(receiver_simulation.commit_window().ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(take(sender.snapshot()), {6.0, 3.0, -1.5},
                                           {-6.0, 2.0, 5.5}, step, 1, 3, shuffled_ids));
    ASSERT_NO_FATAL_FAILURE(
        check_snapshot(take(receiver.snapshot()), {3.0, -1.5, 6.0}, {0.25, 1.5, 4.0}, step, 1, 3));
    for (const auto* simulation : {&sender_simulation, &receiver_simulation}) {
        ASSERT_EQ(simulation->time().accepted_time, step);
        ASSERT_EQ(simulation->time().accepted_steps, 1);
        ASSERT_EQ(simulation->time().trial_attempts, 3);
    }
    ASSERT_TRUE(sender_simulation.finalize().ok());
    ASSERT_TRUE(receiver_simulation.finalize().ok());
}
} // namespace
