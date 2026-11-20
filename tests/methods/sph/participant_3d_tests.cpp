#include "cell_grid_oracle.hpp"
#include <gtest/gtest.h>
#include <nexora/simulation/Simulation.hpp>
#include <participants/sph/Participant3D.hpp>

#include <limits>
#include <optional>

namespace {
using namespace nexora;
using namespace sph_test;
std::vector<Particle> initial_particles() {
    return {{{.25, -.25, .5}, 1, (9ULL << 32) | 17, 0},
            {{2.5, -.25, .5}, 2, (1ULL << 32) | 17, 0},
            {{-.5, -.25, .5}, 3, (5ULL << 32) | 17, 1},
            {{3.75, -.25, .5}, 4, (3ULL << 32) | 17, 1}};
}
void check_snapshot(const sph::Snapshot3D& snapshot, const std::vector<Particle>& input) {
    const auto oracle = explicit_csr(input);
    ASSERT_EQ(snapshot.row_count(), oracle.density.size());
    std::vector<std::uint64_t> seen;
    for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
        const auto id = snapshot.row_id(row);
        seen.push_back(id);
        ASSERT_TRUE(oracle.row_by_id.contains(id));
        const auto expected_row = oracle.row_by_id.at(id);
        const std::vector<std::uint64_t> expected(oracle.ids.begin() + oracle.offsets[expected_row],
                                                  oracle.ids.begin() +
                                                      oracle.offsets[expected_row + 1]);
        std::vector<std::uint64_t> actual;
        snapshot.for_each_neighbor(row, [&](std::uint64_t neighbor, double d2) {
            EXPECT_NE(id, neighbor);
            EXPECT_GE(d2, 0);
            EXPECT_LE(d2, 4);
            actual.push_back(neighbor);
        });
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(std::adjacent_find(actual.begin(), actual.end()), actual.end());
        EXPECT_EQ(actual, expected);
        const auto density = snapshot.density(row);
        const long double reference = oracle.density.at(id);
        const long double relative = std::numeric_limits<real>::epsilon() +
                                     32 * static_cast<long double>(actual.size() + 1) *
                                         std::numeric_limits<double>::epsilon();
        EXPECT_TRUE(std::isfinite(density));
        EXPECT_LE(std::abs(static_cast<long double>(density) - reference), relative * reference);
    }
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
}
TEST(SphParticipant3D, SimulationCommitsStagedGeometryAndRetainsSnapshots) {
    const auto initial = initial_particles();
    Fields input(initial);
    sph::Participant3D participant(input.get(), 1000);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto first = checked(participant.snapshot());
    EXPECT_TRUE(first.stamp.accepted);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(first, initial));
    EXPECT_FALSE(participant.set_trial_fields(input.get()).ok());
    auto same_cell = initial;
    same_cell[0].position[0] = real{1.75};
    same_cell[1].mass = 5;
    Fields staged(same_cell);
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    ASSERT_TRUE(participant.set_trial_fields(staged.get()).ok());
    // Staging owns a snapshot; caller mutation must not alter the staged geometry.
    staged.position(0, 0) = 100;
    staged.mass(1) = 99;
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto trial_stamp = checked(participant.trial_stamp());
    const auto trial = checked(participant.trial_snapshot(trial_stamp));
    EXPECT_FALSE(trial.stamp.accepted);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(trial, same_cell));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), initial));
    ASSERT_TRUE(simulation.commit_window().ok());
    const auto accepted = checked(participant.snapshot());
    EXPECT_TRUE(accepted.stamp.accepted);
    EXPECT_DOUBLE_EQ(accepted.stamp.time.physical_time, .125);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(accepted, same_cell));
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .125);
    EXPECT_FALSE(participant.trial_snapshot(trial_stamp).ok());
    auto crossing = same_cell;
    std::swap(crossing[0].position, crossing[1].position);
    std::swap(crossing[2].position, crossing[3].position);
    std::reverse(crossing.begin(), crossing.end());
    Fields moved(crossing);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(participant.set_trial_fields(moved.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_TRUE(simulation.commit_window().ok());
    const auto latest = checked(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(latest, crossing));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(first, initial));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(trial, same_cell));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(accepted, same_cell));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(latest, crossing));
}
TEST(SphParticipant3D, ExplicitRestoreRetryAndForeignAttemptStamps) {
    const auto initial = initial_particles();
    Fields input(initial);
    sph::Participant3D participant(input.get(), 1000), other(input.get(), 1000);
    Simulation simulation(participant), foreign(other);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(foreign.initialize().ok());
    ASSERT_TRUE(foreign.prepare().ok());
    auto moved = initial;
    moved[0].position[0] = real{1.5};
    Fields staged(moved);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(participant.set_trial_fields(staged.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto stamp = checked(participant.trial_stamp());
    const auto retained = checked(participant.trial_snapshot(stamp));
    const auto restored_trial_input = moved;
    ASSERT_TRUE(foreign.begin_window(.25).ok());
    ASSERT_TRUE(other.set_trial_fields(staged.get()).ok());
    ASSERT_TRUE(foreign.advance_trial().ok());
    EXPECT_FALSE(other.trial_snapshot(stamp).ok());
    EXPECT_FALSE(participant.trial_snapshot(checked(other.trial_stamp())).ok());
    ASSERT_TRUE(simulation.restore_window().ok());
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    EXPECT_FALSE(participant.trial_snapshot(stamp).ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), initial));
    moved[1].mass = 7;
    Fields alternative(moved);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(participant.set_trial_fields(alternative.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    EXPECT_FALSE(participant.trial_snapshot(stamp).ok());
    ASSERT_TRUE(simulation.commit_window().ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), moved));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(retained, restored_trial_input));
    ASSERT_TRUE(foreign.restore_window().ok());
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_TRUE(foreign.finalize().ok());
}
TEST(SphParticipant3D, FailedGeometryAndDensityTrialsRestoreAcceptedAndAllowDifferentStep) {
    const auto initial = initial_particles();
    Fields input(initial);
    sph::Participant3D participant(input.get(), 1000);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto retained = checked(participant.snapshot());
    auto invalid = initial;
    invalid[0].position[0] = std::numeric_limits<real>::infinity();
    Fields invalid_fields(invalid);
    ASSERT_TRUE(simulation.begin_window(.5).ok());
    const auto staged = participant.set_trial_fields(invalid_fields.get());
    if (staged.ok()) {
        EXPECT_FALSE(simulation.advance_trial().ok());
        EXPECT_EQ(simulation.state(), SimulationState::prepared);
    } else {
        ASSERT_TRUE(simulation.restore_window().ok());
    }
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), initial));
    auto overflow = initial;
    for (auto& particle : overflow) {
        particle.position = {0, 0, 0};
        particle.mass = std::numeric_limits<real>::max();
    }
    Fields overflowing(overflow);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(participant.set_trial_fields(overflowing.get()).ok());
    EXPECT_FALSE(simulation.advance_trial().ok());
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), initial));
    auto retry = initial;
    retry[1].position[1] = real{.5};
    Fields legal(retry);
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    ASSERT_TRUE(participant.set_trial_fields(legal.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_TRUE(simulation.commit_window().ok());
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .125);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), retry));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(retained, initial));
}
TEST(SphParticipant3D, CommittedVersionsDescribeGeometryLayoutAndTopologySeparately) {
    auto state = initial_particles();
    Fields initial(state);
    sph::Participant3D participant(initial.get(), 1000);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    auto previous = checked(participant.snapshot());
    const auto commit = [&]() {
        Fields staged(state);
        if (!simulation.begin_window(.125).ok() ||
            !participant.set_trial_fields(staged.get()).ok() || !simulation.advance_trial().ok() ||
            !simulation.commit_window().ok())
            throw std::runtime_error("version test window failed");
        auto snapshot = checked(participant.snapshot());
        check_snapshot(snapshot, state);
        return snapshot;
    };
    state[0].position[0] += real{.125};
    auto current = commit();
    EXPECT_EQ(current.stamp.geometry.value(), previous.stamp.geometry.value() + 1);
    EXPECT_EQ(current.stamp.layout, previous.stamp.layout);
    EXPECT_EQ(current.stamp.topology, previous.stamp.topology);
    previous = current;
    state[0].mass = 7;
    current = commit();
    EXPECT_EQ(current.stamp.geometry, previous.stamp.geometry);
    EXPECT_EQ(current.stamp.layout, previous.stamp.layout);
    EXPECT_EQ(current.stamp.topology, previous.stamp.topology);
    previous = current;
    std::reverse(state.begin(), state.end());
    current = commit();
    EXPECT_EQ(current.stamp.layout.value(), previous.stamp.layout.value() + 1);
    EXPECT_EQ(current.stamp.geometry, previous.stamp.geometry);
    EXPECT_EQ(current.stamp.topology, previous.stamp.topology);
    previous = current;
    state[0].ghost = !state[0].ghost;
    current = commit();
    EXPECT_EQ(current.stamp.topology.value(), previous.stamp.topology.value() + 1);
    EXPECT_EQ(current.stamp.geometry, previous.stamp.geometry);
    previous = current;
    state[0].id += 123;
    current = commit();
    EXPECT_EQ(current.stamp.topology.value(), previous.stamp.topology.value() + 1);
    previous = current;
    state.push_back({{0, 0, 0}, 1, 98765, 0});
    current = commit();
    EXPECT_EQ(current.stamp.topology.value(), previous.stamp.topology.value() + 1);
    ASSERT_TRUE(simulation.finalize().ok());
}
TEST(SphParticipant3D, PrepareFailureResourceScopesRestoreAndRetryBothReaderBranches) {
    for (const bool retain_trial : {false, true}) {
        Fields fields(initial_particles());
        sph::Participant3D participant(fields.get(), 1000, ParticipantId{1}, {2, 1024 * 1024});
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        const auto baseline = participant.resource_usage();
        {
            auto first = checked(participant.snapshot());
            auto shared_copy = first;
            EXPECT_EQ(participant.resource_usage().snapshot_records, 1U);
            {
                auto second = checked(participant.snapshot());
                EXPECT_EQ(participant.resource_usage().snapshot_records, 2U);
                EXPECT_FALSE(participant.snapshot().ok());
            }
            EXPECT_EQ(participant.resource_usage().snapshot_records, 1U);
        }
        EXPECT_EQ(participant.resource_usage().snapshot_records, baseline.snapshot_records);
        ASSERT_TRUE(simulation.begin_window(.25).ok());
        ASSERT_TRUE(simulation.advance_trial().ok());
        const auto failed_stamp = checked(participant.trial_stamp());
        std::optional<sph::Snapshot3D> retained;
        if (retain_trial)
            retained.emplace(checked(participant.trial_snapshot(failed_stamp)));
        participant.fail_next(sph::Participant3D::Fault::prepare);
        EXPECT_FALSE(simulation.commit_window().ok());
        EXPECT_EQ(simulation.state(), SimulationState::prepared);
        EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
        EXPECT_FALSE(participant.trial_snapshot(failed_stamp).ok());
        ASSERT_NO_FATAL_FAILURE(
            check_snapshot(checked(participant.snapshot()), initial_particles()));
        if (retained) {
            ASSERT_NO_FATAL_FAILURE(check_snapshot(*retained, initial_particles()));
            EXPECT_GT(participant.resource_usage().controlled_bytes, baseline.controlled_bytes);
        }
        retained.reset();
        EXPECT_EQ(participant.resource_usage().controlled_bytes, baseline.controlled_bytes);
        EXPECT_EQ(participant.resource_usage().snapshot_records, baseline.snapshot_records);
        ASSERT_TRUE(simulation.advance(.125).ok());
        ASSERT_NO_FATAL_FAILURE(
            check_snapshot(checked(participant.snapshot()), initial_particles()));
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0U);
        EXPECT_EQ(participant.resource_usage().snapshot_records, 0U);
    }
}
TEST(SphParticipant3D, EmptyAndGhostOnlyStatesUseValidZeroRowFieldBindings) {
    for (const auto& state :
         {std::vector<Particle>{}, std::vector<Particle>{{{-.5, .5, 0}, 1, 42, 1}}}) {
        Fields fields(state);
        sph::Participant3D participant(fields.get(), 1000);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        auto snapshot = checked(participant.snapshot());
        EXPECT_EQ(snapshot.row_count(), 0U);
        ASSERT_NO_FATAL_FAILURE(check_snapshot(snapshot, state));
        ASSERT_TRUE(simulation.advance(.125).ok());
        ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), state));
        ASSERT_TRUE(simulation.begin_window(.25).ok());
        ASSERT_TRUE(participant.set_trial_fields(fields.get()).ok());
        ASSERT_TRUE(simulation.advance_trial().ok());
        ASSERT_TRUE(simulation.restore_window().ok());
        ASSERT_TRUE(simulation.finalize().ok());
        ASSERT_NO_FATAL_FAILURE(check_snapshot(snapshot, state));
    }
}
TEST(SphParticipant3D, ReplaysCrossCellTrialFromNoninitialAcceptedState) {
    std::vector<Particle> state;
    for (int i = 0; i < 27; ++i)
        state.push_back({{real(i % 3 - 1) * real{1.25}, real(i / 3 % 3 - 1) * real{1.25},
                          real(i / 9 - 1) * real{1.25}},
                         real(1 + i % 5),
                         (std::uint64_t(i) << 32) | 7,
                         static_cast<std::uint8_t>(i % 4 == 0)});
    Fields initial(state);
    sph::Participant3D participant(initial.get(), 64);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), state));

    // Establish a noninitial accepted state with a same-cell position change.
    state[1].position[1] += real{.375};
    state[5].mass += real{1};
    Fields within(state);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(participant.set_trial_fields(within.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_TRUE(simulation.commit_window().ok());
    const auto accepted = checked(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(accepted, state));

    // Exchange a ghost and an owned particle across all three cell axes.
    auto crossing = state;
    std::swap(crossing.front().position, crossing.back().position);
    std::reverse(crossing.begin(), crossing.end());
    Fields end(crossing);
    const auto try_crossing = [&] {
        EXPECT_TRUE(simulation.begin_window(.125).ok());
        EXPECT_TRUE(participant.set_trial_fields(end.get()).ok());
        EXPECT_TRUE(simulation.advance_trial().ok());
        return checked(participant.trial_snapshot(checked(participant.trial_stamp())));
    };
    const auto rolled_back = try_crossing();
    ASSERT_NO_FATAL_FAILURE(check_snapshot(rolled_back, crossing));
    ASSERT_TRUE(simulation.restore_window().ok());
    const auto restored = checked(participant.snapshot());
    EXPECT_EQ(restored.stamp, accepted.stamp);
    EXPECT_EQ(simulation.time().accepted_steps, 1U);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .25);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(restored, state));
    const auto replay = try_crossing();
    EXPECT_NE(replay.stamp.trial, rolled_back.stamp.trial);
    EXPECT_FALSE(participant.trial_snapshot(rolled_back.stamp).ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(replay, crossing));
    ASSERT_EQ(replay.row_count(), rolled_back.row_count());
    for (std::size_t row = 0; row < replay.row_count(); ++row) {
        EXPECT_EQ(replay.row_id(row), rolled_back.row_id(row));
        EXPECT_EQ(replay.density(row), rolled_back.density(row));
    }
    ASSERT_TRUE(simulation.commit_window().ok());
    EXPECT_EQ(simulation.time().accepted_steps, 2U);
    EXPECT_EQ(simulation.time().trial_attempts, 3U);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .375);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(checked(participant.snapshot()), crossing));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(accepted, state));
    ASSERT_TRUE(simulation.finalize().ok());
}
} // namespace
