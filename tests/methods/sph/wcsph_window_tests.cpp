#include "wcsph_oracle.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <nexora/simulation/Simulation.hpp>
#include <participants/sph/Participant3D.hpp>

namespace {
using namespace nexora;
using namespace sph_test;
TEST(SphWcsphWindow, SourceOutsideOwnedSupportChangesGhostPressureForceOnly) {
    const sph::WcsphParameters parameters{1, 2, 3};
    const std::vector<Particle> initial{
        {{0, 0, 0}, 1, 10, 0}, {{1.25, 0, 0}, 2, 20, 1}, {{2.5, 0, 0}, 1, 30, 1}};
    Fields fields(initial);
    sph::Participant3D participant(fields.get(), 100, parameters);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto accepted = checked(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(accepted, initial, parameters));
    auto changed = initial;
    changed.back().mass = 8;
    Fields staged(changed);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(participant.set_trial_fields(staged.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto trial = checked(participant.trial_snapshot(checked(participant.trial_stamp())));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(trial, changed, parameters));
    EXPECT_EQ(accepted.density(0), trial.density(0));
    EXPECT_EQ(checked(accepted.pressure(0)), checked(trial.pressure(0)));
    EXPECT_NE(checked(accepted.pressure_acceleration(0))[0],
              checked(trial.pressure_acceleration(0))[0]);
    ASSERT_TRUE(simulation.restore_window().ok());
    ASSERT_NO_FATAL_FAILURE(
        check_force_snapshot(checked(participant.snapshot()), initial, parameters));
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphWcsphWindow, LinearEquationOfStateResolvesBothSidesOfRestDensity) {
    Fields fields({{{0, 0, 0}, 1, 1, 0}});
    sph::Participant3D density_only(fields.get(), 1);
    Simulation baseline(density_only);
    ASSERT_TRUE(baseline.initialize().ok());
    const double rho = checked(density_only.snapshot()).density(0);
    ASSERT_TRUE(baseline.finalize().ok());
    for (const double rho0 : {rho, std::nextafter(rho, 0.), std::nextafter(rho, 1.)}) {
        sph::Participant3D particle(fields.get(), 1, sph::WcsphParameters{rho0, 2, 1});
        Simulation simulation(particle);
        ASSERT_TRUE(simulation.initialize().ok());
        const auto snapshot = checked(particle.snapshot());
        const double expected = 4 * (rho - rho0);
        EXPECT_NEAR(checked(snapshot.pressure(0)), expected,
                    8 * std::numeric_limits<real>::epsilon() * std::abs(expected));
        EXPECT_EQ(checked(snapshot.pressure_acceleration(0)), (std::array<real, 3>{}));
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphWcsphWindow, FinitePairAccelerationSurvivesExtremeDensityScaling) {
    const int magnitude = std::numeric_limits<real>::max_exponent / 2 + 32;
    for (const int exponent : {-magnitude, magnitude}) {
        const real mass = std::ldexp(real{1}, exponent);
        const sph::WcsphParameters parameters{exponent < 0 ? 1. : double(mass),
                                              exponent < 0 ? 1. : std::ldexp(1., -exponent / 2), 1};
        Fields fields({{{0, 0, 0}, mass, 11, 0}, {{1, 0, 0}, mass, 22, 0}});
        sph::Participant3D participant(fields.get(), 1, parameters);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        const auto snapshot = checked(participant.snapshot());
        ASSERT_EQ(snapshot.row_count(), 2u);
        for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
            const double rho = snapshot.density(row);
            const double p = (rho - parameters.reference_density) * parameters.sound_speed *
                             parameters.sound_speed;
            const double direction = snapshot.row_id(row) == 11 ? -1 : 1;
            const double expected = direction * (1.5 / std::acos(-1.)) * (mass / rho) * (p / rho);
            ASSERT_NE(expected, 0);
            ASSERT_TRUE(std::isfinite(expected));
            const auto a = checked(snapshot.pressure_acceleration(row));
            EXPECT_NEAR(a[0], expected,
                        64 * std::numeric_limits<real>::epsilon() * std::abs(expected));
            EXPECT_EQ(a[1], 0);
            EXPECT_EQ(a[2], 0);
        }
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphWcsphWindow, StagedThreeAxisCrossingCommitRestoreAndRetryMatchesCompleteCsr) {
    const sph::WcsphParameters parameters{real{.5}, real{2}, real{3}};
    const std::vector<Particle> initial{{{.25, .25, .25}, 1, 91, 0},
                                        {{2.5, .75, .5}, 2, 12, 0},
                                        {{-.5, .25, .25}, 3, 77, 1},
                                        {{3.5, 2.5, 2.5}, 4, 63, 1},
                                        {{.75, 1.25, .75}, 1, 44, 0}};
    Fields fields(initial);
    sph::Participant3D participant(fields.get(), 1000, parameters);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto first = checked(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(first, initial, parameters));
    auto same = initial;
    same[0].position[0] = real{1.75};
    same[1].mass = 5;
    Fields staged(same);
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    ASSERT_TRUE(participant.set_trial_fields(staged.get()).ok());
    staged.mass(0) = 99;
    staged.position(0, 0) = 99;
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto stamp = checked(participant.trial_stamp());
    const auto held_trial = checked(participant.trial_snapshot(stamp));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(held_trial, same, parameters));
    ASSERT_TRUE(simulation.commit_window().ok());
    const auto committed = checked(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(committed, same, parameters));
    auto crossing = same;
    crossing[0].position = {real{2.25}, real{2.25}, real{2.25}};
    crossing[3].position = {real{1.5}, real{1.5}, real{1.5}};
    std::reverse(crossing.begin(), crossing.end());
    Fields moved(crossing);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(participant.set_trial_fields(moved.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto discarded_stamp = checked(participant.trial_stamp());
    const auto discarded = checked(participant.trial_snapshot(discarded_stamp));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(discarded, crossing, parameters));
    ASSERT_TRUE(simulation.restore_window().ok());
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .125);
    EXPECT_FALSE(participant.trial_snapshot(discarded_stamp).ok());
    ASSERT_NO_FATAL_FAILURE(
        check_force_snapshot(checked(participant.snapshot()), same, parameters));
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    ASSERT_TRUE(participant.set_trial_fields(moved.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto retry_stamp = checked(participant.trial_stamp());
    EXPECT_NE(retry_stamp.trial.attempt, discarded_stamp.trial.attempt);
    ASSERT_TRUE(simulation.commit_window().ok());
    const auto latest = checked(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(latest, crossing, parameters));
    ASSERT_EQ(latest.row_count(), discarded.row_count());
    for (std::size_t row = 0; row < latest.row_count(); ++row) {
        EXPECT_EQ(latest.row_id(row), discarded.row_id(row));
        EXPECT_EQ(latest.density(row), discarded.density(row));
        EXPECT_EQ(checked(latest.pressure(row)), checked(discarded.pressure(row)));
        EXPECT_EQ(checked(latest.pressure_acceleration(row)),
                  checked(discarded.pressure_acceleration(row)));
    }
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(first, initial, parameters));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(held_trial, same, parameters));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(committed, same, parameters));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(discarded, crossing, parameters));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(latest, crossing, parameters));
}
TEST(SphWcsphWindow, ChangingOwnedSetPublishesPressureRowsByStableId) {
    const sph::WcsphParameters parameters{1.5, 3, 7};
    std::vector<Particle> initial;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                const auto id = static_cast<std::uint64_t>(initial.size() + 1);
                initial.push_back({{real(i - 1) * real{.75}, real(j - 1) * real{.875}, real(k - 1)},
                                   real(1 + id % 4),
                                   id,
                                   std::uint8_t(id % 3 == 0)});
            }
    Fields fields(initial);
    sph::Participant3D participant(fields.get(), 100, parameters);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto accepted = checked(participant.snapshot());
    ASSERT_EQ(accepted.row_count(), 18u);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(accepted, initial, parameters));

    auto changed = initial;
    for (auto& particle : changed)
        particle.ghost = !particle.ghost;
    changed.erase(changed.begin());
    changed.push_back({{-.5, .25, .125}, 2, 100, 0});
    std::reverse(changed.begin(), changed.end());
    Fields staged(changed);
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    ASSERT_TRUE(participant.set_trial_fields(staged.get()).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto trial = checked(participant.trial_snapshot(checked(participant.trial_stamp())));
    ASSERT_EQ(trial.row_count(), 10u);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(trial, changed, parameters));
    ASSERT_TRUE(simulation.commit_window().ok());
    ASSERT_NO_FATAL_FAILURE(
        check_force_snapshot(checked(participant.snapshot()), changed, parameters));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(accepted, initial, parameters));
}

TEST(SphWcsphWindow, UnequalMassPairAnalyticDirectionAndForceBalanceIncludingTension) {
    const long double pi = std::acos(-1.0L);
    const std::vector<Particle> pair{{{0, 0, 0}, 1, 11, 0}, {{1, 0, 0}, 2, 22, 0}};
    for (const real rho0 : {real{.125}, real{2}}) {
        const sph::WcsphParameters parameters{rho0, real{2}, real{3}};
        Fields fields(pair);
        sph::Participant3D participant(fields.get(), 100, parameters);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        const auto snapshot = checked(participant.snapshot());
        ASSERT_NO_FATAL_FAILURE(check_force_snapshot(snapshot, pair, parameters));
        std::map<std::uint64_t, std::array<real, 3>> acceleration;
        std::map<std::uint64_t, real> pressure;
        for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
            acceleration.emplace(snapshot.row_id(row),
                                 checked(snapshot.pressure_acceleration(row)));
            pressure.emplace(snapshot.row_id(row), checked(snapshot.pressure(row)));
        }
        const real rho_a = static_cast<real>(1.5L / pi), rho_b = static_cast<real>(2.25L / pi);
        const long double coefficient =
            pressure.at(11) / (static_cast<long double>(rho_a) * rho_a) +
            pressure.at(22) / (static_cast<long double>(rho_b) * rho_b);
        const long double expected_a = -2 * coefficient * 3 / (4 * pi);
        EXPECT_LE(std::abs(static_cast<long double>(acceleration.at(11)[0]) - expected_a),
                  128 * std::numeric_limits<real>::epsilon() * std::abs(expected_a));
        EXPECT_EQ(acceleration.at(11)[1], 0);
        EXPECT_EQ(acceleration.at(11)[2], 0);
        EXPECT_EQ(acceleration.at(22)[1], 0);
        EXPECT_EQ(acceleration.at(22)[2], 0);
        const long double imbalance = acceleration.at(11)[0] + 2.L * acceleration.at(22)[0];
        EXPECT_LE(std::abs(imbalance),
                  8 * std::numeric_limits<real>::epsilon() *
                      (std::abs(acceleration.at(11)[0]) + 2.L * std::abs(acceleration.at(22)[0])));
        if (rho0 < real{1}) {
            EXPECT_GT(pressure.at(11), 0);
            EXPECT_LT(acceleration.at(11)[0], 0);
        } else {
            EXPECT_LT(pressure.at(11), 0);
            EXPECT_GT(acceleration.at(11)[0], 0);
        }
        ASSERT_TRUE(simulation.finalize().ok());
    }
}
TEST(SphWcsphWindow, IsolatedCoincidentAndCutoffHaveZeroPressureGradient) {
    const sph::WcsphParameters parameters{real{1}, real{2}, real{3}};
    const std::vector<Particle> state{
        {{0, 0, 0}, 1, 1, 0}, {{0, 0, 0}, 2, 2, 0}, {{2, 0, 0}, 3, 3, 0}, {{8, 0, 0}, 1, 4, 0}};
    Fields fields(state);
    sph::Participant3D participant(fields.get(), 1000, parameters);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto snapshot = checked(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(snapshot, state, parameters));
    for (std::size_t row = 0; row < snapshot.row_count(); ++row)
        for (const real value : checked(snapshot.pressure_acceleration(row)))
            EXPECT_EQ(value, 0);
    ASSERT_TRUE(simulation.finalize().ok());
}
TEST(SphWcsphWindow, InvalidParametersAndFinitePressureOverflowRestoreAccepted) {
    const std::vector<Particle> state{{{0, 0, 0}, 1, 1, 0}, {{1, 0, 0}, 2, 2, 0}};
    Fields fields(state);
    for (const auto parameters : {sph::WcsphParameters{0, 1, 3}, sph::WcsphParameters{1, 0, 3},
                                  sph::WcsphParameters{1, 1, 0}}) {
        sph::Participant3D participant(fields.get(), 100, parameters);
        Simulation simulation(participant);
        EXPECT_FALSE(simulation.initialize().ok());
    }
    const sph::WcsphParameters parameters{real{1}, real{2}, real{3}};
    sph::Participant3D participant(fields.get(), 100, parameters);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto held = checked(participant.snapshot());
    auto huge = state;
    for (auto& particle : huge)
        particle.mass = std::sqrt(std::numeric_limits<real>::max());
    Fields overflowing(huge);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(participant.set_trial_fields(overflowing.get()).ok());
    EXPECT_FALSE(simulation.advance_trial().ok());
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    ASSERT_NO_FATAL_FAILURE(
        check_force_snapshot(checked(participant.snapshot()), state, parameters));
    // Positive representable densities and finite tensile pressure can still
    // produce an acceleration outside real range; this failure must also restore.
    auto tiny = state;
    for (auto& particle : tiny)
        particle.mass = std::numeric_limits<real>::min() / real{32};
    Fields force_overflow(tiny);
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    ASSERT_TRUE(participant.set_trial_fields(force_overflow.get()).ok());
    EXPECT_FALSE(simulation.advance_trial().ok());
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    ASSERT_NO_FATAL_FAILURE(
        check_force_snapshot(checked(participant.snapshot()), state, parameters));
    ASSERT_TRUE(simulation.advance(.125).ok());
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(held, state, parameters));
    sph::Participant3D density_only(fields.get(), 100);
    Simulation density_sim(density_only);
    ASSERT_TRUE(density_sim.initialize().ok());
    ASSERT_TRUE(density_sim.prepare().ok());
    const auto density_snapshot = checked(density_only.snapshot());
    EXPECT_FALSE(density_snapshot.pressure(0).ok());
    EXPECT_FALSE(density_snapshot.pressure_acceleration(0).ok());
    ASSERT_TRUE(density_sim.finalize().ok());
}
TEST(SphWcsphWindow, RetainedTrialBudgetRefusalRestoresAndRetryWithoutReaderSucceeds) {
    const std::vector<Particle> state{{{.25, -.5, .75}, 1, 17, 0}};
    const sph::WcsphParameters parameters{1, 2, 7};
    Fields fields(state);
    std::size_t accepted_bytes = 0;
    {
        sph::Participant3D baseline(fields.get(), 100, parameters);
        ASSERT_TRUE(baseline.initialize(0).ok());
        accepted_bytes = baseline.resource_usage().controlled_bytes;
        ASSERT_TRUE(baseline.finalize().ok());
    }
    // Admit one trial field bank plus method workspace, but not the extra bank
    // required to publish while a caller retains the old trial allocation.
    const std::size_t bank_bytes = 5 * sizeof(real);
    const std::size_t workspace_bytes = sph::pressure_workspace_bytes_per_source(false);
    ASSERT_LT(workspace_bytes, bank_bytes);
    sph::Participant3D participant(fields.get(), 100, parameters, ParticipantId{1},
                                   {32, accepted_bytes + bank_bytes + workspace_bytes});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto stamp = checked(participant.trial_stamp());
    {
        const auto held_trial = checked(participant.trial_snapshot(stamp));
        ASSERT_NO_FATAL_FAILURE(check_force_snapshot(held_trial, state, parameters));
        const auto status = simulation.commit_window();
        EXPECT_EQ(status.code(), StatusCode::resource_exhausted);
        EXPECT_EQ(simulation.state(), SimulationState::prepared);
        EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
        EXPECT_FALSE(participant.trial_snapshot(stamp).ok());
        ASSERT_NO_FATAL_FAILURE(
            check_force_snapshot(checked(participant.snapshot()), state, parameters));
        ASSERT_NO_FATAL_FAILURE(check_force_snapshot(held_trial, state, parameters));
        EXPECT_GT(participant.resource_usage().controlled_bytes, accepted_bytes);
    }
    EXPECT_EQ(participant.resource_usage().controlled_bytes, accepted_bytes);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0U);
    ASSERT_TRUE(simulation.advance(.125).ok());
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .125);
    ASSERT_NO_FATAL_FAILURE(
        check_force_snapshot(checked(participant.snapshot()), state, parameters));
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0U);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0U);
}
TEST(SphWcsphWindow, ConfiguredEmptyAndGhostOnlySnapshotsRejectOutOfRangeObservation) {
    const sph::WcsphParameters parameters{1, 2, 7};
    for (const auto& state :
         {std::vector<Particle>{}, std::vector<Particle>{{{-.25, .5, -.75}, 1, 23, 1}}}) {
        Fields fields(state);
        sph::Participant3D participant(fields.get(), 100, parameters);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        const auto held = checked(participant.snapshot());
        EXPECT_EQ(held.row_count(), 0U);
        ASSERT_NO_FATAL_FAILURE(check_force_snapshot(held, state, parameters));
        EXPECT_EQ(held.pressure(0).status().code(), StatusCode::invalid_argument);
        EXPECT_EQ(held.pressure_acceleration(0).status().code(), StatusCode::invalid_argument);
        EXPECT_FALSE(held.pressure(std::numeric_limits<std::size_t>::max()).ok());
        EXPECT_FALSE(held.pressure_acceleration(std::numeric_limits<std::size_t>::max()).ok());
        ASSERT_TRUE(simulation.advance(.125).ok());
        ASSERT_NO_FATAL_FAILURE(
            check_force_snapshot(checked(participant.snapshot()), state, parameters));
        ASSERT_TRUE(simulation.finalize().ok());
        ASSERT_NO_FATAL_FAILURE(check_force_snapshot(held, state, parameters));
    }
    const std::vector<Particle> state{{{0, 0, 0}, 1, 1, 0}};
    Fields fields(state);
    sph::Participant3D participant(fields.get(), 100, parameters);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto held = checked(participant.snapshot());
    EXPECT_TRUE(held.pressure(0).ok());
    EXPECT_TRUE(held.pressure_acceleration(0).ok());
    EXPECT_EQ(held.pressure(held.row_count()).status().code(), StatusCode::invalid_argument);
    EXPECT_EQ(held.pressure_acceleration(held.row_count()).status().code(),
              StatusCode::invalid_argument);
    ASSERT_TRUE(simulation.finalize().ok());
}
} // namespace
