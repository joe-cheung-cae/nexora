#include "wcsph_oracle.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <methods/sph/Pressure3D.hpp>

namespace {
using namespace nexora;
using namespace sph_test;
// Observe actual density/pressure operator output through the same independent
// all-pairs oracle used for participant snapshots. No participant window is used.
struct PressureEvaluation {
    sph::CellGrid grid;
    sph::Values density_values, pressure_values;
    sph::Acceleration3D acceleration_values;
    std::size_t row_count() const { return grid.owned_slots().extent(0); }
    std::uint64_t row_id(std::size_t row) const {
        return grid.fields().ids(grid.owned_slots()(row));
    }
    real density(std::size_t row) const { return density_values(row); }
    Result<real> pressure(std::size_t row) const { return pressure_values(row); }
    Result<std::array<real, 3>> pressure_acceleration(std::size_t row) const {
        return std::array<real, 3>{acceleration_values(row, 0), acceleration_values(row, 1),
                                   acceleration_values(row, 2)};
    }
    template <class Visitor> void for_each_neighbor(std::size_t row, Visitor&& visit) const {
        grid.for_each_neighbor(row, [&](LocalIndex slot, double d2) {
            visit(grid.fields().ids(slot), d2);
        });
    }
};
PressureEvaluation evaluate_pressure(const std::vector<Particle>& input,
                                     sph::WcsphParameters parameters,
                                     const PressureEvaluation* previous = nullptr) {
    Kokkos::Serial exec;
    Fields fields(input);
    auto grid = checked(previous ? previous->grid.update(exec, fields.get(), 1000)
                                 : sph::CellGrid::build(exec, fields.get(), 1000));
    auto density = checked(sph::evaluate_density_3d(exec, grid));
    sph::Values pressure("test.pressure", density.extent(0));
    sph::Acceleration3D acceleration("test.acceleration", density.extent(0));
    const auto status = sph::evaluate_pressure_3d(exec, grid, sph::TaitEos(parameters), density, pressure,
                                                 acceleration);
    if (!status.ok())
        throw std::runtime_error(status.message());
    return {std::move(grid), std::move(density), std::move(pressure), std::move(acceleration)};
}
TEST(SphWcsphOperators, SourceOutsideOwnedSupportChangesGhostPressureForceOnly) {
    const sph::WcsphParameters parameters{1, 2, 3};
    const std::vector<Particle> initial{
        {{0, 0, 0}, 1, 10, 0}, {{1.25, 0, 0}, 2, 20, 1}, {{2.5, 0, 0}, 1, 30, 1}};
    const auto before = evaluate_pressure(initial, parameters);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(before, initial, parameters));
    auto changed = initial;
    changed.back().mass = 8;
    const auto after = evaluate_pressure(changed, parameters, &before);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(after, changed, parameters));
    EXPECT_EQ(before.density(0), after.density(0));
    EXPECT_EQ(checked(before.pressure(0)), checked(after.pressure(0)));
    EXPECT_NE(checked(before.pressure_acceleration(0))[0],
              checked(after.pressure_acceleration(0))[0]);
    // This ordinary ghost reconstructs density from its complete same-stage
    // local support. It is independent of continuity integration and Adami walls.
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(before, initial, parameters));
}
TEST(SphWcsphOperators, LinearEquationOfStateResolvesBothSidesOfRestDensity) {
    const std::vector<Particle> state{{{0, 0, 0}, 1, 1, 0}};
    const double rho = evaluate_pressure(state, {1, 2, 1}).density(0);
    for (const double rho0 : {rho, std::nextafter(rho, 0.), std::nextafter(rho, 1.)}) {
        const auto result = evaluate_pressure(state, {rho0, 2, 1});
        const double expected = 4 * (rho - rho0);
        EXPECT_NEAR(checked(result.pressure(0)), expected,
                    8 * std::numeric_limits<real>::epsilon() * std::abs(expected));
        EXPECT_EQ(checked(result.pressure_acceleration(0)), (std::array<real, 3>{}));
    }
}
TEST(SphWcsphOperators, ThreeAxisCrossingAndReorderedInputMatchCompleteCsr) {
    const sph::WcsphParameters parameters{real{.5}, real{2}, real{3}};
    const std::vector<Particle> initial{{{.25, .25, .25}, 1, 91, 0},
                                        {{2.5, .75, .5}, 2, 12, 0},
                                        {{-.5, .25, .25}, 3, 77, 1},
                                        {{3.5, 2.5, 2.5}, 4, 63, 1},
                                        {{.75, 1.25, .75}, 1, 44, 0}};
    const auto first = evaluate_pressure(initial, parameters);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(first, initial, parameters));
    auto same = initial;
    same[0].position[0] = real{1.75};
    same[1].mass = 5;
    const auto within = evaluate_pressure(same, parameters, &first);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(within, same, parameters));
    auto crossing = same;
    crossing[0].position = {real{2.25}, real{2.25}, real{2.25}};
    crossing[3].position = {real{1.5}, real{1.5}, real{1.5}};
    std::reverse(crossing.begin(), crossing.end());
    const auto moved = evaluate_pressure(crossing, parameters, &within);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(moved, crossing, parameters));
    // Rebuilding from the same previous geometry gives the same ID-bound fields.
    const auto replay = evaluate_pressure(crossing, parameters, &within);
    ASSERT_EQ(replay.row_count(), moved.row_count());
    for (std::size_t row = 0; row < replay.row_count(); ++row) {
        EXPECT_EQ(replay.row_id(row), moved.row_id(row));
        EXPECT_EQ(replay.density(row), moved.density(row));
        EXPECT_EQ(checked(replay.pressure(row)), checked(moved.pressure(row)));
        EXPECT_EQ(checked(replay.pressure_acceleration(row)),
                  checked(moved.pressure_acceleration(row)));
    }
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(first, initial, parameters));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(within, same, parameters));
}
TEST(SphWcsphOperators, ChangingOwnedSetPublishesPressureRowsByStableId) {
    const sph::WcsphParameters parameters{1.5, 3, 7};
    std::vector<Particle> initial;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                const auto id = static_cast<std::uint64_t>(initial.size() + 1);
                initial.push_back({{real(i - 1) * real{.75}, real(j - 1) * real{.875}, real(k - 1)},
                                   real(1 + id % 4), id, std::uint8_t(id % 3 == 0)});
            }
    const auto before = evaluate_pressure(initial, parameters);
    ASSERT_EQ(before.row_count(), 18u);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(before, initial, parameters));
    auto changed = initial;
    for (auto& particle : changed)
        particle.ghost = !particle.ghost;
    changed.erase(changed.begin());
    changed.push_back({{-.5, .25, .125}, 2, 100, 0});
    std::reverse(changed.begin(), changed.end());
    const auto after = evaluate_pressure(changed, parameters, &before);
    ASSERT_EQ(after.row_count(), 10u);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(after, changed, parameters));
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(before, initial, parameters));
}
TEST(SphWcsphOperators, UnequalMassPairAnalyticDirectionAndForceBalanceIncludingTension) {
    const long double pi = std::acos(-1.0L);
    const std::vector<Particle> pair{{{0, 0, 0}, 1, 11, 0}, {{1, 0, 0}, 2, 22, 0}};
    for (const real rho0 : {real{.125}, real{2}}) {
        const sph::WcsphParameters parameters{rho0, real{2}, real{3}};
        const auto snapshot = evaluate_pressure(pair, parameters);
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
    }
}
TEST(SphWcsphOperators, IsolatedCoincidentAndCutoffHaveZeroPressureGradient) {
    const sph::WcsphParameters parameters{real{1}, real{2}, real{3}};
    const std::vector<Particle> state{
        {{0, 0, 0}, 1, 1, 0}, {{0, 0, 0}, 2, 2, 0}, {{2, 0, 0}, 3, 3, 0}, {{8, 0, 0}, 1, 4, 0}};
    const auto result = evaluate_pressure(state, parameters);
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(result, state, parameters));
    for (std::size_t row = 0; row < result.row_count(); ++row)
        for (const real value : checked(result.pressure_acceleration(row)))
            EXPECT_EQ(value, 0);
}
TEST(SphWcsphOperators, RejectsInvalidParameters) {
    Kokkos::Serial exec;
    Fields fields({{{0, 0, 0}, 1, 1, 0}, {{1, 0, 0}, 2, 2, 0}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 100));
    const auto density = checked(sph::evaluate_density_3d(exec, grid));
    sph::Values pressure("test.pressure", 2);
    sph::Acceleration3D acceleration("test.acceleration", 2);
    for (const auto parameters : {sph::WcsphParameters{0, 1, 3}, sph::WcsphParameters{1, 0, 3},
                                  sph::WcsphParameters{1, 1, 0}}) {
        EXPECT_EQ(parameters.validate().code(), StatusCode::invalid_argument);
        EXPECT_EQ(sph::evaluate_pressure_3d(exec, grid, sph::TaitEos(parameters), density, pressure, acceleration)
                      .code(), StatusCode::invalid_argument);
    }
}
TEST(SphWcsphOperators, EmptyAndGhostOnlyStatesHaveNoPressureRows) {
    const sph::WcsphParameters parameters{1, 2, 7};
    for (const auto& state :
         {std::vector<Particle>{}, std::vector<Particle>{{{-.25, .5, -.75}, 1, 23, 1}}}) {
        const auto result = evaluate_pressure(state, parameters);
        EXPECT_EQ(result.row_count(), 0U);
        ASSERT_NO_FATAL_FAILURE(check_force_snapshot(result, state, parameters));
    }
}
} // namespace
