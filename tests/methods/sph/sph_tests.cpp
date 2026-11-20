#include <Kokkos_Core.hpp>
#include <fixed_csr/Participant.hpp>
#include <fixed_csr/sph_fixture.hpp>
#include <gtest/gtest.h>
#include <nexora/simulation/Simulation.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace nexora;
namespace fixture = sph_fixture;
using fixture::Config;
using fixture::Participant;
using fixture::Snapshot;

static_assert(std::is_same_v<decltype(std::declval<const Snapshot&>().density(0)), real>);
static_assert(std::is_same_v<decltype(std::declval<const Snapshot&>().index(0)), LocalIndex>);

template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw std::runtime_error(result.status().message());
    return std::move(result).value();
}

void check_close(real actual, long double expected, long double epsilon_multiplier) {
    ASSERT_TRUE(std::isfinite(actual));
    const long double tolerance =
        epsilon_multiplier * std::numeric_limits<real>::epsilon() * std::abs(expected);
    const long double error = std::abs(static_cast<long double>(actual) - expected);
    ASSERT_LE(error, tolerance) << "density actual=" << actual << ", oracle=" << expected;
}

void check_frozen(const Snapshot& state, bool reordered) {
    const auto& rows = reordered ? fixture::reordered_rows : fixture::row_ids;
    const auto& columns = reordered ? fixture::reordered_columns : fixture::column_ids;
    const auto& offsets = reordered ? fixture::reordered_offsets : fixture::offsets;
    const auto& indices = reordered ? fixture::reordered_indices : fixture::indices;
    ASSERT_EQ(state.row_count(), rows.size());
    ASSERT_EQ(state.column_count(), columns.size());
    ASSERT_EQ(state.edge_count(), indices.size());
    for (std::size_t row = 0; row < rows.size(); ++row)
        ASSERT_EQ(state.row_id(row), rows[row]);
    for (std::size_t column = 0; column < columns.size(); ++column) {
        ASSERT_EQ(state.column_id(column), columns[column]);
        const auto original =
            std::find(fixture::column_ids.begin(), fixture::column_ids.end(), columns[column]) -
            fixture::column_ids.begin();
        ASSERT_EQ(state.ghost(column), fixture::ghost[original]);
        ASSERT_EQ(state.boundary(column), fixture::boundary[original]);
    }
    for (std::size_t row = 0; row < offsets.size(); ++row)
        ASSERT_EQ(state.offset(row), offsets[row]);
    for (std::size_t edge = 0; edge < indices.size(); ++edge)
        ASSERT_EQ(state.index(edge), indices[edge]);
    ASSERT_EQ(state.stamp.topology, TopologyVersion{1});
    ASSERT_EQ(state.stamp.layout, LayoutVersion{reordered ? 2u : 1u});
    ASSERT_NE(state.stamp.storage_incarnation, 0);
    ASSERT_TRUE(state.stamp.validate().ok());
    if (!state.density_valid)
        return;
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto original =
            std::find(fixture::row_ids.begin(), fixture::row_ids.end(), rows[row]);
        const auto numerator = fixture::density_numerators[original - fixture::row_ids.begin()];
        ASSERT_NO_FATAL_FAILURE(check_close(
            state.density(row), static_cast<long double>(numerator) / fixture::denominator,
            fixture::density_epsilon));
    }
    ASSERT_NO_FATAL_FAILURE(
        check_close(state.owned_density_sum,
                    static_cast<long double>(fixture::sum_numerator) / fixture::denominator,
                    fixture::sum_epsilon));
}

struct OracleRow {
    GlobalEntityId id;
    std::vector<GlobalEntityId> neighbors;
    long double density = 0;
};

std::size_t column_of(const Config& config, GlobalEntityId id) {
    const auto column = std::find(config.column_ids.begin(), config.column_ids.end(), id);
    if (column == config.column_ids.end()) {
        throw std::runtime_error("entity ID is missing from the oracle input columns");
    }
    return static_cast<std::size_t>(column - config.column_ids.begin());
}

// Reconstruct neighbors by scanning physical inputs, independently of the CSR.
// The truncated-power form is equivalent to the frozen piecewise formula.
std::vector<OracleRow> scalar_oracle(const Config& config) {
    const auto cube = [](long double x) { return x * x * x; };
    std::vector<OracleRow> result;
    for (const auto id : config.row_ids) {
        OracleRow row{id, {}};
        const long double position = config.positions[column_of(config, id)];
        for (std::size_t column = 0; column < config.column_ids.size(); ++column) {
            const long double q = std::abs(position - config.positions[column]);
            if (q <= 2 && config.column_ids[column] != id)
                row.neighbors.push_back(config.column_ids[column]);
            const long double weight =
                (cube(std::max(2 - q, 0.0L)) - 4 * cube(std::max(1 - q, 0.0L))) / 6;
            row.density += config.masses[column] * weight;
        }
        result.push_back(std::move(row));
    }
    return result;
}

void check_oracle(const Snapshot& state, const Config& config,
                  const std::vector<OracleRow>& oracle) {
    ASSERT_TRUE(state.density_valid);
    ASSERT_EQ(state.row_count(), oracle.size());
    ASSERT_EQ(state.column_count(), config.column_ids.size());
    ASSERT_EQ(state.offset(0), 0);
    ASSERT_EQ(state.offset(state.row_count()), state.edge_count());
    for (std::size_t column = 0; column < config.column_ids.size(); ++column) {
        const auto input_column = column_of(config, GlobalEntityId{state.column_id(column)});
        ASSERT_EQ(state.ghost(column), config.ghost[input_column]);
        ASSERT_EQ(state.boundary(column), config.boundary[input_column]);
    }
    long double density_sum = 0;
    std::size_t max_degree = 0;
    for (const auto& expected : oracle) {
        std::size_t index = 0;
        while (index < state.row_count() && state.row_id(index) != expected.id.value())
            ++index;
        ASSERT_LT(index, state.row_count());
        const Offset first = state.offset(index);
        const Offset past = state.offset(index + 1);
        ASSERT_LE(first, past);
        ASSERT_LE(past, state.edge_count());
        ASSERT_EQ(past - first, expected.neighbors.size());
        for (Offset edge = first; edge < past; ++edge) {
            ASSERT_LT(state.index(edge), state.column_count());
            ASSERT_EQ(state.column_id(state.index(edge)), expected.neighbors[edge - first].value());
        }
        // On the long-row fixture the q values are dyadic at half-integer support
        // points. Nonnegative terms have up to three roundings apiece followed
        // by at most one addition per edge. Use the degree-dependent bound there.
        ASSERT_NO_FATAL_FAILURE(
            check_close(state.density(index), expected.density, expected.neighbors.size() + 4));
        max_degree = std::max(max_degree, expected.neighbors.size());
        density_sum += expected.density;
    }
    ASSERT_NO_FATAL_FAILURE(
        check_close(state.owned_density_sum, density_sum, max_degree + 4 + oracle.size()));
}

void check_same_accepted(const Snapshot& before, const Snapshot& after) {
    ASSERT_EQ(before.stamp, after.stamp);
    ASSERT_EQ(before.density_valid, after.density_valid);
    ASSERT_EQ(before.owned_density_sum, after.owned_density_sum);
    ASSERT_EQ(before.row_count(), after.row_count());
    ASSERT_EQ(before.column_count(), after.column_count());
    ASSERT_EQ(before.edge_count(), after.edge_count());
    for (std::size_t row = 0; row < before.row_count(); ++row) {
        ASSERT_EQ(before.row_id(row), after.row_id(row));
        ASSERT_EQ(before.offset(row), after.offset(row));
        if (before.density_valid)
            ASSERT_EQ(before.density(row), after.density(row));
    }
    ASSERT_EQ(before.offset(before.row_count()), after.offset(after.row_count()));
    for (std::size_t column = 0; column < before.column_count(); ++column) {
        ASSERT_EQ(before.column_id(column), after.column_id(column));
        ASSERT_EQ(before.ghost(column), after.ghost(column));
        ASSERT_EQ(before.boundary(column), after.boundary(column));
    }
    for (std::size_t edge = 0; edge < before.edge_count(); ++edge)
        ASSERT_EQ(before.index(edge), after.index(edge));
}

void prepare(Simulation& simulation) {
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
}

TEST(Sph, FrozenLayoutAndReorder) {
    const auto config = fixture::frozen_config<Config>();
    const auto rows = fixture::strong_ids<GlobalEntityId>(fixture::reordered_rows);
    const auto columns = fixture::strong_ids<GlobalEntityId>(fixture::reordered_columns);
    Participant participant(config);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto initial = require(participant.snapshot());
    ASSERT_TRUE(initial.stamp.accepted);
    ASSERT_FALSE(initial.density_valid);
    ASSERT_NO_FATAL_FAILURE(check_frozen(initial, false));
    ASSERT_EQ(participant.reorder(rows, columns, initial.stamp).code(), StatusCode::invalid_state);
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(simulation.advance(0.125).ok());
    const auto original = require(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_frozen(original, false));
    ASSERT_NO_FATAL_FAILURE(check_oracle(original, config, scalar_oracle(config)));

    auto stale = original.stamp;
    stale.layout = LayoutVersion{stale.layout.value() + 1};
    ASSERT_EQ(participant.reorder(rows, columns, stale).code(), StatusCode::invalid_state);
    stale = original.stamp;
    stale.geometry = GeometryVersion{stale.geometry.value() + 1};
    ASSERT_EQ(participant.reorder(rows, columns, stale).code(), StatusCode::invalid_state);
    stale = original.stamp;
    stale.topology = TopologyVersion{stale.topology.value() + 1};
    ASSERT_EQ(participant.reorder(rows, columns, stale).code(), StatusCode::invalid_state);
    auto duplicate_rows = rows;
    duplicate_rows[0] = duplicate_rows[1];
    ASSERT_EQ(participant.reorder(duplicate_rows, columns, original.stamp).code(),
              StatusCode::invalid_argument);
    ASSERT_NO_FATAL_FAILURE(check_same_accepted(original, require(participant.snapshot())));

    ASSERT_TRUE(participant.reorder(rows, columns, original.stamp).ok());
    const auto reordered = require(participant.snapshot());
    ASSERT_EQ(reordered.stamp.geometry, original.stamp.geometry);
    ASSERT_EQ(reordered.stamp.accepted_version, original.stamp.accepted_version);
    ASSERT_EQ(reordered.stamp.time, original.stamp.time);
    ASSERT_EQ(reordered.owned_density_sum, original.owned_density_sum);
    ASSERT_NE(reordered.stamp.storage_incarnation, original.stamp.storage_incarnation);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto source = std::find(config.row_ids.begin(), config.row_ids.end(), rows[row]);
        ASSERT_EQ(reordered.density(row), original.density(source - config.row_ids.begin()));
    }
    ASSERT_NO_FATAL_FAILURE(check_frozen(reordered, true));
    ASSERT_EQ(participant.reorder(rows, columns, original.stamp).code(), StatusCode::invalid_state);
    ASSERT_TRUE(simulation.advance(0.125).ok());
    const auto advanced = require(participant.snapshot());
    ASSERT_EQ(advanced.stamp.accepted_version, StateVersion{2});
    ASSERT_EQ(advanced.stamp.time.physical_time, 0.25);
    ASSERT_NO_FATAL_FAILURE(check_frozen(advanced, true));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_FALSE(participant.snapshot().ok());
    ASSERT_NO_FATAL_FAILURE(check_frozen(initial, false));
    ASSERT_NO_FATAL_FAILURE(check_frozen(original, false));
    ASSERT_NO_FATAL_FAILURE(check_frozen(reordered, true));
}

TEST(Sph, RestoreRetryAndReplay) {
    constexpr double step = 0.125;
    const auto config = fixture::frozen_config<Config>();
    Participant participant(config);
    Simulation simulation(participant);
    ASSERT_NO_FATAL_FAILURE(prepare(simulation));
    ASSERT_TRUE(simulation.advance(step).ok());
    const auto accepted = require(participant.snapshot());
    ASSERT_TRUE(accepted.stamp.accepted);
    ASSERT_TRUE(accepted.density_valid);
    ASSERT_TRUE(simulation.begin_window(step).ok());
    ASSERT_EQ(participant.reorder(config.row_ids, config.column_ids, accepted.stamp).code(),
              StatusCode::invalid_state);
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto abandoned_stamp = require(participant.trial_stamp());
    const auto abandoned = require(participant.trial_snapshot(abandoned_stamp));
    ASSERT_FALSE(abandoned.stamp.accepted);
    ASSERT_EQ(abandoned.stamp, abandoned_stamp);
    ASSERT_NE(abandoned.stamp.storage_incarnation, accepted.stamp.storage_incarnation);
    ASSERT_NO_FATAL_FAILURE(check_frozen(abandoned, false));
    ASSERT_NO_FATAL_FAILURE(check_same_accepted(accepted, require(participant.snapshot())));
    ASSERT_TRUE(simulation.restore_window().ok());
    ASSERT_EQ(simulation.state(), SimulationState::prepared);
    ASSERT_EQ(simulation.time().accepted_time, step);
    ASSERT_EQ(simulation.time().accepted_steps, 1);
    ASSERT_EQ(simulation.time().trial_attempts, 2);
    ASSERT_EQ(participant.trial_snapshot(abandoned_stamp).status().code(),
              StatusCode::invalid_state);
    ASSERT_NO_FATAL_FAILURE(check_same_accepted(accepted, require(participant.snapshot())));

    ASSERT_TRUE(simulation.begin_window(step).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto stamp = require(participant.trial_stamp());
    ASSERT_NE(stamp.trial.attempt, abandoned_stamp.trial.attempt);
    ASSERT_EQ(participant.trial_snapshot(abandoned_stamp).status().code(),
              StatusCode::invalid_state);
    auto stale = stamp;
    ++stale.trial.attempt;
    ASSERT_EQ(participant.trial_snapshot(stale).status().code(), StatusCode::invalid_state);
    stale = stamp;
    stale.layout = LayoutVersion{stamp.layout.value() + 1};
    ASSERT_EQ(participant.trial_snapshot(stale).status().code(), StatusCode::invalid_state);
    stale = stamp;
    stale.geometry = GeometryVersion{stamp.geometry.value() + 1};
    ASSERT_EQ(participant.trial_snapshot(stale).status().code(), StatusCode::invalid_state);
    stale = stamp;
    stale.topology = TopologyVersion{stamp.topology.value() + 1};
    ASSERT_EQ(participant.trial_snapshot(stale).status().code(), StatusCode::invalid_state);

    auto retained_trial = require(participant.trial_snapshot(stamp));
    ASSERT_NO_FATAL_FAILURE(check_frozen(retained_trial, false));
    auto trial_copy = retained_trial;
    const auto trial_bytes = retained_trial.row_count() * sizeof(real);
    const auto bytes_before_commit = participant.resource_usage().controlled_bytes;
    ASSERT_TRUE(simulation.commit_window().ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, bytes_before_commit + trial_bytes);
    const auto replayed = require(participant.snapshot());
    ASSERT_TRUE(replayed.stamp.accepted);
    ASSERT_TRUE(replayed.density_valid);
    ASSERT_EQ(retained_trial.stamp, stamp);
    ASSERT_FALSE(retained_trial.stamp.accepted);
    ASSERT_NE(retained_trial.stamp.storage_incarnation, replayed.stamp.storage_incarnation);
    ASSERT_EQ(retained_trial.owned_density_sum, replayed.owned_density_sum);
    for (std::size_t row = 0; row < replayed.row_count(); ++row)
        ASSERT_EQ(retained_trial.density(row), replayed.density(row));
    ASSERT_EQ(participant.trial_snapshot(stamp).status().code(), StatusCode::invalid_state);
    ASSERT_EQ(simulation.time().accepted_steps, 2);
    ASSERT_EQ(simulation.time().trial_attempts, 3);
    ASSERT_TRUE(simulation.finalize().ok());

    Participant continuous(config);
    Simulation baseline(continuous);
    ASSERT_NO_FATAL_FAILURE(prepare(baseline));
    ASSERT_TRUE(baseline.advance(step).ok());
    ASSERT_TRUE(baseline.advance(step).ok());
    const auto direct = require(continuous.snapshot());
    ASSERT_EQ(direct.stamp.accepted_version, replayed.stamp.accepted_version);
    ASSERT_EQ(direct.stamp.time, replayed.stamp.time);
    ASSERT_EQ(direct.owned_density_sum, replayed.owned_density_sum);
    ASSERT_EQ(baseline.time().trial_attempts, 2);
    for (std::size_t row = 0; row < direct.row_count(); ++row)
        ASSERT_EQ(direct.density(row), replayed.density(row));
    ASSERT_NO_FATAL_FAILURE(check_frozen(direct, false));
    ASSERT_NO_FATAL_FAILURE(check_frozen(replayed, false));
    ASSERT_NO_FATAL_FAILURE(check_frozen(abandoned, false));
    ASSERT_NO_FATAL_FAILURE(check_frozen(retained_trial, false));
    const auto bytes_before_trial_release = participant.resource_usage().controlled_bytes;
    retained_trial = {};
    ASSERT_EQ(participant.resource_usage().controlled_bytes, bytes_before_trial_release);
    ASSERT_EQ(trial_copy.density(0), replayed.density(0));
    trial_copy = {};
    ASSERT_EQ(participant.resource_usage().controlled_bytes,
              bytes_before_trial_release - trial_bytes);
    ASSERT_TRUE(baseline.finalize().ok());
}

TEST(Sph, DensitylessLayoutIncarnations) {
    const auto config = fixture::frozen_config<Config>();
    const auto rows = fixture::strong_ids<GlobalEntityId>(fixture::reordered_rows);
    const auto columns = fixture::strong_ids<GlobalEntityId>(fixture::reordered_columns);
    Participant first(config);
    Simulation first_simulation(first);
    ASSERT_NO_FATAL_FAILURE(prepare(first_simulation));
    auto first_initial = require(first.snapshot());
    ASSERT_TRUE(first.reorder(rows, columns, first_initial.stamp).ok());
    auto first_reordered = require(first.snapshot());
    ASSERT_NE(first_initial.stamp.storage_incarnation, first_reordered.stamp.storage_incarnation);
    ASSERT_EQ(first.reorder(config.row_ids, config.column_ids, first_initial.stamp).code(),
              StatusCode::invalid_state);
    ASSERT_TRUE(first_simulation.finalize().ok());
    ASSERT_FALSE(first_initial.density_valid);
    ASSERT_FALSE(first_reordered.density_valid);
    ASSERT_NO_FATAL_FAILURE(check_frozen(first_initial, false));
    ASSERT_NO_FATAL_FAILURE(check_frozen(first_reordered, true));
    const auto first_initial_stamp = first_initial.stamp;
    const auto first_reordered_stamp = first_reordered.stamp;
    first_initial = {};
    first_reordered = {};
    ASSERT_EQ(first.resource_usage().controlled_bytes, 0);

    Participant second(config);
    Simulation second_simulation(second);
    ASSERT_NO_FATAL_FAILURE(prepare(second_simulation));
    const auto second_initial = require(second.snapshot());
    ASSERT_NE(second_initial.stamp.storage_incarnation, first_initial_stamp.storage_incarnation);
    ASSERT_EQ(second.reorder(rows, columns, first_initial_stamp).code(), StatusCode::invalid_state);
    ASSERT_TRUE(second.reorder(rows, columns, second_initial.stamp).ok());
    const auto second_reordered = require(second.snapshot());
    ASSERT_NE(second_reordered.stamp.storage_incarnation,
              first_reordered_stamp.storage_incarnation);
    ASSERT_EQ(second.reorder(config.row_ids, config.column_ids, first_reordered_stamp).code(),
              StatusCode::invalid_state);
    ASSERT_TRUE(second_simulation.advance(0.125).ok());
    const auto accepted = require(second.snapshot());
    ASSERT_NE(accepted.stamp.storage_incarnation, second_reordered.stamp.storage_incarnation);
    ASSERT_EQ(second.reorder(config.row_ids, config.column_ids, second_reordered.stamp).code(),
              StatusCode::invalid_state);
    ASSERT_TRUE(second_simulation.finalize().ok());
    ASSERT_FALSE(second_initial.density_valid);
    ASSERT_FALSE(second_reordered.density_valid);
    ASSERT_NO_FATAL_FAILURE(check_frozen(second_initial, false));
    ASSERT_NO_FATAL_FAILURE(check_frozen(second_reordered, true));
    ASSERT_NO_FATAL_FAILURE(check_frozen(accepted, true));
}

constexpr auto csr_bytes = sizeof(fixture::offsets) + sizeof(fixture::indices);
constexpr auto geometry_bytes =
    csr_bytes + fixture::row_ids.size() * (sizeof(std::uint64_t) + sizeof(LocalIndex)) +
    fixture::column_ids.size() *
        (sizeof(std::uint64_t) + 2 * sizeof(real) + 2 * sizeof(std::uint8_t));
constexpr auto density_bytes = fixture::row_ids.size() * sizeof(real);
constexpr auto reorder_scratch_bytes =
    csr_bytes + 2 * (fixture::row_ids.size() + fixture::column_ids.size()) * sizeof(LocalIndex);

TEST(Sph, RetainedDensityBudget) {
    constexpr double step = 0.125;
    constexpr auto limit = geometry_bytes + 2 * density_bytes;
    Participant participant(fixture::frozen_config<Config>(), ParticipantId{1}, {3, limit});
    Simulation simulation(participant);
    ASSERT_NO_FATAL_FAILURE(prepare(simulation));
    ASSERT_EQ(participant.resource_usage().controlled_bytes, geometry_bytes);
    ASSERT_TRUE(simulation.advance(step).ok());
    auto old = require(participant.snapshot());
    auto old_copy = old;
    ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    ASSERT_TRUE(simulation.advance(step).ok());
    const auto accepted = require(participant.snapshot());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, limit);
    ASSERT_EQ(simulation.begin_window(step).code(), StatusCode::resource_exhausted);
    old = {};
    ASSERT_EQ(simulation.begin_window(step).code(), StatusCode::resource_exhausted);
    ASSERT_EQ(simulation.time().accepted_steps, 2);
    ASSERT_NO_FATAL_FAILURE(check_frozen(old_copy, false));
    ASSERT_NO_FATAL_FAILURE(check_same_accepted(accepted, require(participant.snapshot())));
    old_copy = {};

    ASSERT_TRUE(simulation.begin_window(step).ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, limit);
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto stamp = require(participant.trial_stamp());
    auto trial = require(participant.trial_snapshot(stamp));
    auto trial_copy = trial;
    auto another = require(participant.trial_snapshot(stamp));
    ASSERT_EQ(participant.resource_usage().snapshot_records, 3);
    ASSERT_EQ(participant.trial_snapshot(stamp).status().code(), StatusCode::resource_exhausted);
    ASSERT_EQ(participant.snapshot().status().code(), StatusCode::resource_exhausted);
    another = {};
    // Publishing a separately retained trial would require a third density bank.
    ASSERT_EQ(simulation.commit_window().code(), StatusCode::resource_exhausted);
    ASSERT_EQ(simulation.state(), SimulationState::prepared);
    ASSERT_EQ(simulation.time().accepted_steps, 2);
    ASSERT_NO_FATAL_FAILURE(check_same_accepted(accepted, require(participant.snapshot())));
    ASSERT_NO_FATAL_FAILURE(check_frozen(trial, false));
    trial = {};
    ASSERT_EQ(simulation.begin_window(step).code(), StatusCode::resource_exhausted);
    ASSERT_NO_FATAL_FAILURE(check_frozen(trial_copy, false));
    trial_copy = {};
    ASSERT_EQ(participant.resource_usage().controlled_bytes, geometry_bytes + density_bytes);
    ASSERT_TRUE(simulation.advance(step).ok());
    ASSERT_EQ(simulation.time().accepted_steps, 3);
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, geometry_bytes + density_bytes);
    ASSERT_NO_FATAL_FAILURE(check_frozen(accepted, false));
}

TEST(Sph, RetainedLayoutBudget) {
    constexpr auto limit = 2 * (geometry_bytes + density_bytes) + reorder_scratch_bytes;
    const auto config = fixture::frozen_config<Config>();
    const auto rows = fixture::strong_ids<GlobalEntityId>(fixture::reordered_rows);
    const auto columns = fixture::strong_ids<GlobalEntityId>(fixture::reordered_columns);
    Participant participant(config, ParticipantId{1}, {4, limit});
    Simulation simulation(participant);
    ASSERT_NO_FATAL_FAILURE(prepare(simulation));
    ASSERT_TRUE(simulation.advance(0.125).ok());
    auto old = require(participant.snapshot());
    auto old_copy = old;
    ASSERT_TRUE(participant.reorder(rows, columns, old.stamp).ok());
    auto current = require(participant.snapshot());
    const auto retained_bytes = 2 * (geometry_bytes + density_bytes);
    ASSERT_EQ(participant.resource_usage().controlled_bytes, retained_bytes);
    ASSERT_EQ(participant.reorder(config.row_ids, config.column_ids, current.stamp).code(),
              StatusCode::resource_exhausted);
    ASSERT_NO_FATAL_FAILURE(check_same_accepted(current, require(participant.snapshot())));
    ASSERT_EQ(participant.resource_usage().controlled_bytes, retained_bytes);
    old = {};
    ASSERT_EQ(participant.reorder(config.row_ids, config.column_ids, current.stamp).code(),
              StatusCode::resource_exhausted);
    ASSERT_NO_FATAL_FAILURE(check_frozen(old_copy, false));
    old_copy = {};
    ASSERT_EQ(participant.resource_usage().controlled_bytes, geometry_bytes + density_bytes);
    ASSERT_TRUE(participant.reorder(config.row_ids, config.column_ids, current.stamp).ok());
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, geometry_bytes + density_bytes);
    ASSERT_EQ(participant.resource_usage().snapshot_records, 1);
    ASSERT_NO_FATAL_FAILURE(check_frozen(current, true));
    current = {};
    ASSERT_EQ(participant.resource_usage().controlled_bytes, 0);
    ASSERT_EQ(participant.resource_usage().snapshot_records, 0);
}

TEST(Sph, LongAndEmptyRows) {
    constexpr LocalIndex degree = 4096;
    Config config;
    config.row_ids = {GlobalEntityId{91}, GlobalEntityId{64}};
    config.column_ids = config.row_ids;
    config.positions = {0, 5};
    config.masses = {3, 2};
    config.ghost = {0, 0};
    config.boundary = {1, 0};
    config.offsets = {0, degree, degree};
    for (LocalIndex edge = 0; edge < degree; ++edge) {
        config.column_ids.emplace_back(1000 + edge);
        config.positions.push_back(static_cast<real>(1 + edge % 4) * real{0.5});
        config.masses.push_back(static_cast<real>(1 + edge % 7));
        config.ghost.push_back(1);
        config.boundary.push_back(0);
        config.indices.push_back(edge + 2);
    }
    const auto oracle = scalar_oracle(config);
    Participant participant(config);
    Simulation simulation(participant);
    ASSERT_NO_FATAL_FAILURE(prepare(simulation));
    ASSERT_TRUE(simulation.advance(0.125).ok());
    const auto original = require(participant.snapshot());
    ASSERT_EQ(original.offset(1) - original.offset(0), degree);
    ASSERT_EQ(original.offset(2), original.offset(1));
    ASSERT_NO_FATAL_FAILURE(check_oracle(original, config, oracle));

    auto rows = config.row_ids;
    auto columns = config.column_ids;
    std::ranges::reverse(rows);
    std::ranges::reverse(columns);
    ASSERT_TRUE(participant.reorder(rows, columns, original.stamp).ok());
    ASSERT_NO_FATAL_FAILURE(check_oracle(require(participant.snapshot()), config, oracle));
    ASSERT_TRUE(simulation.advance(0.125).ok());
    const auto reordered = require(participant.snapshot());
    ASSERT_EQ(reordered.row_id(0), rows[0].value());
    ASSERT_EQ(reordered.offset(1), reordered.offset(0));
    ASSERT_EQ(reordered.offset(2) - reordered.offset(1), degree);
    ASSERT_NO_FATAL_FAILURE(check_oracle(reordered, config, oracle));
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(Sph, RejectsInvalidInput) {
    const auto frozen = fixture::frozen_config<Config>();
    const auto reject = [&](auto mutate) {
        auto config = frozen;
        mutate(config);
        Participant participant(std::move(config));
        Simulation simulation(participant);
        ASSERT_EQ(simulation.initialize().code(), StatusCode::invalid_argument);
        ASSERT_FALSE(participant.snapshot().ok());
        ASSERT_TRUE(simulation.finalize().ok());
    };
    ASSERT_NO_FATAL_FAILURE(reject([](Config& config) { config.masses[0] = 0; }));
    ASSERT_NO_FATAL_FAILURE(reject(
        [](Config& config) { config.positions[0] = std::numeric_limits<real>::infinity(); }));
    ASSERT_NO_FATAL_FAILURE(reject([](Config& config) { config.boundary[3] = 2; }));
    ASSERT_NO_FATAL_FAILURE(
        reject([](Config& config) { config.row_ids[0] = config.column_ids.back(); }));
    ASSERT_NO_FATAL_FAILURE(reject([](Config& config) { ++config.offsets.back(); }));
    ASSERT_NO_FATAL_FAILURE(reject([](Config& config) { config.indices[0] = 6; }));
    ASSERT_NO_FATAL_FAILURE(reject([](Config& config) { config.indices[0] = 0; }));
    ASSERT_NO_FATAL_FAILURE(reject([](Config& config) { config.indices[0] = config.indices[1]; }));
    ASSERT_NO_FATAL_FAILURE(reject([](Config& config) { config.indices[0] = 4; }));
    ASSERT_NO_FATAL_FAILURE(reject([](Config& config) {
        config.indices.erase(config.indices.begin() + 3);
        for (std::size_t offset = 2; offset < config.offsets.size(); ++offset)
            --config.offsets[offset];
    }));
}

} // namespace
