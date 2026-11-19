#include <Kokkos_Core.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <participants/sph/Participant.hpp>
#include <sph_fixture.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace nexora;
using sph::Config;
using sph::Participant;
using sph::Snapshot;
using Fault = Participant::Fault;
namespace fixture = sph_fixture;

static_assert(std::is_same_v<decltype(std::declval<const Snapshot&>().density(0)), real>);
static_assert(std::is_same_v<decltype(std::declval<const Snapshot&>().index(0)), LocalIndex>);

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error("line " + std::to_string(line) + " failed check " + expression);
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw std::runtime_error(result.status().message());
    return std::move(result).value();
}

void check_close(real actual, long double expected, long double epsilon_multiplier) {
    CHECK(std::isfinite(actual));
    const long double tolerance = epsilon_multiplier * std::numeric_limits<real>::epsilon() *
                                  std::abs(expected);
    const long double error = std::abs(static_cast<long double>(actual) - expected);
    if (error > tolerance) {
        std::cerr << "density actual=" << actual << " oracle=" << expected << " error=" << error
                  << " tolerance=" << tolerance << '\n';
    }
    CHECK(error <= tolerance);
}

void check_frozen(const Snapshot& state, bool reordered) {
    const auto& rows = reordered ? fixture::reordered_rows : fixture::row_ids;
    const auto& columns = reordered ? fixture::reordered_columns : fixture::column_ids;
    const auto& offsets = reordered ? fixture::reordered_offsets : fixture::offsets;
    const auto& indices = reordered ? fixture::reordered_indices : fixture::indices;
    CHECK(state.row_count() == rows.size());
    CHECK(state.column_count() == columns.size());
    CHECK(state.edge_count() == indices.size());
    for (std::size_t row = 0; row < rows.size(); ++row)
        CHECK(state.row_id(row) == rows[row]);
    for (std::size_t column = 0; column < columns.size(); ++column) {
        CHECK(state.column_id(column) == columns[column]);
        const auto original = std::find(fixture::column_ids.begin(), fixture::column_ids.end(),
                                        columns[column]) - fixture::column_ids.begin();
        CHECK(state.ghost(column) == fixture::ghost[original]);
        CHECK(state.boundary(column) == fixture::boundary[original]);
    }
    for (std::size_t row = 0; row < offsets.size(); ++row)
        CHECK(state.offset(row) == offsets[row]);
    for (std::size_t edge = 0; edge < indices.size(); ++edge)
        CHECK(state.index(edge) == indices[edge]);
    CHECK(state.stamp.topology == TopologyVersion{1});
    CHECK(state.stamp.layout == LayoutVersion{reordered ? 2u : 1u});
    CHECK(state.stamp.storage_incarnation != 0);
    CHECK(state.stamp.validate().ok());
    if (!state.density_valid)
        return;
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto original = std::find(fixture::row_ids.begin(), fixture::row_ids.end(), rows[row]);
        const auto numerator = fixture::density_numerators[original - fixture::row_ids.begin()];
        check_close(state.density(row), static_cast<long double>(numerator) / fixture::denominator,
                    fixture::density_epsilon);
    }
    check_close(state.owned_density_sum,
                static_cast<long double>(fixture::sum_numerator) / fixture::denominator,
                fixture::sum_epsilon);
}

struct OracleRow {
    GlobalEntityId id;
    std::vector<GlobalEntityId> neighbors;
    long double density = 0;
};

std::size_t column_of(const Config& config, GlobalEntityId id) {
    const auto column = std::find(config.column_ids.begin(), config.column_ids.end(), id);
    CHECK(column != config.column_ids.end());
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
            const long double weight = (cube(std::max(2 - q, 0.0L)) -
                                        4 * cube(std::max(1 - q, 0.0L))) /
                                       6;
            row.density += config.masses[column] * weight;
        }
        result.push_back(std::move(row));
    }
    return result;
}

void check_oracle(const Snapshot& state, const Config& config,
                  const std::vector<OracleRow>& oracle) {
    CHECK(state.density_valid);
    CHECK(state.row_count() == oracle.size());
    CHECK(state.column_count() == config.column_ids.size());
    CHECK(state.offset(0) == 0);
    CHECK(state.offset(state.row_count()) == state.edge_count());
    for (std::size_t column = 0; column < config.column_ids.size(); ++column) {
        const auto input_column = column_of(config, GlobalEntityId{state.column_id(column)});
        CHECK(state.ghost(column) == config.ghost[input_column]);
        CHECK(state.boundary(column) == config.boundary[input_column]);
    }
    long double density_sum = 0;
    std::size_t max_degree = 0;
    for (const auto& expected : oracle) {
        std::size_t index = 0;
        while (index < state.row_count() && state.row_id(index) != expected.id.value())
            ++index;
        CHECK(index < state.row_count());
        const Offset first = state.offset(index);
        const Offset past = state.offset(index + 1);
        CHECK(first <= past && past <= state.edge_count());
        CHECK(past - first == expected.neighbors.size());
        for (Offset edge = first; edge < past; ++edge) {
            CHECK(state.index(edge) < state.column_count());
            CHECK(state.column_id(state.index(edge)) == expected.neighbors[edge - first].value());
        }
        // On the long-row fixture the q values are dyadic at half-integer support
        // points. Nonnegative terms have up to three roundings apiece followed
        // by at most one addition per edge. Use the degree-dependent bound there.
        check_close(state.density(index), expected.density, expected.neighbors.size() + 4);
        max_degree = std::max(max_degree, expected.neighbors.size());
        density_sum += expected.density;
    }
    check_close(state.owned_density_sum, density_sum, max_degree + 4 + oracle.size());
}

void check_same_accepted(const Snapshot& before, const Snapshot& after) {
    CHECK(before.stamp == after.stamp);
    CHECK(before.density_valid == after.density_valid);
    CHECK(before.owned_density_sum == after.owned_density_sum);
    CHECK(before.row_count() == after.row_count());
    CHECK(before.column_count() == after.column_count());
    CHECK(before.edge_count() == after.edge_count());
    for (std::size_t row = 0; row < before.row_count(); ++row) {
        CHECK(before.row_id(row) == after.row_id(row));
        CHECK(before.offset(row) == after.offset(row));
        if (before.density_valid)
            CHECK(before.density(row) == after.density(row));
    }
    CHECK(before.offset(before.row_count()) == after.offset(after.row_count()));
    for (std::size_t column = 0; column < before.column_count(); ++column) {
        CHECK(before.column_id(column) == after.column_id(column));
        CHECK(before.ghost(column) == after.ghost(column));
        CHECK(before.boundary(column) == after.boundary(column));
    }
    for (std::size_t edge = 0; edge < before.edge_count(); ++edge)
        CHECK(before.index(edge) == after.index(edge));
}

void prepare(Simulation& simulation) {
    CHECK(simulation.initialize().ok());
    CHECK(simulation.prepare().ok());
}

void support_edge_precision() {
    Config config;
    config.row_ids = {GlobalEntityId{1}};
    config.column_ids = {GlobalEntityId{1}, GlobalEntityId{2}};
    config.ghost = {0, 1};
    config.boundary = {0, 0};
    config.offsets = {0, 1};
    config.indices = {1};
    const auto check_density = [&](long double expected) {
        Participant participant(config);
        Simulation simulation(participant);
        prepare(simulation);
        CHECK(simulation.advance(0.125).ok());
        const auto accepted = require(participant.snapshot());
        CHECK(accepted.density_valid);
        check_close(accepted.density(0), expected, fixture::density_epsilon);
        check_close(accepted.owned_density_sum, expected, fixture::sum_epsilon);
        CHECK(simulation.finalize().ok());
    };

    constexpr int edge_exponent = std::numeric_limits<real>::digits + 1;
    const real gap = std::ldexp(real{1}, -edge_exponent);
    config.masses = {real{1}, std::ldexp(real{1}, 3 * edge_exponent + 3)};
    for (real sign : {real{1}, real{-1}}) {
        config.positions = {sign * real{2}, sign * gap};
        // The ghost contributes 8/6 and the owned self term contributes 2/3.
        check_density(2);
        config.positions[1] = -sign * gap;
        Participant participant(config);
        Simulation simulation(participant);
        CHECK(simulation.initialize().code() == StatusCode::invalid_argument);
        CHECK(simulation.finalize().ok());
    }
    config.positions = {real{2}, real{0}};
    check_density(2.0L / 3.0L);

    // The standalone weight underflows, while mass * gap^3 / 6 is normal.
    constexpr int tiny_gap_exponent = std::is_same_v<real, float> ? -51 : -360;
    constexpr int mass_exponent = std::numeric_limits<real>::max_exponent - 1;
    constexpr int contribution_exponent = mass_exponent + 3 * tiny_gap_exponent;
    config.positions[1] = std::ldexp(real{1}, tiny_gap_exponent);
    config.masses = {std::ldexp(real{1}, contribution_exponent - 4),
                     std::ldexp(real{1}, mass_exponent)};
    check_density(std::ldexp(5.0L / 24.0L, contribution_exponent));
}

void frozen_layout_and_reorder() {
    const auto config = fixture::frozen_config<Config>();
    const auto rows = fixture::strong_ids<GlobalEntityId>(fixture::reordered_rows);
    const auto columns = fixture::strong_ids<GlobalEntityId>(fixture::reordered_columns);
    Participant participant(config);
    Simulation simulation(participant);
    CHECK(simulation.initialize().ok());
    const auto initial = require(participant.snapshot());
    CHECK(initial.stamp.accepted && !initial.density_valid);
    check_frozen(initial, false);
    CHECK(participant.reorder(rows, columns, initial.stamp).code() == StatusCode::invalid_state);
    CHECK(simulation.prepare().ok());
    CHECK(simulation.advance(0.125).ok());
    const auto original = require(participant.snapshot());
    check_frozen(original, false);
    check_oracle(original, config, scalar_oracle(config));

    auto stale = original.stamp;
    stale.layout = LayoutVersion{stale.layout.value() + 1};
    CHECK(participant.reorder(rows, columns, stale).code() == StatusCode::invalid_state);
    stale = original.stamp;
    stale.geometry = GeometryVersion{stale.geometry.value() + 1};
    CHECK(participant.reorder(rows, columns, stale).code() == StatusCode::invalid_state);
    stale = original.stamp;
    stale.topology = TopologyVersion{stale.topology.value() + 1};
    CHECK(participant.reorder(rows, columns, stale).code() == StatusCode::invalid_state);
    auto duplicate_rows = rows;
    duplicate_rows[0] = duplicate_rows[1];
    CHECK(participant.reorder(duplicate_rows, columns, original.stamp).code() ==
          StatusCode::invalid_argument);
    check_same_accepted(original, require(participant.snapshot()));

    CHECK(participant.reorder(rows, columns, original.stamp).ok());
    const auto reordered = require(participant.snapshot());
    CHECK(reordered.stamp.geometry == original.stamp.geometry);
    CHECK(reordered.stamp.accepted_version == original.stamp.accepted_version);
    CHECK(reordered.stamp.time == original.stamp.time);
    CHECK(reordered.owned_density_sum == original.owned_density_sum);
    CHECK(reordered.stamp.storage_incarnation != original.stamp.storage_incarnation);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto source = std::find(config.row_ids.begin(), config.row_ids.end(), rows[row]);
        CHECK(reordered.density(row) == original.density(source - config.row_ids.begin()));
    }
    check_frozen(reordered, true);
    CHECK(participant.reorder(rows, columns, original.stamp).code() == StatusCode::invalid_state);
    CHECK(simulation.advance(0.125).ok());
    const auto advanced = require(participant.snapshot());
    CHECK(advanced.stamp.accepted_version == StateVersion{2});
    CHECK(advanced.stamp.time.physical_time == 0.25);
    check_frozen(advanced, true);
    CHECK(simulation.finalize().ok());
    CHECK(!participant.snapshot().ok());
    check_frozen(initial, false);
    check_frozen(original, false);
    check_frozen(reordered, true);
}

void restore_retry_and_replay() {
    constexpr double step = 0.125;
    const auto config = fixture::frozen_config<Config>();
    Participant participant(config);
    Simulation simulation(participant);
    prepare(simulation);
    CHECK(simulation.advance(step).ok());
    const auto accepted = require(participant.snapshot());
    CHECK(accepted.stamp.accepted && accepted.density_valid);
    CHECK(simulation.begin_window(step).ok());
    CHECK(participant.reorder(config.row_ids, config.column_ids, accepted.stamp).code() ==
          StatusCode::invalid_state);
    CHECK(simulation.advance_trial().ok());
    const auto abandoned_stamp = require(participant.trial_stamp());
    const auto abandoned = require(participant.trial_snapshot(abandoned_stamp));
    CHECK(!abandoned.stamp.accepted);
    CHECK(abandoned.stamp == abandoned_stamp);
    CHECK(abandoned.stamp.storage_incarnation != accepted.stamp.storage_incarnation);
    check_frozen(abandoned, false);
    check_same_accepted(accepted, require(participant.snapshot()));
    CHECK(simulation.restore_window().ok());
    CHECK(participant.trial_snapshot(abandoned_stamp).status().code() == StatusCode::invalid_state);

    participant.fail_next(Fault::advance);
    CHECK(simulation.advance(step).code() == StatusCode::numerical_failure);
    CHECK(simulation.state() == SimulationState::prepared);
    check_same_accepted(accepted, require(participant.snapshot()));
    CHECK(simulation.begin_window(step).ok());
    CHECK(simulation.advance_trial().ok());
    const auto stamp = require(participant.trial_stamp());
    CHECK(stamp.trial.attempt != abandoned_stamp.trial.attempt);
    CHECK(participant.trial_snapshot(abandoned_stamp).status().code() == StatusCode::invalid_state);
    auto stale = stamp;
    ++stale.trial.attempt;
    CHECK(participant.trial_snapshot(stale).status().code() == StatusCode::invalid_state);
    stale = stamp;
    stale.layout = LayoutVersion{stamp.layout.value() + 1};
    CHECK(participant.trial_snapshot(stale).status().code() == StatusCode::invalid_state);
    stale = stamp;
    stale.geometry = GeometryVersion{stamp.geometry.value() + 1};
    CHECK(participant.trial_snapshot(stale).status().code() == StatusCode::invalid_state);
    stale = stamp;
    stale.topology = TopologyVersion{stamp.topology.value() + 1};
    CHECK(participant.trial_snapshot(stale).status().code() == StatusCode::invalid_state);
    check_frozen(require(participant.trial_snapshot(stamp)), false);

    participant.fail_next(Fault::prepare);
    CHECK(simulation.commit_window().code() == StatusCode::execution_failure);
    CHECK(simulation.state() == SimulationState::prepared);
    CHECK(simulation.time().accepted_time == step);
    CHECK(simulation.time().accepted_steps == 1);
    CHECK(simulation.time().trial_attempts == 4);
    CHECK(participant.trial_snapshot(stamp).status().code() == StatusCode::invalid_state);
    check_same_accepted(accepted, require(participant.snapshot()));
    CHECK(simulation.begin_window(step).ok());
    CHECK(simulation.advance_trial().ok());
    const auto successful_stamp = require(participant.trial_stamp());
    auto retained_trial = require(participant.trial_snapshot(successful_stamp));
    auto trial_copy = retained_trial;
    const auto trial_bytes = retained_trial.row_count() * sizeof(real);
    const auto bytes_before_commit = participant.resource_usage().controlled_bytes;
    CHECK(simulation.commit_window().ok());
    CHECK(participant.resource_usage().controlled_bytes == bytes_before_commit + trial_bytes);
    const auto replayed = require(participant.snapshot());
    CHECK(replayed.stamp.accepted && replayed.density_valid);
    CHECK(retained_trial.stamp == successful_stamp && !retained_trial.stamp.accepted);
    CHECK(retained_trial.stamp.storage_incarnation != replayed.stamp.storage_incarnation);
    CHECK(retained_trial.owned_density_sum == replayed.owned_density_sum);
    for (std::size_t row = 0; row < replayed.row_count(); ++row)
        CHECK(retained_trial.density(row) == replayed.density(row));
    CHECK(participant.trial_snapshot(successful_stamp).status().code() == StatusCode::invalid_state);
    CHECK(simulation.time().accepted_steps == 2);
    CHECK(simulation.time().trial_attempts == 5);
    CHECK(simulation.finalize().ok());

    Participant continuous(config);
    Simulation baseline(continuous);
    prepare(baseline);
    CHECK(baseline.advance(step).ok());
    CHECK(baseline.advance(step).ok());
    const auto direct = require(continuous.snapshot());
    CHECK(direct.stamp.accepted_version == replayed.stamp.accepted_version);
    CHECK(direct.stamp.time == replayed.stamp.time);
    CHECK(direct.owned_density_sum == replayed.owned_density_sum);
    CHECK(baseline.time().trial_attempts == 2);
    for (std::size_t row = 0; row < direct.row_count(); ++row)
        CHECK(direct.density(row) == replayed.density(row));
    check_frozen(direct, false);
    check_frozen(replayed, false);
    check_frozen(abandoned, false);
    check_frozen(retained_trial, false);
    const auto bytes_before_trial_release = participant.resource_usage().controlled_bytes;
    retained_trial = {};
    CHECK(participant.resource_usage().controlled_bytes == bytes_before_trial_release);
    CHECK(trial_copy.density(0) == replayed.density(0));
    trial_copy = {};
    CHECK(participant.resource_usage().controlled_bytes == bytes_before_trial_release - trial_bytes);
    CHECK(baseline.finalize().ok());
}

void densityless_layout_incarnations() {
    const auto config = fixture::frozen_config<Config>();
    const auto rows = fixture::strong_ids<GlobalEntityId>(fixture::reordered_rows);
    const auto columns = fixture::strong_ids<GlobalEntityId>(fixture::reordered_columns);
    Participant first(config);
    Simulation first_simulation(first);
    prepare(first_simulation);
    auto first_initial = require(first.snapshot());
    CHECK(first.reorder(rows, columns, first_initial.stamp).ok());
    auto first_reordered = require(first.snapshot());
    CHECK(first_initial.stamp.storage_incarnation != first_reordered.stamp.storage_incarnation);
    CHECK(first.reorder(config.row_ids, config.column_ids, first_initial.stamp).code() ==
          StatusCode::invalid_state);
    CHECK(first_simulation.finalize().ok());
    CHECK(!first_initial.density_valid && !first_reordered.density_valid);
    check_frozen(first_initial, false);
    check_frozen(first_reordered, true);
    const auto first_initial_stamp = first_initial.stamp;
    const auto first_reordered_stamp = first_reordered.stamp;
    first_initial = {};
    first_reordered = {};
    CHECK(first.resource_usage().controlled_bytes == 0);

    Participant second(config);
    Simulation second_simulation(second);
    prepare(second_simulation);
    const auto second_initial = require(second.snapshot());
    CHECK(second_initial.stamp.storage_incarnation != first_initial_stamp.storage_incarnation);
    CHECK(second.reorder(rows, columns, first_initial_stamp).code() == StatusCode::invalid_state);
    CHECK(second.reorder(rows, columns, second_initial.stamp).ok());
    const auto second_reordered = require(second.snapshot());
    CHECK(second_reordered.stamp.storage_incarnation != first_reordered_stamp.storage_incarnation);
    CHECK(second.reorder(config.row_ids, config.column_ids, first_reordered_stamp).code() ==
          StatusCode::invalid_state);
    CHECK(second_simulation.advance(0.125).ok());
    const auto accepted = require(second.snapshot());
    CHECK(accepted.stamp.storage_incarnation != second_reordered.stamp.storage_incarnation);
    CHECK(second.reorder(config.row_ids, config.column_ids, second_reordered.stamp).code() ==
          StatusCode::invalid_state);
    CHECK(second_simulation.finalize().ok());
    CHECK(!second_initial.density_valid && !second_reordered.density_valid);
    check_frozen(second_initial, false);
    check_frozen(second_reordered, true);
    check_frozen(accepted, true);
}

constexpr auto csr_bytes = sizeof(fixture::offsets) + sizeof(fixture::indices);
constexpr auto geometry_bytes = csr_bytes +
    fixture::row_ids.size() * (sizeof(std::uint64_t) + sizeof(LocalIndex)) +
    fixture::column_ids.size() * (sizeof(std::uint64_t) + 2 * sizeof(real) +
                                  2 * sizeof(std::uint8_t));
constexpr auto density_bytes = fixture::row_ids.size() * sizeof(real);
constexpr auto reorder_scratch_bytes = csr_bytes +
    2 * (fixture::row_ids.size() + fixture::column_ids.size()) * sizeof(LocalIndex);

void retained_density_budget() {
    constexpr double step = 0.125;
    constexpr auto limit = geometry_bytes + 2 * density_bytes;
    Participant participant(fixture::frozen_config<Config>(), ParticipantId{1}, {3, limit});
    Simulation simulation(participant);
    prepare(simulation);
    CHECK(participant.resource_usage().controlled_bytes == geometry_bytes);
    CHECK(simulation.advance(step).ok());
    auto old = require(participant.snapshot());
    auto old_copy = old;
    CHECK(participant.resource_usage().snapshot_records == 1);
    CHECK(simulation.advance(step).ok());
    const auto accepted = require(participant.snapshot());
    CHECK(participant.resource_usage().controlled_bytes == limit);
    CHECK(simulation.begin_window(step).code() == StatusCode::resource_exhausted);
    old = {};
    CHECK(simulation.begin_window(step).code() == StatusCode::resource_exhausted);
    CHECK(simulation.time().accepted_steps == 2);
    check_frozen(old_copy, false);
    check_same_accepted(accepted, require(participant.snapshot()));
    old_copy = {};

    CHECK(simulation.begin_window(step).ok());
    CHECK(participant.resource_usage().controlled_bytes == limit);
    CHECK(simulation.advance_trial().ok());
    const auto stamp = require(participant.trial_stamp());
    auto trial = require(participant.trial_snapshot(stamp));
    auto trial_copy = trial;
    auto another = require(participant.trial_snapshot(stamp));
    CHECK(participant.resource_usage().snapshot_records == 3);
    CHECK(participant.trial_snapshot(stamp).status().code() == StatusCode::resource_exhausted);
    CHECK(participant.snapshot().status().code() == StatusCode::resource_exhausted);
    another = {};
    // Publishing a separately retained trial would require a third density bank.
    CHECK(simulation.commit_window().code() == StatusCode::resource_exhausted);
    CHECK(simulation.state() == SimulationState::prepared);
    CHECK(simulation.time().accepted_steps == 2);
    check_same_accepted(accepted, require(participant.snapshot()));
    check_frozen(trial, false);
    trial = {};
    CHECK(simulation.begin_window(step).code() == StatusCode::resource_exhausted);
    check_frozen(trial_copy, false);
    trial_copy = {};
    CHECK(participant.resource_usage().controlled_bytes == geometry_bytes + density_bytes);
    CHECK(simulation.advance(step).ok());
    CHECK(simulation.time().accepted_steps == 3);
    CHECK(simulation.finalize().ok());
    CHECK(participant.resource_usage().controlled_bytes == geometry_bytes + density_bytes);
    check_frozen(accepted, false);
}

void retained_layout_budget() {
    constexpr auto limit = 2 * (geometry_bytes + density_bytes) + reorder_scratch_bytes;
    const auto config = fixture::frozen_config<Config>();
    const auto rows = fixture::strong_ids<GlobalEntityId>(fixture::reordered_rows);
    const auto columns = fixture::strong_ids<GlobalEntityId>(fixture::reordered_columns);
    Participant participant(config, ParticipantId{1}, {4, limit});
    Simulation simulation(participant);
    prepare(simulation);
    CHECK(simulation.advance(0.125).ok());
    auto old = require(participant.snapshot());
    auto old_copy = old;
    CHECK(participant.reorder(rows, columns, old.stamp).ok());
    auto current = require(participant.snapshot());
    const auto retained_bytes = 2 * (geometry_bytes + density_bytes);
    CHECK(participant.resource_usage().controlled_bytes == retained_bytes);
    CHECK(participant.reorder(config.row_ids, config.column_ids, current.stamp).code() ==
          StatusCode::resource_exhausted);
    check_same_accepted(current, require(participant.snapshot()));
    CHECK(participant.resource_usage().controlled_bytes == retained_bytes);
    old = {};
    CHECK(participant.reorder(config.row_ids, config.column_ids, current.stamp).code() ==
          StatusCode::resource_exhausted);
    check_frozen(old_copy, false);
    old_copy = {};
    CHECK(participant.resource_usage().controlled_bytes == geometry_bytes + density_bytes);
    CHECK(participant.reorder(config.row_ids, config.column_ids, current.stamp).ok());
    CHECK(simulation.finalize().ok());
    CHECK(participant.resource_usage().controlled_bytes == geometry_bytes + density_bytes);
    CHECK(participant.resource_usage().snapshot_records == 1);
    check_frozen(current, true);
    current = {};
    CHECK(participant.resource_usage().controlled_bytes == 0);
    CHECK(participant.resource_usage().snapshot_records == 0);
}

void long_and_empty_rows() {
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
    prepare(simulation);
    CHECK(simulation.advance(0.125).ok());
    const auto original = require(participant.snapshot());
    CHECK(original.offset(1) - original.offset(0) == degree);
    CHECK(original.offset(2) == original.offset(1));
    check_oracle(original, config, oracle);

    auto rows = config.row_ids;
    auto columns = config.column_ids;
    std::ranges::reverse(rows);
    std::ranges::reverse(columns);
    CHECK(participant.reorder(rows, columns, original.stamp).ok());
    check_oracle(require(participant.snapshot()), config, oracle);
    CHECK(simulation.advance(0.125).ok());
    const auto reordered = require(participant.snapshot());
    CHECK(reordered.row_id(0) == rows[0].value());
    CHECK(reordered.offset(1) == reordered.offset(0));
    CHECK(reordered.offset(2) - reordered.offset(1) == degree);
    check_oracle(reordered, config, oracle);
    CHECK(simulation.finalize().ok());
}

void invalid_input_and_nonfinite_reduction() {
    const auto frozen = fixture::frozen_config<Config>();
    const auto reject = [&](auto mutate) {
        auto config = frozen;
        mutate(config);
        Participant participant(std::move(config));
        Simulation simulation(participant);
        CHECK(simulation.initialize().code() == StatusCode::invalid_argument);
        CHECK(!participant.snapshot().ok());
        CHECK(simulation.finalize().ok());
    };
    reject([](Config& config) { config.masses[0] = 0; });
    reject([](Config& config) { config.positions[0] = std::numeric_limits<real>::infinity(); });
    reject([](Config& config) { config.boundary[3] = 2; });
    reject([](Config& config) { config.row_ids[0] = config.column_ids.back(); });
    reject([](Config& config) { ++config.offsets.back(); });
    reject([](Config& config) { config.indices[0] = 6; });
    reject([](Config& config) { config.indices[0] = 0; });
    reject([](Config& config) { config.indices[0] = config.indices[1]; });
    reject([](Config& config) { config.indices[0] = 4; });
    reject([](Config& config) {
        config.indices.erase(config.indices.begin() + 3);
        for (std::size_t offset = 2; offset < config.offsets.size(); ++offset)
            --config.offsets[offset];
    });

    Config overflow;
    overflow.row_ids = fixture::strong_ids<GlobalEntityId>(std::array{1u, 2u});
    overflow.column_ids = overflow.row_ids;
    overflow.positions = {0, 3};
    overflow.masses.assign(2, std::numeric_limits<real>::max());
    overflow.boundary = {0, 0};
    overflow.ghost = {0, 0};
    overflow.offsets = {0, 0, 0};
    Participant participant(std::move(overflow));
    Simulation simulation(participant);
    prepare(simulation);
    const auto initial = require(participant.snapshot());
    CHECK(simulation.advance(0.125).code() == StatusCode::numerical_failure);
    CHECK(simulation.state() == SimulationState::prepared);
    CHECK(simulation.time().accepted_steps == 0);
    check_same_accepted(initial, require(participant.snapshot()));
    CHECK(simulation.finalize().ok());
}

void finite_rounded_reduction_at_range_limit() {
    if constexpr (std::is_same_v<real, double>) {
        Config config;
        config.row_ids = fixture::strong_ids<GlobalEntityId>(std::array{1u, 2u, 3u});
        config.column_ids = config.row_ids;
        config.positions = {0, 3, 6};
        config.ghost.resize(3);
        config.boundary.resize(3);
        config.offsets.resize(4);
        // The first two rounded rows sum to DBL_MAX. The third is positive
        // and too small to change the rounded double diagnostic.
        config.masses = {static_cast<real>(0x1.8p+1023),
                         static_cast<real>(0x1.7ffffffffffffp+1023), real{1}};

        Participant participant(config);
        Simulation simulation(participant);
        prepare(simulation);
        CHECK(simulation.advance(0.125).ok());
        const auto accepted = require(participant.snapshot());
        CHECK(accepted.stamp.accepted && accepted.density_valid);
        CHECK(accepted.density(2) > 0);
        CHECK(accepted.owned_density_sum == std::numeric_limits<double>::max());
        CHECK(simulation.finalize().ok());
    }
}

void rounded_reduction_depends_on_row_order() {
    if constexpr (std::is_same_v<real, double>) {
        Config config;
        for (unsigned i = 0; i < 13; ++i) {
            config.row_ids.emplace_back(i + 1);
            config.positions.push_back(static_cast<real>(3 * i));
        }
        config.column_ids = config.row_ids;
        config.ghost.resize(13);
        config.boundary.resize(13);
        config.offsets.resize(14);
        config.masses.assign(11, static_cast<real>(0x1.2p+971));
        config.masses.resize(13, static_cast<real>(0x1.7fffffffffff8p+1023));

        Participant participant(config);
        Simulation simulation(participant);
        prepare(simulation);
        CHECK(simulation.advance(0.125).ok());
        const auto first = require(participant.snapshot());
        CHECK(first.owned_density_sum == static_cast<real>(0x1.ffffffffffffep+1023));

        auto reversed = config.row_ids;
        std::ranges::reverse(reversed);
        CHECK(participant.reorder(reversed, config.column_ids, first.stamp).ok());
        const auto reordered = require(participant.snapshot());
        CHECK(reordered.owned_density_sum == first.owned_density_sum);
        // Starting with the two largest rows makes the same rounded row values
        // overflow the diagnostic during the next Serial evaluation.
        CHECK(simulation.advance(0.125).code() == StatusCode::numerical_failure);
        CHECK(simulation.state() == SimulationState::prepared);
        CHECK(simulation.time().accepted_steps == 1);
        check_same_accepted(reordered, require(participant.snapshot()));
        CHECK(simulation.finalize().ok());
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        Kokkos::ScopeGuard runtime(argc, argv);
        support_edge_precision();
        frozen_layout_and_reorder();
        restore_retry_and_replay();
        densityless_layout_incarnations();
        retained_density_budget();
        retained_layout_budget();
        long_and_empty_rows();
        invalid_input_and_nonfinite_reduction();
        finite_rounded_reduction_at_range_limit();
        rounded_reduction_depends_on_row_order();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
