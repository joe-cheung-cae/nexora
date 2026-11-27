#include "wcsph_oracle.hpp"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <new>
#include <optional>
#include <methods/sph/PrescribedMovingWall3D.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <participants/sph/WcsphParticipant3D.hpp>
#include <participants/sph/WcsphStep3D.hpp>
#include <prescribed_motion/PrescribedMotion.hpp>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace {

struct HostAllocationFault {
    std::size_t fail_at{}, calls{};
    bool active{}, threw{};
    const std::unique_ptr<nexora::sph::CellGrid>* committed_grid{};
};

thread_local HostAllocationFault host_fault;

void* allocate_test_storage(std::size_t bytes) {
    if (host_fault.active && !*host_fault.committed_grid && ++host_fault.calls == host_fault.fail_at) {
        host_fault.active = false;
        host_fault.threw = true;
        throw std::bad_alloc{};
    }
    for (;;) {
        if (void* allocation = std::malloc(bytes ? bytes : 1))
            return allocation;
        const auto handler = std::get_new_handler();
        if (!handler)
            throw std::bad_alloc{};
        handler();
    }
}

} // namespace

void* operator new(std::size_t bytes) { return allocate_test_storage(bytes); }
void* operator new[](std::size_t bytes) { return allocate_test_storage(bytes); }
void operator delete(void* allocation) noexcept { std::free(allocation); }
void operator delete[](void* allocation) noexcept { std::free(allocation); }
void operator delete(void* allocation, std::size_t) noexcept { std::free(allocation); }
void operator delete[](void* allocation, std::size_t) noexcept { std::free(allocation); }

namespace nexora::sph {

struct WcsphTestAccess {
    struct PublishedIdentity {
        FieldStamp state_stamp;
        double maximum_step;
        std::array<std::uintptr_t, 5> allocation;
        std::array<FieldStamp, 4> grant;
        bool operator==(const PublishedIdentity&) const = default;
    };

    static PublishedIdentity published_identity(const WcsphState3D& state) {
        const auto address = [](const auto& view) {
            return reinterpret_cast<std::uintptr_t>(view.data());
        };
        return {state.stamp,
                state.maximum_step,
                {address(state.density->view()), address(state.pressure->view()),
                 address(state.acceleration->view()), address(state.velocity->view()),
                 address(state.geometry.owned_row_to_input)},
                {state.density->stamp(), state.pressure->stamp(), state.acceleration->stamp(),
                 state.velocity->stamp()}};
    }

    static const auto& committed_grid(const WcsphParticipant3D& participant) {
        return participant.step_->committed_grid_;
    }

    static PublishedIdentity accepted_identity(const WcsphParticipant3D& participant) {
        return published_identity(*participant.accepted_);
    }

    static auto reserve_test_budget(const WcsphParticipant3D& participant, std::size_t bytes) {
        return participant.budget_.reserve(bytes);
    }

    static FieldStamp stamp(const WcsphParticipant3D& participant) {
        return participant.accepted_->stamp;
    }
    static double maximum_step(const WcsphParticipant3D& participant) {
        return participant.accepted_->maximum_step;
    }
    static std::vector<LocalIndex> row_to_input(const WcsphParticipant3D& participant) {
        const auto& rows = participant.accepted_->geometry.owned_row_to_input;
        std::vector<LocalIndex> result;
        for (std::size_t row = 0; row < rows.extent(0); ++row)
            result.push_back(rows(row));
        return result;
    }
    static std::array<std::vector<real>, 2> rates(const WcsphParticipant3D& participant) {
        const auto& state = *participant.accepted_;
        std::array<std::vector<real>, 2> result;
        for (std::size_t row = 0; row < state.history.density_rate.extent(0); ++row)
            result[0].push_back(state.history.density_rate(row));
        for (std::size_t row = 0; row < state.density_rate.extent(0); ++row)
            result[1].push_back(state.density_rate(row));
        return result;
    }
    struct WallGroupValue {
        double initial_phase, elapsed, maximum_step;
        friend bool operator==(const WallGroupValue&, const WallGroupValue&) = default;
    };
    static std::map<std::uint64_t, WallGroupValue>
    wall_groups(const WcsphParticipant3D& participant) {
        const auto& group_states = participant.accepted_->wall_phases;
        std::map<std::uint64_t, WallGroupValue> result;
        for (std::size_t i = 0; i < group_states.extent(0); ++i) {
            const auto& group = group_states(i);
            result.emplace(group.group_id, WallGroupValue{group.time.initial_phase, group.time.elapsed,
                                                         group.maximum_step});
        }
        return result;
    }
};

} // namespace nexora::sph

namespace {
using namespace nexora;
using namespace sph_test;
using sph::WcsphTestAccess;

using NeighborPairs = std::vector<std::pair<std::uint64_t, double>>;

struct StageSample {
    std::vector<Particle> particles;
    std::vector<std::uint64_t> owned;
    std::map<std::uint64_t, NeighborPairs> neighbors;
};

StageSample committed_grid_sample(const sph::WcsphParticipant3D& participant) {
    const auto& grid = WcsphTestAccess::committed_grid(participant);
    if (!grid)
        throw std::logic_error("expected a retained WCSPH grid");
    const auto query = grid->neighbors();
    StageSample result;
    const auto& fields = query.fields();
    for (std::size_t slot = 0; slot < fields.ids.extent(0); ++slot) {
        const auto id = fields.ids(slot);
        result.particles.push_back(
            {{fields.position(slot, 0), fields.position(slot, 1), fields.position(slot, 2)},
             fields.mass(slot), id, fields.ghost(slot)});
        if (fields.ghost(slot)) {
            auto& pairs = result.neighbors[id];
            query.for_each_neighbor_of_slot(static_cast<LocalIndex>(slot),
                [&](LocalIndex other, double distance) {
                    pairs.emplace_back(fields.ids(other), distance);
                });
        }
    }
    const auto owned = query.owned_slots();
    for (std::size_t row = 0; row < owned.extent(0); ++row) {
        const auto id = fields.ids(owned(row));
        result.owned.push_back(id);
        auto& pairs = result.neighbors[id];
        query.for_each_neighbor(row, [&](LocalIndex other, double distance) {
            pairs.emplace_back(fields.ids(other), distance);
        });
    }
    return result;
}

struct CommittedSample {
    FieldStamp stamp;
    std::vector<sph::CommittedRow3D> rows;
};

CommittedSample committed(const sph::WcsphParticipant3D& participant) {
    CommittedSample result{WcsphTestAccess::stamp(participant), {}};
    const auto status = participant.visit_committed_rows(
        [&](sph::CommittedRow3D row) { result.rows.push_back(row); });
    if (!status.ok())
        throw std::runtime_error(status.message());
    return result;
}

void expect_same_rows(const CommittedSample& actual, const CommittedSample& expected) {
    ASSERT_EQ(actual.rows.size(), expected.rows.size());
    for (std::size_t i = 0; i < actual.rows.size(); ++i)
        EXPECT_EQ(actual.rows[i], expected.rows[i]) << "row " << i << ", expected ID " << expected.rows[i].id;
}

void expect_same_sample(const CommittedSample& actual, const CommittedSample& expected) {
    EXPECT_EQ(actual.stamp, expected.stamp);
    expect_same_rows(actual, expected);
}

// The shared force oracle consumes independent host values and neighbor data
// read from the committed grid.
struct ForceObservation {
    const CommittedSample& fields;
    const StageSample& stage;
    std::size_t row_count() const { return fields.rows.size(); }
    std::uint64_t row_id(std::size_t row) const { return fields.rows[row].id; }
    real density(std::size_t row) const { return fields.rows[row].density; }
    Result<real> pressure(std::size_t row) const { return fields.rows[row].pressure; }
    Result<std::array<real, 3>> pressure_acceleration(std::size_t row) const {
        return fields.rows[row].pressure_acceleration;
    }
    template <class Visit> void for_each_neighbor(std::size_t row, Visit&& visit) const {
        for (const auto& [id, distance] : stage.neighbors.at(row_id(row)))
            visit(id, distance);
    }
};

using Velocities = std::map<std::uint64_t, std::array<real, 3>>;
struct FluidSample {
    long double density{}, density_rate{};
    std::array<long double, 3> velocity{}, acceleration{};
};
using FluidSamples = std::map<std::uint64_t, FluidSample>;

struct Trajectory {
    std::vector<Particle> particles;
    Velocities velocity;
    std::map<std::uint64_t, long double> density{};
    FluidSamples previous{};
    double previous_dt{};
    int centered_steps{};
};
sph::Velocity3D input_velocity(const Trajectory& state) {
    sph::Velocity3D result("test.initial_velocity", state.particles.size());
    for (std::size_t i = 0; i < state.particles.size(); ++i)
        for (int k = 0; k < 3; ++k)
            result(i, k) = state.velocity.at(state.particles[i].id)[k];
    return result;
}
sph::Values seed_density(const std::vector<Particle>& particles) {
    const auto sum = explicit_csr(particles);
    sph::Values density("test.initial_density", particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i)
        density(i) = static_cast<real>(sum.density.at(particles[i].id));
    return density;
}
long double density_rate(const Particle& particle, const std::vector<Particle>& cloud,
                         const Velocities& velocity) {
    long double sum{};
    for (const auto& other : cloud) {
        std::array<long double, 3> delta;
        long double dot{};
        for (int k = 0; k < 3; ++k) {
            delta[k] = static_cast<long double>(particle.position[k]) - other.position[k];
            dot +=
                (static_cast<long double>(velocity.at(particle.id)[k]) - velocity.at(other.id)[k]) *
                delta[k];
        }
        const long double r = std::hypot(delta[0], delta[1], delta[2]);
        if (r > 0 && r < 2) {
            const auto radial = r < 1 ? (2.25L * r - 3) : -.75L * (2 - r) * (2 - r) / r;
            sum += other.mass * radial * dot / std::acos(-1.L);
        }
    }
    return sum;
}
// All-pairs reference in stable ID order. Integrate the linear interpolant
// of the two latest derivatives for width changes and periodic recoupling.
Trajectory advance_reference(const Trajectory& state, const FluidSamples& current, double dt) {
    Trajectory next = state;
    const bool centered = dt == state.previous_dt && state.centered_steps < 39;
    for (auto& p : next.particles) {
        const auto& now = current.at(p.id);
        const auto& old = state.previous.empty() ? now : state.previous.at(p.id);
        const auto integrate = [&](long double y, long double previous, long double f, long double f0) {
            if (centered)
                return previous + 2 * dt * f;
            const auto slope = state.previous_dt ? (f - f0) / state.previous_dt : 0.L;
            return y + dt * (f + .5L * dt * slope);
        };
        next.density[p.id] = static_cast<real>(
            integrate(now.density, old.density, now.density_rate, old.density_rate));
        for (int axis = 0; axis < 3; ++axis) {
            p.position[axis] += static_cast<long double>(dt) *
                                (now.velocity[axis] + .5L * dt * now.acceleration[axis]);
            next.velocity[p.id][axis] = static_cast<real>(integrate(
                now.velocity[axis], old.velocity[axis], now.acceleration[axis], old.acceleration[axis]));
        }
    }
    next.previous = current;
    next.previous_dt = dt;
    next.centered_steps = centered ? state.centered_steps + 1 : 0;
    return next;
}

Trajectory all_pairs_step(const Trajectory& accepted, sph::WcsphParameters eos, double dt) {
    const auto force = force_oracle(accepted.particles, eos, accepted.density);
    FluidSamples current;
    for (const auto& p : accepted.particles) {
        auto& now = current[p.id];
        now.density = static_cast<real>(force.csr.density.at(p.id));
        now.density_rate = density_rate(p, accepted.particles, accepted.velocity);
        now.acceleration = force.acceleration.at(p.id);
        std::copy_n(accepted.velocity.at(p.id).begin(), 3, now.velocity.begin());
    }
    return advance_reference(accepted, current, dt);
}
struct TrajectoryErrors {
    long double position{}, velocity{}, density{}, pressure{};
    std::array<long double, 3> acceleration{};
};
void check_trajectory(const CommittedSample& sample, const Trajectory& expected,
                      sph::WcsphParameters eos, std::size_t steps,
                      TrajectoryErrors* maximum = nullptr,
                      const sph::WcsphParticipant3D* participant = nullptr) {
    // Check fields against the oracle at the observed geometry, avoiding force
    // cancellation bounds that pretend accumulated trajectory roundoff is zero.
    auto observed = expected.particles;
    const auto independent_force = force_oracle(expected.particles, eos, expected.density);
    const long double tolerance = 256 * (steps + 1) * std::numeric_limits<real>::epsilon();
    std::vector<std::uint64_t> seen;
    for (const auto& row : sample.rows) {
        const auto id = row.id;
        seen.push_back(id);
        const auto source = std::find_if(expected.particles.begin(), expected.particles.end(),
                                         [&](const auto& p) { return p.id == id; });
        ASSERT_NE(source, expected.particles.end());
        EXPECT_EQ(row.mass, source->mass);
        const auto position = row.position;
        const auto velocity = row.velocity;
        for (int k = 0; k < 3; ++k) {
            EXPECT_LE(std::abs(static_cast<long double>(position[k]) - source->position[k]),
                      tolerance *
                          std::max(1.0L, std::abs(static_cast<long double>(source->position[k]))));
            EXPECT_LE(std::abs(static_cast<long double>(velocity[k]) - expected.velocity.at(id)[k]),
                      tolerance * std::max(1.0L, std::abs(static_cast<long double>(
                                                     expected.velocity.at(id)[k]))));
        }
        const auto pressure = row.pressure;
        const auto acceleration = row.pressure_acceleration;
        const long double density_error = std::abs(static_cast<long double>(row.density) -
                                                   independent_force.csr.density.at(id));
        const long double pressure_error =
            std::abs(static_cast<long double>(pressure) - independent_force.pressure.at(id));
        EXPECT_LE(density_error, tolerance * independent_force.csr.density.at(id));
        EXPECT_LE(pressure_error, tolerance * independent_force.pressure_scale.at(id));
        for (int k = 0; k < 3; ++k) {
            const long double acceleration_error =
                std::abs(static_cast<long double>(acceleration[k]) -
                         independent_force.acceleration.at(id)[k]);
            EXPECT_LE(acceleration_error,
                      4 * tolerance * independent_force.acceleration_scale.at(id)[k]);
            if (maximum) {
                maximum->position =
                    std::max(maximum->position,
                             std::abs(static_cast<long double>(position[k]) - source->position[k]));
                maximum->velocity =
                    std::max(maximum->velocity, std::abs(static_cast<long double>(velocity[k]) -
                                                         expected.velocity.at(id)[k]));
                maximum->acceleration[k] = std::max(maximum->acceleration[k], acceleration_error);
            }
        }
        if (maximum) {
            maximum->density = std::max(maximum->density, density_error);
            maximum->pressure = std::max(maximum->pressure, pressure_error);
        }
        observed[std::distance(expected.particles.begin(), source)].position = position;
    }
    ASSERT_EQ(sample.rows.size(), expected.particles.size());
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
    if (!participant)
        return;
    const auto stage = committed_grid_sample(*participant);
    ASSERT_NO_FATAL_FAILURE(
        check_force_snapshot(ForceObservation{sample, stage}, observed, eos, expected.density));
    // Neighbor identity must also match the independently evolved geometry.
    const auto csr = explicit_csr(expected.particles);
    for (const auto& row : sample.rows) {
        const auto index = csr.row_by_id.at(row.id);
        std::vector<std::uint64_t> neighbors;
        for (const auto& [id, distance] : stage.neighbors.at(row.id))
            neighbors.push_back(id);
        std::sort(neighbors.begin(), neighbors.end());
        EXPECT_EQ(neighbors,
                  (std::vector<std::uint64_t>(csr.ids.begin() + csr.offsets[index],
                                              csr.ids.begin() + csr.offsets[index + 1])));
    }
}
TEST(SphVerlet, ConnectedThreeDimensionalCloudChangesNeighborsAcrossSteps) {
    const sph::WcsphVerletParameters parameters{{real{.75}, real{.5}, real{3}}, .5};
    constexpr std::array input_order{5, 0, 6, 3, 7, 1, 4, 2};
    Trajectory expected;
    for (const int corner : input_order) {
        Particle particle{};
        particle.id = 100 + 17 * corner;
        particle.mass = real{1} + real{.125} * corner;
        auto& velocity = expected.velocity[particle.id];
        for (int axis = 0; axis < 3; ++axis) {
            const bool upper = (corner >> axis) & 1;
            particle.position[axis] = upper ? real{.875} : real{-.375};
            const real speed = real{.375} + real{.015625} * (corner + axis);
            velocity[axis] = upper ? -speed : speed;
        }
        expected.particles.push_back(particle);
    }
    const auto initial_csr = explicit_csr(expected.particles);
    const auto first_id = expected.particles.front().id;
    const auto first_row = initial_csr.row_by_id.at(first_id);
    ASSERT_GE(initial_csr.offsets[first_row + 1] - initial_csr.offsets[first_row], 3);
    const auto initial_force = force_oracle(expected.particles, parameters.eos);
    for (const auto acceleration : initial_force.acceleration.at(first_id))
        ASSERT_NE(acceleration, 0);

    Fields fields(expected.particles);
    const auto velocity = input_velocity(expected);
    sph::WcsphParticipant3D participant(fields.get(), 1000, parameters, velocity,
                                   seed_density(expected.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_NO_FATAL_FAILURE(
        check_trajectory(committed(participant), expected, parameters.eos, 0,
                         nullptr, &participant));

    bool neighbors_changed = false;
    constexpr double dt = .04;
    for (std::size_t step = 1; step <= 8; ++step) {
        expected = all_pairs_step(expected, parameters.eos, dt);
        const auto csr = explicit_csr(expected.particles);
        neighbors_changed |= csr.offsets != initial_csr.offsets || csr.ids != initial_csr.ids;
        ASSERT_TRUE(simulation.run(dt).ok());
        ASSERT_NO_FATAL_FAILURE(
            check_trajectory(committed(participant), expected, parameters.eos, step,
                             nullptr, &participant));
    }
    EXPECT_TRUE(neighbors_changed);
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphVerlet, UniformVelocityCrossesCells) {
    const sph::WcsphVerletParameters parameters{{1, .5, 3}, 1};
    Trajectory expected{{{{8, 8, 8}, 1, 99, 0}, {{-.25, .5, .75}, 2, 11, 0}},
                        {{99, {4, 1, -2}}, {11, {4, 1, -2}}}};
    const Fields fields(expected.particles);
    const auto velocity = input_velocity(expected);
    sph::WcsphParticipant3D participant(fields.get(), 10000, parameters, velocity,
                                        seed_density(expected.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const double bound = participant.time_constraint().maximum_step;
    const double expected_bound = 1.0 / (.5 + std::sqrt(21.0));
    EXPECT_NEAR(bound, expected_bound, 4 * std::numeric_limits<real>::epsilon() * expected_bound);
    for (std::size_t step = 1; step <= 6; ++step) {
        expected = all_pairs_step(expected, parameters.eos, .125);
        ASSERT_TRUE(simulation.run(.125).ok());
        const auto current = committed(participant);
        ASSERT_NO_FATAL_FAILURE(check_trajectory(current, expected, parameters.eos, step, nullptr,
                                                 &participant));
        for (const auto& row : current.rows)
            EXPECT_EQ(row.velocity, (std::array<real, 3>{4, 1, -2}));
    }
    ASSERT_TRUE(simulation.finalize().ok());
}
Trajectory pair_state() {
    return {{{{static_cast<real>(.05), static_cast<real>(.05), static_cast<real>(.05)}, 1, 91, 0},
             {{static_cast<real>(.8), static_cast<real>(.55), static_cast<real>(.3)}, 2, 12, 0}},
            {{91, {0, 0, 0}}, {12, {0, 0, 0}}}};
}
TEST(SphVerlet, UnequalMassTrajectoryMatchesIndependentCsrAndMomentum) {
    const sph::WcsphVerletParameters parameters{{real{.125}, real{.5}, real{3}}, .5};
    auto expected = pair_state();
    Fields fields(expected.particles);
    auto velocity = input_velocity(expected);
    sph::WcsphParticipant3D participant(fields.get(), 1000, parameters, velocity,
                                   seed_density(expected.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto initial = committed(participant);
    TrajectoryErrors maximum;
    const double initial_cfl = participant.time_constraint().maximum_step;
    // Step 41 rejoins the parity branches. The following requests exercise
    // changing widths and then returning to a centered step.
    for (std::size_t step = 1; step <= 45; ++step) {
        const double dt = step <= 41 ? .02 : step == 42 ? .015 : step == 43 ? .01 : .0125;
        const auto next = all_pairs_step(expected, parameters.eos, dt);
        ASSERT_TRUE(simulation.run(dt).ok());
        const auto snapshot = committed(participant);
        ASSERT_NO_FATAL_FAILURE(check_trajectory(snapshot, next, parameters.eos, step, &maximum,
                                                 &participant));
        std::array<long double, 3> momentum{}, absolute{};
        for (const auto& row : snapshot.rows) {
            const auto mass = row.mass;
            const auto v = row.velocity;
            for (int k = 0; k < 3; ++k) {
                momentum[k] += mass * static_cast<long double>(v[k]);
                absolute[k] += mass * std::abs(static_cast<long double>(v[k]));
            }
        }
        for (int k = 0; k < 3; ++k)
            EXPECT_LE(std::abs(momentum[k]),
                      64 * (step + 1) * std::numeric_limits<real>::epsilon() * absolute[k]);
        long double vmax = 0;
        for (const auto& row : snapshot.rows) {
            const auto v = row.velocity;
            long double square = 0;
            for (const real component : v)
                square += static_cast<long double>(component) * component;
            vmax = std::max(vmax, std::sqrt(square));
        }
        const double expected_cfl = parameters.cfl / (parameters.eos.sound_speed + vmax);
        EXPECT_NEAR(participant.time_constraint().maximum_step,
                    expected_cfl, 16 * std::numeric_limits<real>::epsilon() * expected_cfl);
        expected = next;
    }
    const auto latest = committed(participant);
    EXPECT_GT(latest.stamp.layout.value(), initial.stamp.layout.value());
    EXPECT_LT(participant.time_constraint().maximum_step,
              initial_cfl);
    const auto record = [&](const char* key, long double value) {
        std::ostringstream text;
        text << std::scientific << std::setprecision(17) << value;
        ::testing::Test::RecordProperty(key, text.str());
    };
    record("trajectory_max_position_error", maximum.position);
    record("trajectory_max_velocity_error", maximum.velocity);
    record("trajectory_max_density_error", maximum.density);
    record("trajectory_max_pressure_error", maximum.pressure);
    record("trajectory_max_ax_error", maximum.acceleration[0]);
    record("trajectory_max_ay_error", maximum.acceleration[1]);
    record("trajectory_max_az_error", maximum.acceleration[2]);
    ASSERT_TRUE(simulation.finalize().ok());
}
TEST(SphVerlet, GridCapacityFailurePreservesRowsAndTime) {
    const sph::WcsphVerletParameters parameters{{1, .25, 3}, .5};
    const Trajectory initial{{{{.125, 0, 0}, 1, 11, 0}, {{3.875, 0, 0}, 1, 22, 0}},
                             {{11, {-.5, 0, 0}}, {22, {.5, 0, 0}}}};
    Fields fields(initial.particles);
    sph::WcsphParticipant3D participant(fields.get(), 2, parameters, input_velocity(initial),
                                   seed_density(initial.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto before = committed(participant);
    const auto bytes = participant.resource_usage().controlled_bytes;
    const auto limit = participant.time_constraint().maximum_step;
    ASSERT_GE(limit, .5);
    // Both particles cross the outer grid boundary. Their new box spans four cells.
    const auto failed = simulation.run(.5);
    EXPECT_EQ(failed.code(), StatusCode::resource_exhausted);
    EXPECT_EQ(simulation.time().accepted_steps, 0u);
    EXPECT_EQ(simulation.time().accepted_time, 0);
    EXPECT_EQ(participant.time_constraint().maximum_step, limit);
    EXPECT_EQ(participant.resource_usage().controlled_bytes, bytes);
    const auto after = committed(participant);
    expect_same_sample(after, before);
    ASSERT_NO_FATAL_FAILURE(check_trajectory(after, initial, parameters.eos, 0));
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

TEST(SphVerlet, DriftOverflowPreservesRowsAndTime) {
    Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const real gravity = std::numeric_limits<real>::max() / 4;
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, 1, {gravity, 0, 0}, 8};
    sph::Velocity3D velocity("test.velocity", 1);
    sph::Values density("test.density", 1);
    density(0) = 1;
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto before = committed(participant);
    ASSERT_EQ(simulation.run(4).code(), StatusCode::numerical_failure);
    expect_same_sample(committed(participant), before);
    EXPECT_EQ(simulation.time().accepted_time, 0);
    ASSERT_TRUE(simulation.finalize().ok());
}

#if !NEXORA_ENABLE_DOUBLE_PRECISION
TEST(SphVerlet, DurationOutsideSolverRangePreservesRowsAndTime) {
    const Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const sph::Velocity3D velocity("test.velocity", 1);
    constexpr real smallest = std::numeric_limits<real>::denorm_min();
    const double too_small = static_cast<double>(smallest) / 4;
    constexpr double largest = std::numeric_limits<real>::max();
    const double too_large = std::nextafter(largest, std::numeric_limits<double>::infinity());
    ASSERT_GT(too_small, 0);
    ASSERT_EQ(static_cast<real>(too_small), 0);
    for (double dt : {too_small, too_large}) {
        sph::WcsphParticipant3D participant(fields.get(), 1, {{1, smallest, 1}, 1}, velocity, fields.mass);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_EQ(participant.time_constraint().maximum_step, largest);
        const auto before = committed(participant);
        const auto usage = participant.resource_usage();
        EXPECT_EQ(simulation.run(dt).code(), StatusCode::invalid_argument);
        EXPECT_EQ(simulation.time().accepted_steps, 0u);
        EXPECT_EQ(simulation.time().accepted_time, 0);
        EXPECT_EQ(participant.resource_usage().controlled_bytes, usage.controlled_bytes);
        expect_same_sample(committed(participant), before);
        ASSERT_TRUE(simulation.finalize().ok());
    }
}
#endif

TEST(SphVerlet, ChecksInitialVelocityDensityCflAndOwnershipBeforePublishing) {
    const sph::WcsphVerletParameters parameters{{1, 1, 3}, .5};
    Fields fields({{{0, 0, 0}, 1, 1, 0}});
    sph::Velocity3D velocity("test.initial_velocity", 1);
    sph::Values density("test.initial_density", 1);
    density(0) = 1;
    const auto rejects = [&](sph::WcsphVerletParameters p, sph::Velocity3D v, sph::Values rho,
                             StatusCode reason) {
        sph::WcsphParticipant3D participant(fields.get(), 1, p, v, rho);
        Simulation simulation(participant);
        EXPECT_EQ(simulation.initialize().code(), reason);
        const auto visit = participant.visit_committed_rows([](sph::CommittedRow3D) {
            ADD_FAILURE() << "Rejected initialization published a fluid row";
        });
        EXPECT_EQ(visit.code(), StatusCode::invalid_state);
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
        EXPECT_TRUE(simulation.finalize().ok());
    };
    for (real cfl : {real{0}, real{1.001}, std::numeric_limits<real>::quiet_NaN()}) {
        SCOPED_TRACE(cfl);
        rejects({parameters.eos, cfl}, velocity, density, StatusCode::invalid_argument);
    }
#if !NEXORA_ENABLE_DOUBLE_PRECISION
    rejects({{1, 2, 3}, std::numeric_limits<real>::denorm_min()}, velocity, density,
            StatusCode::numerical_failure);
#endif
    rejects(parameters, {}, density, StatusCode::invalid_argument);
    rejects(parameters, velocity, {}, StatusCode::invalid_argument);
    const sph::Values oversized_density("test.oversized_density", 2);
    Kokkos::deep_copy(oversized_density, real{1});
    rejects(parameters, velocity, oversized_density, StatusCode::invalid_argument);
    velocity(0, 2) = std::numeric_limits<real>::quiet_NaN();
    rejects(parameters, velocity, density, StatusCode::invalid_argument);
    // The component is finite, while its squared speed exceeds real range.
    velocity(0, 2) = std::ldexp(real{1}, std::numeric_limits<real>::max_exponent / 2);
    rejects(parameters, velocity, density, StatusCode::invalid_argument);
    velocity(0, 2) = 0;
    for (const real invalid : {real{0}, -real{1}, std::numeric_limits<real>::infinity()}) {
        density(0) = invalid;
        rejects(parameters, velocity, density, StatusCode::numerical_failure);
    }
    density(0) = 1;
    fields.ghost(0) = 1;
    rejects(parameters, velocity, density, StatusCode::unsupported);
}

TEST(SphVerlet, TimeConstraintRequiresAnAcceptedState) {
    const Fields fields({{{0, 0, 0}, 1, 1, 0}});
    const sph::Velocity3D velocity("test.velocity", 1);
    const sph::Values density("test.density", 1);
    density(0) = 0;
    sph::WcsphParticipant3D participant(fields.get(), 1, {{1, 1, 3}, .5}, velocity, density);
    const auto expect_unavailable = [&] {
        EXPECT_FALSE(participant.time_constraint().available);
        EXPECT_EQ(participant.step({0, 2, 2}).code(), StatusCode::invalid_state);
    };
    expect_unavailable();
    ASSERT_EQ(participant.initialize(0).code(), StatusCode::numerical_failure);
    expect_unavailable();
    density(0) = 1;
    ASSERT_TRUE(participant.initialize(0).ok());
    EXPECT_TRUE(participant.time_constraint().available);
    EXPECT_DOUBLE_EQ(participant.time_constraint().maximum_step, .5);
    ASSERT_TRUE(participant.finalize().ok());
    expect_unavailable();
}

TEST(SphVerlet, DirectStepRequiresTheAcceptedBeginAndMatchingEnd) {
    const sph::WcsphVerletParameters parameters{{1, 1, 3}, .5};
    const Fields fields({{{0, 0, 0}, 1, 1, 0}});
    sph::Velocity3D velocity("test.velocity", 1);
    velocity(0, 0) = .125;
    sph::Values density("test.density", 1);
    density(0) = 1;
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density);
    ASSERT_TRUE(participant.initialize(1).ok());
    const TimeInterval interval{1, 1.125, .125};
    const auto before = committed(participant);
    for (const auto invalid : {TimeInterval{1.25, 1.375, .125},
                               TimeInterval{1, 1.126, .125}})
        EXPECT_EQ(participant.step(invalid).code(), StatusCode::invalid_argument);
    expect_same_sample(committed(participant), before);
    ASSERT_TRUE(participant.step(interval).ok());
    const auto after = committed(participant);
    EXPECT_EQ(after.stamp.time.physical_time, 1.125);
    EXPECT_EQ(after.rows.at(0).position[0], .015625);
    EXPECT_EQ(after.rows.at(0).velocity, before.rows.at(0).velocity);
    ASSERT_TRUE(participant.finalize().ok());
}

void expect_metadata(const sph::WcsphParticipant3D& participant, const Simulation& simulation,
                     double time, std::size_t steps, double maximum_step) {
    const auto stamp = WcsphTestAccess::stamp(participant);
    EXPECT_TRUE(stamp.accepted);
    EXPECT_EQ(stamp.time.physical_time, time);
    EXPECT_EQ(stamp.valid_from, time);
    EXPECT_EQ(stamp.valid_until, time);
    EXPECT_EQ(stamp.accepted_version.value(), steps);
    EXPECT_EQ(simulation.time().accepted_time, time);
    EXPECT_EQ(simulation.time().accepted_steps, steps);
    EXPECT_DOUBLE_EQ(WcsphTestAccess::maximum_step(participant), maximum_step);
    EXPECT_DOUBLE_EQ(participant.time_constraint().maximum_step, maximum_step);
}

TEST(SphVerlet, AcceptedAcousticLimitAppliesToTheNextStep) {
    const Fields fluid({{{0, 0, 0}, 1, 11, 0}});
    const sph::Velocity3D velocity("test.acoustic_velocity", 1);
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, .5, {.5, 0, 0}};
    sph::WcsphParticipant3D participant(fluid.get(), 1, parameters, velocity, fluid.mass);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    expect_metadata(participant, simulation, 0., 0, .5);

    // The nearest float to 0.4 is above the physical CFL bound.
    constexpr double next_bound = std::is_same_v<real, float> ? 0x1.999998p-2 : .4;
    const auto above = std::nextafter(next_bound, std::numeric_limits<double>::infinity());
    const std::array requests{.5, above, next_bound};
    const auto result = simulation.run(requests);
    ASSERT_EQ(result.code(), StatusCode::invalid_argument) << result.message();
    expect_metadata(participant, simulation, .5, 1, next_bound);
    const auto accepted = committed(participant);
    ASSERT_EQ(accepted.rows.size(), 1u);
    EXPECT_EQ(accepted.rows.front().velocity[0], .25);
    EXPECT_EQ(accepted.rows.front().position[0], .0625);
    EXPECT_EQ(accepted.rows.front().density, 1);
    for (const auto& stamp : WcsphTestAccess::accepted_identity(participant).grant) {
        EXPECT_TRUE(stamp.accepted);
        EXPECT_EQ(stamp.accepted_version.value(), 1u);
        EXPECT_EQ(stamp.time.physical_time, .5);
    }
    EXPECT_EQ(simulation.run(next_bound).code(), StatusCode::invalid_state);
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);

    sph::WcsphParticipant3D at_bound(fluid.get(), 1, parameters, velocity, fluid.mass);
    Simulation accepted_simulation(at_bound);
    ASSERT_TRUE(accepted_simulation.initialize().ok());
    ASSERT_TRUE(accepted_simulation.run(.5).ok());
    ASSERT_TRUE(accepted_simulation.run(at_bound.time_constraint().maximum_step).ok());
    EXPECT_EQ(accepted_simulation.time().accepted_time, .5 + next_bound);
    EXPECT_EQ(accepted_simulation.time().accepted_steps, 2u);
    ASSERT_TRUE(accepted_simulation.finalize().ok());
}

TEST(SphVerlet, CommittedMetadataWithNoWallOrRows) {
    constexpr sph::WcsphVerletParameters parameters{{1, 1, 3}, .5};
    for (const bool has_fluid : {false, true}) {
        const Fields fields(has_fluid ? std::vector<Particle>{{{.25, .5, .75}, 1, 5, 0}}
                                      : std::vector<Particle>{});
        sph::Velocity3D velocity("test.initial_velocity", has_fluid ? 1 : 0);
        sph::Values density("test.initial_density", has_fluid ? 1 : 0);
        if (has_fluid) {
            velocity(0, 0) = .25;
            density(0) = 1.5;
        }
        constexpr double fluid_bound = std::is_same_v<real, float> ? 0x1.999998p-2 : .4;
        const double maximum_step = has_fluid ? fluid_bound : .5;
        sph::WcsphParticipant3D participant(fields.get(), has_fluid ? 1 : 0,
                                            parameters, velocity, density);
        Simulation simulation(participant, 2);
        ASSERT_TRUE(simulation.initialize().ok());
        expect_metadata(participant, simulation, 2, 0, maximum_step);
        const auto start = committed(participant);
        EXPECT_EQ(start.rows.size(), has_fluid ? 1u : 0u);
        ASSERT_TRUE(simulation.run(.125).ok());
        expect_metadata(participant, simulation, 2.125, 1, maximum_step);
        const auto end = committed(participant);
        EXPECT_EQ(end.rows.size(), has_fluid ? 1u : 0u);
        EXPECT_EQ(end.stamp.topology, start.stamp.topology);
        EXPECT_EQ(end.stamp.layout, start.stamp.layout);
        EXPECT_EQ(end.stamp.geometry.value(), start.stamp.geometry.value() + (has_fluid ? 1 : 0));
        EXPECT_FALSE(participant.read_committed_wall(5).ok());
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphVerlet, FixedWallOnlyCommitsStaticGeometryAndAcousticLimit) {
    constexpr sph::WcsphVerletParameters parameters{{1, 1, 3}, .5};
    const Fields empty({}), fixed({{{.25, .5, .75}, 1, 19, 1}});
    const sph::Velocity3D velocity("test.empty_velocity", 0);
    const sph::Values density("test.empty_density", 0);
    sph::WcsphParticipant3D participant(
        empty.get(), 1, parameters, velocity, density,
        sph::FixedWallParticles3D{fixed.position, fixed.mass, fixed.ids});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto stamp = WcsphTestAccess::stamp(participant);
    for (std::size_t step = 0; step <= 2; ++step) {
        expect_metadata(participant, simulation, .5 * step, step, .5);
        const auto state = committed(participant);
        EXPECT_TRUE(state.rows.empty());
        const auto wall = checked(participant.read_committed_wall(19));
        EXPECT_EQ(wall.id, 19u);
        EXPECT_EQ(wall.mass, 1);
        EXPECT_EQ(wall.position, (std::array<real, 3>{.25, .5, .75}));
        EXPECT_EQ(wall.stamp, state.stamp);
        EXPECT_EQ(wall.stamp.topology, stamp.topology);
        EXPECT_EQ(wall.stamp.geometry, stamp.geometry);
        EXPECT_EQ(wall.stamp.layout, stamp.layout);
        EXPECT_EQ(wall.phase, 0);
        if (step < 2)
            ASSERT_TRUE(simulation.run(.5).ok());
    }
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

double independent_acoustic_step(const Velocities& velocity,
                                 const sph::WcsphVerletParameters& parameters) {
    double maximum_speed{};
    for (const auto& [id, v] : velocity)
        maximum_speed = std::max(maximum_speed, std::hypot(double(v[0]), double(v[1]), double(v[2])));
    return parameters.cfl * parameters.smoothing_length /
           (parameters.eos.sound_speed + maximum_speed);
}

TEST(SphVerlet, InitializationMapsExplicitValuesAcrossInputPermutations) {
    const sph::WcsphVerletParameters parameters{{1, .25, 3}, .5};
    const Trajectory initial{
        {{{2.25, .25, .5}, 1.25, 91, 0}, {{.25, .375, .5}, .75, 12, 0},
         {{1.25, .25, .375}, 1.5, 63, 0}},
        {{91, {1, .125, -.25}}, {12, {-.5, .25, .875}}, {63, {.25, -.5, .125}}},
        {{91, 1.25}, {12, .875}, {63, 1.5}}};
    auto reordered = initial;
    std::rotate(reordered.particles.begin(), reordered.particles.begin() + 2,
                reordered.particles.end());
    Fields first(initial.particles), second(reordered.particles);
    auto first_velocity = input_velocity(initial);
    const auto second_velocity = input_velocity(reordered);
    const auto density = [](const Trajectory& state) {
        sph::Values result("test.explicit_density", state.particles.size());
        for (std::size_t i = 0; i < state.particles.size(); ++i)
            result(i) = static_cast<real>(state.density.at(state.particles[i].id));
        return result;
    };
    auto first_density = density(initial);
    sph::WcsphParticipant3D participant(first.get(), 64, parameters, first_velocity, first_density);
    sph::WcsphParticipant3D direct(second.get(), 64, parameters, second_velocity, density(reordered));
    Simulation simulation(participant), direct_sim(direct);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(direct_sim.initialize().ok());
    const auto original = committed(participant);
    expect_same_rows(committed(direct), original);
    const std::vector<LocalIndex> expected_slots{1, 2, 0};
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), expected_slots);
    ASSERT_EQ(original.rows.size(), 3u);
    EXPECT_EQ(original.rows[0].id, 12u);
    EXPECT_EQ(original.rows[1].id, 63u);
    EXPECT_EQ(original.rows[2].id, 91u);
    for (const auto& row : original.rows) {
        EXPECT_EQ(row.density, initial.density.at(row.id));
        EXPECT_EQ(row.velocity, initial.velocity.at(row.id));
    }
    ASSERT_NO_FATAL_FAILURE(check_trajectory(original, initial, parameters.eos, 0,
                                             nullptr, &participant));
    const auto bound = independent_acoustic_step(initial.velocity, parameters);
    EXPECT_NEAR(participant.time_constraint().maximum_step, bound,
                8 * std::numeric_limits<real>::epsilon() * bound);
    EXPECT_DOUBLE_EQ(participant.time_constraint().maximum_step,
                     direct.time_constraint().maximum_step);

    for (std::size_t i = 0; i < initial.particles.size(); ++i) {
        first.position(i, 0) += 10;
        first_density(i) = 9;
        for (int axis = 0; axis < 3; ++axis)
            first_velocity(i, axis) = 10;
    }
    expect_same_rows(committed(participant), original);
    ASSERT_TRUE(simulation.run(.125).ok());
    ASSERT_TRUE(direct_sim.run(.125).ok());
    const auto endpoint = all_pairs_step(initial, parameters.eos, .125);
    const auto current = committed(participant);
    ASSERT_NO_FATAL_FAILURE(check_trajectory(current, endpoint, parameters.eos, 1,
                                             nullptr, &participant));
    expect_same_rows(current, committed(direct));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_TRUE(direct_sim.finalize().ok());
}

class StepAllocations {
  public:
    struct Event {
        bool allocated;
        std::string label;
        const void* address;
        std::uint64_t bytes;
        std::size_t controlled;
    };

    explicit StepAllocations(const sph::WcsphParticipant3D& participant)
        : participant_(participant), previous_(Kokkos::Tools::Experimental::get_callbacks()) {
        active_ = this;
        Kokkos::Tools::Experimental::set_allocate_data_callback(record<true>);
        Kokkos::Tools::Experimental::set_deallocate_data_callback(record<false>);
    }
    ~StepAllocations() {
        Kokkos::Tools::Experimental::set_callbacks(previous_);
        active_ = nullptr;
        EXPECT_FALSE(failed_);
    }
    StepAllocations(const StepAllocations&) = delete;
    StepAllocations& operator=(const StepAllocations&) = delete;

    std::size_t count(std::size_t begin, std::string_view label, bool allocated) const {
        return std::count_if(events.begin() + begin, events.end(), [&](const auto& event) {
            return event.allocated == allocated && event.label == label;
        });
    }

    std::size_t peak(std::size_t first, std::size_t last) const {
        std::size_t result{};
        for (std::size_t i = first; i < last; ++i)
            result = std::max(result, events[i].controlled);
        return result;
    }

    void expect_all_released() const {
        std::map<const void*, const Event*> live;
        for (const auto& event : events) {
            if (event.allocated) {
                EXPECT_TRUE(live.emplace(event.address, &event).second) << event.label;
            } else {
                const auto found = live.find(event.address);
                ASSERT_NE(found, live.end()) << event.label;
                EXPECT_EQ(found->second->bytes, event.bytes) << event.label;
                EXPECT_EQ(found->second->label, event.label);
                live.erase(found);
            }
        }
        EXPECT_TRUE(live.empty());
    }

    std::vector<Event> events;

  private:
    template <bool allocated>
    static void record(Kokkos::Tools::SpaceHandle, const char* label, const void* address,
                       std::uint64_t bytes) noexcept {
        if (!address || !std::string_view(label).starts_with("sph."))
            return;
        try {
            auto& recorder = *active_;
            recorder.events.push_back({allocated, label, address, bytes,
                                       recorder.participant_.resource_usage().controlled_bytes});
        } catch (...) {
            active_->failed_ = true;
        }
    }

    inline static StepAllocations* active_{};
    const sph::WcsphParticipant3D& participant_;
    Kokkos::Tools::Experimental::EventSet previous_;
    bool failed_{};
};

TEST(SphVerlet, ReusedWorkspaceIsReleasedWithoutChangingCommittedRows) {
    const sph::WcsphVerletParameters parameters{{1, .5, 3}, .5};
    const Trajectory initial{{{{.25, .5, .75}, 1, 11, 0}}, {{11, {0, 0, 0}}}};
    const Fields fields(initial.particles);
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, input_velocity(initial),
                                        seed_density(initial.particles));
    StepAllocations observed(participant);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto initialized = observed.events.size();
    ASSERT_TRUE(simulation.run(.125).ok());
    const auto after_cold = observed.events.size();
    const auto warmed_bytes = participant.resource_usage().controlled_bytes;
    EXPECT_EQ(observed.count(initialized, "sph.central_difference.position", true), 1u);
    ASSERT_TRUE(simulation.run(.125, 3).ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, warmed_bytes);
    EXPECT_EQ(observed.count(after_cold, "sph.central_difference.position", true), 0u);

    // Both input orders remain the identity. The immutable committed row table
    // follows the coordinates through all four steps and survives cache eviction.
    EXPECT_EQ(observed.count(initialized, "sph.geometry.owned_row_to_input", true), 0u);
    const auto before = committed(participant);
    const auto rates_before_eviction = WcsphTestAccess::rates(participant);
    const auto release_start = observed.events.size();
    ASSERT_TRUE(participant.release_workspace().ok());
    EXPECT_EQ(observed.count(release_start, "sph.central_difference.position", false), 1u);
    EXPECT_EQ(observed.count(release_start, "sph.grid.cell_begin", false), 1u);
    EXPECT_LT(participant.resource_usage().controlled_bytes, warmed_bytes);
    EXPECT_GT(participant.resource_usage().controlled_bytes, sizeof(LocalIndex));
    const auto read_start = observed.events.size();
    const auto after = committed(participant);
    expect_same_sample(after, before);
    EXPECT_EQ(observed.events.size(), read_start);
    EXPECT_EQ(WcsphTestAccess::rates(participant), rates_before_eviction);
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), (std::vector<LocalIndex>{0}));
    ASSERT_NO_FATAL_FAILURE(check_trajectory(after, initial, parameters.eos, 4));
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0u);
    observed.expect_all_released();
}

TEST(SphVerlet, HistoryAndCandidatesRespectControlledByteLimit) {
    const sph::WcsphVerletParameters parameters{{.375, .5, 3}, .5};
    const Trajectory initial{{{{.25, .5, .75}, 1, 11, 0}, {{1.25, .5, .75}, .75, 22, 0}},
                             {{11, {.125, 0, 0}}, {22, {-.125, 0, 0}}}};
    const Fields fields(initial.particles);
    const auto velocity = input_velocity(initial);
    const auto density = seed_density(initial.particles);
    constexpr std::size_t r = sizeof(real), i = sizeof(LocalIndex);
    constexpr std::size_t historical = 2 * (12 * r + 9 + 3 * i) + 2 * i;
    constexpr std::size_t insufficient = historical + 6 * r;
    constexpr std::size_t limit = historical + 2 * (17 + 9) * r; // Two rows of multistep history.
    constexpr double dt = .125;
    {
        sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density,
                                            ParticipantId{1}, {32, insufficient});
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        const auto before = committed(participant);
        const auto bytes = participant.resource_usage().controlled_bytes;
        EXPECT_EQ(bytes, historical + 2 * i + 2 * r); // Initial continuity rates.
        ::testing::Test::RecordProperty("controlled_initial", bytes);
        const auto maximum_step = participant.time_constraint().maximum_step;
        EXPECT_EQ(simulation.run(dt, 2).code(), StatusCode::resource_exhausted);
        expect_metadata(participant, simulation, 0, 0, maximum_step);
        EXPECT_EQ(participant.resource_usage().controlled_bytes, bytes);
        expect_same_sample(committed(participant), before);
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    }

    {
        sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density,
                                            ParticipantId{1}, {32, limit});
        StepAllocations observed(participant);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        const auto after_initialize = observed.events.size();
        ::testing::Test::RecordProperty("controlled_peak_initialization",
                                        observed.peak(0, after_initialize));
        ASSERT_LT(observed.peak(0, after_initialize), limit);

        auto expected = initial;
        std::size_t after_first_commit{};
        for (int step = 1; step <= 3; ++step) {
            expected = all_pairs_step(expected, parameters.eos, dt);
            ASSERT_TRUE(simulation.run(dt).ok());
            EXPECT_LE(participant.resource_usage().controlled_bytes, limit);
            ASSERT_NO_FATAL_FAILURE(
                check_trajectory(committed(participant), expected, parameters.eos, step));
            if (step == 1)
                after_first_commit = observed.events.size();
        }
        EXPECT_LE(observed.peak(after_initialize, after_first_commit), limit);
        EXPECT_LE(observed.peak(after_first_commit, observed.events.size()), limit);
        ::testing::Test::RecordProperty("controlled_peak_first_step",
                                        observed.peak(after_initialize, after_first_commit));
        ::testing::Test::RecordProperty("controlled_peak_hot_step",
                                        observed.peak(after_first_commit, observed.events.size()));
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
        EXPECT_EQ(participant.resource_usage().snapshot_records, 0u);
        observed.expect_all_released();
    }

    const auto run_at = [&](std::size_t budget) {
        sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density,
                                            ParticipantId{1}, {32, budget});
        Simulation simulation(participant);
        auto status = simulation.initialize();
        if (status.ok())
            status = simulation.run(dt, 3);
        EXPECT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
        return status.code();
    };
    std::size_t minimum = insufficient + 1;
    for (; minimum <= limit; ++minimum)
        if (run_at(minimum) == StatusCode::success)
            break;
    ASSERT_LE(minimum, limit);
    EXPECT_EQ(run_at(minimum), StatusCode::success);
    EXPECT_EQ(run_at(minimum - 1), StatusCode::resource_exhausted);
    ::testing::Test::RecordProperty("minimum_controlled_budget", minimum);
    ::testing::Test::RecordProperty("one_byte_smaller_rejected", minimum - 1);
    ::testing::Test::RecordProperty("controlled_budget_limit", limit);
}

// Independent all-pairs Adami and pressure reference for h=1, gamma=1. Positions,
// densities and pressures use the same public field precision at storage boundaries.
struct FixedWallForce {
    std::map<std::uint64_t, real> pressure, acceleration;
};

FixedWallForce fixed_wall_force(const Trajectory& state, const Particle& wall, double gravity) {
    constexpr double c2 = .25;
    std::map<std::uint64_t, double> coefficient;
    FixedWallForce result;
    long double weighted_pressure{}, weights{};
    for (const auto& fluid : state.particles) {
        const auto rho = static_cast<real>(state.density.at(fluid.id));
        const auto pressure = static_cast<real>(c2 * (rho - 1));
        result.pressure.emplace(fluid.id, pressure);
        coefficient.emplace(fluid.id, pressure / (double(rho) * rho));
        const auto separation = static_cast<long double>(wall.position[0]) - fluid.position[0];
        const auto distance = std::abs(separation);
        if (distance < 2) {
            const auto weight = std::pow(2 - distance, 3) -
                                4 * std::pow(std::max(1 - distance, 0.L), 3);
            weighted_pressure += weight * (pressure + rho * gravity * separation);
            weights += weight;
        }
    }
    const auto wall_pressure = static_cast<real>(weighted_pressure / weights);
    const auto wall_density = static_cast<real>(1.L + wall_pressure / c2);
    coefficient.emplace(wall.id, wall_pressure / (double(wall_density) * wall_density));

    auto sources = state.particles;
    sources.push_back(wall);
    for (const auto& fluid : state.particles) {
        long double acceleration{};
        for (const auto& other : sources) {
            const auto delta = static_cast<long double>(fluid.position[0]) - other.position[0];
            const auto distance = std::abs(delta);
            if (distance > 0 && distance < 2) {
                const long double radial = distance < 1
                    ? 3 - 2.25L * distance
                    : .75L * std::pow(2 - distance, 2) / distance;
                acceleration += other.mass * radial * delta *
                    (coefficient.at(fluid.id) + coefficient.at(other.id)) / std::acos(-1.L);
            }
        }
        result.acceleration.emplace(fluid.id, static_cast<real>(acceleration));
    }
    return result;
}

struct FixedWallStep {
    Trajectory end;
    std::map<std::uint64_t, long double> start_rate, end_rate;
};

FixedWallStep wall_reference_step(const Trajectory& state, const std::map<std::uint64_t, real>& pressure_force,
                                 double gravity, double dt, const std::vector<Particle>& walls,
                                 const Velocities& wall_velocity, const std::vector<Particle>& next_walls,
                                 const Velocities& next_wall_velocity) {
    auto sources = state.particles;
    sources.insert(sources.end(), walls.begin(), walls.end());
    auto velocity = state.velocity;
    velocity.insert(wall_velocity.begin(), wall_velocity.end());
    FluidSamples current;
    FixedWallStep result;
    for (const auto& p : state.particles) {
        auto& now = current[p.id];
        now.velocity[0] = velocity.at(p.id)[0];
        now.acceleration[0] = pressure_force.at(p.id) + gravity;
        now.density = state.density.at(p.id);
        now.density_rate = density_rate(p, sources, velocity);
        result.start_rate[p.id] = now.density_rate;
    }
    result.end = advance_reference(state, current, dt);
    sources = result.end.particles;
    sources.insert(sources.end(), next_walls.begin(), next_walls.end());
    velocity = result.end.velocity;
    velocity.insert(next_wall_velocity.begin(), next_wall_velocity.end());
    for (const auto& p : result.end.particles)
        result.end_rate[p.id] = density_rate(p, sources, velocity);
    return result;
}

FixedWallStep fixed_wall_step(const Trajectory& start, const Particle& wall, double gravity,
                              double dt) {
    const Velocities wall_velocity{{wall.id, {}}};
    return wall_reference_step(start, fixed_wall_force(start, wall, gravity).acceleration, gravity,
                               dt, {wall}, wall_velocity, {wall}, wall_velocity);
}

std::vector<Particle> ordered_with_wall(const Trajectory& state, const Particle& wall) {
    auto sorted = state.particles;
    sorted.push_back(wall);
    const auto key = [](const Particle& particle) {
        const auto& x = particle.position;
        return std::tuple{std::floor(.5 * x[2]), std::floor(.5 * x[1]),
                          std::floor(.5 * x[0]), particle.id};
    };
    std::sort(sorted.begin(), sorted.end(),
              [&](const auto& a, const auto& b) { return key(a) < key(b); });
    return sorted;
}

void expect_stage_query(const StageSample& sampled, const Trajectory& state,
                        const Particle& wall) {
    constexpr double eps = std::numeric_limits<real>::epsilon();
    const auto expected = ordered_with_wall(state, wall);
    ASSERT_EQ(sampled.particles.size(), expected.size());
    ASSERT_EQ(sampled.neighbors.size(), expected.size());
    std::vector<std::uint64_t> expected_owned;
    for (std::size_t slot = 0; slot < expected.size(); ++slot) {
        const auto& self = expected[slot];
        EXPECT_EQ(sampled.particles[slot].id, self.id);
        EXPECT_EQ(sampled.particles[slot].mass, self.mass);
        EXPECT_EQ(sampled.particles[slot].ghost, self.ghost);
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(sampled.particles[slot].position[k], self.position[k], 8 * eps);
        if (!self.ghost)
            expected_owned.push_back(self.id);
        NeighborPairs pairs;
        for (const auto& other : expected) {
            if (self.id == other.id)
                continue;
            double d2{};
            for (int k = 0; k < 3; ++k) {
                const double dx = double(self.position[k]) - other.position[k];
                d2 += dx * dx;
            }
            if (d2 <= 4)
                pairs.emplace_back(other.id, d2);
        }
        const auto& actual = sampled.neighbors.at(self.id);
        ASSERT_EQ(actual.size(), pairs.size());
        for (std::size_t j = 0; j < pairs.size(); ++j) {
            EXPECT_EQ(actual[j].first, pairs[j].first);
            EXPECT_NEAR(actual[j].second, pairs[j].second, 16 * eps);
        }
    }
    EXPECT_EQ(sampled.owned, expected_owned);
}

void expect_fixed_rows(const CommittedSample& current, const Trajectory& expected,
                       const Particle& wall, double gravity) {
    const auto force = fixed_wall_force(expected, wall, gravity);
    constexpr double eps = std::numeric_limits<real>::epsilon();
    ASSERT_EQ(current.rows.size(), expected.particles.size());
    std::size_t row_index{};
    for (const auto& fluid : ordered_with_wall(expected, wall)) {
        if (fluid.ghost)
            continue;
        const auto& row = current.rows[row_index++];
        ASSERT_EQ(row.id, fluid.id);
        EXPECT_EQ(row.mass, fluid.mass);
        EXPECT_NEAR(row.density, expected.density.at(row.id), 16 * eps);
        EXPECT_NEAR(row.pressure, force.pressure.at(row.id), 16 * eps);
        EXPECT_NEAR(row.pressure_acceleration[0], force.acceleration.at(row.id), 32 * eps);
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(row.position[k], fluid.position[k], 8 * eps);
            EXPECT_NEAR(row.velocity[k], expected.velocity.at(row.id)[k], 16 * eps);
        }
        EXPECT_EQ(row.pressure_acceleration[1], 0);
        EXPECT_EQ(row.pressure_acceleration[2], 0);
    }
}

TEST(SphVerlet, CommittedNeighborsAndRatesSurviveGridEviction) {
    constexpr double gravity = .03125, dt = .25;
    const sph::WcsphVerletParameters parameters{{1, .5, 1}, 1, {gravity, 0, 0}};
    const Trajectory initial{
        {{{2.125, 0, 0}, 1.25, 77, 0}, {{1.8125, 0, 0}, .75, 42, 0}},
        {{77, {-1.5, 0, 0}}, {42, {1.25, 0, 0}}}, {{77, 1.5}, {42, 1.25}}};
    const Particle wall{{2, 0, 0}, 2.25, 50, 1};
    const Fields fields(initial.particles), fixed({wall});
    const sph::FixedWallParticles3D boundary{fixed.position, fixed.mass, fixed.ids};
    sph::Values density("test.crossing_density", 2);
    density(0) = 1.5;
    density(1) = 1.25;
    const auto velocity = input_velocity(initial);
    sph::WcsphParticipant3D participant(fields.get(), 64, parameters, velocity, density, boundary);
    Simulation simulation(participant);
    StepAllocations allocations(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto initial_grid = committed_grid_sample(participant);
    expect_stage_query(initial_grid, initial, wall);
    const auto start = committed(participant);
    ASSERT_NO_FATAL_FAILURE(expect_fixed_rows(start, initial, wall, gravity));

    const auto first = fixed_wall_step(initial, wall, gravity, dt);
    ASSERT_TRUE(simulation.run(dt).ok());
    const auto first_grid = committed_grid_sample(participant);
    expect_stage_query(first_grid, first.end, wall);
    const auto rates = WcsphTestAccess::rates(participant);
    ASSERT_EQ(rates[0].size(), 2u);
    ASSERT_EQ(rates[1].size(), 2u);
    constexpr double eps = std::numeric_limits<real>::epsilon();
    for (std::size_t row = 0; row < 2; ++row) {
        EXPECT_NEAR(rates[0][row], first.start_rate.at(initial_grid.owned[row]), 16 * eps);
        EXPECT_NEAR(rates[1][row], first.end_rate.at(first_grid.owned[row]), 16 * eps);
    }
    EXPECT_EQ(initial_grid.owned, (std::vector<std::uint64_t>{42, 77}));
    EXPECT_EQ(first_grid.owned, (std::vector<std::uint64_t>{77, 42}));
    const auto crossed = committed(participant);
    EXPECT_EQ(crossed.stamp.topology, start.stamp.topology);
    EXPECT_EQ(crossed.stamp.geometry.value(), start.stamp.geometry.value() + 1);
    EXPECT_EQ(crossed.stamp.layout.value(), start.stamp.layout.value() + 1);
    ASSERT_NO_FATAL_FAILURE(expect_fixed_rows(crossed, first.end, wall, gravity));
    const auto acoustic = independent_acoustic_step(first.end.velocity, parameters);
    EXPECT_NEAR(participant.time_constraint().maximum_step, acoustic, 16 * eps * acoustic);

    const auto before_release = allocations.events.size();
    const auto warm_bytes = participant.resource_usage().controlled_bytes;
    const auto row_table = WcsphTestAccess::row_to_input(participant);
    ASSERT_EQ(row_table.size(), 2u);
    ASSERT_TRUE(participant.release_workspace().ok());
    EXPECT_EQ(allocations.count(before_release, "sph.grid.cell_begin", false), 1u);
    EXPECT_EQ(allocations.count(before_release, "sph.grid.to_input", false), 1u);
    EXPECT_EQ(allocations.count(before_release, "sph.geometry.owned_row_to_input", false), 0u);
    EXPECT_LT(participant.resource_usage().controlled_bytes, warm_bytes);
    const auto after_release = allocations.events.size();
    const auto readback = committed(participant);
    expect_same_sample(readback, crossed);
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), row_table);
    EXPECT_EQ(allocations.events.size(), after_release);
    ASSERT_NO_FATAL_FAILURE(expect_fixed_rows(readback, first.end, wall, gravity));

    constexpr double next_dt = .125;
    const auto next = fixed_wall_step(first.end, wall, gravity, next_dt);
    ASSERT_TRUE(simulation.run(next_dt).ok());
    EXPECT_GE(allocations.count(after_release, "sph.grid.keys", true), 1u);
    expect_stage_query(committed_grid_sample(participant), next.end, wall);
    const auto published = committed(participant);
    ASSERT_NO_FATAL_FAILURE(expect_fixed_rows(published, next.end, wall, gravity));
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

// The second fluid crosses the x-cell edge on the second step. The wall
// remains outside the fluid support while retaining real committed wall data.
struct M4FixedWallFixture {
    static constexpr double first_step = .125, step = .25;
    Fields fluid{{{{.75, 0, 0}, 1, 10, 0}, {{1.875, 0, 0}, 2, 20, 0}}};
    Fields wall{{{{4, 0, 0}, 3, 30, 1}}};
    sph::Velocity3D velocity{"m4.velocity", 2};
    sph::Values density{"m4.density", 2};

    M4FixedWallFixture() {
        velocity(0, 0) = velocity(1, 0) = .5;
        density(0) = 2;
        density(1) = 3;
    }

    sph::WcsphParticipant3D participant(sph::WcsphParticipant3D::ResourceLimits limits = {}) const {
        return {fluid.get(), 16, {{2, 1, 1}, 1, {}, .5}, velocity, density,
                sph::FixedWallParticles3D{wall.position, wall.mass, wall.ids},
                ParticipantId{1}, limits};
    }
};

TEST(SphVerlet, RegularStepUpdatesOneGridAndEvaluatesEachDerivativeOnce) {
    // Two neighboring fluid rows share one wall in support; the second wall is out of range.
    const Fields fluid({{{.25, 0, 0}, 1, 10, 0}, {{.75, 0, 0}, 1, 20, 0}});
    const Fields walls({{{1.25, 0, 0}, 1, 31, 1}, {{5, 0, 0}, 1, 32, 1}});
    sph::Velocity3D velocity("probe.velocity", 2);
    velocity(0, 0) = .03125;
    velocity(1, 0) = -.015625;
    sph::WcsphParticipant3D participant(fluid.get(), 32, {{1, .5, 1}, .5}, velocity, fluid.mass,
                                       sph::FixedWallParticles3D{walls.position, walls.mass, walls.ids});
    constexpr double dt = .03125;
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.run(dt).ok());

    using namespace Kokkos::Tools::Experimental;
    static std::array<unsigned, 3> launches;
    launches = {};
    const auto count = [](const char* label, std::uint32_t, std::uint64_t*) {
        const std::string_view kernel(label);
        if (kernel == "sph.grid.validate") ++launches[0];
        if (kernel == "sph.continuity.evaluate_density_rate") ++launches[1];
        if (kernel == "sph.pressure.acceleration") ++launches[2];
    };
    const auto previous = get_callbacks();
    set_begin_parallel_for_callback(count);
    set_begin_parallel_reduce_callback(count);
    const auto status = simulation.run(dt);
    set_callbacks(previous);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(launches, (std::array<unsigned, 3>{1, 1, 1}));
    ASSERT_TRUE(simulation.finalize().ok());
}

// The larger second step moves a particle into a different grid cell. The
// smaller step reuses cell membership and fits the very same byte budget.
TEST(SphVerlet, RegularStepBudgetFailureAllowsShorterRetry) {
    const M4FixedWallFixture fixture;
    constexpr double first = fixture.first_step, large = fixture.step, small = first / 4;
    const auto run_short = [&](std::size_t limit) {
        auto probe = fixture.participant({32, limit});
        Simulation run(probe);
        auto status = run.initialize();
        if (status.ok())
            status = run.run(first);
        if (status.ok())
            status = run.run(small, 2);
        EXPECT_TRUE(status.ok() || status.code() == StatusCode::resource_exhausted);
        EXPECT_TRUE(run.finalize().ok());
        return status.ok();
    };
    std::size_t lower = 0, upper = 4096;
    ASSERT_TRUE(run_short(upper));
    while (upper - lower > 1) {
        const auto middle = lower + (upper - lower) / 2;
        if (run_short(middle))
            upper = middle;
        else
            lower = middle;
    }

    auto reference = fixture.participant(), participant = fixture.participant({32, upper});
    Simulation direct(reference), simulation(participant);
    ASSERT_TRUE(direct.initialize().ok());
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(direct.run(first).ok());
    ASSERT_TRUE(simulation.run(first).ok());
    const auto identity = WcsphTestAccess::accepted_identity(participant);
    const auto before = committed(participant);
    const auto wall = checked(participant.read_committed_wall(30));
    const auto rates = WcsphTestAccess::rates(participant);

    const auto failure = simulation.run(large);
    ASSERT_EQ(failure.code(), StatusCode::resource_exhausted) << failure.message();
    EXPECT_EQ(failure.message(), "participant resource budget exceeded");
    EXPECT_EQ(WcsphTestAccess::accepted_identity(participant), identity);
    EXPECT_EQ(WcsphTestAccess::rates(participant), rates);
    expect_metadata(participant, simulation, first, 1, identity.maximum_step);
    expect_same_rows(committed(participant), before);
    EXPECT_EQ(checked(participant.read_committed_wall(30)), wall);

    double time = first;
    for (int retry = 0; retry < 2; ++retry) {
        ASSERT_TRUE(direct.run(small).ok());
        const auto status = participant.step({time, time + small, small});
        ASSERT_TRUE(status.ok()) << status.message();
        time += small;
        expect_same_sample(committed(participant), committed(reference));
        EXPECT_EQ(WcsphTestAccess::rates(participant), WcsphTestAccess::rates(reference));
        EXPECT_EQ(participant.time_constraint().maximum_step, reference.time_constraint().maximum_step);
    }
    RecordProperty("retry_budget_bytes", upper);
    ASSERT_TRUE(direct.finalize().ok());
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

// Pressure coefficients are the last Kokkos scratch allocation destroyed before field
// publication. The Serial HostSpace deallocation callback runs after building its label.
// Stop counting when the cold grid is installed, before old-state teardown can allocate.
class PressurePublicationFault {
  public:
    PressurePublicationFault(std::size_t fail_at, const sph::WcsphParticipant3D& participant)
        : previous_(Kokkos::Tools::Experimental::get_callbacks()) {
        host_fault = {.fail_at = fail_at, .committed_grid = &WcsphTestAccess::committed_grid(participant)};
        Kokkos::Tools::Experimental::set_deallocate_data_callback(
            [](Kokkos::Tools::SpaceHandle, const char* label, const void*, std::uint64_t) {
                if (std::string_view(label) == "sph.pressure_over_density_squared")
                    host_fault.active = true;
            });
    }
    ~PressurePublicationFault() {
        host_fault.active = false;
        Kokkos::Tools::Experimental::set_callbacks(previous_);
    }
    PressurePublicationFault(const PressurePublicationFault&) = delete;
    PressurePublicationFault& operator=(const PressurePublicationFault&) = delete;

  private:
    Kokkos::Tools::Experimental::EventSet previous_;
};

TEST(SphVerlet, PressurePublicationAllocationFailurePreservesHistory) {
    const M4FixedWallFixture fixture;
    constexpr auto first = fixture.first_step, dt = fixture.step;
    auto reference = fixture.participant();
    Simulation direct(reference);
    ASSERT_TRUE(direct.initialize().ok());
    ASSERT_TRUE(direct.run(first).ok());
    ASSERT_TRUE(direct.run(dt).ok());
    const auto expected = committed(reference);
    const auto expected_rates = WcsphTestAccess::rates(reference);
    ASSERT_TRUE(direct.finalize().ok());

    std::size_t publication_allocations{};
    const auto exercise = [&](std::size_t fail_at) {
        SCOPED_TRACE(fail_at);
        auto participant = fixture.participant();
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.run(first).ok());
        ASSERT_TRUE(participant.release_workspace().ok());
        ASSERT_FALSE(WcsphTestAccess::committed_grid(participant));
        const auto before = committed(participant);
        const auto identity = WcsphTestAccess::accepted_identity(participant);
        const auto before_rates = WcsphTestAccess::rates(participant);
        const auto row_map = WcsphTestAccess::row_to_input(participant);
        const auto wall = checked(participant.read_committed_wall(30));
        const auto usage = participant.resource_usage();

        Status result;
        {
            const PressurePublicationFault fault(fail_at, participant);
            result = simulation.run(dt);
        }
        if (fail_at) {
            ASSERT_TRUE(host_fault.threw);
            EXPECT_EQ(host_fault.calls, fail_at);
            ASSERT_EQ(result.code(), StatusCode::resource_exhausted) << result.message();
            expect_metadata(participant, simulation, first, 1, identity.maximum_step);
            expect_same_rows(committed(participant), before);
            EXPECT_EQ(checked(participant.read_committed_wall(30)), wall);
            EXPECT_EQ(WcsphTestAccess::accepted_identity(participant), identity);
            EXPECT_EQ(WcsphTestAccess::row_to_input(participant), row_map);
            EXPECT_EQ(WcsphTestAccess::rates(participant), before_rates);
            EXPECT_FALSE(WcsphTestAccess::committed_grid(participant));
            EXPECT_EQ(participant.resource_usage().controlled_bytes, usage.controlled_bytes);
            EXPECT_EQ(participant.resource_usage().snapshot_records, usage.snapshot_records);
            ASSERT_TRUE(participant.step({first, first + dt, dt}).ok());
        } else {
            ASSERT_TRUE(result.ok()) << result.message();
            publication_allocations = host_fault.calls;
        }
        expect_same_sample(committed(participant), expected);
        EXPECT_EQ(WcsphTestAccess::rates(participant), expected_rates);
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    };
    ASSERT_NO_FATAL_FAILURE(exercise(0));
    ASSERT_GT(publication_allocations, 0u);
    RecordProperty("pressure_publication_host_allocations", publication_allocations);
    for (std::size_t fail_at = 1; fail_at <= publication_allocations; ++fail_at)
        ASSERT_NO_FATAL_FAILURE(exercise(fail_at));
}

void expect_committed_reads_rejected(const sph::WcsphParticipant3D& participant) {
    std::size_t calls = 0;
    const auto rows = participant.visit_committed_rows([&](sph::CommittedRow3D) { ++calls; });
    EXPECT_EQ(rows.code(), StatusCode::invalid_state);
    EXPECT_EQ(calls, 0u);
    const auto wall = participant.read_committed_wall(30);
    EXPECT_FALSE(wall.ok());
    EXPECT_EQ(wall.status().code(), StatusCode::invalid_state);
}

TEST(SphVerlet, CommittedReadsGuardReentrancyAndRecoverAfterVisitorException) {
    const M4FixedWallFixture fixture;
    auto participant = fixture.participant();
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto before = committed(participant);
    ASSERT_EQ(before.rows.size(), 2u);
    EXPECT_EQ(before.rows[0].id, 10u);
    EXPECT_EQ(before.rows[1].id, 20u);
    const auto wall = checked(participant.read_committed_wall(30));
    EXPECT_EQ(wall.id, 30u);
    EXPECT_EQ(wall.mass, 3);
    EXPECT_EQ(wall.position, (std::array<real, 3>{4, 0, 0}));
    EXPECT_EQ(wall.stamp, before.stamp);
    EXPECT_EQ(wall.phase, 0);
    const double maximum = participant.time_constraint().maximum_step;

    std::vector<std::uint64_t> visited;
    const auto normal = participant.visit_committed_rows([&](sph::CommittedRow3D row) {
        visited.push_back(row.id);
        if (visited.size() == 1) {
            EXPECT_EQ(participant.release_workspace().code(), StatusCode::invalid_state);
            expect_committed_reads_rejected(participant);
        }
    });
    EXPECT_TRUE(normal.ok()) << normal.message();
    EXPECT_EQ(visited, (std::vector<std::uint64_t>{10, 20}));
    std::size_t calls = 0;
    const auto thrown = participant.visit_committed_rows([&](sph::CommittedRow3D) {
        ++calls;
        throw std::runtime_error("observer failed");
    });
    EXPECT_EQ(thrown.code(), StatusCode::execution_failure);
    EXPECT_EQ(calls, 1u);
    expect_same_rows(committed(participant), before);
    EXPECT_EQ(checked(participant.read_committed_wall(30)), wall);
    expect_metadata(participant, simulation, 0, 0, maximum);
    ASSERT_TRUE(simulation.run(fixture.step).ok());
    EXPECT_TRUE(simulation.finalize().ok());
}

TEST(SphVerlet, DirectFinalizeRejectedDuringCommittedRowVisit) {
    const M4FixedWallFixture fixture;
    auto participant = fixture.participant();
    ASSERT_TRUE(participant.initialize(0).ok());
    std::vector<std::uint64_t> ids;
    const auto read = participant.visit_committed_rows([&](sph::CommittedRow3D row) {
        ids.push_back(row.id);
        if (ids.size() == 1)
            EXPECT_EQ(participant.finalize().code(), StatusCode::invalid_state);
    });
    EXPECT_TRUE(read.ok()) << read.message();
    EXPECT_EQ(ids, (std::vector<std::uint64_t>{10, 20}));
    const TimeInterval interval{0, fixture.first_step, fixture.first_step};
    ASSERT_TRUE(participant.step(interval).ok());
    EXPECT_EQ(committed(participant).stamp.accepted_version.value(), 1u);
    EXPECT_TRUE(participant.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

TEST(SphVerlet, SimulationFinalizeFailsDuringCommittedRowVisit) {
    const M4FixedWallFixture fixture;
    auto participant = fixture.participant();
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.run(fixture.first_step).ok());
    const auto before = simulation.time();
    std::vector<std::uint64_t> ids;
    const auto read = participant.visit_committed_rows([&](sph::CommittedRow3D row) {
        ids.push_back(row.id);
        if (ids.size() == 1) {
            EXPECT_EQ(simulation.finalize().code(), StatusCode::invalid_state);
        }
    });
    EXPECT_TRUE(read.ok()) << read.message();
    EXPECT_EQ(ids, (std::vector<std::uint64_t>{10, 20}));
    EXPECT_EQ(simulation.time().accepted_time, before.accepted_time);
    EXPECT_EQ(simulation.time().accepted_steps, before.accepted_steps);
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}


TEST(SphVerlet, LateCflFailurePreservesPublishedFieldsAndReleasesCandidate) {
    const Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const sph::Velocity3D velocity("late_cfl.velocity", 1);
    const sph::Values density("late_cfl.density", 1);
    density(0) = 1;
    constexpr int exponent = std::numeric_limits<real>::max_exponent;
    const double failed_dt = std::ldexp(1.0, 7 - exponent / 2);
    const real gravity = std::ldexp(real{1}, exponent - 6);
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, .5, {gravity, 0, 0}, 1};
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density);
    StepAllocations allocations(participant);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(participant.release_workspace().ok());
    const auto before = committed(participant);
    const auto identity = WcsphTestAccess::accepted_identity(participant);
    const auto row_map = WcsphTestAccess::row_to_input(participant);
    const auto usage = participant.resource_usage();
    const auto mark = allocations.events.size();
    const auto failure = simulation.run(failed_dt, 2);
    EXPECT_EQ(failure.code(), StatusCode::numerical_failure);
    EXPECT_EQ(allocations.count(mark, "sph.participant3d.velocity", true), 1u);
    EXPECT_EQ(allocations.count(mark, "sph.participant3d.velocity", false), 1u);
    expect_metadata(participant, simulation, 0, 0, .5);
    expect_same_rows(committed(participant), before);
    EXPECT_EQ(WcsphTestAccess::accepted_identity(participant), identity);
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), row_map);
    ASSERT_TRUE(participant.release_workspace().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, usage.controlled_bytes);
    EXPECT_EQ(participant.resource_usage().snapshot_records, usage.snapshot_records);
    EXPECT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

TEST(SphVerlet, FixedWallBudgetRejectionPreservesRowsAndWall) {
    const M4FixedWallFixture fixture;
    fixture.fluid.position(0, 0) = 1.125;
    constexpr auto dt = fixture.first_step;
    constexpr std::size_t source_count = 3;
    constexpr std::size_t vector_bytes = source_count * 3 * sizeof(real);
    constexpr std::size_t grid_position_bytes = source_count * 3 * sizeof(real);

    auto reference = fixture.participant();
    std::optional<StepAllocations> reference_allocations(std::in_place, reference);
    Simulation direct(reference);
    ASSERT_TRUE(direct.initialize().ok());
    const auto after_initialize = reference_allocations->events.size();
    const auto initial_bytes = reference.resource_usage().controlled_bytes;
    const auto initial_peak = reference_allocations->peak(0, after_initialize);
    ASSERT_TRUE(direct.run(dt).ok());
    const auto expected = committed(reference);
    const auto expected_maximum_step = reference.time_constraint().maximum_step;
    const auto step_peak = reference_allocations->peak(after_initialize,
                                                       reference_allocations->events.size());
    ASSERT_GT(step_peak, initial_peak);
    ASSERT_TRUE(direct.finalize().ok());
    reference_allocations.reset();

    // The old grid has the same box, input order, and cell membership on this
    // step. The drift positions fit; the new grid position allocation needs one
    // more byte. Its mass, identity and permutation arrays are shared.
    const auto refusal_limit = initial_bytes + vector_bytes + grid_position_bytes - 1;
    ASSERT_GE(refusal_limit, initial_peak);
    ASSERT_LT(refusal_limit, step_peak);
    ::testing::Test::RecordProperty("fixed_wall_initialization_peak_bytes", initial_peak);
    ::testing::Test::RecordProperty("fixed_wall_step_peak_bytes", step_peak);
    ::testing::Test::RecordProperty("fixed_wall_refusal_limit_bytes", refusal_limit);
    ::testing::Test::RecordProperty("rejected_grid_position_bytes", grid_position_bytes);

    const auto cap = step_peak + grid_position_bytes;
    auto participant = fixture.participant({32, cap});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);
    const auto before = committed(participant);
    const auto before_wall = checked(participant.read_committed_wall(30));
    const auto identity = WcsphTestAccess::accepted_identity(participant);
    const auto row_map = WcsphTestAccess::row_to_input(participant);
    const auto maximum_step = participant.time_constraint().maximum_step;
    const auto held_bytes = cap - refusal_limit;
    auto held = checked(WcsphTestAccess::reserve_test_budget(participant, held_bytes));
    {
        StepAllocations rejected(participant);
        const auto failure = simulation.run(dt, 2);
        EXPECT_EQ(failure.code(), StatusCode::resource_exhausted);
        EXPECT_EQ(failure.message(), "participant resource budget exceeded");
        EXPECT_EQ(rejected.count(0, "sph.central_difference.position", true), 1u);
        EXPECT_EQ(rejected.count(0, "sph.grid.position", true), 0u);
        EXPECT_EQ(rejected.count(0, "sph.grid.mass", true), 0u);
        EXPECT_EQ(rejected.count(0, "sph.grid.identity", true), 0u);
        EXPECT_EQ(rejected.count(0, "sph.grid.to_input", true), 0u);
        EXPECT_EQ(rejected.count(0, "sph.grid.from_input", true), 0u);
        EXPECT_EQ(rejected.peak(0, rejected.events.size()), cap - (grid_position_bytes - 1));
        rejected.expect_all_released();
    }
    expect_metadata(participant, simulation, 0, 0, maximum_step);
    expect_same_rows(committed(participant), before);
    EXPECT_EQ(checked(participant.read_committed_wall(30)), before_wall);
    EXPECT_EQ(WcsphTestAccess::accepted_identity(participant), identity);
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), row_map);
    EXPECT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + held_bytes);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0u);

    held.reset();
    EXPECT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);
    // Simulation closes its run path after failure. Exercise participant retry
    // directly without asking that failed coordinator to advance again.
    ASSERT_TRUE(participant.step(TimeInterval{0, dt, dt}).ok());
    expect_same_rows(committed(participant), expected);
    EXPECT_EQ(WcsphTestAccess::stamp(participant).time.physical_time, dt);
    EXPECT_EQ(WcsphTestAccess::stamp(participant).accepted_version.value(), 1u);
    EXPECT_EQ(participant.time_constraint().maximum_step, expected_maximum_step);
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

} // namespace

namespace {

namespace wall = sph::SphBoundary;
using WallMotion = wall::RigidWallMotion3D;
using WallTime = wall::WallTrajectoryTime3D;
using TrajectoryInput = wall::PrescribedWallTrajectory3D;
using WallGroups = std::vector<wall::WallGroupInput3D>;

sph::FixedWallParticles3D wall_fields(const Fields& fields) {
    return {fields.position, fields.mass, fields.ids};
}

struct TrajectoryCall {
    WallTime time;
    double phase;
};

using TrajectoryCalls = std::vector<TrajectoryCall>;

TrajectoryInput stationary_motion(double limit) {
    TrajectoryInput result;
    result.time_resolution = 0;
    result.motion = [](WallTime, double) -> Result<WallMotion> { return WallMotion{}; };
    result.sample_step_limit = limit;
    return result;
}

TEST(SphVerlet, WallAssemblyRejectsInvalidParticles) {
    const Fields empty({});
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, .5};
    const auto assemble = [&](const Fields& fields,
                              std::optional<TrajectoryInput> trajectory = std::nullopt) {
        sph::WcsphParticipant3D participant(
            empty.get(), 4, parameters, {}, {}, WallGroups{{1, wall_fields(fields), trajectory}});
        const auto result = participant.initialize(0);
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
        EXPECT_FALSE(participant.read_committed_wall(21).ok());
        EXPECT_TRUE(participant.finalize().ok());
        return result.code();
    };
    Fields particles({{{.25, 0, 0}, 1, 21, 1}});
    for (const real mass : {real{0}, std::numeric_limits<real>::infinity()}) {
        particles.mass(0) = mass;
        EXPECT_EQ(assemble(particles), StatusCode::invalid_argument);
    }
    particles.mass(0) = 1;
    std::size_t calls{};
    auto moving = stationary_motion(.5);
    moving.motion = [&](WallTime, double) -> Result<WallMotion> {
        ++calls;
        return WallMotion{};
    };
    for (const real x : {std::numeric_limits<real>::quiet_NaN(),
                         std::numeric_limits<real>::infinity()}) {
        particles.position(0, 0) = x;
        EXPECT_EQ(assemble(particles), StatusCode::invalid_argument);
        EXPECT_EQ(assemble(particles, moving), StatusCode::invalid_argument);
        EXPECT_EQ(calls, 0u);
    }
    const Fields duplicate({{{.25, 0, 0}, 1, 21, 1}, {{.5, 0, 0}, 2, 21, 1}});
    EXPECT_EQ(assemble(duplicate), StatusCode::invalid_argument);

    particles.position(0, 0) = real{0x1p64};
    EXPECT_EQ(assemble(particles), StatusCode::unsupported);
}

TEST(SphVerlet, MovingWallRejectsUnrepresentableGroupDurationBeforeSampling) {
    constexpr double t = 0x1p52;
    const Fields empty({}), first({{{.25, 0, 0}, 2, 70, 1}}),
        second({{{.5, 0, 0}, 3, 71, 1}});
    const sph::WcsphVerletParameters parameters{{1, 0x1p-54, 7}, .5};
    struct Case {
        double request, resolution, span;
    };
    for (const auto [request, resolution, span] : {Case{.25, .25, 0}, Case{1.75, .125, 2}}) {
        SCOPED_TRACE(request);
        EXPECT_EQ((t + request) - t, span);
        TrajectoryCalls calls;
        auto precise = stationary_motion(t);
        precise.time_resolution = resolution;
        precise.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
            calls.push_back({time, phase});
            return WallMotion{};
        };
        auto permissive = precise;
        permissive.time_resolution = 1;
        const WallGroups groups{{10, wall_fields(first), permissive}, {11, wall_fields(second), precise}};
        sph::WcsphParticipant3D participant(empty.get(), 64, parameters, {}, {}, groups);
        Simulation simulation(participant, -t);
        ASSERT_TRUE(simulation.initialize().ok());
        expect_metadata(participant, simulation, -t, 0, t);
        ASSERT_TRUE(simulation.run(t).ok());
        expect_metadata(participant, simulation, 0, 1, t);
        const auto before = checked(participant.read_committed_wall(70));
        EXPECT_EQ(before.phase, t);
        EXPECT_EQ(before.mass, 2);
        EXPECT_EQ(before.position, (std::array<real, 3>{.25, 0, 0}));
        const auto accepted_phases = WcsphTestAccess::wall_groups(participant);
        calls.clear();

        const auto failure = simulation.run(request, 2);
        EXPECT_EQ(failure.code(), StatusCode::numerical_failure);
        EXPECT_TRUE(calls.empty());
        expect_metadata(participant, simulation, 0, 1, t);
        EXPECT_EQ(checked(participant.read_committed_wall(70)), before);
        EXPECT_EQ(WcsphTestAccess::wall_groups(participant), accepted_phases);
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    }
}

TEST(SphVerlet, MovingWallActualSpanMustFitItsExplicitLimit) {
    const Fields fluid({}), particles({{{.25, 0, 0}, 2, 70, 1}});
    const double limit = std::nextafter(1.75, 0.);
    auto trajectory = stationary_motion(limit);
    trajectory.time_resolution = 1e-12;
    sph::WcsphParticipant3D participant(fluid.get(), 1, {{1, .25, 1}, 1}, {}, {},
                                       WallGroups{{1, wall_fields(particles), trajectory}});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.run(1.).ok());
    ASSERT_EQ(participant.time_constraint().maximum_step, limit);
    ASSERT_GT((1. + limit) - 1., limit);
    const auto before = checked(participant.read_committed_wall(70));
    EXPECT_EQ(simulation.run(limit).code(), StatusCode::numerical_failure);
    EXPECT_EQ(checked(participant.read_committed_wall(70)), before);
    EXPECT_EQ(simulation.time().accepted_steps, 1u);
    EXPECT_TRUE(simulation.finalize().ok());
}

TEST(SphVerlet, MovingWallFiniteEndpointPublishesAndRejectsAnotherStep) {
    const Fields empty({}), particles({{{.25, 0, 0}, 2, 70, 1}});
    const sph::WcsphVerletParameters parameters{{1, .25, 7}, .5};
    for (const double next_step : {0., .5, 0x1p-54}) {
        SCOPED_TRACE(next_step);
        TrajectoryCalls calls;
        auto trajectory = stationary_motion(1);
        trajectory.first_phase = 0;
        trajectory.last_phase = 1;
        trajectory.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
            calls.push_back({time, phase});
            WallMotion result;
            result.translation[0] = phase / 4;
            result.velocity[0] = .25;
            return result;
        };
        const WallGroups groups{{12, wall_fields(particles), trajectory}};
        sph::WcsphParticipant3D participant(empty.get(), 64, parameters, {}, {}, groups);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_EQ(calls.size(), 1u);
        expect_metadata(participant, simulation, 0, 0, 1);
        const auto before = checked(participant.read_committed_wall(70));
        const auto result = next_step > 0 ? simulation.run(std::array{.5, .5, next_step})
                                          : simulation.run(.5, 2);
        if (next_step > 0) {
            EXPECT_EQ(result.code(), StatusCode::invalid_argument);
            EXPECT_EQ(result.message(), "participant finite time domain is exhausted");
        } else {
            EXPECT_TRUE(result.ok()) << result.message();
        }
        expect_metadata(participant, simulation, 1, 2, 0);
        const auto after = checked(participant.read_committed_wall(70));
        EXPECT_EQ(after.id, 70u);
        EXPECT_EQ(after.mass, 2);
        EXPECT_EQ(after.phase, 1);
        EXPECT_EQ(after.position, (std::array<real, 3>{.5, 0, 0}));
        EXPECT_EQ(after.stamp.geometry.value(), before.stamp.geometry.value() + 2);
        EXPECT_EQ(after.stamp.topology, before.stamp.topology);
        EXPECT_EQ(WcsphTestAccess::wall_groups(participant).at(12),
                  (WcsphTestAccess::WallGroupValue{0, 1, 0}));
        const auto count = calls.size();
        ASSERT_EQ(count, 3u);
        constexpr std::array expected_elapsed{0., .5, 1.};
        for (std::size_t i = 0; i < count; ++i) {
            EXPECT_EQ(calls[i].time.elapsed, expected_elapsed[i]);
            EXPECT_EQ(calls[i].phase, expected_elapsed[i]);
        }
        EXPECT_TRUE(participant.time_constraint().available);
        EXPECT_EQ(participant.time_constraint().maximum_step, 0.);
        EXPECT_EQ(checked(participant.read_committed_wall(70)), after);
        EXPECT_EQ(calls.size(), count);
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    }
}

constexpr double piston_reference_position = -2 - 1. / 512;
constexpr double piston_mass = 8192;
constexpr double piston_sound_speed = .875;
constexpr double piston_limit = 1. / 32;

std::array<double, 3> piston_signal(double phase) {
    const double q = phase * (1 - phase);
    const double derivative = 1 - 2 * phase;
    const double square = q * q;
    return {2 * square * square, 8 * q * square * derivative,
            24 * square * derivative * derivative - 16 * q * square};
}

struct PistonReference {
    double density, velocity, acceleration;
};

std::vector<PistonReference> integrate_piston_reference(int steps, const auto& wall_at) {
    constexpr double c2 = piston_sound_speed * piston_sound_speed;
    constexpr double inverse_four_pi = 1 / (4 * std::numbers::pi);
    const auto reference = [&](double time, double x, double velocity) {
        const auto [translation, wall_acceleration] = wall_at(time);
        const double wall_x = piston_reference_position + translation;
        const double r = x - wall_x;
        const double gap = std::max(2 - r, 0.);
        const double rho = 2 + piston_mass * std::pow(gap, 3) * inverse_four_pi;
        double fluid_acceleration{};
        if (r < 2) {
            const double p = c2 * (rho - 1);
            const double pw = p + rho * wall_acceleration * r;
            const double rhow = 1 + pw / c2;
            fluid_acceleration = piston_mass * 3 * std::pow(gap, 2) * inverse_four_pi *
                                 (p / (rho * rho) + pw / (rhow * rhow));
        }
        return PistonReference{rho, velocity, fluid_acceleration};
    };
    const auto acceleration = [&](double time, double x) {
        return reference(time, x, 0.).acceleration;
    };

    const double dt = 1. / steps;
    double x{}, velocity{};
    std::vector<PistonReference> result{reference(0, x, velocity)};
    for (int step = 0; step < steps; ++step) {
        const double time = step * dt;
        const double k1x = velocity, k1v = acceleration(time, x);
        const double k2x = velocity + .5 * dt * k1v;
        const double k2v = acceleration(time + .5 * dt, x + .5 * dt * k1x);
        const double k3x = velocity + .5 * dt * k2v;
        const double k3v = acceleration(time + .5 * dt, x + .5 * dt * k2x);
        const double k4x = velocity + dt * k3v;
        const double k4v = acceleration(time + dt, x + dt * k3x);
        x += dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6;
        velocity += dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6;
        result.push_back(reference(time + dt, x, velocity));
    }
    return result;
}

sph::WcsphParticipant3D piston_participant(const TrajectoryInput& trajectory) {
    const Fields fluid({{{0, 0, 0}, 1, 11, 0}});
    const Fields particles({{{piston_reference_position, 0, 0}, piston_mass, 29, 1}});
    const sph::Velocity3D velocity("test.piston_fluid_velocity", 1);
    sph::Values density("test.piston_fluid_density", 1);
    density(0) = 2;
    const sph::WcsphVerletParameters parameters{{1, piston_sound_speed, 1}, 1};
    return {fluid.get(), 64, parameters, velocity, density, WallGroups{{21, wall_fields(particles), trajectory}}};
}

struct ReplayMotion {
    double translation, velocity, acceleration;
};

ReplayMotion replay_piston(double time) {
    struct Sample { double time, velocity; };
    constexpr std::array<Sample, 4> samples{{{0, 0}, {.25, 1. / 32}, {.625, -1. / 64}, {1, 0}}};
    double translation{};
    for (std::size_t i = 1; i < samples.size(); ++i) {
        const auto [start, v0] = samples[i - 1];
        const auto [end, v1] = samples[i];
        const double width = end - start;
        if (time >= start && time <= end) {
            // Cubic Hermite velocity with zero slope at every knot. Integrate it
            // exactly for pose and differentiate it for the Adami acceleration.
            const double u = (time - start) / width;
            const double dv = v1 - v0;
            return {translation + width * (v0 * u + dv * u * u * u * (1 - .5 * u)),
                    v0 + dv * u * u * (3 - 2 * u), dv * 6 * u * (1 - u) / width};
        }
        translation += width * (v0 + v1) / 2;
    }
    throw std::out_of_range("piston sample time outside the recording");
}

TEST(SphVerlet, DiscreteWallReplayMatchesReferenceFluidImpulse) {
    // At the midpoint of the second interval the preceding sample contributes
    // 1/256 to the displacement and the wall is decelerating through contact.
    const auto midpoint = replay_piston(7. / 16);
    EXPECT_DOUBLE_EQ(midpoint.translation, 133. / 16384);
    EXPECT_DOUBLE_EQ(midpoint.velocity, 1. / 128);
    EXPECT_DOUBLE_EQ(midpoint.acceleration, -3. / 16);
    constexpr int reference_steps = 8192;
    const auto wall_at = [](double time) {
        const auto motion = replay_piston(time);
        return std::pair{motion.translation, motion.acceleration};
    };
    // The independent one-neighbor RK4 reference integrates the pressure force
    // on the unit fluid mass. Both solvers query the same recording at physical time.
    const auto integrated = integrate_piston_reference(reference_steps, wall_at);
    const double reference = integrated.back().velocity;
    const double coarser = integrate_piston_reference(reference_steps / 2, wall_at).back().velocity;
    ASSERT_NEAR(reference, coarser, 1.e-12);
    ASSERT_GT(reference, 1.e-4);

    constexpr int steps = 256;
    constexpr double dt = 1. / steps;
    auto trajectory = stationary_motion(dt);
    trajectory.first_phase = 0;
    trajectory.last_phase = 1;
    trajectory.motion = [](WallTime time, double) -> Result<WallMotion> {
        const auto motion = replay_piston(time.elapsed);
        return WallMotion{.translation = {static_cast<real>(motion.translation), 0, 0},
                          .velocity = {static_cast<real>(motion.velocity), 0, 0},
                          .acceleration = {static_cast<real>(motion.acceleration), 0, 0}};
    };
    auto participant = piston_participant(trajectory);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto initial = committed(participant).rows.at(0);
    double previous_force = initial.mass * initial.pressure_acceleration[0];
    double impulse{}, momentum_change{};
    for (int step = 1; step <= steps; ++step) {
        const auto status = simulation.run(dt);
        ASSERT_TRUE(status.ok()) << status.message();
        const auto state = committed(participant);
        const auto& fluid = state.rows.at(0);
        const double time = step * dt;
        const auto wall_value = checked(participant.read_committed_wall(29));
        ASSERT_EQ(state.stamp.time.physical_time, time);
        ASSERT_EQ(wall_value.stamp.time.physical_time, time);
        EXPECT_NEAR(wall_value.position[0], piston_reference_position + replay_piston(time).translation,
                    2 * std::numeric_limits<real>::epsilon());
        if (time == 7. / 16) {
            // Density integrates the continuity equation driven by fluid-wall relative velocity.
            EXPECT_NEAR(fluid.density, integrated[7 * reference_steps / 16].density, 4.e-6);
        }
        // Gravity is zero and the wall is the only neighbor. Every accepted
        // endpoint contributes its instantaneous wall force to the trapezoid.
        const double force = fluid.mass * fluid.pressure_acceleration[0];
        impulse += .5 * dt * (previous_force + force);
        previous_force = force;
        momentum_change = fluid.mass * fluid.velocity[0] - initial.mass * initial.velocity[0];
    }
    const double tolerance = 1.e-4 * reference;
    EXPECT_NEAR(impulse, reference, tolerance);
    EXPECT_NEAR(momentum_change, reference, tolerance);
    EXPECT_NEAR(momentum_change, impulse, tolerance);
    std::ostringstream report;
    report << std::setprecision(16) << "reference=" << reference << ", rk4_refinement=" << std::abs(reference - coarser)
           << ", sph_force_integral=" << impulse << ", fluid_momentum_change=" << momentum_change;
    RecordProperty("replay_impulse", report.str());
    ASSERT_TRUE(simulation.finalize().ok());
}

struct CallerEulerWall {
    WallMotion accepted, candidate;
    double accepted_time{}, candidate_time{};

    void stage(double end, double dt, WallMotion endpoint) {
        candidate_time = end;
        endpoint.center = accepted.center;
        endpoint.axis = accepted.axis;
        for (int axis = 0; axis < 3; ++axis)
            endpoint.translation[axis] = std::fma(static_cast<real>(dt), accepted.velocity[axis],
                                                 accepted.translation[axis]);
        endpoint.angle = std::fma(static_cast<real>(dt), accepted.angular_speed, accepted.angle);
        candidate = endpoint;
    }

    Result<WallMotion> sample(WallTime time) const {
        if (time.elapsed != accepted_time && time.elapsed != candidate_time)
            return Status{StatusCode::invalid_state, "caller has no staged wall motion at this time"};
        const auto& motion = time.elapsed == accepted_time ? accepted : candidate;
        // The caller needs these rates for its next Euler step, even for a wall-only run.
        if (!std::isfinite(motion.angular_speed) ||
            !std::all_of(motion.velocity.begin(), motion.velocity.end(), [](real v) { return std::isfinite(v); }))
            return Status{StatusCode::numerical_failure, "nonfinite caller Euler rate"};
        return motion;
    }

    void publish() {
        accepted = candidate;
        accepted_time = candidate_time;
    }
};

void check_euler_piston(bool expression) {
    const auto endpoint = [](double time) {
        const auto signal = piston_signal(time);
        return WallMotion{.velocity = {static_cast<real>(signal[1]), 0, 0},
                          .acceleration = {static_cast<real>(signal[2]), 0, 0}};
    };
    double previous_wall_error{};
    for (int steps : {32, 64, 128, 256}) {
        SCOPED_TRACE(steps);
        const double dt = 1. / steps;
        CallerEulerWall owner{.accepted = endpoint(0)};
        auto trajectory = stationary_motion(piston_limit);
        trajectory.motion = [&](WallTime time, double) { return owner.sample(time); };
        std::unique_ptr<app::PrescribedMotion> expression_owner;
        if (expression) {
            app::MotionConfig config{.mode = app::MotionMode::velocity};
            config.expressions[0] = "8*(t*(1-t))^3*(1-2*t)";
            expression_owner = checked(app::PrescribedMotion::create(config));
            trajectory = expression_owner->trajectory(piston_limit);
        }
        trajectory.period = 1;
        auto participant = piston_participant(trajectory);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        double maximum_wall_error{};
        double euler_translation{};
        for (int step = 1; step <= steps; ++step) {
            euler_translation += dt * piston_signal((step - 1) * dt)[1];
            const double time = step * dt;
            if (expression_owner)
                ASSERT_TRUE(expression_owner->prepare(dt).ok());
            else
                owner.stage(time, dt, endpoint(time));
            const auto [translation, wall_velocity, wall_acceleration] = piston_signal(time);
            const double epsilon = std::numeric_limits<real>::epsilon();
            // Check the group's endpoint rates even while the wall is outside kernel support.
            const auto trial_motion = checked(trajectory.motion({0, time}, std::fmod(time, trajectory.period)));
            EXPECT_NEAR(trial_motion.velocity[0], wall_velocity, 4 * epsilon);
            EXPECT_NEAR(trial_motion.acceleration[0], wall_acceleration, 4 * epsilon);
            ASSERT_TRUE(simulation.run(dt).ok());
            if (expression_owner)
                expression_owner->commit();
            else
                owner.publish();
            const auto wall_value = checked(participant.read_committed_wall(29));
            const auto state = committed(participant);
            ASSERT_EQ(state.rows.size(), 1u);
            const auto& particle = state.rows.front();
            EXPECT_EQ(wall_value.stamp.time.physical_time, time);
            EXPECT_EQ(state.stamp.time.physical_time, time);
            EXPECT_NEAR(wall_value.position[0], piston_reference_position + euler_translation, 4 * epsilon);
            maximum_wall_error = std::max(maximum_wall_error,
                std::abs(wall_value.position[0] - (piston_reference_position + translation)));

            // Independent one-neighbor M4 continuity and Adami force at the published endpoint.
            const double separation = static_cast<real>(particle.position[0] - wall_value.position[0]);
            const double gap = std::max(0., 2 - separation);
            const double gradient = piston_mass * 3 * gap * gap / (4 * std::numbers::pi);
            const double density_rate = gradient * (wall_velocity - particle.velocity[0]);
            const auto rates = WcsphTestAccess::rates(participant);
            ASSERT_EQ(rates[1].size(), 1u);
            EXPECT_NEAR(rates[1][0], density_rate, 16 * epsilon * std::max(.01, std::abs(density_rate)));
            const double fluid_density = particle.density;
            const double c2 = piston_sound_speed * piston_sound_speed;
            const double fluid_pressure = c2 * (fluid_density - 1);
            const double wall_pressure = fluid_pressure + fluid_density * wall_acceleration * separation;
            const double wall_density = 1 + wall_pressure / c2;
            const double acceleration = gradient * (fluid_pressure / (fluid_density * fluid_density)
                + wall_pressure / (wall_density * wall_density));
            EXPECT_NEAR(particle.pressure_acceleration[0], acceleration,
                        32 * epsilon * std::max(.01, std::abs(acceleration)));
        }
        std::ostringstream text;
        text << std::scientific << std::setprecision(16) << maximum_wall_error;
        ::testing::Test::RecordProperty("max_wall_position_error_" + std::to_string(steps), text.str());
        if (previous_wall_error)
            EXPECT_NEAR(previous_wall_error / maximum_wall_error, 2., .05);
        previous_wall_error = maximum_wall_error;
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphVerlet, CallerEulerPistonChecksWallConvergenceAndFluidRates) {
    check_euler_piston(false);
}

TEST(SphLeptonWall, VelocityPistonChecksWallConvergenceAndFluidRates) {
    check_euler_piston(true);
}

void expect_wall_vector(const std::array<real, 3>& value, const std::array<double, 3>& expected) {
    for (int axis = 0; axis < 3; ++axis)
        EXPECT_NEAR(value[axis], expected[axis], 16 * std::numeric_limits<real>::epsilon());
}

TEST(SphCallerEulerWall, ConstantTranslationAndRotationPreserveRigidGeometry) {
    const Fields empty({}), body({{{2, -1, .5}, 1, 41, 1}, {{3, -1, .5}, 1, 42, 1},
                                   {{2, 1, .5}, 1, 43, 1}}),
        prescribed({{{5, 0, 0}, 1, 51, 1}});
    const WallMotion initial{.translation = {-1, .5, .25}, .velocity = {.5, -.25, .125},
                             .center = {2, -1, .5}, .axis = {0, 0, 1},
                             .angle = .25, .angular_speed = .75};
    CallerEulerWall owner{.accepted = initial};
    auto motion = stationary_motion(.5);
    std::vector<double> sampled;
    motion.motion = [&](WallTime time, double) {
        sampled.push_back(time.elapsed);
        return owner.sample(time);
    };
    auto analytic = stationary_motion(.5);
    analytic.motion = [](WallTime time, double) -> Result<WallMotion> {
        WallMotion result;
        result.translation[2] = time.elapsed * time.elapsed;
        result.velocity[2] = 2 * time.elapsed;
        result.acceleration[2] = 2;
        return result;
    };
    // Register the prescribed group first to exercise the group-to-sample mapping.
    const WallGroups groups{{7, wall_fields(prescribed), analytic}, {3, wall_fields(body), motion}};
    sph::WcsphParticipant3D participant(empty.get(), 128, {{1, 1, 1}, 1}, {}, {}, groups);
    Simulation simulation(participant, 8);
    ASSERT_TRUE(simulation.initialize().ok());
    std::vector<double> expected_samples{0};
    for (const double time : {0., .125, .375, .4375}) {
        if (time > 0) {
            const double dt = time - expected_samples.back();
            owner.stage(time, dt, initial);
            const WallTime request{0, time};
            const auto first = checked(motion.motion(request, time));
            const auto second = checked(motion.motion(request, time));
            EXPECT_EQ(first.translation, second.translation);
            EXPECT_EQ(first.angle, second.angle);
            EXPECT_EQ(first.velocity, second.velocity);
            EXPECT_EQ(first.angular_speed, second.angular_speed);
            expected_samples.insert(expected_samples.end(), 2, time);
            ASSERT_TRUE(simulation.run(dt).ok());
            owner.publish();
            expected_samples.push_back(time);
        }
        EXPECT_EQ(sampled, expected_samples);
        const auto center = checked(participant.read_committed_wall(41));
        const auto x_arm = checked(participant.read_committed_wall(42));
        const auto y_arm = checked(participant.read_committed_wall(43));
        const double x = 1 + .5 * time, y = -.5 - .25 * time, z = .75 + .125 * time;
        const double angle = .25 + .75 * time;
        expect_wall_vector(center.position, {x, y, z});
        expect_wall_vector(x_arm.position, {x + std::cos(angle), y + std::sin(angle), z});
        expect_wall_vector(y_arm.position, {x - 2 * std::sin(angle), y + 2 * std::cos(angle), z});
        EXPECT_NEAR(std::hypot(x_arm.position[0] - y_arm.position[0], x_arm.position[1] - y_arm.position[1]),
                    std::sqrt(5.), 16 * std::numeric_limits<real>::epsilon());
        EXPECT_EQ(x_arm.stamp.time.physical_time, 8 + time);
        EXPECT_EQ(checked(participant.read_committed_wall(51)).position[2], time * time);
    }
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphLeptonWall, LinearVelocityAndAngularSpeedConvergeAtFirstOrder) {
    const Fields empty({}), walls({{{0, 0, 0}, 1, 41, 1}, {{1, 0, 0}, 1, 42, 1}});
    app::MotionConfig config{.mode = app::MotionMode::velocity};
    config.expressions = {"1+t", "0", "0", "0.5+t"};
    config.axis = {0, 0, 1};
    double previous_x_error{}, previous_angle_error{};
    for (const int steps : {4, 8, 16}) {
        const auto owner = checked(app::PrescribedMotion::create(config));
        const WallGroups groups{{3, wall_fields(walls), owner->trajectory(.25)}};
        sph::WcsphParticipant3D participant(empty.get(), 128, {{1, 1, 1}, 1}, {}, {}, groups);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        const double dt = 1. / steps;
        for (int step = 1; step <= steps; ++step) {
            ASSERT_TRUE(owner->prepare(dt).ok());
            ASSERT_TRUE(simulation.run(dt).ok());
            owner->commit();
        }
        const auto center = checked(participant.read_committed_wall(41)).position;
        const auto arm = checked(participant.read_committed_wall(42)).position;
        const double angle = std::atan2(double(arm[1]) - center[1], double(arm[0]) - center[0]);
        // Integral of 1+t on [0,1] is 1.5; integral of .5+t is 1 radian.
        const double x_error = 1.5 - center[0], angle_error = 1. - angle;
        EXPECT_EQ(x_error, .5 / steps);
        EXPECT_NEAR(angle_error, .5 / steps, 8 * std::numeric_limits<real>::epsilon());
        if (previous_x_error) {
            EXPECT_EQ(previous_x_error / x_error, 2.);
            EXPECT_NEAR(previous_angle_error / angle_error, 2., 2.e-5);
        }
        ::testing::Test::RecordProperty("translation_error_" + std::to_string(steps), std::to_string(x_error));
        ::testing::Test::RecordProperty("angle_error_" + std::to_string(steps), std::to_string(angle_error));
        previous_x_error = x_error;
        previous_angle_error = angle_error;
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphLeptonWall, FailedAdamiEndpointPreservesPoseAndRatesForShorterRetry) {
    const Fields fluid({{{0, 0, 0}, .125, 11, 0}}), body({{{1, 0, 0}, .125, 41, 1}});
    const sph::Velocity3D velocity("test.retry_velocity", 1);
    sph::Values density("test.retry_density", 1);
    density(0) = 1;
    app::MotionConfig config{.mode = app::MotionMode::velocity};
    config.expressions[0] = "0.125+4*t^2";
    const auto owner = checked(app::PrescribedMotion::create(config));
    const auto reference_owner = checked(app::PrescribedMotion::create(config));
    const auto motion = owner->trajectory(.5);
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, 1};
    sph::WcsphParticipant3D participant(fluid.get(), 32, parameters, velocity, density,
                                        WallGroups{{7, wall_fields(body), motion}});
    sph::WcsphParticipant3D reference(fluid.get(), 32, parameters, velocity, density,
                                      WallGroups{{7, wall_fields(body), reference_owner->trajectory(.5)}});
    ASSERT_TRUE(participant.initialize(0).ok());
    ASSERT_TRUE(reference.initialize(0).ok());
    constexpr double first = 1. / 16, large = .25, small = 1. / 64;
    const auto step = [](sph::WcsphParticipant3D& solver, app::PrescribedMotion& driver,
                         double start, double dt) {
        if (const auto status = driver.prepare(dt); !status.ok())
            return status;
        const auto status = solver.step({start, start + dt, dt});
        if (status.ok())
            driver.commit();
        else
            driver.discard();
        return status;
    };
    ASSERT_TRUE(step(participant, *owner, 0, first).ok());
    ASSERT_TRUE(step(reference, *reference_owner, 0, first).ok());
    const auto identity = WcsphTestAccess::accepted_identity(participant);
    const auto before = committed(participant);
    const auto before_wall = checked(participant.read_committed_wall(41));
    const auto before_rates = WcsphTestAccess::rates(participant);
    const auto before_motion = checked(motion.motion({0, first}, first));
    // At t=5/16, the wall's positive acceleration gives a negative inverse-EOS
    // density after SPH samples the staged endpoint.
    const auto failure = step(participant, *owner, first, large);
    ASSERT_EQ(failure.code(), StatusCode::numerical_failure) << failure.message();
    EXPECT_EQ(failure.message(), "invalid Adami wall pressure or inverse Tait density");
    EXPECT_FALSE(motion.motion({0, first + large}, first + large).ok());
    const auto restored = checked(motion.motion({0, first}, first));
    EXPECT_EQ(restored.translation, before_motion.translation);
    EXPECT_EQ(restored.velocity, before_motion.velocity);
    EXPECT_EQ(restored.acceleration, before_motion.acceleration);
    EXPECT_EQ(WcsphTestAccess::accepted_identity(participant), identity);
    EXPECT_EQ(WcsphTestAccess::rates(participant), before_rates);
    expect_same_rows(committed(participant), before);
    EXPECT_EQ(checked(participant.read_committed_wall(41)), before_wall);
    ASSERT_TRUE(participant.release_workspace().ok());
    for (const double start : {first, first + small}) {
        ASSERT_TRUE(step(reference, *reference_owner, start, small).ok());
        const auto retry = step(participant, *owner, start, small);
        ASSERT_TRUE(retry.ok()) << retry.message();
        expect_same_rows(committed(participant), committed(reference));
        EXPECT_EQ(checked(participant.read_committed_wall(41)), checked(reference.read_committed_wall(41)));
        EXPECT_EQ(WcsphTestAccess::rates(participant), WcsphTestAccess::rates(reference));
    }
    ASSERT_TRUE(participant.finalize().ok());
    ASSERT_TRUE(reference.finalize().ok());
}

TEST(SphLeptonWall, RejectsInvalidExpressionAndUnknownVariable) {
    app::MotionConfig config{.mode = app::MotionMode::pose};
    for (const char* input : {"sin(", "t+unknown", "1e999"}) {
        SCOPED_TRACE(input);
        config.expressions[0] = input;
        const auto result = app::PrescribedMotion::create(config);
        ASSERT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::invalid_argument);
    }
}

TEST(SphLeptonWall, SamplesAngularDerivativesAtPhysicalTime) {
    app::MotionConfig config{.mode = app::MotionMode::pose};
    config.initial_time = 2;
    config.axis = {0, 0, 1};
    config.expressions[3] = "t^3";
    const auto owner = checked(app::PrescribedMotion::create(config));
    const auto trajectory = owner->trajectory(.5);
    const auto initial = checked(trajectory.motion({0, 0}, 0));
    EXPECT_EQ(initial.angle, 8);
    ASSERT_TRUE(owner->prepare(.5).ok());
    for (int sample = 0; sample < 2; ++sample) {
        const auto motion = checked(trajectory.motion({0, .5}, .5));
        EXPECT_EQ(motion.angle, 15.625);
        EXPECT_EQ(motion.angular_speed, 18.75);
        EXPECT_EQ(motion.angular_acceleration, 15);
        EXPECT_EQ(motion.axis, (std::array<real, 3>{0, 0, 1}));
        EXPECT_EQ(checked(trajectory.motion({0, 0}, 0)).angle, initial.angle);
    }
    owner->commit();
    EXPECT_FALSE(trajectory.motion({0, 0}, 0).ok());
    EXPECT_EQ(checked(trajectory.motion({0, .5}, .5)).angle, 15.625);
}

TEST(SphLeptonWall, MatchesAcceptedPhysicalClockAtLargeOrigin) {
    app::MotionConfig config{.mode = app::MotionMode::pose};
    config.initial_time = 0x1p52;
    config.expressions[0] = "t-4503599627370496";
    const auto owner = checked(app::PrescribedMotion::create(config));
    const auto motion = owner->trajectory(1).motion;
    double physical_time = config.initial_time;
    for (int step = 1; step <= 3; ++step) {
        constexpr double dt = .75;
        physical_time += dt;
        ASSERT_TRUE(owner->prepare(dt).ok());
        owner->commit();
        const double elapsed = step * dt;
        EXPECT_EQ(checked(motion({0, elapsed}, elapsed)).translation[0], physical_time - config.initial_time);
        EXPECT_EQ(physical_time - config.initial_time, step);
    }
}

TEST(SphLeptonWall, FailedPrepareDiscardsCandidateAndPreservesInitialPose) {
    app::MotionConfig config{.mode = app::MotionMode::velocity};
    config.expressions = {"0.5", "1/(1-t)", "-2", "2"};
    config.initial_pose = {1, 2, 3, .125};
    const auto owner = checked(app::PrescribedMotion::create(config));
    const auto motion = owner->trajectory(1).motion;
    const auto initial = checked(motion({0, 0}, 0));
    EXPECT_EQ(initial.translation, (std::array<real, 3>{1, 2, 3}));
    EXPECT_EQ(initial.angle, .125);
    ASSERT_TRUE(owner->prepare(.5).ok());
    EXPECT_EQ(checked(motion({0, .5}, .5)).translation, (std::array<real, 3>{1.25, 2.5, 2}));
    EXPECT_EQ(owner->prepare(1).code(), StatusCode::numerical_failure);
    owner->commit();
    EXPECT_FALSE(motion({0, .5}, .5).ok());
    EXPECT_FALSE(motion({0, 1}, 1).ok());
    EXPECT_EQ(checked(motion({0, 0}, 0)).translation, initial.translation);
    ASSERT_TRUE(owner->prepare(.25).ok());
    owner->commit();
    const auto shorter = checked(motion({0, .25}, .25));
    EXPECT_EQ(shorter.translation, (std::array<real, 3>{1.125, 2.25, 2.5}));
    EXPECT_EQ(shorter.angle, .625);
    EXPECT_EQ(shorter.velocity, (std::array<real, 3>{.5, static_cast<real>(4. / 3), -2}));
    EXPECT_EQ(shorter.angular_speed, 2);
}

TEST(SphVerlet, MovingWallsSampleTranslationAndRotationAtGroupPhase) {
    const Fields empty({}), translated({{{-4, .25, 0}, 2, 10, 1}}),
        rotated({{{1, 0, 0}, 3, 20, 1}});
    constexpr double dt = .25, initial_time = 12., rotation_offset = .5, translation_offset = 1.25;
    const sph::WcsphVerletParameters parameters{{1, .5, 7}, 1};
    const sph::Velocity3D velocity("test.empty_velocity", 0);
    const sph::Values density("test.empty_density", 0);
    auto translation = stationary_motion(.5);
    translation.initial_phase = translation_offset;
    translation.motion = [](WallTime, double phase) -> Result<WallMotion> {
        WallMotion motion;
        motion.translation = {static_cast<real>(.5 * phase * phase), static_cast<real>(-.25 * phase), 0};
        motion.velocity = {static_cast<real>(phase), -.25, 0};
        motion.acceleration = {1, 0, 0};
        return motion;
    };
    auto rotation = stationary_motion(.5);
    rotation.initial_phase = rotation_offset;
    rotation.motion = [](WallTime, double phase) -> Result<WallMotion> {
        WallMotion motion;
        motion.axis = {0, 0, 1};
        motion.angle = phase + .5 * phase * phase;
        motion.angular_speed = 1 + phase;
        motion.angular_acceleration = 1;
        return motion;
    };

    const WallGroups groups{{31, wall_fields(translated), translation},
                            {41, wall_fields(rotated), rotation}};
    sph::WcsphParticipant3D participant(empty.get(), 128, parameters, velocity, density, groups);
    Simulation simulation(participant, initial_time);
    ASSERT_TRUE(simulation.initialize().ok());

    for (int step = 0; step <= 1; ++step) {
        const double elapsed = step * dt;
        const double translation_phase = translation_offset + elapsed;
        const double rotation_phase = rotation_offset + elapsed;
        const double angle = rotation_phase * (1 + .5 * rotation_phase);
        const double angular_speed = 1 + rotation_phase;
        const double x = std::cos(angle), y = std::sin(angle);
        const std::array<double, 3> translation_position{
            -4 + .5 * translation_phase * translation_phase, .25 - .25 * translation_phase, 0};
        const std::array<double, 3> rotation_position{x, y, 0};
        const auto committed_left = checked(participant.read_committed_wall(10));
        const auto committed_right = checked(participant.read_committed_wall(20));
        expect_wall_vector(committed_left.position, translation_position);
        expect_wall_vector(committed_right.position, rotation_position);

        const std::array<real, 3> reference{1, 0, 0};
        wall::SampledWallMotion3D spin(checked(rotation.motion({rotation_offset, elapsed}, rotation_phase)),
                                       wall::WallMotionUse3D::full);
        const auto v = spin.velocity(reference);
        ASSERT_TRUE(v);
        expect_wall_vector(*v, {-angular_speed * y, angular_speed * x, 0});
        expect_wall_vector(spin.acceleration(reference), {-y - angular_speed * angular_speed * x,
                                                           x - angular_speed * angular_speed * y, 0});
        EXPECT_EQ(committed_left.phase, translation_phase);
        EXPECT_EQ(committed_right.phase, rotation_phase);
        EXPECT_EQ(committed_left.mass, 2);
        EXPECT_EQ(committed_right.mass, 3);
        expect_metadata(participant, simulation, initial_time + elapsed, step, .5);
        if (step == 0)
            ASSERT_TRUE(simulation.run(dt).ok());
    }
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphVerlet, MovingWallRejectsInvalidGroupsAndSamples) {
    const Fields fluid({{{0, 0, 0}, 1, 11, 0}});
    const Fields wall_a({{{1, 0, 0}, 1, 21, 1}}), wall_b({{{3, 0, 0}, 2, 21, 1}}),
        duplicate_fluid({{{4, 0, 0}, 2, 11, 1}}), other_wall({{{5, 0, 0}, 3, 32, 1}});
    const sph::WcsphVerletParameters parameters{{1, .5, 7}, 1};
    const sph::Velocity3D velocity("test.velocity", 1);
    sph::Values density("test.density", 1);
    density(0) = 1;
    const auto rejects = [&](WallGroups groups) {
        sph::WcsphParticipant3D participant(fluid.get(), 64, parameters, velocity, density, groups);
        Simulation simulation(participant);
        const auto result = simulation.initialize();
        EXPECT_FALSE(result.ok());
        EXPECT_FALSE(participant.read_committed_wall(21).ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
        EXPECT_TRUE(simulation.finalize().ok());
    };
    const auto legal = stationary_motion(.5);
    rejects({{1, wall_fields(wall_a), std::nullopt}, {2, wall_fields(wall_b), legal}});
    rejects({{1, wall_fields(duplicate_fluid), legal}});
    rejects({{1, wall_fields(wall_a), legal}, {1, wall_fields(other_wall), std::nullopt}});
    auto invalid = legal;
    invalid.motion = [](WallTime, double) -> Result<WallMotion> {
        WallMotion motion;
        motion.acceleration[0] = std::numeric_limits<double>::quiet_NaN();
        return motion;
    };
    rejects({{1, wall_fields(wall_a), invalid}});
    invalid = legal;
    invalid.motion = {};
    rejects({{1, wall_fields(wall_a), invalid}});
    std::size_t samples = 0;
    for (const double limit : {0., -.25, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
        SCOPED_TRACE(limit);
        auto invalid_limit = stationary_motion(limit);
        invalid_limit.motion = [&](WallTime, double) -> Result<WallMotion> {
            ++samples;
            return WallMotion{};
        };
        rejects({{1, wall_fields(wall_a), invalid_limit}});
    }
    EXPECT_EQ(samples, 0u);
}

TEST(SphCallerEulerWall, InvalidEndpointRatesPreserveWallForShorterRetry) {
    const Fields empty({}), boundary({{{0, 0, 0}, 1, 21, 1}});
    const WallMotion uniform{.velocity = {.5, 0, 0}};
    CallerEulerWall owner{.accepted = uniform};
    auto motion = stationary_motion(.25);
    motion.motion = [&](WallTime time, double) { return owner.sample(time); };
    sph::WcsphParticipant3D participant(empty.get(), 16, {{1, 1, 1}, 1}, {}, {},
                                        WallGroups{{1, wall_fields(boundary), motion}});
    ASSERT_TRUE(participant.initialize(0).ok());
    owner.stage(.125, .125, uniform);
    ASSERT_TRUE(participant.step({0, .125, .125}).ok());
    owner.publish();
    const auto before = checked(participant.read_committed_wall(21));

    for (bool invalid_spin : {false, true}) {
        SCOPED_TRACE(invalid_spin);
        auto invalid = uniform;
        const auto infinity = std::numeric_limits<real>::infinity();
        if (invalid_spin)
            invalid.angular_speed = infinity;
        else
            invalid.velocity[0] = infinity;
        owner.stage(.25, .125, invalid);
        const auto status = participant.step({.125, .25, .125});
        EXPECT_EQ(status.code(), StatusCode::numerical_failure);
        EXPECT_EQ(status.message(), "nonfinite caller Euler rate");
        EXPECT_EQ(checked(participant.read_committed_wall(21)), before);
        EXPECT_EQ(owner.accepted_time, .125);
    }
    owner.stage(.1875, .0625, uniform);
    ASSERT_TRUE(participant.step({.125, .1875, .0625}).ok());
    owner.publish();
    EXPECT_EQ(checked(participant.read_committed_wall(21)).position[0], .5 * .1875);
    EXPECT_TRUE(participant.finalize().ok());
}

TEST(SphVerlet, UnusedMovingWallKinematicsDoNotRejectTheStep) {
    const Fields fluid({{{0, 0, 0}, 1, 11, 0}}), boundary({{{2, 0, 0}, 1, 21, 1}});
    const sph::Velocity3D velocity("test.velocity", 1);
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto run = [&](bool invalid_velocity) {
        SCOPED_TRACE(invalid_velocity);
        bool initialized = false;
        auto trajectory = stationary_motion(.25);
        trajectory.motion = [&](WallTime time, double) -> Result<WallMotion> {
            WallMotion m;
            // The wall enters support on the second endpoint. Until then neither
            // continuity nor Adami consumes this particle's velocity or acceleration.
            m.translation[0] = initialized && time.elapsed == 0 ? nan : (time.elapsed == .25 ? -.5 : 0.);
            m.velocity[0] = invalid_velocity || !initialized ? nan : 0.;
            m.acceleration[0] = invalid_velocity ? 0. : nan;
            return m;
        };
        sph::WcsphParticipant3D participant(fluid.get(), 16, {{1, .5, 1}, 1},
            velocity, fluid.mass, WallGroups{{1, wall_fields(boundary), trajectory}});
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        initialized = true;
        const auto first = simulation.run(.125);
        ASSERT_TRUE(first.ok()) << first.message();
        const auto grid = committed_grid_sample(participant);
        EXPECT_EQ(grid.neighbors.at(11), (NeighborPairs{{21, 4.}}));
        const auto committed_wall = checked(participant.read_committed_wall(21));
        EXPECT_EQ(committed_wall.position[0], 2);
        EXPECT_EQ(simulation.run(.125).code(), StatusCode::numerical_failure);
        EXPECT_EQ(checked(participant.read_committed_wall(21)), committed_wall);
        EXPECT_EQ(simulation.time().accepted_steps, 1u);
        EXPECT_TRUE(simulation.finalize().ok());
    };
    for (bool invalid_velocity : {false, true})
        ASSERT_NO_FATAL_FAILURE(run(invalid_velocity));
}

TEST(SphVerlet, MovingWallPositionFailurePreservesTheLastGridAndClock) {
    const Fields empty({}), boundary({{{1, 0, 0}, 1, 21, 1}});
    const sph::Velocity3D velocity("test.empty_velocity", 0);
    for (bool bad_initial : {false, true}) {
        auto trajectory = stationary_motion(.25);
        trajectory.motion = [bad_initial](WallTime time, double) -> Result<WallMotion> {
            WallMotion motion;
            if (bad_initial || time.elapsed >= .25)
                motion.translation[0] = std::numeric_limits<double>::infinity();
            return motion;
        };
        sph::WcsphParticipant3D participant(empty.get(), 8, {{1, 1, 1}, 1}, velocity, empty.mass,
                                            WallGroups{{1, wall_fields(boundary), trajectory}});
        Simulation simulation(participant);
        const auto initial = simulation.initialize();
        if (bad_initial) {
            EXPECT_EQ(initial.code(), StatusCode::invalid_argument);
            EXPECT_FALSE(participant.read_committed_wall(21).ok());
        } else {
            ASSERT_TRUE(initial.ok()) << initial.message();
            ASSERT_TRUE(simulation.run(.125).ok());
            const auto before = checked(participant.read_committed_wall(21));
            const auto stamp = WcsphTestAccess::stamp(participant);
            EXPECT_EQ(simulation.run(.125).code(), StatusCode::numerical_failure);
            EXPECT_EQ(checked(participant.read_committed_wall(21)), before);
            EXPECT_EQ(WcsphTestAccess::stamp(participant), stamp);
            EXPECT_EQ(simulation.time().accepted_steps, 1u);
            EXPECT_EQ(simulation.time().accepted_time, .125);
        }
        EXPECT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphVerlet, MotionUsesRotationGeometryOnlyWhenRotating) {
    const std::array<real, 3> reference{1, 0, 0};
    WallMotion motion;
    motion.translation[0] = .5;
    motion.center[0] = motion.axis[1] = std::numeric_limits<real>::quiet_NaN();
    wall::SampledWallMotion3D translated(motion, wall::WallMotionUse3D::full);
    EXPECT_EQ(translated.position(reference), (std::array<real, 3>{1.5, 0, 0}));
    motion.angular_speed = 1;
    wall::SampledWallMotion3D bad_axis(motion, wall::WallMotionUse3D::full);
    EXPECT_FALSE(bad_axis.velocity(reference));

    for (const real z : {real{0}, std::numeric_limits<real>::infinity(), std::numeric_limits<real>::quiet_NaN(),
                         std::numeric_limits<real>::max(), std::numeric_limits<real>::denorm_min()}) {
        WallMotion rotation;
        rotation.axis = {0, 0, z};
        rotation.angular_speed = 1;
        wall::SampledWallMotion3D spin(rotation, wall::WallMotionUse3D::full);
        const auto velocity = spin.velocity(reference);
        if (z > 0 && std::isfinite(z)) {
            ASSERT_TRUE(velocity);
            EXPECT_EQ(*velocity, (std::array<real, 3>{0, 1, 0}));
            EXPECT_EQ(spin.acceleration(reference), (std::array<real, 3>{-1, 0, 0}));
        } else {
            EXPECT_FALSE(velocity);
        }
    }
}

TEST(SphVerlet, MovingWallLocalDurationSurvivesLargeAbsoluteTime) {
    constexpr double dt = .75 + 0x1p-51, large_time = 0x1p52;
    constexpr real solver_dt = static_cast<real>(dt);
    const Fields fluid({{{0, 0, 0}, 2, 11, 0}}), moving({{{-.5, 0, 0}, 3, 22, 1}});
    sph::Velocity3D velocity("test.uniform_velocity", 1);
    velocity(0, 0) = 1;
    sph::Values density("test.uniform_density", 1);
    density(0) = 1;
    const sph::WcsphVerletParameters parameters{{1, .125, 1}, 1};
    TrajectoryCalls calls;
    auto trajectory = stationary_motion(1);
    trajectory.initial_phase = .375;
    trajectory.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
        calls.push_back({time, phase});
        WallMotion result;
        result.translation[0] = static_cast<real>(time.elapsed);
        result.velocity[0] = 1;
        return result;
    };
    const WallGroups groups{{5, wall_fields(moving), trajectory}};
    sph::WcsphParticipant3D small(fluid.get(), 64, parameters, velocity, density, groups);
    sph::WcsphParticipant3D large(fluid.get(), 64, parameters, velocity, density, groups);
    Simulation small_sim(small), large_sim(large, large_time);
    ASSERT_TRUE(small_sim.initialize().ok());
    ASSERT_TRUE(large_sim.initialize().ok());
    const auto check = [&](Simulation& simulation) {
        calls.clear();
        ASSERT_TRUE(simulation.run(dt).ok());
        ASSERT_EQ(calls.size(), 1u);
        EXPECT_EQ(calls[0].time.initial_phase, trajectory.initial_phase);
        EXPECT_EQ(calls[0].time.elapsed, dt);
        EXPECT_EQ(calls[0].phase, trajectory.initial_phase + dt);
    };
    ASSERT_NO_FATAL_FAILURE(check(small_sim));
    ASSERT_NO_FATAL_FAILURE(check(large_sim));
    EXPECT_EQ(small_sim.time().accepted_time, dt);
    EXPECT_EQ(large_sim.time().accepted_time, large_time + 1);

    const auto expected = committed(small);
    const auto actual = committed(large);
    expect_same_rows(actual, expected);
    ASSERT_EQ(actual.rows.size(), 1u);
    EXPECT_EQ(actual.rows.front().position[0], solver_dt);
    EXPECT_EQ(actual.rows.front().pressure, 0);
    EXPECT_EQ(actual.rows.front().density, 1);
    EXPECT_EQ(actual.rows.front().velocity, (std::array<real, 3>{1, 0, 0}));
    const auto wall_small = checked(small.read_committed_wall(22));
    const auto wall_large = checked(large.read_committed_wall(22));
    EXPECT_EQ(wall_large.position, wall_small.position);
    EXPECT_EQ(wall_large.position[0], solver_dt - real{.5});
    EXPECT_EQ(wall_large.phase, wall_small.phase);
    EXPECT_EQ(WcsphTestAccess::wall_groups(small), WcsphTestAccess::wall_groups(large));
    const auto small_grid = committed_grid_sample(small);
    EXPECT_EQ(small_grid.neighbors, committed_grid_sample(large).neighbors);
    EXPECT_EQ(small_grid.neighbors.at(11), (NeighborPairs{{22, .25}}));
    ASSERT_TRUE(small_sim.finalize().ok());
    ASSERT_TRUE(large_sim.finalize().ok());
}

TEST(SphVerlet, MovingWallLimitsComposeAndUseNextEndpointBound) {
    const Fields fluid({{{0, 0, 0}, 1, 11, 0}}), left({{{-6, 0, 0}, 2, 31, 1}}),
        right({{{6, 0, 0}, 3, 32, 1}});
    sph::Velocity3D velocity("test.fluid_velocity", 1);
    velocity(0, 0) = .25;
    sph::Values density("test.fluid_density", 1);
    density(0) = 1;
    const sph::WcsphVerletParameters parameters{{1, .25, 1}, 1};
    constexpr double acoustic = 2;
    for (const auto limits : std::array{std::array{.125, .25}, std::array{.25, .125},
                                       std::array{3., 4.}}) {
        const WallGroups groups{{5, wall_fields(left), stationary_motion(limits[0])},
                                {7, wall_fields(right), stationary_motion(limits[1])}};
        const auto maximum = std::min({acoustic, limits[0], limits[1]});
        sph::WcsphParticipant3D participant(fluid.get(), 64, parameters, velocity, density, groups);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        expect_metadata(participant, simulation, 0, 0, maximum);
        ASSERT_TRUE(simulation.run(.125).ok());
        expect_metadata(participant, simulation, .125, 1, maximum);
        const auto groups_after = WcsphTestAccess::wall_groups(participant);
        EXPECT_EQ(groups_after.at(5), (WcsphTestAccess::WallGroupValue{0, .125, limits[0]}));
        EXPECT_EQ(groups_after.at(7), (WcsphTestAccess::WallGroupValue{0, .125, limits[1]}));
        ASSERT_TRUE(simulation.finalize().ok());
    }

    auto finite = stationary_motion(1);
    finite.last_phase = .75;
    TrajectoryCalls calls;
    finite.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
        calls.push_back({time, phase});
        return WallMotion{};
    };
    const WallGroups groups{{5, wall_fields(left), finite},
                            {7, wall_fields(right), stationary_motion(4)}};
    sph::WcsphParticipant3D participant(fluid.get(), 64, parameters, velocity, density, groups);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    expect_metadata(participant, simulation, 0, 0, .75);
    const auto above = std::nextafter(.25, std::numeric_limits<double>::infinity());
    const std::array requests{.5, above, .25};
    EXPECT_EQ(simulation.run(requests).code(), StatusCode::invalid_argument);
    expect_metadata(participant, simulation, .5, 1, .25);
    EXPECT_EQ(WcsphTestAccess::wall_groups(participant).at(5),
              (WcsphTestAccess::WallGroupValue{0, .5, .25}));
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].time.elapsed, 0.);
    EXPECT_EQ(calls[0].phase, 0.);
    EXPECT_EQ(calls[1].time.elapsed, .5);
    EXPECT_EQ(calls[1].phase, .5);
    const auto after = committed(participant);
    ASSERT_EQ(after.rows.size(), 1u);
    EXPECT_EQ(after.rows[0].position[0], .125);
    EXPECT_EQ(after.rows[0].velocity[0], .25);
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphVerlet, MovingWallOnlyPublishesCrossingAndGroupPhases) {
    const Fields empty({}), left({{{1.5, 0, 0}, 2, 80, 1}}), right({{{2.5, 0, 0}, 3, 10, 1}});
    const sph::Velocity3D velocity("test.empty_velocity", 0);
    const sph::Values density("test.empty_density", 0);
    const sph::WcsphVerletParameters parameters{{1, .5, 7}, .5};
    auto left_motion = stationary_motion(.5);
    auto right_motion = stationary_motion(.75);
    left_motion.initial_phase = .25;
    right_motion.initial_phase = .75;
    left_motion.motion = [](WallTime time, double) -> Result<WallMotion> {
        WallMotion result;
        result.translation[0] = 2 * time.elapsed;
        result.velocity[0] = 2;
        return result;
    };
    right_motion.motion = [](WallTime time, double) -> Result<WallMotion> {
        WallMotion result;
        result.translation[0] = -2 * time.elapsed;
        result.velocity[0] = -2;
        return result;
    };
    left_motion.last_phase = 1.125;
    const WallGroups groups{{4, wall_fields(left), left_motion},
                            {7, wall_fields(right), right_motion}};
    sph::WcsphParticipant3D participant(empty.get(), 64, parameters, velocity, density, groups);
    StepAllocations observed(participant);
    Simulation simulation(participant, 3);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto initial = WcsphTestAccess::stamp(participant);
    const auto check = [&](double elapsed, std::size_t steps, double maximum_step, std::size_t layout_changes) {
        expect_metadata(participant, simulation, 3 + elapsed, steps, maximum_step);
        const auto state = committed(participant);
        EXPECT_TRUE(state.rows.empty());
        EXPECT_EQ(state.stamp.geometry.value(), initial.geometry.value() + steps);
        EXPECT_EQ(state.stamp.layout.value(), initial.layout.value() + layout_changes);
        EXPECT_EQ(state.stamp.topology, initial.topology);
        const auto l = checked(participant.read_committed_wall(80));
        const auto r = checked(participant.read_committed_wall(10));
        EXPECT_EQ(l.position, (std::array<real, 3>{static_cast<real>(1.5 + 2 * elapsed), 0, 0}));
        EXPECT_EQ(r.position, (std::array<real, 3>{static_cast<real>(2.5 - 2 * elapsed), 0, 0}));
        EXPECT_EQ(l.id, 80u);
        EXPECT_EQ(r.id, 10u);
        EXPECT_EQ(l.mass, 2);
        EXPECT_EQ(r.mass, 3);
        EXPECT_EQ(l.phase, .25 + elapsed);
        EXPECT_EQ(r.phase, .75 + elapsed);
        EXPECT_EQ(l.stamp, state.stamp);
        EXPECT_EQ(r.stamp, state.stamp);
    };
    ASSERT_NO_FATAL_FAILURE(check(0, 0, .5, 0));
    ASSERT_EQ(committed_grid_sample(participant).particles.size(), 2u);
    EXPECT_EQ(committed_grid_sample(participant).particles.front().id, 80u);
    ASSERT_TRUE(simulation.run(.5).ok());
    ASSERT_NO_FATAL_FAILURE(check(.5, 1, .375, 1));
    ASSERT_EQ(committed_grid_sample(participant).particles.size(), 2u);
    EXPECT_EQ(committed_grid_sample(participant).particles.front().id, 10u);
    const auto accepted = WcsphTestAccess::wall_groups(participant);
    EXPECT_EQ(accepted.at(4), (WcsphTestAccess::WallGroupValue{.25, .5, .375}));
    EXPECT_EQ(accepted.at(7), (WcsphTestAccess::WallGroupValue{.75, .5, .75}));

    ASSERT_TRUE(simulation.run(.375).ok());
    // Sorted IDs remain [10,80] on the second step. The sorted-to-input
    // map returns to [0,1] because input slots follow the prior step's order.
    ASSERT_NO_FATAL_FAILURE(check(.875, 2, 0, 2));
    EXPECT_EQ(committed_grid_sample(participant).particles.front().id, 10u);
    for (const auto label : {"sph.pressure_over_density_squared", "sph.pressure.needed_ghost",
                             "sph.wall.fluid_row", "sph.wall.acceleration"})
        EXPECT_EQ(observed.count(0, label, true), 0u) << label;
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

struct MixedBoundaryState {
    std::vector<Particle> walls;
    Velocities velocity;
    std::map<std::uint64_t, double> acceleration;
};

MixedBoundaryState mixed_boundary_at(const std::vector<Particle>& fixed,
                                     const std::vector<Particle>& moving, double time) {
    MixedBoundaryState result{fixed, {}, {}};
    for (const auto& particle : fixed) {
        result.velocity.emplace(particle.id, std::array<real, 3>{});
        result.acceleration.emplace(particle.id, 0);
    }
    for (auto particle : moving) {
        particle.position[0] = static_cast<real>(particle.position[0] + time * (.75 + time / 16));
        result.walls.push_back(particle);
        result.velocity.emplace(
            particle.id, std::array<real, 3>{static_cast<real>(.75 + time / 8), 0, 0});
        result.acceleration.emplace(particle.id, .125);
    }
    return result;
}

FixedWallForce mixed_boundary_force(const Trajectory& state, const MixedBoundaryState& boundary,
                                   double gravity) {
    constexpr double c2 = .25;
    const auto force_coefficient = [](real p, real rho) { return double(p) / (double(rho) * rho); };
    std::map<std::uint64_t, double> coefficient;
    FixedWallForce result;
    for (const auto& particle : state.particles) {
        const auto rho = static_cast<real>(state.density.at(particle.id));
        const auto pressure = static_cast<real>(c2 * (rho - 1));
        result.pressure.emplace(particle.id, pressure);
        coefficient.emplace(particle.id, force_coefficient(pressure, rho));
    }
    for (const auto& particle : boundary.walls) {
        long double weighted_pressure{}, weights{};
        for (const auto& fluid : state.particles) {
            const long double delta = static_cast<long double>(particle.position[0]) - fluid.position[0];
            const auto distance = std::abs(delta);
            if (distance >= 2)
                continue;
            const long double w = std::pow(2 - distance, 3) -
                                  4 * std::pow(std::max(1 - distance, 0.L), 3);
            const auto head = (gravity - boundary.acceleration.at(particle.id)) * delta;
            weighted_pressure += w * (result.pressure.at(fluid.id) + state.density.at(fluid.id) * head);
            weights += w;
        }
        if (weights > 0) {
            const auto pressure = static_cast<real>(weighted_pressure / weights);
            const auto rho = static_cast<real>(1.L + pressure / c2);
            coefficient.emplace(particle.id, force_coefficient(pressure, rho));
        }
    }
    auto all = state.particles;
    all.insert(all.end(), boundary.walls.begin(), boundary.walls.end());
    for (const auto& p : state.particles) {
        long double force{};
        for (const auto& neighbor : all) {
            const auto d = static_cast<long double>(p.position[0]) - neighbor.position[0];
            const auto r = std::abs(d);
            if (r == 0 || r >= 2)
                continue;
            const long double radial = r < 1 ? 3 - 2.25L * r : .75L * (2 - r) * (2 - r) / r;
            force += (coefficient.at(p.id) + coefficient.at(neighbor.id)) * neighbor.mass *
                     radial * d / std::acos(-1.L);
        }
        result.acceleration.emplace(p.id, static_cast<real>(force));
    }
    return result;
}

FixedWallStep mixed_boundary_step(const Trajectory& state, const MixedBoundaryState& before,
                                  const MixedBoundaryState& after, double gravity, double dt) {
    return wall_reference_step(state, mixed_boundary_force(state, before, gravity).acceleration,
                               gravity, dt, before.walls, before.velocity, after.walls, after.velocity);
}

std::vector<Particle> mixed_boundary_order(const Trajectory& stage, const MixedBoundaryState& boundary) {
    auto result = stage.particles;
    result.insert(result.end(), boundary.walls.begin(), boundary.walls.end());
    const auto key = [](const Particle& p) { return std::pair{std::floor(double(p.position[0]) / 2), p.id}; };
    std::sort(result.begin(), result.end(), [&](const auto& a, const auto& b) { return key(a) < key(b); });
    return result;
}

void expect_mixed_query(const StageSample& sample, const Trajectory& fluid,
                        const MixedBoundaryState& walls) {
    constexpr double epsilon = std::numeric_limits<real>::epsilon();
    const auto expected = mixed_boundary_order(fluid, walls);
    ASSERT_EQ(sample.particles.size(), expected.size());
    std::vector<std::uint64_t> owned;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& center = expected[i];
        EXPECT_EQ(sample.particles[i].id, center.id);
        EXPECT_EQ(sample.particles[i].mass, center.mass);
        EXPECT_EQ(sample.particles[i].ghost, center.ghost);
        EXPECT_NEAR(sample.particles[i].position[0], center.position[0], 8 * epsilon);
        if (!center.ghost)
            owned.push_back(center.id);
        NeighborPairs expected_neighbors;
        for (const auto& neighbor : expected) {
            const double distance = std::abs(double(center.position[0]) - neighbor.position[0]);
            if (neighbor.id != center.id && distance <= 2)
                expected_neighbors.emplace_back(neighbor.id, distance * distance);
        }
        const auto& actual_neighbors = sample.neighbors.at(center.id);
        ASSERT_EQ(actual_neighbors.size(), expected_neighbors.size()) << center.id;
        for (std::size_t j = 0; j < expected_neighbors.size(); ++j) {
            EXPECT_EQ(actual_neighbors[j].first, expected_neighbors[j].first);
            EXPECT_NEAR(actual_neighbors[j].second, expected_neighbors[j].second, 32 * epsilon);
        }
    }
    EXPECT_EQ(sample.owned, owned);
}

void expect_mixed_rows(const CommittedSample& actual, const Trajectory& fluid,
                       const MixedBoundaryState& walls, double gravity) {
    const auto force = mixed_boundary_force(fluid, walls, gravity);
    ASSERT_EQ(actual.rows.size(), fluid.particles.size());
    constexpr double eps = std::numeric_limits<real>::epsilon();
    std::size_t i{};
    for (const auto& p : mixed_boundary_order(fluid, walls)) {
        if (p.ghost)
            continue;
        const auto& row = actual.rows[i++];
        ASSERT_EQ(row.id, p.id);
        EXPECT_EQ(row.mass, p.mass);
        EXPECT_NEAR(row.position[0], p.position[0], 8 * eps);
        EXPECT_NEAR(row.velocity[0], fluid.velocity.at(p.id)[0], 16 * eps);
        EXPECT_NEAR(row.density, fluid.density.at(p.id), 16 * eps);
        EXPECT_NEAR(row.pressure, force.pressure.at(p.id), 16 * eps);
        EXPECT_NEAR(row.pressure_acceleration[0], force.acceleration.at(p.id), 32 * eps);
        EXPECT_EQ(row.pressure_acceleration[1], 0);
        EXPECT_EQ(row.pressure_acceleration[2], 0);
    }
}

TEST(SphVerlet, MovingWallGroupsPreserveIdentityAcrossPermutations) {
    constexpr double dt = .25, gravity = .03125;
    const sph::WcsphVerletParameters parameters{{1, .5, 1}, 1, {gravity, 0, 0}};
    const Trajectory initial{
        {{{2.375, 0, 0}, 1.25, 11, 0}, {{1.625, 0, 0}, .875, 41, 0}},
        {{11, {-.125, 0, 0}}, {41, {.25, 0, 0}}}, {{11, 1.5}, {41, 1.25}}};
    const std::vector<Particle> fixed{{{.5, 0, 0}, 1.125, 25, 1}, {{3.25, 0, 0}, 1.5, 95, 1}};
    const std::vector<Particle> moving{{{1.875, 0, 0}, 1.75, 70, 1}, {{2.625, 0, 0}, 2.25, 80, 1}};
    const auto reversed = [](std::vector<Particle> particles) {
        std::reverse(particles.begin(), particles.end());
        return particles;
    };
    const Fields fluid(initial.particles), fluid_reversed(reversed(initial.particles));
    const Fields fixed_fields(fixed), moving_fields(moving);
    const Fields fixed_reversed(reversed(fixed)), moving_reversed(reversed(moving));
    auto second_initial = initial;
    second_initial.particles = reversed(initial.particles);
    auto density = [](const Trajectory& seed) {
        sph::Values values("test.mixed_density", seed.particles.size());
        for (std::size_t i = 0; i < seed.particles.size(); ++i)
            values(i) = static_cast<real>(seed.density.at(seed.particles[i].id));
        return values;
    };
    auto motion = stationary_motion(.5);
    motion.motion = [](WallTime, double phase) -> Result<WallMotion> {
        WallMotion result;
        result.translation[0] = .75 * phase + .0625 * phase * phase;
        result.velocity[0] = .75 + .125 * phase;
        result.acceleration[0] = .125;
        return result;
    };
    const WallGroups groups{{4, wall_fields(fixed_fields), std::nullopt},
                            {9, wall_fields(moving_fields), motion}};
    const WallGroups reversed_groups{{9, wall_fields(moving_reversed), motion},
                                     {4, wall_fields(fixed_reversed), std::nullopt}};
    sph::WcsphParticipant3D participant(fluid.get(), 64, parameters, input_velocity(initial),
                                        density(initial), groups);
    sph::WcsphParticipant3D permuted(fluid_reversed.get(), 64, parameters, input_velocity(second_initial),
                                     density(second_initial), reversed_groups);
    Simulation simulation(participant), permuted_sim(permuted);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(permuted_sim.initialize().ok());

    const auto start = mixed_boundary_at(fixed, moving, 0);
    const auto end = mixed_boundary_at(fixed, moving, dt);
    const auto expected = mixed_boundary_step(initial, start, end, gravity, dt);
    ASSERT_LT(start.walls[2].position[0], 2);
    ASSERT_GT(end.walls[2].position[0], 2);
    EXPECT_NE(start.velocity.at(70), end.velocity.at(70));
    const auto check = [&](const Trajectory& fluid_state, const MixedBoundaryState& stage) {
        const auto grid = committed_grid_sample(participant);
        expect_mixed_query(grid, fluid_state, stage);
        EXPECT_EQ(grid.neighbors, committed_grid_sample(permuted).neighbors);
        const auto observed = committed(participant);
        expect_same_rows(observed, committed(permuted));
        expect_mixed_rows(observed, fluid_state, stage, gravity);
        for (const auto& p : stage.walls) {
            const auto row = checked(participant.read_committed_wall(p.id));
            EXPECT_EQ(row, checked(permuted.read_committed_wall(p.id))) << "wall ID " << p.id;
            EXPECT_EQ(row.id, p.id);
            EXPECT_EQ(row.mass, p.mass);
            EXPECT_EQ(row.position, p.position);
        }
    };
    const auto initial_grid = committed_grid_sample(participant);
    ASSERT_NO_FATAL_FAILURE(check(initial, start));
    ASSERT_TRUE(simulation.run(dt).ok());
    ASSERT_TRUE(permuted_sim.run(dt).ok());
    ASSERT_NO_FATAL_FAILURE(check(expected.end, end));
    const auto first_grid = committed_grid_sample(participant);
    const auto rates = WcsphTestAccess::rates(participant);
    ASSERT_EQ(rates[0].size(), 2u);
    ASSERT_EQ(rates[1].size(), 2u);
    constexpr double eps = std::numeric_limits<real>::epsilon();
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(rates[0][i], expected.start_rate.at(initial_grid.owned[i]), 32 * eps);
        EXPECT_NEAR(rates[1][i], expected.end_rate.at(first_grid.owned[i]), 32 * eps);
    }
    EXPECT_EQ(WcsphTestAccess::wall_groups(participant), WcsphTestAccess::wall_groups(permuted));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_TRUE(permuted_sim.finalize().ok());

    auto prescribed_constant = groups;
    prescribed_constant[1].trajectory = stationary_motion(4);
    auto all_fixed = groups;
    all_fixed[1].trajectory.reset();
    sph::WcsphParticipant3D constant(fluid.get(), 64, parameters, input_velocity(initial),
                                     density(initial), prescribed_constant);
    sph::WcsphParticipant3D fixed_only(fluid.get(), 64, parameters, input_velocity(initial),
                                       density(initial), all_fixed);
    Simulation constant_sim(constant), fixed_sim(fixed_only);
    ASSERT_TRUE(constant_sim.initialize().ok());
    ASSERT_TRUE(fixed_sim.initialize().ok());
    expect_same_rows(committed(constant), committed(fixed_only));
    ASSERT_TRUE(constant_sim.run(dt).ok());
    ASSERT_TRUE(fixed_sim.run(dt).ok());
    expect_same_rows(committed(constant), committed(fixed_only));
    EXPECT_DOUBLE_EQ(constant.time_constraint().maximum_step,
                     fixed_only.time_constraint().maximum_step);
    ASSERT_TRUE(constant_sim.finalize().ok());
    ASSERT_TRUE(fixed_sim.finalize().ok());
}

TEST(SphVerlet, MovingWallFiniteDomainRejectsOversizedStepBeforeCallback) {
    const Fields fluid({}), particles({{{.25, 0, 0}, 1, 70, 1}});
    TrajectoryCalls calls;
    auto trajectory = stationary_motion(2);
    trajectory.initial_phase = trajectory.first_phase = -.75;
    trajectory.last_phase = .5;
    trajectory.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
        calls.push_back({time, phase});
        return WallMotion{};
    };
    sph::WcsphParticipant3D participant(fluid.get(), 1, {{1, .25, 1}, .5}, {}, {},
                                       WallGroups{{1, wall_fields(particles), trajectory}});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(participant.time_constraint().maximum_step, 1.25);
    EXPECT_EQ(simulation.run(1.5).code(), StatusCode::invalid_argument);
    EXPECT_EQ(calls.size(), 1u);
    EXPECT_EQ(simulation.time().accepted_time, 0.);
    EXPECT_EQ(simulation.time().accepted_steps, 0u);
    EXPECT_TRUE(simulation.finalize().ok());
}

TEST(SphVerlet, MovingWallPeriodicPhaseNormalizesOrdinaryTimes) {
    const Fields fluid({}), particles({{{.25, 0, 0}, 1, 70, 1}});
    struct Case { double initial, elapsed, period, expected; };
    for (const auto [initial, elapsed, period, expected] : {
             Case{0, 0, 1, 0}, Case{-2, 0, 1, 0}, Case{-2.25, 0, 1, .75},
             Case{-.25, .5, 1, .25}, Case{1.25, 0, .5, .25}, Case{-2.25, 0, 2, 1.75}}) {
        SCOPED_TRACE(initial);
        double sampled = -1;
        auto trajectory = stationary_motion(1);
        trajectory.initial_phase = initial;
        trajectory.period = period;
        trajectory.motion = [&](WallTime, double phase) -> Result<WallMotion> {
            sampled = phase;
            return WallMotion{};
        };
        sph::WcsphParticipant3D participant(fluid.get(), 1, {{1, 1, 1}, 1}, {}, {},
                                           WallGroups{{1, wall_fields(particles), trajectory}});
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        if (elapsed > 0)
            ASSERT_TRUE(simulation.run(elapsed).ok());
        EXPECT_EQ(sampled, expected);
        EXPECT_GE(sampled, 0);
        EXPECT_LT(sampled, period);
        EXPECT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphVerlet, MovingWallPeriodicFoldAndThrowingCallbackPreserveLastStep) {
    constexpr double period = 1.;
    const Fields fluid({}), particles({{{.25, 0, 0}, 1, 70, 1}});
    std::vector<double> phases;
    sph::WcsphParticipant3D* active{};
    auto trajectory = stationary_motion(1.);
    trajectory.initial_phase = -.25;
    trajectory.period = period;
    trajectory.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
        phases.push_back(phase);
        if (time.elapsed == 1.) {
            EXPECT_EQ(active->step({.5, 1., .5}).code(), StatusCode::invalid_state);
            EXPECT_EQ(active->finalize().code(), StatusCode::invalid_state);
            throw 1;
        }
        WallMotion sample;
        sample.translation[0] = time.elapsed;
        sample.velocity[0] = 1;
        return sample;
    };
    sph::WcsphParticipant3D participant(fluid.get(), 1, {{1, .25, 1}, .5}, {}, {},
                                       WallGroups{{1, wall_fields(particles), trajectory}});
    active = &participant;
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_EQ(phases.size(), 1u);
    EXPECT_GE(phases.front(), 0.);
    EXPECT_EQ(phases.front(), .75);
    EXPECT_EQ(simulation.run(.5, 3).code(), StatusCode::execution_failure);
    EXPECT_EQ(simulation.time().accepted_steps, 1u);
    EXPECT_EQ(simulation.time().accepted_time, .5);
    const auto published = checked(participant.read_committed_wall(70));
    EXPECT_EQ(published.phase, .25);
    EXPECT_EQ(published.position[0], .75);
    EXPECT_EQ(published.stamp.accepted_version.value(), 1u);
    ASSERT_EQ(phases.size(), 3u);
    EXPECT_EQ(phases.back(), .75);
    EXPECT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

TEST(SphVerlet, MovingWallRejectsReentrantSimulationMutation) {
    constexpr double dt = .5;
    const Fields fluid({}), boundary({{{.25, 0, 0}, 1, 70, 1}});
    const sph::WcsphVerletParameters parameters{{1, .25, 1}, .5};
    for (const bool propagate : {true, false}) {
        SCOPED_TRACE(propagate);
        Simulation* active{};
        std::optional<Status> rejection;
        auto motion = stationary_motion(1.);
        motion.motion = [&](WallTime time, double) -> Result<WallMotion> {
            if (time.elapsed == dt) {
                EXPECT_EQ(active->initialize().code(), StatusCode::invalid_state);
                EXPECT_EQ(active->finalize().code(), StatusCode::invalid_state);
                rejection = active->run(dt);
                EXPECT_EQ(rejection->code(), StatusCode::invalid_state);
                EXPECT_EQ(active->time().accepted_steps, 0u);
                if (propagate)
                    return *rejection;
            }
            WallMotion sample;
            sample.translation[0] = time.elapsed;
            sample.velocity[0] = 1.;
            return sample;
        };
        const WallGroups groups{{1, wall_fields(boundary), motion}};
        sph::WcsphParticipant3D participant(fluid.get(), 8, parameters, {}, {}, groups);
        Simulation simulation(participant);
        active = &simulation;
        ASSERT_TRUE(simulation.initialize().ok());
        const auto initial = checked(participant.read_committed_wall(70));
        const auto result = simulation.run(dt);
        ASSERT_TRUE(rejection.has_value());
        if (propagate) {
            EXPECT_EQ(result.code(), rejection->code());
            EXPECT_EQ(result.message(), rejection->message());
            EXPECT_EQ(simulation.time().accepted_time, 0.);
            EXPECT_EQ(simulation.time().accepted_steps, 0u);
            EXPECT_EQ(checked(participant.read_committed_wall(70)), initial);
        } else {
            ASSERT_TRUE(result.ok()) << result.message();
            EXPECT_EQ(simulation.time().accepted_time, dt);
            EXPECT_EQ(simulation.time().accepted_steps, 1u);
            EXPECT_EQ(checked(participant.read_committed_wall(70)).position[0], initial.position[0] + dt);
        }
        EXPECT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    }
}

TEST(SphVerlet, MovingWallTrajectoryStorageIsAdmittedBeforeCapture) {
    const Fields empty({}), moving_wall({{{.25, 0, 0}, 1, 82, 1}});
    const auto compact = stationary_motion(.5);
    auto large = compact;
    std::array<double, 1024> displacement{};
    displacement.back() = .125;
    large.motion = [displacement](WallTime, double) -> Result<WallMotion> {
        WallMotion motion;
        motion.translation[0] = displacement.back();
        return motion;
    };
    const auto captured_bytes = large.motion.retained_bytes() - compact.motion.retained_bytes();
    ASSERT_GT(captured_bytes, 8000u);

    const sph::WcsphVerletParameters parameters{{1, .25, 1}, .5};
    const auto make = [&](const TrajectoryInput& trajectory,
                          sph::WcsphParticipant3D::ResourceLimits limits = {}) {
        return sph::WcsphParticipant3D(empty.get(), 1, parameters, {}, {},
            WallGroups{{81, wall_fields(moving_wall), trajectory}}, ParticipantId{1}, limits);
    };
    auto baseline = make(compact);
    ASSERT_TRUE(baseline.initialize(0).ok());
    const auto compact_bytes = baseline.resource_usage().controlled_bytes;
    EXPECT_TRUE(baseline.finalize().ok());

    auto accepted = make(large);
    ASSERT_TRUE(accepted.initialize(0).ok());
    EXPECT_EQ(accepted.resource_usage().controlled_bytes, compact_bytes + captured_bytes);
    EXPECT_EQ(checked(accepted.read_committed_wall(82)).position[0], .375);
    EXPECT_TRUE(accepted.finalize().ok());
    EXPECT_EQ(accepted.resource_usage().controlled_bytes, 0u);

    auto refused = make(large, {32, 2048});
    StepAllocations events(refused);
    EXPECT_EQ(refused.initialize(0).code(), StatusCode::resource_exhausted);
    EXPECT_EQ(events.count(0, "sph.wall.particles", true), 0u);
    EXPECT_EQ(refused.resource_usage().controlled_bytes, 0u);
    EXPECT_TRUE(refused.finalize().ok());
}

} // namespace
