#include "wcsph_oracle.hpp"

#include <iomanip>
#include <nexora/simulation/Simulation.hpp>
#include <participants/sph/WcsphParticipant3D.hpp>
#include <participants/sph/WcsphStep3D.hpp>
#include <sstream>
#include <string_view>
#include <tuple>

namespace nexora::sph {

struct WcsphTestAccess {
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
    static std::array<real, 3> stage_velocity(const WcsphParticipant3D& participant,
                                              std::size_t slot) {
        const auto& source = participant.step_->workspace_.stage->values;
        return {source(slot, 0), source(slot, 1), source(slot, 2)};
    }
    static std::array<real, 3> half_velocity(const WcsphParticipant3D& participant,
                                             std::size_t slot) {
        const auto& source = participant.step_->workspace_.half->values;
        return {source(slot, 0), source(slot, 1), source(slot, 2)};
    }
    static std::array<std::vector<double>, 2> rates(const WcsphParticipant3D& participant) {
        const auto& workspace = participant.step_->workspace_;
        std::array<std::vector<double>, 2> result;
        if (!workspace.rate_n || !workspace.rate_star)
            return result;
        for (std::size_t row = 0; row < workspace.rate_n->values.extent(0); ++row)
            result[0].push_back(workspace.rate_n->values(row));
        for (std::size_t row = 0; row < workspace.rate_star->values.extent(0); ++row)
            result[1].push_back(workspace.rate_star->values(row));
        return result;
    }
    static void observe(WcsphParticipant3D& participant, void* context,
                        void (*callback)(void*, StepStage3D, const NeighborView3D&)) {
        participant.observer_context_ = context;
        participant.observe_ = callback;
        if (participant.step_) {
            participant.step_->observer_context_ = context;
            participant.step_->observe_ = callback;
        }
    }
};

} // namespace nexora::sph

namespace {
using namespace nexora;
using namespace sph_test;
using sph::StepStage3D;
using sph::WcsphTestAccess;

using NeighborPairs = std::vector<std::pair<std::uint64_t, double>>;

struct StageSample {
    StepStage3D stage;
    std::vector<Particle> particles;
    std::vector<std::uint64_t> owned;
    std::map<std::uint64_t, NeighborPairs> neighbors;
    std::map<std::uint64_t, std::array<real, 3>> velocity, half;
};

class StageSamples {
  public:
    explicit StageSamples(sph::WcsphParticipant3D& participant, bool observe_kinematics = false)
        : participant_(participant), observe_kinematics_(observe_kinematics) {
        WcsphTestAccess::observe(participant_, this, &record);
    }
    ~StageSamples() { WcsphTestAccess::observe(participant_, nullptr, nullptr); }

    StageSamples(const StageSamples&) = delete;
    StageSamples& operator=(const StageSamples&) = delete;
    std::vector<StageSample> samples;

  private:
    static void record(void* context, StepStage3D stage, const sph::NeighborView3D& query) {
        auto& observer = *static_cast<StageSamples*>(context);
        StageSample result{stage, {}, {}, {}, {}, {}};
        const auto& fields = query.fields();
        for (std::size_t slot = 0; slot < fields.ids.extent(0); ++slot) {
            const auto id = fields.ids(slot);
            result.particles.push_back(
                {{fields.position(slot, 0), fields.position(slot, 1), fields.position(slot, 2)},
                 fields.mass(slot), id, fields.ghost(slot)});
            if (observer.observe_kinematics_ && stage != StepStage3D::initial)
                result.velocity.emplace(id, WcsphTestAccess::stage_velocity(observer.participant_, slot));
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
            if (observer.observe_kinematics_ && stage == StepStage3D::start)
                result.half.emplace(id, WcsphTestAccess::half_velocity(observer.participant_, owned(row)));
            auto& pairs = result.neighbors[id];
            query.for_each_neighbor(row, [&](LocalIndex other, double distance) {
                pairs.emplace_back(fields.ids(other), distance);
            });
        }
        observer.samples.push_back(std::move(result));
    }
    sph::WcsphParticipant3D& participant_;
    bool observe_kinematics_;
};

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
    for (std::size_t i = 0; i < actual.rows.size(); ++i) {
        const auto& a = actual.rows[i];
        const auto& e = expected.rows[i];
        EXPECT_EQ(a.id, e.id);
        EXPECT_EQ(a.mass, e.mass);
        EXPECT_EQ(a.position, e.position);
        EXPECT_EQ(a.density, e.density);
        EXPECT_EQ(a.pressure, e.pressure);
        EXPECT_EQ(a.pressure_acceleration, e.pressure_acceleration);
        EXPECT_EQ(a.velocity, e.velocity);
    }
}

// The shared force oracle consumes independent host values and the callback data
// captured from the stage used by the stepper.
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
struct Trajectory {
    std::vector<Particle> particles;
    Velocities velocity;
    std::map<std::uint64_t, long double> density{};
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
Trajectory explicit_verlet(const Trajectory& accepted, sph::WcsphParameters eos, double dt) {
    // All forces originate in the independent all-pairs CSR oracle. Match the
    // public real state boundaries at each half-kick/drift/final-kick, not the
    // production cell ordering or numerical kernels.
    const auto old_force = force_oracle(accepted.particles, eos, accepted.density);
    Trajectory next = accepted;
    Velocities half, predictor;
    for (auto& particle : next.particles) {
        for (int k = 0; k < 3; ++k) {
            const real acceleration = static_cast<real>(old_force.acceleration.at(particle.id)[k]);
            predictor[particle.id][k] = static_cast<real>(
                std::fma(dt, double(acceleration), double(accepted.velocity.at(particle.id)[k])));
            half[particle.id][k] = static_cast<real>(std::fma(
                dt / 2, double(acceleration), double(accepted.velocity.at(particle.id)[k])));
            particle.position[k] = static_cast<real>(
                std::fma(dt, double(half.at(particle.id)[k]), double(particle.position[k])));
        }
    }
    for (std::size_t i = 0; i < next.particles.size(); ++i) {
        const auto id = next.particles[i].id;
        const auto before =
            density_rate(accepted.particles[i], accepted.particles, accepted.velocity);
        const auto after = density_rate(next.particles[i], next.particles, predictor);
        next.density[id] = static_cast<real>(static_cast<real>(old_force.csr.density.at(id)) +
                                             dt * .5L * (before + after));
    }
    const auto refreshed = force_oracle(next.particles, eos, next.density);
    for (const auto& particle : next.particles)
        for (int k = 0; k < 3; ++k)
            next.velocity[particle.id][k] = static_cast<real>(std::fma(
                dt / 2, double(static_cast<real>(refreshed.acceleration.at(particle.id)[k])),
                double(half.at(particle.id)[k])));
    return next;
}
struct TrajectoryErrors {
    long double position{}, velocity{}, density{}, pressure{};
    std::array<long double, 3> acceleration{};
};
void check_trajectory(const CommittedSample& sample, const Trajectory& expected,
                      sph::WcsphParameters eos, std::size_t steps,
                      TrajectoryErrors* maximum = nullptr,
                      const StageSample* stage = nullptr) {
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
    if (!stage)
        return;
    ASSERT_NO_FATAL_FAILURE(
        check_force_snapshot(ForceObservation{sample, *stage}, observed, eos, expected.density));
    // Neighbor identity must also match the independently evolved geometry.
    const auto csr = explicit_csr(expected.particles);
    for (const auto& row : sample.rows) {
        const auto index = csr.row_by_id.at(row.id);
        std::vector<std::uint64_t> neighbors;
        for (const auto& [id, distance] : stage->neighbors.at(row.id))
            neighbors.push_back(id);
        std::sort(neighbors.begin(), neighbors.end());
        EXPECT_EQ(neighbors,
                  (std::vector<std::uint64_t>(csr.ids.begin() + csr.offsets[index],
                                              csr.ids.begin() + csr.offsets[index + 1])));
    }
}
TEST(SphVerletWindow, ConnectedThreeDimensionalCloudChangesNeighborsAcrossSteps) {
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
    StageSamples stages(participant);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_NO_FATAL_FAILURE(
        check_trajectory(committed(participant), expected, parameters.eos, 0,
                         nullptr, &stages.samples.back()));

    bool neighbors_changed = false;
    constexpr double dt = .04;
    for (std::size_t step = 1; step <= 8; ++step) {
        expected = explicit_verlet(expected, parameters.eos, dt);
        const auto csr = explicit_csr(expected.particles);
        neighbors_changed |= csr.offsets != initial_csr.offsets || csr.ids != initial_csr.ids;
        ASSERT_TRUE(simulation.advance(dt).ok());
        ASSERT_NO_FATAL_FAILURE(
            check_trajectory(committed(participant), expected, parameters.eos, step,
                             nullptr, &stages.samples.back()));
    }
    EXPECT_TRUE(neighbors_changed);
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphVerletWindow, UniformVelocityCrossesCellsAndStrictCfl) {
    const sph::WcsphVerletParameters parameters{{1, .5, 3}, 1};
    Trajectory expected{{{{8, 8, 8}, 1, 99, 0}, {{-.25, .5, .75}, 2, 11, 0}},
                        {{99, {4, 1, -2}}, {11, {4, 1, -2}}}};
    const Fields fields(expected.particles);
    const auto velocity = input_velocity(expected);
    sph::WcsphParticipant3D participant(fields.get(), 10000, parameters, velocity,
                                        seed_density(expected.particles));
    StageSamples stages(participant);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const double bound = participant.time_constraint(0).maximum_step;
    EXPECT_DOUBLE_EQ(bound, 1.0 / (.5 + std::sqrt(21.0)));
    EXPECT_FALSE(
        simulation.begin_window(std::nextafter(bound, std::numeric_limits<double>::infinity()))
            .ok());
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    ASSERT_TRUE(simulation.begin_window(bound).ok());
    ASSERT_TRUE(simulation.restore_window().ok());
    for (std::size_t step = 1; step <= 6; ++step) {
        if (step == 5) {
            const auto before = committed(participant);
            ASSERT_TRUE(simulation.begin_window(.125).ok());
            ASSERT_TRUE(simulation.advance_trial().ok());
            ASSERT_TRUE(simulation.restore_window().ok());
            const auto after = committed(participant);
            EXPECT_EQ(after.stamp, before.stamp);
            expect_same_rows(after, before);
            EXPECT_EQ(simulation.time().accepted_steps, 4U);
            EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .5);
        }
        expected = explicit_verlet(expected, parameters.eos, .125);
        ASSERT_TRUE(simulation.advance(.125).ok());
        ASSERT_EQ(stages.samples.back().stage, StepStage3D::end);
        const auto current = committed(participant);
        ASSERT_NO_FATAL_FAILURE(check_trajectory(current, expected, parameters.eos, step, nullptr,
                                                 &stages.samples.back()));
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
TEST(SphVerletWindow, UnequalMassTrajectoryMatchesIndependentCsrAndMomentum) {
    const sph::WcsphVerletParameters parameters{{real{.125}, real{.5}, real{3}}, .5};
    auto expected = pair_state();
    Fields fields(expected.particles);
    auto velocity = input_velocity(expected);
    sph::WcsphParticipant3D participant(fields.get(), 1000, parameters, velocity,
                                   seed_density(expected.particles));
    StageSamples stages(participant);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto initial = committed(participant);
    TrajectoryErrors maximum;
    const double initial_cfl = participant.time_constraint(0).maximum_step;
    for (std::size_t step = 1; step <= 12; ++step) {
        const auto next = explicit_verlet(expected, parameters.eos, .02);
        ASSERT_TRUE(simulation.advance(.02).ok());
        const auto snapshot = committed(participant);
        ASSERT_NO_FATAL_FAILURE(check_trajectory(snapshot, next, parameters.eos, step, &maximum,
                                                 &stages.samples.back()));
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
        EXPECT_NEAR(participant.time_constraint(simulation.time().accepted_time).maximum_step,
                    expected_cfl, 16 * std::numeric_limits<double>::epsilon() * expected_cfl);
        expected = next;
    }
    const auto latest = committed(participant);
    EXPECT_GT(latest.stamp.layout.value(), initial.stamp.layout.value());
    EXPECT_LT(participant.time_constraint(simulation.time().accepted_time).maximum_step,
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
TEST(SphVerletWindow, GridCapacityFailureRestoresAllFieldsAndAllowsShorterStep) {
    const sph::WcsphVerletParameters parameters{{1, .25, 3}, .5};
    const Trajectory initial{{{{.125, 0, 0}, 1, 11, 0}, {{3.875, 0, 0}, 1, 22, 0}},
                             {{11, {-.5, 0, 0}}, {22, {.5, 0, 0}}}};
    Fields fields(initial.particles);
    sph::WcsphParticipant3D participant(fields.get(), 2, parameters, input_velocity(initial),
                                   seed_density(initial.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto before = committed(participant);
    const auto bytes = participant.resource_usage().controlled_bytes;
    const auto limit = participant.time_constraint(0).maximum_step;
    ASSERT_GE(limit, .5);
    // Both particles cross the outer grid boundary. Their new box spans four cells.
    const auto failed = simulation.advance(.5);
    EXPECT_EQ(failed.code(), StatusCode::resource_exhausted);
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    EXPECT_EQ(simulation.time().accepted_steps, 0u);
    EXPECT_EQ(simulation.time().accepted_time, 0);
    EXPECT_EQ(participant.time_constraint(0).maximum_step, limit);
    EXPECT_EQ(participant.resource_usage().controlled_bytes, bytes);
    EXPECT_EQ(committed(participant).stamp, before.stamp);
    expect_same_rows(committed(participant), before);
    ASSERT_NO_FATAL_FAILURE(
        check_trajectory(committed(participant), initial, parameters.eos, 0));
    ASSERT_TRUE(simulation.advance(.125).ok());
    ASSERT_NO_FATAL_FAILURE(check_trajectory(committed(participant),
                                             explicit_verlet(initial, parameters.eos, .125),
                                             parameters.eos, 1));
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphVerletWindow, DriftOverflowRestoresAndSmallerStepSucceeds) {
    Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const double gravity = std::numeric_limits<real>::max() / 4;
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, 1, {gravity, 0, 0}, 8};
    sph::Velocity3D velocity("test.velocity", 1);
    sph::Values density("test.density", 1);
    density(0) = 1;
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto before = committed(participant);
    ASSERT_EQ(simulation.advance(4).code(), StatusCode::numerical_failure);
    EXPECT_EQ(committed(participant).stamp, before.stamp);
    expect_same_rows(committed(participant), before);
    EXPECT_EQ(simulation.time().accepted_time, 0);
    const double dt = std::ldexp(1.0, -std::numeric_limits<real>::max_exponent / 2);
    ASSERT_TRUE(simulation.advance(dt).ok());
    const auto after = committed(participant);
    EXPECT_EQ(after.rows.at(0).position[0], static_cast<real>(.5 * dt * gravity * dt));
    EXPECT_EQ(after.rows.at(0).velocity[0], static_cast<real>(gravity * dt));
    ASSERT_TRUE(simulation.finalize().ok());
}

#if !NEXORA_ENABLE_DOUBLE_PRECISION
TEST(SphVerletWindow, DriftJustAboveFloatRangeCannotRoundBackToLargestPosition) {
    const real largest = std::numeric_limits<real>::max();
    Fields fields({{{largest, 0, 0}, 1, 11, 0}});
    sph::Velocity3D velocity("test.velocity", 1);
    sph::Values density("test.density", 1);
    density(0) = 1;
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, 1, {0x1p100, 0, 0}, 0x1p70};
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    // The double drift is FLT_MAX + 2^99, below the midpoint for rounding to infinity.
    EXPECT_EQ(simulation.advance(1).code(), StatusCode::numerical_failure);
    EXPECT_EQ(simulation.time().accepted_steps, 0u);
    EXPECT_EQ(committed(participant).rows.at(0).position[0], largest);
    ASSERT_TRUE(simulation.finalize().ok());
}
#endif


TEST(SphVerletWindow, NoninitialRestoreAndSmallerRetryPreserveCommittedRows) {
    const sph::WcsphVerletParameters parameters{{.125, .5, 3}, .5};
    const auto initial = pair_state();
    const Fields fields(initial.particles);
    auto velocity = input_velocity(initial);
    auto density = seed_density(initial.particles);
    sph::WcsphParticipant3D participant(fields.get(), 1000, parameters, velocity, density);
    sph::WcsphParticipant3D direct(fields.get(), 1000, parameters, velocity, density);
    StageSamples stages(participant);
    Simulation simulation(participant), direct_sim(direct);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(direct_sim.initialize().ok());
    ASSERT_TRUE(direct_sim.prepare().ok());
    const auto accepted_reference = explicit_verlet(initial, parameters.eos, .02);
    ASSERT_TRUE(simulation.advance(.02).ok());
    ASSERT_TRUE(direct_sim.advance(.02).ok());
    const auto accepted = committed(participant);
    const auto accepted_cfl = participant.time_constraint(.02).maximum_step;
    ASSERT_TRUE(simulation.begin_window(.04).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_TRUE(simulation.restore_window().ok());
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .02);
    EXPECT_EQ(simulation.time().accepted_steps, 1u);
    EXPECT_DOUBLE_EQ(participant.time_constraint(.02).maximum_step, accepted_cfl);
    const auto restored = committed(participant);
    EXPECT_EQ(restored.stamp, accepted.stamp);
    expect_same_rows(restored, accepted);
    const auto retry_reference = explicit_verlet(accepted_reference, parameters.eos, .01);
    ASSERT_TRUE(simulation.advance(.01).ok());
    ASSERT_TRUE(direct_sim.advance(.01).ok());
    const auto retried = committed(participant);
    ASSERT_NO_FATAL_FAILURE(check_trajectory(retried, retry_reference, parameters.eos, 2,
                                             nullptr, &stages.samples.back()));
    expect_same_rows(retried, committed(direct));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_TRUE(direct_sim.finalize().ok());
}

TEST(SphVerletWindow, ChecksInitialVelocityCflAndOwnershipBeforePublishing) {
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
        EXPECT_EQ(participant.visit_committed_rows([](sph::CommittedRow3D) {}).code(),
                  StatusCode::invalid_state);
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
        EXPECT_TRUE(simulation.finalize().ok());
    };
    for (double cfl : {0., 1.001, std::numeric_limits<double>::quiet_NaN()}) {
        SCOPED_TRACE(cfl);
        rejects({parameters.eos, cfl}, velocity, density, StatusCode::invalid_argument);
    }
    rejects(parameters, {}, density, StatusCode::invalid_argument);
    rejects(parameters, velocity, {}, StatusCode::invalid_argument);
    velocity(0, 2) = std::numeric_limits<real>::quiet_NaN();
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

TEST(SphVerletWindow, MismatchedTrialIntervalsKeepWindowOpen) {
    const sph::WcsphVerletParameters parameters{{1, 1, 3}, .5};
    const Fields fields({{{0, 0, 0}, 1, 1, 0}});
    sph::Velocity3D velocity("test.velocity", 1);
    velocity(0, 0) = .125;
    sph::Values density("test.density", 1);
    density(0) = 1;
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density);
    ASSERT_TRUE(participant.initialize(1).ok());
    ASSERT_TRUE(participant.prepare().ok());
    const TimeInterval interval{1, 1.125, .125};
    const auto before = committed(participant);
    ASSERT_TRUE(participant.begin_window(interval).ok());
    for (const auto mismatched : {TimeInterval{1.001, 1.125, .125},
                                  TimeInterval{1, 1.126, .125},
                                  TimeInterval{1, 1.125, .124}})
        EXPECT_EQ(participant.advance_trial(mismatched).code(), StatusCode::invalid_state);
    ASSERT_TRUE(participant.advance_trial(interval).ok());
    ASSERT_TRUE(participant.prepare_commit(interval).ok());
    ASSERT_TRUE(participant.commit_window(interval).ok());
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
    EXPECT_DOUBLE_EQ(participant.time_constraint(time).maximum_step, maximum_step);
}

TEST(SphVerletWindow, CommittedMetadataWithNoWallOrRows) {
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
        const auto maximum_step = has_fluid ? .5 / 1.25 : .5;
        sph::WcsphParticipant3D participant(fields.get(), has_fluid ? 1 : 0,
                                            parameters, velocity, density);
        Simulation simulation(participant, 2);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        expect_metadata(participant, simulation, 2, 0, maximum_step);
        const auto start = committed(participant);
        EXPECT_EQ(start.rows.size(), has_fluid ? 1u : 0u);
        ASSERT_TRUE(simulation.advance(.125).ok());
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

TEST(SphVerletWindow, FixedWallOnlyCommitsStaticGeometryAndAcousticLimit) {
    constexpr sph::WcsphVerletParameters parameters{{1, 1, 3}, .5};
    const Fields empty({}), fixed({{{.25, .5, .75}, 1, 19, 1}});
    const sph::Velocity3D velocity("test.empty_velocity", 0);
    const sph::Values density("test.empty_density", 0);
    sph::WcsphParticipant3D participant(
        empty.get(), 1, parameters, velocity, density,
        sph::FixedWallParticles3D{fixed.position, fixed.mass, fixed.ids});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
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
            ASSERT_TRUE(simulation.advance(.5).ok());
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

TEST(SphVerletWindow, InitializationMapsExplicitValuesAcrossInputPermutations) {
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
    StageSamples stages(participant);
    Simulation simulation(participant), direct_sim(direct);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(direct_sim.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(direct_sim.prepare().ok());
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
    ASSERT_EQ(stages.samples.size(), 1u);
    ASSERT_NO_FATAL_FAILURE(check_trajectory(original, initial, parameters.eos, 0,
                                             nullptr, &stages.samples.front()));
    const auto bound = independent_acoustic_step(initial.velocity, parameters);
    EXPECT_NEAR(participant.time_constraint(0).maximum_step, bound,
                8 * std::numeric_limits<real>::epsilon() * bound);
    EXPECT_DOUBLE_EQ(participant.time_constraint(0).maximum_step,
                     direct.time_constraint(0).maximum_step);

    for (std::size_t i = 0; i < initial.particles.size(); ++i) {
        first.position(i, 0) += 10;
        first_density(i) = 9;
        for (int axis = 0; axis < 3; ++axis)
            first_velocity(i, axis) = 10;
    }
    expect_same_rows(committed(participant), original);
    ASSERT_TRUE(simulation.advance(.125).ok());
    ASSERT_TRUE(direct_sim.advance(.125).ok());
    const auto endpoint = explicit_verlet(initial, parameters.eos, .125);
    const auto current = committed(participant);
    ASSERT_NO_FATAL_FAILURE(check_trajectory(current, endpoint, parameters.eos, 1,
                                             nullptr, &stages.samples.back()));
    expect_same_rows(current, committed(direct));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_TRUE(direct_sim.finalize().ok());
}

class WindowAllocations {
  public:
    struct Event {
        bool allocated;
        std::string label;
        const void* address;
        std::uint64_t bytes;
        std::size_t controlled;
    };

    explicit WindowAllocations(const sph::WcsphParticipant3D& participant)
        : participant_(participant), previous_(Kokkos::Tools::Experimental::get_callbacks()) {
        active_ = this;
        Kokkos::Tools::Experimental::set_allocate_data_callback(record<true>);
        Kokkos::Tools::Experimental::set_deallocate_data_callback(record<false>);
    }
    ~WindowAllocations() {
        Kokkos::Tools::Experimental::set_callbacks(previous_);
        active_ = nullptr;
        EXPECT_FALSE(failed_);
    }
    WindowAllocations(const WindowAllocations&) = delete;
    WindowAllocations& operator=(const WindowAllocations&) = delete;

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

    inline static WindowAllocations* active_{};
    const sph::WcsphParticipant3D& participant_;
    Kokkos::Tools::Experimental::EventSet previous_;
    bool failed_{};
};

constexpr std::array workspace_labels{
    "sph.verlet.half_velocity", "sph.verlet.stage", "sph.continuity.old_row",
    "sph.continuity.accepted_rate", "sph.continuity.predicted_rate", "sph.continuity.density"};

TEST(SphVerletWindow, ReusedWorkspaceIsReleasedWithoutChangingCommittedRows) {
    const sph::WcsphVerletParameters parameters{{1, .5, 3}, .5};
    const Trajectory initial{{{{.25, .5, .75}, 1, 11, 0}}, {{11, {0, 0, 0}}}};
    const Fields fields(initial.particles);
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, input_velocity(initial),
                                        seed_density(initial.particles));
    WindowAllocations observed(participant);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto initialized = observed.events.size();
    ASSERT_TRUE(simulation.advance(.125).ok());
    const auto after_cold = observed.events.size();
    const auto warmed_bytes = participant.resource_usage().controlled_bytes;
    for (const auto label : workspace_labels)
        EXPECT_EQ(observed.count(initialized, label, true), 1u) << label;
    for (int step = 0; step < 3; ++step) {
        ASSERT_TRUE(simulation.advance(.125).ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, warmed_bytes);
    }
    for (const auto label : workspace_labels)
        EXPECT_EQ(observed.count(after_cold, label, true), 0u) << label;

    const auto before = committed(participant);
    const auto release_start = observed.events.size();
    ASSERT_TRUE(participant.release_workspace().ok());
    for (const auto label : workspace_labels)
        EXPECT_EQ(observed.count(release_start, label, false), 1u) << label;
    EXPECT_EQ(observed.count(release_start, "sph.grid.cell_begin", false), 1u);
    EXPECT_LT(participant.resource_usage().controlled_bytes, warmed_bytes);
    EXPECT_GT(participant.resource_usage().controlled_bytes, sizeof(LocalIndex));
    const auto read_start = observed.events.size();
    const auto after = committed(participant);
    EXPECT_EQ(after.stamp, before.stamp);
    expect_same_rows(after, before);
    EXPECT_EQ(observed.events.size(), read_start);
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), (std::vector<LocalIndex>{0}));
    ASSERT_NO_FATAL_FAILURE(check_trajectory(after, initial, parameters.eos, 4));
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0u);
    observed.expect_all_released();
}

TEST(SphVerletWindow, WorkspaceBudgetPressurePreservesHistoricalStepCapacity) {
    const sph::WcsphVerletParameters parameters{{.375, .5, 3}, .5};
    const Trajectory initial{{{{.25, .5, .75}, 1, 11, 0}, {{1.25, .5, .75}, .75, 22, 0}},
                             {{11, {.125, 0, 0}}, {22, {-.125, 0, 0}}}};
    const Fields fields(initial.particles);
    const auto velocity = input_velocity(initial);
    const auto density = seed_density(initial.particles);
    constexpr std::size_t r = sizeof(real), i = sizeof(LocalIndex), d = sizeof(double);
    constexpr std::size_t historical = 2 * (12 * r + 9 + 3 * i) + 2 * i;
    constexpr std::size_t insufficient = historical + 6 * r;
    constexpr std::size_t limit = historical + 2 * (16 * r + d);
    constexpr double dt = .125;
    {
        sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density,
                                            ParticipantId{1}, {32, insufficient});
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        const auto before = committed(participant);
        const auto bytes = participant.resource_usage().controlled_bytes;
        EXPECT_EQ(bytes, historical + 2 * i);
        ::testing::Test::RecordProperty("controlled_initial", bytes);
        const auto maximum_step = participant.time_constraint(0).maximum_step;
        for (int attempt = 0; attempt < 2; ++attempt) {
            EXPECT_EQ(simulation.advance(dt).code(), StatusCode::resource_exhausted);
            EXPECT_EQ(simulation.state(), SimulationState::prepared);
            expect_metadata(participant, simulation, 0, 0, maximum_step);
            EXPECT_EQ(participant.resource_usage().controlled_bytes, bytes);
            const auto after = committed(participant);
            EXPECT_EQ(after.stamp, before.stamp);
            expect_same_rows(after, before);
        }
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    }

    {
        sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density,
                                            ParticipantId{1}, {32, limit});
        WindowAllocations observed(participant);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        const auto after_initialize = observed.events.size();
        const auto before = committed(participant);
        ASSERT_TRUE(simulation.begin_window(dt).ok());
        ASSERT_TRUE(simulation.advance_trial().ok());
        ASSERT_TRUE(simulation.restore_window().ok());
        const auto after_restore = observed.events.size();
        ::testing::Test::RecordProperty("controlled_peak_initialization",
                                        observed.peak(0, after_initialize));
        ::testing::Test::RecordProperty("controlled_peak_first_trial",
                                        observed.peak(after_initialize, after_restore));
        const auto restored = committed(participant);
        EXPECT_EQ(restored.stamp, before.stamp);
        expect_same_rows(restored, before);
        EXPECT_EQ(simulation.time().accepted_steps, 0u);
        ASSERT_LT(observed.peak(0, after_initialize), limit);
        EXPECT_LE(observed.peak(after_initialize, after_restore), limit);

        auto expected = initial;
        std::size_t after_first_commit{};
        for (int step = 1; step <= 3; ++step) {
            expected = explicit_verlet(expected, parameters.eos, dt);
            ASSERT_TRUE(simulation.advance(dt).ok());
            EXPECT_LE(participant.resource_usage().controlled_bytes, limit);
            ASSERT_NO_FATAL_FAILURE(
                check_trajectory(committed(participant), expected, parameters.eos, step));
            if (step == 1)
                after_first_commit = observed.events.size();
        }
        EXPECT_LE(observed.peak(after_first_commit, observed.events.size()), limit);
        EXPECT_GE(observed.count(after_initialize, "sph.continuity.density", false), 1u);
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
            status = simulation.prepare();
        if (status.ok())
            status = simulation.begin_window(dt);
        if (status.ok())
            status = simulation.advance_trial();
        if (status.ok())
            status = simulation.restore_window();
        for (int step = 0; step < 3 && status.ok(); ++step)
            status = simulation.advance(dt);
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
    ::testing::Test::RecordProperty("historical_budget_limit", limit);
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
    Velocities half, predicted;
    std::map<std::uint64_t, long double> start_rate, end_rate;
};

FixedWallStep fixed_wall_step(const Trajectory& start, const Particle& wall, double gravity,
                              double dt) {
    const auto first_force = fixed_wall_force(start, wall, gravity);
    FixedWallStep result{start, {}, {}, {}, {}};
    auto start_sources = start.particles;
    auto start_velocity = start.velocity;
    start_sources.push_back(wall);
    start_velocity.emplace(wall.id, std::array<real, 3>{});
    result.predicted.emplace(wall.id, std::array<real, 3>{});
    for (auto& fluid : result.end.particles) {
        const auto id = fluid.id;
        const auto prior = start.velocity.at(id)[0];
        const double acceleration = first_force.acceleration.at(id) + gravity;
        result.half[id][0] = static_cast<real>(std::fma(.5 * dt, acceleration, double(prior)));
        result.predicted[id][0] = static_cast<real>(std::fma(dt, acceleration, double(prior)));
        fluid.position[0] = static_cast<real>(
            std::fma(dt, double(result.half.at(id)[0]), double(fluid.position[0])));
    }
    auto end_sources = result.end.particles;
    end_sources.push_back(wall);
    for (std::size_t i = 0; i < result.end.particles.size(); ++i) {
        const auto id = start.particles[i].id;
        const auto r0 = density_rate(start.particles[i], start_sources, start_velocity);
        const auto r1 = density_rate(result.end.particles[i], end_sources, result.predicted);
        result.start_rate.emplace(id, r0);
        result.end_rate.emplace(id, r1);
        result.end.density[id] = static_cast<real>(
            std::fma(.5 * dt, static_cast<double>(r0 + r1), double(start.density.at(id))));
    }
    const auto last_force = fixed_wall_force(result.end, wall, gravity);
    for (const auto& fluid : result.end.particles) {
        const double acceleration = last_force.acceleration.at(fluid.id) + gravity;
        result.end.velocity[fluid.id][0] = static_cast<real>(
            std::fma(.5 * dt, acceleration, double(result.half.at(fluid.id)[0])));
    }
    return result;
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

void expect_stage_query(const StageSample& sampled, StepStage3D stage, const Trajectory& state,
                        const Particle& wall) {
    constexpr double eps = std::numeric_limits<real>::epsilon();
    const auto expected = ordered_with_wall(state, wall);
    EXPECT_EQ(sampled.stage, stage);
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

void expect_stage_kinematics(const StageSample& sample, const Velocities& velocity,
                             const Velocities& half = {}) {
    EXPECT_EQ(sample.velocity.size(), velocity.size());
    EXPECT_EQ(sample.half.size(), half.size());
    constexpr double eps = std::numeric_limits<real>::epsilon();
    for (const auto& [id, v] : velocity)
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(sample.velocity.at(id)[k], v[k], 16 * eps);
    for (const auto& [id, v] : half)
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(sample.half.at(id)[k], v[k], 16 * eps);
}

TEST(SphVerletWindow, ObservesStageNeighborsAfterEvictionAndRestore) {
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
    sph::WcsphParticipant3D direct(fields.get(), 64, parameters, velocity, density, boundary);
    sph::WcsphParticipant3D participant(fields.get(), 64, parameters, velocity, density, boundary);
    StageSamples stages(participant, true), reference_stages(direct);
    Simulation simulation(participant), direct_sim(direct);
    ASSERT_TRUE(direct_sim.initialize().ok());
    ASSERT_TRUE(direct_sim.prepare().ok());
    WindowAllocations allocations(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_EQ(stages.samples.size(), 1u);
    expect_stage_query(stages.samples[0], StepStage3D::initial, initial, wall);
    const auto start = committed(participant);
    ASSERT_NO_FATAL_FAILURE(expect_fixed_rows(start, initial, wall, gravity));

    const auto first = fixed_wall_step(initial, wall, gravity, dt);
    ASSERT_TRUE(simulation.advance(dt).ok());
    ASSERT_EQ(stages.samples.size(), 3u);
    expect_stage_query(stages.samples[1], StepStage3D::start, initial, wall);
    expect_stage_query(stages.samples[2], StepStage3D::end, first.end, wall);
    auto old_velocity = initial.velocity;
    old_velocity.emplace(wall.id, std::array<real, 3>{});
    expect_stage_kinematics(stages.samples[1], old_velocity, first.half);
    expect_stage_kinematics(stages.samples[2], first.predicted);
    const auto rates = WcsphTestAccess::rates(participant);
    ASSERT_EQ(rates[0].size(), 2u);
    ASSERT_EQ(rates[1].size(), 2u);
    constexpr double eps = std::numeric_limits<real>::epsilon();
    for (std::size_t row = 0; row < 2; ++row) {
        EXPECT_NEAR(rates[0][row], first.start_rate.at(stages.samples[1].owned[row]), 16 * eps);
        EXPECT_NEAR(rates[1][row], first.end_rate.at(stages.samples[2].owned[row]), 16 * eps);
    }
    EXPECT_EQ(stages.samples[1].owned, (std::vector<std::uint64_t>{42, 77}));
    EXPECT_EQ(stages.samples[2].owned, (std::vector<std::uint64_t>{77, 42}));
    const auto crossed = committed(participant);
    EXPECT_EQ(crossed.stamp.topology, start.stamp.topology);
    EXPECT_EQ(crossed.stamp.geometry.value(), start.stamp.geometry.value() + 1);
    EXPECT_EQ(crossed.stamp.layout.value(), start.stamp.layout.value() + 1);
    ASSERT_NO_FATAL_FAILURE(expect_fixed_rows(crossed, first.end, wall, gravity));
    const auto acoustic = independent_acoustic_step(first.end.velocity, parameters);
    EXPECT_NEAR(participant.time_constraint(dt).maximum_step, acoustic, 16 * eps * acoustic);

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
    expect_same_rows(readback, crossed);
    EXPECT_EQ(readback.stamp, crossed.stamp);
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), row_table);
    EXPECT_EQ(allocations.events.size(), after_release);
    EXPECT_EQ(stages.samples.size(), 3u);
    ASSERT_NO_FATAL_FAILURE(expect_fixed_rows(readback, first.end, wall, gravity));

    constexpr double replay_dt = .125;
    const auto next = fixed_wall_step(first.end, wall, gravity, replay_dt);
    ASSERT_TRUE(simulation.begin_window(replay_dt).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    EXPECT_GE(allocations.count(after_release, "sph.grid.keys", true), 1u);
    ASSERT_EQ(stages.samples.size(), 5u);
    expect_stage_query(stages.samples[3], StepStage3D::start, first.end, wall);
    expect_stage_query(stages.samples[4], StepStage3D::end, next.end, wall);
    ASSERT_TRUE(simulation.restore_window().ok());
    EXPECT_EQ(simulation.time().accepted_time, dt);
    EXPECT_EQ(simulation.time().accepted_steps, 1u);
    EXPECT_DOUBLE_EQ(participant.time_constraint(dt).maximum_step, WcsphTestAccess::maximum_step(participant));
    const auto restored = committed(participant);
    EXPECT_EQ(crossed.stamp, restored.stamp);
    expect_same_rows(restored, crossed);
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), row_table);

    ASSERT_TRUE(direct_sim.advance(dt).ok());
    ASSERT_TRUE(direct_sim.advance(replay_dt).ok());
    ASSERT_TRUE(simulation.advance(replay_dt).ok());
    ASSERT_EQ(stages.samples.size(), 7u);
    ASSERT_EQ(reference_stages.samples.size(), 5u);
    for (std::size_t i = 0; i < 2; ++i) {
        const auto& replayed = stages.samples[5 + i];
        expect_stage_query(replayed, i == 0 ? StepStage3D::start : StepStage3D::end,
                           i == 0 ? first.end : next.end, wall);
        EXPECT_EQ(replayed.neighbors, stages.samples[3 + i].neighbors);
        EXPECT_EQ(replayed.neighbors, reference_stages.samples[3 + i].neighbors);
    }
    const auto published = committed(participant);
    expect_same_rows(published, committed(direct));
    ASSERT_NO_FATAL_FAILURE(expect_fixed_rows(published, next.end, wall, gravity));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_TRUE(direct_sim.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

} // namespace
