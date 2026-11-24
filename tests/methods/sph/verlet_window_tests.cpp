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
#include <numbers>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

namespace {
struct HostAllocationObservation {
    std::size_t calls{}, requested_bytes{};
    std::size_t fail_at{};
    bool active{};
    bool threw{};
};
thread_local HostAllocationObservation host_allocations;

class ScopedHostAllocations {
  public:
    explicit ScopedHostAllocations(std::size_t fail_at) {
        host_allocations = {.fail_at = fail_at, .active = true};
    }
    ~ScopedHostAllocations() { host_allocations.active = false; }
};

void count_host_allocation(std::size_t bytes) {
    auto& observation = host_allocations;
    if (!observation.active)
        return;
    observation.requested_bytes += bytes;
    if (++observation.calls == observation.fail_at) {
        observation.active = false;
        observation.threw = true;
        throw std::bad_alloc{};
    }
}

void* allocate_test_host_storage(std::size_t bytes) {
    count_host_allocation(bytes);
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

void* operator new(std::size_t bytes) {
    return allocate_test_host_storage(bytes);
}

void* operator new[](std::size_t bytes) {
    return allocate_test_host_storage(bytes);
}

void operator delete(void* allocation) noexcept {
    std::free(allocation);
}

void operator delete[](void* allocation) noexcept {
    std::free(allocation);
}

void operator delete(void* allocation, std::size_t) noexcept {
    std::free(allocation);
}

void operator delete[](void* allocation, std::size_t) noexcept {
    std::free(allocation);
}

namespace nexora::sph {

struct WcsphTestAccess {
    struct PrecommitIdentity {
        FieldStamp state_stamp;
        double maximum_step;
        std::array<std::uintptr_t, 5> allocation;
        std::array<FieldStamp, 4> grant;
        bool operator==(const PrecommitIdentity&) const = default;
    };

    static PrecommitIdentity precommit_identity(const WcsphState3D& state) {
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

    static PrecommitIdentity accepted_precommit_identity(const WcsphParticipant3D& participant) {
        return precommit_identity(*participant.accepted_);
    }

    static PrecommitIdentity trial_precommit_identity(const WcsphParticipant3D& participant) {
        return precommit_identity(*participant.trial_);
    }

    static bool has_unpublished_precommit_state(const WcsphParticipant3D& participant) {
        return participant.trial_ || participant.pending_;
    }

    static auto reserve_test_budget(const WcsphParticipant3D& participant, std::size_t bytes) {
        return participant.budget_.reserve(bytes);
    }

    static PrecommitIdentity pending_precommit_identity(const WcsphParticipant3D& participant) {
        return precommit_identity(*participant.pending_);
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
    struct WallGroupValue {
        double initial_phase, elapsed, maximum_step;
        friend bool operator==(const WallGroupValue&, const WallGroupValue&) = default;
    };
    static std::map<std::uint64_t, WallGroupValue>
    wall_groups(const WcsphParticipant3D& participant, bool trial = false) {
        return wall_groups(*(trial ? participant.trial_ : participant.accepted_));
    }
    static std::map<std::uint64_t, WallGroupValue>
    pending_wall_groups(const WcsphParticipant3D& participant) {
        return wall_groups(*participant.pending_);
    }
    static std::map<std::uint64_t, WallGroupValue> wall_groups(const WcsphState3D& state) {
        const auto& group_states = state.wall_phases;
        std::map<std::uint64_t, WallGroupValue> result;
        for (std::size_t i = 0; i < group_states.extent(0); ++i) {
            const auto& group = group_states(i);
            result.emplace(group.group_id, WallGroupValue{group.time.initial_phase, group.time.elapsed,
                                                         group.maximum_step});
        }
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

// Forward the real lifecycle. Scope the host fault to prepare_commit so
// neither Simulation recovery nor the subsequent publication can be faulted.
class ObservePrecommit final : public ParticipantIF {
  public:
    explicit ObservePrecommit(sph::WcsphParticipant3D& participant) : participant_(participant) {}
    ParticipantId id() const noexcept override { return participant_.id(); }
    ParticipantCapabilities capabilities() const override { return participant_.capabilities(); }
    std::vector<PortDescriptor> ports() const override { return participant_.ports(); }
    TimeConstraint time_constraint(double time) const override {
        return participant_.time_constraint(time);
    }
    Status initialize(double time) override { return participant_.initialize(time); }
    Status prepare() override { return participant_.prepare(); }
    Status begin_window(const TimeInterval& interval) override {
        return participant_.begin_window(interval);
    }
    Status advance_trial(const TimeInterval& interval) override {
        return participant_.advance_trial(interval);
    }
    Status prepare_commit(const TimeInterval& interval) override {
        ++prepare_calls;
        const auto failure = std::exchange(fail_at, std::nullopt);
        if (!failure)
            return participant_.prepare_commit(interval);
        const ScopedHostAllocations observe(*failure);
        return participant_.prepare_commit(interval);
    }
    Status commit_window(const TimeInterval& interval) override {
        return participant_.commit_window(interval);
    }
    Status restore_window() override { return participant_.restore_window(); }
    Status finalize() override { return participant_.finalize(); }

    std::size_t prepare_calls{};
    std::optional<std::size_t> fail_at;

  private:
    sph::WcsphParticipant3D& participant_;
};

class PrecommitKokkosEvents {
  public:
    PrecommitKokkosEvents() : previous_(Kokkos::Tools::Experimental::get_callbacks()) {
        active_ = this;
        using namespace Kokkos::Tools::Experimental;
        set_allocate_data_callback(record_allocation);
        set_deallocate_data_callback(record_release);
        set_begin_deep_copy_callback(record_copy);
    }

    ~PrecommitKokkosEvents() {
        Kokkos::Tools::Experimental::set_callbacks(previous_);
        active_ = nullptr;
    }

    std::size_t allocations{}, copies{};
    std::array<std::size_t, 5> releases{};
    std::array<std::uintptr_t, 5> candidate_allocations{};

  private:
    static void record_allocation(Kokkos::Tools::SpaceHandle, const char* label,
                                  const void* address, std::uint64_t) noexcept {
        constexpr std::array<std::string_view, 5> candidate_labels{
            "sph.participant3d.density", "sph.participant3d.pressure",
            "sph.participant3d.pressure_acceleration", "sph.participant3d.velocity",
            "sph.geometry.owned_row_to_input"};
        ++active_->allocations;
        for (std::size_t i = 0; i < candidate_labels.size(); ++i)
            if (candidate_labels[i] == label)
                active_->candidate_allocations[i] = reinterpret_cast<std::uintptr_t>(address);
    }

    static void record_copy(Kokkos::Tools::SpaceHandle, const char*, const void*,
                            Kokkos::Tools::SpaceHandle, const char*, const void*,
                            std::uint64_t) noexcept {
        ++active_->copies;
    }

    static void record_release(Kokkos::Tools::SpaceHandle, const char*, const void* address,
                               std::uint64_t) noexcept {
        const auto pointer = reinterpret_cast<std::uintptr_t>(address);
        for (std::size_t i = 0; i < active_->candidate_allocations.size(); ++i)
            if (active_->candidate_allocations[i] == pointer)
                ++active_->releases[i];
    }

    inline static PrecommitKokkosEvents* active_{};
    Kokkos::Tools::Experimental::EventSet previous_;
};

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

void expect_same_wall(const sph::WallValue3D& actual, const sph::WallValue3D& expected) {
    EXPECT_EQ(actual.id, expected.id);
    EXPECT_EQ(actual.mass, expected.mass);
    EXPECT_EQ(actual.position, expected.position);
    EXPECT_EQ(actual.stamp, expected.stamp);
    EXPECT_EQ(actual.phase, expected.phase);
}

TEST(SphVerletWindow, HostAllocationFailureRestoresAndRetries) {
    using Access = sph::WcsphTestAccess;
    const M4FixedWallFixture fixture;
    constexpr double first_step = fixture.first_step, step = fixture.step;
    std::size_t precommit_host_calls = 0;
    CommittedSample reference;
    double reference_limit = 0;
    const auto run = [&](std::size_t fail_at) {
        SCOPED_TRACE(fail_at);
        auto participant = fixture.participant();
        ObservePrecommit observed(participant);
        Simulation simulation(observed);
        const auto initialized = simulation.initialize();
        ASSERT_TRUE(initialized.ok()) << initialized.message();
        const auto prepared = simulation.prepare();
        ASSERT_TRUE(prepared.ok()) << prepared.message();
        const auto first = simulation.advance(first_step);
        ASSERT_TRUE(first.ok()) << first.message();
        ASSERT_TRUE(participant.release_workspace().ok());
        const auto before = Access::accepted_precommit_identity(participant);
        const auto before_rows = committed(participant);
        const auto before_wall = checked(participant.read_committed_wall(30));
        const auto cold_usage = participant.resource_usage();
        const auto opened = simulation.begin_window(step);
        ASSERT_TRUE(opened.ok()) << opened.message();
        Access::PrecommitIdentity trial{};
        detail::ResourceBudget::Usage trial_usage;
        Status commit_status;
        {
            PrecommitKokkosEvents numeric;
            const auto advanced = simulation.advance_trial();
            ASSERT_TRUE(advanced.ok()) << advanced.message();
            trial = Access::trial_precommit_identity(participant);
            for (std::size_t i = 0; i < trial.allocation.size(); ++i) {
                ASSERT_NE(trial.allocation[i], 0u);
                ASSERT_NE(numeric.candidate_allocations[i], 0u);
                EXPECT_NE(trial.allocation[i], before.allocation[i]);
            }
            trial_usage = participant.resource_usage();
            EXPECT_GT(trial_usage.controlled_bytes, cold_usage.controlled_bytes);
            EXPECT_EQ(simulation.state(), SimulationState::trial_advanced);
            const auto allocations = numeric.allocations;
            const auto copies = numeric.copies;
            observed.fail_at = fail_at;
            commit_status = simulation.commit_window();
            EXPECT_EQ(numeric.allocations, allocations);
            EXPECT_EQ(numeric.copies, copies);
            const std::array<std::size_t, 5> expected_releases =
                fail_at ? std::array<std::size_t, 5>{1, 1, 1, 1, 1}
                        : std::array<std::size_t, 5>{};
            EXPECT_EQ(numeric.releases, expected_releases);
        }

        EXPECT_FALSE(Access::has_unpublished_precommit_state(participant));
        EXPECT_EQ(simulation.state(), SimulationState::prepared);
        if (fail_at) {
            EXPECT_TRUE(host_allocations.threw);
            EXPECT_EQ(host_allocations.calls, fail_at);
            ASSERT_EQ(commit_status.code(), StatusCode::resource_exhausted) << commit_status.message();
            EXPECT_EQ(Access::accepted_precommit_identity(participant), before);
            expect_same_rows(committed(participant), before_rows);
            expect_same_wall(checked(participant.read_committed_wall(30)), before_wall);
            EXPECT_EQ(simulation.time().accepted_time, first_step);
            EXPECT_EQ(simulation.time().accepted_steps, 1u);
            EXPECT_EQ(simulation.time().trial_attempts, 2u);
            EXPECT_EQ(participant.time_constraint(first_step).maximum_step, before.maximum_step);
            ASSERT_TRUE(participant.release_workspace().ok());
            const auto restored_usage = participant.resource_usage();
            EXPECT_EQ(restored_usage.controlled_bytes, cold_usage.controlled_bytes);
            EXPECT_EQ(restored_usage.snapshot_records, cold_usage.snapshot_records);
            EXPECT_EQ(restored_usage.known_external_owner_bytes, cold_usage.known_external_owner_bytes);

            ASSERT_TRUE(simulation.advance(step).ok());
            expect_same_rows(committed(participant), reference);
            EXPECT_EQ(participant.time_constraint(first_step + step).maximum_step, reference_limit);
        } else {
            ASSERT_TRUE(commit_status.ok()) << commit_status.message();
            EXPECT_FALSE(host_allocations.threw);
            precommit_host_calls = host_allocations.calls;
            RecordProperty("successful_prepare_host_allocation_bytes",
                           std::to_string(host_allocations.requested_bytes));
            RecordProperty("controlled_bytes_before_window",
                           std::to_string(cold_usage.controlled_bytes));
            RecordProperty("controlled_bytes_after_trial",
                           std::to_string(trial_usage.controlled_bytes));
            const auto accepted = Access::accepted_precommit_identity(participant);
            EXPECT_EQ(accepted.allocation, trial.allocation);
            for (std::size_t i = 0; i < accepted.grant.size(); ++i) {
                EXPECT_TRUE(accepted.grant[i].accepted);
                EXPECT_NE(accepted.grant[i].storage_incarnation, trial.grant[i].storage_incarnation);
            }
            reference = committed(participant);
            reference_limit = participant.time_constraint(first_step + step).maximum_step;
        }
        EXPECT_EQ(simulation.time().accepted_time, first_step + step);
        EXPECT_EQ(simulation.time().accepted_steps, 2u);
        EXPECT_EQ(simulation.time().trial_attempts, fail_at ? 3u : 2u);
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    };

    ASSERT_NO_FATAL_FAILURE(run(0));
    ASSERT_GT(precommit_host_calls, 0u);
    RecordProperty("successful_prepare_host_allocations", std::to_string(precommit_host_calls));
    for (std::size_t failure = 1; failure <= precommit_host_calls; ++failure)
        ASSERT_NO_FATAL_FAILURE(run(failure));
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

TEST(SphVerletWindow, CommittedReadsRejectActiveAndReentrantWindowsAndRecoverAfterException) {
    const M4FixedWallFixture fixture;
    auto participant = fixture.participant();
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
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
    const double maximum = participant.time_constraint(0).maximum_step;

    EXPECT_EQ(simulation.begin_window(2 * maximum).code(), StatusCode::invalid_argument);
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    expect_same_rows(committed(participant), before);
    expect_same_wall(checked(participant.read_committed_wall(30)), wall);

    ASSERT_TRUE(simulation.begin_window(fixture.step).ok());
    expect_committed_reads_rejected(participant);
    ASSERT_TRUE(simulation.restore_window().ok());

    std::vector<std::uint64_t> visited;
    const auto normal = participant.visit_committed_rows([&](sph::CommittedRow3D row) {
        visited.push_back(row.id);
        if (visited.size() == 1) {
            EXPECT_EQ(simulation.begin_window(fixture.step).code(), StatusCode::invalid_state);
            expect_committed_reads_rejected(participant);
        }
    });
    EXPECT_TRUE(normal.ok()) << normal.message();
    EXPECT_EQ(visited, (std::vector<std::uint64_t>{10, 20}));
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    ASSERT_TRUE(simulation.begin_window(fixture.step).ok());
    ASSERT_TRUE(simulation.restore_window().ok());
    expect_same_rows(committed(participant), before);
    expect_same_wall(checked(participant.read_committed_wall(30)), wall);

    std::size_t calls = 0;
    const auto thrown = participant.visit_committed_rows([&](sph::CommittedRow3D) {
        ++calls;
        throw std::runtime_error("observer failed");
    });
    EXPECT_EQ(thrown.code(), StatusCode::execution_failure);
    EXPECT_EQ(calls, 1u);
    ASSERT_TRUE(simulation.begin_window(fixture.step).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_TRUE(simulation.restore_window().ok());
    expect_same_rows(committed(participant), before);
    expect_same_wall(checked(participant.read_committed_wall(30)), wall);
    expect_metadata(participant, simulation, 0, 0, maximum);
    EXPECT_EQ(simulation.time().trial_attempts, 1u);
    EXPECT_TRUE(simulation.finalize().ok());
}

TEST(SphVerletWindow, DirectFinalizeRejectedDuringCommittedRowVisit) {
    const M4FixedWallFixture fixture;
    auto participant = fixture.participant();
    ASSERT_TRUE(participant.initialize(0).ok());
    ASSERT_TRUE(participant.prepare().ok());
    std::vector<std::uint64_t> ids;
    const auto read = participant.visit_committed_rows([&](sph::CommittedRow3D row) {
        ids.push_back(row.id);
        if (ids.size() == 1)
            EXPECT_EQ(participant.finalize().code(), StatusCode::invalid_state);
    });
    EXPECT_TRUE(read.ok()) << read.message();
    EXPECT_EQ(ids, (std::vector<std::uint64_t>{10, 20}));
    const TimeInterval interval{0, fixture.first_step, fixture.first_step};
    ASSERT_TRUE(participant.begin_window(interval).ok());
    ASSERT_TRUE(participant.advance_trial(interval).ok());
    ASSERT_TRUE(participant.prepare_commit(interval).ok());
    ASSERT_TRUE(participant.commit_window(interval).ok());
    EXPECT_EQ(committed(participant).stamp.accepted_version.value(), 1u);
    EXPECT_TRUE(participant.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

TEST(SphVerletWindow, SimulationFinalizeFailsDuringCommittedRowVisit) {
    const M4FixedWallFixture fixture;
    auto participant = fixture.participant();
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(simulation.advance(fixture.first_step).ok());
    const auto before = simulation.time();
    std::vector<std::uint64_t> ids;
    const auto read = participant.visit_committed_rows([&](sph::CommittedRow3D row) {
        ids.push_back(row.id);
        if (ids.size() == 1) {
            EXPECT_EQ(simulation.finalize().code(), StatusCode::invalid_state);
            EXPECT_EQ(simulation.state(), SimulationState::failed);
        }
    });
    EXPECT_TRUE(read.ok()) << read.message();
    EXPECT_EQ(ids, (std::vector<std::uint64_t>{10, 20}));
    EXPECT_EQ(simulation.state(), SimulationState::failed);
    EXPECT_EQ(simulation.begin_window(fixture.step).code(), StatusCode::invalid_state);
    EXPECT_EQ(simulation.time().accepted_time, before.accepted_time);
    EXPECT_EQ(simulation.time().accepted_steps, before.accepted_steps);
    EXPECT_EQ(simulation.time().trial_attempts, before.trial_attempts);
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(simulation.state(), SimulationState::finalized);
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

TEST(SphVerletWindow, PreparedCandidateRestoresAndReplaysWithoutChangingCommittedAllocations) {
    using Access = WcsphTestAccess;
    const M4FixedWallFixture fixture;
    constexpr auto first = fixture.first_step, full = fixture.step;
    for (const double dt : {full, full / 2}) {
        SCOPED_TRACE(dt);
        auto participant = fixture.participant();
        auto direct = fixture.participant();
        const auto advance = [](sph::WcsphParticipant3D& item, TimeInterval interval) {
            ASSERT_TRUE(item.begin_window(interval).ok());
            ASSERT_TRUE(item.advance_trial(interval).ok());
            ASSERT_TRUE(item.prepare_commit(interval).ok());
            ASSERT_TRUE(item.commit_window(interval).ok());
        };
        for (auto* item : {&participant, &direct}) {
            ASSERT_TRUE(item->initialize(0).ok());
            ASSERT_TRUE(item->prepare().ok());
            ASSERT_NO_FATAL_FAILURE(advance(*item, {0, first, first}));
        }
        ASSERT_TRUE(participant.release_workspace().ok());
        const auto baseline_usage = participant.resource_usage();
        const auto before = committed(participant);
        const auto before_identity = Access::accepted_precommit_identity(participant);
        const auto before_map = Access::row_to_input(participant);
        const auto before_wall = checked(participant.read_committed_wall(30));

        const TimeInterval trial_interval{first, first + full, full};
        ASSERT_TRUE(participant.begin_window(trial_interval).ok());
        {
            PrecommitKokkosEvents numeric;
            ASSERT_TRUE(participant.advance_trial(trial_interval).ok());
            const auto candidate = Access::trial_precommit_identity(participant);
            const auto allocations = numeric.allocations, copies = numeric.copies;
            ASSERT_TRUE(participant.prepare_commit(trial_interval).ok());
            const auto pending = Access::pending_precommit_identity(participant);
            EXPECT_EQ(pending.allocation, candidate.allocation);
            for (std::size_t i = 0; i < pending.grant.size(); ++i) {
                EXPECT_TRUE(pending.grant[i].accepted);
                EXPECT_NE(pending.grant[i].storage_incarnation,
                          candidate.grant[i].storage_incarnation);
            }
            EXPECT_EQ(numeric.allocations, allocations);
            EXPECT_EQ(numeric.copies, copies);
            expect_committed_reads_rejected(participant);
            ASSERT_TRUE(participant.restore_window().ok());
            EXPECT_EQ(numeric.releases, (std::array<std::size_t, 5>{1, 1, 1, 1, 1}));
        }
        EXPECT_FALSE(Access::has_unpublished_precommit_state(participant));
        EXPECT_EQ(Access::accepted_precommit_identity(participant), before_identity);
        EXPECT_EQ(Access::row_to_input(participant), before_map);
        expect_same_rows(committed(participant), before);
        expect_same_wall(checked(participant.read_committed_wall(30)), before_wall);
        ASSERT_TRUE(participant.release_workspace().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, baseline_usage.controlled_bytes);
        EXPECT_EQ(participant.resource_usage().snapshot_records, baseline_usage.snapshot_records);
        EXPECT_EQ(participant.time_constraint(first).maximum_step, before_identity.maximum_step);

        const TimeInterval retry{first, first + dt, dt};
        ASSERT_NO_FATAL_FAILURE(advance(direct, retry));
        ASSERT_TRUE(participant.begin_window(retry).ok());
        {
            PrecommitKokkosEvents numeric;
            ASSERT_TRUE(participant.advance_trial(retry).ok());
            ASSERT_TRUE(participant.prepare_commit(retry).ok());
            const auto pending = Access::pending_precommit_identity(participant);
            const auto allocations = numeric.allocations, copies = numeric.copies;
            ASSERT_TRUE(participant.commit_window(retry).ok());
            EXPECT_EQ(Access::accepted_precommit_identity(participant), pending);
            EXPECT_EQ(numeric.allocations, allocations);
            EXPECT_EQ(numeric.copies, copies);
            EXPECT_EQ(numeric.releases, (std::array<std::size_t, 5>{}));
        }
        expect_same_rows(committed(participant), committed(direct));
        EXPECT_EQ(Access::stamp(participant), Access::stamp(direct));
        EXPECT_EQ(Access::row_to_input(participant), Access::row_to_input(direct));
        EXPECT_EQ(Access::maximum_step(participant), Access::maximum_step(direct));
        expect_same_wall(checked(participant.read_committed_wall(30)),
                         checked(direct.read_committed_wall(30)));
        EXPECT_TRUE(participant.finalize().ok());
        EXPECT_TRUE(direct.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    }
}

#if NEXORA_ENABLE_DOUBLE_PRECISION
TEST(SphVerletWindow, LateCflFailureRestoresAndShorterRetryMatchesDirectStep) {
    const Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const sph::Velocity3D velocity("late_cfl.velocity", 1);
    const sph::Values density("late_cfl.density", 1);
    density(0) = 1;
    constexpr double failed_dt = 0x1p-505, legal_dt = 0x1p-508;
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, .5, {0x1p1018, 0, 0}, 1};
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, density);
    sph::WcsphParticipant3D direct(fields.get(), 1, parameters, velocity, density);
    ObservePrecommit observed(participant);
    StageSamples samples(participant);
    WindowAllocations allocations(participant);
    Simulation simulation(observed), baseline(direct);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(baseline.initialize().ok());
    ASSERT_TRUE(baseline.prepare().ok());
    ASSERT_TRUE(participant.release_workspace().ok());
    const auto before = committed(participant);
    const auto identity = WcsphTestAccess::accepted_precommit_identity(participant);
    const auto row_map = WcsphTestAccess::row_to_input(participant);
    const auto usage = participant.resource_usage();
    for (std::size_t attempt = 1; attempt <= 2; ++attempt) {
        SCOPED_TRACE(attempt);
        const auto mark = allocations.events.size();
        const auto stage_mark = samples.samples.size();
        ASSERT_TRUE(simulation.begin_window(failed_dt).ok());
        const auto failure = simulation.advance_trial();
        EXPECT_EQ(failure.code(), StatusCode::numerical_failure);
        EXPECT_EQ(failure.message(), "invalid WCSPH CFL step");
        EXPECT_EQ(observed.prepare_calls, 0u);
        EXPECT_FALSE(WcsphTestAccess::has_unpublished_precommit_state(participant));
        EXPECT_EQ(allocations.count(mark, "sph.participant3d.velocity", true), 1u);
        EXPECT_EQ(allocations.count(mark, "sph.participant3d.velocity", false), 1u);
        ASSERT_EQ(samples.samples.size(), stage_mark + 2);
        EXPECT_EQ(samples.samples.back().stage, StepStage3D::end);
        EXPECT_EQ(samples.samples.back().particles.at(0).position[0], 128);
        EXPECT_EQ(simulation.state(), SimulationState::prepared);
        EXPECT_EQ(simulation.time().trial_attempts, attempt);
        expect_metadata(participant, simulation, 0, 0, .5);
        expect_same_rows(committed(participant), before);
        EXPECT_EQ(WcsphTestAccess::accepted_precommit_identity(participant), identity);
        EXPECT_EQ(WcsphTestAccess::row_to_input(participant), row_map);
        ASSERT_TRUE(participant.release_workspace().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, usage.controlled_bytes);
        EXPECT_EQ(participant.resource_usage().snapshot_records, usage.snapshot_records);
    }
    ASSERT_TRUE(simulation.advance(legal_dt).ok());
    ASSERT_TRUE(baseline.advance(legal_dt).ok());
    EXPECT_EQ(observed.prepare_calls, 1u);
    EXPECT_EQ(simulation.time().trial_attempts, 3u);
    expect_metadata(participant, simulation, legal_dt, 1, 0x1p-511);
    const auto after = committed(participant);
    expect_same_rows(after, committed(direct));
    ASSERT_EQ(after.rows.size(), 1u);
    EXPECT_EQ(after.rows[0].position[0], 2);
    EXPECT_EQ(after.rows[0].velocity[0], 0x1p510);
    EXPECT_TRUE(simulation.finalize().ok());
    EXPECT_TRUE(baseline.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}
#endif

TEST(SphVerletWindow, FixedWallBudgetRejectionRestoresCommittedRowsAndWall) {
    const M4FixedWallFixture fixture;
    fixture.fluid.position(0, 0) = 1.125;
    constexpr auto dt = fixture.first_step;
    constexpr std::size_t source_count = 3;
    constexpr std::size_t vector_bytes = source_count * 3 * sizeof(real);
    constexpr std::size_t grid_values_bytes = source_count * 4 * sizeof(real);

    auto reference = fixture.participant();
    std::optional<WindowAllocations> reference_allocations(std::in_place, reference);
    StageSamples reference_stages(reference);
    Simulation direct(reference);
    ASSERT_TRUE(direct.initialize().ok());
    ASSERT_TRUE(direct.prepare().ok());
    const auto after_initialize = reference_allocations->events.size();
    const auto initial_bytes = reference.resource_usage().controlled_bytes;
    const auto initial_peak = reference_allocations->peak(0, after_initialize);
    ASSERT_TRUE(direct.begin_window(dt).ok());
    ASSERT_TRUE(direct.advance_trial().ok());
    const auto trial_peak = reference_allocations->peak(after_initialize,
                                                       reference_allocations->events.size());
    ASSERT_GT(trial_peak, initial_peak);
    ASSERT_TRUE(direct.commit_window().ok());
    const auto end = committed(reference);
    const auto end_wall = checked(reference.read_committed_wall(30));
    const double end_step = reference.time_constraint(dt).maximum_step;
    ASSERT_TRUE(direct.finalize().ok());
    reference_allocations.reset();
    ASSERT_EQ(reference_stages.samples.size(), 3u);

    // The old grid has the same box, input order, and cell membership on this
    // step. Both Verlet buffers fit; the subsequent reuse values need one more byte.
    const auto refusal_limit = initial_bytes + 2 * vector_bytes + grid_values_bytes - 1;
    ASSERT_GE(refusal_limit, initial_peak);
    ASSERT_LT(refusal_limit, trial_peak);
    ::testing::Test::RecordProperty("fixed_wall_initialization_peak_bytes", initial_peak);
    ::testing::Test::RecordProperty("fixed_wall_trial_peak_bytes", trial_peak);
    ::testing::Test::RecordProperty("fixed_wall_refusal_limit_bytes", refusal_limit);
    ::testing::Test::RecordProperty("rejected_grid_values_bytes", grid_values_bytes);

    const auto cap = trial_peak + grid_values_bytes;
    auto participant = fixture.participant({32, cap});
    StageSamples samples(participant);
    ObservePrecommit observed(participant);
    Simulation simulation(observed);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_EQ(samples.samples.size(), 1u);
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);
    const auto before = committed(participant);
    const auto before_wall = checked(participant.read_committed_wall(30));
    const auto identity = WcsphTestAccess::accepted_precommit_identity(participant);
    const auto row_map = WcsphTestAccess::row_to_input(participant);
    const auto maximum_step = participant.time_constraint(0).maximum_step;
    const auto held_bytes = cap - refusal_limit;
    auto held = checked(WcsphTestAccess::reserve_test_budget(participant, held_bytes));
    ASSERT_TRUE(simulation.begin_window(dt).ok());
    {
        WindowAllocations rejected(participant);
        const auto failure = simulation.advance_trial();
        EXPECT_EQ(failure.code(), StatusCode::resource_exhausted);
        EXPECT_EQ(failure.message(), "participant resource budget exceeded");
        EXPECT_EQ(rejected.count(0, "sph.verlet.half_velocity", true), 1u);
        EXPECT_EQ(rejected.count(0, "sph.verlet.stage", true), 1u);
        EXPECT_EQ(rejected.count(0, "sph.grid.position", true), 0u);
        EXPECT_EQ(rejected.count(0, "sph.grid.mass", true), 0u);
        EXPECT_EQ(rejected.peak(0, rejected.events.size()), cap - (grid_values_bytes - 1));
        rejected.expect_all_released();
    }
    EXPECT_EQ(observed.prepare_calls, 0u);
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    EXPECT_EQ(simulation.time().trial_attempts, 1u);
    EXPECT_EQ(samples.samples.size(), 1u);
    expect_metadata(participant, simulation, 0, 0, maximum_step);
    expect_same_rows(committed(participant), before);
    expect_same_wall(checked(participant.read_committed_wall(30)), before_wall);
    EXPECT_EQ(WcsphTestAccess::accepted_precommit_identity(participant), identity);
    EXPECT_EQ(WcsphTestAccess::row_to_input(participant), row_map);
    EXPECT_EQ(participant.resource_usage().controlled_bytes, initial_bytes + held_bytes);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0u);

    held.reset();
    ASSERT_EQ(participant.resource_usage().controlled_bytes, initial_bytes);
    ASSERT_TRUE(simulation.advance(dt).ok());
    ASSERT_EQ(samples.samples.size(), reference_stages.samples.size());
    for (std::size_t i = 1; i < samples.samples.size(); ++i) {
        const auto& actual = samples.samples[i];
        const auto& expected = reference_stages.samples[i];
        EXPECT_EQ(actual.stage, expected.stage);
        EXPECT_EQ(actual.owned, (std::vector<std::uint64_t>{10, 20}));
        EXPECT_EQ(actual.neighbors, expected.neighbors);
        ASSERT_EQ(actual.particles.size(), source_count);
        for (std::size_t slot = 0; slot < source_count; ++slot) {
            EXPECT_EQ(actual.particles[slot].id, expected.particles[slot].id);
            EXPECT_EQ(actual.particles[slot].position, expected.particles[slot].position);
        }
        const double dx = double(actual.particles[1].position[0]) - actual.particles[0].position[0];
        EXPECT_EQ(actual.neighbors.at(10), (NeighborPairs{{20, dx * dx}}));
        EXPECT_EQ(actual.neighbors.at(20), (NeighborPairs{{10, dx * dx}}));
        EXPECT_TRUE(actual.neighbors.at(30).empty());
    }
    EXPECT_EQ(observed.prepare_calls, 1u);
    EXPECT_EQ(simulation.time().trial_attempts, 2u);
    expect_metadata(participant, simulation, dt, 1, end_step);
    expect_same_rows(committed(participant), end);
    expect_same_wall(checked(participant.read_committed_wall(30)), end_wall);
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
    enum class Kind { motion, limit };
    Kind kind;
    WallTime time;
    double phase;
};

using TrajectoryCalls = std::vector<TrajectoryCall>;

TrajectoryInput stationary_motion(double limit) {
    TrajectoryInput result;
    result.time_resolution = 0;
    result.motion = [](WallTime, double) -> Result<WallMotion> { return WallMotion{}; };
    result.maximum_sample_step = [limit](WallTime) -> Result<double> { return limit; };
    return result;
}

void expect_no_terminal_sample(const TrajectoryCalls& calls, double old_elapsed) {
    for (const auto& call : calls)
        EXPECT_EQ(call.time.elapsed, old_elapsed);
}

TEST(SphVerletWindow, MovingWallRelativeTimeResolutionRejectsInBoundRoundedSpan) {
    constexpr double t = 0x1p52;
    const Fields empty({}), particles({{{.25, 0, 0}, 2, 70, 1}});
    const sph::WcsphVerletParameters parameters{{1, 0x1p-54, 7}, .5};
    const sph::Velocity3D velocity("test.empty_velocity", 0);
    const sph::Values density("test.empty_density", 0);
    struct Case {
        double request, limit, resolution, span;
    };
    const std::array cases{Case{.25, 2, .25, 0}, Case{1.75, 1.75, .25, 2},
                           Case{1.75, 2, .125, 2}};
    for (std::size_t index = 0; index < cases.size(); ++index) {
        SCOPED_TRACE(index);
        const auto [request, limit, resolution, span] = cases[index];
        EXPECT_EQ((t + request) - t, span);
        ASSERT_LE(request, limit);
        TrajectoryCalls calls;
        auto trajectory = stationary_motion(t);
        trajectory.first_phase = 0;
        trajectory.last_phase = 2 * t;
        trajectory.time_resolution = resolution;
        trajectory.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
            calls.push_back({TrajectoryCall::Kind::motion, time, phase});
            return WallMotion{};
        };
        trajectory.maximum_sample_step = [&](WallTime time) -> Result<double> {
            calls.push_back({TrajectoryCall::Kind::limit, time, time.elapsed});
            if (time.elapsed == 0)
                return t;
            return limit;
        };
        const WallGroups groups{{11, wall_fields(particles), trajectory}};
        sph::WcsphParticipant3D participant(empty.get(), 64, parameters, velocity, density, groups);
        Simulation simulation(participant, -t);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        expect_metadata(participant, simulation, -t, 0, t);
        ASSERT_TRUE(simulation.advance(t).ok());
        expect_metadata(participant, simulation, 0, 1, limit);
        ASSERT_TRUE(committed(participant).rows.empty());
        const auto before = checked(participant.read_committed_wall(70));
        const auto before_groups = WcsphTestAccess::wall_groups(participant);
        ASSERT_EQ(before_groups.at(11), (WcsphTestAccess::WallGroupValue{0, t, limit}));
        EXPECT_EQ(before.phase, t);
        EXPECT_EQ(before.mass, 2);
        EXPECT_EQ(before.position, (std::array<real, 3>{.25, 0, 0}));
        calls.clear();

        ASSERT_TRUE(simulation.begin_window(request).ok());
        const auto failure = simulation.advance_trial();
        EXPECT_FALSE(failure.ok());
        expect_no_terminal_sample(calls, t);
        EXPECT_EQ(simulation.state(), SimulationState::prepared);
        expect_metadata(participant, simulation, 0, 1, limit);
        EXPECT_EQ(simulation.time().trial_attempts, 2u);
        expect_same_wall(checked(participant.read_committed_wall(70)), before);
        EXPECT_EQ(WcsphTestAccess::wall_groups(participant), before_groups);

        if (index == 2) {
            sph::WcsphParticipant3D direct(empty.get(), 64, parameters, velocity, density, groups);
            Simulation direct_sim(direct, -t);
            ASSERT_TRUE(direct_sim.initialize().ok());
            ASSERT_TRUE(direct_sim.prepare().ok());
            ASSERT_TRUE(direct_sim.advance(t).ok());
            ASSERT_TRUE(direct_sim.advance(1).ok());
            ASSERT_TRUE(simulation.advance(1).ok());
            expect_metadata(participant, simulation, 1, 2, limit);
            const auto retried = checked(participant.read_committed_wall(70));
            EXPECT_EQ(retried.phase, t + 1);
            expect_same_wall(retried, checked(direct.read_committed_wall(70)));
            EXPECT_EQ(WcsphTestAccess::wall_groups(participant),
                      WcsphTestAccess::wall_groups(direct));
            ASSERT_TRUE(direct_sim.finalize().ok());
        }
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphVerletWindow, MovingWallFiniteDomainRequiresPositivePostCommitSpan) {
    const Fields empty({}), particles({{{.25, 0, 0}, 1, 70, 1}});
    const sph::WcsphVerletParameters parameters{{1, .25, 7}, .5};
    const sph::Velocity3D velocity("test.empty_velocity", 0);
    const sph::Values density("test.empty_density", 0);
    for (double domain_end : {1., 1.5}) {
        SCOPED_TRACE(domain_end);
        TrajectoryCalls calls;
        auto trajectory = stationary_motion(1);
        trajectory.first_phase = 0;
        trajectory.last_phase = domain_end;
        trajectory.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
            calls.push_back({TrajectoryCall::Kind::motion, time, phase});
            WallMotion result;
            result.translation[0] = phase / 4;
            result.velocity[0] = .25;
            return result;
        };
        trajectory.maximum_sample_step = [&](WallTime time) -> Result<double> {
            calls.push_back({TrajectoryCall::Kind::limit, time, time.elapsed});
            const double remaining = domain_end - time.elapsed;
            if (remaining <= 0)
                return Status{StatusCode::numerical_failure, "test trajectory has no post-end span"};
            return std::min(1., remaining);
        };
        const WallGroups groups{{12, wall_fields(particles), trajectory}};
        sph::WcsphParticipant3D participant(empty.get(), 64, parameters, velocity, density, groups);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        expect_metadata(participant, simulation, 0, 0, 1);
        const auto before = checked(participant.read_committed_wall(70));
        ASSERT_EQ(before.position, (std::array<real, 3>{.25, 0, 0}));
        calls.clear();

        ASSERT_TRUE(simulation.begin_window(1).ok());
        const auto status = simulation.advance_trial();
        std::vector<TrajectoryCall::Kind> endpoint_calls;
        for (const auto& call : calls) {
            if (call.time.elapsed == 1) {
                EXPECT_EQ(call.phase, 1);
                endpoint_calls.push_back(call.kind);
            }
        }
        ASSERT_GE(endpoint_calls.size(), 2u);
        EXPECT_EQ(endpoint_calls.front(), TrajectoryCall::Kind::motion);
        EXPECT_NE(std::find(endpoint_calls.begin(), endpoint_calls.end(), TrajectoryCall::Kind::limit),
                  endpoint_calls.end());
        if (domain_end == 1) {
            EXPECT_FALSE(status.ok());
            EXPECT_EQ(simulation.state(), SimulationState::prepared);
            expect_metadata(participant, simulation, 0, 0, 1);
            expect_same_wall(checked(participant.read_committed_wall(70)), before);
            EXPECT_EQ(WcsphTestAccess::wall_groups(participant).at(12),
                      (WcsphTestAccess::WallGroupValue{0, 0, 1}));
        } else {
            ASSERT_TRUE(status.ok()) << status.message();
            EXPECT_EQ(WcsphTestAccess::wall_groups(participant, true).at(12),
                      (WcsphTestAccess::WallGroupValue{0, 1, .5}));
            ASSERT_TRUE(simulation.commit_window().ok());
            expect_metadata(participant, simulation, 1, 1, .5);
            const auto after = checked(participant.read_committed_wall(70));
            EXPECT_EQ(after.id, 70u);
            EXPECT_EQ(after.mass, 1);
            EXPECT_EQ(after.phase, 1);
            EXPECT_EQ(after.position, (std::array<real, 3>{.5, 0, 0}));
            EXPECT_EQ(after.stamp.geometry.value(), before.stamp.geometry.value() + 1);
            EXPECT_EQ(after.stamp.topology, before.stamp.topology);
            EXPECT_EQ(WcsphTestAccess::wall_groups(participant).at(12),
                      (WcsphTestAccess::WallGroupValue{0, 1, .5}));
        }
        EXPECT_EQ(simulation.time().trial_attempts, 1u);
        ASSERT_TRUE(simulation.finalize().ok());
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
    double separation, density, velocity;
};

std::array<PistonReference, 2> integrate_piston_reference(int steps) {
    constexpr double c2 = piston_sound_speed * piston_sound_speed;
    constexpr double inverse_four_pi = 1 / (4 * std::numbers::pi);
    const auto reference = [&](double time, double x, double velocity) {
        const double u = time - std::floor(time);
        const double wall_x = piston_reference_position + 2 * std::pow(u * (1 - u), 4);
        const double separation = x - wall_x;
        const double gap = std::max(2 - separation, 0.);
        return PistonReference{separation, 2 + piston_mass * std::pow(gap, 3) * inverse_four_pi,
                               velocity};
    };
    const auto acceleration = [&](double time, double x) {
        const auto [r, rho, v] = reference(time, x, 0);
        if (r >= 2)
            return 0.;
        const double u = time - std::floor(time);
        const double a = 8 * u*u * (1-u)*(1-u) * (3 - 14*u*(1-u));
        const double p = c2 * (rho - 1);
        const double pw = p + rho * a * r;
        const double rhow = 1 + pw / c2;
        return piston_mass * 3 * std::pow(2 - r, 2) * inverse_four_pi *
               (p / (rho * rho) + pw / (rhow * rhow));
    };

    const double dt = 1. / steps;
    double x{}, velocity{};
    std::array<PistonReference, 2> result{};
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
        if (step + 1 == steps / 2 || step + 1 == steps)
            result[(step + 1 == steps)] = reference(time + dt, x, velocity);
    }
    return result;
}

TEST(SphVerletWindow, RoundTripPistonResolvesTransientContact) {
    const auto reference = integrate_piston_reference(8192);
    const auto doubled = integrate_piston_reference(4096);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(reference[i].density, doubled[i].density, 4.e-14);
        EXPECT_NEAR(reference[i].velocity, doubled[i].velocity, 4.e-14);
    }
    EXPECT_NEAR(reference[0].separation, 1.994283047164, 1.e-12);
    EXPECT_NEAR(reference[1].separation, 2.002527610449, 1.e-12);
    EXPECT_NEAR(reference[0].density, 2.000121807465783, 1.e-14);
    EXPECT_NEAR(reference[0].velocity, .000611834268412, 1.e-14);
    EXPECT_NEAR(reference[1].velocity, .001113282602584, 1.e-14);
    EXPECT_GT(std::abs(reference[0].density - 2), 3.e-6);
    EXPECT_GT(std::abs(reference[1].velocity), 2.e-6);

    const Fields fluid({{{0, 0, 0}, 1, 11, 0}});
    const Fields particles({{{piston_reference_position, 0, 0}, piston_mass, 29, 1}});
    const sph::Velocity3D velocity("test.piston_fluid_velocity", 1);
    sph::Values density("test.piston_fluid_density", 1);
    density(0) = 2;
    const sph::WcsphVerletParameters parameters{{1, piston_sound_speed, 1}, 1};
    std::array<double, 3> density_error{}, speed_error{};
    for (int run = 0; run < 3; ++run) {
        const int steps = 32 << run;
        const double dt = 1. / steps;
        SCOPED_TRACE(steps);
        TrajectoryCalls calls;
        auto trajectory = stationary_motion(piston_limit);
        trajectory.initial_phase = 0;
        trajectory.period = 1;
        trajectory.motion = [&](WallTime time, double phase) -> Result<WallMotion> {
            calls.push_back({TrajectoryCall::Kind::motion, time, phase});
            const auto [x, v, a] = piston_signal(phase);
            WallMotion motion;
            motion.translation[0] = x;
            motion.velocity[0] = v;
            motion.acceleration[0] = a;
            return motion;
        };
        const WallGroups groups{{21, wall_fields(particles), trajectory}};
        sph::WcsphParticipant3D participant(fluid.get(), 64, parameters, velocity, density, groups);
        StageSamples stages(participant, true);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        EXPECT_DOUBLE_EQ(participant.time_constraint(0).maximum_step, piston_limit);
        EXPECT_LT(1., 1 / piston_sound_speed);
        if (run == 0) {
            ASSERT_EQ(stages.samples.size(), 1u);
            EXPECT_EQ(stages.samples.front().stage, StepStage3D::initial);
            EXPECT_TRUE(stages.samples.front().neighbors.at(11).empty());
            const auto before = committed(participant);
            EXPECT_FALSE(simulation.begin_window(1).ok());
            EXPECT_EQ(simulation.time().accepted_time, 0);
            EXPECT_EQ(simulation.time().accepted_steps, 0u);
            EXPECT_EQ(simulation.time().trial_attempts, 0u);
            expect_same_rows(committed(participant), before);
        }
        for (int step = 1; step <= steps; ++step) {
            const auto status = simulation.advance(dt);
            ASSERT_TRUE(status.ok()) << status.message();
            const double time = step * dt;
            EXPECT_DOUBLE_EQ(participant.time_constraint(time).maximum_step, piston_limit);
            if (step != steps / 2 && step != steps)
                continue;

            const auto now = committed(participant);
            ASSERT_EQ(now.rows.size(), 1u);
            const auto& row = now.rows.front();
            EXPECT_EQ(row.id, 11u);
            const auto wall_value = checked(participant.read_committed_wall(29));
            const auto& target = reference[step == steps];
            EXPECT_EQ(wall_value.phase, time);
            EXPECT_EQ(wall_value.position[0],
                      piston_reference_position + (step == steps / 2 ? 1. / 128 : 0.));
            EXPECT_EQ(wall_value.mass, piston_mass);
            EXPECT_NEAR(row.density, target.density, 3.e-6);
            EXPECT_NEAR(row.velocity[0], target.velocity, 2.e-6);
            EXPECT_NEAR(row.velocity[1], 0, 2.e-6);
            EXPECT_NEAR(row.velocity[2], 0, 2.e-6);
            density_error[run] = std::max(density_error[run], std::abs(row.density - target.density));
            speed_error[run] = std::max(speed_error[run], std::abs(row.velocity[0] - target.velocity));

            ASSERT_EQ(stages.samples.size(), 1 + 2 * static_cast<std::size_t>(step));
            const auto& endpoint = stages.samples.back();
            EXPECT_EQ(endpoint.stage, StepStage3D::end);
            const auto& pairs = endpoint.neighbors.at(11);
            if (step == steps / 2) {
                ASSERT_EQ(pairs.size(), 1u);
                EXPECT_EQ(pairs.front().first, 29u);
                const double distance = double(row.position[0]) - wall_value.position[0];
                EXPECT_DOUBLE_EQ(pairs.front().second, distance * distance);
            } else {
                EXPECT_TRUE(pairs.empty());
                EXPECT_EQ(endpoint.velocity.at(29), (std::array<real, 3>{}));
                ASSERT_FALSE(calls.empty());
                EXPECT_EQ(calls.back().time.elapsed, 1);
                EXPECT_EQ(calls.back().phase, 0);
                EXPECT_EQ(piston_signal(calls.back().phase), (std::array<double, 3>{}));
            }
        }
        ASSERT_TRUE(simulation.finalize().ok());
        std::ostringstream density_text, speed_text;
        density_text << std::scientific << std::setprecision(16) << density_error[run];
        speed_text << std::scientific << std::setprecision(16) << speed_error[run];
        ::testing::Test::RecordProperty("density_error_" + std::to_string(steps), density_text.str());
        ::testing::Test::RecordProperty("speed_error_" + std::to_string(steps), speed_text.str());
    }
    if constexpr (std::is_same_v<real, double>) {
        EXPECT_GT(density_error[0], density_error[1]);
        EXPECT_GT(density_error[1], density_error[2]);
        EXPECT_GT(speed_error[0], speed_error[1]);
        EXPECT_GT(speed_error[1], speed_error[2]);
    }
}


void expect_kinematics(const wall::WallKinematics3D& actual, const std::array<double, 3>& position,
                       const std::array<double, 3>& velocity, const std::array<double, 3>& acceleration) {
    const auto eps = std::numeric_limits<real>::epsilon();
    for (std::size_t axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(actual.position[axis], position[axis], 16 * eps);
        EXPECT_NEAR(actual.velocity[axis], velocity[axis], 16 * eps);
        EXPECT_NEAR(actual.acceleration[axis], acceleration[axis], 16 * eps);
    }
}

TEST(SphVerletWindow, MovingWallsSampleTranslationAndRotationAtGroupPhase) {
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
        motion.translation = {.5 * phase * phase, -.25 * phase, 0};
        motion.velocity = {phase, -.25, 0};
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

    const Kokkos::Serial exec;
    const wall::PrescribedMovingWall3D moving_left(
        checked(wall::capture_wall_particles(exec, wall_fields(translated))), translation);
    const wall::PrescribedMovingWall3D moving_right(
        checked(wall::capture_wall_particles(exec, wall_fields(rotated))), rotation);
    const WallGroups groups{{31, wall_fields(translated), translation},
                            {41, wall_fields(rotated), rotation}};
    sph::WcsphParticipant3D participant(empty.get(), 128, parameters, velocity, density, groups);
    StageSamples stages(participant, true);
    Simulation simulation(participant, initial_time);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());

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
        const auto left = checked(moving_left.sample(10, WallTime{translation_offset, elapsed}));
        const auto right = checked(moving_right.sample(20, WallTime{rotation_offset, elapsed}));
        expect_kinematics(left, translation_position, {translation_phase, -.25, 0}, {1, 0, 0});
        expect_kinematics(right, rotation_position, {-angular_speed * y, angular_speed * x, 0},
                          {-y - angular_speed * angular_speed * x,
                           x - angular_speed * angular_speed * y, 0});
        const auto committed_left = checked(participant.read_committed_wall(10));
        const auto committed_right = checked(participant.read_committed_wall(20));
        EXPECT_EQ(committed_left.position, left.position);
        EXPECT_EQ(committed_right.position, right.position);
        EXPECT_EQ(committed_left.phase, translation_phase);
        EXPECT_EQ(committed_right.phase, rotation_phase);
        EXPECT_EQ(committed_left.mass, 2);
        EXPECT_EQ(committed_right.mass, 3);
        expect_metadata(participant, simulation, initial_time + elapsed, step, .5);
        if (step == 0) {
            ASSERT_TRUE(simulation.advance(dt).ok());
        } else {
            ASSERT_EQ(stages.samples.size(), 3u);
            EXPECT_EQ(stages.samples.back().velocity.at(10), left.velocity);
            EXPECT_EQ(stages.samples.back().velocity.at(20), right.velocity);
        }
    }
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphVerletWindow, MovingWallRejectsInvalidGroupsAndSamples) {
    const Fields fluid({{{0, 0, 0}, 1, 11, 0}});
    const Fields wall_a({{{2, 0, 0}, 1, 21, 1}}), wall_b({{{3, 0, 0}, 2, 21, 1}}),
        duplicate_fluid({{{4, 0, 0}, 2, 11, 1}}), other_wall({{{5, 0, 0}, 3, 32, 1}});
    const sph::WcsphVerletParameters parameters{{1, .5, 7}, 1};
    const sph::Velocity3D velocity("test.velocity", 1);
    sph::Values density("test.density", 1);
    density(0) = 1;
    const auto rejects = [&](WallGroups groups) {
        sph::WcsphParticipant3D participant(fluid.get(), 64, parameters, velocity, density, groups);
        Simulation simulation(participant);
        EXPECT_FALSE(simulation.initialize().ok());
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
    invalid.maximum_sample_step = {};
    rejects({{1, wall_fields(wall_a), invalid}});
    using LimitFunction = Result<double> (*)(WallTime);
    invalid.maximum_sample_step = static_cast<LimitFunction>(nullptr);
    rejects({{1, wall_fields(wall_a), invalid}});
    rejects({{1, wall_fields(wall_a), stationary_motion(0)}});
    rejects({{1, wall_fields(wall_a), stationary_motion(std::numeric_limits<double>::infinity())}});
}

TEST(SphVerletWindow, MovingWallLocalDurationSurvivesLargeAbsoluteTime) {
    constexpr double dt = .75, large_time = 0x1p52;
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
        calls.push_back({TrajectoryCall::Kind::motion, time, phase});
        WallMotion result;
        result.translation[0] = time.elapsed;
        result.velocity[0] = 1;
        return result;
    };
    const WallGroups groups{{5, wall_fields(moving), trajectory}};
    sph::WcsphParticipant3D small(fluid.get(), 64, parameters, velocity, density, groups);
    sph::WcsphParticipant3D large(fluid.get(), 64, parameters, velocity, density, groups);
    StageSamples small_stages(small, true), large_stages(large, true);
    Simulation small_sim(small), large_sim(large, large_time);
    ASSERT_TRUE(small_sim.initialize().ok());
    ASSERT_TRUE(large_sim.initialize().ok());
    ASSERT_TRUE(small_sim.prepare().ok());
    ASSERT_TRUE(large_sim.prepare().ok());
    const auto check = [&](Simulation& simulation) {
        calls.clear();
        ASSERT_TRUE(simulation.advance(dt).ok());
        ASSERT_FALSE(calls.empty());
        bool saw_start = false, saw_end = false;
        for (const auto& call : calls) {
            EXPECT_EQ(call.time.initial_phase, .375);
            if (call.time.elapsed == dt) {
                EXPECT_EQ(call.phase, 1.125);
                saw_end = true;
            } else {
                EXPECT_EQ(call.time.elapsed, 0);
                EXPECT_EQ(call.phase, .375);
                saw_start = true;
            }
        }
        EXPECT_TRUE(saw_start);
        EXPECT_TRUE(saw_end);
    };
    ASSERT_NO_FATAL_FAILURE(check(small_sim));
    ASSERT_NO_FATAL_FAILURE(check(large_sim));
    EXPECT_EQ(small_sim.time().accepted_time, dt);
    EXPECT_EQ(large_sim.time().accepted_time, large_time + 1);

    const auto expected = committed(small);
    const auto actual = committed(large);
    expect_same_rows(actual, expected);
    ASSERT_EQ(actual.rows.size(), 1u);
    EXPECT_EQ(actual.rows.front().position[0], .75);
    EXPECT_EQ(actual.rows.front().pressure, 0);
    EXPECT_EQ(actual.rows.front().density, 1);
    EXPECT_EQ(actual.rows.front().velocity, (std::array<real, 3>{1, 0, 0}));
    const auto wall_small = checked(small.read_committed_wall(22));
    const auto wall_large = checked(large.read_committed_wall(22));
    EXPECT_EQ(wall_large.position, wall_small.position);
    EXPECT_EQ(wall_large.position[0], .25);
    EXPECT_EQ(wall_large.phase, wall_small.phase);
    EXPECT_EQ(WcsphTestAccess::wall_groups(small), WcsphTestAccess::wall_groups(large));
    ASSERT_EQ(small_stages.samples.size(), 3u);
    ASSERT_EQ(large_stages.samples.size(), 3u);
    for (std::size_t stage = 0; stage < 3; ++stage) {
        const auto& a = small_stages.samples[stage];
        const auto& b = large_stages.samples[stage];
        EXPECT_EQ(a.neighbors, b.neighbors);
        EXPECT_EQ(a.neighbors.at(11), (NeighborPairs{{22, .25}}));
        EXPECT_EQ(a.velocity, b.velocity);
    }
    ASSERT_TRUE(small_sim.finalize().ok());
    ASSERT_TRUE(large_sim.finalize().ok());
}

TEST(SphVerletWindow, MovingWallLimitsComposeAndUseNextEndpointBound) {
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
        ASSERT_TRUE(simulation.prepare().ok());
        expect_metadata(participant, simulation, 0, 0, maximum);
        ASSERT_TRUE(simulation.begin_window(.125).ok());
        ASSERT_TRUE(simulation.advance_trial().ok());
        const auto candidate = WcsphTestAccess::wall_groups(participant, true);
        EXPECT_EQ(candidate.at(5), (WcsphTestAccess::WallGroupValue{0, .125, limits[0]}));
        EXPECT_EQ(candidate.at(7), (WcsphTestAccess::WallGroupValue{0, .125, limits[1]}));
        ASSERT_TRUE(simulation.commit_window().ok());
        expect_metadata(participant, simulation, .125, 1, maximum);
        EXPECT_EQ(WcsphTestAccess::wall_groups(participant), candidate);
        ASSERT_TRUE(simulation.finalize().ok());
    }

    auto changing_limit = stationary_motion(1);
    changing_limit.maximum_sample_step = [](WallTime time) -> Result<double> {
        return time.elapsed == 0 ? 1. : .25;
    };
    const WallGroups groups{{5, wall_fields(left), changing_limit},
                            {7, wall_fields(right), stationary_motion(4)}};
    sph::WcsphParticipant3D participant(fluid.get(), 64, parameters, velocity, density, groups);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    expect_metadata(participant, simulation, 0, 0, 1);
    ASSERT_TRUE(simulation.advance(.5).ok());
    expect_metadata(participant, simulation, .5, 1, .25);
    EXPECT_EQ(WcsphTestAccess::wall_groups(participant).at(5),
              (WcsphTestAccess::WallGroupValue{0, .5, .25}));
    const auto before = committed(participant);
    const auto above = std::nextafter(.25, std::numeric_limits<double>::infinity());
    EXPECT_FALSE(simulation.begin_window(above).ok());
    expect_metadata(participant, simulation, .5, 1, .25);
    expect_same_rows(committed(participant), before);
    ASSERT_TRUE(simulation.begin_window(.25).ok());
    ASSERT_TRUE(simulation.restore_window().ok());
    ASSERT_TRUE(simulation.finalize().ok());

    sph::WcsphParticipant3D direct(fluid.get(), 64, parameters, velocity, density, groups);
    ASSERT_TRUE(direct.initialize(0).ok());
    ASSERT_TRUE(direct.prepare().ok());
    const TimeInterval start{0, .5, .5};
    ASSERT_TRUE(direct.begin_window(start).ok());
    ASSERT_TRUE(direct.advance_trial(start).ok());
    ASSERT_TRUE(direct.prepare_commit(start).ok());
    ASSERT_TRUE(direct.commit_window(start).ok());
    EXPECT_FALSE(direct.begin_window({.5, .5 + above, above}).ok());
    ASSERT_TRUE(direct.begin_window({.5, .75, .25}).ok());
    ASSERT_TRUE(direct.restore_window().ok());
    ASSERT_TRUE(direct.finalize().ok());
}

TEST(SphVerletWindow, MovingWallOnlyPublishesCrossingAndRestoresGroupState) {
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
    left_motion.maximum_sample_step = [](WallTime time) -> Result<double> {
        return time.elapsed == 0 ? .5 : (time.elapsed == .5 ? .375 : .25);
    };
    right_motion.maximum_sample_step = [](WallTime time) -> Result<double> {
        return time.elapsed == 0 ? .75 : .625;
    };
    const WallGroups groups{{4, wall_fields(left), left_motion},
                            {7, wall_fields(right), right_motion}};
    for (double retry : {.375, .25}) {
        SCOPED_TRACE(retry);
        sph::WcsphParticipant3D participant(empty.get(), 64, parameters, velocity, density, groups);
        sph::WcsphParticipant3D direct(empty.get(), 64, parameters, velocity, density, groups);
        StageSamples stages(participant);
        WindowAllocations observed(participant);
        Simulation simulation(participant, 3), direct_sim(direct, 3);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        ASSERT_TRUE(direct_sim.initialize().ok());
        ASSERT_TRUE(direct_sim.prepare().ok());
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
        ASSERT_EQ(stages.samples.size(), 1u);
        ASSERT_EQ(stages.samples[0].particles.size(), 2u);
        EXPECT_EQ(stages.samples[0].particles.front().id, 80u);
        ASSERT_TRUE(simulation.advance(.5).ok());
        ASSERT_TRUE(direct_sim.advance(.5).ok());
        ASSERT_NO_FATAL_FAILURE(check(.5, 1, .375, 1));
        ASSERT_EQ(stages.samples.size(), 3u);
        ASSERT_EQ(stages.samples.back().particles.size(), 2u);
        EXPECT_EQ(stages.samples.back().particles.front().id, 10u);
        const auto accepted = WcsphTestAccess::wall_groups(participant);
        EXPECT_EQ(accepted.at(4), (WcsphTestAccess::WallGroupValue{.25, .5, .375}));
        EXPECT_EQ(accepted.at(7), (WcsphTestAccess::WallGroupValue{.75, .5, .625}));
        const auto wall_before = checked(participant.read_committed_wall(80));

        ASSERT_TRUE(simulation.begin_window(.375).ok());
        ASSERT_TRUE(simulation.advance_trial().ok());
        const auto trial = WcsphTestAccess::wall_groups(participant, true);
        EXPECT_EQ(trial.at(4), (WcsphTestAccess::WallGroupValue{.25, .875, .25}));
        EXPECT_EQ(trial.at(7), (WcsphTestAccess::WallGroupValue{.75, .875, .625}));
        ASSERT_TRUE(simulation.restore_window().ok());
        ASSERT_NO_FATAL_FAILURE(check(.5, 1, .375, 1));
        EXPECT_EQ(WcsphTestAccess::wall_groups(participant), accepted);
        expect_same_wall(checked(participant.read_committed_wall(80)), wall_before);
        ASSERT_TRUE(simulation.advance(retry).ok());
        ASSERT_TRUE(direct_sim.advance(retry).ok());
        // Independently, x/2 bins give initial order [80,10], then [10,80] on
        // both commits. Input slots follow the previous sorted order, changing
        // the sorted-to-input map from [0,1] to [1,0] and back to [0,1].
        ASSERT_NO_FATAL_FAILURE(check(.5 + retry, 2, .25, 2));
        ASSERT_FALSE(stages.samples.empty());
        ASSERT_EQ(stages.samples.back().particles.size(), 2u);
        EXPECT_EQ(stages.samples.back().particles.front().id, 10u);
        EXPECT_EQ(WcsphTestAccess::wall_groups(participant), WcsphTestAccess::wall_groups(direct));
        for (const auto id : {10u, 80u})
            expect_same_wall(checked(participant.read_committed_wall(id)),
                             checked(direct.read_committed_wall(id)));
        for (const auto label : {"sph.pressure_over_density_squared", "sph.pressure.needed_ghost",
                                 "sph.wall.fluid_row", "sph.wall.acceleration"})
            EXPECT_EQ(observed.count(0, label, true), 0u) << label;
        ASSERT_TRUE(simulation.finalize().ok());
        ASSERT_TRUE(direct_sim.finalize().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    }
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
    FixedWallStep result{state, {}, after.velocity, {}, {}};
    const auto first_force = mixed_boundary_force(state, before, gravity);
    auto old_sources = state.particles;
    old_sources.insert(old_sources.end(), before.walls.begin(), before.walls.end());
    auto old_velocity = state.velocity;
    old_velocity.insert(before.velocity.begin(), before.velocity.end());
    for (auto& p : result.end.particles) {
        const double a = first_force.acceleration.at(p.id) + gravity;
        const double prior = state.velocity.at(p.id)[0];
        result.half[p.id][0] = static_cast<real>(std::fma(.5 * dt, a, prior));
        result.predicted[p.id][0] = static_cast<real>(std::fma(dt, a, prior));
        p.position[0] = static_cast<real>(
            std::fma(dt, double(result.half.at(p.id)[0]), double(p.position[0])));
    }
    auto new_sources = result.end.particles;
    new_sources.insert(new_sources.end(), after.walls.begin(), after.walls.end());
    for (std::size_t i = 0; i < state.particles.size(); ++i) {
        const auto id = state.particles[i].id;
        const auto r0 = density_rate(state.particles[i], old_sources, old_velocity);
        const auto r1 = density_rate(result.end.particles[i], new_sources, result.predicted);
        result.start_rate.emplace(id, r0);
        result.end_rate.emplace(id, r1);
        result.end.density[id] = static_cast<real>(
            std::fma(.5 * dt, double(r0 + r1), double(state.density.at(id))));
    }
    const auto final_force = mixed_boundary_force(result.end, after, gravity);
    for (const auto& p : result.end.particles) {
        const double a = final_force.acceleration.at(p.id) + gravity;
        result.end.velocity[p.id][0] = static_cast<real>(
            std::fma(.5 * dt, a, double(result.half.at(p.id)[0])));
    }
    return result;
}

std::vector<Particle> mixed_boundary_order(const Trajectory& stage, const MixedBoundaryState& boundary) {
    auto result = stage.particles;
    result.insert(result.end(), boundary.walls.begin(), boundary.walls.end());
    const auto key = [](const Particle& p) { return std::pair{std::floor(double(p.position[0]) / 2), p.id}; };
    std::sort(result.begin(), result.end(), [&](const auto& a, const auto& b) { return key(a) < key(b); });
    return result;
}

void expect_mixed_query(const StageSample& sample, StepStage3D stage, const Trajectory& fluid,
                        const MixedBoundaryState& walls) {
    constexpr double epsilon = std::numeric_limits<real>::epsilon();
    const auto expected = mixed_boundary_order(fluid, walls);
    ASSERT_EQ(sample.particles.size(), expected.size());
    EXPECT_EQ(sample.stage, stage);
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

TEST(SphVerletWindow, MovingWallGroupsPreserveIdentityAcrossPermutations) {
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
    StageSamples stages(participant, true), reordered_stages(permuted, true);
    Simulation simulation(participant), permuted_sim(permuted);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(permuted_sim.initialize().ok());
    ASSERT_TRUE(permuted_sim.prepare().ok());

    const auto start = mixed_boundary_at(fixed, moving, 0);
    const auto end = mixed_boundary_at(fixed, moving, dt);
    const auto expected = mixed_boundary_step(initial, start, end, gravity, dt);
    ASSERT_LT(start.walls[2].position[0], 2);
    ASSERT_GT(end.walls[2].position[0], 2);
    EXPECT_NE(start.velocity.at(70), end.velocity.at(70));
    const auto check = [&](const Trajectory& fluid_state, const MixedBoundaryState& stage) {
        const auto observed = committed(participant);
        expect_same_rows(observed, committed(permuted));
        expect_mixed_rows(observed, fluid_state, stage, gravity);
        for (const auto& p : stage.walls) {
            const auto row = checked(participant.read_committed_wall(p.id));
            expect_same_wall(row, checked(permuted.read_committed_wall(p.id)));
            EXPECT_EQ(row.id, p.id);
            EXPECT_EQ(row.mass, p.mass);
            EXPECT_EQ(row.position, p.position);
        }
    };
    ASSERT_NO_FATAL_FAILURE(check(initial, start));
    ASSERT_TRUE(simulation.advance(dt).ok());
    ASSERT_TRUE(permuted_sim.advance(dt).ok());
    ASSERT_NO_FATAL_FAILURE(check(expected.end, end));
    ASSERT_EQ(stages.samples.size(), 3u);
    ASSERT_EQ(reordered_stages.samples.size(), 3u);
    expect_mixed_query(stages.samples[0], StepStage3D::initial, initial, start);
    expect_mixed_query(stages.samples[1], StepStage3D::start, initial, start);
    expect_mixed_query(stages.samples[2], StepStage3D::end, expected.end, end);
    for (std::size_t i = 0; i < stages.samples.size(); ++i)
        EXPECT_EQ(stages.samples[i].neighbors, reordered_stages.samples[i].neighbors);
    auto start_velocity = initial.velocity;
    start_velocity.insert(start.velocity.begin(), start.velocity.end());
    expect_stage_kinematics(stages.samples[1], start_velocity, expected.half);
    expect_stage_kinematics(stages.samples[2], expected.predicted);
    const auto rates = WcsphTestAccess::rates(participant);
    ASSERT_EQ(rates[0].size(), 2u);
    ASSERT_EQ(rates[1].size(), 2u);
    constexpr double eps = std::numeric_limits<real>::epsilon();
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(rates[0][i], expected.start_rate.at(stages.samples[1].owned[i]), 32 * eps);
        EXPECT_NEAR(rates[1][i], expected.end_rate.at(stages.samples[2].owned[i]), 32 * eps);
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
    ASSERT_TRUE(constant_sim.prepare().ok());
    ASSERT_TRUE(fixed_sim.prepare().ok());
    expect_same_rows(committed(constant), committed(fixed_only));
    ASSERT_TRUE(constant_sim.advance(dt).ok());
    ASSERT_TRUE(fixed_sim.advance(dt).ok());
    expect_same_rows(committed(constant), committed(fixed_only));
    EXPECT_DOUBLE_EQ(constant.time_constraint(dt).maximum_step,
                     fixed_only.time_constraint(dt).maximum_step);
    ASSERT_TRUE(constant_sim.finalize().ok());
    ASSERT_TRUE(fixed_sim.finalize().ok());
}

TEST(SphVerletWindow, MovingWallFiniteDomainRejectsRoundedUpRelativeWidth) {
    using namespace sph::SphBoundary;
    constexpr double T = 0x1p52;
    const Fields fluid({}), wall({{{.25, 0, 0}, 1, 70, 1}});
    std::vector<double> sampled;
    PrescribedWallTrajectory3D trajectory;
    trajectory.initial_phase = -.75;
    trajectory.first_phase = -.75;
    trajectory.last_phase = T;
    trajectory.motion = [&](WallTrajectoryTime3D time, double) -> Result<RigidWallMotion3D> {
        sampled.push_back(time.elapsed);
        return RigidWallMotion3D{};
    };
    trajectory.maximum_sample_step = [](WallTrajectoryTime3D) -> Result<double> { return 2 * T; };

    const Kokkos::Serial exec;
    const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
    const PrescribedMovingWall3D moving(checked(capture_wall_particles(exec, boundary)), trajectory);
    ASSERT_TRUE(moving.validate().ok());
    EXPECT_EQ(checked(moving.maximum_sample_step({-.75, 0})), T);
    EXPECT_FALSE(moving.advance_time({-.75, 0}, T + 1, T + 1).ok());
    EXPECT_FALSE(moving.motion_at({-.75, T + 1}).ok());
    EXPECT_TRUE(sampled.empty());

    sph::WcsphParticipant3D participant(fluid.get(), 1, {{1, 0x1p-54, 1}, .5}, {},
        sph::Values{}, std::vector<WallGroupInput3D>{{1, boundary, trajectory}});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_EQ(sampled, (std::vector<double>{0.}));
    EXPECT_EQ(participant.time_constraint(0).maximum_step, T);
    EXPECT_EQ(simulation.begin_window(T + 1).code(), StatusCode::invalid_argument);
    EXPECT_EQ(sampled, (std::vector<double>{0.}));
    EXPECT_EQ(simulation.time().accepted_time, 0.);
    EXPECT_EQ(simulation.time().trial_attempts, 0u);
    EXPECT_TRUE(simulation.finalize().ok());
}

TEST(SphVerletWindow, MovingWallPeriodicFoldAndThrowingCallbackRecover) {
    using namespace sph::SphBoundary;
    constexpr double T = 0x1p52;
    const Fields fluid({}), wall({{{.25, 0, 0}, 1, 70, 1}});
    std::vector<double> phases;
    sph::WcsphParticipant3D* active{};
    StatusCode reinitialize{}, restore{}, finalize{};
    PrescribedWallTrajectory3D trajectory;
    trajectory.initial_phase = -.25;
    trajectory.period = T;
    trajectory.motion = [&](WallTrajectoryTime3D time, double phase) -> Result<RigidWallMotion3D> {
        if (phases.empty())
            reinitialize = active->initialize(0).code();
        if (time.elapsed == .5) {
            restore = active->restore_window().code();
            finalize = active->finalize().code();
            throw 1;
        }
        phases.push_back(phase);
        return RigidWallMotion3D{};
    };
    trajectory.maximum_sample_step = [](WallTrajectoryTime3D) -> Result<double> { return 1.; };
    const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
    sph::WcsphParticipant3D participant(fluid.get(), 1, {{1, .25, 1}, .5}, {}, sph::Values{},
        std::vector<WallGroupInput3D>{{1, boundary, trajectory}});
    active = &participant;
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_EQ(phases.size(), 1u);
    EXPECT_EQ(reinitialize, StatusCode::invalid_state);
    EXPECT_GE(phases.front(), 0.);
    EXPECT_LT(phases.front(), T);
    const auto before = checked(participant.read_committed_wall(70));

    ASSERT_TRUE(simulation.begin_window(.5).ok());
    EXPECT_EQ(simulation.advance_trial().code(), StatusCode::execution_failure);
    EXPECT_EQ(restore, StatusCode::invalid_state);
    EXPECT_EQ(finalize, StatusCode::invalid_state);
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    EXPECT_EQ(checked(participant.read_committed_wall(70)), before);
    EXPECT_EQ(simulation.time().accepted_steps, 0u);
    EXPECT_EQ(simulation.time().accepted_time, 0.);
    ASSERT_TRUE(simulation.advance(1.).ok());
    EXPECT_EQ(phases.back(), .75);
    EXPECT_EQ(simulation.time().accepted_time, 1.);
    EXPECT_EQ(simulation.time().accepted_steps, 1u);
    EXPECT_EQ(simulation.time().trial_attempts, 2u);
    EXPECT_TRUE(simulation.finalize().ok());
}

TEST(SphVerletWindow, MovingWallRejectsReentrantSimulationMutation) {
    struct NestedCall {
        const char* name;
        Status (Simulation::*call)();
    };
    constexpr double full_step = .5;
    const Fields fluid({}), boundary({{{.25, 0, 0}, 1, 70, 1}});
    const sph::WcsphVerletParameters parameters{{1, .25, 1}, .5};
    for (const auto nested : {NestedCall{"restore_window", &Simulation::restore_window},
                              NestedCall{"finalize", &Simulation::finalize},
                              NestedCall{"advance_trial", &Simulation::advance_trial}}) {
        for (const bool propagate : {true, false}) {
            SCOPED_TRACE(nested.name);
            SCOPED_TRACE(propagate);
            Simulation* active{};
            std::optional<Status> rejection;
            auto motion = stationary_motion(1.);
            motion.motion = [&](WallTime time, double) -> Result<WallMotion> {
                if (time.elapsed == full_step) {
                    EXPECT_EQ(active->state(), SimulationState::window_open);
                    rejection = (active->*nested.call)();
                    EXPECT_EQ(rejection->code(), StatusCode::invalid_state);
                    EXPECT_EQ(active->state(), SimulationState::window_open);
                    EXPECT_EQ(active->time().trial_attempts, 1u);
                    if (propagate && !rejection->ok())
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
            ASSERT_TRUE(simulation.prepare().ok());
            const auto initial = checked(participant.read_committed_wall(70));

            double accepted_step = full_step;
            const auto attempted = simulation.advance(full_step);
            ASSERT_TRUE(rejection.has_value());
            EXPECT_EQ(simulation.state(), SimulationState::prepared);
            EXPECT_EQ(simulation.time().trial_attempts, 1u);
            if (propagate) {
                EXPECT_EQ(attempted.code(), rejection->code());
                EXPECT_EQ(attempted.message(), rejection->message());
                EXPECT_EQ(simulation.time().accepted_time, 0.);
                EXPECT_EQ(simulation.time().accepted_steps, 0u);
                EXPECT_EQ(checked(participant.read_committed_wall(70)), initial);
                accepted_step = full_step / 2;
                const auto retry = simulation.advance(accepted_step);
                ASSERT_TRUE(retry.ok()) << retry.message();
            } else {
                ASSERT_TRUE(attempted.ok()) << attempted.message();
            }
            EXPECT_EQ(simulation.state(), SimulationState::prepared);
            EXPECT_EQ(simulation.time().accepted_time, accepted_step);
            EXPECT_EQ(simulation.time().accepted_steps, 1u);
            EXPECT_EQ(simulation.time().trial_attempts, propagate ? 2u : 1u);
            EXPECT_EQ(checked(participant.read_committed_wall(70)).position[0],
                      initial.position[0] + accepted_step);
            EXPECT_TRUE(simulation.finalize().ok());
            EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
        }
    }
}

TEST(SphVerletWindow, MovingWallPreparedCandidateRestoresPhasesBeforeReplay) {
    const Fields empty({}), wall_fields_input({{{.25, 0, 0}, 3, 71, 1}});
    auto motion = stationary_motion(.5);
    motion.initial_phase = .25;
    motion.motion = [](WallTime time, double) -> Result<WallMotion> {
        WallMotion value;
        value.translation[0] = time.elapsed;
        value.velocity[0] = 1;
        return value;
    };
    motion.maximum_sample_step = [](WallTime time) -> Result<double> {
        return time.elapsed <= .25 ? .5 : .25;
    };
    const WallGroups groups{{70, wall_fields(wall_fields_input), motion}};
    const sph::WcsphVerletParameters parameters{{1, .25, 1}, .5};
    sph::WcsphParticipant3D participant(empty.get(), 1, parameters, {}, {}, groups);
    sph::WcsphParticipant3D direct(empty.get(), 1, parameters, {}, {}, groups);
    for (auto* current : {&participant, &direct}) {
        ASSERT_TRUE(current->initialize(2).ok());
        ASSERT_TRUE(current->prepare().ok());
        constexpr TimeInterval start{2, 2.25, .25};
        ASSERT_TRUE(current->begin_window(start).ok());
        ASSERT_TRUE(current->advance_trial(start).ok());
        ASSERT_TRUE(current->prepare_commit(start).ok());
        ASSERT_TRUE(current->commit_window(start).ok());
        ASSERT_TRUE(current->release_workspace().ok());
    }
    const auto before = checked(participant.read_committed_wall(71));
    const auto accepted = WcsphTestAccess::wall_groups(participant);
    const auto bytes = participant.resource_usage().controlled_bytes;
    ASSERT_EQ(accepted.at(70), (WcsphTestAccess::WallGroupValue{.25, .25, .5}));
    constexpr TimeInterval retry{2.25, 2.75, .5};
    ASSERT_TRUE(participant.begin_window(retry).ok());
    ASSERT_TRUE(participant.advance_trial(retry).ok());
    const auto trial = WcsphTestAccess::wall_groups(participant, true);
    ASSERT_EQ(trial.at(70), (WcsphTestAccess::WallGroupValue{.25, .75, .25}));
    ASSERT_TRUE(participant.prepare_commit(retry).ok());
    EXPECT_EQ(WcsphTestAccess::pending_wall_groups(participant), trial);
    EXPECT_EQ(participant.read_committed_wall(71).status().code(), StatusCode::invalid_state);
    ASSERT_TRUE(participant.restore_window().ok());
    EXPECT_EQ(checked(participant.read_committed_wall(71)), before);
    EXPECT_EQ(WcsphTestAccess::wall_groups(participant), accepted);
    EXPECT_EQ(participant.time_constraint(2.25).maximum_step, .5);
    ASSERT_TRUE(participant.release_workspace().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, bytes);

    for (auto* current : {&participant, &direct}) {
        ASSERT_TRUE(current->begin_window(retry).ok());
        ASSERT_TRUE(current->advance_trial(retry).ok());
        ASSERT_TRUE(current->prepare_commit(retry).ok());
        ASSERT_TRUE(current->commit_window(retry).ok());
    }
    EXPECT_EQ(checked(participant.read_committed_wall(71)), checked(direct.read_committed_wall(71)));
    EXPECT_EQ(WcsphTestAccess::wall_groups(participant), trial);
    EXPECT_EQ(participant.time_constraint(2.75).maximum_step, .25);
    EXPECT_TRUE(participant.finalize().ok());
    EXPECT_TRUE(direct.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}


TEST(SphVerletWindow, MovingWallTrajectoryStorageIsAdmittedBeforeCapture) {
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
    WindowAllocations events(refused);
    EXPECT_EQ(refused.initialize(0).code(), StatusCode::resource_exhausted);
    EXPECT_EQ(events.count(0, "sph.wall.particles", true), 0u);
    EXPECT_EQ(refused.resource_usage().controlled_bytes, 0u);
    EXPECT_TRUE(refused.finalize().ok());
}

} // namespace
