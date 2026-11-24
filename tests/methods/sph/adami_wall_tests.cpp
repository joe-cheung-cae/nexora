#include "cell_grid_oracle.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <methods/sph/AdamiWallPressure3D.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <numbers>
#include <participants/sph/WcsphParticipant3D.hpp>

namespace {
using namespace nexora;
using namespace sph_test;

std::vector<sph::CommittedRow3D> committed_rows(const sph::WcsphParticipant3D& participant) {
    std::vector<sph::CommittedRow3D> rows;
    const auto status =
        participant.visit_committed_rows([&](sph::CommittedRow3D row) { rows.push_back(row); });
    if (!status.ok())
        throw std::runtime_error(status.message());
    return rows;
}

TEST(SphAdamiWall, InitialAdamiFieldsAndAcousticBoundScaleWithSmoothingLength) {
    struct Initial {
        double density, pressure, acceleration;
    };
    std::array<Initial, 2> reference{};
    for (const real h : {real(1), real(.5), real(2)}) {
        const real volume = h * h * h;
        Fields fluid({{{0, 0, 0}, volume, 10, 0}, {{real(.75) * h, 0, 0}, 2 * volume, 20, 0}});
        Fields wall({{{real(-.5) * h, 0, 0}, volume, 30, 1}});
        const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
        sph::Velocity3D velocity("test.h_scaled_velocity", 2);
        velocity(0, 0) = 1;
        const sph::WcsphVerletParameters parameters{{.5, 3, 1}, .25, {-.125 / h, 0, 0}, h};
        sph::Values density("test.scaled_physical_density", 2);
        density(0) = fluid.mass(0) / volume;
        density(1) = fluid.mass(1) / volume;
        sph::WcsphParticipant3D participant(fluid.get(), 100, parameters, velocity, density, boundary);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        const auto rows = committed_rows(participant);
        ASSERT_EQ(rows.size(), reference.size());
        EXPECT_EQ(participant.time_constraint(0).maximum_step, .25 * h / 4);
        for (const auto& row : rows) {
            const auto i = row.id == 10 ? 0 : 1;
            const Initial observed{row.density, row.pressure, h * row.pressure_acceleration[0]};
            if (h == 1) {
                reference[i] = observed;
                EXPECT_NE(observed.acceleration, 0);
            } else {
                EXPECT_EQ(observed.density, reference[i].density);
                EXPECT_EQ(observed.pressure, reference[i].pressure);
                EXPECT_EQ(observed.acceleration, reference[i].acceleration);
            }
        }
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphAdamiWall, GravityOnlyMatchesFreeFall) {
    Fields fluid({{{0, 0, 0}, 1, 41, 0}});
    sph::Velocity3D velocity("wall_test.initial_velocity", 1);
    const std::array<double, 3> gravity{1, -2, -4};
    const std::array<double, 3> initial_velocity{2, 1, -1};
    for (int axis = 0; axis < 3; ++axis)
        velocity(0, axis) = static_cast<real>(initial_velocity[axis]);
    sph::WcsphParticipant3D participant(fluid.get(), 512, {{1, 2, 2}, .8, gravity}, velocity, fluid.mass);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    for (int step = 1; step <= 16; ++step) {
        constexpr double dt = 1. / 32;
        ASSERT_TRUE(simulation.advance(dt).ok());
        const double t = step * dt;
        const auto rows = committed_rows(participant);
        ASSERT_EQ(rows.size(), 1u);
        const auto& x = rows.front().position;
        const auto& v = rows.front().velocity;
        EXPECT_EQ(rows.front().pressure_acceleration, (std::array<real, 3>{}));
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_EQ(v[axis], initial_velocity[axis] + gravity[axis] * t);
            EXPECT_EQ(x[axis], initial_velocity[axis] * t + .5 * gravity[axis] * t * t);
        }
    }
    EXPECT_TRUE(simulation.finalize().ok());
}

struct WallForceReference {
    std::map<std::uint64_t, real> fluid_pressure;
    std::map<std::uint64_t, long double> wall_pressure;
    std::map<std::uint64_t, double> coefficient;
    std::map<std::uint64_t, std::array<long double, 3>> acceleration;
};

WallForceReference independent_wall_forces(
    const std::vector<Particle>& particles, const std::map<std::uint64_t, real>& density,
    const sph::WcsphVerletParameters& parameters,
    const std::array<double, 3>& wall_acceleration = {}) {
    constexpr long double pi = std::numbers::pi_v<long double>;
    const long double rho0 = parameters.eos.reference_density;
    const long double gamma = parameters.eos.gamma;
    const long double h = parameters.smoothing_length;
    const long double bulk = rho0 * std::pow(parameters.eos.sound_speed, 2) / gamma;
    const auto distance = [](const Particle& a, const Particle& b) {
        return std::hypot(static_cast<long double>(a.position[0]) - b.position[0],
                          static_cast<long double>(a.position[1]) - b.position[1],
                          static_cast<long double>(a.position[2]) - b.position[2]);
    };
    const auto coefficient = [](real p, real rho) { return double(p) / (double(rho) * rho); };
    WallForceReference result;
    for (const auto& [id, rho] : density) {
        const auto p = static_cast<real>(bulk * (std::pow(rho / rho0, gamma) - 1));
        result.fluid_pressure.emplace(id, p);
        result.coefficient.emplace(id, coefficient(p, rho));
    }
    for (const auto& wall : particles) {
        if (!wall.ghost)
            continue;
        long double weight_sum{}, weighted_pressure{};
        for (const auto& fluid : particles) {
            const auto q = distance(wall, fluid) / h;
            if (fluid.ghost || q >= 2)
                continue;
            const auto weight =
                (std::pow(2 - q, 3) - 4 * std::pow(std::max(1 - q, 0.L), 3)) / (4 * pi);
            long double head{};
            for (int k = 0; k < 3; ++k)
                head += (static_cast<long double>(parameters.gravity[k]) - wall_acceleration[k]) *
                        (static_cast<long double>(wall.position[k]) - fluid.position[k]);
            weight_sum += weight;
            weighted_pressure += weight *
                (result.fluid_pressure.at(fluid.id) + density.at(fluid.id) * head);
        }
        if (weight_sum > 0) {
            const long double raw = weighted_pressure / weight_sum;
            const auto p = static_cast<real>(raw);
            const auto rho = static_cast<real>(rho0 * std::pow(1 + p / bulk, 1 / gamma));
            result.wall_pressure.emplace(wall.id, raw);
            result.coefficient.emplace(wall.id, coefficient(p, rho));
        }
    }
    for (const auto& fluid : particles) {
        if (fluid.ghost)
            continue;
        std::array<long double, 3> acceleration{};
        for (const auto& other : particles) {
            const auto q = distance(fluid, other) / h;
            if (q == 0 || q >= 2)
                continue;
            const auto grad = q < 1 ? (3 - 2.25L * q) / pi : .75L * (2 - q) * (2 - q) / (pi * q);
            const auto pair = other.mass * grad / std::pow(h, 5) *
                              (result.coefficient.at(fluid.id) + result.coefficient.at(other.id));
            for (int k = 0; k < 3; ++k)
                acceleration[k] += pair *
                    (static_cast<long double>(fluid.position[k]) - other.position[k]);
        }
        result.acceleration.emplace(fluid.id, acceleration);
    }
    return result;
}

void expect_wall_reference(const std::vector<sph::CommittedRow3D>& rows, const WallForceReference& reference,
                           const sph::WcsphVerletParameters& parameters) {
    const double eps = std::numeric_limits<real>::epsilon();
    const double c2 = parameters.eos.sound_speed * parameters.eos.sound_speed;
    const double pressure_scale = parameters.eos.reference_density * c2 / parameters.eos.gamma;
    ASSERT_EQ(rows.size(), reference.fluid_pressure.size());
    for (const auto& row : rows) {
        const auto id = row.id;
        ASSERT_TRUE(reference.fluid_pressure.contains(id));
        EXPECT_NEAR(row.pressure, reference.fluid_pressure.at(id),
                    64 * eps * pressure_scale);
        const auto observed = row.pressure_acceleration;
        const auto& expected = reference.acceleration.at(id);
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(observed[k], expected[k], 128 * eps *
                        std::max<long double>(c2 / parameters.eos.gamma, std::abs(expected[k])));
    }
}

// The all-pairs reference leaves the detached dry wall at its original coordinates.
void check_wall_forces(const std::vector<sph::CommittedRow3D>& rows, std::vector<Particle> particles,
                       const sph::WcsphVerletParameters& parameters) {
    std::map<std::uint64_t, real> density;
    std::map<std::uint64_t, std::array<real, 3>> positions;
    for (const auto& row : rows) {
        ASSERT_TRUE(positions.emplace(row.id, row.position).second);
        density.emplace(row.id, row.density);
    }
    for (auto& particle : particles)
        if (!particle.ghost) {
            ASSERT_TRUE(positions.contains(particle.id));
            particle.position = positions.at(particle.id);
        }
    expect_wall_reference(rows, independent_wall_forces(particles, density, parameters),
                          parameters);
}

TEST(SphAdamiWall, WallPressureAndForcesMatchIndependentReference) {
    const std::vector<Particle> fluid{
        {{.875, .5, 0}, 2.5, 20, 0}, {{.125, 0, 0}, 1.5, 40, 0}};
    const Particle wall{{.125, 0, 0}, 4, 99, 1};
    auto particles = fluid;
    particles.push_back(wall);
    const sph::WcsphVerletParameters parameters{{1, 2, 3}, .75, {-.25, .125, -.375}, 1.25};
    const sph::TaitEos eos(parameters.eos);
    const std::array<double, 3> fixed_acceleration{};
    const Kokkos::Serial exec;
    const auto grid = checked(sph::CellGrid::build(exec, Fields(particles).get(), 64, {},
                                                   parameters.smoothing_length));
    const auto& sources = grid.fields();
    const auto owned = grid.owned_slots();
    sph::CellGrid::Indices::non_const_type row_of_slot("wall_test.row_of_slot", particles.size());
    LocalIndex wall_slot{};
    for (std::size_t slot = 0; slot < particles.size(); ++slot)
        if (sources.ids(slot) == wall.id)
            wall_slot = static_cast<LocalIndex>(slot);
    ASSERT_EQ(sources.ids(wall_slot), wall.id);
    for (std::size_t row = 0; row < fluid.size(); ++row)
        row_of_slot(owned(row)) = static_cast<LocalIndex>(row);

    const Fields fluid_fields(fluid), wall_fields({wall});
    const sph::FixedWallParticles3D boundary{wall_fields.position, wall_fields.mass, wall_fields.ids};
    const sph::Velocity3D velocity("wall_test.velocity", fluid.size());
    for (const real coincident_density : {real{1.125}, real{1.375}}) {
        SCOPED_TRACE(coincident_density);
        // The offset fluid marks the wall. Only the coincident fluid's density changes.
        const std::map<std::uint64_t, real> density{{20, real{1.25}}, {40, coincident_density}};
        const auto reference = independent_wall_forces(particles, density, parameters);
        sph::Values owned_density("wall_test.owned_density", fluid.size());
        sph::Values owned_pressure("wall_test.owned_pressure", fluid.size());
        sph::Values input_density("wall_test.input_density", fluid.size());
        for (std::size_t row = 0; row < fluid.size(); ++row) {
            const auto id = sources.ids(owned(row));
            owned_density(row) = density.at(id);
            owned_pressure(row) = reference.fluid_pressure.at(id);
            input_density(row) = density.at(fluid[row].id);
        }
        const double raw = sph::wall_pressure(grid.neighbors(), wall_slot, row_of_slot, owned_density,
                                               owned_pressure, fixed_acceleration, parameters.gravity);
        const auto coefficient = sph::wall_coefficient(grid.neighbors(), wall_slot, row_of_slot, owned_density,
                                                       owned_pressure, fixed_acceleration,
                                                       parameters.gravity, eos);
        ASSERT_TRUE(coefficient.has_value());
        EXPECT_NEAR(raw, reference.wall_pressure.at(wall.id),
                    32 * std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(*coefficient, reference.coefficient.at(wall.id),
                    16 * std::numeric_limits<real>::epsilon());

        sph::WcsphParticipant3D participant(fluid_fields.get(), 64, parameters, velocity, input_density,
                                       boundary);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        expect_wall_reference(committed_rows(participant), reference, parameters);
        EXPECT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphAdamiWall, AcceleratedWallPressureAndForcesMatchIndependentHead) {
    const std::vector<Particle> particles{
        {{.125, .75, .5}, 1.5, 20, 0},
        {{.875, .125, .5}, 2.5, 40, 0},
        {{.25, .375, .5}, 4, 99, 1}};
    const std::map<std::uint64_t, real> density{{20, real{1.25}}, {40, real{1.125}}};
    const sph::WcsphVerletParameters parameters{{1, 2, 3}, .75, {-.25, .125, -.375}, 1.25};
    const std::array<double, 3> wall_acceleration{1.5, -1, .5};
    const auto expected = independent_wall_forces(particles, density, parameters, wall_acceleration);
    const auto stationary = independent_wall_forces(particles, density, parameters);
    EXPECT_GT(std::abs(expected.wall_pressure.at(99) - stationary.wall_pressure.at(99)), .1);

    const Kokkos::Serial exec;
    const auto grid = checked(sph::CellGrid::build(exec, Fields(particles).get(), 64, {},
                                                   parameters.smoothing_length));
    const auto& query = grid.neighbors();
    const auto& sources = query.fields();
    const auto owned = query.owned_slots();
    sph::Values rho("moving_wall_test.density", 2);
    sph::Values fluid_pressure("moving_wall_test.fluid_pressure", 2);
    sph::NeighborView3D::Indices::non_const_type row_of_slot("moving_wall_test.row_of_slot", 3);
    LocalIndex wall_slot{};
    for (std::size_t slot = 0; slot < sources.ids.extent(0); ++slot)
        if (sources.ids(slot) == 99)
            wall_slot = static_cast<LocalIndex>(slot);
    ASSERT_EQ(sources.ids(wall_slot), 99u);
    for (std::size_t row = 0; row < owned.extent(0); ++row) {
        const auto slot = owned(row);
        const auto id = sources.ids(slot);
        row_of_slot(slot) = static_cast<LocalIndex>(row);
        rho(row) = density.at(id);
        fluid_pressure(row) = expected.fluid_pressure.at(id);
    }

    const sph::TaitEos eos(parameters.eos);
    const auto raw = sph::wall_pressure(query, wall_slot, row_of_slot, rho, fluid_pressure,
                                        wall_acceleration, parameters.gravity);
    const auto coefficient = sph::wall_coefficient(query, wall_slot, row_of_slot, rho, fluid_pressure,
                                                   wall_acceleration, parameters.gravity, eos);
    ASSERT_TRUE(coefficient.has_value());
    EXPECT_NEAR(raw, expected.wall_pressure.at(99), 64 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(*coefficient, expected.coefficient.at(99), 32 * std::numeric_limits<real>::epsilon());

    sph::WallAcceleration3D wall_a("moving_wall_test.acceleration", 3);
    for (int k = 0; k < 3; ++k)
        wall_a(wall_slot, k) = wall_acceleration[k];
    sph::Values pressure("moving_wall_test.pressure", 2);
    sph::Acceleration3D acceleration("moving_wall_test.fluid_acceleration", 2);
    ASSERT_TRUE(sph::evaluate_pressure_3d(exec, query, eos, rho, pressure, acceleration, wall_a,
                                          parameters.gravity).ok());
    std::vector<sph::CommittedRow3D> observed;
    for (std::size_t row = 0; row < owned.extent(0); ++row)
        observed.push_back({.id = sources.ids(owned(row)),
                            .pressure = pressure(row),
                            .pressure_acceleration =
                                {acceleration(row, 0), acceleration(row, 1), acceleration(row, 2)}});
    ASSERT_NO_FATAL_FAILURE(expect_wall_reference(observed, expected, parameters));
}

TEST(SphAdamiWall, FixedSourcesMatchIndependentWallForcesAcrossRestore) {
    std::vector<Particle> fluid{{{1.875, .25, .5}, 2, 33, 0}, {{1, .5, .5}, 3, 11, 0}};
    const std::vector<Particle> walls{
        {{1.5, 0, -.25}, 2, 44, 1}, {{.5, -.5, .25}, 1, 20, 1}, {{12, 0, 0}, 2, 55, 1}};
    Fields input(fluid), fixed(walls);
    sph::Velocity3D velocity("wall_test.velocity", fluid.size());
    velocity(0, 0) = 5;
    const sph::WcsphVerletParameters parameters{{1, 2, 3}, .75, {0, -.25, -1}};
    const sph::FixedWallParticles3D boundary{fixed.position, fixed.mass, fixed.ids};
    sph::Values initial_density("wall_test.initial_density", fluid.size());
    initial_density(0) = real{1.125};
    initial_density(1) = real{1.25};
    sph::WcsphParticipant3D participant(input.get(), 4096, parameters, velocity, initial_density, boundary);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    // Caller may reuse all wall input buffers after initialization.
    fixed.position(0, 2) = 10;
    fixed.mass(0) = 7;
    fixed.ids(0) = 33;
    fluid.insert(fluid.end(), walls.begin(), walls.end());
    const auto initial = committed_rows(participant);
    ASSERT_EQ(initial.size(), 2u);
    ASSERT_NO_FATAL_FAILURE(check_wall_forces(initial, fluid, parameters));
    for (const auto& row : initial) {
        const double vx = row.id == 33 ? 5 : 0;
        EXPECT_EQ(row.density, row.id == 33 ? 1.125 : 1.25);
        EXPECT_EQ(row.velocity[0], vx);
    }
    ASSERT_TRUE(simulation.begin_window(.05).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_TRUE(simulation.restore_window().ok());
    EXPECT_EQ(committed_rows(participant), initial);

    // Inspect the same endpoint through an independent publication from the initial state.
    const Fields direct_wall(walls);
    const sph::FixedWallParticles3D direct_boundary{direct_wall.position, direct_wall.mass,
                                                   direct_wall.ids};
    sph::WcsphParticipant3D direct(input.get(), 4096, parameters, velocity, initial_density,
                                   direct_boundary);
    Simulation direct_sim(direct);
    ASSERT_TRUE(direct_sim.initialize().ok());
    ASSERT_TRUE(direct_sim.prepare().ok());
    ASSERT_TRUE(direct_sim.advance(.05).ok());
    ASSERT_NO_FATAL_FAILURE(check_wall_forces(committed_rows(direct), fluid, parameters));
    ASSERT_TRUE(direct_sim.finalize().ok());

    ASSERT_TRUE(simulation.advance(.025).ok());
    ASSERT_NO_FATAL_FAILURE(check_wall_forces(committed_rows(participant), fluid, parameters));
    EXPECT_TRUE(simulation.finalize().ok());
}

TEST(SphAdamiWall, RejectsInvalidGravityWallIdentityAndGhostFluid) {
    Fields input({{{0, 0, 0}, 1, 12, 0}}), wall({{{0, .25, 1}, 2, 21, 0}});
    sph::Velocity3D velocity("wall_test.validation_velocity", 1);
    const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
    sph::WcsphVerletParameters parameters{{1, 2, 3}, .5};
    parameters.gravity[1] = std::numeric_limits<double>::infinity();
    sph::WcsphParticipant3D invalid_g(input.get(), 128, parameters, velocity, input.mass, boundary);
    EXPECT_FALSE(invalid_g.initialize(0).ok());
    parameters.gravity[1] = 0;
    wall.ids(0) = 12;
    sph::WcsphParticipant3D duplicate(input.get(), 128, parameters, velocity, input.mass, boundary);
    EXPECT_FALSE(duplicate.initialize(0).ok());
    wall.ids(0) = 21;
    input.ghost(0) = 1;
    sph::WcsphParticipant3D ghost(input.get(), 128, parameters, velocity, input.mass, boundary);
    EXPECT_FALSE(ghost.initialize(0).ok());
}

TEST(SphAdamiWall, InverseEosFailureRetainsAcceptedStateForShorterRetry) {
    Fields fluid({{{0, 0, 0}, 1, 1, 0}}), wall({{{0, 0, 2.125}, 1, 2, 0}});
    sph::Velocity3D velocity("wall_test.approach", 1);
    velocity(0, 2) = 3;
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, 1, {0, 0, -10}};
    const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
    sph::WcsphParticipant3D participant(fluid.get(), 128, parameters, velocity, fluid.mass, boundary);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto accepted = committed_rows(participant);
    // The longer step brings an overhead wall into support. Its extrapolated
    // pressure falls below -B, where the wall has no positive Tait density.
    const auto status = simulation.advance(.1);
    EXPECT_EQ(status.code(), StatusCode::numerical_failure);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    EXPECT_EQ(committed_rows(participant), accepted);
    ASSERT_TRUE(simulation.advance(.01).ok());
    EXPECT_NEAR(committed_rows(participant).at(0).velocity[2], 2.9,
                4 * std::numeric_limits<real>::epsilon());
    EXPECT_TRUE(simulation.finalize().ok());
}
} // namespace
