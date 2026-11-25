#include "cell_grid_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <nexora/simulation/Simulation.hpp>
#include <numbers>
#include <participants/sph/WcsphParticipant3D.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace nexora;
using namespace sph_test;

constexpr int width = 6;
constexpr int depth = 6;
constexpr int shell_layers = 2;
constexpr double g = 9.81;
constexpr sph::WcsphParameters eos{1000, 80, 7};
constexpr double pressure_scale = eos.reference_density * g * depth;
const double gravity_time = std::sqrt(depth / g);
const double gravity_speed = std::sqrt(depth * g);
constexpr std::array face_names{"x_min", "x_max", "y_min", "y_max", "bottom"};
constexpr std::array region_names{"bulk", "side", "edge", "bottom", "free_surface"};

double exact_density(double z) {
    // Integrate dp/dz = -rho*g with p(depth)=0 and the Tait EOS.
    const double relative_depth =
        (eos.gamma - 1) * g * (depth - z) / (eos.sound_speed * eos.sound_speed);
    return eos.reference_density * std::pow(1 + relative_depth, 1 / (eos.gamma - 1));
}

double exact_pressure(double z) {
    // For the all-particle error, extend the hydrostatic formula below the
    // bottom and use ambient gauge pressure above the reference free surface.
    const double relative_depth =
        (eos.gamma - 1) * g * std::max(depth - z, 0.) / (eos.sound_speed * eos.sound_speed);
    return eos.reference_density * eos.sound_speed * eos.sound_speed / eos.gamma *
           std::expm1(eos.gamma / (eos.gamma - 1) * std::log1p(relative_depth));
}

struct Tank {
    std::vector<Particle> fluid;
    std::vector<Particle> wall;
};

Tank make_tank() {
    Tank tank;
    for (int z = -shell_layers; z < depth; ++z)
        for (int y = -shell_layers; y < width + shell_layers; ++y)
            for (int x = -shell_layers; x < width + shell_layers; ++x) {
                const bool is_fluid = x >= 0 && x < width && y >= 0 && y < width && z >= 0;
                const auto id = is_fluid ? tank.fluid.size() + 1 : 10000 + tank.wall.size();
                const auto z_center = static_cast<real>(z + .5);
                Particle particle{{static_cast<real>(x + .5), static_cast<real>(y + .5), z_center},
                                  static_cast<real>(exact_density(z_center)),
                                  id,
                                  0};
                (is_fluid ? tank.fluid : tank.wall).push_back(particle);
            }
    return tank;
}

double weight(const Particle& lhs, const Particle& rhs) {
    const double q = std::hypot(double(lhs.position[0]) - rhs.position[0],
                                double(lhs.position[1]) - rhs.position[1],
                                double(lhs.position[2]) - rhs.position[2]);
    const auto cube = [](double value) { return value * value * value; };
    return (cube(std::max(2 - q, 0.)) - 4 * cube(std::max(1 - q, 0.))) / (4 * std::numbers::pi);
}

void fit_fluid_mass(Tank& tank) {
    // One-time fixture quadrature. The support integral is truncated at the open
    // surface. Choose positive fluid masses so the same lattice starts near the
    // continuum EOS density. No data are recalibrated during integration.
    struct Row {
        double wall_density{};
        std::vector<std::pair<std::size_t, double>> fluid_weights;
    };
    std::vector<Row> rows(tank.fluid.size());
    std::vector<double> masses;
    for (std::size_t i = 0; i < tank.fluid.size(); ++i) {
        const auto& particle = tank.fluid[i];
        masses.push_back(particle.mass);
        for (const auto& wall : tank.wall)
            rows[i].wall_density += weight(particle, wall) * wall.mass;
        for (std::size_t j = 0; j < tank.fluid.size(); ++j) {
            const double w = weight(particle, tank.fluid[j]);
            if (w > 0)
                rows[i].fluid_weights.emplace_back(j, w);
        }
    }
    auto updated = masses;
    for (int iteration = 0; iteration < 200; ++iteration) {
        for (std::size_t i = 0; i < masses.size(); ++i) {
            double fluid_density = 0;
            for (const auto& [j, w] : rows[i].fluid_weights)
                fluid_density += w * masses[j];
            const double target = exact_density(tank.fluid[i].position[2]) - rows[i].wall_density;
            updated[i] = masses[i] * target / fluid_density;
        }
        masses.swap(updated);
    }
    for (std::size_t i = 0; i < masses.size(); ++i)
        tank.fluid[i].mass = static_cast<real>(masses[i]);
}

struct SquaredError {
    double mass{};
    double square_sum{};
    double max_absolute{};
    void add(double weight, double error) {
        mass += weight;
        square_sum += weight * error * error;
        max_absolute = std::max(max_absolute, std::abs(error));
    }
    double rms() const {
        return std::sqrt(square_sum / mass);
    }
};

using Center = std::array<real, 3>;

std::array<double, face_names.size()> distances_to_planes(const Center& x) {
    return {double(x[0]), width - double(x[0]), double(x[1]), width - double(x[1]), double(x[2])};
}

bool hits_rectangular_face(std::size_t face, const Center& from, const Center& to,
                           double fraction) {
    const std::array limits{width, width, depth};
    const auto normal_axis = face / 2;
    for (std::size_t axis = 0; axis < limits.size(); ++axis) {
        if (axis == normal_axis)
            continue;
        const double coordinate = std::lerp(double(from[axis]), double(to[axis]), fraction);
        if (coordinate < 0 || coordinate > limits[axis])
            return false;
    }
    return true;
}

struct Observation {
    SquaredError pressure, in_tank_pressure, speed, force_residual;
    std::size_t in_tank_count{};
    std::array<SquaredError, region_names.size()> pressure_region;
    std::array<double, face_names.size()> clearance;
    std::array<std::size_t, face_names.size()> outside{};
    std::vector<Center> centers_by_id;
    struct Layer {
        std::size_t count{};
        double mass{}, pressure_sum{}, reference_sum{};
    };
    std::array<Layer, depth> pressure_layer;

    Observation() {
        clearance.fill(std::numeric_limits<double>::infinity());
    }
};

Observation observe(sph::WcsphParticipant3D& participant, const Tank& tank) {
    Observation observed;
    observed.centers_by_id.resize(tank.fluid.size());
    std::size_t row_count = 0;
    const auto read = participant.visit_committed_rows([&](sph::CommittedRow3D row) {
        ++row_count;
        const auto id = row.id;
        const double mass = row.mass;
        EXPECT_EQ(mass, tank.fluid.at(id - 1).mass);
        const auto& x = row.position;
        const auto& v = row.velocity;
        const auto& a = row.pressure_acceleration;
        const double pressure = row.pressure;
        const auto finite = [](const Center& vector) {
            return std::all_of(vector.begin(), vector.end(),
                               [](real component) { return std::isfinite(component); });
        };
        if (!std::isfinite(pressure) || !std::isfinite(row.density) || !finite(x) ||
            !finite(v) || !finite(a)) {
            ADD_FAILURE() << "Nonfinite tank fields for fluid particle " << id;
            return;
        }
        const double reference_pressure = exact_pressure(x[2]);
        const double error = pressure - reference_pressure;
        observed.centers_by_id.at(id - 1) = x;
        const auto distances = distances_to_planes(x);
        for (std::size_t face = 0; face < face_names.size(); ++face) {
            observed.clearance[face] = std::min(observed.clearance[face], distances[face]);
            observed.outside[face] += distances[face] <= 0;
        }
        const bool near_x = std::min(distances[0], distances[1]) < 2;
        const bool near_y = std::min(distances[2], distances[3]) < 2;
        const bool near_bottom = distances[4] < 2;
        const bool near_surface = depth - x[2] < 2;
        // Regions overlap deliberately. Edge includes the intersection of two
        // solid faces. Bottom and free surface include their lateral edges.
        if (!near_x && !near_y && !near_bottom && !near_surface)
            observed.pressure_region[0].add(mass, error);
        if (near_x || near_y)
            observed.pressure_region[1].add(mass, error);
        if (int(near_x) + int(near_y) + int(near_bottom) >= 2)
            observed.pressure_region[2].add(mass, error);
        if (near_bottom)
            observed.pressure_region[3].add(mass, error);
        if (near_surface)
            observed.pressure_region[4].add(mass, error);
        // The profile measures surviving fluid in the initial water volume.
        // The upper bin includes centers on the reference free-surface plane.
        if (x[2] <= depth &&
            std::all_of(distances.begin(), distances.end(), [](double d) { return d >= 0; })) {
            auto& layer = observed.pressure_layer[std::min(static_cast<int>(x[2]), depth - 1)];
            ++layer.count;
            layer.mass += mass;
            layer.pressure_sum += mass * pressure;
            layer.reference_sum += mass * reference_pressure;
            ++observed.in_tank_count;
            observed.in_tank_pressure.add(mass, error);
        }
        observed.pressure.add(mass, error);
        observed.speed.add(mass, std::hypot(v[0], v[1], v[2]));
        observed.force_residual.add(mass, std::hypot(double(a[0]), double(a[1]), a[2] - g));
    });
    EXPECT_TRUE(read.ok()) << read.message();
    EXPECT_EQ(row_count, tank.fluid.size());
    return observed;
}

void record(std::string name, double value) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(17) << value;
    ::testing::Test::RecordProperty(name, text.str());
}

void record_endpoint(const std::string& prefix, const Observation& observed) {
    record(prefix + "_pressure_rms_pa", observed.pressure.rms());
    record(prefix + "_pressure_max_pa", observed.pressure.max_absolute);
    record(prefix + "_speed_rms_m_per_s", observed.speed.rms());
    record(prefix + "_speed_max_m_per_s", observed.speed.max_absolute);
    record(prefix + "_pressure_rms_over_hydrostatic_head",
           observed.pressure.rms() / pressure_scale);
    record(prefix + "_pressure_max_over_hydrostatic_head",
           observed.pressure.max_absolute / pressure_scale);
    record(prefix + "_speed_rms_over_gravity_speed", observed.speed.rms() / gravity_speed);
    record(prefix + "_speed_max_over_gravity_speed", observed.speed.max_absolute / gravity_speed);
    record(prefix + "_force_residual_rms_over_g", observed.force_residual.rms() / g);
    record(prefix + "_in_tank_count", observed.in_tank_count);
    record(prefix + "_in_tank_mass_kg", observed.in_tank_pressure.mass);
    if (observed.in_tank_count > 0)
        record(prefix + "_in_tank_pressure_rms_pa", observed.in_tank_pressure.rms());
    for (std::size_t layer = 0; layer < observed.pressure_layer.size(); ++layer) {
        const auto& profile = observed.pressure_layer[layer];
        const std::string label = prefix + "_z_bin_" + std::to_string(layer);
        record(label + "_count", profile.count);
        record(label + "_mass_kg", profile.mass);
        if (profile.count == 0)
            continue;
        record(label + "_mean_pressure", profile.pressure_sum / profile.mass);
        record(label + "_mean_reference_pressure", profile.reference_sum / profile.mass);
    }
    for (std::size_t region = 0; region < region_names.size(); ++region) {
        const auto& error = observed.pressure_region[region];
        if (error.mass == 0)
            continue;
        const std::string label = prefix + "_" + region_names[region];
        record(label + "_pressure_rms_over_hydrostatic_head", error.rms() / pressure_scale);
        record(label + "_pressure_max_over_hydrostatic_head", error.max_absolute / pressure_scale);
    }
}

void record_masses(const Tank& tank, const std::string& prefix) {
    double total_mass = 0;
    double minimum_mass = std::numeric_limits<double>::infinity();
    double maximum_mass = 0;
    for (const auto& particle : tank.fluid) {
        ASSERT_TRUE(std::isfinite(particle.mass));
        ASSERT_GT(particle.mass, 0);
        total_mass += particle.mass;
        minimum_mass = std::min(minimum_mass, double(particle.mass));
        maximum_mass = std::max(maximum_mass, double(particle.mass));
    }
    record(prefix + "fluid_total_mass_kg", total_mass);
    record(prefix + "fluid_minimum_mass_kg", minimum_mass);
    record(prefix + "fluid_maximum_mass_kg", maximum_mass);
}

void run_tank(const Tank& tank, bool uncompensated) {
    const std::string prefix = uncompensated ? "uncompensated_" : "";
    ASSERT_NO_FATAL_FAILURE(record_masses(tank, prefix));
    const Fields wall(tank.wall);
    const sph::Velocity3D rest("tank.initial_rest", tank.fluid.size());
    const sph::WcsphVerletParameters parameters{eos, .5, {0, 0, -g}};
    const sph::FixedWallParticles3D fixed{wall.position, wall.mass, wall.ids};
    const Fields fluid(tank.fluid);
    const sph::Values initial_density("tank.initial_density", tank.fluid.size());
    for (std::size_t i = 0; i < tank.fluid.size(); ++i)
        initial_density(i) = static_cast<real>(exact_density(tank.fluid[i].position[2]));
    sph::WcsphParticipant3D participant(fluid.get(), 10000, parameters, rest, initial_density, fixed);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());

    const auto initial = observe(participant, tank);
    record_endpoint(prefix + "initial", initial);

    constexpr double maximum_dt = 1. / 256;
    const double end_time = std::ceil(3 * gravity_time / maximum_dt) * maximum_dt;
    double minimum_dt = maximum_dt;
    double peak_rms_speed = 0;
    double peak_max_speed = 0;
    double peak_rms_pressure_error = initial.pressure.rms();
    auto clearance = initial.clearance;
    auto outside_samples = initial.outside;
    std::array<std::size_t, face_names.size()> plane_crossings{}, face_crossings{};
    std::vector<bool> crossed_face(tank.fluid.size());
    auto previous_centers = initial.centers_by_id;
    Observation last = initial;
    int sample = 1;
    while (simulation.time().accepted_time < end_time) {
        const double time = simulation.time().accepted_time;
        const double cfl_bound = participant.time_constraint().maximum_step;
        const double dt = std::min({maximum_dt, .9 * cfl_bound, end_time - time});
        ASSERT_GT(time + dt, time);
        minimum_dt = std::min(minimum_dt, dt);
        const auto result = simulation.run(dt);
        ASSERT_TRUE(result.ok()) << "time " << time << " s, " << result.message();
        last = observe(participant, tank);
        peak_rms_speed = std::max(peak_rms_speed, last.speed.rms());
        peak_max_speed = std::max(peak_max_speed, last.speed.max_absolute);
        peak_rms_pressure_error = std::max(peak_rms_pressure_error, last.pressure.rms());
        for (std::size_t face = 0; face < face_names.size(); ++face) {
            clearance[face] = std::min(clearance[face], last.clearance[face]);
            outside_samples[face] += last.outside[face];
        }
        // Each Verlet drift segment is linear between its accepted endpoints.
        // Plane events can include escapes over the rim. Count actual crossings
        // only when the intersection lies on the closed rectangular face.
        for (std::size_t i = 0; i < tank.fluid.size(); ++i) {
            const auto& old_center = previous_centers[i];
            const auto& new_center = last.centers_by_id[i];
            const auto old_distances = distances_to_planes(old_center);
            const auto new_distances = distances_to_planes(new_center);
            for (std::size_t face = 0; face < face_names.size(); ++face) {
                if (!(old_distances[face] > 0 && new_distances[face] <= 0))
                    continue;
                ++plane_crossings[face];
                const double fraction =
                    old_distances[face] / (old_distances[face] - new_distances[face]);
                if (hits_rectangular_face(face, old_center, new_center, fraction)) {
                    ++face_crossings[face];
                    crossed_face[i] = true;
                }
            }
        }
        if (sample <= 3 && simulation.time().accepted_time >= sample * gravity_time) {
            const auto sample_prefix = prefix + "sample_" + std::to_string(sample++);
            record(sample_prefix + "_elapsed_over_gravity_time",
                   simulation.time().accepted_time / gravity_time);
            record(sample_prefix + "_speed_rms_over_gravity_speed",
                   last.speed.rms() / gravity_speed);
            record(sample_prefix + "_speed_max_over_gravity_speed",
                   last.speed.max_absolute / gravity_speed);
            record(sample_prefix + "_pressure_rms_over_hydrostatic_head",
                   last.pressure.rms() / pressure_scale);
        }
        previous_centers = last.centers_by_id;
    }

    record_endpoint(prefix + "final", last);
    record(prefix + "accepted_steps", simulation.time().accepted_steps);
    record(prefix + "minimum_dt_s", minimum_dt);
    record(prefix + "elapsed_over_gravity_time", simulation.time().accepted_time / gravity_time);
    record(prefix + "peak_speed_rms_m_per_s", peak_rms_speed);
    record(prefix + "peak_speed_max_m_per_s", peak_max_speed);
    record(prefix + "peak_speed_rms_over_gravity_speed", peak_rms_speed / gravity_speed);
    record(prefix + "peak_speed_max_over_gravity_speed", peak_max_speed / gravity_speed);
    record(prefix + "peak_pressure_rms_pa", peak_rms_pressure_error);
    record(prefix + "peak_pressure_rms_over_hydrostatic_head",
           peak_rms_pressure_error / pressure_scale);
    record(prefix + "unique_particles_crossing_solid_faces",
           std::count(crossed_face.begin(), crossed_face.end(), true));
    for (std::size_t face = 0; face < face_names.size(); ++face) {
        const std::string label = prefix + face_names[face];
        record(label + "_minimum_clearance_over_h", clearance[face]);
        record(label + "_outside_particle_step_samples", outside_samples[face]);
        record(label + "_outward_plane_crossings", plane_crossings[face]);
        record(label + "_outward_solid_face_crossings", face_crossings[face]);
        if (!uncompensated) {
            EXPECT_GT(clearance[face], 0) << label;
            EXPECT_EQ(outside_samples[face], 0u) << label;
            EXPECT_EQ(face_crossings[face], 0u) << label;
        }
    }
    EXPECT_GE(simulation.time().accepted_time / gravity_time, 3);
    ASSERT_TRUE(simulation.finalize().ok());
    if (uncompensated) {
        return;
    }
    record("fluid_particles", tank.fluid.size());
    record("fixed_wall_particles", tank.wall.size());
    record("width_over_h", width);
    record("depth_over_h", depth);
    record("wall_layers", shell_layers);
    record("gravity_m_per_s2", g);
    record("reference_density_kg_per_m3", eos.reference_density);
    record("sound_speed_m_per_s", eos.sound_speed);
    record("tait_gamma", eos.gamma);
    record("maximum_dt_s", maximum_dt);
    record("hydrostatic_head_pa", pressure_scale);
    record("gravity_speed_m_per_s", gravity_speed);
    record("gravity_time_s", gravity_time);
    // Require a small pressure error relative to the full hydrostatic head and
    // parasitic motion well below the gravity-wave speed over the whole run.
    EXPECT_LT(initial.pressure.rms() / pressure_scale, 1e-4);
    EXPECT_LT(peak_rms_pressure_error / pressure_scale, .25);
    EXPECT_LT(peak_rms_speed / gravity_speed, .2);
    EXPECT_LT(peak_max_speed / gravity_speed, .5);
    for (const auto& region : last.pressure_region)
        EXPECT_GT(region.mass, 0);
    for (const auto& layer : last.pressure_layer)
        ASSERT_GT(layer.mass, 0);
    for (std::size_t k = 1; k < last.pressure_layer.size(); ++k) {
        const auto& lower = last.pressure_layer[k - 1];
        const auto& upper = last.pressure_layer[k];
        EXPECT_GT(lower.pressure_sum / lower.mass, upper.pressure_sum / upper.mass);
    }
}

TEST(SphHydrostaticTank, OpenFiveFaceTankUnderGravity) {
    // The control uses continuum density times the one-cubic-metre lattice
    // volume, including its free-surface particles.
    auto tank = make_tank();
    ASSERT_NO_FATAL_FAILURE(run_tank(tank, true));
    fit_fluid_mass(tank);
    ASSERT_NO_FATAL_FAILURE(run_tank(tank, false));
}
} // namespace
