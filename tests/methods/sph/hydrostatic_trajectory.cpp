#include "hydrostatic_tank_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nexora/simulation/Simulation.hpp>
#include <participants/sph/WcsphParticipant3D.hpp>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace {
using namespace nexora;
using namespace sph_test;
using namespace sph_test::hydrostatic;
using Center = std::array<real, 3>;
constexpr double length = tank_size, g = gravity;
constexpr std::array frame_times{0., .38671875, 1.5625, 2.34765625};
constexpr std::array face_names{"x_min", "x_max", "y_min", "y_max", "bottom"};
constexpr auto steps_csv = "steps.csv", frames_csv = "frames.csv";
constexpr auto profiles_csv = "profiles.csv", crossings_csv = "crossings.csv";

class OutputClaim {
public:
    explicit OutputClaim(const std::filesystem::path& path)
        : marker_(path / ".nexora-tank-trajectory") {
        std::filesystem::create_directories(path);
        if (!std::filesystem::create_directory(marker_))
            throw std::runtime_error("tank trajectory output is already reserved in " + path.string());
    }

    ~OutputClaim() {
        std::error_code ignored;
        std::filesystem::remove(marker_, ignored);
    }

    OutputClaim(const OutputClaim&) = delete;
    OutputClaim& operator=(const OutputClaim&) = delete;

private:
    std::filesystem::path marker_;
};

std::array<double, 5> distances(const Center& x) {
    return {x[0], length - x[0], x[1], length - x[1], x[2]};
}

std::ofstream csv(const std::filesystem::path& path, const char* header) {
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path);
    file << header << '\n' << std::setprecision(17);
    return file;
}

void require(Status status) {
    if (!status.ok())
        throw std::runtime_error(status.message());
}

struct Layer {
    std::size_t count{};
    double mass{}, pressure_sum{}, reference_sum{}, net_acceleration_sum{};
};

// Stable fluid IDs track a center across sorted cell rows and multiple crossings.
struct Crossings {
    explicit Crossings(std::size_t count) : previous(count), crossed(count) {}
    std::vector<Center> previous;
    std::vector<unsigned> crossed;
    std::array<std::size_t, 5> unique{};
    std::size_t unique_any{};
    void observe(int n, std::uint64_t step, double time, std::uint64_t id, const Center& x,
                 const std::array<double, 5>& d, std::ostream& events) {
        auto& last = previous[id - 1];
        auto& mask = crossed[id - 1];
        if (step != 0) {
            const auto old = distances(last);
            for (std::size_t f = 0; f < d.size(); ++f) {
                if (!(old[f] > 0 && d[f] <= 0))
                    continue;
                const double fraction = old[f] / (old[f] - d[f]);
                std::array<double, 3> hit{};
                bool on_face = true;
                for (std::size_t k = 0; k < 3; ++k) {
                    hit[k] = std::lerp(double(last[k]), double(x[k]), fraction);
                    if (k != f / 2 && (hit[k] < 0 || hit[k] > length))
                        on_face = false;
                }
                if (on_face) {
                    events << n << ',' << step << ',' << time << ',' << id << ',' << face_names[f]
                           << ',' << fraction << ',' << hit[0] << ',' << hit[1] << ',' << hit[2]
                           << '\n';
                    unique_any += mask == 0;
                    unique[f] += (mask & (1U << f)) == 0;
                    mask |= 1U << f;
                }
            }
        }
        last = x;
    }
};

void run(int n, const std::filesystem::path& path) {
    // Hold the claim from the preflight through closing all CSV streams.
    const OutputClaim claim(path);
    for (const auto* name : {steps_csv, frames_csv, profiles_csv, crossings_csv}) {
        const auto output = path / name;
        if (std::filesystem::symlink_status(output).type() != std::filesystem::file_type::not_found)
            throw std::runtime_error("tank trajectory output already exists at " + output.string());
    }

    const real h = real(length) / real(n);
    const auto tank = make_tank(n, h, std::max(n * n * n + 1, 10000));
    const Fields fluid(tank.fluid), wall(tank.wall);
    const sph::Velocity3D rest("trajectory.rest", tank.fluid.size());
    const sph::WcsphVerletParameters parameters{eos, real(.5), {0, 0, real(-g)}, h};
    const sph::FixedWallParticles3D fixed{wall.position, wall.mass, wall.ids};
    const sph::Values initial_density("trajectory.density", tank.fluid.size());
    for (std::size_t row = 0; row < tank.fluid.size(); ++row)
        initial_density(row) = hydrostatic_density(tank.fluid[row].position[2]);
    // Escaped particles enlarge the occupied bounding box of the dense grid.
    // The admission limits bound its storage independently of physical inputs.
    constexpr std::size_t max_cells = 50000000;
    constexpr sph::WcsphParticipant3D::ResourceLimits limits{64, 1024ULL * 1024 * 1024};
    sph::WcsphParticipant3D participant(fluid.get(), max_cells, parameters, rest, initial_density,
                                        fixed, ParticipantId{1}, limits);
    Simulation sim(participant);
    const auto fail = [&](std::string context, const Status& status) {
        context += status.message();
        if (const auto finalized = sim.finalize(); !finalized.ok())
            context += " (finalization failed with " + finalized.message() + ")";
        throw std::runtime_error(context);
    };
    if (const auto status = sim.initialize(); !status.ok())
        fail("initialize tank: ", status);

    auto steps =
        csv(path / steps_csv, "n,step,time,dt,cfl_bound,pressure_rms,speed_rms,speed_max,in_tank,"
                              "unique_any,x_min,x_max,y_min,y_max,bottom,controlled_bytes,net_acceleration_rms");
    auto frames =
        csv(path / frames_csv, "n,step,time,id,x,y,z,density,pressure,speed,crossed,mass,ap_x,ap_y,ap_z");
    auto profiles = csv(path / profiles_csv,
                        "n,step,time,bin,count,mass,mean_pressure,mean_reference_pressure,net_acceleration_rms");
    auto events = csv(path / crossings_csv, "n,step,time,id,face,fraction,x,y,z");
    Crossings crossings(tank.fluid.size());
    std::size_t frame = 0;
    double dt = 0, bound = participant.time_constraint().maximum_step;
    while (true) {
        const double time = sim.time().accepted_time;
        const auto step = sim.time().accepted_steps;
        const bool save = time == frame_times.at(frame);
        double mass_sum = 0, error_sum = 0, speed_sum = 0, max_speed = 0, net_acceleration_sum = 0;
        std::size_t inside = 0;
        std::array<Layer, 6> layers{};
        std::size_t row_count = 0;
        require(participant.visit_committed_rows([&](sph::CommittedRow3D row) {
            ++row_count;
            const auto id = row.id;
            const real mass = row.mass;
            if (mass != tank.fluid.at(id - 1).mass)
                throw std::runtime_error("fluid mass changed at step " + std::to_string(step) +
                                         " for ID " + std::to_string(id));
            const auto& x = row.position;
            const auto& v = row.velocity;
            const double rho = row.density, p = row.pressure;
            const auto& ap = row.pressure_acceleration;
            const double acceleration = std::hypot(ap[0], ap[1], double(ap[2]) - g);
            const double speed = std::hypot(v[0], v[1], v[2]);
            const double rho0 = eos.reference_density, sound_speed = eos.sound_speed, gamma = eos.gamma;
            const double reference = rho0 * sound_speed * sound_speed / gamma *
                                     std::expm1(gamma / (gamma - 1) *
                                                std::log1p(std::max(0., depth_term(x[2]))));
            if (!std::isfinite(rho) || !std::isfinite(p) || !std::isfinite(speed) ||
                !std::isfinite(acceleration) ||
                !std::all_of(x.begin(), x.end(), [](double c) { return std::isfinite(c); }))
                throw std::runtime_error("nonfinite particle field");
            mass_sum += mass;
            error_sum += mass * (p - reference) * (p - reference);
            speed_sum += mass * speed * speed;
            const double acceleration_moment = mass * acceleration * acceleration;
            net_acceleration_sum += acceleration_moment;
            max_speed = std::max(max_speed, speed);
            const auto d = distances(x);
            crossings.observe(n, step, time, id, x, d, events);
            if (x[2] <= length && *std::min_element(d.begin(), d.end()) >= 0) {
                ++inside;
                auto& bin = layers[std::min(5, static_cast<int>(x[2]))];
                ++bin.count;
                bin.mass += mass;
                bin.pressure_sum += mass * p;
                bin.reference_sum += mass * reference;
                bin.net_acceleration_sum += acceleration_moment;
            }
            if (save)
                frames << n << ',' << step << ',' << time << ',' << id << ',' << x[0] << ','
                       << x[1] << ',' << x[2] << ',' << rho << ',' << p << ',' << speed << ','
                       << (crossings.crossed[id - 1] != 0) << ',' << mass << ',' << ap[0] << ','
                       << ap[1] << ',' << ap[2] << '\n';
        }));
        if (row_count != tank.fluid.size())
            throw std::runtime_error("missing fluid rows");
        steps << n << ',' << step << ',' << time << ',' << dt << ',' << bound << ','
              << std::sqrt(error_sum / mass_sum) << ',' << std::sqrt(speed_sum / mass_sum) << ','
              << max_speed << ',' << inside << ',' << crossings.unique_any;
        for (auto count : crossings.unique)
            steps << ',' << count;
        steps << ',' << participant.resource_usage().controlled_bytes << ','
              << std::sqrt(net_acceleration_sum / mass_sum) << std::endl;
        if (save) {
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const auto& bin = layers[i];
                profiles << n << ',' << step << ',' << time << ',' << i << ',' << bin.count << ','
                         << bin.mass << ',';
                if (bin.count)
                    profiles << bin.pressure_sum / bin.mass;
                profiles << ',';
                if (bin.count)
                    profiles << bin.reference_sum / bin.mass;
                profiles << ',';
                if (bin.count)
                    profiles << std::sqrt(bin.net_acceleration_sum / bin.mass);
                profiles << '\n';
            }
            profiles.flush();
            frames.flush();
            events.flush();
            if (++frame == frame_times.size())
                break;
        }
        bound = participant.time_constraint().maximum_step;
        dt = std::min({h / 256., .9 * bound, frame_times[frame] - time});
        if (const auto status = sim.run(dt); !status.ok())
            fail("run tank at t=" + std::to_string(time) + ": ", status);
    }
    require(sim.finalize());
    steps.close();
    profiles.close();
    frames.close();
    events.close();
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: nexora_sph_tank_trajectory {6|12|24} output_case_directory\n";
        return 2;
    }
    Kokkos::ScopeGuard kokkos;
    try {
        const std::string_view argument(argv[1]);
        if (argument != "6" && argument != "12" && argument != "24")
            throw std::runtime_error("unsupported tank resolution");
        run(std::stoi(argv[1]), argv[2]);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
