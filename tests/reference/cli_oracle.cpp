#include <Kokkos_Core.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <participants/reference/Participant.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace nexora;
using namespace nexora::reference;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void run_oracle() {
    // This fixture schedule is intentionally independent of the parsed CLI case.
    const Config config{{8, 13}, {real{1.25}, real{-2}}, {real{-0.5}, real{4}}, real{0.7}};
    constexpr double start = 0.1;
    constexpr std::array<double, 3> steps{0.3, 0.3, 0.25};
    Participant participant(config);
    Simulation simulation(participant, start);
    require(simulation.initialize().ok(), "oracle initialize failed");
    require(simulation.run(steps).ok(), "oracle run failed");

    const auto state_result = participant.snapshot();
    require(state_result.ok(), "oracle snapshot failed");
    const auto& state = state_result.value();
    const auto ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state.ids);
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state.x);
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state.q);

    // Independent high-precision schedule result: .79 * .79 * .825 = .5148825.
    // The float build permits its expected rounding while the double build stays tight.
    constexpr double expected_x0 = 0.643603125;
    constexpr double expected_x1 = -1.029765;
    constexpr double expected_q0 = 0.106396875;
    constexpr double expected_q1 = 3.029765;
    constexpr double tolerance = sizeof(real) == sizeof(float) ? 2.0e-6 : 2.0e-14;
    require(std::abs(static_cast<double>(x(0)) - expected_x0) <= tolerance,
            "oracle x[0] differs from fixed expected value");
    require(std::abs(static_cast<double>(x(1)) - expected_x1) <= tolerance,
            "oracle x[1] differs from fixed expected value");
    require(std::abs(static_cast<double>(q(0)) - expected_q0) <= tolerance,
            "oracle q[0] differs from fixed expected value");
    require(std::abs(static_cast<double>(q(1)) - expected_q1) <= tolerance,
            "oracle q[1] differs from fixed expected value");
    require(state.time == 0.95 && state.version.value() == 3 &&
                simulation.time().accepted_steps == steps.size(),
            "oracle time/version/step count differs from fixed schedule");

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
              << "accepted_time=" << state.time << '\n'
              << "state_version=" << state.version.value() << '\n'
              << "accepted_steps=" << simulation.time().accepted_steps << '\n'
              << "entities=" << ids.extent(0) << '\n'
              << std::setprecision(std::numeric_limits<real>::max_digits10);
    for (std::size_t i = 0; i < ids.extent(0); ++i) {
        std::cout << "entity." << i << ".id=" << ids(i) << '\n'
                  << "entity." << i << ".x=" << x(i) << '\n'
                  << "entity." << i << ".q=" << q(i) << '\n';
    }
    require(simulation.finalize().ok(), "oracle finalize failed");
}

} // namespace

int main() {
    try {
        Kokkos::ScopeGuard runtime;
        run_oracle();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
