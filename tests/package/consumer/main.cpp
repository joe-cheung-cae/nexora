#include <nexora/core/Version.hpp>
#include <nexora/exec/Buffer.hpp>
#include <nexora/exec/ExecContext.hpp>
#include <nexora/participant/TimeConstraint.hpp>
#include <nexora/simulation/Simulation.hpp>

#include <cstddef>

bool check_installed_time_interval();

int main() {
    nexora::Buffer buffer{1};
    const auto done = nexora::CpuContext{}.submit([&buffer] {
        buffer.view()[0] = std::byte{42};
        return nexora::Status{};
    });
    const nexora::TimeInterval interval{0.0, 0.5, 0.5};
    const bool valid = nexora::version[0] != '\0' && done.wait().ok() &&
                       buffer.view()[0] == std::byte{42} && interval.duration() == 0.5 &&
                       check_installed_time_interval();
    return valid ? 0 : 1;
}
