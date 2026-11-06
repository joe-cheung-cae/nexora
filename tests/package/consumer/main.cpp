#include <nexora/core/Version.hpp>
#include <nexora/exec/Buffer.hpp>
#include <nexora/exec/ExecContext.hpp>
#include <nexora/simulation/Simulation.hpp>

#include <cstddef>

int main() {
    nexora::Buffer buffer{1};
    const auto done = nexora::CpuContext{}.submit([&buffer] {
        buffer.view()[0] = std::byte{42};
        return nexora::Status{};
    });
    return nexora::version[0] != '\0' && done.wait().ok() && buffer.view()[0] == std::byte{42} ? 0
                                                                                               : 1;
}
