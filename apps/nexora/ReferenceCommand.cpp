#include "ReferenceCommand.hpp"

#include "BuildInfo.hpp"
#include "ReferenceCase.hpp"

#include <Kokkos_Core.hpp>
#include <nexora/core/Real.hpp>
#include <nexora/core/Status.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <participants/reference/Participant.hpp>

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nexora::cli {
namespace {

void print_plan(const ReferenceCase& input) {
    static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>);
    static_assert(std::is_same_v<reference::Values::array_layout, Kokkos::LayoutRight>);
    static_assert(std::is_same_v<reference::Values::memory_space, Kokkos::HostSpace>);
    static_assert(reference::Values::rank() == 1);
    constexpr auto dtype = std::is_same_v<real, double> ? "float64" : "float32";

    // The compiled Serial profile uses the process host.
    std::cout << "execution=serial\n"
              << "built=true\n"
              << "available=true\n"
              << "validated=" << (build_info::reference_validated ? "true" : "false") << '\n'
              << "validation_evidence=" << build_info::reference_validation_evidence << '\n'
              << "participant=reference\n"
              << "entities=" << input.config.ids.size() << '\n'
              << std::setprecision(std::numeric_limits<double>::max_digits10)
              << "start=" << input.start << '\n'
              << "end=" << input.end << '\n'
              << "steps=" << input.steps.size() << '\n'
              << "first_step=" << input.steps.front() << '\n'
              << "last_step=" << input.steps.back() << '\n'
              << "memory_space=HostSpace\n"
              << "field.x.dtype=" << dtype << '\n'
              << "field.x.layout=LayoutRight\n"
              << "field.q.dtype=" << dtype << '\n'
              << "field.q.layout=LayoutRight\n"
              << "field.stable_ids.dtype=uint64\n"
              << "field.stable_ids.layout=LayoutRight\n"
              << "field.history.dtype=uint64\n"
              << "field.history.layout=LayoutRight\n"
              << "layout_conversion=none\n"
              << "permissions.granularity=whole allocation\n"
              << "permissions.accepted=read-only leases\n"
              << "permissions.step=exclusive x and q overwrite and history accumulate leases\n"
              << "permissions.publish=register completed fields for read-only publication\n"
              << "copy.initialize=host input to stable_ids and x and q plus history seed\n"
              << "copy.step=accepted history to independent next history\n"
              << "copy.step_x_q=none\n"
              << "copy.summary=none\n"
              << "synchronize.initialize=Serial fence before publishing initial state\n"
              << "synchronize.step=Serial fence before host diagnostics and publication\n"
              << "synchronize.summary=host reads the completed accepted snapshot\n"
              << "synchronize.finalize=Serial fence before releasing participant state\n";
}

Status run_and_summarize(Simulation& simulation, const reference::Participant& participant,
                         const std::vector<double>& steps, double end, std::ostream& out) {
    if (auto status = simulation.initialize(); !status.ok()) {
        return status;
    }
    if (const auto status = simulation.run(steps); !status.ok()) {
        return {status.code(), "step " + std::to_string(simulation.time().accepted_steps + 1) +
                                   ": " + status.message()};
    }

    const auto& time = simulation.time();
    if (time.accepted_time != end) {
        return {StatusCode::execution_failure, "simulation did not reach the planned end time"};
    }
    const auto snapshot = participant.snapshot();
    if (!snapshot.ok()) {
        return snapshot.status();
    }
    const auto& accepted = snapshot.value();
    const auto entities = accepted.ids.extent(0);
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "accepted_time=" << time.accepted_time << '\n'
        << "state_version=" << accepted.version.value() << '\n'
        << "accepted_steps=" << time.accepted_steps << '\n'
        << "entities=" << entities << '\n'
        << std::setprecision(std::numeric_limits<real>::max_digits10);
    for (std::size_t entity = 0; entity < entities; ++entity) {
        out << "entity." << entity << ".id=" << accepted.ids(entity) << '\n'
            << "entity." << entity << ".x=" << accepted.x(entity) << '\n'
            << "entity." << entity << ".q=" << accepted.q(entity) << '\n';
    }
    return {};
}

int run_reference_case(ReferenceCase input) {
    const Kokkos::ScopeGuard runtime;
    reference::Participant participant{std::move(input.config)};
    Simulation simulation{participant, input.start};
    std::ostringstream summary;
    summary.exceptions(std::ios::badbit | std::ios::failbit);

    Status run_status;
    try {
        // The snapshot and its leases are destroyed before finalizing Simulation.
        run_status = run_and_summarize(simulation, participant, input.steps, input.end, summary);
    } catch (...) {
        if (const auto status = simulation.finalize(); !status.ok()) {
            std::cerr << "reference finalize failed: " << status.message() << '\n';
        }
        throw;
    }
    const auto finalized = simulation.finalize();
    if (!run_status.ok()) {
        std::cerr << "reference run failed: " << run_status.message() << '\n';
    }
    if (!finalized.ok()) {
        std::cerr << "reference finalize failed: " << finalized.message() << '\n';
    }
    if (!run_status.ok() || !finalized.ok()) {
        return 1;
    }
    std::cout << std::move(summary).str();
    return std::cout ? 0 : 1;
}

} // namespace

int reference_command(ReferenceCommand command, const char* path) try {
    if (command == ReferenceCommand::run_fixture && !NEXORA_REFERENCE_TEST_FIXTURE) {
        std::cerr << "controlled reference fixture execution requires a build with BUILD_TESTING enabled\n";
        return 2;
    }
    auto input = load_reference_case(path);
    if (!input.ok()) {
        std::cerr << input.status().message() << '\n';
        return 2;
    }
    switch (command) {
    case ReferenceCommand::validate:
        std::cout << "valid=true\n";
        return std::cout ? 0 : 1;
    case ReferenceCommand::plan:
        print_plan(input.value());
        return std::cout ? 0 : 1;
    case ReferenceCommand::run:
        if (!build_info::reference_validated) {
            std::cerr << "reference Serial profile requires validation evidence for this build\n";
            return 2;
        }
        return run_reference_case(std::move(input).value());
    case ReferenceCommand::run_fixture:
        std::cerr << "running controlled reference fixture\n";
        return run_reference_case(std::move(input).value());
    }
    return 2;
} catch (const std::exception& error) {
    std::cerr << path << ": " << error.what() << '\n';
    return 1;
}

} // namespace nexora::cli
