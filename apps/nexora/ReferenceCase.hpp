#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <participants/reference/Participant.hpp>
#include <vector>

namespace nexora::cli {
struct ReferenceCase {
    std::unique_ptr<reference::Participant> participant;
    std::size_t entities = 0;
    double start = 0.0;
    double end = 0.0;
    // Authoritative requested durations, including a separately validated tail.
    // Replaying these additions reaches end without recovering dt by subtraction.
    std::vector<double> steps;
};

// Metadata-only preflight: no Kokkos initialization or numerical allocations.
Result<ReferenceCase> load_reference_case(const std::filesystem::path& path);
} // namespace nexora::cli
