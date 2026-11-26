#pragma once

#include <array>
#include <cstdint>
#include <nexora/core/Types.hpp>
#include <vector>

namespace nexora::sph_fixture {

// Transcribed from validation/baselines/sph-density-1d-v1.json.
inline constexpr std::array<std::uint64_t, 5> row_ids{10, 30, 20, 40, 50};
inline constexpr std::array<std::uint64_t, 6> column_ids{10, 30, 20, 40, 50, 60};
inline constexpr std::array<real, 6> positions{0, .5, 1.5, 2, 5, 2.5};
inline constexpr std::array masses{1, 2, 3, 4, 5, 6};
inline constexpr std::array<std::uint8_t, 6> ghost{0, 0, 0, 0, 0, 1};
inline constexpr std::array<std::uint8_t, 6> boundary{0, 0, 0, 1, 0, 0};
inline constexpr std::array<std::uint64_t, 6> offsets{0, 3, 7, 11, 15, 15};
inline constexpr std::array<std::uint32_t, 15> indices{1, 2, 3, 0, 2, 3, 5, 0, 1, 3, 5, 0, 1, 2, 5};

inline constexpr std::array<std::uint64_t, 5> reordered_rows{40, 50, 10, 20, 30};
inline constexpr std::array<std::uint64_t, 6> reordered_columns{60, 40, 50, 10, 20, 30};
inline constexpr std::array<std::uint64_t, 6> reordered_offsets{0, 4, 4, 7, 11, 15};
inline constexpr std::array<std::uint32_t, 15> reordered_indices{3, 5, 4, 0, 5, 4, 1, 3, 5, 1, 0, 3, 4, 1, 0};

inline constexpr std::array density_numerators{81, 115, 253, 337, 160};
inline constexpr int sum_numerator = 946;
inline constexpr int denominator = 48;
inline constexpr int density_epsilon = 8;
inline constexpr int sum_epsilon = 16;

template <class Id, class Range>
std::vector<Id> strong_ids(const Range& ids) {
    return std::vector<Id>(ids.begin(), ids.end());
}

template <class Config>
Config frozen_config() {
    Config config;
    using Id = typename decltype(config.row_ids)::value_type;
    config.row_ids = strong_ids<Id>(row_ids);
    config.column_ids = strong_ids<Id>(column_ids);
    config.positions.assign(positions.begin(), positions.end());
    config.masses.assign(masses.begin(), masses.end());
    config.ghost.assign(ghost.begin(), ghost.end());
    config.boundary.assign(boundary.begin(), boundary.end());
    config.offsets.assign(offsets.begin(), offsets.end());
    config.indices.assign(indices.begin(), indices.end());
    return config;
}

} // namespace nexora::sph_fixture
