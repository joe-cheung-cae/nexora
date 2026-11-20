#include "cell_grid_oracle.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <methods/sph/Density3D.hpp>
#include <methods/sph/neighbors/CellGrid.hpp>
#include <participants/sph/Geometry3D.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace nexora;
using namespace sph_test;
void compare(const sph::CellGrid& grid, const std::vector<Particle>& input,
             const Kokkos::Serial& exec) {
    const auto oracle = explicit_csr(input);
    const auto packed = grid.fields();
    const auto owned = grid.owned_slots();
    const auto forward = grid.sorted_to_input();
    const auto inverse = grid.input_to_sorted();
    ASSERT_EQ(forward.extent(0), input.size());
    ASSERT_EQ(inverse.extent(0), input.size());
    ASSERT_EQ(owned.extent(0), oracle.density.size());
    const auto begins = grid.cell_begin();
    ASSERT_GE(begins.extent(0), 1U);
    EXPECT_EQ(begins(0), 0U);
    EXPECT_EQ(begins(begins.extent(0) - 1), input.size());
    for (std::size_t c = 1; c < begins.extent(0); ++c)
        EXPECT_LE(begins(c - 1), begins(c));
    for (std::size_t slot = 0; slot < input.size(); ++slot) {
        ASSERT_LT(static_cast<std::size_t>(forward(slot)), input.size());
        EXPECT_EQ(static_cast<std::size_t>(inverse(forward(slot))), slot);
        const auto& source = input[forward(slot)];
        EXPECT_EQ(packed.ids(slot), source.id);
        EXPECT_EQ(packed.mass(slot), source.mass);
        EXPECT_EQ(packed.ghost(slot), source.ghost);
        for (int k = 0; k < 3; ++k)
            EXPECT_EQ(packed.position(slot, k), source.position[k]);
    }
    const auto density = checked(sph::evaluate_density_3d(exec, grid));
    ASSERT_EQ(density.extent(0), owned.extent(0));
    std::vector<std::uint64_t> seen_owned;
    for (std::size_t row = 0; row < owned.extent(0); ++row) {
        const auto slot = owned(row);
        ASSERT_LT(static_cast<std::size_t>(slot), input.size());
        ASSERT_EQ(packed.ghost(slot), 0);
        const auto id = packed.ids(slot);
        seen_owned.push_back(id);
        std::vector<std::uint64_t> neighbors;
        grid.for_each_neighbor(row, [&](LocalIndex other, double r2) {
            ASSERT_LT(static_cast<std::size_t>(other), input.size());
            EXPECT_NE(packed.ids(other), id);
            EXPECT_GE(r2, 0);
            EXPECT_LE(r2, 4);
            neighbors.push_back(packed.ids(other));
        });
        std::sort(neighbors.begin(), neighbors.end());
        EXPECT_EQ(std::adjacent_find(neighbors.begin(), neighbors.end()), neighbors.end());
        ASSERT_TRUE(oracle.row_by_id.contains(id));
        const auto oracle_row = oracle.row_by_id.at(id);
        const std::vector<std::uint64_t> expected_neighbors(
            oracle.ids.begin() + oracle.offsets[oracle_row],
            oracle.ids.begin() + oracle.offsets[oracle_row + 1]);
        EXPECT_EQ(neighbors, expected_neighbors);
        const auto expected = oracle.density.at(id);
        ASSERT_TRUE(std::isfinite(density(row)));
        // Kernel and accumulation are double; only the final conversion uses real.
        // The real epsilon covers final rounding, while the degree-scaled double
        // term bounds positive summation and independent kernel evaluation error.
        const long double relative_bound = std::numeric_limits<real>::epsilon() +
                                           32 * static_cast<long double>(neighbors.size() + 1) *
                                               std::numeric_limits<double>::epsilon();
        EXPECT_LE(std::abs(static_cast<long double>(density(row)) - expected),
                  relative_bound * expected)
            << "owned ID " << id;
    }
    std::sort(seen_owned.begin(), seen_owned.end());
    EXPECT_EQ(std::adjacent_find(seen_owned.begin(), seen_owned.end()), seen_owned.end());
}
TEST(SphCellGrid, ThreeDimensionalShuffledStableIdsAndGhosts) {
    std::vector<Particle> particles;
    std::mt19937 random(32091);
    std::uniform_int_distribution<int> coordinate(-16, 16);
    for (std::size_t i = 0; i < 137; ++i)
        particles.push_back({{real(coordinate(random)) / 4, real(coordinate(random)) / 4,
                              real(coordinate(random)) / 4},
                             real(1 + i % 9) / 4,
                             (std::uint64_t{1} << 60) + 19 * i,
                             static_cast<std::uint8_t>(i % 7 == 0)});
    std::shuffle(particles.begin(), particles.end(), random);
    Fields fields(particles);
    Kokkos::Serial exec;
    auto grid = checked(sph::CellGrid::build(exec, fields.get(), 100000));
    ASSERT_NO_FATAL_FAILURE(compare(grid, particles, exec));
}
TEST(SphCellGrid, InclusiveSupportBoundaryAndSelfExclusion) {
    const real outside = std::nextafter(real{2}, std::numeric_limits<real>::infinity());
    const real inside = std::nextafter(real{2}, real{0});
    std::vector<Particle> particles{{{0, 0, 0}, 1, 91, 0},
                                    {{2, 0, 0}, 2, 72, 1},
                                    {{outside, 0, 0}, 3, 53, 1},
                                    {{inside, 0, 0}, 4, 34, 1},
                                    {{0, -2, 0}, 1, 15, 1}};
    Fields fields(particles);
    Kokkos::Serial exec;
    auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1000));
    ASSERT_NO_FATAL_FAILURE(compare(grid, particles, exec));
}
TEST(SphCellGrid, EmptySentinelsAndGhostOnly) {
    Kokkos::Serial exec;
    for (const auto& particles :
         {std::vector<Particle>{}, std::vector<Particle>{{{-1, 2, -3}, 1, 80, 1}}}) {
        Fields fields(particles);
        auto grid = checked(sph::CellGrid::build(exec, fields.get(), 100));
        ASSERT_NO_FATAL_FAILURE(compare(grid, particles, exec));
        EXPECT_EQ(grid.owned_slots().extent(0), 0U);
    }
}
TEST(SphCellGrid, RebuildMovesWithinAndAcrossCellsAndPreservesOldIndex) {
    std::vector<Particle> original{
        {{-1.75, 0, 0}, 1, 5, 0}, {{-.25, .5, 0}, 2, 6, 0}, {{2.25, .5, -1}, 3, 7, 1}};
    Fields fields(original);
    Kokkos::Serial exec;
    const auto old = checked(sph::CellGrid::build(exec, fields.get(), 1000));
    auto moved = original;
    moved[0].position[1] += real{.125};
    moved[1].position[0] += real{4};
    moved[0].mass = real{7};
    fields.mass(0) = moved[0].mass;
    for (std::size_t i = 0; i < moved.size(); ++i)
        for (int k = 0; k < 3; ++k)
            fields.position(i, k) = moved[i].position[k];
    const auto rebuilt = checked(sph::CellGrid::build(exec, fields.get(), 1000));
    ASSERT_NO_FATAL_FAILURE(compare(rebuilt, moved, exec));
    ASSERT_NO_FATAL_FAILURE(compare(old, original, exec));
}
TEST(SphCellGrid, RejectsDuplicateIdsBadExtentsAndCellBudget) {
    Kokkos::Serial exec;
    Fields fields({{{0, 0, 0}, 1, 40, 0}, {{100, 1, 1}, 1, 41, 0}});
    fields.ids(1) = fields.ids(0);
    EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 1000).status().code(),
              StatusCode::invalid_argument);
    fields.ids(1) = 41;
    auto bad = fields.get();
    bad.mass = Array<real>("bad.mass", 1);
    EXPECT_EQ(sph::CellGrid::build(exec, bad, 1000).status().code(), StatusCode::invalid_argument);
    for (const auto budget : {0u, 1u})
        EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), budget).status().code(),
                  StatusCode::resource_exhausted);
}
TEST(SphCellGrid, HighDegreeHasNoTruncation) {
    constexpr std::size_t degree = 4096;
    std::vector<Particle> particles{{{-.5, .25, -.75}, 1, 9000, 0}};
    for (std::size_t i = 0; i < degree; ++i)
        particles.push_back({{real(i % 7) / 16, real(i % 11) / 32, real(i % 13) / 64},
                             real(1 + i % 3),
                             10000 + i,
                             1});
    Fields fields(particles);
    Kokkos::Serial exec;
    auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1000));
    ASSERT_NO_FATAL_FAILURE(compare(grid, particles, exec));
}
TEST(SphCellGrid, DistinctCoincidentIdsUseAll64BitsAndRetainSelfOnce) {
    // Every ID has identical low 32 bits and every particle occupies the same cell.
    const std::uint64_t low = 23;
    std::vector<Particle> particles{{{-.25, -.25, -.25}, 1, (9ULL << 32) | low, 0},
                                    {{-.25, -.25, -.25}, 2, (1ULL << 32) | low, 1},
                                    {{-.25, -.25, -.25}, 3, (5ULL << 32) | low, 0}};
    Fields fields(particles);
    Kokkos::Serial exec;
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 100));
    ASSERT_NO_FATAL_FAILURE(compare(grid, particles, exec));
    EXPECT_EQ(grid.fields().ids(0), particles[1].id);
    EXPECT_EQ(grid.fields().ids(1), particles[2].id);
    EXPECT_EQ(grid.fields().ids(2), particles[0].id);
}
TEST(SphCellGrid, SquaredDistanceContractAtDiagonal) {
    std::vector<Particle> particles{
        {{0, 0, 0}, 1, 1, 0},
        {{static_cast<real>(1.2), static_cast<real>(1.6), 0}, 1, 2, 1},
        {{std::nextafter(static_cast<real>(1.2), real{2}), static_cast<real>(1.6), 0}, 1, 3, 1},
        {{std::nextafter(static_cast<real>(1.2), real{0}), static_cast<real>(1.6), 0}, 1, 4, 1},
        {{-2, 0, 0}, 1, 6, 1}};
    Fields fields(particles);
    Kokkos::Serial exec;
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1000));
    ASSERT_NO_FATAL_FAILURE(compare(grid, particles, exec));
}
TEST(SphCellGrid, CaseSupportControlsCellsPhysicalDistanceAndUpdates) {
    Kokkos::Serial exec;
    Fields fields({{{-1.5, 0, 0}, 1, 1, 0}, {{0, 0, 0}, 1, 2, 1}, {{.25, 0, 0}, 1, 3, 1}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 2, {}, .75));
    EXPECT_EQ(grid.smoothing_length(), .75);
    EXPECT_EQ(grid.support_radius(), 1.5);
    EXPECT_EQ(grid.cell_begin().extent(0), 3U);
    std::vector<std::uint64_t> ids;
    grid.for_each_neighbor(0, [&](LocalIndex slot, double distance_squared) {
        ids.push_back(grid.fields().ids(slot));
        EXPECT_EQ(distance_squared, 2.25);
    });
    EXPECT_EQ(ids, (std::vector<std::uint64_t>{2}));
    fields.position(2, 0) = .5;
    const auto next = checked(grid.update(exec, fields.get(), 2));
    EXPECT_EQ(next.smoothing_length(), .75);
    EXPECT_EQ(grid.cell_begin().data(), next.cell_begin().data());
    fields.position(2, 0) = 1.5;
    EXPECT_FALSE(grid.update(exec, fields.get(), 2).ok());
    const auto rebuilt = checked(grid.update(exec, fields.get(), 3));
    EXPECT_EQ(rebuilt.support_radius(), 1.5);
    EXPECT_EQ(rebuilt.cell_begin().extent(0), 4U);
}

TEST(SphCellGrid, NonbinaryCellWidthUsesExactFloorOnRoundedQuotient) {
    Kokkos::Serial exec;
    Fields fields({{{0, 0, 0}, 1, 1, 0}, {{1, 0, 0}, 1, 2, 1}});
    // The double width 0.2 exceeds 1/5, although 1/width rounds to five.
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 5, {}, .1));
    EXPECT_EQ(grid.cell_begin().extent(0), 6U);
    fields.position(1, 0) = -1;
    EXPECT_FALSE(sph::CellGrid::build(exec, fields.get(), 5, {}, .1).ok());
}

TEST(SphCellGrid, ValidatesCaseSupport) {
    Kokkos::Serial exec;
    Fields fields({{{0, 0, 0}, 1, 1, 0}});
    for (const double h : {0., std::numeric_limits<double>::infinity()})
        EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 1, {}, h).status().code(),
                  StatusCode::invalid_argument)
            << h;
}

TEST(SphCellGrid, RejectsInvalidPhysicalFields) {
    Kokkos::Serial exec;
    Fields fields({{{0, 0, 0}, 1, 1, 0}});
    for (const real mass : {real{0}, std::numeric_limits<real>::infinity()}) {
        fields.mass(0) = mass;
        EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 100).status().code(),
                  StatusCode::invalid_argument)
            << mass;
    }
    fields.mass(0) = 1;
    fields.ghost(0) = 2;
    EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 100).status().code(),
              StatusCode::invalid_argument);
    fields.ghost(0) = 0;
    fields.position(0, 0) = std::numeric_limits<real>::quiet_NaN();
    EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 100).status().code(),
              StatusCode::invalid_argument);
}
TEST(SphCellGrid, ExactComponentSupportBothDirectionsAndInputOrders) {
    const real inside_two = std::nextafter(real{2}, real{0});
    const real outside_two = std::nextafter(real{2}, real{4});
    std::vector<Particle> particles{
        {{inside_two, 10, 0}, 1, 83, 0},
        {{4, 10, 0}, 1, 84, 0},
        {{0, 20, 0}, 1, 85, 0},
        {{2, 20, 0}, 1, 86, 0},
        {{0, 30, 0}, 1, 87, 0},
        {{outside_two, 30, 0}, 1, 88, 0},
        {{-2, 40, 0}, 1, 89, 0},
        {{.25, 40, 0}, 1, 90, 0}};
    EXPECT_FALSE(exact_component_support(inside_two, real{4}));
    EXPECT_FALSE(exact_component_support(real{4}, inside_two));
    EXPECT_TRUE(exact_component_support(real{0}, real{2}));
    Kokkos::Serial exec;
    Fields fields(particles);
    const auto first = checked(sph::CellGrid::build(exec, fields.get(), 1000));
    ASSERT_NO_FATAL_FAILURE(compare(first, particles, exec));
    std::reverse(particles.begin(), particles.end());
    Fields reversed(particles);
    const auto second = checked(sph::CellGrid::build(exec, reversed.get(), 1000));
    ASSERT_NO_FATAL_FAILURE(compare(second, particles, exec));
    for (std::size_t slot = 0; slot < particles.size(); ++slot)
        EXPECT_EQ(first.fields().ids(slot), second.fields().ids(slot));
}
TEST(SphCellGrid, EmptyCellsShareOffsetsBetweenOccupiedRuns) {
    // Three x cells, two y cells. In x-fastest order the occupied cells are 0, 2 and 5.
    std::vector<Particle> particles{
        {{4, 2, 0}, 1, 9, 0}, {{4, 0, 0}, 1, 7, 0}, {{0, 0, 0}, 1, 5, 1}, {{5, 0, 0}, 1, 3, 1}};
    Kokkos::Serial exec;
    Fields fields(particles);
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 6));
    const std::array<LocalIndex, 7> expected{0, 1, 1, 3, 3, 3, 4};
    ASSERT_EQ(grid.cell_begin().extent(0), expected.size());
    for (std::size_t c = 0; c < expected.size(); ++c)
        EXPECT_EQ(grid.cell_begin()(c), expected[c]);
    ASSERT_NO_FATAL_FAILURE(compare(grid, particles, exec));
}
TEST(SphCellGrid, CrossStepReuseAndRebuildMatchesExplicitCsr) {
    Kokkos::Serial exec;
    const std::vector<Particle> initial{{{.25, 0, 0}, 1, 101, 0},
                                        {{2.5, 0, 0}, 2, 202, 0},
                                        {{-.5, 0, 0}, 3, 303, 1},
                                        {{3.75, 0, 0}, 4, 404, 1}};
    Fields original(initial);
    const auto first = checked(sph::CellGrid::build(exec, original.get(), 100));
    const auto advance = [&](const sph::CellGrid& previous, const std::vector<Particle>& state) {
        Fields fields(state);
        return checked(previous.update(exec, fields.get(), 100));
    };
    auto within = initial;
    within[0].position[0] = real{1.75};
    within[1].mass = real{5};
    const auto reused = advance(first, within);
    EXPECT_EQ(first.cell_begin().data(), reused.cell_begin().data());
    EXPECT_EQ(first.owned_slots().data(), reused.owned_slots().data());
    EXPECT_EQ(first.sorted_to_input().data(), reused.sorted_to_input().data());
    EXPECT_EQ(first.input_to_sorted().data(), reused.input_to_sorted().data());
    EXPECT_NE(first.fields().position.data(), reused.fields().position.data());
    ASSERT_NO_FATAL_FAILURE(compare(reused, within, exec));
    const auto before = checked(sph::evaluate_density_3d(exec, first));
    const auto after = checked(sph::evaluate_density_3d(exec, reused));
    EXPECT_NE(before(0), after(0));
    // Both particles retain their cells, but one support edge enters and another leaves.
    EXPECT_NE(explicit_csr(initial).ids, explicit_csr(within).ids);

    auto ghost_crossing = within;
    std::swap(ghost_crossing[2].position, ghost_crossing[3].position);
    const auto ghost_rebuilt = advance(reused, ghost_crossing);
    // Histogram is unchanged: ghost identities exchange cells and still require rebuild.
    EXPECT_NE(reused.cell_begin().data(), ghost_rebuilt.cell_begin().data());
    ASSERT_NO_FATAL_FAILURE(compare(ghost_rebuilt, ghost_crossing, exec));
    auto owned_crossing = ghost_crossing;
    std::swap(owned_crossing[0].position, owned_crossing[1].position);
    const auto owned_rebuilt = advance(ghost_rebuilt, owned_crossing);
    EXPECT_NE(ghost_rebuilt.cell_begin().data(), owned_rebuilt.cell_begin().data());
    ASSERT_NO_FATAL_FAILURE(compare(owned_rebuilt, owned_crossing, exec));
    auto following = owned_crossing;
    following[0].position[0] += real{.125};
    following[1].position[0] += real{.125};
    const auto subsequent = advance(owned_rebuilt, following);
    EXPECT_EQ(owned_rebuilt.cell_begin().data(), subsequent.cell_begin().data());
    ASSERT_NO_FATAL_FAILURE(compare(subsequent, following, exec));
    // Retained old snapshots survive every reuse/rebuild and input Fields destruction.
    ASSERT_NO_FATAL_FAILURE(compare(first, initial, exec));
    ASSERT_NO_FATAL_FAILURE(compare(reused, within, exec));
    ASSERT_NO_FATAL_FAILURE(compare(ghost_rebuilt, ghost_crossing, exec));
    ASSERT_NO_FATAL_FAILURE(compare(owned_rebuilt, owned_crossing, exec));
}
TEST(SphCellGrid, UpdateReorderedInputSharesCellsWithFreshMappings) {
    Kokkos::Serial exec;
    std::vector<Particle> initial{
        {{-.5, .25, 0}, 1, 44, 0}, {{.5, .25, 0}, 2, 11, 1}, {{2.5, .25, 0}, 3, 77, 0}};
    Fields fields(initial);
    const auto first = checked(sph::CellGrid::build(exec, fields.get(), 100));
    std::reverse(initial.begin(), initial.end());
    Fields reversed(initial);
    const auto updated = checked(first.update(exec, reversed.get(), 100));
    EXPECT_EQ(first.cell_begin().data(), updated.cell_begin().data());
    EXPECT_NE(first.sorted_to_input().data(), updated.sorted_to_input().data());
    ASSERT_NO_FATAL_FAILURE(compare(updated, initial, exec));
    for (std::size_t slot = 0; slot < initial.size(); ++slot)
        EXPECT_EQ(first.fields().ids(slot), updated.fields().ids(slot));
    initial[0].mass += real{1};
    reversed.mass(0) = initial[0].mass;
    const auto next = checked(updated.update(exec, reversed.get(), 100));
    EXPECT_EQ(next.sorted_to_input().data(), updated.sorted_to_input().data());
    EXPECT_EQ(next.input_to_sorted().data(), updated.input_to_sorted().data());
    ASSERT_NO_FATAL_FAILURE(compare(next, initial, exec));
}
TEST(SphCellGrid, ShuffledThreeDimensionalTrajectoryMatchesCsrAtEveryStep) {
    // Two distinct high-bit IDs per cell, spread over a 3 x 3 x 3 box.
    std::vector<std::array<int, 3>> bins;
    for (int i = 0; i < 54; ++i)
        bins.push_back({i / 2 % 3 - 1, i / 6 % 3 - 1, i / 18 - 1});
    std::mt19937 random(2073);
    std::uniform_int_distribution<int> offset(1, 31);
    auto stage = [&] {
        std::vector<Particle> particles;
        for (std::size_t i = 0; i < bins.size(); ++i) {
            Particle p{{},
                       real(1 + offset(random)) / 8,
                       (std::uint64_t(i) << 32) | 7,
                       static_cast<std::uint8_t>(i % 3 == 0)};
            for (int axis = 0; axis < 3; ++axis)
                p.position[axis] = real(2 * bins[i][axis]) + real(offset(random)) / 16;
            particles.push_back(p);
        }
        std::shuffle(particles.begin(), particles.end(), random);
        return particles;
    };

    Kokkos::Serial exec;
    auto state = stage();
    auto grid = checked(sph::CellGrid::build(exec, Fields(state).get(), 27));
    ASSERT_NO_FATAL_FAILURE(compare(grid, state, exec));
    for (int step = 1; step <= 7; ++step) {
        SCOPED_TRACE(step);
        const bool cross_cell = step % 2 == 0;
        if (cross_cell) {
            // Exchange two IDs' x, then y, then z bins. All cell counts stay fixed.
            const int axis = step / 2 - 1;
            constexpr std::array<int, 3> first{0, 1, 3}, second{2, 7, 21};
            std::swap(bins[first[axis]][axis], bins[second[axis]][axis]);
        }
        state = stage();
        auto next = checked(grid.update(exec, Fields(state).get(), 27));
        EXPECT_EQ(grid.cell_begin().data() == next.cell_begin().data(), !cross_cell);
        ASSERT_NO_FATAL_FAILURE(compare(next, state, exec));
        grid = std::move(next);
    }
}
TEST(SphCellGrid, UpdateOwnershipAndMembershipChangesRebuild) {
    Kokkos::Serial exec;
    std::vector<Particle> initial{{{.25, 0, 0}, 1, 1, 0}, {{.5, 0, 0}, 2, 2, 1}};
    Fields fields(initial);
    const auto first = checked(sph::CellGrid::build(exec, fields.get(), 100));
    auto changed = initial;
    std::swap(changed[0].ghost, changed[1].ghost);
    Fields ownership(changed);
    const auto owner_changed = checked(first.update(exec, ownership.get(), 100));
    EXPECT_NE(first.cell_begin().data(), owner_changed.cell_begin().data());
    ASSERT_NO_FATAL_FAILURE(compare(owner_changed, changed, exec));
    changed[0].id = 900;
    Fields identity(changed);
    const auto member_changed = checked(owner_changed.update(exec, identity.get(), 100));
    EXPECT_NE(owner_changed.cell_begin().data(), member_changed.cell_begin().data());
    ASSERT_NO_FATAL_FAILURE(compare(member_changed, changed, exec));
    changed.push_back({{.75, 0, 0}, 3, 901, 0});
    Fields added(changed);
    const auto grown = checked(member_changed.update(exec, added.get(), 100));
    EXPECT_NE(member_changed.cell_begin().data(), grown.cell_begin().data());
    ASSERT_NO_FATAL_FAILURE(compare(grown, changed, exec));
    ASSERT_NO_FATAL_FAILURE(compare(first, initial, exec));
}
TEST(SphCellGrid, UpdateFailureRetryBudgetAndEmptyTransitions) {
    Kokkos::Serial exec;
    const std::vector<Particle> initial{{{.25, 0, 0}, 1, 10, 0}, {{2.25, 0, 0}, 2, 20, 1}};
    Fields fields(initial);
    const auto first = checked(sph::CellGrid::build(exec, fields.get(), 100));
    EXPECT_FALSE(first.update(exec, fields.get(), 1).ok());
    fields.mass(0) = -1;
    EXPECT_FALSE(first.update(exec, fields.get(), 100).ok());
    fields.mass(0) = 1;
    fields.ids(1) = fields.ids(0);
    EXPECT_FALSE(first.update(exec, fields.get(), 100).ok());
    fields.ids(1) = 20;
    const auto retried = checked(first.update(exec, fields.get(), 100));
    EXPECT_EQ(first.cell_begin().data(), retried.cell_begin().data());
    ASSERT_NO_FATAL_FAILURE(compare(retried, initial, exec));
    ASSERT_NO_FATAL_FAILURE(compare(first, initial, exec));
    Fields empty({});
    const auto cleared = checked(retried.update(exec, empty.get(), 0));
    ASSERT_NO_FATAL_FAILURE(compare(cleared, {}, exec));
    const auto still_empty = checked(cleared.update(exec, empty.get(), 0));
    ASSERT_NO_FATAL_FAILURE(compare(still_empty, {}, exec));
    const auto repopulated = checked(still_empty.update(exec, fields.get(), 100));
    ASSERT_NO_FATAL_FAILURE(compare(repopulated, initial, exec));
}
std::vector<Particle> initial_particles() {
    return {{{.25, -.25, .5}, 1, (9ULL << 32) | 17, 0},
            {{2.5, -.25, .5}, 2, (1ULL << 32) | 17, 0},
            {{-.5, -.25, .5}, 3, (5ULL << 32) | 17, 1},
            {{3.75, -.25, .5}, 4, (3ULL << 32) | 17, 1}};
}
TEST(SphGeometry3D, UpdatesOwnGeometryAndRetainPreviousStates) {
    Kokkos::Serial exec;
    detail::ResourceBudget budget;
    const auto initial = initial_particles();
    Fields input(initial);
    const auto first = checked(sph::prepare_geometry_3d(exec, input.get(), 1000, budget, 1));
    auto same_cell = initial;
    same_cell[0].position[0] = real{1.75};
    same_cell[1].mass = 5;
    Fields fields(same_cell);
    const auto next = checked(sph::prepare_geometry_3d(exec, fields.get(), 1000, budget, 1,
                                                       first.get()));
    // Geometry owns its values; caller reuse cannot change any retained result.
    fields.position(0, 0) = 100;
    fields.mass(1) = 99;
    ASSERT_NO_FATAL_FAILURE(compare(next->grid, same_cell, exec));
    auto crossing = same_cell;
    std::swap(crossing[0].position, crossing[1].position);
    std::swap(crossing[2].position, crossing[3].position);
    std::reverse(crossing.begin(), crossing.end());
    const auto latest = checked(sph::prepare_geometry_3d(exec, Fields(crossing).get(), 1000,
                                                         budget, 1, next.get()));
    EXPECT_NE(next->grid.cell_begin().data(), latest->grid.cell_begin().data());
    ASSERT_NO_FATAL_FAILURE(compare(latest->grid, crossing, exec));
    ASSERT_NO_FATAL_FAILURE(compare(first->grid, initial, exec));
    ASSERT_NO_FATAL_FAILURE(compare(next->grid, same_cell, exec));
}
TEST(SphGeometry3D, VersionsDescribeGeometryLayoutAndTopologySeparately) {
    Kokkos::Serial exec;
    detail::ResourceBudget budget;
    auto state = initial_particles();
    auto previous = checked(sph::prepare_geometry_3d(exec, Fields(state).get(), 1000, budget, 1));
    const auto update = [&] {
        auto geometry = checked(sph::prepare_geometry_3d(exec, Fields(state).get(), 1000,
                                                         budget, 1, previous.get()));
        compare(geometry->grid, state, exec);
        return geometry;
    };
    state[0].position[0] += real{.125};
    auto current = update();
    EXPECT_EQ(current->geometry.value(), previous->geometry.value() + 1);
    EXPECT_EQ(current->layout, previous->layout);
    EXPECT_EQ(current->topology, previous->topology);
    previous = current;
    state[0].mass = 7;
    current = update();
    EXPECT_EQ(current->geometry, previous->geometry);
    EXPECT_EQ(current->layout, previous->layout);
    EXPECT_EQ(current->topology, previous->topology);
    previous = current;
    std::reverse(state.begin(), state.end());
    current = update();
    EXPECT_EQ(current->layout.value(), previous->layout.value() + 1);
    EXPECT_EQ(current->geometry, previous->geometry);
    EXPECT_EQ(current->topology, previous->topology);
    previous = current;
    state[0].ghost = !state[0].ghost;
    current = update();
    EXPECT_EQ(current->topology.value(), previous->topology.value() + 1);
    EXPECT_EQ(current->geometry, previous->geometry);
    previous = current;
    state[0].id += 123;
    current = update();
    EXPECT_EQ(current->topology.value(), previous->topology.value() + 1);
    previous = current;
    state.push_back({{0, 0, 0}, 1, 98765, 0});
    current = update();
    EXPECT_EQ(current->topology.value(), previous->topology.value() + 1);
}
TEST(SphGeometry3D, ThreeAxisCrossingAndInputReorderCompareGeometryById) {
    std::vector<Particle> state;
    for (int i = 0; i < 27; ++i)
        state.push_back({{real(i % 3 - 1) * real{1.25}, real(i / 3 % 3 - 1) * real{1.25},
                          real(i / 9 - 1) * real{1.25}},
                         real(1 + i % 5), (std::uint64_t(i) << 32) | 7,
                         static_cast<std::uint8_t>(i % 4 == 0)});
    Kokkos::Serial exec;
    detail::ResourceBudget budget;
    const auto first = checked(sph::prepare_geometry_3d(exec, Fields(state).get(), 64, budget, 1));
    auto crossing = state;
    std::swap(crossing.front().position, crossing.back().position);
    std::reverse(crossing.begin(), crossing.end());
    const auto next = checked(sph::prepare_geometry_3d(exec, Fields(crossing).get(), 64,
                                                       budget, 1, first.get()));
    EXPECT_EQ(next->geometry.value(), first->geometry.value() + 1);
    EXPECT_EQ(next->layout.value(), first->layout.value() + 1);
    EXPECT_EQ(next->topology, first->topology);
    ASSERT_NO_FATAL_FAILURE(compare(next->grid, crossing, exec));
    ASSERT_NO_FATAL_FAILURE(compare(first->grid, state, exec));
    std::reverse(crossing.begin(), crossing.end());
    const auto reordered = checked(sph::prepare_geometry_3d(exec, Fields(crossing).get(), 64,
                                                            budget, 1, next.get()));
    EXPECT_EQ(reordered->geometry, next->geometry);
    EXPECT_EQ(reordered->layout.value(), next->layout.value() + 1);
    EXPECT_EQ(reordered->topology, next->topology);
    ASSERT_NO_FATAL_FAILURE(compare(reordered->grid, crossing, exec));
}
} // namespace
