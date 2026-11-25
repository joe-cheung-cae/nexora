#include "cell_grid_oracle.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <methods/sph/Density3D.hpp>
#include <methods/sph/Verlet3D.hpp>
#include <methods/sph/neighbors/CellGrid.hpp>
#include <participants/ResourceBudget.hpp>
#include <participants/sph/GeometryVersions3D.hpp>
#include <methods/sph/Continuity3D.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string_view>
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
    const auto density = checked(sph::evaluate_density_3d(exec, grid.neighbors()));
    ASSERT_EQ(density.extent(0), owned.extent(0));
    std::vector<std::uint64_t> seen_owned;
    for (std::size_t row = 0; row < owned.extent(0); ++row) {
        const auto slot = owned(row);
        ASSERT_LT(static_cast<std::size_t>(slot), input.size());
        ASSERT_EQ(packed.ghost(slot), 0);
        const auto id = packed.ids(slot);
        seen_owned.push_back(id);
        std::vector<std::uint64_t> neighbors;
        grid.neighbors().for_each_neighbor(row, [&](LocalIndex other, double r2) {
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
// Record actual Serial allocation lifetimes. The selected eight-particle fixture has
// aligned payloads, so Kokkos profiling bytes equal the independently computed sizes.
class GridAllocations {
  public:
    struct Event {
        bool allocated;
        std::string label;
        const void* address;
        std::size_t bytes, controlled;
    };
    explicit GridAllocations(const detail::ResourceBudget& budget)
        : budget_(budget), previous_(Kokkos::Tools::Experimental::get_callbacks()) {
        active_ = this;
        using namespace Kokkos::Tools::Experimental;
        set_allocate_data_callback(record<true>);
        set_deallocate_data_callback(record<false>);
    }
    ~GridAllocations() {
        Kokkos::Tools::Experimental::set_callbacks(previous_);
        active_ = nullptr;
        EXPECT_FALSE(failed_);
    }
    std::size_t count(std::size_t begin, std::string_view label, bool allocated = true) const {
        return std::count_if(events.begin() + begin, events.end(), [&](const auto& e) {
            return e.allocated == allocated && e.label == label;
        });
    }
    std::size_t first(std::size_t begin, std::string_view label, bool allocated) const {
        const auto it = std::find_if(events.begin() + begin, events.end(), [&](const auto& e) {
            return e.allocated == allocated && e.label == label;
        });
        return it - events.begin();
    }
    // Pair each deallocation with its original allocation, even across pointer reuse.
    std::size_t peak_bytes() const {
        std::map<const void*, Event> live;
        std::size_t bytes = 0, peak = 0;
        for (const auto& e : events) {
            if (e.allocated) {
                EXPECT_TRUE(live.emplace(e.address, e).second);
                bytes += e.bytes;
                peak = std::max(peak, bytes);
                EXPECT_LE(bytes, e.controlled);
            } else {
                EXPECT_LE(bytes, e.controlled) << e.label;
                const auto found = live.find(e.address);
                if (found == live.end()) {
                    ADD_FAILURE() << "unmatched deallocation " << e.label;
                    continue;
                }
                EXPECT_EQ(found->second.bytes, e.bytes);
                EXPECT_EQ(found->second.label, e.label);
                bytes -= e.bytes;
                live.erase(found);
            }
        }
        return peak;
    }
    std::vector<Event> events;

  private:
    template <bool Allocate>
    static void record(Kokkos::Tools::SpaceHandle, const char* name, const void* address,
                       std::uint64_t bytes) noexcept {
        const std::string_view label(name);
        if (!label.starts_with("sph.grid.") || label == "sph.grid.density")
            return;
        try {
            active_->events.push_back({Allocate, name, address, bytes, active_->budget_.usage().controlled_bytes});
        } catch (...) {
            active_->failed_ = true;
        }
    }
    inline static GridAllocations* active_{};
    const detail::ResourceBudget& budget_;
    Kokkos::Tools::Experimental::EventSet previous_;
    bool failed_{};
};

std::vector<Particle> budget_particles() {
    std::vector<Particle> result;
    for (std::uint64_t i = 0; i < 8; ++i)
        result.push_back({{real(i) * real{.625}, 0, 0}, real(i + 1), 101 + i, 0});
    return result;
}

TEST(SphCellGrid, AdmissionPrecedesRemapAndFollowsSharedQueryLifetime) {
    using Grid = sph::CellGrid;
    using Group = Grid::StorageGroup;
    using Request = std::pair<Group, std::size_t>;
    constexpr std::size_t limit = 4096, n = 8;
    constexpr std::size_t values = n * 4 * sizeof(real), mappings = n * 2 * sizeof(LocalIndex);
    constexpr std::size_t cells = n * (sizeof(std::uint64_t) + sizeof(std::uint8_t) + sizeof(LocalIndex)) +
                                  4 * sizeof(LocalIndex);
    // Size of the actual (cell, id, input slot) key, including ABI padding.
    struct Key { std::size_t cell; std::uint64_t id; LocalIndex input; };
    Kokkos::Serial exec;
    detail::ResourceBudget budget({0, limit});
    GridAllocations trace(budget);
    std::vector<Request> requests;
    const auto admit = [&](Group group, std::size_t bytes) {
        requests.emplace_back(group, bytes);
        return budget.reserve(bytes);
    };
    auto input = budget_particles();
    auto reordered = input;
    std::reverse(reordered.begin(), reordered.end());
    const Fields original(input), permuted(reordered);
    std::optional<sph::NeighborView3D> held_query;
    {
        const auto old = checked(Grid::build(exec, original.get(), 3, admit));
        const std::vector<Request> rebuild{{Group::position, 3 * n * sizeof(real)}, {Group::mass, n * sizeof(real)},
                                           {Group::identity, 9 * n}, {Group::cells, cells - 9 * n},
                                           {Group::permutations, mappings}, {Group::sort_keys, n * sizeof(Key)}};
        EXPECT_EQ(requests, rebuild);
        const auto retained = values + cells + mappings;
        ASSERT_EQ(budget.usage().controlled_bytes, retained);
        EXPECT_EQ(trace.peak_bytes(), retained + n * sizeof(Key));
        held_query = old.neighbors();

        auto occupied = checked(budget.reserve(limit - retained - (mappings - 1)));
        const auto mark = trace.events.size();
        requests.clear();
        EXPECT_EQ(old.update(exec, permuted.get(), 3, admit).status().code(), StatusCode::resource_exhausted);
        EXPECT_EQ(requests, (std::vector<Request>{{Group::permutations, mappings}}));
        EXPECT_EQ(trace.events.size(), mark);
        EXPECT_EQ(limit - budget.usage().controlled_bytes, mappings - 1);
        ASSERT_NO_FATAL_FAILURE(compare(old, input, exec));

        occupied.reset();
        requests.clear();
        const auto remap = checked(old.update(exec, permuted.get(), 3, admit));
        EXPECT_EQ(requests, (std::vector<Request>{{Group::permutations, mappings}, {Group::position, 3 * n * sizeof(real)}, {Group::mass, n * sizeof(real)}}));
        EXPECT_EQ(trace.count(mark, "sph.grid.to_input"), 1u);
        EXPECT_EQ(trace.count(mark, "sph.grid.from_input"), 1u);
        EXPECT_EQ(old.cell_begin().data(), remap.cell_begin().data());
        EXPECT_EQ(budget.usage().controlled_bytes, retained + mappings + values);
        requests.clear();
        const auto reuse = checked(remap.update(exec, permuted.get(), 3, admit));
        EXPECT_EQ(requests, (std::vector<Request>{{Group::position, 3 * n * sizeof(real)}, {Group::mass, n * sizeof(real)}}));
        EXPECT_EQ(remap.sorted_to_input().data(), reuse.sorted_to_input().data());
        EXPECT_EQ(remap.input_to_sorted().data(), reuse.input_to_sorted().data());
        EXPECT_EQ(budget.usage().controlled_bytes, retained + mappings + 2 * values);
        ASSERT_NO_FATAL_FAILURE(compare(reuse, reordered, exec));
    }
    // Stencils retain values and cells. Neither input permutation is needed after grid destruction.
    EXPECT_EQ(budget.usage().controlled_bytes, cells + values);
    const auto rho = checked(sph::evaluate_density_3d(exec, *held_query));
    const auto oracle = explicit_csr(input);
    for (std::size_t r = 0; r < n; ++r) {
        const auto id = held_query->fields().ids(held_query->owned_slots()(r));
        EXPECT_NEAR(rho(r), oracle.density.at(id), 4 * std::numeric_limits<real>::epsilon() * rho(r));
    }
    held_query.reset();
    EXPECT_EQ(budget.usage().controlled_bytes, 0u);
    EXPECT_EQ(trace.peak_bytes(), std::max(cells + values + mappings + n * sizeof(Key),
                                          cells + 3 * values + 2 * mappings));
}

TEST(SphCellGrid, AssignmentKeepsReplacedArraysChargedUntilDeallocation) {
    Kokkos::Serial exec;
    detail::ResourceBudget budget;
    GridAllocations trace(budget);
    const auto admit = [&](sph::CellGrid::StorageGroup, std::size_t bytes) {
        return budget.reserve(bytes);
    };
    const Fields input(budget_particles());
    {
        auto current = checked(sph::CellGrid::build(exec, input.get(), 3, admit));
        auto query = current.neighbors();
        auto replacement = checked(sph::CellGrid::build(exec, input.get(), 3, admit));
        current = std::move(replacement); // original permutations lose their last holder
        query = current.neighbors();      // original packed values and cells lose theirs
    }
    // Each callback, including deallocation, sees at least its current live payload charged.
    EXPECT_GT(trace.peak_bytes(), 0u);
    EXPECT_EQ(budget.usage().controlled_bytes, 0u);
}

// Drive the method and the version comparator directly with the grid's actual stage layout.
struct IndexedStage {
    sph::CellGrid grid;
    TopologyVersion topology{1};
    GeometryVersion geometry{1};
    LayoutVersion layout{1};
};

Result<std::shared_ptr<const IndexedStage>>
indexed_stage(const Kokkos::Serial& exec, const sph::ParticleFields3D& input, std::size_t max_cells,
              const detail::ResourceBudget& budget, double h, const IndexedStage* old = nullptr) {
    const auto admit = [&](sph::CellGrid::StorageGroup, std::size_t bytes) { return budget.reserve(bytes); };
    auto grid = old ? old->grid.update(exec, input, max_cells, admit)
                    : sph::CellGrid::build(exec, input, max_cells, admit, h);
    if (!grid.ok())
        return grid.status();
    IndexedStage result{std::move(grid).value()};
    if (old) {
        const auto layout = [](const sph::CellGrid& grid) -> sph::GeometryLayout3D {
            return {grid.fields(), grid.owned_slots(), grid.sorted_to_input(), grid.input_to_sorted()};
        };
        result.topology = old->topology;
        result.geometry = old->geometry;
        result.layout = old->layout;
        const auto reserve = [&](std::size_t bytes) { return budget.reserve(bytes); };
        const auto compared = sph::update_geometry_versions(exec, layout(old->grid), layout(result.grid),
            reserve, result.topology, result.geometry, result.layout);
        if (!compared.ok())
            return compared;
    }
    return std::make_shared<const IndexedStage>(std::move(result));
}

TEST(SphCellGrid, RebuildFallbackAndComparisonRespectScratchLifetimes) {
    constexpr std::size_t limit = 8192, n = 8;
    struct Key { std::size_t cell; std::uint64_t id; LocalIndex input; };
    struct ById { std::uint64_t id; LocalIndex slot; };
    Kokkos::Serial exec;
    detail::ResourceBudget budget({0, limit});
    GridAllocations trace(budget);
    auto original = budget_particles();
    auto crossing = original;
    std::swap(crossing.front().position, crossing.back().position);
    std::reverse(crossing.begin(), crossing.end());
    const Fields before(original), after(crossing);
    auto initial = checked(indexed_stage(exec, before.get(), 3, budget, 1));
    const auto retained = budget.usage().controlled_bytes;
    const auto update = [&] { return indexed_stage(exec, after.get(), 3, budget, 1, initial.get()); };
    // Rebuild outputs fit, but adding its keys would exceed the available bytes by one.
    auto occupied = checked(budget.reserve(limit - 2 * retained - n * sizeof(Key) + 1));
    auto mark = trace.events.size();
    ASSERT_EQ(update().status().code(), StatusCode::resource_exhausted);
    EXPECT_EQ(trace.count(mark, "sph.grid.to_input"), 1u); // remap probe only
    EXPECT_EQ(trace.count(mark, "sph.grid.from_input"), 1u);
    EXPECT_EQ(trace.count(mark, "sph.grid.keys"), 0u);
    EXPECT_EQ(trace.count(mark, "sph.grid.position"), 0u);
    occupied.reset();
    EXPECT_EQ(budget.usage().controlled_bytes, retained);

    // Rebuild (including keys) now fits. The genuinely reordered ID comparison does not.
    occupied = checked(budget.reserve(limit - 2 * retained - 2 * n * sizeof(ById) + 1));
    mark = trace.events.size();
    ASSERT_EQ(update().status().code(), StatusCode::resource_exhausted);
    ASSERT_EQ(trace.count(mark, "sph.grid.keys"), 1u);
    EXPECT_EQ(trace.count(mark, "sph.grid.to_input"), 2u);
    EXPECT_EQ(trace.count(mark, "sph.grid.from_input"), 2u);
    EXPECT_EQ(trace.count(mark, "sph.grid.old_by_id"), 0u);
    EXPECT_EQ(trace.count(mark, "sph.grid.new_by_id"), 0u);
    const auto keys = trace.first(mark, "sph.grid.keys", true);
    EXPECT_LT(trace.first(mark, "sph.grid.to_input", false), keys);
    EXPECT_LT(trace.first(mark, "sph.grid.from_input", false), keys);
    occupied.reset();
    EXPECT_EQ(budget.usage().controlled_bytes, retained);
    ASSERT_NO_FATAL_FAILURE(compare(initial->grid, original, exec));
    EXPECT_EQ(initial->topology.value(), 1u);
    EXPECT_EQ(initial->geometry.value(), 1u);
    EXPECT_EQ(initial->layout.value(), 1u);

    mark = trace.events.size();
    auto candidate = checked(update());
    EXPECT_EQ(trace.count(mark, "sph.grid.old_by_id"), 1u);
    EXPECT_EQ(trace.count(mark, "sph.grid.new_by_id"), 1u);
    EXPECT_EQ(trace.peak_bytes(), 2 * retained + 2 * n * sizeof(ById));
    EXPECT_EQ(budget.usage().controlled_bytes, 2 * retained);
    EXPECT_EQ(candidate->topology, initial->topology);
    EXPECT_EQ(candidate->geometry.value(), 2u);
    EXPECT_EQ(candidate->layout.value(), 2u);
    ASSERT_NO_FATAL_FAILURE(compare(candidate->grid, crossing, exec));
    const auto rebuilt = checked(sph::CellGrid::build(exec, after.get(), 3));
    ASSERT_NO_FATAL_FAILURE(compare(rebuilt, crossing, exec));
    for (std::size_t s = 0; s < n; ++s)
        EXPECT_EQ(candidate->grid.sorted_to_input()(s), rebuilt.sorted_to_input()(s));
    candidate.reset();
    initial.reset();
    EXPECT_EQ(budget.usage().controlled_bytes, 0u);
}

TEST(SphCellGrid, NeighborViewPreservesCallbackOrder) {
    using Neighbors = std::vector<std::pair<std::uint64_t, double>>;
    const Neighbors expected{{90, 1.0 / 16}, {10, 9.0 / 64}};
    enum class Center { fluid, wall, ghost };
    for (auto center : {Center::fluid, Center::wall, Center::ghost}) {
        std::vector<Particle> input{{{.75, 0, 0}, 3, 90, 0}, {{1.375, 0, 0}, 1, 10, 0},
                                    {{1, 0, 0}, 2, 50, std::uint8_t(center != Center::fluid)}};
        if (center == Center::ghost)
            std::rotate(input.begin(), input.end() - 1, input.end());
        for (int permutation = 0; permutation < 2; ++permutation) {
            const auto grid = checked(sph::CellGrid::build(Kokkos::Serial{}, Fields(input).get(), 2, {}, .5));
            const auto& query = grid.neighbors();
            const auto index = std::find_if(input.begin(), input.end(), [](auto p) { return p.id == 50; }) - input.begin();
            const auto slot = grid.input_to_sorted()(index);
            Neighbors actual;
            const auto append = [&](LocalIndex j, double d2) { actual.emplace_back(query.fields().ids(j), d2); };
            query.for_each_neighbor_of_slot(slot, append);
            EXPECT_EQ(actual, expected);
            if (center == Center::fluid) {
                const auto rows = query.owned_slots();
                const auto row = std::find(rows.data(), rows.data() + rows.extent(0), slot) - rows.data();
                ASSERT_LT(std::size_t(row), rows.extent(0));
                actual.clear();
                query.for_each_neighbor(row, append);
                EXPECT_EQ(actual, expected);
            }
            // Wall input stays behind the fluid prefix.
            std::reverse(input.begin(), center == Center::wall ? input.end() - 1 : input.end());
        }
    }
}

TEST(SphCellGrid, IsolatedMassAndSmoothingLengthSetSelfDensity) {
    Kokkos::Serial exec;
    const auto grid = checked(sph::CellGrid::build(exec, Fields({{{.25, 1, 0}, 5, 6, 0}}).get(), 1, {}, .5));
    const auto& query = grid.neighbors();
    std::size_t callbacks = 0;
    query.for_each_neighbor(0, [&](LocalIndex, double) { ++callbacks; });
    EXPECT_EQ(callbacks, 0u);
    const auto rho = checked(sph::evaluate_density_3d(exec, query));
    EXPECT_NEAR(rho(0), 40 / std::numbers::pi, 40 * std::numeric_limits<real>::epsilon());
}

TEST(SphCellGrid, PairMirrorsAndRelativeVelocityReversalMatchAnalyticRatesAndForces) {
    // For h=1 at unit separation W=1/(4*pi) and |dW/dr|=3/(4*pi).
    // Velocities are centered on the same mass-weighted mean of 1.
    Kokkos::Serial exec;
    const double pi = std::numbers::pi;
    const double gradient = 3 / (4 * pi);
    for (const auto [mirror, relative] : {std::pair{1, 1}, {-1, 1}, {1, -1}}) {
        const std::vector<Particle> pair{{{real(mirror), 0, 0}, 2, 22, 0}, {{0, 0, 0}, 1, 11, 0}};
        const auto grid = checked(sph::CellGrid::build(exec, Fields(pair).get(), 10));
        const auto& query = grid.neighbors();
        sph::Velocity3D velocity("test.velocity", 2);
        for (LocalIndex s = 0; s < 2; ++s)
            velocity(s, 0) = mirror * (1 + relative * (query.fields().ids(s) == 11 ? 2 : -1));
        const auto rho = checked(sph::evaluate_density_3d(exec, query));
        sph::DensityRate3D rate("test.rate", 2);
        ASSERT_TRUE(sph::continuity_rate_3d(exec, query, velocity, rate).ok());
        sph::Values pressure("test.pressure", 2);
        sph::Acceleration3D acceleration("test.acceleration", 2);
        const sph::WcsphParameters parameters{.25, 2, 1};
        ASSERT_TRUE(sph::evaluate_pressure_3d(exec, query, sph::TaitEos(parameters), rho, pressure, acceleration).ok());
        const double rho_a = real(1.5 / pi), rho_b = real(2.25 / pi);
        const double p_a = real(4 * (rho_a - .25)), p_b = real(4 * (rho_b - .25));
        const double coefficient = p_a / (rho_a * rho_a) + p_b / (rho_b * rho_b);
        for (std::size_t row = 0; row < 2; ++row) {
            const bool a = query.fields().ids(query.owned_slots()(row)) == 11;
            const double neighbor_mass = a ? 2 : 1;
            const double expected_rate = relative * 3 * neighbor_mass * gradient;
            EXPECT_NEAR(rate(row), expected_rate, 4 * std::numeric_limits<double>::epsilon() * std::abs(expected_rate));
            EXPECT_EQ(rho(row), a ? rho_a : rho_b);
            EXPECT_GT(pressure(row), 0);
            const double force = (a ? -mirror : mirror) * neighbor_mass * gradient * coefficient;
            EXPECT_NEAR(acceleration(row, 0), force, 8 * std::numeric_limits<real>::epsilon() * std::abs(force));
            EXPECT_EQ(acceleration(row, 1), 0);
            EXPECT_EQ(acceleration(row, 2), 0);
        }
    }
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
    grid.neighbors().for_each_neighbor(0, [&](LocalIndex slot, double distance_squared) {
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

TEST(SphCellGrid, TranslatedCellRangesKeepIntegerPrecision) {
    Kokkos::Serial exec;
    const real base = sizeof(real) == sizeof(double) ? real{0x1p54} : real{0x1p24};
    // At width 3/2, the exact bins of 2^54 + {0,4,8} are
    // 12009599006321322, 12009599006321325 and 12009599006321328.
    // Mirroring gives -12009599006321323, -12009599006321326 and -12009599006321328.
    // 2^24 has the same residues mod 3 and keeps the float coordinates distinct.
    for (const real sign : {real{1}, real{-1}}) {
        for (const real shift : {real{0}, real{4}}) {
            SCOPED_TRACE(::testing::Message() << "sign=" << sign << ", shift=" << shift);
            const real x = base + shift;
            Fields fields({{{sign * x, 0, 0}, 1, 10, 0}, {{sign * (x + 4), 0, 0}, 1, 20, 0}});
            const std::size_t count = sign < 0 && shift == 4 ? 3 : 4;
            EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), count - 1, {}, .75).status().code(),
                      StatusCode::resource_exhausted);
            const auto grid = checked(sph::CellGrid::build(exec, fields.get(), count, {}, .75));
            ASSERT_EQ(grid.cell_begin().extent(0), count + 1);
            EXPECT_EQ(grid.cell_begin()(0), 0U);
            for (std::size_t cell = 1; cell < count; ++cell)
                EXPECT_EQ(grid.cell_begin()(cell), 1U);
            EXPECT_EQ(grid.cell_begin()(count), 2U);
            EXPECT_EQ(grid.fields().ids(0), sign > 0 ? 10U : 20U);
            EXPECT_EQ(grid.fields().ids(1), sign > 0 ? 20U : 10U);
            for (LocalIndex slot = 0; slot < 2; ++slot)
                grid.neighbors().for_each_neighbor_of_slot(slot, [](LocalIndex, double) {
                    ADD_FAILURE() << "particles four units apart exceed the 1.5 support radius";
                });
        }
    }
}

TEST(SphCellGrid, SignedCellRangeIncludesOnlyItsLowerEndpoint) {
    Kokkos::Serial exec;
    const real limit = real{1.5} * real{0x1p63};
    Fields fields({{{0, 0, 0}, 1, 10, 0}, {{0, .5, 0}, 1, 20, 1}});
    for (const real x : {-limit, std::nextafter(-limit, real{0}), std::nextafter(limit, real{0})}) {
        fields.position(0, 0) = fields.position(1, 0) = x;
        const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1, {}, .75));
        std::vector<std::uint64_t> ids;
        grid.neighbors().for_each_neighbor(0, [&](LocalIndex slot, double d2) {
            ids.push_back(grid.fields().ids(slot));
            EXPECT_EQ(d2, .25);
        });
        EXPECT_EQ(ids, (std::vector<std::uint64_t>{20}));
    }
    for (const real x : {std::nextafter(-limit, -std::numeric_limits<real>::infinity()), limit}) {
        fields.position(0, 0) = x;
        EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 1, {}, .75).status().code(),
                  StatusCode::unsupported);
    }
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
    const auto accepted = checked(sph::CellGrid::build(exec, fields.get(), 100));
    for (const real mass : {real{0}, std::numeric_limits<real>::infinity()}) {
        fields.mass(0) = mass;
        EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 100).status().code(),
                  StatusCode::invalid_argument)
            << mass;
        EXPECT_EQ(accepted.update(exec, fields.get(), 100).status().code(),
                  StatusCode::invalid_argument)
            << mass;
    }
    fields.mass(0) = 1;
    fields.ghost(0) = 2;
    EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 100).status().code(),
              StatusCode::invalid_argument);
    EXPECT_EQ(accepted.update(exec, fields.get(), 100).status().code(),
              StatusCode::invalid_argument);
    fields.ghost(0) = 0;
    fields.position(0, 0) = std::numeric_limits<real>::quiet_NaN();
    EXPECT_EQ(sph::CellGrid::build(exec, fields.get(), 100).status().code(),
              StatusCode::invalid_argument);
    EXPECT_EQ(accepted.update(exec, fields.get(), 100).status().code(),
              StatusCode::invalid_argument);
}

TEST(SphCellGrid, PositionUpdateRetainsValidatedFieldsAndRejectsInvalidCoordinates) {
    Kokkos::Serial exec;
    const Fields fields({{{0, 0, 0}, 2, 10, 0}, {{1, 0, 0}, 3, 20, 1}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 4));
    const Fields shifted({{{0, 0, 0}, 2, 10, 0}, {{2.5, 0, 0}, 3, 20, 1}});
    const auto updated = checked(grid.update_positions(exec, shifted.position, 4));
    EXPECT_EQ(updated.cell_begin().extent(0), 3u);
    compare(updated, {{{0, 0, 0}, 2, 10, 0}, {{2.5, 0, 0}, 3, 20, 1}}, exec);
    const auto previous = grid.neighbors();
    std::vector<std::uint64_t> ids;
    previous.for_each_neighbor(0, [&](LocalIndex slot, double) {
        ids.push_back(previous.fields().ids(slot));
    });
    EXPECT_EQ(ids, (std::vector<std::uint64_t>{20}));

    const auto limit = real{0x1p64};
    shifted.position(0, 0) = limit;
    EXPECT_EQ(grid.update_positions(exec, shifted.position, 4).status().code(), StatusCode::unsupported);
    shifted.position(1, 0) = std::numeric_limits<real>::quiet_NaN();
    EXPECT_EQ(grid.update_positions(exec, shifted.position, 4).status().code(), StatusCode::invalid_argument);
}

TEST(SphCellGrid, PositionUpdatesReuseSortedColumnsAndResetNonidentityPermutations) {
    Kokkos::Serial exec;
    const std::vector<Particle> initial{{{2.25, .125, 0}, 3, 30, 1},
                                        {{.75, .25, 0}, 2, 20, 0},
                                        {{.25, .125, 0}, 1, 10, 0}};
    const Fields original_source(initial);
    const auto original = checked(sph::CellGrid::build(exec, original_source.get(), 4));
    ASSERT_EQ(original.sorted_to_input()(0), 2u);
    std::vector<Particle> sorted{{{.5, .125, 0}, 1, 10, 0},
                                  {{1, .25, 0}, 2, 20, 0},
                                  {{2.5, .125, 0}, 3, 30, 1}};
    std::vector<sph::CellGrid::StorageGroup> admissions;
    const auto admit = [&](sph::CellGrid::StorageGroup group, std::size_t)
        -> Result<sph::CellGrid::Ticket> {
        admissions.push_back(group);
        return sph::CellGrid::Ticket{};
    };
    const Fields first_source(sorted);
    const auto first = checked(original.update_positions(exec, first_source.position, 4, admit));
    // Const input Views may have writable caller aliases. The accepted grid's
    // copied values, including the newly shared mass column, must not alias them.
    original_source.mass(0) = 99;
    original_source.ids(0) = 999;
    original_source.ghost(0) = 0;
    first_source.position(0, 0) = 100;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        EXPECT_EQ(first.sorted_to_input()(i), i);
        EXPECT_EQ(first.input_to_sorted()(i), i);
    }
    EXPECT_EQ(first.cell_begin().data(), original.cell_begin().data());
    EXPECT_EQ(first.fields().mass.data(), original.fields().mass.data());
    EXPECT_EQ(first.fields().ids.data(), original.fields().ids.data());
    EXPECT_EQ(first.sorted_to_input().data(), first.input_to_sorted().data());
    EXPECT_EQ(admissions, (std::vector{sph::CellGrid::StorageGroup::permutations,
                                      sph::CellGrid::StorageGroup::position}));
    ASSERT_NO_FATAL_FAILURE(compare(first, sorted, exec));

    for (auto& particle : sorted)
        particle.position[0] += real{.125};
    admissions.clear();
    const auto second = checked(first.update_positions(exec, Fields(sorted).position, 4, admit));
    EXPECT_EQ(second.cell_begin().data(), first.cell_begin().data());
    EXPECT_EQ(second.fields().mass.data(), first.fields().mass.data());
    EXPECT_EQ(second.sorted_to_input().data(), first.sorted_to_input().data());
    EXPECT_EQ(second.input_to_sorted().data(), first.input_to_sorted().data());
    EXPECT_EQ(admissions, (std::vector{sph::CellGrid::StorageGroup::position}));
    ASSERT_NO_FATAL_FAILURE(compare(second, sorted, exec));

    // Exchange cells without changing the bounding box. Mass and ownership must
    // follow IDs when the positional membership test requires a fresh ordering.
    sorted[0].position[0] = real{2.875};
    sorted[2].position[0] = real{.875};
    const auto crossed = checked(second.update_positions(exec, Fields(sorted).position, 4));
    EXPECT_EQ(crossed.sorted_to_input()(0), 1u);
    EXPECT_EQ(crossed.input_to_sorted()(0), 2u);
    EXPECT_NE(crossed.cell_begin().data(), second.cell_begin().data());
    EXPECT_EQ(crossed.fields().ids(0), 20u);
    EXPECT_EQ(crossed.fields().ids(1), 30u);
    EXPECT_EQ(crossed.fields().ids(2), 10u);
    ASSERT_NO_FATAL_FAILURE(compare(crossed, sorted, exec));
    ASSERT_NO_FATAL_FAILURE(compare(original, initial, exec));
}

TEST(SphCellGrid, PositionUpdatesResolveExternalRemapsWithoutExtraPermutationStorage) {
    Kokkos::Serial exec;
    const std::vector<Particle> sorted{{{.25, 0, 0}, 1, 10, 0},
                                       {{.75, 0, 0}, 2, 20, 0},
                                       {{2.25, 0, 0}, 3, 30, 1}};
    auto reversed = sorted;
    std::reverse(reversed.begin(), reversed.end());
    const Fields positions(sorted);
    const auto built = checked(sph::CellGrid::build(exec, positions.get(), 4));
    const auto remapped = checked(built.update(exec, Fields(reversed).get(), 4));
    const auto restored = checked(remapped.update(exec, positions.get(), 4));
    std::vector<sph::CellGrid::StorageGroup> admissions;
    const auto admit = [&](sph::CellGrid::StorageGroup group, std::size_t)
        -> Result<sph::CellGrid::Ticket> {
        admissions.push_back(group);
        return sph::CellGrid::Ticket{};
    };
    const auto check_identity_reuse = [&](const sph::CellGrid& old) {
        admissions.clear();
        const auto next = checked(old.update_positions(exec, positions.position, 4, admit));
        EXPECT_EQ(admissions, (std::vector{sph::CellGrid::StorageGroup::position}));
        EXPECT_EQ(next.sorted_to_input().data(), old.sorted_to_input().data());
        EXPECT_EQ(next.input_to_sorted().data(), old.input_to_sorted().data());
        EXPECT_EQ(next.fields().mass.data(), old.fields().mass.data());
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            EXPECT_EQ(next.sorted_to_input()(i), i);
            EXPECT_EQ(next.input_to_sorted()(i), i);
        }
        ASSERT_NO_FATAL_FAILURE(compare(next, sorted, exec));
    };
    ASSERT_NO_FATAL_FAILURE(check_identity_reuse(built));
    ASSERT_NO_FATAL_FAILURE(check_identity_reuse(restored));
    admissions.clear();
    const auto next = checked(remapped.update_positions(exec, positions.position, 4, admit));
    EXPECT_EQ(admissions, (std::vector{sph::CellGrid::StorageGroup::permutations,
                                      sph::CellGrid::StorageGroup::position}));
    ASSERT_NO_FATAL_FAILURE(compare(next, sorted, exec));
    ASSERT_NO_FATAL_FAILURE(compare(remapped, reversed, exec));
}

TEST(SphCellGrid, PositionUpdateBudgetRejectsIdentityAndPositionBeforePublishing) {
    constexpr std::size_t limit = 4096, n = 3;
    constexpr std::size_t identity_bytes = n * sizeof(LocalIndex);
    constexpr std::size_t position_bytes = 3 * n * sizeof(real);
    Kokkos::Serial exec;
    detail::ResourceBudget budget({0, limit});
    GridAllocations trace(budget);
    const auto admit = [&](sph::CellGrid::StorageGroup, std::size_t bytes) {
        return budget.reserve(bytes);
    };
    const std::vector<Particle> initial{{{2.25, 0, 0}, 3, 30, 1},
                                        {{.75, 0, 0}, 2, 20, 0},
                                        {{.25, 0, 0}, 1, 10, 0}};
    const auto original = checked(sph::CellGrid::build(exec, Fields(initial).get(), 4, admit));
    const auto retained = budget.usage().controlled_bytes;
    const std::vector<Particle> sorted{{{.5, 0, 0}, 1, 10, 0},
                                       {{1, 0, 0}, 2, 20, 0},
                                       {{2.5, 0, 0}, 3, 30, 1}};
    const Fields positions(sorted);
    auto occupied = checked(budget.reserve(limit - retained - identity_bytes + 1));
    auto mark = trace.events.size();
    EXPECT_EQ(original.update_positions(exec, positions.position, 4, admit).status().code(),
              StatusCode::resource_exhausted);
    EXPECT_EQ(trace.events.size(), mark);
    occupied.reset();
    EXPECT_EQ(budget.usage().controlled_bytes, retained);

    occupied = checked(budget.reserve(limit - retained - identity_bytes - position_bytes + 1));
    mark = trace.events.size();
    EXPECT_EQ(original.update_positions(exec, positions.position, 4, admit).status().code(),
              StatusCode::resource_exhausted);
    EXPECT_EQ(trace.count(mark, "sph.grid.identity", true), 1u);
    EXPECT_EQ(trace.count(mark, "sph.grid.identity", false), 1u);
    EXPECT_EQ(trace.count(mark, "sph.grid.position"), 0u);
    EXPECT_EQ(trace.count(mark, "sph.grid.mass"), 0u);
    occupied.reset();
    EXPECT_EQ(budget.usage().controlled_bytes, retained);
    ASSERT_NO_FATAL_FAILURE(compare(original, initial, exec));

    {
        const auto first = checked(original.update_positions(exec, positions.position, 4, admit));
        const auto first_retained = retained + identity_bytes + position_bytes;
        EXPECT_EQ(budget.usage().controlled_bytes, first_retained);
        occupied = checked(budget.reserve(limit - first_retained - position_bytes + 1));
        mark = trace.events.size();
        EXPECT_EQ(first.update_positions(exec, positions.position, 4, admit).status().code(),
                  StatusCode::resource_exhausted);
        EXPECT_EQ(trace.events.size(), mark);
        occupied.reset();
        EXPECT_EQ(budget.usage().controlled_bytes, first_retained);
        const auto second = checked(first.update_positions(exec, positions.position, 4, admit));
        EXPECT_EQ(budget.usage().controlled_bytes, first_retained + position_bytes);
        EXPECT_EQ(second.fields().mass.data(), original.fields().mass.data());
        EXPECT_EQ(second.input_to_sorted().data(), first.input_to_sorted().data());
        ASSERT_NO_FATAL_FAILURE(compare(second, sorted, exec));
    }
    EXPECT_EQ(budget.usage().controlled_bytes, retained);
    ASSERT_NO_FATAL_FAILURE(compare(original, initial, exec));
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
    const auto before = checked(sph::evaluate_density_3d(exec, first.neighbors()));
    const auto after = checked(sph::evaluate_density_3d(exec, reused.neighbors()));
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
    const auto first = checked(indexed_stage(exec, input.get(), 1000, budget, 1));
    auto same_cell = initial;
    same_cell[0].position[0] = real{1.75};
    same_cell[1].mass = 5;
    Fields fields(same_cell);
    const auto next = checked(indexed_stage(exec, fields.get(), 1000, budget, 1,
                                                       first.get()));
    // Geometry owns its values; caller reuse cannot change any retained result.
    fields.position(0, 0) = 100;
    fields.mass(1) = 99;
    ASSERT_NO_FATAL_FAILURE(compare(next->grid, same_cell, exec));
    auto crossing = same_cell;
    std::swap(crossing[0].position, crossing[1].position);
    std::swap(crossing[2].position, crossing[3].position);
    std::reverse(crossing.begin(), crossing.end());
    const auto latest = checked(indexed_stage(exec, Fields(crossing).get(), 1000,
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
    auto previous = checked(indexed_stage(exec, Fields(state).get(), 1000, budget, 1));
    const auto update = [&] {
        auto geometry = checked(indexed_stage(exec, Fields(state).get(), 1000,
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
    const auto first = checked(indexed_stage(exec, Fields(state).get(), 64, budget, 1));
    auto crossing = state;
    std::swap(crossing.front().position, crossing.back().position);
    std::reverse(crossing.begin(), crossing.end());
    const auto next = checked(indexed_stage(exec, Fields(crossing).get(), 64,
                                                       budget, 1, first.get()));
    EXPECT_EQ(next->geometry.value(), first->geometry.value() + 1);
    EXPECT_EQ(next->layout.value(), first->layout.value() + 1);
    EXPECT_EQ(next->topology, first->topology);
    ASSERT_NO_FATAL_FAILURE(compare(next->grid, crossing, exec));
    ASSERT_NO_FATAL_FAILURE(compare(first->grid, state, exec));
    std::reverse(crossing.begin(), crossing.end());
    const auto reordered = checked(indexed_stage(exec, Fields(crossing).get(), 64,
                                                            budget, 1, next.get()));
    EXPECT_EQ(reordered->geometry, next->geometry);
    EXPECT_EQ(reordered->layout.value(), next->layout.value() + 1);
    EXPECT_EQ(reordered->topology, next->topology);
    ASSERT_NO_FATAL_FAILURE(compare(reordered->grid, crossing, exec));
}
} // namespace
