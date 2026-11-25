#include <Kokkos_Core.hpp>
#include <fixed_csr/Participant.hpp>
#include <fixed_csr/sph_fixture.hpp>
#include <nexora/simulation/Simulation.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace nexora;
namespace fixture = sph_fixture;
using fixture::Config;
using fixture::Participant;
using fixture::Snapshot;
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

constexpr std::size_t samples = 3;
constexpr std::size_t default_clusters = 262144;
constexpr std::size_t max_clusters = 524288; // Keeps translated half-integer positions exact.
constexpr std::size_t max_benchmark_view_bytes = 512 * 1024 * 1024; // Allows max_clusters on double.
constexpr LocalIndex long_degree = 4096;
constexpr std::uint64_t id_stride = 100;
constexpr double step = 0.125;

enum Phase { input, initialize, advance, reorder, advance_reordered, cleanup, phase_count };
constexpr std::array<std::string_view, phase_count> phase_names{
    "input", "initialize", "advance", "reorder", "advance_reordered", "cleanup"};

void require(const Status& status) {
    if (!status.ok())
        throw std::runtime_error(status.message());
}

template <class T> T require(Result<T> result) {
    require(result.status());
    return std::move(result).value();
}

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct Counts {
    std::uint64_t allocations{}, allocation_bytes{}, deallocations{}, deep_copies{}, copy_bytes{};
    std::uint64_t fences{}, peak_live_bytes{};
    double fence_ms{};
};

struct Observer {
    std::array<Counts, phase_count> counts{};
    std::unordered_map<const void*, std::uint64_t> live;
    std::uint64_t live_bytes{}, fence_id{};
    Clock::time_point fence_begin;
    Phase phase = input;
    bool failed{}, fence_open{};

    void enter(Phase next) {
        phase = next;
        counts[phase].peak_live_bytes = live_bytes;
    }

    static void allocate(Kokkos::Tools::SpaceHandle, const char*, const void* ptr,
                         std::uint64_t bytes) noexcept;
    static void deallocate(Kokkos::Tools::SpaceHandle, const char*, const void* ptr,
                           std::uint64_t bytes) noexcept;
    static void copy(Kokkos::Tools::SpaceHandle, const char*, const void*, Kokkos::Tools::SpaceHandle,
                     const char*, const void*, std::uint64_t bytes) noexcept;
    static void begin_fence(const char*, std::uint32_t, std::uint64_t* id) noexcept;
    static void end_fence(std::uint64_t id) noexcept;
};

Observer* observed = nullptr;

void Observer::allocate(Kokkos::Tools::SpaceHandle, const char*, const void* ptr,
                        std::uint64_t bytes) noexcept {
    try {
        if (!observed->live.emplace(ptr, bytes).second)
            observed->failed = true;
        observed->live_bytes += bytes;
        auto& counts = observed->counts[observed->phase];
        ++counts.allocations;
        counts.allocation_bytes += bytes;
        counts.peak_live_bytes = std::max(counts.peak_live_bytes, observed->live_bytes);
    } catch (...) {
        observed->failed = true;
    }
}

void Observer::deallocate(Kokkos::Tools::SpaceHandle, const char*, const void* ptr,
                          std::uint64_t bytes) noexcept {
    const auto found = observed->live.find(ptr);
    if (found == observed->live.end() || found->second != bytes) {
        observed->failed = true;
        return;
    }
    observed->live.erase(found);
    observed->live_bytes -= bytes;
    ++observed->counts[observed->phase].deallocations;
}

void Observer::copy(Kokkos::Tools::SpaceHandle, const char*, const void*, Kokkos::Tools::SpaceHandle,
                    const char*, const void*, std::uint64_t bytes) noexcept {
    auto& counts = observed->counts[observed->phase];
    ++counts.deep_copies;
    counts.copy_bytes += bytes;
}

void Observer::begin_fence(const char*, std::uint32_t, std::uint64_t* id) noexcept {
    if (observed->fence_open)
        observed->failed = true;
    observed->fence_open = true;
    observed->fence_begin = Clock::now();
    *id = ++observed->fence_id;
}

void Observer::end_fence(std::uint64_t id) noexcept {
    if (!observed->fence_open || id != observed->fence_id)
        observed->failed = true;
    observed->fence_open = false;
    auto& counts = observed->counts[observed->phase];
    ++counts.fences;
    counts.fence_ms += Milliseconds(Clock::now() - observed->fence_begin).count();
}

struct ScopedObservation {
    explicit ScopedObservation(Observer& observer) {
        using namespace Kokkos::Tools::Experimental;
        observed = &observer;
        set_allocate_data_callback(Observer::allocate);
        set_deallocate_data_callback(Observer::deallocate);
        set_begin_deep_copy_callback(Observer::copy);
        set_begin_fence_callback(Observer::begin_fence);
        set_end_fence_callback(Observer::end_fence);
    }

    ~ScopedObservation() {
        using namespace Kokkos::Tools::Experimental;
        set_allocate_data_callback(nullptr);
        set_deallocate_data_callback(nullptr);
        set_begin_deep_copy_callback(nullptr);
        set_begin_fence_callback(nullptr);
        set_end_fence_callback(nullptr);
        observed = nullptr;
    }
};

real mass_scale(std::size_t cluster) {
    return real{1} + static_cast<real>(cluster % 4) * real{0.125};
}

std::uint64_t long_row_id(std::size_t clusters) {
    return id_stride * (clusters + 1) + 91;
}

Config large_config(std::size_t clusters) {
    Config config;
    const auto columns = clusters * fixture::column_ids.size() + 1 + long_degree;
    const auto rows = clusters * fixture::row_ids.size() + 1;
    config.column_ids.reserve(columns);
    config.positions.reserve(columns);
    config.masses.reserve(columns);
    config.ghost.reserve(columns);
    config.boundary.reserve(columns);
    config.row_ids.reserve(rows);
    config.offsets.reserve(rows + 1);
    config.indices.reserve(clusters * fixture::indices.size() + long_degree);
    for (std::size_t cluster = 0; cluster < clusters; ++cluster) {
        const auto column_base = static_cast<LocalIndex>(config.column_ids.size());
        const auto id_base = id_stride * (cluster + 1);
        const real origin = static_cast<real>(8 * cluster);
        for (std::size_t column = 0; column < fixture::column_ids.size(); ++column) {
            config.column_ids.emplace_back(id_base + fixture::column_ids[column]);
            config.positions.push_back(origin + static_cast<real>(fixture::positions[column]));
            config.masses.push_back(mass_scale(cluster) * static_cast<real>(fixture::masses[column]));
            config.ghost.push_back(fixture::ghost[column]);
            config.boundary.push_back(fixture::boundary[column]);
        }
        for (std::size_t row = 0; row < fixture::row_ids.size(); ++row) {
            config.row_ids.emplace_back(id_base + fixture::row_ids[row]);
            config.offsets.push_back(config.indices.size());
            for (auto edge = fixture::offsets[row]; edge < fixture::offsets[row + 1]; ++edge)
                config.indices.push_back(column_base + fixture::indices[edge]);
        }
    }

    const auto id = long_row_id(clusters);
    const real origin = static_cast<real>(8 * clusters);
    config.row_ids.emplace_back(id);
    config.column_ids.emplace_back(id);
    config.positions.push_back(origin);
    config.masses.push_back(3);
    config.ghost.push_back(0);
    config.boundary.push_back(0);
    config.offsets.push_back(config.indices.size());
    for (LocalIndex edge = 0; edge < long_degree; ++edge) {
        config.indices.push_back(static_cast<LocalIndex>(config.column_ids.size()));
        config.column_ids.emplace_back(id + 1 + edge);
        config.positions.push_back(origin + static_cast<real>(1 + edge % 4) * real{0.5});
        config.masses.push_back(static_cast<real>(1 + edge % 7));
        config.ghost.push_back(1);
        config.boundary.push_back(0);
    }
    config.offsets.push_back(config.indices.size());
    return config;
}

enum class Layout { initial, reordered };

std::size_t initial_row_index(std::size_t row, std::size_t rows, std::size_t clusters,
                              Layout layout) {
    if (layout == Layout::initial)
        return row;
    if (clusters > 0)
        return rows - 1 - row;
    return std::ranges::find(fixture::row_ids, fixture::reordered_rows[row]) -
           fixture::row_ids.begin();
}

void verify(const Snapshot& state, std::size_t clusters, std::size_t rows, std::size_t columns,
            std::size_t edges, Layout layout, const Snapshot* before_reorder = nullptr) {
    check(state.stamp.accepted && state.density_valid && state.row_count() == rows &&
              state.column_count() == columns && state.edge_count() == edges &&
              state.offset(rows) == edges,
          "unexpected SPH result shape or stamp");
    constexpr long double eps = std::numeric_limits<real>::epsilon();
    long double rounded_sum = 0;
    for (std::size_t row = rows; row-- > 0;) {
        const auto initial = initial_row_index(row, rows, clusters, layout);
        const auto group = initial / fixture::row_ids.size();
        const auto particle = initial % fixture::row_ids.size();
        const bool long_row = clusters > 0 && group == clusters;
        const auto id_base = clusters > 0 ? id_stride * (group + 1) : 0;
        const auto expected_id = long_row ? long_row_id(clusters)
                                          : id_base + fixture::row_ids[particle];
        check(state.row_id(row) == expected_id, "unexpected SPH row ID or order");

        long double expected = 0;
        long double epsilon_multiplier = fixture::density_epsilon;
        if (long_row) {
            check(state.offset(row + 1) - state.offset(row) == long_degree,
                  "unexpected SPH long-row degree");
            // Half-integer weights at q = 0.5, 1, 1.5, 2, multiplied by 48.
            constexpr std::array<unsigned, 4> weights{23, 8, 1, 0};
            unsigned numerator = 3 * 32;
            for (LocalIndex edge = 0; edge < long_degree; ++edge)
                numerator += (1 + edge % 7) * weights[edge % weights.size()];
            expected = static_cast<long double>(numerator) / fixture::denominator;
            epsilon_multiplier = long_degree + 4;
        } else {
            const auto numerator = fixture::density_numerators[particle];
            const real scale = mass_scale(group);
            expected = static_cast<long double>(numerator) * scale / fixture::denominator;
        }
        const real result = state.density(row);
        check(std::abs(result - expected) <=
                                          epsilon_multiplier * eps * std::abs(expected),
              "unexpected SPH density");
        if (before_reorder)
            check(result == before_reorder->density(initial), "SPH reorder changed an accepted density");
        rounded_sum += result;
    }

    // Both double accumulation and this independent reverse sum contribute at
    // most gamma_n each. Allow one final rounding to the configured real type.
    const long double nu = static_cast<long double>(rows - 1) *
                           std::numeric_limits<double>::epsilon() / 2;
    const long double reference = clusters > 0
                                      ? rounded_sum
                                      : static_cast<long double>(fixture::sum_numerator) /
                                            fixture::denominator;
    const long double relative_tolerance =
        clusters > 0 ? 2 * nu / (1 - nu) + eps : fixture::sum_epsilon * eps;
    check(std::abs(state.owned_density_sum - reference) <= relative_tolerance * reference,
          "unexpected SPH owned density sum");
    if (before_reorder)
        check(state.owned_density_sum == before_reorder->owned_density_sum,
              "SPH reorder changed the accepted sum");
}

struct Sample {
    std::array<double, phase_count> elapsed{};
    double end_to_end_ms{};
    std::size_t rows{}, columns{}, edges{}, payload_bytes{};
    real owned_density_sum{};

    template <class Function> void measure(Phase phase, Function&& run) {
        if (observed) {
            observed->enter(phase);
            run();
            return;
        }
        const auto start = Clock::now();
        run();
        elapsed[phase] = Milliseconds(Clock::now() - start).count();
    }
};

Sample run_sample(std::size_t clusters) {
    Sample sample;
    // Check intermediate snapshots in the independent profiling pass, so the
    // oracle neither contributes elapsed time nor warms data for a timed phase.
    const bool validate = observed != nullptr;
    const auto started = Clock::now();
    Config config;
    sample.measure(input, [&] {
        config = clusters > 0 ? large_config(clusters) : fixture::frozen_config<Config>();
    });
    sample.rows = config.row_ids.size();
    sample.columns = config.column_ids.size();
    sample.edges = config.indices.size();
    sample.payload_bytes = sample.rows * (sizeof(GlobalEntityId) + sizeof(real)) +
                           sample.columns * (sizeof(GlobalEntityId) + 2 * sizeof(real) +
                                             2 * sizeof(std::uint8_t)) +
                           config.offsets.size() * sizeof(Offset) +
                           sample.edges * sizeof(LocalIndex);

    Participant participant(std::move(config), ParticipantId{1},
                            {.controlled_bytes = max_benchmark_view_bytes});
    Simulation simulation(participant);
    const auto require_lifecycle = [&](const Status& status) {
        if (!status.ok()) {
            const auto cleanup_status = simulation.finalize();
            if (!cleanup_status.ok())
                std::cerr << "benchmark finalization failed: " << cleanup_status.message() << '\n';
            require(status);
        }
    };
    sample.measure(initialize, [&] {
        require_lifecycle(simulation.initialize());
    });
    Snapshot committed;
    sample.measure(advance, [&] {
        require_lifecycle(simulation.run(step));
        committed = require(participant.snapshot());
    });
    if (validate)
        verify(committed, clusters, sample.rows, sample.columns, sample.edges, Layout::initial);
    Snapshot reordered;
    sample.measure(reorder, [&] {
        std::vector<GlobalEntityId> rows, columns;
        if (clusters > 0) {
            rows.resize(sample.rows);
            for (std::size_t i = 0; i < sample.rows; ++i)
                rows[i] = GlobalEntityId{committed.row_id(i)};
            columns.resize(sample.columns);
            for (std::size_t i = 0; i < sample.columns; ++i)
                columns[i] = GlobalEntityId{committed.column_id(i)};
            std::ranges::reverse(rows);
            std::ranges::rotate(
                columns, columns.begin() + static_cast<std::ptrdiff_t>(sample.columns / 2));
        } else {
            rows = fixture::strong_ids<GlobalEntityId>(fixture::reordered_rows);
            columns = fixture::strong_ids<GlobalEntityId>(fixture::reordered_columns);
        }
        require(participant.reorder(rows, columns, committed.stamp));
        reordered = require(participant.snapshot());
        if (validate)
            verify(reordered, clusters, sample.rows, sample.columns, sample.edges,
                   Layout::reordered, &committed);
        committed = {};
    });

    Snapshot advanced;
    sample.measure(advance_reordered, [&] {
        require_lifecycle(simulation.run(step));
        advanced = require(participant.snapshot());
    });
    if (validate)
        verify(advanced, clusters, sample.rows, sample.columns, sample.edges, Layout::reordered);
    else
        sample.end_to_end_ms = Milliseconds(Clock::now() - started).count();
    sample.owned_density_sum = advanced.owned_density_sum;
    sample.measure(cleanup, [&] {
        require(simulation.finalize());
        reordered = {};
        advanced = {};
    });
    return sample;
}

void measure_case(std::string_view name, std::size_t clusters) {
    std::array<Sample, samples + 1> timings;
    for (auto& sample : timings)
        sample = run_sample(clusters);

    Observer observer;
    {
        const ScopedObservation profile(observer);
        run_sample(clusters);
    }
    check(!observer.failed && !observer.fence_open && observer.live.empty() && observer.live_bytes == 0,
          "unpaired Kokkos profiling event");

    for (std::size_t run = 0; run <= samples; ++run) {
        const auto& sample = timings[run];
        std::cout << "case=" << name << " run=" << run << " warmup=" << (run == 0)
                  << " rows=" << sample.rows << " columns=" << sample.columns
                  << " edges=" << sample.edges << " payload_bytes=" << sample.payload_bytes
                  << " owned_density_sum=" << sample.owned_density_sum
                  << " end_to_end_ms=" << sample.end_to_end_ms;
        for (std::size_t phase = 0; phase < phase_count; ++phase)
            std::cout << ' ' << phase_names[phase] << "_ms=" << sample.elapsed[phase];
        std::cout << " edge_visits_per_second=" << sample.edges * 1000.0 / sample.elapsed[advance]
                  << '\n';
    }

    for (std::size_t phase = 0; phase < phase_count; ++phase) {
        const auto& counts = observer.counts[phase];
        std::cout << "case=" << name << " profile_phase=" << phase_names[phase]
                  << " allocations=" << counts.allocations << " allocation_bytes="
                  << counts.allocation_bytes << " deallocations=" << counts.deallocations
                  << " peak_live_kokkos_bytes=" << counts.peak_live_bytes
                  << " deep_copies=" << counts.deep_copies << " deep_copy_bytes=" << counts.copy_bytes
                  << " fences=" << counts.fences << " observed_fence_ms=" << counts.fence_ms << '\n';
    }
}

std::size_t parse_clusters(int argc, char* argv[]) {
    if (argc == 1)
        return default_clusters;
    check(argc == 2, "usage: nexora_sph_benchmark [cluster_count]");
    const std::string_view argument(argv[1]);
    std::size_t clusters{};
    const auto result = std::from_chars(argument.data(), argument.data() + argument.size(), clusters);
    check(result.ec == std::errc{} && result.ptr == argument.data() + argument.size() &&
              clusters > 0 && clusters <= max_clusters,
          "cluster_count must be between 1 and 524288");
    return clusters;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const auto clusters = parse_clusters(argc, argv);
        for (const auto* name : {"KOKKOS_TOOLS_LIBS", "KOKKOS_PROFILE_LIBRARY", "KOKKOS_TOOLS_LIBRARY"}) {
            const auto* value = std::getenv(name);
            if (value && *value)
                throw std::runtime_error(std::string("external Kokkos profiling library set by ") + name);
        }
        Kokkos::Tools::Experimental::set_request_tool_settings_callback(
            [](std::uint32_t, Kokkos::Tools::Experimental::ToolSettings* settings) {
                settings->requires_global_fencing = false;
            });
        Kokkos::ScopeGuard runtime;
        std::cout << std::boolalpha << std::setprecision(std::numeric_limits<real>::max_digits10)
                  << "execution=Serial precision=" << (sizeof(real) == 4 ? "float" : "double")
                  << " kokkos_version=" << KOKKOS_VERSION << " large_clusters=" << clusters << '\n';
        measure_case("frozen", 0);
        measure_case("large", clusters);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
