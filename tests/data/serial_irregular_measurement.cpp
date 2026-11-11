#ifdef NDEBUG
#undef NDEBUG
#endif
#include "structure_fixture.hpp"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

// A test-only observer for the exact locked Serial profile. No allocator,
// execution wrapper, or accounting is introduced in production code.
namespace {
using namespace nexora::test_fixture;
using Space = Kokkos::Tools::SpaceHandle;
struct Event {
    std::string kind, phase, label, source, space;
    std::uint64_t bytes = 0, allocation_id = 0, live_bytes = 0;
};
struct Allocation {
    std::string label, space;
    std::uint64_t bytes, id;
};
struct Observer {
    std::string phase = "initialize", peak_phase;
    std::vector<Event> events;
    std::map<const void*, Allocation> live;
    std::map<std::uint64_t, std::string> open_fences, open_kernels;
    std::uint64_t next_id = 0, next_operation = 0;
    std::uint64_t allocated = 0, deallocated = 0, allocated_bytes = 0, deallocated_bytes = 0;
    std::uint64_t live_bytes = 0, peak_bytes = 0;
    std::uint64_t packed_live = 0, packed_peak = 0, copies = 0, copy_bytes = 0, copy_ends = 0;
    std::uint64_t explicit_fences = 0, internal_fences = 0, fence_ends = 0;
    std::uint64_t kernels = 0, kernel_ends = 0, finalizations = 0;
    bool failed = false, copy_active = false;
    void event(std::string kind, std::string label = {}, std::uint64_t bytes = 0,
               std::uint64_t id = 0, std::string source = {}, std::string space = {}) {
        events.push_back({std::move(kind), phase, std::move(label), std::move(source),
                          std::move(space), bytes, id, live_bytes});
    }
    void stage(const char* name) {
        phase = name;
        event("phase");
    }
};
Observer observed;
// Never let an instrumentation exception cross a Kokkos callback boundary.
template<class F> void callback(F&& f) noexcept {
    try { f(); } catch (...) { observed.failed = true; }
}
bool packed(std::string_view label) { return label.starts_with("transport."); }
void allocate(Space space, const char* label, const void* pointer, std::uint64_t bytes) {
    callback([&] {
        const auto id = ++observed.next_id;
        if (!observed.live.emplace(pointer, Allocation{label, space.name, bytes, id}).second)
            observed.failed = true;
        ++observed.allocated;
        observed.allocated_bytes += bytes;
        observed.live_bytes += bytes;
        if (observed.live_bytes > observed.peak_bytes) {
            observed.peak_bytes = observed.live_bytes;
            observed.peak_phase = observed.phase;
        }
        if (packed(label)) observed.packed_live += bytes;
        observed.packed_peak = std::max(observed.packed_peak, observed.packed_live);
        observed.event("allocate", label, bytes, id, {}, space.name);
    });
}
void deallocate(Space space, const char* label, const void* pointer, std::uint64_t bytes) {
    callback([&] {
        const auto found = observed.live.find(pointer);
        if (found == observed.live.end() || found->second.bytes != bytes ||
            found->second.label != label || found->second.space != space.name) {
            observed.failed = true;
            return;
        }
        const auto id = found->second.id;
        observed.live.erase(found);
        ++observed.deallocated;
        observed.deallocated_bytes += bytes;
        observed.live_bytes -= bytes;
        if (packed(label)) observed.packed_live -= bytes;
        observed.event("deallocate", label, bytes, id, {}, space.name);
    });
}
void begin_copy(Space destination, const char* label, const void*, Space source,
                const char* source_label, const void*, std::uint64_t bytes) {
    callback([&] {
        if (observed.copy_active) observed.failed = true;
        observed.copy_active = true;
        ++observed.copies;
        observed.copy_bytes += bytes;
        observed.event("deep_copy", label, bytes, 0, source_label,
                       std::string(source.name) + "->" + destination.name);
    });
}
void end_copy() {
    if (!observed.copy_active) observed.failed = true;
    observed.copy_active = false;
    ++observed.copy_ends;
}
bool explicit_fence(std::string_view name) {
    constexpr std::array names{
        "structure initial state ready", "transport stable IDs ready for validation",
        "transport payload ready", "entity matching source ready",
        "structure partial candidate ready", "structure candidate diagnostics ready",
        "entity order source ready", "entity order snapshot ready", "transport receive ready",
        "structure value commit ready"};
    return std::find(names.begin(), names.end(), name) != names.end();
}
void begin_fence(const char* name, std::uint32_t, std::uint64_t* id) {
    callback([&] {
        *id = ++observed.next_operation;
        observed.open_fences.emplace(*id, name);
        const bool explicit_call = explicit_fence(name);
        // Unknown labels fail instead of silently attributing a new call site.
        if (!explicit_call && !std::string_view(name).starts_with("Kokkos::"))
            observed.failed = true;
        if (explicit_call) ++observed.explicit_fences;
        else ++observed.internal_fences;
        observed.event(explicit_call ? "explicit_fence" : "kokkos_internal_fence", name);
    });
}
void end_fence(std::uint64_t id) {
    if (observed.open_fences.erase(id) != 1) observed.failed = true;
    ++observed.fence_ends;
}
void begin_kernel(const char* name, std::uint32_t, std::uint64_t* id) {
    callback([&] {
        *id = ++observed.next_operation;
        observed.open_kernels.emplace(*id, name);
        ++observed.kernels;
        observed.event("parallel_for", name);
    });
}
void end_kernel(std::uint64_t id) {
    if (observed.open_kernels.erase(id) != 1) observed.failed = true;
    ++observed.kernel_ends;
}
void finalize() { ++observed.finalizations; }
void settings(std::uint32_t, Kokkos::Tools::Experimental::ToolSettings* value) {
    // Observation must not request extra profiling fences around kernels.
    value->requires_global_fencing = false;
}
void install_observer() {
    using namespace Kokkos::Tools::Experimental;
    set_allocate_data_callback(allocate);
    set_deallocate_data_callback(deallocate);
    set_begin_deep_copy_callback(begin_copy);
    set_end_deep_copy_callback(end_copy);
    set_begin_fence_callback(begin_fence);
    set_end_fence_callback(end_fence);
    set_begin_parallel_for_callback(begin_kernel);
    set_end_parallel_for_callback(end_kernel);
    set_finalize_callback(finalize);
    set_request_tool_settings_callback(settings);
}
void run(const Kokkos::Serial& exec) {
    observed.stage("input_and_create");
    auto config = fixture_config();
    auto state = require(StructureState::create(exec, config));
    auto original = state->snapshot();
    expect_bank(*original, false);
    observed.stage("pack");
    auto payload = pack_bank(exec, *original);
    observed.stage("reorder");
    assert(state->reorder(reordered_ids).ok());
    auto reordered = state->snapshot();
    expect_bank(*original, false);
    expect_bank(*reordered, true);
    observed.stage("receive");
    receive_reordered(exec, payload, *reordered);
    observed.stage("commit");
    assert(state->commit_values(std::array<real, 4>{21, 11, 41, 31}).ok());
    auto committed = state->snapshot();
    expect_bank(*committed, true, true);
    assert(committed->logical.accepted_version == StateVersion{12});
    assert(committed->logical.layout == LayoutVersion{8});
    assert(committed->adjacency.indices().data() == reordered->adjacency.indices().data());
    observed.stage("owner_close");
    const auto before_close = observed.live_bytes;
    state.reset();
    assert(observed.live_bytes == before_close);
    expect_bank(*original, false);
    expect_bank(*reordered, true);
    expect_bank(*committed, true, true);
    assert(payload.values()(0, 0) == real{100});
    observed.stage("release_snapshots");
    original.reset();
    reordered.reset();
    committed.reset();
    // The independent transport payload is the last numerical owner.
    assert(observed.live_bytes == 4 * (sizeof(std::uint64_t) + sizeof(real)));
    assert(payload.values()(3, 0) == real{400});
    observed.stage("release_payload");
    payload.complete();
    assert(observed.live_bytes == 0);
    observed.stage("input_cleanup");
}
void print_report() {
    const auto& o = observed;
    std::cout << "{\"schema_version\":1,\"fixture\":\"structure_reorder_irregular\","
              << "\"precision\":" << std::quoted(sizeof(real) == 4 ? "float" : "double")
              << ",\"kokkos_version\":" << KOKKOS_VERSION
              << ",\"numerical_validation\":\"passed\",\"resource_budget\":{\"applicable\":false,"
              << "\"reason\":\"The measured StructureState path has no ResourceBudget\"},"
              << "\"counts\":{\"allocations\":" << o.allocated << ",\"deallocations\":" << o.deallocated
              << ",\"deep_copy_calls\":" << o.copies << ",\"deep_copy_bytes\":" << o.copy_bytes
              << ",\"deep_copy_ends\":" << o.copy_ends << ",\"explicit_fences\":" << o.explicit_fences
              << ",\"kokkos_internal_fences\":" << o.internal_fences << ",\"fence_ends\":" << o.fence_ends
              << ",\"parallel_for_calls\":" << o.kernels << ",\"parallel_for_ends\":" << o.kernel_ends
              << ",\"finalize_callbacks\":" << o.finalizations << "},"
              << "\"bytes\":{\"allocated_payload\":" << o.allocated_bytes
              << ",\"deallocated_payload\":" << o.deallocated_bytes
              << ",\"peak_live_payload\":" << o.peak_bytes << ",\"peak_phase\":" << std::quoted(o.peak_phase)
              << ",\"peak_live_transport_payload\":" << o.packed_peak << ",\"final_live_payload\":" << o.live_bytes << "},"
              << "\"interpretation\":{\"allocation_bytes\":\"Callback-reported HostSpace View bytes include 8-byte span rounding. Allocation and free callbacks are paired by address.\","
              << "\"deep_copy_bytes\":\"API-reported destination bytes include scalar fill. Gather, pack, unpack and host-loop traffic are excluded.\","
              << "\"fences\":\"Counts describe observed fence begin/end events. The profiler requests no additional global fences.\","
              << "\"in_flight\":\"Serial is synchronous. Live View bytes include retained snapshots, candidates, temporaries and packed consumer payload.\","
              << "\"excluded\":\"STL/config/observer storage, allocation headers and allocator overhead, RSS, host-loop traffic, kernel memory traffic and physical device transfers\"},"
              << "\"events\":[";
    bool first = true;
    for (const auto& event : o.events) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"kind\":" << std::quoted(event.kind) << ",\"phase\":" << std::quoted(event.phase)
                  << ",\"label\":" << std::quoted(event.label) << ",\"source\":" << std::quoted(event.source)
                  << ",\"space\":" << std::quoted(event.space) << ",\"bytes\":" << event.bytes
                  << ",\"allocation_id\":" << event.allocation_id << ",\"live_bytes\":" << event.live_bytes << '}';
    }
    std::cout << "]}\n";
}
} // namespace
int main() {
    static_assert(KOKKOS_VERSION == 50202, "measurement is pinned to Kokkos 5.2.2");
    // An externally injected tool could replace callbacks or add fences.
    for (const char* name : {"KOKKOS_TOOLS_LIBS", "KOKKOS_PROFILE_LIBRARY", "KOKKOS_TOOLS_LIBRARY"}) {
        if (const char* value = std::getenv(name); value && *value) {
            std::cerr << "measurement rejects external Kokkos Tools library: " << name << '\n';
            return 2;
        }
    }
    install_observer();
    observed.stage("initialize");
    {
        Kokkos::ScopeGuard guard;
        const Kokkos::Serial exec;
        run(exec);
        assert(observed.live.empty());
        observed.stage("finalize");
    }
    assert(Kokkos::is_finalized());
    assert(!observed.failed && observed.live.empty() && observed.live_bytes == 0);
    assert(observed.allocated == observed.deallocated && observed.allocated > 0);
    assert(observed.copy_ends == observed.copies && !observed.copy_active);
    assert(observed.fence_ends == observed.explicit_fences + observed.internal_fences);
    assert(observed.open_fences.empty() && observed.open_kernels.empty());
    assert(observed.kernel_ends == observed.kernels && observed.finalizations == 1);
    // Independent source ledger for this fixed path: create 6 + pack 2 +
    // reorder 12 + receive 4 + commit 4 allocations. Eleven four-real arrays
    // contribute 44*sizeof(real); ID/index/CSR arrays contribute 480 bytes,
    // including three five-LocalIndex spans rounded from 20 to 24 by Kokkos.
    assert(observed.allocated == 28 && observed.allocated_bytes == 480 + 44 * sizeof(real));
    assert(observed.deallocated_bytes == observed.allocated_bytes);
    // Seven four-real copies/fills plus three four-uint64 ID copies.
    assert(observed.copies == 10 && observed.copy_bytes == 96 + 28 * sizeof(real));
    assert(observed.explicit_fences == 15 && observed.internal_fences == 28);
    // Pack, two field gathers, two CSR kernels, scalar fill, receive unpack.
    assert(observed.kernels == 7);
    // Peak: retained original + packed payload + all reorder intermediates
    // and new bank; seven four-real arrays and 384 bytes of ID/index/CSR.
    assert(observed.peak_bytes == 384 + 28 * sizeof(real));
    assert(observed.packed_peak == 32 + 4 * sizeof(real));
    print_report();
}
