#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <array>
#include <cstdint>
#include <limits>
#include <nexora/data/FieldDescriptor.hpp>
#include <span>
#include <utility>
#include <data/CsrAdjacency.hpp>
#include <data/EntityCapacity.hpp>
#include <data/EntityOrder.hpp>
#include <data/Permutation.hpp>

namespace {
using namespace nexora;
using namespace nexora::detail;
using Ids = std::array<GlobalEntityId, 4>;

EntityNamespace particles(DomainId domain = DomainId{1},
                          Association association = Association::particle,
                          std::string set = "particles") {
    return {domain, association, std::move(set)};
}

TEST(EntityTopology, OrderAndPermutation) {
    const Kokkos::Serial exec;
    const auto ns = particles();
    Ids old_ids{GlobalEntityId{10}, GlobalEntityId{30}, GlobalEntityId{20},
                GlobalEntityId{40}};
    const Ids new_ids{GlobalEntityId{20}, GlobalEntityId{10}, GlobalEntityId{40},
                      GlobalEntityId{30}};
    const auto old_order = EntityOrder::create(exec, ns, LayoutVersion{3}, old_ids);
    const auto new_order = EntityOrder::create(exec, ns, LayoutVersion{4}, new_ids);
    ASSERT_TRUE(old_order.ok() && new_order.ok());
    old_ids[0] = GlobalEntityId{999}; // Order owns a stable snapshot, not the caller's span.
    const auto& old = old_order.value();
    const auto& next = new_order.value();
    ASSERT_TRUE(old.validate_for(ns, LayoutVersion{3}).ok());
    ASSERT_FALSE(old.validate_for(ns, LayoutVersion{4}).ok());
    const std::array<std::uint64_t, 4> expected_old_ids{10, 30, 20, 40};
    for (std::size_t i = 0; i < expected_old_ids.size(); ++i)
        ASSERT_EQ(old.ids()(i), expected_old_ids[i]);

    const auto permutation = Permutation::between(exec, old, next);
    ASSERT_TRUE(permutation.ok());
    const std::array<LocalIndex, 4> expected_new_to_old{2, 0, 3, 1};
    const std::array<LocalIndex, 4> expected_old_to_new{1, 3, 0, 2};
    for (std::size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(permutation.value().new_to_old()(i), expected_new_to_old[i]);
        ASSERT_EQ(permutation.value().old_to_new()(i), expected_old_to_new[i]);
    }
    ASSERT_TRUE(permutation.value().validate_for(old, next).ok());

    // Independent gather check for field alignment under this permutation.
    const std::array<int, 4> old_field{100, 300, 200, 400};
    std::array<int, 4> gathered{};
    for (std::size_t i = 0; i < gathered.size(); ++i)
        gathered[i] = old_field[permutation.value().new_to_old()(i)];
    ASSERT_TRUE((gathered == std::array<int, 4>{200, 100, 400, 300}));

    const auto wrong_domain = EntityOrder::create(exec, particles(DomainId{2}),
                                                   LayoutVersion{5}, new_ids);
    ASSERT_TRUE(wrong_domain.ok());
    ASSERT_FALSE(Permutation::between(exec, old, wrong_domain.value()).ok());
    const auto duplicate = EntityOrder::create(
        exec, ns, LayoutVersion{5}, std::array{GlobalEntityId{10}, GlobalEntityId{10}});
    ASSERT_FALSE(duplicate.ok());
    const auto missing = EntityOrder::create(
        exec, ns, LayoutVersion{5}, std::array{GlobalEntityId{10}, GlobalEntityId{30},
                                               GlobalEntityId{20}, GlobalEntityId{99}});
    ASSERT_TRUE(missing.ok());
    ASSERT_FALSE(Permutation::between(exec, old, missing.value()).ok());
    const auto same_version_wrong_order = EntityOrder::create(
        exec, ns, LayoutVersion{3}, std::array{GlobalEntityId{30}, GlobalEntityId{10},
                                               GlobalEntityId{20}, GlobalEntityId{40}});
    ASSERT_TRUE(same_version_wrong_order.ok());
    ASSERT_FALSE(Permutation::between(exec, old, same_version_wrong_order.value()).ok());
    const std::array<LocalIndex, 4> bad_forward{2, 0, 3, 3};
    const std::array<LocalIndex, 4> bad_reverse{1, 3, 0, 2};
    ASSERT_FALSE(Permutation::create(exec, old, next, bad_forward, bad_reverse).ok());
    const std::array<LocalIndex, 4> out_of_range{2, 0, 3, 4};
    ASSERT_FALSE(Permutation::create(exec, old, next, out_of_range, expected_old_to_new).ok());

    const auto empty = EntityOrder::create(exec, ns, LayoutVersion{6}, std::span<const GlobalEntityId>{});
    ASSERT_TRUE(empty.ok() && empty.value().size() == 0);
    const auto empty_perm = Permutation::between(exec, empty.value(), empty.value());
    ASSERT_TRUE(empty_perm.ok());
    ASSERT_TRUE(!EntityOrder::create(exec, particles(DomainId{1}, Association::particle, ""),
                                LayoutVersion{6}, std::span<const GlobalEntityId>{})
                .ok());

    Kokkos::View<std::uint64_t**, Kokkos::LayoutRight, Kokkos::HostSpace> strided_backing(
        "strided entity IDs", 4, 2);
    for (std::size_t i = 0; i < 4; ++i) {
        strided_backing(i, 0) = static_cast<std::uint64_t>(100 + 2 * i);
        strided_backing(i, 1) = static_cast<std::uint64_t>(101 + 2 * i);
    }
    const auto every_other = Kokkos::subview(strided_backing, Kokkos::ALL, 0);
    static_assert(decltype(every_other)::rank() == 1);
    const auto copied = EntityOrder::copy_from(exec, particles(), LayoutVersion{17}, every_other);
    ASSERT_TRUE(copied.ok() && copied.value().size() == 4);
    strided_backing(0, 0) = 999;
    strided_backing(1, 0) = 998;
    ASSERT_TRUE(copied.value().ids()(0) == 100 && copied.value().ids()(1) == 102 &&
           copied.value().ids()(2) == 104 && copied.value().ids()(3) == 106);
    Kokkos::View<std::uint64_t**, Kokkos::Serial> wrong_rank("rank two IDs", 2, 2);
    Kokkos::View<std::int32_t*, Kokkos::Serial> wrong_dtype("wrong type IDs", 2);
    ASSERT_FALSE(EntityOrder::copy_from(exec, particles(), LayoutVersion{18}, wrong_rank).ok());
    ASSERT_FALSE(EntityOrder::copy_from(exec, particles(), LayoutVersion{18}, wrong_dtype).ok());

    auto supplied_forward = expected_new_to_old;
    auto supplied_inverse = expected_old_to_new;
    const auto supplied = Permutation::create(exec, old, next, supplied_forward, supplied_inverse);
    ASSERT_TRUE(supplied.ok());
    supplied_forward.fill(0);
    supplied_inverse.fill(0);
    for (std::size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(supplied.value().new_to_old()(i), expected_new_to_old[i]);
        ASSERT_EQ(supplied.value().old_to_new()(i), expected_old_to_new[i]);
    }
}

TEST(EntityTopology, Csr) {
    const Kokkos::Serial exec;
    const auto ns = particles();
    const auto row_order = EntityOrder::create(
        exec, ns, LayoutVersion{7}, std::array{GlobalEntityId{10}, GlobalEntityId{30},
                                               GlobalEntityId{20}, GlobalEntityId{40}});
    const auto reordered = EntityOrder::create(
        exec, ns, LayoutVersion{8}, std::array{GlobalEntityId{20}, GlobalEntityId{10},
                                               GlobalEntityId{40}, GlobalEntityId{30}});
    ASSERT_TRUE(row_order.ok() && reordered.ok());
    std::array<Offset, 5> offsets{0, 2, 2, 3, 5};
    std::array<LocalIndex, 5> indices{1, 2, 3, 0, 2};
    const auto graph = CsrAdjacency::create(exec, row_order.value(), row_order.value(),
                                            TopologyVersion{9}, offsets, indices);
    ASSERT_TRUE(graph.ok());
    offsets[1] = 0;
    indices[0] = 3; // CSR owns snapshots of both caller spans.
    const std::array<Offset, 5> expected_offsets{0, 2, 2, 3, 5};
    const std::array<LocalIndex, 5> expected_indices{1, 2, 3, 0, 2};
    for (std::size_t i = 0; i < 5; ++i) {
        ASSERT_EQ(graph.value().offsets()(i), expected_offsets[i]);
        ASSERT_EQ(graph.value().indices()(i), expected_indices[i]);
    }
    const std::array<Offset, 5> reordered_offsets{0, 1, 3, 5, 5};
    const std::array<LocalIndex, 5> reordered_indices{2, 3, 0, 1, 0};
    const auto reordered_graph = CsrAdjacency::create(
        exec, reordered.value(), reordered.value(), TopologyVersion{9}, reordered_offsets,
        reordered_indices);
    ASSERT_TRUE(reordered_graph.ok());
    // Explicit stable-ID edge list: (20,40), (10,30), (10,20), (40,10), (40,20).
    const std::array<std::array<std::uint64_t, 2>, 5> expected_stable_edges{{
        {20, 40}, {10, 30}, {10, 20}, {40, 10}, {40, 20}}};
    const std::array<std::uint64_t, 4> reordered_row_ids{20, 10, 40, 30};
    for (std::size_t row = 0, edge = 0; row < reordered_row_ids.size(); ++row) {
        for (Offset k = reordered_graph.value().offsets()(row);
             k < reordered_graph.value().offsets()(row + 1); ++k, ++edge) {
            ASSERT_EQ(reordered_row_ids[row], expected_stable_edges[edge][0]);
            const auto col = reordered_graph.value().indices()(k);
            ASSERT_EQ(reordered.value().ids()(col), expected_stable_edges[edge][1]);
        }
    }
    ASSERT_TRUE(graph.value().validate_for(row_order.value(), row_order.value(), TopologyVersion{9}).ok());
    ASSERT_FALSE(graph.value().validate_for(reordered.value(), row_order.value(), TopologyVersion{9}).ok());
    ASSERT_FALSE(graph.value().validate_for(row_order.value(), row_order.value(), TopologyVersion{10}).ok());
    const auto same_version_different_order = EntityOrder::create(
        exec, ns, LayoutVersion{7}, std::array{GlobalEntityId{30}, GlobalEntityId{10},
                                               GlobalEntityId{20}, GlobalEntityId{40}});
    ASSERT_TRUE(same_version_different_order.ok());
    ASSERT_TRUE(!graph.value().validate_for(same_version_different_order.value(), row_order.value(),
                                       TopologyVersion{9}).ok());

    const std::array<Offset, 5> decreasing{0, 2, 1, 3, 5};
    ASSERT_TRUE(!CsrAdjacency::create(exec, row_order.value(), row_order.value(), TopologyVersion{9},
                                 decreasing, indices).ok());
    const std::array<Offset, 5> wrong_end{0, 2, 2, 3, 4};
    ASSERT_TRUE(!CsrAdjacency::create(exec, row_order.value(), row_order.value(), TopologyVersion{9},
                                 wrong_end, indices).ok());
    const std::array<LocalIndex, 5> bad_column{1, 2, 3, 0, 4};
    ASSERT_TRUE(!CsrAdjacency::create(exec, row_order.value(), row_order.value(), TopologyVersion{9},
                                 offsets, bad_column).ok());
    ASSERT_TRUE(!CsrAdjacency::create(exec, row_order.value(), row_order.value(), TopologyVersion{9},
                                 std::array<Offset, 4>{0, 2, 3, 5}, indices).ok());
    ASSERT_TRUE(!CsrAdjacency::create(exec, row_order.value(), row_order.value(), TopologyVersion{9},
                                 std::array<Offset, 5>{1, 2, 2, 3, 5}, indices).ok());

    const auto empty_rows = EntityOrder::create(exec, ns, LayoutVersion{11},
                                                std::span<const GlobalEntityId>{});
    const auto empty_graph = CsrAdjacency::create(exec, empty_rows.value(), empty_rows.value(),
                                                  TopologyVersion{12}, std::array<Offset, 1>{0},
                                                  std::span<const LocalIndex>{});
    ASSERT_TRUE(empty_graph.ok() && empty_graph.value().offsets()(0) == 0);

    const auto col_ns = particles(DomainId{2}, Association::node, "nodes");
    const auto columns = EntityOrder::create(
        exec, col_ns, LayoutVersion{13}, std::array{GlobalEntityId{5}, GlobalEntityId{6}});
    const auto rows = EntityOrder::create(exec, ns, LayoutVersion{13},
                                          std::array{GlobalEntityId{50}, GlobalEntityId{60}});
    ASSERT_TRUE(columns.ok() && rows.ok());
    const auto bipartite = CsrAdjacency::create(exec, rows.value(), columns.value(),
                                                TopologyVersion{14},
                                                std::array<Offset, 3>{0, 1, 3},
                                                std::array<LocalIndex, 3>{1, 0, 1});
    ASSERT_TRUE(bipartite.ok());
    const auto stale_columns = EntityOrder::create(
        exec, col_ns, LayoutVersion{14}, std::array{GlobalEntityId{5}, GlobalEntityId{6}});
    ASSERT_TRUE(stale_columns.ok());
    ASSERT_TRUE(!bipartite.value().validate_for(rows.value(), stale_columns.value(),
                                           TopologyVersion{14}).ok());
    ASSERT_TRUE(!CsrAdjacency::create(exec, rows.value(), columns.value(), TopologyVersion{14},
                                 std::array<Offset, 3>{0, 1, 3},
                                 std::array<LocalIndex, 3>{1, 0, 2}).ok());

    const auto one_row = EntityOrder::create(exec, ns, LayoutVersion{15},
                                             std::array{GlobalEntityId{70}});
    const auto long_columns = EntityOrder::create(
        exec, col_ns, LayoutVersion{15},
        std::array{GlobalEntityId{5}, GlobalEntityId{6}, GlobalEntityId{7}});
    std::array<LocalIndex, 8193> long_indices{};
    for (std::size_t i = 0; i < long_indices.size(); ++i)
        long_indices[i] = static_cast<LocalIndex>(i % 3);
    const auto long_row = CsrAdjacency::create(exec, one_row.value(), long_columns.value(),
                                               TopologyVersion{16},
                                               std::array<Offset, 2>{0, long_indices.size()},
                                               long_indices);
    ASSERT_TRUE(long_row.ok());
}

TEST(EntityTopology, CapacityAndVersions) {
    const auto max_count = static_cast<std::uint64_t>(std::numeric_limits<LocalIndex>::max());
    ASSERT_TRUE(checked_entity_count(max_count).ok());
    ASSERT_FALSE(checked_entity_count(max_count + 1).ok());
    ASSERT_TRUE(checked_local_index(max_count - 1, max_count).ok());
    ASSERT_FALSE(checked_local_index(max_count, max_count).ok());
    ASSERT_FALSE(checked_local_index(0, 0).ok());
    ASSERT_TRUE(checked_array_bytes(3, sizeof(Offset)).ok());
    ASSERT_TRUE(!checked_array_bytes(std::numeric_limits<std::uint64_t>::max(),
                                std::numeric_limits<std::size_t>::max()).ok());
    ASSERT_FALSE(checked_csr_capacity(max_count, max_count, std::numeric_limits<std::uint64_t>::max()).ok());
    const auto max_entries_without_multiply_overflow =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(LocalIndex));
    ASSERT_FALSE(checked_csr_capacity(1, 1, max_entries_without_multiply_overflow).ok());
    ASSERT_FALSE(checked_csr_capacity(std::numeric_limits<std::uint64_t>::max(), 0, 0).ok());
    ASSERT_FALSE(checked_next_version(LayoutVersion{std::numeric_limits<std::uint64_t>::max()}).ok());
    ASSERT_FALSE(checked_increment(std::numeric_limits<std::uint64_t>::max()).ok());
}
} // namespace
