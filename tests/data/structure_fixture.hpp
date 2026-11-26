#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <cassert>
#include <cmath>
#include <data/CsrAdjacency.hpp>
#include <data/FieldIssuer.hpp>
#include <validation/structure/StructureState.hpp>
#include <data/TransportBuilder.hpp>
#include <data/TransportReceiver.hpp>
#include <limits>
#include <memory>
#include <nexora/core/Real.hpp>
#include <span>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>

// Shared literal fixture and independent checks from structure_reorder_tests.
namespace nexora::test_fixture {
using namespace nexora;
using namespace nexora::detail;
constexpr std::array original_ids{GlobalEntityId{10}, GlobalEntityId{30},
                                  GlobalEntityId{20}, GlobalEntityId{40}};
constexpr std::array reordered_ids{GlobalEntityId{20}, GlobalEntityId{10},
                                   GlobalEntityId{40}, GlobalEntityId{30}};

inline void require_check(bool passed, const char* expression,
                          std::source_location where = std::source_location::current()) {
    if (!passed)
        throw std::runtime_error(std::string(where.file_name()) + ":" + std::to_string(where.line())
                                 + ": " + expression);
}
#define NEXORA_FIXTURE_CHECK(expression) require_check((expression), #expression)

template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw std::runtime_error(result.status().message());
    return std::move(result).value();
}
inline FieldDescriptor descriptor(unsigned id, bool ids = false) {
    FieldDescriptor result;
    result.id = FieldId{id};
    result.name = ids ? "stable IDs" : "structure field";
    result.domain_id = DomainId{1};
    result.association = Association::particle;
    result.entity_set = "structure";
    result.units = "1";
    if (ids)
        result.scalar_type = ScalarType::uint64;
    return result;
}
using nexora::reference::StructureState;
using nexora::reference::StructureConfig;
using Bank = nexora::reference::StructureBank;

inline StructureConfig fixture_config(LayoutVersion layout = LayoutVersion{7}) {
    StructureConfig config;
    config.entity_namespace = {DomainId{1}, Association::particle, "structure"};
    config.ids.assign(original_ids.begin(), original_ids.end());
    config.x = {1, 3, 2, 4};
    config.q = {100, 300, 200, 400};
    config.history = {-10, -30, -20, -40};
    config.offsets = {0, 2, 2, 3, 5};
    config.indices = {1, 2, 3, 0, 2};
    config.fields = {descriptor(1, true), descriptor(2), descriptor(3), descriptor(4)};
    config.stamp.topology = TopologyVersion{3};
    config.stamp.geometry = GeometryVersion{5};
    config.stamp.layout = layout;
    config.stamp.accepted_version = StateVersion{11};
    config.stamp.accepted = true;
    return config;
}

inline void expect_bank(const Bank& bank, bool reordered, bool committed = false) {
    // Independent literals, never derived from the permutation under test.
    const std::array<std::uint64_t, 4> original{10, 30, 20, 40}, next{20, 10, 40, 30};
    const std::array<real, 4> oq{100, 300, 200, 400}, nq{200, 100, 400, 300};
    const std::array<real, 4> ox{1, 3, 2, 4}, nx{2, 1, 4, 3};
    const std::array<real, 4> oh{-10, -30, -20, -40}, nh{-20, -10, -40, -30};
    const std::array<Offset, 5> oo{0, 2, 2, 3, 5}, no{0, 1, 3, 5, 5};
    const std::array<LocalIndex, 5> oi{1, 2, 3, 0, 2}, ni{2, 3, 0, 1, 0};
    const std::array<std::array<std::uint64_t, 2>, 5> oe{{{10, 30}, {10, 20}, {20, 40}, {40, 10}, {40, 20}}};
    const std::array<std::array<std::uint64_t, 2>, 5> ne{{{20, 40}, {10, 30}, {10, 20}, {40, 10}, {40, 20}}};
    for (std::size_t i = 0; i < 4; ++i) {
        NEXORA_FIXTURE_CHECK(bank.ids.grant.view()(i) == (reordered ? next : original)[i]);
        NEXORA_FIXTURE_CHECK(bank.x.grant.view()(i) == (reordered ? nx : ox)[i]);
        NEXORA_FIXTURE_CHECK(bank.q.grant.view()(i) == (committed ? std::array<real, 4>{21, 11, 41, 31} : (reordered ? nq : oq))[i]);
        NEXORA_FIXTURE_CHECK(bank.history.grant.view()(i) == (reordered ? nh : oh)[i]);
    }
    for (std::size_t i = 0; i < 5; ++i) {
        NEXORA_FIXTURE_CHECK(bank.adjacency.offsets()(i) == (reordered ? no : oo)[i]);
        NEXORA_FIXTURE_CHECK(bank.adjacency.indices()(i) == (reordered ? ni : oi)[i]);
    }
    std::size_t edge = 0;
    for (std::size_t row = 0; row < 4; ++row)
        for (Offset k = bank.adjacency.offsets()(row); k < bank.adjacency.offsets()(row + 1); ++k, ++edge) {
            NEXORA_FIXTURE_CHECK(bank.order.ids()(row) == (reordered ? ne : oe)[edge][0]);
            NEXORA_FIXTURE_CHECK(bank.order.ids()(bank.adjacency.indices()(k)) == (reordered ? ne : oe)[edge][1]);
        }
    NEXORA_FIXTURE_CHECK(bank.adjacency.rows().same_order(bank.order));
    NEXORA_FIXTURE_CHECK(bank.adjacency.columns().same_order(bank.order));
    NEXORA_FIXTURE_CHECK(bank.adjacency.topology() == bank.logical.topology);
    auto check_stamp = [&](const FieldStamp& field) {
        auto logical = field;
        NEXORA_FIXTURE_CHECK(logical.storage_incarnation != 0);
        logical.storage_incarnation = 0;
        NEXORA_FIXTURE_CHECK(logical == bank.logical);
    };
    check_stamp(bank.ids.grant.stamp());
    check_stamp(bank.x.grant.stamp());
    check_stamp(bank.q.grant.stamp());
    check_stamp(bank.history.grant.stamp());
}

inline TransportView<real> pack_bank(const Kokkos::Serial& exec, const Bank& bank) {
    const std::array axes{FieldAxis{4}};
    auto q = require(bank.q.issuer.read(bank.q.values, descriptor(3), bank.q.grant.stamp(),
                                       {Association::particle, "1", axes}));
    auto ids = require(bank.ids.issuer.read(bank.ids.values, descriptor(1, true), bank.ids.grant.stamp(),
                                           {Association::particle, "1", axes}));
    return require(TransportBuilder::prepare(exec, TransportPurpose::coupling, std::move(q),
                                            std::move(ids), TransportBuilder::SourceLifetime::until_packed));
}
inline void receive_reordered(const Kokkos::Serial& exec, const TransportView<real>& source, const Bank& target) {
    // A real ghost destination uses component/entity axes and LayoutLeft,
    // unlike the packed LayoutRight entity/component payload.
    Kokkos::View<real**, Kokkos::LayoutLeft, Kokkos::HostSpace> values(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "receive ghost"), 1, 4);
    Kokkos::deep_copy(exec, values, real{-999});
    auto stamp = target.logical;
    stamp.accepted = false;
    auto value_issuer = require(FieldIssuer::create_at_validated_time(values, stamp));
    auto ids = require(EntityOrder::copy_from(exec, target.order.entity_namespace(), stamp.layout,
                                              target.order.ids()));
    auto id_issuer = require(FieldIssuer::create_at_validated_time(ids.ids(), stamp));
    auto field = descriptor(3);
    field.ownership = FieldOwnership::ghost;
    field.access = FieldAccess::overwrite;
    const std::array value_axes{FieldAxis{1, true}, FieldAxis{4}};
    const std::array id_axes{FieldAxis{4}};
    const auto write = require(value_issuer.write(values, field, value_issuer.stamp(),
                                                  {Association::particle, "1", value_axes},
                                                  FieldAccess::overwrite));
    const auto read_ids = require(id_issuer.read(ids.ids(), descriptor(1, true), id_issuer.stamp(),
                                                 {Association::particle, "1", id_axes}));
    auto stale = source.stamp();
    ++stale.storage_incarnation;
    NEXORA_FIXTURE_CHECK(!TransportReceiver::unpack(exec, source, write, read_ids, stale).ok());
    for (std::size_t i = 0; i < 4; ++i)
        NEXORA_FIXTURE_CHECK(values(0, i) == real{-999});
    NEXORA_FIXTURE_CHECK(TransportReceiver::unpack(exec, source, write, read_ids, source.stamp()).ok());
    const std::array<real, 4> expected{200, 100, 400, 300};
    for (std::size_t i = 0; i < 4; ++i)
        NEXORA_FIXTURE_CHECK(values(0, i) == expected[i]);
}

#undef NEXORA_FIXTURE_CHECK
} // namespace nexora::test_fixture
