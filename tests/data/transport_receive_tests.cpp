#include <gtest/gtest.h>
#include <stdexcept>
#include <Kokkos_Core.hpp>
#include <nexora/core/Real.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <data/FieldIssuer.hpp>
#include <data/TransportBuilder.hpp>
#include <data/TransportReceiver.hpp>
#include <span>
#include <utility>

namespace {
using namespace nexora;
using nexora::detail::FieldAxis;
using nexora::detail::FieldBindingSpec;
using nexora::detail::FieldIssuer;
using nexora::detail::TransportBuilder;
using nexora::detail::TransportReceiver;
using Id = std::uint64_t;
using Ids = Kokkos::View<Id*, Kokkos::Serial>;
using Scalars = Kokkos::View<real*, Kokkos::Serial>;

template <class T> T take(Result<T> result) {
    if (!result.ok())
        throw std::runtime_error(result.status().message());
    return std::move(result).value();
}

FieldDescriptor field() {
    FieldDescriptor descriptor;
    descriptor.id = FieldId{7};
    descriptor.name = "displacement";
    descriptor.domain_id = DomainId{5};
    descriptor.association = Association::particle;
    descriptor.entity_set = "particles";
    descriptor.units = "m";
    descriptor.access = FieldAccess::overwrite;
    return descriptor;
}

FieldDescriptor id_field() {
    auto descriptor = field();
    descriptor.id = FieldId{8};
    descriptor.name = "stable IDs";
    descriptor.units = "1";
    descriptor.scalar_type = ScalarType::uint64;
    return descriptor;
}

FieldStamp trial_stamp() {
    FieldStamp stamp;
    stamp.topology = TopologyVersion{2};
    stamp.geometry = GeometryVersion{3};
    stamp.layout = LayoutVersion{4};
    stamp.accepted_version = StateVersion{5};
    stamp.trial = TrialEpoch{6, 7, 8, 9};
    stamp.time = TimeStamp{0.5, 6, 1, 9};
    stamp.valid_from = 0.25;
    stamp.valid_until = 0.75;
    return stamp;
}

Ids make_ids(std::span<const Id> values) {
    Ids ids("receive test IDs", values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        ids(i) = values[i];
    return ids;
}

template <class View>
TransportView<real> pack(const Kokkos::Serial& exec, TransportPurpose purpose, const View& values,
                           FieldDescriptor descriptor, std::span<const FieldAxis> axes,
                           const Ids& ids, FieldStamp stamp) {
    auto value_issuer = take(FieldIssuer::create_at_validated_time(values, stamp));
    auto id_issuer = take(FieldIssuer::create_at_validated_time(ids, stamp));
    const std::array id_axes{FieldAxis{ids.extent(0)}};
    auto read_values = take(value_issuer.read(values, descriptor, value_issuer.stamp(),
                                              {descriptor.association, descriptor.units, axes}));
    auto read_ids = take(id_issuer.read(ids, id_field(), id_issuer.stamp(),
                                       {Association::particle, "1", id_axes}));
    // Release grants after packing so another snapshot can stamp the same input Views.
    return take(TransportBuilder::prepare(exec, purpose, std::move(read_values),
                                          std::move(read_ids),
                                          TransportBuilder::SourceLifetime::until_packed));
}

TEST(TransportReceive, ValidationAndRetry) {
    const Kokkos::Serial exec;
    const std::array axes{FieldAxis{3}};
    const FieldBindingSpec value_spec{Association::particle, "m", axes};
    const FieldBindingSpec id_spec{Association::particle, "1", axes};
    const auto descriptor = field();
    const auto stamp = trial_stamp();

    const auto source_ids = make_ids(std::array<Id, 3>{41, 7, 29});
    Scalars source_values("coupling source", 3);
    source_values(0) = real{18};
    source_values(1) = real{-4};
    source_values(2) = real{9.5};
    const auto valid = pack(exec, TransportPurpose::coupling, source_values, descriptor, axes,
                            source_ids, stamp);

    Scalars target_values("trial target", 3);
    Kokkos::deep_copy(target_values, real{-77});
    const auto target_ids = make_ids(std::array<Id, 3>{29, 41, 7});
    auto target_stamp = stamp;
    target_stamp.layout = LayoutVersion{25};
    auto target_issuer = take(FieldIssuer::create_at_validated_time(target_values, target_stamp));
    auto id_issuer = take(FieldIssuer::create_at_validated_time(target_ids, target_stamp));
    auto target = take(target_issuer.write(target_values, descriptor, target_issuer.stamp(),
                                           value_spec, FieldAccess::overwrite));
    auto ids = take(id_issuer.read(target_ids, id_field(), id_issuer.stamp(), id_spec));

    auto expect_unchanged = [&] {
        for (std::size_t i = 0; i < target_values.extent(0); ++i)
            ASSERT_EQ(target.view()(i), real{-77});
    };
    auto reject_snapshot = [&](const TransportView<real>& source, const FieldStamp& expected) {
        ASSERT_FALSE(TransportReceiver::unpack(exec, source, target, ids, expected).ok());
        ASSERT_NO_FATAL_FAILURE(expect_unchanged());
    };

    auto stale_plan = valid.stamp();
    ++stale_plan.storage_incarnation;
    ASSERT_NO_FATAL_FAILURE(reject_snapshot(valid, stale_plan));

    auto stale_epoch = stamp;
    ++stale_epoch.trial.attempt;
    const auto stale = pack(exec, TransportPurpose::coupling, source_values, descriptor, axes,
                            source_ids, stale_epoch);
    ASSERT_NO_FATAL_FAILURE(reject_snapshot(stale, stale.stamp()));

    auto wrong_units = descriptor;
    wrong_units.units = "cm";
    const auto incompatible = pack(exec, TransportPurpose::coupling, source_values, wrong_units,
                                   axes, source_ids, stamp);
    ASSERT_NO_FATAL_FAILURE(reject_snapshot(incompatible, incompatible.stamp()));

    const auto visualization = pack(exec, TransportPurpose::visualization, source_values,
                                    descriptor, axes, source_ids, stamp);
    ASSERT_NO_FATAL_FAILURE(reject_snapshot(visualization, visualization.stamp()));
    const auto halo = pack(exec, TransportPurpose::halo, source_values, descriptor, axes,
                           source_ids, stamp);
    ASSERT_NO_FATAL_FAILURE(reject_snapshot(halo, halo.stamp()));

    const auto wrong_ids = make_ids(std::array<Id, 3>{41, 7, 999});
    const auto missing_id = pack(exec, TransportPurpose::coupling, source_values, descriptor, axes,
                                 wrong_ids, stamp);
    ASSERT_NO_FATAL_FAILURE(reject_snapshot(missing_id, missing_id.stamp()));

    auto short_ids = Kokkos::subview(target_ids, std::pair<std::size_t, std::size_t>{0, 2});
    const std::array short_axes{FieldAxis{2}};
    auto wrong_shape = take(id_issuer.read(short_ids, id_field(), id_issuer.stamp(),
                                           {Association::particle, "1", short_axes}));
    ASSERT_FALSE(TransportReceiver::unpack(exec, valid, target, wrong_shape, valid.stamp()).ok());
    ASSERT_NO_FATAL_FAILURE(expect_unchanged());

    const auto stale_target_ids = make_ids(std::array<Id, 3>{29, 41, 7});
    auto stale_id_stamp = target_stamp;
    stale_id_stamp.layout = LayoutVersion{26};
    auto stale_id_issuer = take(FieldIssuer::create_at_validated_time(stale_target_ids, stale_id_stamp));
    auto stale_ids = take(stale_id_issuer.read(stale_target_ids, id_field(),
                                               stale_id_issuer.stamp(), id_spec));
    ASSERT_FALSE(TransportReceiver::unpack(exec, valid, target, stale_ids, valid.stamp()).ok());
    ASSERT_NO_FATAL_FAILURE(expect_unchanged());

    const auto duplicate_ids = make_ids(std::array<Id, 3>{29, 41, 41});
    auto duplicate_issuer = take(FieldIssuer::create_at_validated_time(duplicate_ids, target_stamp));
    auto duplicate = take(duplicate_issuer.read(duplicate_ids, id_field(), duplicate_issuer.stamp(),
                                                id_spec));
    ASSERT_FALSE(TransportReceiver::unpack(exec, valid, target, duplicate, valid.stamp()).ok());
    ASSERT_NO_FATAL_FAILURE(expect_unchanged());

    ASSERT_TRUE(TransportReceiver::unpack(exec, valid, target, ids, valid.stamp()).ok());
    ASSERT_TRUE(target.view()(0) == real{9.5} && target.view()(1) == real{18} &&
           target.view()(2) == real{-4});
}

TEST(TransportReceive, GhostSubview) {
    const Kokkos::Serial exec;
    auto stamp = trial_stamp();
    stamp.trial = {};
    Kokkos::View<real**, Kokkos::LayoutRight, Kokkos::Serial> source("halo source", 3, 2);
    for (std::size_t entity = 0; entity < 3; ++entity) {
        source(entity, 0) = static_cast<real>(entity + 1);
        source(entity, 1) = static_cast<real>(entity + 10);
    }
    const auto source_ids = make_ids(std::array<Id, 3>{41, 7, 29});
    const std::array source_axes{FieldAxis{3}, FieldAxis{2, true}};
    auto descriptor = field();
    descriptor.components = 2;
    const auto snapshot = pack(exec, TransportPurpose::halo, source, descriptor, source_axes,
                               source_ids, stamp);
    const auto one_component = Kokkos::subview(source, Kokkos::ALL, 0);
    const std::array id_axes{FieldAxis{3}};
    const auto wrong_components = pack(exec, TransportPurpose::halo, one_component, field(),
                                        id_axes, source_ids, stamp);

    Kokkos::View<real**, Kokkos::LayoutRight, Kokkos::Serial> backing("ghost storage", 2, 5);
    Kokkos::deep_copy(backing, real{-99});
    const auto ghosts =
        Kokkos::subview(backing, Kokkos::ALL, std::pair<std::size_t, std::size_t>{1, 4});
    const auto target_ids = make_ids(std::array<Id, 3>{7, 29, 41});
    const std::array target_axes{FieldAxis{2, true}, FieldAxis{3}};
    stamp.layout = LayoutVersion{26};
    auto target_issuer = take(FieldIssuer::create_at_validated_time(backing, stamp));
    auto id_issuer = take(FieldIssuer::create_at_validated_time(target_ids, stamp));
    descriptor.ownership = FieldOwnership::ghost;
    auto target = take(target_issuer.write(ghosts, descriptor, target_issuer.stamp(),
                                           {Association::particle, "m", target_axes},
                                           FieldAccess::overwrite));
    auto ids = take(id_issuer.read(target_ids, id_field(), id_issuer.stamp(),
                                   {Association::particle, "1", id_axes}));

    ASSERT_TRUE(!TransportReceiver::unpack(exec, wrong_components, target, ids,
                                      wrong_components.stamp()).ok());
    for (std::size_t component = 0; component < backing.extent(0); ++component)
        for (std::size_t entity = 0; entity < backing.extent(1); ++entity)
            ASSERT_EQ(backing(component, entity), real{-99});

    ASSERT_TRUE(TransportReceiver::unpack(exec, snapshot, target, ids, snapshot.stamp()).ok());
    ASSERT_TRUE(backing(0, 1) == real{2} && backing(0, 2) == real{3} && backing(0, 3) == real{1});
    ASSERT_TRUE(backing(1, 1) == real{11} && backing(1, 2) == real{12} && backing(1, 3) == real{10});
    ASSERT_TRUE(backing(0, 0) == real{-99} && backing(0, 4) == real{-99});
    ASSERT_TRUE(backing(1, 0) == real{-99} && backing(1, 4) == real{-99});
}
} // namespace
