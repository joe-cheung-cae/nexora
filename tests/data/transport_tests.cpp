#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <nexora/core/Real.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <data/ExternalField.hpp>
#include <data/FieldIssuer.hpp>
#include <data/TransportBuilder.hpp>
#include <limits>
#include <memory>
#include <nexora/data/TransportView.hpp>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
using namespace nexora;
using nexora::detail::ExternalField;
using nexora::detail::FieldAxis;
using nexora::detail::FieldBindingSpec;
using nexora::detail::FieldIssuer;
using nexora::detail::TransportBuilder;
using Id = std::uint64_t;
using Snapshot = TransportView<real>;

template <class View>
concept HasRvalueValues = requires(View view) { std::move(view).values(); };

static_assert(std::is_move_constructible_v<Snapshot> && !std::is_copy_constructible_v<Snapshot>);
static_assert(!HasRvalueValues<Snapshot>);
static_assert(
    !std::is_assignable_v<decltype(std::declval<const Snapshot&>().values()(0, 0)), real>);
static_assert(!std::is_assignable_v<decltype(std::declval<const Snapshot&>().stable_ids()(0)), Id>);
using PackedValues = decltype(std::declval<const Snapshot&>().values());
static_assert(std::is_same_v<typename PackedValues::array_layout, Kokkos::LayoutRight>);

template <class T> T take(Result<T> result) {
    if (!result.ok())
        throw std::runtime_error(result.status().message());
    return std::move(result).value();
}

FieldDescriptor field(FieldOwnership ownership = FieldOwnership::owned,
                      std::size_t components = 1) {
    FieldDescriptor out;
    out.id = FieldId{7};
    out.name = "displacement";
    out.domain_id = DomainId{5};
    out.association = Association::particle;
    out.entity_set = "particles";
    out.units = "m";
    out.components = components;
    out.ownership = ownership;
    out.access = FieldAccess::overwrite;
    return out;
}

FieldDescriptor id_field(FieldOwnership ownership = FieldOwnership::owned) {
    auto out = field(ownership);
    out.id = FieldId{8};
    out.name = "particle ids";
    out.units = "1";
    out.scalar_type = ScalarType::uint64;
    return out;
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

TEST(Transport, ExternalValidation) {
    auto storage = std::make_shared<real[]>(6);
    auto* base = storage.get();
    const auto bytes = 6 * sizeof(real);
    const std::shared_ptr<const void> owner(storage, base);
    using External = ExternalField<real>;
    ASSERT_FALSE(External::create(base, 3, 2, bytes, {}, trial_stamp()).ok());
    const std::shared_ptr<const void> unowned(std::shared_ptr<const void>{}, base);
    ASSERT_TRUE(unowned.get() == base && unowned.use_count() == 0);
    ASSERT_FALSE(External::create(base, 3, 2, bytes, unowned, trial_stamp()).ok());
    ASSERT_FALSE(External::create(base + 1, 3, 1, bytes, owner, trial_stamp()).ok());
    ASSERT_FALSE(External::create(nullptr, 3, 1, bytes, owner, trial_stamp()).ok());
    ASSERT_FALSE(External::create(base, 3, 0, bytes, owner, trial_stamp()).ok());
    ASSERT_FALSE(External::create(base, 3, 2, 5 * sizeof(real) - 1, owner, trial_stamp()).ok());
    const auto max_size = std::numeric_limits<std::size_t>::max();
    ASSERT_FALSE(External::create(base, 3, max_size, bytes, owner, trial_stamp()).ok());
    ASSERT_FALSE(External::create(base, 1, 1, max_size, owner, trial_stamp()).ok());

    auto* misaligned = reinterpret_cast<std::byte*>(base) + 1;
    const std::shared_ptr<const void> wrong_alignment(storage, misaligned);
    ASSERT_TRUE(!External::create(reinterpret_cast<real*>(misaligned), 1, 1, bytes - 1,
                             wrong_alignment, trial_stamp()).ok());

    const std::array axes{FieldAxis{3}};
    const FieldBindingSpec spec{Association::particle, "m", axes};
    const auto descriptor = field(FieldOwnership::external);
    auto invalid_stamp = trial_stamp();
    invalid_stamp.valid_until = 0.0;
    EXPECT_EQ(External::create(base, 3, 2, bytes, owner, invalid_stamp).status().code(),
              StatusCode::invalid_argument);
    invalid_stamp = trial_stamp();
    invalid_stamp.time.physical_time = std::numeric_limits<double>::infinity();
    EXPECT_EQ(External::create(base, 3, 2, bytes, owner, invalid_stamp).status().code(),
              StatusCode::invalid_argument);

    auto imported = take(External::create(base, 3, 2, bytes, owner, trial_stamp()));
    ASSERT_TRUE(External::create(base, 3, 2, bytes, owner, trial_stamp()).status().code() ==
           StatusCode::already_exists);
    const std::shared_ptr<const void> overlapping(storage, base + 1);
    ASSERT_TRUE(!External::create(base + 1, 3, 1, bytes - sizeof(real), overlapping, trial_stamp())
                .ok());
    auto retaining_owner = std::make_shared<decltype(storage)>(storage);
    const std::shared_ptr<const void> independent_owner(retaining_owner, base + 1);
    ASSERT_TRUE(owner.owner_before(independent_owner) || independent_owner.owner_before(owner));
    ASSERT_TRUE(External::create(base + 1, 3, 1, bytes - sizeof(real), independent_owner,
                            trial_stamp()).status().code() == StatusCode::already_exists);

    auto owned_descriptor = descriptor;
    owned_descriptor.ownership = FieldOwnership::owned;
    ASSERT_FALSE(imported.read(owned_descriptor, imported.stamp(), spec).ok());
    ASSERT_FALSE(imported.read(descriptor, trial_stamp(), spec).ok());
    auto wrong_dtype = descriptor;
    wrong_dtype.scalar_type =
        real_scalar_type == ScalarType::float32 ? ScalarType::float64 : ScalarType::float32;
    ASSERT_FALSE(imported.read(wrong_dtype, imported.stamp(), spec).ok());

    auto accepted = trial_stamp();
    accepted.accepted = true;
    accepted.trial = {};
    auto accepted_storage = std::make_shared<real[]>(3);
    auto accepted_field = take(External::create(accepted_storage.get(), 3, 1, 3 * sizeof(real),
                                                accepted_storage, accepted));
    ASSERT_TRUE(accepted_field.read(descriptor, accepted_field.stamp(), spec).ok());
    ASSERT_TRUE(accepted_field.write(descriptor, accepted_field.stamp(), spec, FieldAccess::overwrite)
               .status().code() == StatusCode::invalid_state);

    const auto const_storage = std::make_shared<const real>(real{4});
    auto readonly = take(ExternalField<const real>::create(const_storage.get(), 1, 1,
                                                              sizeof(real), const_storage,
                                                              trial_stamp()));
    const std::array scalar_axis{FieldAxis{1}};
    auto read = take(readonly.read(descriptor, readonly.stamp(),
                                   {Association::particle, "m", scalar_axis}));
    ASSERT_EQ(read.view()(0), real{4});
}

TEST(Transport, ManagedExternalOverlap) {
    using Managed = Kokkos::View<real*, Kokkos::Serial>;
    auto storage = std::make_shared<Managed>("shared allocation", 4);
    auto* const data = storage->data();
    const std::shared_ptr<const void> alias(storage, data);
    {
        auto managed = take(FieldIssuer::create_at_validated_time(*storage, trial_stamp()));
        ASSERT_TRUE(!ExternalField<real>::create(data, 4, 1, 4 * sizeof(real), alias,
                                              trial_stamp()).ok());
    }
    {
        auto external = take(ExternalField<real>::create(data, 4, 1, 4 * sizeof(real), alias,
                                                           trial_stamp()));
        ASSERT_FALSE(FieldIssuer::create_at_validated_time(*storage, trial_stamp()).ok());
    }
    ASSERT_TRUE(FieldIssuer::create_at_validated_time(*storage, trial_stamp()).ok());
}

TEST(Transport, ConsumerCompletion) {
    const Kokkos::Serial exec;
    int value_releases = 0;
    int id_releases = 0;
    std::optional<Snapshot> fast;
    std::optional<Snapshot> slow;
    const std::array axes{FieldAxis{3}};
    const FieldBindingSpec value_spec{Association::particle, "m", axes};
    const FieldBindingSpec ids_spec{Association::particle, "1", axes};
    const auto value_descriptor = field(FieldOwnership::external);
    const auto ids_descriptor = id_field(FieldOwnership::external);
    {
        auto value_owner = std::shared_ptr<real[]>(new real[6]{3, -1, 6, -1, 9, -1},
                                                   [&](real* p) {
                                                       ++value_releases;
                                                       delete[] p;
                                                   });
        auto id_owner = std::shared_ptr<Id[]>(new Id[3]{41, 7, 29}, [&](Id* p) {
            ++id_releases;
            delete[] p;
        });
        auto values = take(ExternalField<real>::create(value_owner.get(), 3, 2,
                                                         6 * sizeof(real), value_owner,
                                                         trial_stamp()));
        auto ids = take(ExternalField<Id>::create(id_owner.get(), 3, 1, 3 * sizeof(Id), id_owner,
                                                 trial_stamp()));
        value_owner.reset();
        id_owner.reset();
        ASSERT_TRUE(value_releases == 0 && id_releases == 0);

        auto start_consumer = [&](TransportPurpose purpose) {
            auto [value_read, ids_read] = take(FieldIssuer::grant(
                values.request_read(value_descriptor, values.stamp(), value_spec),
                ids.request_read(ids_descriptor, ids.stamp(), ids_spec)));
            return take(TransportBuilder::prepare(exec, purpose, std::move(value_read),
                                                   std::move(ids_read)));
        };
        fast.emplace(start_consumer(TransportPurpose::coupling));
        slow.emplace(start_consumer(TransportPurpose::visualization));
        ASSERT_TRUE(fast->descriptor().id == value_descriptor.id && fast->stamp() == values.stamp());
        ASSERT_EQ(fast->purpose(), TransportPurpose::coupling);
        ASSERT_EQ(slow->purpose(), TransportPurpose::visualization);
        ASSERT_TRUE(fast->values().extent(0) == 3 && fast->values().extent(1) == 1);
        ASSERT_TRUE(fast->values()(0, 0) == real{3} && fast->values()(1, 0) == real{6} &&
               fast->values()(2, 0) == real{9});
        ASSERT_TRUE(fast->stable_ids()(0) == 41 && fast->stable_ids()(1) == 7 &&
               fast->stable_ids()(2) == 29);

        auto writer_blocked = [&] {
            const auto attempted = values.write(value_descriptor, values.stamp(), value_spec,
                                                FieldAccess::overwrite);
            return attempted.status().code() == StatusCode::invalid_state;
        };
        ASSERT_TRUE(writer_blocked());
        fast->complete();
        fast->complete();
        ASSERT_TRUE(!*fast && writer_blocked());
        ASSERT_TRUE(ids.write(ids_descriptor, ids.stamp(), ids_spec, FieldAccess::overwrite)
                   .status().code() == StatusCode::invalid_state);
        slow.reset(); // The consumer record also completes when its final owner leaves scope.
        {
            auto writer = take(values.write(value_descriptor, values.stamp(), value_spec,
                                             FieldAccess::overwrite));
            const auto writable = writer.view();
            Kokkos::parallel_for("update external values",
                Kokkos::RangePolicy<Kokkos::Serial>(exec, 0, 3),
                KOKKOS_LAMBDA(const int index) { writable(index) += real{1}; });
            exec.fence("external values ready");
            ASSERT_FALSE(values.read(value_descriptor, values.stamp(), value_spec).ok());
        }
        ASSERT_TRUE(ids.write(ids_descriptor, ids.stamp(), ids_spec, FieldAccess::overwrite).ok());

        fast.emplace(start_consumer(TransportPurpose::checkpoint));
        slow.emplace(start_consumer(TransportPurpose::halo));
        values.close();
        ids.close();
        ASSERT_FALSE(values.read(value_descriptor, values.stamp(), value_spec).ok());
        ASSERT_FALSE(ids.read(ids_descriptor, ids.stamp(), ids_spec).ok());
        ASSERT_TRUE(value_releases == 0 && id_releases == 0);
    }
    fast->complete();
    ASSERT_TRUE(value_releases == 0 && id_releases == 0);
    ASSERT_TRUE(slow->values()(0, 0) == real{4} && slow->values()(1, 0) == real{7} &&
           slow->values()(2, 0) == real{10});
    ASSERT_TRUE(slow->stable_ids()(0) == 41 && slow->stable_ids()(1) == 7 &&
           slow->stable_ids()(2) == 29);
    slow->complete();
    ASSERT_TRUE(value_releases == 1 && id_releases == 1);
    bool completed_view_rejected = false;
    try {
        (void)slow->values();
    } catch (const std::logic_error&) {
        completed_view_rejected = true;
    }
    ASSERT_TRUE(completed_view_rejected);
    slow.reset();
    fast.reset();
    ASSERT_TRUE(value_releases == 1 && id_releases == 1);
}

TEST(Transport, PackComponents) {
    const Kokkos::Serial exec;
    Kokkos::View<real**, Kokkos::LayoutRight, Kokkos::Serial> values("component first", 2, 3);
    Kokkos::View<Id*, Kokkos::Serial> ids("particle ids", 3);
    for (std::size_t entity = 0; entity < 3; ++entity) {
        ids(entity) = 20 - entity;
        values(0, entity) = static_cast<real>(entity + 1);
        values(1, entity) = static_cast<real>(10 + entity);
    }
    const std::array values_axes{FieldAxis{2, true}, FieldAxis{3}};
    const std::array id_axes{FieldAxis{3}};
    auto values_issuer = take(FieldIssuer::create_at_validated_time(values, trial_stamp()));
    auto ids_issuer = take(FieldIssuer::create_at_validated_time(ids, trial_stamp()));
    auto packed = take(TransportBuilder::prepare(
        exec, TransportPurpose::coupling,
        take(values_issuer.read(values, field(FieldOwnership::owned, 2), values_issuer.stamp(),
                                {Association::particle, "m", values_axes})),
        take(ids_issuer.read(ids, id_field(), ids_issuer.stamp(),
                            {Association::particle, "1", id_axes}))));
    ASSERT_TRUE(packed.values().extent(0) == 3 && packed.values().extent(1) == 2);
    for (std::size_t entity = 0; entity < 3; ++entity) {
        ASSERT_EQ(packed.stable_ids()(entity), ids(entity));
        ASSERT_EQ(packed.values()(entity, 0), values(0, entity));
        ASSERT_EQ(packed.values()(entity, 1), values(1, entity));
    }

    const std::array invalid_axes{FieldAxis{2}, FieldAxis{3}};
    auto wrong_roles = TransportBuilder::prepare(
        exec, TransportPurpose::coupling,
        take(values_issuer.read(values, field(), values_issuer.stamp(),
                                {Association::particle, "m", invalid_axes})),
        take(ids_issuer.read(ids, id_field(), ids_issuer.stamp(),
                            {Association::particle, "1", id_axes})));
    ASSERT_EQ(wrong_roles.status().code(), StatusCode::invalid_argument);
}

TEST(Transport, PrepareRejectionAndRetry) {
    const Kokkos::Serial exec;
    Kokkos::View<real*, Kokkos::Serial> values("values", 2);
    Kokkos::View<Id*, Kokkos::Serial> ids("ids", 2);
    Kokkos::View<Id*, Kokkos::Serial> short_ids("too few ids", 1);
    const std::array axes{FieldAxis{2}};
    const std::array short_axes{FieldAxis{1}};
    const FieldBindingSpec value_spec{Association::particle, "m", axes};
    const FieldBindingSpec ids_spec{Association::particle, "1", axes};
    const auto descriptor = field();
    auto values_issuer = take(FieldIssuer::create_at_validated_time(values, trial_stamp()));
    auto short_issuer = take(FieldIssuer::create_at_validated_time(short_ids, trial_stamp()));

    auto with_ids = [&](const FieldIssuer& issuer, FieldDescriptor ids_descriptor,
                        TransportPurpose purpose = TransportPurpose::coupling) {
        return TransportBuilder::prepare(
            exec, purpose, take(values_issuer.read(values, descriptor, values_issuer.stamp(),
                                                  value_spec)),
            take(issuer.read(ids, std::move(ids_descriptor), issuer.stamp(), ids_spec)));
    };
    {
        auto different_layout = trial_stamp();
        different_layout.layout = LayoutVersion{1};
        auto stale_issuer = take(FieldIssuer::create_at_validated_time(ids, different_layout));
        ASSERT_FALSE(with_ids(stale_issuer, id_field()).ok());
    }

    auto ids_issuer = take(FieldIssuer::create_at_validated_time(ids, trial_stamp()));
    ASSERT_FALSE(with_ids(ids_issuer, id_field(), static_cast<TransportPurpose>(255)).ok());
    auto foreign_ids = id_field();
    foreign_ids.entity_set = "foreign particles";
    ASSERT_FALSE(with_ids(ids_issuer, foreign_ids).ok());
    auto wrong_size = TransportBuilder::prepare(
        exec, TransportPurpose::coupling,
        take(values_issuer.read(values, descriptor, values_issuer.stamp(), value_spec)),
        take(short_issuer.read(short_ids, id_field(), short_issuer.stamp(),
                              {Association::particle, "1", short_axes})));
    ASSERT_FALSE(wrong_size.ok());
    ASSERT_FALSE(with_ids(ids_issuer, id_field()).ok()); // Both uninitialized IDs are zero.

    // No failure above may retain an input lease. A corrected payload can be prepared.
    {
        auto write_values = take(values_issuer.write(values, descriptor, values_issuer.stamp(),
                                                     value_spec, FieldAccess::overwrite));
        auto write_ids = take(ids_issuer.write(ids, id_field(), ids_issuer.stamp(), ids_spec,
                                               FieldAccess::overwrite));
        write_values.view()(0) = real{-3};
        write_values.view()(1) = real{8};
        write_ids.view()(0) = 4;
        write_ids.view()(1) = 2;
    }
    auto packed = take(with_ids(ids_issuer, id_field()));
    ASSERT_TRUE(packed.stable_ids()(0) == 4 && packed.stable_ids()(1) == 2);
    ASSERT_TRUE(packed.values()(0, 0) == real{-3} && packed.values()(1, 0) == real{8});
}

TEST(Transport, EmptyTransport) {
    const Kokkos::Serial exec;
    const auto backing = std::make_shared<real>(real{0});
    auto values = take(ExternalField<real>::create(backing.get(), 0, 1, sizeof(real), backing,
                                                     trial_stamp()));
    Kokkos::View<Id*, Kokkos::Serial> ids("empty ids", 0);
    auto ids_issuer = take(FieldIssuer::create_at_validated_time(ids, trial_stamp()));
    const std::array axes{FieldAxis{0}};
    auto empty = take(TransportBuilder::prepare(
        exec, TransportPurpose::coupling,
        take(values.read(field(FieldOwnership::external), values.stamp(),
                         {Association::particle, "m", axes})),
        take(ids_issuer.read(ids, id_field(), ids_issuer.stamp(),
                            {Association::particle, "1", axes}))));
    ASSERT_TRUE(empty.values().extent(0) == 0 && empty.values().extent(1) == 1);
    ASSERT_EQ(empty.stable_ids().extent(0), 0);
}
} // namespace
