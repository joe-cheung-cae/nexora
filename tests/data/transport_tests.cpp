#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Kokkos_Core.hpp>
#include <nexora/core/Real.hpp>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <data/ExternalField.hpp>
#include <data/FieldIssuer.hpp>
#include <data/TransportBuilder.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <nexora/data/TransportView.hpp>
#include <optional>
#include <source_location>
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

template <class T>
T take(Result<T> result, const std::source_location source = std::source_location::current()) {
    if (!result.ok()) {
        std::cerr << source.file_name() << ':' << source.line() << ": " << result.status().message()
                  << '\n';
        std::abort();
    }
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

void test_external_validation() {
    auto storage = std::make_shared<real[]>(6);
    auto* base = storage.get();
    const auto bytes = 6 * sizeof(real);
    const std::shared_ptr<const void> owner(storage, base);
    using External = ExternalField<real>;
    assert(!External::create(base, 3, 2, bytes, {}, trial_stamp()).ok());
    const std::shared_ptr<const void> unowned(std::shared_ptr<const void>{}, base);
    assert(unowned.get() == base && unowned.use_count() == 0);
    assert(!External::create(base, 3, 2, bytes, unowned, trial_stamp()).ok());
    assert(!External::create(base + 1, 3, 1, bytes, owner, trial_stamp()).ok());
    assert(!External::create(nullptr, 3, 1, bytes, owner, trial_stamp()).ok());
    assert(!External::create(base, 3, 0, bytes, owner, trial_stamp()).ok());
    assert(!External::create(base, 3, 2, 5 * sizeof(real) - 1, owner, trial_stamp()).ok());
    const auto max_size = std::numeric_limits<std::size_t>::max();
    assert(!External::create(base, 3, max_size, bytes, owner, trial_stamp()).ok());
    assert(!External::create(base, 1, 1, max_size, owner, trial_stamp()).ok());

    auto* misaligned = reinterpret_cast<std::byte*>(base) + 1;
    const std::shared_ptr<const void> wrong_alignment(storage, misaligned);
    assert(!External::create(reinterpret_cast<real*>(misaligned), 1, 1, bytes - 1,
                             wrong_alignment, trial_stamp()).ok());

    const std::array axes{FieldAxis{3}};
    const FieldBindingSpec spec{Association::particle, "m", axes};
    const auto descriptor = field(FieldOwnership::external);
    auto invalid_stamp = trial_stamp();
    invalid_stamp.valid_until = 0.0;
    assert(!External::create(base, 3, 2, bytes, owner, invalid_stamp).ok());

    auto imported = take(External::create(base, 3, 2, bytes, owner, trial_stamp()));
    assert(External::create(base, 3, 2, bytes, owner, trial_stamp()).status().code() ==
           StatusCode::already_exists);
    const std::shared_ptr<const void> overlapping(storage, base + 1);
    assert(!External::create(base + 1, 3, 1, bytes - sizeof(real), overlapping, trial_stamp())
                .ok());
    auto retaining_owner = std::make_shared<decltype(storage)>(storage);
    const std::shared_ptr<const void> independent_owner(retaining_owner, base + 1);
    assert(owner.owner_before(independent_owner) || independent_owner.owner_before(owner));
    assert(External::create(base + 1, 3, 1, bytes - sizeof(real), independent_owner,
                            trial_stamp()).status().code() == StatusCode::already_exists);

    auto owned_descriptor = descriptor;
    owned_descriptor.ownership = FieldOwnership::owned;
    assert(!imported.read(owned_descriptor, imported.stamp(), spec).ok());
    assert(!imported.read(descriptor, trial_stamp(), spec).ok());
    auto wrong_dtype = descriptor;
    wrong_dtype.scalar_type =
        real_scalar_type == ScalarType::float32 ? ScalarType::float64 : ScalarType::float32;
    assert(!imported.read(wrong_dtype, imported.stamp(), spec).ok());

    auto accepted = trial_stamp();
    accepted.accepted = true;
    accepted.trial = {};
    auto accepted_storage = std::make_shared<real[]>(3);
    auto accepted_field = take(External::create(accepted_storage.get(), 3, 1, 3 * sizeof(real),
                                                accepted_storage, accepted));
    assert(accepted_field.read(descriptor, accepted_field.stamp(), spec).ok());
    assert(accepted_field.write(descriptor, accepted_field.stamp(), spec, FieldAccess::overwrite)
               .status().code() == StatusCode::invalid_state);

    const auto const_storage = std::make_shared<const real>(real{4});
    auto readonly = take(ExternalField<const real>::create(const_storage.get(), 1, 1,
                                                              sizeof(real), const_storage,
                                                              trial_stamp()));
    const std::array scalar_axis{FieldAxis{1}};
    auto read = take(readonly.read(descriptor, readonly.stamp(),
                                   {Association::particle, "m", scalar_axis}));
    assert(read.view()(0) == real{4});
}

void test_managed_external_overlap() {
    using Managed = Kokkos::View<real*, Kokkos::Serial>;
    auto storage = std::make_shared<Managed>("shared allocation", 4);
    auto* const data = storage->data();
    const std::shared_ptr<const void> alias(storage, data);
    {
        auto managed = take(FieldIssuer::create(*storage, trial_stamp()));
        assert(!ExternalField<real>::create(data, 4, 1, 4 * sizeof(real), alias,
                                              trial_stamp()).ok());
    }
    {
        auto external = take(ExternalField<real>::create(data, 4, 1, 4 * sizeof(real), alias,
                                                           trial_stamp()));
        assert(!FieldIssuer::create(*storage, trial_stamp()).ok());
    }
    assert(FieldIssuer::create(*storage, trial_stamp()).ok());
}

void test_consumer_completion(const Kokkos::Serial& exec) {
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
        assert(value_releases == 0 && id_releases == 0);

        auto start_consumer = [&](TransportPurpose purpose) {
            auto [value_read, ids_read] = take(FieldIssuer::grant(
                values.request_read(value_descriptor, values.stamp(), value_spec),
                ids.request_read(ids_descriptor, ids.stamp(), ids_spec)));
            return take(TransportBuilder::prepare(exec, purpose, std::move(value_read),
                                                   std::move(ids_read)));
        };
        fast.emplace(start_consumer(TransportPurpose::coupling));
        slow.emplace(start_consumer(TransportPurpose::visualization));
        assert(fast->descriptor().id == value_descriptor.id && fast->stamp() == values.stamp());
        assert(fast->purpose() == TransportPurpose::coupling);
        assert(slow->purpose() == TransportPurpose::visualization);
        assert(fast->values().extent(0) == 3 && fast->values().extent(1) == 1);
        assert(fast->values()(0, 0) == real{3} && fast->values()(1, 0) == real{6} &&
               fast->values()(2, 0) == real{9});
        assert(fast->stable_ids()(0) == 41 && fast->stable_ids()(1) == 7 &&
               fast->stable_ids()(2) == 29);

        auto writer_blocked = [&] {
            const auto attempted = values.write(value_descriptor, values.stamp(), value_spec,
                                                FieldAccess::overwrite);
            return attempted.status().code() == StatusCode::invalid_state;
        };
        assert(writer_blocked());
        fast->complete();
        fast->complete();
        assert(!*fast && writer_blocked());
        assert(ids.write(ids_descriptor, ids.stamp(), ids_spec, FieldAccess::overwrite)
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
            assert(!values.read(value_descriptor, values.stamp(), value_spec).ok());
        }
        assert(ids.write(ids_descriptor, ids.stamp(), ids_spec, FieldAccess::overwrite).ok());

        fast.emplace(start_consumer(TransportPurpose::checkpoint));
        slow.emplace(start_consumer(TransportPurpose::halo));
        values.close();
        ids.close();
        assert(!values.read(value_descriptor, values.stamp(), value_spec).ok());
        assert(!ids.read(ids_descriptor, ids.stamp(), ids_spec).ok());
        assert(value_releases == 0 && id_releases == 0);
    }
    fast->complete();
    assert(value_releases == 0 && id_releases == 0);
    assert(slow->values()(0, 0) == real{4} && slow->values()(1, 0) == real{7} &&
           slow->values()(2, 0) == real{10});
    assert(slow->stable_ids()(0) == 41 && slow->stable_ids()(1) == 7 &&
           slow->stable_ids()(2) == 29);
    slow->complete();
    assert(value_releases == 1 && id_releases == 1);
    bool completed_view_rejected = false;
    try {
        (void)slow->values();
    } catch (const std::logic_error&) {
        completed_view_rejected = true;
    }
    assert(completed_view_rejected);
    slow.reset();
    fast.reset();
    assert(value_releases == 1 && id_releases == 1);
}

void test_pack_components(const Kokkos::Serial& exec) {
    Kokkos::View<real**, Kokkos::LayoutRight, Kokkos::Serial> values("component first", 2, 3);
    Kokkos::View<Id*, Kokkos::Serial> ids("particle ids", 3);
    for (std::size_t entity = 0; entity < 3; ++entity) {
        ids(entity) = 20 - entity;
        values(0, entity) = static_cast<real>(entity + 1);
        values(1, entity) = static_cast<real>(10 + entity);
    }
    const std::array values_axes{FieldAxis{2, true}, FieldAxis{3}};
    const std::array id_axes{FieldAxis{3}};
    auto values_issuer = take(FieldIssuer::create(values, trial_stamp()));
    auto ids_issuer = take(FieldIssuer::create(ids, trial_stamp()));
    auto packed = take(TransportBuilder::prepare(
        exec, TransportPurpose::coupling,
        take(values_issuer.read(values, field(FieldOwnership::owned, 2), values_issuer.stamp(),
                                {Association::particle, "m", values_axes})),
        take(ids_issuer.read(ids, id_field(), ids_issuer.stamp(),
                            {Association::particle, "1", id_axes}))));
    assert(packed.values().extent(0) == 3 && packed.values().extent(1) == 2);
    for (std::size_t entity = 0; entity < 3; ++entity) {
        assert(packed.stable_ids()(entity) == ids(entity));
        assert(packed.values()(entity, 0) == values(0, entity));
        assert(packed.values()(entity, 1) == values(1, entity));
    }

    const std::array invalid_axes{FieldAxis{2}, FieldAxis{3}};
    auto wrong_roles = TransportBuilder::prepare(
        exec, TransportPurpose::coupling,
        take(values_issuer.read(values, field(), values_issuer.stamp(),
                                {Association::particle, "m", invalid_axes})),
        take(ids_issuer.read(ids, id_field(), ids_issuer.stamp(),
                            {Association::particle, "1", id_axes})));
    assert(wrong_roles.status().code() == StatusCode::invalid_argument);
}

void test_prepare_rejection_and_retry(const Kokkos::Serial& exec) {
    Kokkos::View<real*, Kokkos::Serial> values("values", 2);
    Kokkos::View<Id*, Kokkos::Serial> ids("ids", 2);
    Kokkos::View<Id*, Kokkos::Serial> short_ids("too few ids", 1);
    const std::array axes{FieldAxis{2}};
    const std::array short_axes{FieldAxis{1}};
    const FieldBindingSpec value_spec{Association::particle, "m", axes};
    const FieldBindingSpec ids_spec{Association::particle, "1", axes};
    const auto descriptor = field();
    auto values_issuer = take(FieldIssuer::create(values, trial_stamp()));
    auto short_issuer = take(FieldIssuer::create(short_ids, trial_stamp()));

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
        auto stale_issuer = take(FieldIssuer::create(ids, different_layout));
        assert(!with_ids(stale_issuer, id_field()).ok());
    }

    auto ids_issuer = take(FieldIssuer::create(ids, trial_stamp()));
    assert(!with_ids(ids_issuer, id_field(), static_cast<TransportPurpose>(255)).ok());
    auto foreign_ids = id_field();
    foreign_ids.entity_set = "foreign particles";
    assert(!with_ids(ids_issuer, foreign_ids).ok());
    auto wrong_size = TransportBuilder::prepare(
        exec, TransportPurpose::coupling,
        take(values_issuer.read(values, descriptor, values_issuer.stamp(), value_spec)),
        take(short_issuer.read(short_ids, id_field(), short_issuer.stamp(),
                              {Association::particle, "1", short_axes})));
    assert(!wrong_size.ok());
    assert(!with_ids(ids_issuer, id_field()).ok()); // Both uninitialized IDs are zero.

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
    assert(packed.stable_ids()(0) == 4 && packed.stable_ids()(1) == 2);
    assert(packed.values()(0, 0) == real{-3} && packed.values()(1, 0) == real{8});
}

void test_empty_transport(const Kokkos::Serial& exec) {
    const auto backing = std::make_shared<real>(real{0});
    auto values = take(ExternalField<real>::create(backing.get(), 0, 1, sizeof(real), backing,
                                                     trial_stamp()));
    Kokkos::View<Id*, Kokkos::Serial> ids("empty ids", 0);
    auto ids_issuer = take(FieldIssuer::create(ids, trial_stamp()));
    const std::array axes{FieldAxis{0}};
    auto empty = take(TransportBuilder::prepare(
        exec, TransportPurpose::coupling,
        take(values.read(field(FieldOwnership::external), values.stamp(),
                         {Association::particle, "m", axes})),
        take(ids_issuer.read(ids, id_field(), ids_issuer.stamp(),
                            {Association::particle, "1", axes}))));
    assert(empty.values().extent(0) == 0 && empty.values().extent(1) == 1);
    assert(empty.stable_ids().extent(0) == 0);
}
} // namespace

int main(int argc, char* argv[]) {
    const auto owner = std::make_shared<real>(real{0});
    assert(ExternalField<real>::create(owner.get(), 1, 1, sizeof(real), owner, trial_stamp())
               .status().code() == StatusCode::invalid_state);
    const Kokkos::ScopeGuard runtime(argc, argv);
    const Kokkos::Serial exec;
    test_external_validation();
    test_managed_external_overlap();
    test_consumer_completion(exec);
    test_pack_components(exec);
    test_prepare_rejection_and_retry(exec);
    test_empty_transport(exec);
}
