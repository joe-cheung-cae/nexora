#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Kokkos_Core.hpp>
#include <nexora/core/Real.hpp>
#include <array>
#include <cassert>
#include <cstdlib>
#include <data/FieldIssuer.hpp>
#include <iostream>
#include <limits>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
using namespace nexora;
using nexora::detail::FieldAxis;
using nexora::detail::FieldBindingSpec;
using nexora::detail::FieldIssuer;
using SerialView = Kokkos::View<real*, Kokkos::Serial>;
using ReadBinding = ReadField<SerialView>;
using WriteBinding = WriteField<SerialView>;

template <class Binding>
concept HasRvalueView = requires(Binding binding) { std::move(binding).view(); };

static_assert(!std::is_copy_constructible_v<ReadLease> && std::is_move_constructible_v<ReadLease>);
static_assert(!std::is_copy_constructible_v<WriteLease> &&
              std::is_move_constructible_v<WriteLease>);
static_assert(!std::is_copy_constructible_v<ReadBinding>);
static_assert(!std::is_copy_constructible_v<WriteBinding>);
static_assert(!HasRvalueView<ReadBinding> && !HasRvalueView<WriteBinding>);
static_assert(std::is_same_v<decltype(std::declval<const ReadBinding&>().view()),
                             SerialView::const_type>);
static_assert(
    !std::is_assignable_v<decltype(std::declval<const ReadBinding&>().view()(0)), real>);
static_assert(
    std::is_assignable_v<decltype(std::declval<const WriteBinding&>().view()(0)), real>);

template <class T>
void require_ok(const Result<T>& result,
                const std::source_location source = std::source_location::current()) {
    if (!result.ok()) {
        std::cerr << source.file_name() << ':' << source.line() << ": " << result.status().message()
                  << '\n';
        std::abort();
    }
}

FieldDescriptor descriptor(std::size_t components = 1,
                           FieldAccess access = FieldAccess::overwrite) {
    FieldDescriptor field;
    field.id = FieldId{7};
    field.name = "values";
    field.entity_set = "particles";
    field.association = Association::particle;
    field.units = "m";
    field.components = components;
    field.access = access;
    return field;
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

void test_access_and_serial_work() {
    SerialView values("lease values", 4);
    const std::array axes{FieldAxis{4, false}};
    const FieldBindingSpec spec{Association::particle, "m", axes};
    const auto field = descriptor();
    const auto accumulation = descriptor(1, FieldAccess::accumulate);
    auto created = FieldIssuer::create(values, trial_stamp());
    require_ok(created);
    auto& issuer = created.value();
    const auto stamp = issuer.stamp();

    const Kokkos::Serial exec;
    {
        auto granted = issuer.write(values, field, stamp, spec, FieldAccess::overwrite);
        require_ok(granted);
        auto writer = std::move(granted).value();
        assert(writer.descriptor().id == field.id && writer.stamp() == stamp);
        assert(!issuer.read(values, field, stamp, spec).ok());
        assert(!issuer.write(values, accumulation, stamp, spec, FieldAccess::accumulate).ok());
        const auto write = writer.view();
        assert(write.data() == values.data());
        Kokkos::parallel_for(
            "overwrite leased field", Kokkos::RangePolicy<Kokkos::Serial>(exec, 0, 4),
            KOKKOS_LAMBDA(int index) { write(index) = real{2} * index; });
        exec.fence("overwrite complete");
    }
    {
        auto granted = issuer.write(values, accumulation, stamp, spec, FieldAccess::accumulate);
        require_ok(granted);
        const auto write = granted.value().view();
        Kokkos::parallel_for(
            "accumulate leased field", Kokkos::RangePolicy<Kokkos::Serial>(exec, 0, 4),
            KOKKOS_LAMBDA(int index) { write(index) += real{1}; });
        exec.fence("accumulation complete");
    }
    {
        auto first = issuer.read(values, field, stamp, spec);
        require_ok(first);
        auto reader = std::move(first).value();
        {
            const SerialView::const_type const_values = values;
            auto second = issuer.read(const_values, field, stamp, spec);
            require_ok(second);
            assert(second.value().view()(3) == real{7});
        }
        assert(!issuer.write(values, field, stamp, spec, FieldAccess::overwrite).ok());
        const auto read = reader.view();
        assert(reader.descriptor().id == field.id && reader.stamp() == stamp);
        assert(read.data() == values.data());
        real sum{};
        Kokkos::parallel_reduce(
            "read leased field", Kokkos::RangePolicy<Kokkos::Serial>(exec, 0, 4),
            KOKKOS_LAMBDA(int index, real& total) { total += read(index); }, sum);
        exec.fence("read complete");
        assert(sum == real{16});
    }
    assert(issuer.write(values, field, stamp, spec, FieldAccess::overwrite).ok());
}

void test_layouts_and_allocation_aliases() {
    const std::array matrix_axes{FieldAxis{3, false}, FieldAxis{2, true}};
    const FieldBindingSpec matrix_spec{Association::particle, "m", matrix_axes};
    const auto matrix_field = descriptor(2);
    Kokkos::View<real**, Kokkos::LayoutRight, Kokkos::Serial> right("layout right", 3, 2);
    Kokkos::View<real**, Kokkos::LayoutLeft, Kokkos::Serial> left("layout left", 3, 2);
    auto right_issuer = FieldIssuer::create(right, trial_stamp());
    auto left_issuer = FieldIssuer::create(left, trial_stamp());
    require_ok(right_issuer);
    require_ok(left_issuer);
    const auto right_stamp = right_issuer.value().stamp();
    const auto left_stamp = left_issuer.value().stamp();
    {
        auto right_read = right_issuer.value().read(right, matrix_field, right_stamp, matrix_spec);
        auto left_read = left_issuer.value().read(left, matrix_field, left_stamp, matrix_spec);
        require_ok(right_read);
        require_ok(left_read);
        assert(right_read.value().view().stride(0) == 2);
        assert(left_read.value().view().stride(0) == 1);
    }

    const auto first = Kokkos::subview(right, Kokkos::ALL, 0);
    const auto second = Kokkos::subview(right, Kokkos::ALL, 1);
    static_assert(std::is_same_v<typename decltype(first)::array_layout, Kokkos::LayoutStride>);
    const std::array column_axes{FieldAxis{3, false}};
    const FieldBindingSpec column_spec{Association::particle, "m", column_axes};
    const auto first_field = descriptor();
    auto second_field = descriptor();
    second_field.id = FieldId{8};
    second_field.name = "other component";
    {
        auto reader = right_issuer.value().read(first, first_field, right_stamp, column_spec);
        require_ok(reader);
        assert(reader.value().view().stride(0) == 2);
        assert(!FieldIssuer::create(second, trial_stamp()).ok());
        assert(!right_issuer.value()
                    .write(second, second_field, right_stamp, column_spec, FieldAccess::overwrite)
                    .ok());
        assert(left_issuer.value()
                   .write(left, matrix_field, left_stamp, matrix_spec, FieldAccess::overwrite)
                   .ok());
    }
    {
        auto writer = right_issuer.value().write(second, second_field, right_stamp, column_spec,
                                                FieldAccess::overwrite);
        require_ok(writer);
        assert(!right_issuer.value().read(first, first_field, right_stamp, column_spec).ok());
    }
    auto other_allocation = right_issuer.value().read(left, matrix_field, right_stamp, matrix_spec);
    assert(other_allocation.status().code() == StatusCode::invalid_argument);

    // Entity axis last and two component axes exercise method-specific tensor layout.
    const std::array tensor_axes{FieldAxis{2, true}, FieldAxis{2, true}, FieldAxis{3, false}};
    const FieldBindingSpec tensor_spec{Association::particle, "m", tensor_axes};
    Kokkos::View<real***, Kokkos::LayoutLeft, Kokkos::Serial> tensor("tensor", 2, 2, 3);
    auto tensor_issuer = FieldIssuer::create(tensor, trial_stamp());
    require_ok(tensor_issuer);
    const auto tensor_stamp = tensor_issuer.value().stamp();
    auto tensor_read = tensor_issuer.value().read(tensor, descriptor(4), tensor_stamp, tensor_spec);
    require_ok(tensor_read);
    assert(tensor_read.value().view().extent(2) == 3);
    assert(!tensor_issuer.value().read(tensor, descriptor(2), tensor_stamp, tensor_spec).ok());

    const auto empty = Kokkos::subview(right, std::pair{1, 1}, 0);
    const std::array empty_axes{FieldAxis{0, false}};
    const FieldBindingSpec empty_spec{Association::particle, "m", empty_axes};
    auto empty_read = right_issuer.value().read(empty, first_field, right_stamp, empty_spec);
    require_ok(empty_read);
    assert(empty_read.value().view().extent(0) == 0);
    assert(!right_issuer.value()
                .read(empty, first_field, right_stamp, {Association::particle, "s", empty_axes})
                .ok());

    SerialView empty_allocation("empty field", 0);
    auto empty_issuer = FieldIssuer::create(empty_allocation, trial_stamp());
    require_ok(empty_issuer);
    auto empty_binding = empty_issuer.value().read(empty_allocation, first_field,
                                                  empty_issuer.value().stamp(), empty_spec);
    require_ok(empty_binding);
    assert(empty_binding.value().view().extent(0) == 0);
}

void test_rejection() {
    SerialView values("field to validate", 4);
    auto invalid_stamp = trial_stamp();
    invalid_stamp.time.physical_time = std::numeric_limits<double>::infinity();
    assert(!FieldIssuer::create(values, invalid_stamp).ok());
    auto created = FieldIssuer::create(values, trial_stamp());
    require_ok(created);
    auto& issuer = created.value();
    const auto stamp = issuer.stamp();
    assert(stamp.storage_incarnation != 0);
    const std::array axes{FieldAxis{4, false}};
    const FieldBindingSpec spec{Association::particle, "m", axes};
    const auto field = descriptor();
    const std::array wrong_extent{FieldAxis{5, false}};
    const std::array wrong_rank{FieldAxis{4, false}, FieldAxis{1, true}};
    assert(!issuer.read(values, field, stamp, {Association::node, "m", axes}).ok());
    assert(!issuer.read(values, field, stamp, {Association::particle, "s", axes}).ok());
    assert(!issuer.read(values, field, stamp, {Association::particle, "m", wrong_extent}).ok());
    assert(!issuer.read(values, field, stamp, {Association::particle, "m", wrong_rank}).ok());

    auto invalid_field = field;
    invalid_field.name.clear();
    assert(!issuer.read(values, invalid_field, stamp, spec).ok());
    invalid_field = field;
    invalid_field.scalar_type =
        real_scalar_type == ScalarType::float32 ? ScalarType::float64 : ScalarType::float32;
    assert(!issuer.read(values, invalid_field, stamp, spec).ok());
    invalid_field = field;
    invalid_field.ownership = FieldOwnership::external;
    auto external = issuer.read(values, invalid_field, stamp, spec);
    assert(external.status().code() == StatusCode::unsupported);
    invalid_field = field;
    invalid_field.access = FieldAccess::read;
    assert(!issuer.write(values, invalid_field, stamp, spec, FieldAccess::overwrite).ok());
    invalid_field.access = FieldAccess::overwrite;
    assert(!issuer.write(values, invalid_field, stamp, spec, FieldAccess::accumulate).ok());
    assert(!issuer.write(values, field, stamp, spec, FieldAccess::read).ok());

    std::array stale_stamps{stamp, stamp, stamp, stamp, stamp, stamp, stamp, stamp, stamp};
    stale_stamps[0].topology = TopologyVersion{1};
    stale_stamps[1].geometry = GeometryVersion{1};
    stale_stamps[2].layout = LayoutVersion{1};
    stale_stamps[3].accepted_version = StateVersion{1};
    stale_stamps[4].trial.attempt = 1;
    stale_stamps[5].time.physical_time = 0.375;
    stale_stamps[6].valid_until = 1.0;
    stale_stamps[7].accepted = true;
    stale_stamps[8].storage_incarnation = 0;
    for (const auto& stale : stale_stamps) {
        auto rejected = issuer.read(values, field, stale, spec);
        assert(rejected.status().code() == StatusCode::invalid_state);
    }

    using Borrowed = Kokkos::View<real*, Kokkos::Serial, Kokkos::MemoryUnmanaged>;
    const Borrowed borrowed(values.data(), 4);
    assert(FieldIssuer::create(borrowed, trial_stamp()).status().code() == StatusCode::unsupported);
    assert(issuer.read(borrowed, field, stamp, spec).status().code() == StatusCode::unsupported);

    auto accepted_stamp = trial_stamp();
    accepted_stamp.accepted = true;
    accepted_stamp.trial = {};
    SerialView accepted("accepted field", 4);
    auto accepted_issuer = FieldIssuer::create(accepted, accepted_stamp);
    require_ok(accepted_issuer);
    accepted_stamp = accepted_issuer.value().stamp();
    assert(accepted_issuer.value().read(accepted, field, accepted_stamp, spec).ok());
    assert(!accepted_issuer.value()
                .write(accepted, field, accepted_stamp, spec, FieldAccess::overwrite)
                .ok());
    assert(!accepted_issuer.value()
                .write(accepted, descriptor(1, FieldAccess::accumulate), accepted_stamp, spec,
                       FieldAccess::accumulate)
                .ok());

    SerialView storage("const registration", 4);
    const SerialView::const_type const_storage = storage;
    auto const_issuer = FieldIssuer::create(const_storage, trial_stamp());
    require_ok(const_issuer);
    const auto& const_stamp = const_issuer.value().stamp();
    require_ok(const_issuer.value().read(const_storage, field, const_stamp, spec));
    auto denied_write =
        const_issuer.value().write(storage, field, const_stamp, spec, FieldAccess::overwrite);
    assert(denied_write.status().code() == StatusCode::invalid_state);
}

void test_move_assignment() {
    const auto field = descriptor();
    const std::array axes{FieldAxis{1, false}};
    const FieldBindingSpec spec{Association::particle, "m", axes};
    SerialView first("first movable field", 1);
    SerialView second("second movable field", 1);
    auto first_issuer = FieldIssuer::create(first, trial_stamp());
    auto second_issuer = FieldIssuer::create(second, trial_stamp());
    require_ok(first_issuer);
    require_ok(second_issuer);
    const auto& first_stamp = first_issuer.value().stamp();
    const auto& second_stamp = second_issuer.value().stamp();
    {
        auto first_read = first_issuer.value().read(first, field, first_stamp, spec);
        auto second_read = second_issuer.value().read(second, field, second_stamp, spec);
        require_ok(first_read);
        require_ok(second_read);
        first_read.value() = std::move(second_read).value();
        assert(first_read.value().view().data() == second.data());
        assert(first_read.value().stamp() == second_stamp);
        bool moved_from_rejected = false;
        try {
            (void)second_read.value().view();
        } catch (const std::logic_error&) {
            moved_from_rejected = true;
        }
        assert(moved_from_rejected);
        require_ok(
            first_issuer.value().write(first, field, first_stamp, spec, FieldAccess::overwrite));
        auto blocked =
            second_issuer.value().write(second, field, second_stamp, spec, FieldAccess::overwrite);
        assert(blocked.status().code() == StatusCode::invalid_state);
    }
    require_ok(
        second_issuer.value().write(second, field, second_stamp, spec, FieldAccess::overwrite));
}

void test_close_and_lease_lifetime() {
    SerialView values("retained field", 1);
    Kokkos::deep_copy(values, real{42});
    const auto field = descriptor();
    const std::array axes{FieldAxis{1, false}};
    const FieldBindingSpec spec{Association::particle, "m", axes};
    std::optional<ReadField<SerialView>> retained;
    FieldStamp old_stamp;
    {
        auto created = FieldIssuer::create(values, trial_stamp());
        require_ok(created);
        auto& issuer = created.value();
        old_stamp = issuer.stamp();
        auto read = issuer.read(values, field, old_stamp, spec);
        require_ok(read);
        retained.emplace(std::move(read).value());
        issuer.close();
        assert(!issuer.read(values, field, old_stamp, spec).ok());
        assert(!issuer.write(values, field, old_stamp, spec, FieldAccess::overwrite).ok());
    }
    auto duplicate = FieldIssuer::create(values, trial_stamp());
    assert(duplicate.status().code() == StatusCode::already_exists);
    assert(retained->view()(0) == real{42});
    retained.reset();
    {
        auto created = FieldIssuer::create(values, trial_stamp());
        require_ok(created);
        auto& issuer = created.value();
        assert(issuer.stamp().storage_incarnation > old_stamp.storage_incarnation);
        assert(!issuer.read(values, field, old_stamp, spec).ok());
        auto read = issuer.read(values, field, issuer.stamp(), spec);
        require_ok(read);
        retained.emplace(std::move(read).value());
    }
    values = SerialView{};
    assert(retained->view()(0) == real{42});
}
} // namespace

int main(int argc, char* argv[]) {
    assert(FieldIssuer::create(SerialView{}, trial_stamp()).status().code() ==
           StatusCode::invalid_state);
    const Kokkos::ScopeGuard runtime(argc, argv);
    test_access_and_serial_work();
    test_layouts_and_allocation_aliases();
    test_rejection();
    test_move_assignment();
    test_close_and_lease_lifetime();
}
