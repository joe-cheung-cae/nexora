#include <participants/reference/StructureState.hpp>
#include <data/Permutation.hpp>
#include <cmath>
#include <exception>
#include <new>
#include <utility>

namespace nexora::reference {
namespace {
using namespace nexora::detail;
using Values = StructureValues;
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
struct Failure { Status status; };
template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw Failure{result.status()};
    return std::move(result).value();
}
template <class View>
StructureField<View> accepted_field(View values, const FieldDescriptor& field, FieldStamp stamp) {
    auto issuer = require(FieldIssuer::create_at_validated_time(values, stamp));
    const std::array axes{FieldAxis{values.extent(0)}};
    auto grant = require(issuer.read(values, field, issuer.stamp(),
                                     {field.association, field.units, axes}));
    return {std::move(values), std::move(issuer), std::move(grant)};
}
std::array<FieldDescriptor, 4> fields(const StructureBank& bank) {
    return {bank.ids.grant.descriptor(), bank.x.grant.descriptor(),
            bank.q.grant.descriptor(), bank.history.grant.descriptor()};
}
} // namespace

StructureBank::StructureBank(ConstructionKey, detail::EntityOrder entity_order, detail::CsrAdjacency graph,
                             FieldStamp stamp, const std::array<FieldDescriptor, 4>& fields,
                             ConstStructureValues positions, ConstStructureValues values,
                             ConstStructureValues previous)
    : order(std::move(entity_order)), adjacency(std::move(graph)), logical(stamp),
      ids(accepted_field(order.ids(), fields[0], stamp)),
      x(accepted_field(std::move(positions), fields[1], stamp)),
      q(accepted_field(std::move(values), fields[2], stamp)),
      history(accepted_field(std::move(previous), fields[3], stamp)) {}

void StructureBank::close() noexcept {
    ids.issuer.close();
    x.issuer.close();
    q.issuer.close();
    history.issuer.close();
}

Result<std::unique_ptr<StructureState>> StructureState::create(
    const Kokkos::Serial& exec, const StructureConfig& config) {
    const auto count = config.ids.size();
    if (count == 0 || config.x.size() != count || config.q.size() != count ||
        config.history.size() != count)
        return Status{StatusCode::invalid_argument, "structure fields require equal nonzero counts"};
    if (!config.stamp.accepted || config.stamp.trial != TrialEpoch{} ||
        config.stamp.storage_incarnation != 0)
        return Status{StatusCode::invalid_argument, "structure requires an accepted logical stamp"};
    if (auto valid = config.stamp.validate(); !valid.ok())
        return valid;
    for (std::size_t i = 0; i < config.fields.size(); ++i) {
        const auto& field = config.fields[i];
        if (auto valid = field.validate(); !valid.ok())
            return valid;
        if (field.components != 1 ||
            field.scalar_type != (i == 0 ? ScalarType::uint64 : real_scalar_type) ||
            (i == 0 && field.units != "1"))
            return Status{StatusCode::invalid_argument, "structure field scalar shape or type differs"};
        for (std::size_t j = 0; j < i; ++j)
            if (field.id == config.fields[j].id)
                return Status{StatusCode::invalid_argument, "structure field IDs must be unique"};
        if (field.domain_id != config.entity_namespace.domain_id ||
            field.association != config.entity_namespace.association ||
            field.entity_set != config.entity_namespace.entity_set ||
            field.ownership != FieldOwnership::owned || field.access != FieldAccess::read)
            return Status{StatusCode::invalid_argument, "structure field namespace or ownership differs"};
    }
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(config.x[i]) || !std::isfinite(config.q[i]) ||
            !std::isfinite(config.history[i]))
            return Status{StatusCode::invalid_argument, "structure fields must be finite"};
    try {
        auto order = require(detail::EntityOrder::create(exec, config.entity_namespace,
                                                         config.stamp.layout, config.ids));
        auto graph = require(detail::CsrAdjacency::create(exec, order, order, config.stamp.topology,
                                                          config.offsets, config.indices));
        StructureValues x(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "structure.x"), count);
        StructureValues q(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "structure.q"), count);
        StructureValues h(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "structure.history"), count);
        using Input = Kokkos::View<const real*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        Kokkos::deep_copy(exec, x, Input(config.x.data(), count));
        Kokkos::deep_copy(exec, q, Input(config.q.data(), count));
        Kokkos::deep_copy(exec, h, Input(config.history.data(), count));
        exec.fence("structure initial state ready");
        auto bank = std::make_shared<StructureBank>(
            StructureBank::ConstructionKey{}, std::move(order), std::move(graph), config.stamp,
            config.fields, x, q, h);
        return std::unique_ptr<StructureState>(new StructureState(exec, std::move(bank)));
    } catch (const Failure& failure) {
        return failure.status;
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate structure state"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "cannot initialize structure state"};
    }
}

StructureState::StructureState(const Kokkos::Serial& exec, std::shared_ptr<StructureBank> bank)
    : exec_(exec), accepted_(std::move(bank)) {}
StructureState::~StructureState() { accepted_->close(); }
std::shared_ptr<const StructureBank> StructureState::snapshot() const { return accepted_; }

Status StructureState::reorder(std::span<const GlobalEntityId> requested, Fault fault) {
    try {
        auto stamp = accepted_->logical;
        stamp.layout = require(checked_next_version(stamp.layout));
        auto requested_order = require(EntityOrder::create(
            exec_, accepted_->order.entity_namespace(), stamp.layout, requested));
        const auto permutation = require(Permutation::between(exec_, accepted_->order,
                                                               requested_order));
        const auto forward = permutation.new_to_old();
        const auto inverse = permutation.old_to_new();
        const auto count = accepted_->order.size();
        Kokkos::View<std::uint64_t*, Kokkos::HostSpace> ids(
            Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "reorder.ids"), count);
        Values x(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "reorder.x"), count);
        const auto old_ids = accepted_->ids.grant.view();
        const auto old_x = accepted_->x.grant.view();
        Kokkos::parallel_for("structure.gather IDs and geometry", Policy(exec_, 0, count),
                            KOKKOS_LAMBDA(std::size_t i) {
            ids(i) = old_ids(forward(i));
            x(i) = old_x(forward(i));
        });
        exec_.fence("structure partial candidate ready");
        if (fault == Fault::after_partial_candidate)
            return {StatusCode::execution_failure, "injected partial candidate failure"};

        Values q(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "reorder.q"), count);
        Values h(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "reorder.history"), count);
        const auto old_q = accepted_->q.grant.view();
        const auto old_h = accepted_->history.grant.view();
        Kokkos::parallel_for("structure.gather values and history", Policy(exec_, 0, count),
                            KOKKOS_LAMBDA(std::size_t i) {
            q(i) = old_q(forward(i));
            h(i) = old_h(forward(i));
        });
        const auto old_offsets = accepted_->adjacency.offsets();
        const auto old_indices = accepted_->adjacency.indices();
        Kokkos::View<Offset*, Kokkos::HostSpace> offsets(
            Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "reorder.offsets"), count + 1);
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> indices(
            Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "reorder.indices"),
            old_indices.extent(0));
        // One Serial kernel builds the exclusive prefix; row gathers then
        // use that prefix on the same execution instance.
        Kokkos::parallel_for("structure CSR row offsets", Policy(exec_, 0, 1),
                            KOKKOS_LAMBDA(std::size_t) {
            offsets(0) = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto row = forward(i);
                offsets(i + 1) = offsets(i) + (old_offsets(row + 1) - old_offsets(row));
            }
        });
        Kokkos::parallel_for("structure CSR rows and columns", Policy(exec_, 0, count),
                            KOKKOS_LAMBDA(std::size_t i) {
            const auto row = forward(i);
            for (Offset j = old_offsets(row); j < old_offsets(row + 1); ++j)
                indices(offsets(i) + (j - old_offsets(row))) = inverse(old_indices(j));
        });
        exec_.fence("structure candidate diagnostics ready");
        auto order = require(EntityOrder::copy_from(exec_, requested_order.entity_namespace(),
                                                     stamp.layout, ids));
        if (!order.same_order(requested_order))
            return {StatusCode::execution_failure, "gathered IDs differ from requested order"};
        auto graph = require(CsrAdjacency::create(
            exec_, order, order, stamp.topology,
            std::span<const Offset>(offsets.data(), offsets.extent(0)),
            std::span<const LocalIndex>(indices.data(), indices.extent(0))));
        // The validated permutation gathers immutable accepted finite fields
        // exactly; no arithmetic can create a new invalid numerical value.
        auto candidate = std::make_shared<StructureBank>(
            StructureBank::ConstructionKey{}, std::move(order), std::move(graph), stamp,
            fields(*accepted_), x, q, h);
        if (fault == Fault::before_publish)
            return {StatusCode::execution_failure, "injected prepared publication failure"};
        publish(std::move(candidate));
        return {};
    } catch (const Failure& failure) {
        return failure.status;
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate reorder candidate"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "cannot build reorder candidate"};
    }
}

Status StructureState::commit_values(std::span<const real> supplied) {
    if (supplied.size() != accepted_->order.size())
        return {StatusCode::invalid_argument, "replacement value count differs"};
    for (real value : supplied)
        if (!std::isfinite(value))
            return {StatusCode::invalid_argument, "replacement value is not finite"};
    try {
        auto stamp = accepted_->logical;
        stamp.accepted_version = require(checked_next_version(stamp.accepted_version));
        const auto count = supplied.size();
        Values x(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "commit.x"), count);
        Values q(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "commit.q"), count);
        Values h(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "commit.history"), count);
        Kokkos::deep_copy(exec_, x, accepted_->x.grant.view());
        Kokkos::deep_copy(exec_, h, accepted_->history.grant.view());
        // Explicit copy from caller-owned host storage before publication.
        Kokkos::View<const real*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            input(supplied.data(), count);
        Kokkos::deep_copy(exec_, q, input);
        exec_.fence("structure value commit ready");
        auto order = require(EntityOrder::copy_from(exec_, accepted_->order.entity_namespace(),
                                                     stamp.layout, accepted_->order.ids()));
        // Same logical order/topology: the immutable CSR may stay shared.
        auto candidate = std::make_shared<StructureBank>(
            StructureBank::ConstructionKey{}, std::move(order), accepted_->adjacency, stamp,
            fields(*accepted_), x, q, h);
        publish(std::move(candidate));
        return {};
    } catch (const Failure& failure) {
        return failure.status;
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate commit candidate"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "cannot build commit candidate"};
    }
}

void StructureState::publish(std::shared_ptr<StructureBank> candidate) noexcept {
    // All fallible work is finished. Closing denies fresh old bindings;
    // existing snapshots retain their independent grants and allocations.
    accepted_->close();
    accepted_.swap(candidate);
}
} // namespace nexora::reference
