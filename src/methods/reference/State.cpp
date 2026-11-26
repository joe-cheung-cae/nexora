#include <methods/reference/State.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace nexora::reference {
namespace {
Status invalid_state(const char* message) {
    return {StatusCode::invalid_state, message};
}
Status invalid_argument(const char* message) {
    return {StatusCode::invalid_argument, message};
}

template <class Exec, class ReadView, class WriteView>
bool relaxation(const Exec& exec, real factor, const ReadView& x,
                const ReadView& q, const WriteView& tx, const WriteView& tq) {
    static_assert(Kokkos::is_view<ReadView>::value && Kokkos::is_view<WriteView>::value);
    static_assert(ReadView::rank() == 1 && WriteView::rank() == 1);
    static_assert(std::is_same_v<typename ReadView::value_type, const real> &&
                  std::is_same_v<typename WriteView::value_type, real>);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename ReadView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<Exec, typename WriteView::memory_space>::accessible);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::uint64_t>>;
    std::uint64_t invalid_count = 0;
    Kokkos::parallel_reduce(
        "reference.relaxation", Policy(exec, 0, x.extent(0)),
        KOKKOS_LAMBDA(const std::uint64_t j, std::uint64_t& invalid) {
            // Admitted 0 < factor <= 1 makes this a finite, same-sign transfer
            // no larger than x in magnitude. Only addition to q can overflow.
            const real d = factor * x(j);
            tx(j) = x(j) - d;
            tq(j) = q(j) + d;
            if (!Kokkos::isfinite(tq(j)))
                ++invalid;
        }, invalid_count);
    return invalid_count == 0;
}

template <class T>
using HostInput = Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
} // namespace

State::State(Config config) : config_(std::move(config)) {}

Snapshot& Snapshot::operator=(Snapshot other) noexcept {
    using std::swap;
    swap(ids, other.ids);
    swap(x, other.x);
    swap(q, other.q);
    swap(history, other.history);
    swap(control, other.control);
    swap(diagnostic, other.diagnostic);
    swap(version, other.version);
    swap(time, other.time);
    swap(x_descriptor_, other.x_descriptor_);
    swap(q_descriptor_, other.q_descriptor_);
    swap(grants_, other.grants_);
    return *this;
}

Result<AllocationSize> validate_initial_fields(const Config& config, ConfigField* invalid_field) {
    const auto invalid = [&](ConfigField field, const char* message) {
        if (invalid_field)
            *invalid_field = field;
        return invalid_argument(message);
    };
    constexpr auto bad_shape = "invalid reference initial state or lambda";
    if (config.ids.empty())
        return invalid(ConfigField::ids, bad_shape);
    if (config.ids.size() != config.x.size())
        return invalid(ConfigField::x, bad_shape);
    if (config.ids.size() != config.q.size())
        return invalid(ConfigField::q, bad_shape);
    if (config.ids.size() > std::numeric_limits<LocalIndex>::max()) {
        return Status{StatusCode::resource_exhausted, "reference entity count exceeds local index capacity"};
    }
    auto sorted = config.ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        return invalid_argument("reference stable IDs must be unique");
    }
    constexpr auto nonfinite_values = "reference initial values must be finite";
    for (std::size_t i = 0; i < config.ids.size(); ++i) {
        if (!std::isfinite(config.x[i]))
            return invalid(ConfigField::x, nonfinite_values);
        if (!std::isfinite(config.q[i]))
            return invalid(ConfigField::q, nonfinite_values);
    }
    const auto n = config.ids.size();
    return AllocationSize{n * sizeof(std::uint64_t),
                          2 * n * sizeof(real) + sizeof(std::uint64_t)};
}

Status State::preflight_initialize() const {
    if (phase_ != Phase::created) {
        return invalid_state("reference state already initialized");
    }
    if (!Kokkos::is_initialized()) {
        return invalid_state("Kokkos must be initialized by the application");
    }
    return {};
}

void State::initialize(double time, const Kokkos::Serial& exec) {
    // Participant already validated input and reserved IDs plus bank payload.
    const auto n = config_.ids.size();
    EntityIds ids(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "reference.ids"), n);
    Values x(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "reference.accepted.x"), n);
    Values q(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "reference.accepted.q"), n);
    History history(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                                       "reference.accepted.history"), 1);
    Kokkos::deep_copy(exec, history, config_.history_seed);
    // The config retains all three host buffers until the instance completes the copies.
    Kokkos::deep_copy(exec, ids, HostInput<std::uint64_t>(config_.ids.data(), n));
    Kokkos::deep_copy(exec, x, HostInput<real>(config_.x.data(), n));
    Kokkos::deep_copy(exec, q, HostInput<real>(config_.q.data(), n));
    exec.fence("reference initial state ready");
    ids_ = ids;
    accepted_x_ = x;
    accepted_q_ = q;
    accepted_history_ = history;
    accepted_control_ = config_.control_seed;
    accepted_diagnostic_ = config_.history_seed + config_.control_seed.sequence;
    time_ = time;
    phase_ = Phase::initialized;
}

void State::finish_initialization() noexcept {
    std::vector<std::uint64_t>().swap(config_.ids);
    std::vector<real>().swap(config_.x);
    std::vector<real>().swap(config_.q);
}

void State::cancel_initialization() noexcept {
    ids_ = {};
    accepted_x_ = {};
    accepted_q_ = {};
    accepted_history_ = {};
    accepted_control_ = {};
    accepted_diagnostic_ = 0;
    time_ = 0;
    phase_ = Phase::created;
}

FieldStamp State::accepted_stamp() const noexcept {
    return Snapshot::accepted_stamp(StateVersion{version_}, time_);
}

Result<std::size_t> State::preflight_step(const TimeInterval& i) const {
    if (i.begin != time_)
        return invalid_argument("invalid reference time interval");
    if (version_ == std::numeric_limits<std::uint64_t>::max())
        return Status{StatusCode::resource_exhausted, "reference state version exhausted"};
    if (accepted_control_.sequence == std::numeric_limits<std::uint64_t>::max())
        return Status{StatusCode::resource_exhausted, "reference control sequence exhausted"};
    return 2 * accepted_x_.extent(0) * sizeof(real) + sizeof(std::uint64_t);
}

Status State::evaluate(const Kokkos::Serial& exec, real factor,
                        const ConstValues& x, const ConstValues& q,
                        const Values& next_x, const Values& next_q,
                        const History& history, std::uint64_t sequence) {
    if (!relaxation(exec, factor, x, q, next_x, next_q))
        return {StatusCode::numerical_failure, "reference step produced nonfinite values"};
    // Unsigned modular arithmetic is intentional in this history fixture.
    Kokkos::parallel_for("reference.history", Kokkos::RangePolicy<Kokkos::Serial>(exec, 0, 1),
                         KOKKOS_LAMBDA(const int) { history(0) = 33 * history(0) + sequence; });
    exec.fence("reference candidate ready");
    return {};
}

Status State::finalize() {
    if (phase_ == Phase::finalized)
        return invalid_state("reference state already finalized");
    accepted_x_ = {};
    accepted_q_ = {};
    accepted_history_ = {};
    ids_ = {};
    finish_initialization();
    phase_ = Phase::finalized;
    return {};
}

FieldDescriptor State::field_descriptor(Field field, FieldAccess access) {
    const bool is_x = field == Field::x;
    FieldDescriptor descriptor;
    descriptor.id = FieldId{is_x ? std::uint64_t{1} : std::uint64_t{2}};
    descriptor.name = is_x ? "x" : "q";
    descriptor.association = Association::node;
    descriptor.entity_set = "reference_samples";
    descriptor.units = "1";
    descriptor.scalar_type = real_scalar_type;
    descriptor.components = 1;
    descriptor.measure = MeasureSemantics::point_value;
    descriptor.access = access;
    descriptor.time_level = access == FieldAccess::read ? "accepted" : "step";
    return descriptor;
}

FieldDescriptor State::history_descriptor(FieldAccess access) {
    auto descriptor = field_descriptor(Field::x, access);
    descriptor.id = FieldId{4};
    descriptor.name = "replay_history";
    descriptor.entity_set = "reference_replay_history";
    descriptor.scalar_type = ScalarType::uint64;
    return descriptor;
}

Snapshot State::snapshot() const {
    Snapshot out;
    out.ids = ids_;
    out.history = accepted_history_;
    out.control = accepted_control_;
    out.diagnostic = accepted_diagnostic_;
    out.version = StateVersion{version_};
    out.time = time_;
    out.x_descriptor_ = field_descriptor(Field::x, FieldAccess::read);
    out.q_descriptor_ = field_descriptor(Field::q, FieldAccess::read);
    return out;
}
} // namespace nexora::reference
