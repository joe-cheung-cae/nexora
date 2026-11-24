#include "WcsphParticipant3D.hpp"

#include <algorithm>
#include <cmath>
#include <data/EntityCapacity.hpp>
#include <data/FieldIssuer.hpp>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace nexora::sph {
namespace {

using SphBoundary::FixedWall3D;

struct Failure {
    Status status;
};

template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw Failure{result.status()};
    return std::move(result).value();
}

template <class F> auto guarded(F&& operation) -> decltype(operation()) {
    try {
        return operation();
    } catch (const Failure& error) {
        return error.status;
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate 3D SPH participant state"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "3D SPH participant operation failed"};
    }
}

FieldDescriptor particle_field(FieldId id, const char* name, const char* units,
                               std::size_t components, FieldAccess access) {
    FieldDescriptor field;
    field.id = id;
    field.name = name;
    field.domain_id = DomainId{1};
    field.association = Association::particle;
    field.entity_set = "sph_owned";
    field.units = units;
    field.components = components;
    field.ownership = FieldOwnership::owned;
    field.access = access;
    return field;
}

FieldDescriptor density_field(FieldAccess access) {
    auto field = particle_field(FieldId{3}, "density", "kg/m^3", 1, access);
    field.measure = MeasureSemantics::density;
    return field;
}

FieldDescriptor pressure_field(FieldAccess access) {
    auto field = particle_field(FieldId{4}, "pressure", "Pa", 1, access);
    field.pressure_convention = "gauge";
    return field;
}

FieldDescriptor acceleration_field(FieldAccess access) {
    auto field = particle_field(FieldId{5}, "pressure_acceleration", "m/s^2", 3, access);
    field.reference_frame = "grid_cartesian";
    field.tensor_interpretation = "vector";
    return field;
}

FieldDescriptor velocity_field(FieldAccess access) {
    auto field = particle_field(FieldId{6}, "velocity", "m/s", 3, access);
    field.reference_frame = "grid_cartesian";
    field.tensor_interpretation = "vector";
    return field;
}

template <class View> auto owned_axes(const View& values) {
    if constexpr (View::rank() == 1)
        return std::array{detail::FieldAxis{values.extent(0)}};
    else
        return std::array{detail::FieldAxis{values.extent(0)}, detail::FieldAxis{3, true}};
}

template <class View>
auto read_field(const View& values, FieldDescriptor descriptor, FieldStamp stamp) {
    stamp.storage_incarnation = 0;
    auto issuer = require(detail::FieldIssuer::create(values, stamp));
    const auto axes = owned_axes(values);
    const detail::FieldBindingSpec spec{Association::particle, descriptor.units, axes};
    return require(issuer.read(values, descriptor, issuer.stamp(), spec));
}

template <class View>
auto write_field(const View& values, FieldDescriptor descriptor, FieldStamp stamp) {
    stamp.accepted = false;
    stamp.storage_incarnation = 0;
    auto issuer = require(detail::FieldIssuer::create(values, stamp));
    const auto axes = owned_axes(values);
    const detail::FieldBindingSpec spec{Association::particle, descriptor.units, axes};
    return require(issuer.write(values, descriptor, issuer.stamp(), spec, FieldAccess::overwrite));
}

template <class View>
View allocate(const Kokkos::Serial& exec, const char* label, std::size_t rows) {
    return View(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(label)), rows);
}

struct ActiveScope {
    explicit ActiveScope(bool& active) : active(active) { active = true; }
    ~ActiveScope() { active = false; }
    bool& active;
};

template <class View> std::array<real, 3> vector_at(const View& values, std::size_t row) {
    return {values(row, 0), values(row, 1), values(row, 2)};
}

} // namespace

WcsphParticipant3D::WcsphParticipant3D(ParticleFields3D fluid, std::size_t max_cells,
                                       WcsphVerletParameters parameters,
                                       Velocity3D::const_type velocity, Values::const_type density,
                                       ParticipantId id, ResourceLimits limits)
    : initial_(std::move(fluid)), max_cells_(max_cells), parameters_(parameters),
      initial_velocity_(std::move(velocity)), initial_density_(std::move(density)), id_(id),
      budget_(limits) {}

WcsphParticipant3D::WcsphParticipant3D(ParticleFields3D fluid, std::size_t max_cells,
                                       WcsphVerletParameters parameters,
                                       Velocity3D::const_type velocity, Values::const_type density,
                                       FixedWallParticles3D walls, ParticipantId id,
                                       ResourceLimits limits)
    : WcsphParticipant3D(std::move(fluid), max_cells, parameters, std::move(velocity),
                         std::move(density), id, limits) {
    initial_walls_.push_back({0, std::move(walls), std::nullopt});
}

WcsphParticipant3D::WcsphParticipant3D(ParticleFields3D fluid, std::size_t max_cells,
                                       WcsphVerletParameters parameters,
                                       Velocity3D::const_type velocity, Values::const_type density,
                                       std::vector<SphBoundary::WallGroupInput3D> walls,
                                       ParticipantId id, ResourceLimits limits)
    : WcsphParticipant3D(std::move(fluid), max_cells, parameters, std::move(velocity),
                         std::move(density), id, limits) {
    initial_walls_ = std::move(walls);
}

BindFields3D WcsphParticipant3D::bind_fields(WcsphState3D& candidate) const {
    return [this, &candidate, base = candidate.stamp](WcsphState3D& target, const FieldStamp& stamp,
                                                     const Geometry3D::Indices& owned_rows,
                                                     FieldGroup3D group,
                                                     const FieldWrite3D& write) {
        auto expected_stamp = base;
        expected_stamp.topology = candidate.geometry.topology;
        expected_stamp.geometry = candidate.geometry.geometry;
        expected_stamp.layout = candidate.geometry.layout;
        if (&target != &candidate || &owned_rows != &candidate.geometry.owned_row_to_input ||
            stamp != expected_stamp || stamp != target.stamp)
            return Status{StatusCode::invalid_state, "WCSPH field grant does not match its stage"};
        using Group = FieldGroup3D;
        const auto order = base.accepted ? std::array{Group::density, Group::velocity, Group::pressure}
                                         : std::array{Group::density, Group::pressure, Group::velocity};
        const auto bound = std::size_t(target.density.has_value()) +
                           std::size_t(target.pressure.has_value()) +
                           std::size_t(target.velocity.has_value());
        if (bound >= order.size() || group != order[bound])
            return Status{StatusCode::invalid_state, "WCSPH field grant is out of order"};

        return guarded([&]() -> Status {
            const auto rows = owned_rows.extent(0);
            const auto reserve = [&](std::size_t components) {
                return require(budget_.reserve(
                    require(detail::checked_array_bytes(rows, components * sizeof(real)))));
            };
            switch (group) {
            case FieldGroup3D::density: {
                auto storage = reserve(1);
                const auto values = allocate<Values>(*exec_, "sph.participant3d.density", rows);
                {
                    const auto grant = write_field(values, density_field(FieldAccess::overwrite),
                                                   stamp);
                    if (auto status = write({.density = grant.view()}); !status.ok())
                        return status;
                }
                auto field = read_field(Values::const_type(values), density_field(FieldAccess::read),
                                        stamp);
                target.field_storage[0] = std::move(storage);
                target.density.emplace(std::move(field));
                return {};
            }
            case FieldGroup3D::pressure: {
                auto storage = reserve(4);
                const auto values = allocate<Values>(*exec_, "sph.participant3d.pressure", rows);
                const auto force =
                    allocate<Acceleration3D>(*exec_, "sph.participant3d.pressure_acceleration", rows);
                {
                    const auto pressure =
                        write_field(values, pressure_field(FieldAccess::overwrite), stamp);
                    const auto acceleration =
                        write_field(force, acceleration_field(FieldAccess::overwrite), stamp);
                    if (auto status = write({.pressure = pressure.view(),
                                             .acceleration = acceleration.view()});
                        !status.ok())
                        return status;
                }
                auto pressure = read_field(Values::const_type(values),
                                           pressure_field(FieldAccess::read), stamp);
                auto acceleration = read_field(Acceleration3D::const_type(force),
                                               acceleration_field(FieldAccess::read), stamp);
                target.field_storage[1] = std::move(storage);
                target.pressure.emplace(std::move(pressure));
                target.acceleration.emplace(std::move(acceleration));
                return {};
            }
            case FieldGroup3D::velocity: {
                auto storage = reserve(3);
                const auto values = allocate<Velocity3D>(*exec_, "sph.participant3d.velocity", rows);
                {
                    const auto grant =
                        write_field(values, velocity_field(FieldAccess::overwrite), stamp);
                    if (auto status = write({.velocity = grant.view()}); !status.ok())
                        return status;
                }
                auto velocity = read_field(Velocity3D::const_type(values),
                                           velocity_field(FieldAccess::read), stamp);
                target.field_storage[2] = std::move(storage);
                target.velocity.emplace(std::move(velocity));
                return {};
            }
            }
            return {StatusCode::invalid_state, "unknown WCSPH field grant group"};
        });
    };
}

Status WcsphParticipant3D::complete_step(WcsphState3D& candidate) const {
    const auto& parameters = step_->parameters();
    const auto maximum = cfl_step_3d(*exec_, candidate.velocity->view(), parameters.eos.sound_speed,
                                     parameters.cfl, parameters.smoothing_length);
    if (!maximum.ok())
        return maximum.status();
    candidate.maximum_step = maximum.value();
    for (std::size_t g = 0; g < candidate.wall_phases.extent(0); ++g)
        candidate.maximum_step = std::min(candidate.maximum_step, candidate.wall_phases(g).maximum_step);
    return {};
}

TimeConstraint WcsphParticipant3D::time_constraint(double) const {
    TimeConstraint result;
    if (accepted_)
        result.maximum_step = accepted_->maximum_step;
    return result;
}

Status WcsphParticipant3D::initialize(double time) {
    if (stage_ != Stage::created || evaluating_ || !Kokkos::is_initialized() || Kokkos::is_finalized())
        return {StatusCode::invalid_state,
                "3D SPH initialization requires a new participant and active Kokkos"};
    if (!std::isfinite(time))
        return {StatusCode::invalid_argument, "invalid 3D SPH initial time"};

    const ActiveScope evaluation(evaluating_);
    const auto result = guarded([&]() -> Status {
        step_.emplace(exec_.emplace(), budget_, max_cells_, parameters_);
        step_->observer_context_ = observer_context_;
        step_->observe_ = observe_;
        auto candidate = std::make_unique<WcsphState3D>();
        auto& stamp = candidate->stamp;
        stamp.accepted = true;
        stamp.time.physical_time = stamp.valid_from = stamp.valid_until = time;
        if (auto status = step_->initialize_state(time, initial_, initial_velocity_, initial_density_,
                                                  initial_walls_, *candidate, bind_fields(*candidate));
            !status.ok())
            return status;
        if (auto status = complete_step(*candidate); !status.ok())
            return status;
        step_->commit();
        accepted_ = std::move(candidate);
        initial_ = {};
        initial_velocity_ = {};
        initial_density_ = {};
        decltype(initial_walls_){}.swap(initial_walls_);
        stage_ = Stage::initialized;
        return {};
    });
    if (!result.ok()) {
        step_.reset();
        exec_.reset();
    }
    return result;
}

Status WcsphParticipant3D::prepare() {
    if (stage_ != Stage::initialized)
        return {StatusCode::invalid_state, "3D SPH is not initialized"};
    stage_ = Stage::ready;
    return {};
}

Status WcsphParticipant3D::begin_window(const TimeInterval& interval) {
    if (stage_ != Stage::ready || reading_)
        return {StatusCode::invalid_state, "3D SPH window requires an idle prepared participant"};
    if (auto status = validate_time_step(time_constraint(interval.begin), interval.duration());
        !status.ok())
        return status;
    if (interval.duration() > accepted_->maximum_step)
        return {StatusCode::invalid_argument, "requested 3D SPH step exceeds accepted bound"};
    if (interval.begin != accepted_->stamp.time.physical_time || !std::isfinite(interval.end) ||
        interval.end <= interval.begin || interval.end != interval.begin + interval.duration())
        return {StatusCode::invalid_argument, "invalid 3D SPH window interval"};
    if (attempt_ == std::numeric_limits<std::uint64_t>::max() ||
        !detail::checked_next_version(accepted_->stamp.accepted_version).ok())
        return {StatusCode::resource_exhausted, "3D SPH attempt or accepted version exhausted"};
    return guarded([&]() -> Status {
        // The immutable accepted owner remains in place throughout the window.
        recovery_ = require(budget_.reserve(0, 1));
        ++attempt_;
        interval_ = interval;
        stage_ = Stage::open;
        return {};
    });
}

Status WcsphParticipant3D::advance_trial(const TimeInterval& interval) {
    if (stage_ != Stage::open || !matches(interval))
        return {StatusCode::invalid_state, "no matching open 3D SPH window"};
    const ActiveScope evaluation(evaluating_);
    stage_ = Stage::failed;
    return guarded([&]() -> Status {
        auto candidate = std::make_unique<WcsphState3D>();
        auto& stamp = candidate->stamp;
        stamp = accepted_->stamp;
        stamp.accepted = false;
        stamp.time = {interval.end, stamp.accepted_version.value() + 1, 0, 0};
        stamp.valid_from = stamp.valid_until = interval.end;
        stamp.trial = {stamp.time.window, attempt_, 0, 0};
        if (auto status = step_->try_step(interval, *accepted_, *candidate, bind_fields(*candidate));
            !status.ok())
            return status;
        if (auto status = complete_step(*candidate); !status.ok())
            return status;
        trial_ = std::move(candidate);
        stage_ = Stage::advanced;
        return {};
    });
}

Status WcsphParticipant3D::prepare_commit(const TimeInterval& interval) {
    if (stage_ != Stage::advanced || !matches(interval))
        return {StatusCode::invalid_state, "no matching advanced 3D SPH window"};
    stage_ = Stage::failed;
    return guarded([&]() -> Status {
        auto stamp = trial_->stamp;
        stamp.accepted = true;
        stamp.trial = {};
        stamp.accepted_version = StateVersion{stamp.time.window};

        decltype(trial_->field_storage) storage;
        auto density = trial_->density->view();
        auto pressure = trial_->pressure->view();
        auto acceleration = trial_->acceleration->view();
        auto velocity = trial_->velocity->view();
        storage = std::move(trial_->field_storage);
        trial_->density.reset();
        trial_->pressure.reset();
        trial_->acceleration.reset();
        trial_->velocity.reset();

        auto candidate = std::make_unique<WcsphState3D>();
        candidate->density.emplace(read_field(density, density_field(FieldAccess::read), stamp));
        candidate->pressure.emplace(read_field(pressure, pressure_field(FieldAccess::read), stamp));
        candidate->acceleration.emplace(
            read_field(acceleration, acceleration_field(FieldAccess::read), stamp));
        candidate->velocity.emplace(read_field(velocity, velocity_field(FieldAccess::read), stamp));
        candidate->field_storage = std::move(storage);
        candidate->phase_storage = std::move(trial_->phase_storage);
        candidate->wall_phases = std::move(trial_->wall_phases);
        candidate->geometry = std::move(trial_->geometry);
        candidate->maximum_step = trial_->maximum_step;
        candidate->stamp = stamp;
        pending_ = std::move(candidate);
        stage_ = Stage::precommitted;
        return {};
    });
}

Status WcsphParticipant3D::commit_window(const TimeInterval& interval) {
    if (stage_ != Stage::precommitted || !matches(interval))
        return {StatusCode::invalid_state, "no matching prepared 3D SPH commit"};
    accepted_.swap(pending_);
    step_->commit();
    close_window();
    return {};
}

Status WcsphParticipant3D::restore_window() {
    if (evaluating_ || (stage_ != Stage::open && stage_ != Stage::advanced &&
                        stage_ != Stage::precommitted && stage_ != Stage::failed))
        return {StatusCode::invalid_state, "no 3D SPH window to restore"};
    close_window();
    step_->restore();
    return {};
}

Status WcsphParticipant3D::finalize() {
    if (stage_ == Stage::finalized || reading_ || evaluating_)
        return {StatusCode::invalid_state, "3D SPH cannot finalize during an active operation or after closing"};
    close_window();
    accepted_.reset();
    step_.reset();
    initial_ = {};
    initial_velocity_ = {};
    initial_density_ = {};
    decltype(initial_walls_){}.swap(initial_walls_);
    exec_.reset();
    stage_ = Stage::finalized;
    return {};
}

Status WcsphParticipant3D::visit_committed_rows(
    const std::function<void(CommittedRow3D)>& visit) const {
    if (!can_read())
        return {StatusCode::invalid_state, "3D SPH committed rows are unavailable"};
    const ActiveScope access(reading_);
    try {
        const auto& state = *accepted_;
        const auto& geometry = state.geometry;
        const auto& fields = geometry.fields();
        const auto density = state.density->view(), pressure = state.pressure->view();
        const auto acceleration = state.acceleration->view(), velocity = state.velocity->view();
        const auto& rows = geometry.owned_row_to_input;
        for (std::size_t row = 0; row < rows.extent(0); ++row) {
            const auto slot = rows(row);
            visit({fields.ids(slot), fields.mass(slot), vector_at(fields.position, slot),
                   density(row), pressure(row), vector_at(acceleration, row),
                   vector_at(velocity, row)});
        }
        return {};
    } catch (...) {
        return {StatusCode::execution_failure, "3D SPH committed row callback failed"};
    }
}

Result<WallValue3D> WcsphParticipant3D::read_committed_wall(std::uint64_t id) const {
    if (!can_read())
        return Status{StatusCode::invalid_state, "3D SPH committed wall is unavailable"};
    const ActiveScope access(reading_);
    const auto& state = *accepted_;
    const auto& fields = state.geometry.fields();
    for (std::size_t slot = 0; slot < fields.ids.extent(0); ++slot) {
        if (fields.ghost(slot) && fields.ids(slot) == id) {
            const auto time = step_->wall_time(id, state.wall_phases);
            if (!time.ok())
                return time.status();
            return WallValue3D{id, fields.mass(slot), vector_at(fields.position, slot), state.stamp,
                               time.value().initial_phase + time.value().elapsed};
        }
    }
    return Status{StatusCode::invalid_argument, "3D SPH wall ID is unknown"};
}

Status WcsphParticipant3D::release_workspace() {
    if (!can_read())
        return {StatusCode::invalid_state, "3D SPH workspace release requires an idle participant"};
    step_->release_workspace();
    return {};
}

bool WcsphParticipant3D::matches(const TimeInterval& interval) const {
    return interval.begin == interval_.begin && interval.end == interval_.end &&
           interval.duration() == interval_.duration();
}

bool WcsphParticipant3D::can_read() const {
    return !reading_ && (stage_ == Stage::initialized || stage_ == Stage::ready);
}

void WcsphParticipant3D::close_window() noexcept {
    pending_.reset();
    trial_.reset();
    recovery_.reset();
    interval_ = {};
    stage_ = Stage::ready;
}

} // namespace nexora::sph
