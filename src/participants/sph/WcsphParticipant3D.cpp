#include "WcsphParticipant3D.hpp"

#include <cmath>
#include <data/EntityCapacity.hpp>
#include <data/FieldIssuer.hpp>
#include <new>
#include <string>
#include <utility>

namespace nexora::sph {
namespace {

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
auto read_field(const View& values, FieldDescriptor descriptor, const FieldStamp& stamp) {
    auto issuer = require(detail::FieldIssuer::create_at_validated_time(values, stamp));
    const auto axes = owned_axes(values);
    const detail::FieldBindingSpec spec{Association::particle, descriptor.units, axes};
    return require(issuer.read(values, descriptor, issuer.stamp(), spec));
}

template <class View>
auto write_field(const View& values, FieldDescriptor descriptor, FieldStamp stamp) {
    stamp.accepted = false;
    auto issuer = require(detail::FieldIssuer::create_at_validated_time(values, stamp));
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
    return [this, &candidate](FieldGroup3D group, const FieldWrite3D& write,
                              const std::function<bool()>& evict_unused) {
        return guarded([&]() -> Status {
            const auto rows = candidate.geometry.owned_row_to_input.extent(0);
            const auto& stamp = candidate.stamp;
            const auto reserve = [&](std::size_t components) {
                const auto bytes = require(detail::checked_array_bytes(rows, components * sizeof(real)));
                auto storage = budget_.reserve(bytes);
                if (!storage.ok() && storage.status().code() == StatusCode::resource_exhausted &&
                    evict_unused && evict_unused())
                    storage = budget_.reserve(bytes);
                return require(std::move(storage));
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
                candidate.field_storage[0] = std::move(storage);
                candidate.density.emplace(std::move(field));
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
                candidate.field_storage[1] = std::move(storage);
                candidate.pressure.emplace(std::move(pressure));
                candidate.acceleration.emplace(std::move(acceleration));
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
                candidate.field_storage[2] = std::move(storage);
                candidate.velocity.emplace(std::move(velocity));
                return {};
            }
            }
            return {StatusCode::invalid_state, "unknown WCSPH field grant group"};
        });
    };
}

TimeConstraint WcsphParticipant3D::time_constraint() const {
    TimeConstraint result;
    if (accepted_)
        result.maximum_step = accepted_->maximum_step;
    return result;
}

Status WcsphParticipant3D::initialize(double time) {
    if (!std::isfinite(time))
        return {StatusCode::invalid_argument, "invalid 3D SPH initial time"};
    return initialize_at_finite_time(time);
}

Status WcsphParticipant3D::initialize_at_finite_time(double time) {
    if (finalized_ || accepted_ || active_ || !Kokkos::is_initialized() || Kokkos::is_finalized())
        return {StatusCode::invalid_state,
                "3D SPH initialization requires a new participant and active Kokkos"};

    const ActiveScope evaluation(active_);
    const auto result = guarded([&]() -> Status {
        step_.emplace(exec_.emplace(), budget_, max_cells_, parameters_);
        step_->observer_context_ = observer_context_;
        step_->observe_ = observe_;
        step_->observe_kinematics_ = observe_kinematics_;
        auto candidate = std::make_unique<WcsphState3D>();
        auto& stamp = candidate->stamp;
        stamp.accepted = true;
        stamp.time.physical_time = stamp.valid_from = stamp.valid_until = time;
        if (auto status = step_->initialize_state(initial_, initial_velocity_, initial_density_,
                                                  initial_walls_, *candidate, bind_fields(*candidate));
            !status.ok())
            return status;
        accepted_ = std::move(candidate);
        initial_ = {};
        initial_velocity_ = {};
        initial_density_ = {};
        decltype(initial_walls_){}.swap(initial_walls_);
        return {};
    });
    if (!result.ok()) {
        step_.reset();
        exec_.reset();
    }
    return result;
}

Status WcsphParticipant3D::do_step(const TimeInterval& interval) {
    if (!idle_with_state())
        return {StatusCode::invalid_state, "3D SPH step requires an idle initialized participant"};
    if (interval.begin != accepted_->stamp.time.physical_time)
        return {StatusCode::invalid_argument, "3D SPH step must begin at the accepted time"};

    const ActiveScope evaluation(active_);
    return guarded([&]() -> Status {
        auto candidate = std::make_unique<WcsphState3D>();
        auto& stamp = candidate->stamp;
        stamp.accepted = true;
        stamp.accepted_version = require(detail::checked_next_version(accepted_->stamp.accepted_version));
        stamp.time = {interval.end, stamp.accepted_version.value(), 0, 0};
        stamp.valid_from = stamp.valid_until = interval.end;
        if (auto status = step_->try_step(interval, *accepted_, *candidate, bind_fields(*candidate));
            !status.ok())
            return status;
        accepted_ = std::move(candidate);
        return {};
    });
}

Status WcsphParticipant3D::finalize() {
    if (finalized_ || active_)
        return {StatusCode::invalid_state, "3D SPH cannot finalize during an active operation or after closing"};
    accepted_.reset();
    step_.reset();
    initial_ = {};
    initial_velocity_ = {};
    initial_density_ = {};
    decltype(initial_walls_){}.swap(initial_walls_);
    exec_.reset();
    finalized_ = true;
    return {};
}

Status WcsphParticipant3D::visit_committed_rows(
    const std::function<void(CommittedRow3D)>& visit) const {
    if (!idle_with_state())
        return {StatusCode::invalid_state, "3D SPH committed rows are unavailable"};
    const ActiveScope access(active_);
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
    if (!idle_with_state())
        return Status{StatusCode::invalid_state, "3D SPH committed wall is unavailable"};
    const ActiveScope access(active_);
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
    if (!idle_with_state())
        return {StatusCode::invalid_state, "3D SPH workspace release requires an idle participant"};
    step_->release_workspace();
    return {};
}

bool WcsphParticipant3D::idle_with_state() const {
    return accepted_ && !active_;
}

} // namespace nexora::sph
