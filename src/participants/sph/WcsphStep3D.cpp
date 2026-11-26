#include "WcsphStep3D.hpp"

#include <algorithm>
#include <data/EntityCapacity.hpp>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
using Budget = detail::ResourceBudget;
using Ticket = Budget::Ticket;
using namespace SphBoundary;
struct Failure { Status status; };

void require(Status status) {
    if (!status.ok())
        throw Failure{std::move(status)};
}
template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw Failure{result.status()};
    return std::move(result).value();
}
template <class F> Status guarded(F&& operation) {
    try {
        operation();
        return {};
    } catch (const Failure& f) {
        return f.status;
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate WCSPH step storage"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "WCSPH step operation failed"};
    }
}

template <class V> V allocate(const Kokkos::Serial& exec, const char* label, std::size_t n) {
    return V(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(label)), n);
}

template <class... V> bool valid_extent(std::size_t n, const V&... fields) {
    return ((fields.extent(0) == n && (!n || fields.data())) && ...);
}

void stamp_geometry(WcsphState3D& state) {
    state.stamp.topology = state.geometry.topology;
    state.stamp.geometry = state.geometry.geometry;
    state.stamp.layout = state.geometry.layout;
}

// An initial copy is needed only for wall assembly or when grid packing changes
// fluid input order. All four input arrays in this copy share one lifetime.
struct InitialInput {
    Ticket ticket;
    ParticleFields3D fields;
};

template <class CaptureWalls>
InitialInput capture_input(const Kokkos::Serial& exec, const Budget& budget,
                           const ParticleFields3D& fluid, std::size_t nw, const CaptureWalls& capture_walls) {
    const auto nf = fluid.ids.extent(0);
    const auto n = nf + nw; // Bounded by initial source admission.
    auto ticket = require(budget.reserve(
        n * (4 * sizeof(real) + sizeof(std::uint64_t) + sizeof(std::uint8_t))));
    auto position = allocate<Velocity3D>(exec, "sph.geometry.input_position", n);
    auto mass = allocate<Values>(exec, "sph.geometry.input_mass", n);
    auto ids = allocate<decltype(fluid.ids)::non_const_type>(exec, "sph.geometry.input_ids", n);
    auto ghost = allocate<decltype(fluid.ghost)::non_const_type>(exec, "sph.geometry.input_ghost", n);
    Kokkos::parallel_for("sph.geometry.capture_fluid", Policy(exec, 0, nf), [&](std::size_t i) {
        ghost(i) = 0;
        ids(i) = fluid.ids(i);
        mass(i) = fluid.mass(i);
        for (int k = 0; k < 3; ++k)
            position(i, k) = fluid.position(i, k);
    });
    if (nw)
        capture_walls(position, mass, ids, ghost);
    exec.fence("sph.geometry.initial_inputs_ready");
    return {std::move(ticket), {position, mass, ids, ghost}};
}

// Budget admission can release scratch whose access has finished or has yet to begin.
template <class Reserve, class Trim>
auto reserve_with_eviction(const Reserve& reserve, const Trim& trim) {
    auto result = reserve();
    if (!result.ok() && result.status().code() == StatusCode::resource_exhausted && trim()) {
        return reserve();
    }
    return result;
}

} // namespace

WcsphStep3D::WcsphStep3D(const Kokkos::Serial& exec, const Budget& budget,
                         std::size_t max_cells, WcsphVerletParameters parameters)
    : exec_(exec), budget_(budget), max_cells_(max_cells), parameters_(parameters) {}

bool WcsphStep3D::Workspace::release_rates() noexcept {
    const bool released = old_row || rate_n || rate_star;
    old_row.reset();
    rate_n.reset();
    rate_star.reset();
    return released;
}
bool WcsphStep3D::Workspace::release_continuity() noexcept {
    const bool released = release_rates() || corrected || stage;
    corrected.reset();
    stage.reset();
    return released;
}

const SolidWallBoundary3D& WcsphStep3D::WallGroup::source() const {
    return std::visit([](const auto& group) -> const SolidWallBoundary3D& { return group; }, boundary);
}

const PrescribedMovingWall3D* WcsphStep3D::WallGroup::moving() const {
    return std::get_if<PrescribedMovingWall3D>(&boundary);
}

void WcsphStep3D::capture_groups(const std::vector<WallGroupInput3D>& inputs) {
    if (inputs.empty())
        return;
    const auto groups = require(detail::checked_array_bytes(inputs.size(), sizeof(WallGroup)));
    std::size_t bytes = wall_count_ * sizeof(WallParticle3D);
    const auto add_bytes = [&](std::size_t extra) {
        if (extra > std::numeric_limits<std::size_t>::max() - bytes)
            throw Failure{{StatusCode::resource_exhausted, "WCSPH wall group storage exceeds capacity"}};
        bytes += extra;
    };
    add_bytes(groups);
    for (const auto& input : inputs) {
        if (input.trajectory)
            add_bytes(input.trajectory->motion.retained_bytes());
    }
    wall_storage_ = require(budget_.reserve(bytes));
    walls_.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (std::any_of(walls_.begin(), walls_.end(), [&](const auto& g) { return g.id == input.id; }))
            throw Failure{{StatusCode::invalid_argument, "duplicate WCSPH wall group ID"}};
        auto particles = require(SolidWallBoundary3D::capture_for_grid(
            exec_, input.particles, input.trajectory.has_value()));
        if (input.trajectory) {
            PrescribedMovingWall3D moving(particles, *input.trajectory);
            walls_.push_back({input.id, moving_groups_++, std::move(moving)});
        } else {
            walls_.push_back({input.id, 0, FixedWall3D(std::move(particles))});
        }
    }
}

WcsphStep3D::Ticket WcsphStep3D::reserve_wall_stage(std::size_t bytes) {
    // Wall clocks and samples are created before the first half kick. Any work
    // buffers left by a previous step are disposable here.
    auto reserved = budget_.reserve(bytes);
    if (!reserved.ok()) {
        const bool released = workspace_.release_continuity() || workspace_.half;
        workspace_.half.reset();
        if (released)
            reserved = budget_.reserve(bytes);
    }
    return require(std::move(reserved));
}

WallPhases3D::non_const_type WcsphStep3D::phase_storage(WcsphState3D& state) {
    if (!moving_groups_)
        return {};
    state.phase_storage = reserve_wall_stage(require(detail::checked_array_bytes(moving_groups_, sizeof(WallPhase3D))));
    return allocate<WallPhases3D::non_const_type>(exec_, "sph.wall.group_phases", moving_groups_);
}

WcsphStep3D::WallMotions WcsphStep3D::sample_groups(const WallPhases3D& phases, WallMotionUse3D use) {
    WallMotions motions;
    if (!moving_groups_)
        return motions;
    const auto bytes = require(detail::checked_array_bytes(moving_groups_, sizeof(SampledWallMotion3D)));
    motions.ticket = reserve_wall_stage(bytes);
    motions.values = decltype(motions.values)(
        Kokkos::view_alloc(exec_, "sph.wall.stage_motions"), moving_groups_);
    for (const auto& group : walls_) {
        const auto* moving = group.moving();
        if (moving && moving->particles().extent(0))
            motions.values(group.phase_index) = require(
                moving->motion_at_certified_time(phases(group.phase_index).time, use));
    }
    return motions;
}

void WcsphStep3D::set_next_step(WcsphState3D& candidate, real largest_speed_squared) const {
    candidate.maximum_step = require(verlet_step::acoustic_step(largest_speed_squared,
        parameters_.eos.sound_speed, parameters_.cfl, parameters_.smoothing_length));
    for (std::size_t g = 0; g < candidate.wall_phases.extent(0); ++g)
        candidate.maximum_step = std::min(candidate.maximum_step, candidate.wall_phases(g).maximum_step);
}

bool WcsphStep3D::write_wall_slot(const ParticleFields3D& fields, const WallMotions& motions,
                                 const Velocity3D& output, LocalIndex slot, bool positions) const {
    for (const auto& group : walls_) {
        const auto* moving = group.moving();
        if (!moving)
            continue;
        const auto* particle = moving->find(fields.ids(slot));
        if (!particle)
            continue;
        auto& motion = motions.values(group.phase_index);
        if (positions) {
            const auto position = motion.position(particle->reference_position);
            for (int k = 0; k < 3; ++k)
                output(slot, k) = position[k];
            return true;
        }
        const auto sampled = motion.velocity(particle->reference_position);
        if (!sampled)
            return false;
        for (int k = 0; k < 3; ++k)
            output(slot, k) = (*sampled)[k];
        return true;
    }
    for (int k = 0; k < 3; ++k)
        output(slot, k) = positions ? fields.position(slot, k) : real{0};
    return true;
}

void WcsphStep3D::write_wall_positions(const ParticleFields3D& fields, const WallMotions& motions,
                                       const Velocity3D& output) const {
    if (!wall_count_)
        return;
    if (!moving_groups_) {
        Kokkos::parallel_for("sph.wall.fixed_stage", Policy(exec_, 0, fields.ids.extent(0)),
            [&](std::size_t slot) {
                if (fields.ghost(slot))
                    for (int k = 0; k < 3; ++k)
                        output(slot, k) = fields.position(slot, k);
            });
        return;
    }
    // Only drift needs all moving wall slots. Continuity samples their velocity on demand.
    // update_positions checks every drifted coordinate before converting it to an index.
    Kokkos::parallel_for("sph.wall.drift_position", Policy(exec_, 0, fields.ids.extent(0)),
        [&](std::size_t slot) {
            if (fields.ghost(slot))
                write_wall_slot(fields, motions, output, static_cast<LocalIndex>(slot), true);
        });
}

WallTrajectoryTime3D WcsphStep3D::committed_wall_time(std::uint64_t particle_id, const WallPhases3D& phases) const {
    for (const auto& group : walls_)
        if (const auto* moving = group.moving(); moving && moving->find(particle_id))
            return phases(group.phase_index).time;
    return {}; // A committed ghost outside the moving groups belongs to a fixed wall.
}

void WcsphStep3D::evaluate_pressure(WcsphState3D& state, const CellGrid& grid,
                                     const BindFields3D& bind, Workspace* work, const WallMotions& motions) {
    const auto& q = grid.neighbors();
    const auto& sources = q.fields();
    const auto n = sources.ids.extent(0), rows = q.owned_slots().extent(0);
    const auto trim = [&] { return work && work->release_continuity(); };
    const auto write = [&](const StepOutputs3D& out) -> Status {
        Ticket scratch;
        if (rows) {
            const auto stride = step_pressure_workspace_bytes_per_source(n != rows);
            const auto bytes = n * stride;
            auto reserved = reserve_with_eviction([&] { return budget_.reserve(bytes); }, trim);
            if (!reserved.ok())
                return reserved.status();
            scratch = std::move(reserved).value();
        }
        const auto acceleration_at = [&](LocalIndex slot) {
            for (const auto& group : walls_) {
                if (const auto* moving = group.moving()) {
                    if (const auto* particle = moving->find(sources.ids(slot)))
                        return motions.values(group.phase_index).acceleration(particle->reference_position);
                }
            }
            return std::array<real, 3>{};
        };
        const TaitEos eos(parameters_.eos);
        return evaluate_step_pressure_3d(exec_, q, eos, state.density->view(), out.pressure,
                                         out.acceleration, std::cref(acceleration_at), parameters_.gravity, work == nullptr);
    };
    require(bind(FieldGroup3D::pressure, write, trim));
}

Status WcsphStep3D::initialize_state(const ParticleFields3D& fluid,
                                     const Velocity3D::const_type& velocity,
                                     const Values::const_type& density,
                                     const std::vector<WallGroupInput3D>& initial_walls,
                                     WcsphState3D& candidate, const BindFields3D& bind) {
    return guarded([&] {
        require(parameters_.validate());
        const auto nf = fluid.position.extent(0);
        if (!valid_extent(nf, fluid.position, fluid.ids, fluid.mass, fluid.ghost, velocity, density))
            throw Failure{{StatusCode::invalid_argument, "WCSPH fluid field extents disagree"}};
        std::size_t remaining = std::numeric_limits<LocalIndex>::max();
        const auto admit_sources = [&](std::size_t count) {
            if (count > remaining)
                throw Failure{{StatusCode::resource_exhausted, "WCSPH source count exceeds capacity"}};
            remaining -= count;
        };
        admit_sources(nf);
        for (const auto& group : initial_walls)
            admit_sources(group.particles.ids.extent(0));
        wall_count_ = std::numeric_limits<LocalIndex>::max() - remaining - nf;
        std::size_t ghosts{};
        Kokkos::parallel_reduce("sph.wall.validate_ownership", Policy(exec_, 0, nf),
            [&](std::size_t i, std::size_t& count) { count += fluid.ghost(i) != 0; }, ghosts);
        if (ghosts)
            throw Failure{{StatusCode::unsupported, "WCSPH requires all-owned fluid input"}};
        capture_groups(initial_walls);
        const auto phases = phase_storage(candidate);
        for (const auto& group : walls_)
            if (const auto* moving = group.moving()) {
                const auto time = moving->initial_time();
                phases(group.phase_index) = {group.id, time, require(moving->initial_step_limit())};
            }
        candidate.wall_phases = phases;
        const auto samples = sample_groups(phases, nf ? WallMotionUse3D::full : WallMotionUse3D::position);
        const auto capture_walls = [&](const auto& position, const auto& mass,
                                       const auto& ids, const auto& ghost) {
            std::size_t offset = nf;
            for (const auto& group : walls_) {
                const auto& particles = group.source().particles();
                const auto count = particles.extent(0);
                const auto write = [&](std::size_t w, const std::array<real, 3>& x) {
                    const auto i = offset + w;
                    ghost(i) = 1;
                    ids(i) = particles(w).id;
                    mass(i) = particles(w).mass;
                    for (int k = 0; k < 3; ++k)
                        position(i, k) = x[k];
                };
                if (group.moving()) {
                    // Grid assembly validates the sampled coordinates before publication.
                    Kokkos::parallel_for("sph.geometry.capture_moving_walls", Policy(exec_, 0, count),
                        [&](std::size_t w) {
                            write(w, samples.values(group.phase_index).position(particles(w).reference_position));
                        });
                } else {
                    // The initial grid validates fixed coordinates and masses.
                    Kokkos::parallel_for("sph.geometry.capture_fixed_walls", Policy(exec_, 0, count),
                        [&](std::size_t w) { write(w, particles(w).reference_position); });
                }
                offset += count;
            }
        };
        {
            std::optional<InitialInput> captured;
            if (wall_count_)
                captured.emplace(capture_input(exec_, budget_, fluid, wall_count_, capture_walls));
            const auto admit = [&](CellGrid::StorageGroup, std::size_t bytes) { return budget_.reserve(bytes); };
            auto grid = require(CellGrid::build_initial(exec_, captured ? captured->fields : fluid,
                                                        nf, max_cells_, admit,
                                                        parameters_.smoothing_length));
            auto& geometry = candidate.geometry;
            if (grid.input_is_sorted()) {
                geometry.particle_storage = grid.field_storage();
                geometry.input = grid.fields();
            } else {
                if (!captured)
                    captured.emplace(capture_input(exec_, budget_, fluid, 0, capture_walls));
                geometry.particle_storage.fill(captured->ticket);
                geometry.input = captured->fields;
            }
            require(establish_owned_rows(exec_, geometry, grid.owned_slots(), grid.sorted_to_input(),
                                         [&](std::size_t bytes) { return budget_.reserve(bytes); }));
            committed_grid_ = std::make_unique<CellGrid>(std::move(grid));
        }
        stamp_geometry(candidate);
        const auto& map = candidate.geometry.owned_row_to_input;
        const auto rows = Policy(exec_, 0, map.extent(0));
        require(bind(FieldGroup3D::density,
            [&](const StepOutputs3D& out) -> Status {
                Kokkos::parallel_for("sph.continuity.initial_density", rows,
                    [&](std::size_t r) { out.density(r) = density(map(r)); });
                return {};
            }, {}));
        real largest_speed_squared{};
        require(bind(FieldGroup3D::velocity,
            [&](const StepOutputs3D& out) -> Status {
                const auto speed = verlet_step::capture_velocity(exec_, map, velocity, out.velocity);
                if (speed.ok())
                    largest_speed_squared = speed.value();
                return speed.status();
            }, {}));
        evaluate_pressure(candidate, *committed_grid_, bind, nullptr, samples);
        set_next_step(candidate, largest_speed_squared);
    });
}

Status WcsphStep3D::try_step(const TimeInterval& interval, const WcsphState3D& current,
                             WcsphState3D& candidate, const BindFields3D& bind) {
    const double requested_step = interval.duration();
    const real dt = static_cast<real>(requested_step);
    if (!(dt > 0))
        return {StatusCode::invalid_argument, "WCSPH step rounds to zero in real precision"};

    const auto status = guarded([&] {
        const auto phases = phase_storage(candidate);
        // Validate every group's unwrapped interval before asking any trajectory for an endpoint.
        for (const auto& group : walls_) {
            if (const auto* moving = group.moving()) {
                const auto index = group.phase_index;
                const auto& phase = current.wall_phases(index);
                const auto [time, limit] = require(
                    moving->advance_certified_time(phase.time, requested_step, phase.maximum_step));
                phases(index) = {phase.group_id, time, limit};
            }
        }
        const bool use_stage_velocity = current.geometry.owned_row_to_input.extent(0);
        auto start_samples = use_stage_velocity ? sample_groups(current.wall_phases, WallMotionUse3D::velocity) : WallMotions{};
        auto end_samples = sample_groups(phases, use_stage_velocity ? WallMotionUse3D::full : WallMotionUse3D::position);
        candidate.wall_phases = phases;
        auto& work = workspace_;
        // These hot buffers have no live contents until after the drifted grid is ready.
        const auto trim_drift = [&] {
            const bool released = work.release_rates() || work.corrected;
            work.corrected.reset();
            return released;
        };
        const auto reserve_geometry = [&](std::size_t bytes) {
            return reserve_with_eviction([&] { return budget_.reserve(bytes); }, trim_drift);
        };
        const auto admit_grid = [&](CellGrid::StorageGroup, std::size_t bytes) {
            return reserve_geometry(bytes);
        };
        std::optional<CellGrid> rebuilt_grid;
        if (!committed_grid_) {
            // Accepted geometry owns a read-only snapshot from a successful grid
            // build/update. Recreate its derived index without certifying the same
            // values or checking the sort/gather algorithm's own mappings again.
            rebuilt_grid.emplace(require(CellGrid::rebuild_committed(
                exec_, current.geometry.fields(), current.geometry.owned_row_to_input.extent(0),
                max_cells_, admit_grid, parameters_.smoothing_length)));
        }
        const auto& old = committed_grid_ ? *committed_grid_ : *rebuilt_grid;
        const auto& old_fields = old.fields();
        const auto& old_owned = old.owned_slots();
        const auto n = old_fields.ids.extent(0), nf = old_owned.extent(0);
        const auto ensure = [&](auto& buffer, std::size_t count, std::size_t stride,
                                const char* label, const auto& trim) {
            if (!buffer) {
                using V = std::remove_cvref_t<decltype(buffer->values)>;
                const auto bytes = count * stride;
                auto ticket = require(reserve_with_eviction([&] { return budget_.reserve(bytes); }, trim));
                buffer.emplace(std::move(ticket), allocate<V>(exec_, label, count));
            }
        };
        const auto ensure_stage = [&] {
            ensure(work.stage, n, 3 * sizeof(real), "sph.verlet.stage", trim_drift);
        };
        ensure(work.half, n, 3 * sizeof(real), "sph.verlet.half_velocity", trim_drift);
        ensure_stage();
        const auto& half = work.half->values;
        const auto& v = current.velocity->view();
        const auto& a = current.acceleration->view();
        auto& geometry = candidate.geometry;
        std::unique_ptr<CellGrid> next_grid;
        {
            const auto& drift = work.stage->values;
            verlet_step::kick_drift(exec_, old, v, a, dt, drift, half, parameters_.gravity);
            write_wall_positions(old_fields, end_samples, drift);
            const ParticleFields3D input{drift, old_fields.mass, old_fields.ids, old_fields.ghost};
            auto updated = old.update_positions(exec_, drift, max_cells_, admit_grid);
            if (updated.status().code() == StatusCode::invalid_argument)
                throw Failure{{StatusCode::numerical_failure, "invalid drifted WCSPH position"}};
            next_grid = std::make_unique<CellGrid>(require(std::move(updated)));
            const auto& next = *next_grid;
            if (next.input_is_sorted()) {
                geometry.particle_storage = next.field_storage();
                geometry.input = next.fields();
            } else {
                // The new grid's input was in old sorted order. Retain this exact position
                // allocation and the old co-ordered immutable fields before recycling stage.
                geometry.particle_storage = {work.stage->ticket, old.field_storage()[1], old.field_storage()[2]};
                geometry.input = input;
                work.stage.reset();
            }
        }
        const auto& next = *next_grid;
        // The new grid input follows old sorted slots and preserves their IDs and
        // ownership. A nonidentity new permutation moves distinct IDs; otherwise
        // the layout changes only if the previous input permutation was nonidentity.
        const bool layout_changed = !old.input_is_sorted() || !next.input_is_sorted();
        if (layout_changed) {
            require(establish_owned_rows(exec_, geometry, next.owned_slots(), next.sorted_to_input(),
                                         reserve_geometry));
        } else {
            geometry.row_storage = current.geometry.row_storage;
            geometry.owned_row_to_input = current.geometry.owned_row_to_input;
        }
        const auto next_geometry = next.positions_changed_
            ? require(detail::checked_next_version(current.geometry.geometry)) : current.geometry.geometry;
        const auto next_layout = layout_changed ? require(detail::checked_next_version(current.geometry.layout))
                                                : current.geometry.layout;
        geometry.topology = current.geometry.topology;
        geometry.geometry = next_geometry;
        geometry.layout = next_layout;
        stamp_geometry(candidate);
        ensure_stage();
        const auto no_trim = [] { return false; };
        ensure(work.old_row, n, sizeof(LocalIndex), "sph.continuity.old_row", no_trim);
        ensure(work.rate_n, nf, sizeof(real), "sph.continuity.accepted_rate", no_trim);
        const Policy rows(exec_, 0, nf);
        {
            const auto& stage = work.stage->values;
            const auto& old_row = work.old_row->values;
            Kokkos::parallel_for("sph.continuity.accepted_velocity", rows, [&](std::size_t r) {
                const auto s0 = old_owned(r);
                old_row(s0) = static_cast<LocalIndex>(r);
                for (int k = 0; k < 3; ++k)
                    stage(s0, k) = v(r, k);
            });
            const auto write_wall = [&](const ParticleFields3D& fields, const WallMotions& motions,
                                        LocalIndex slot) {
                return write_wall_slot(fields, motions, stage, slot, false);
            };
            std::optional<Buffer<StageVelocityReady3D>> ready;
            const bool use_lazy_prediction = nf && n > 1;
            if (use_lazy_prediction && wall_count_)
                ensure(ready, n, sizeof(std::uint8_t), "sph.continuity.velocity_ready", no_trim);
            if (ready && !moving_groups_) {
                Kokkos::parallel_for("sph.wall.fixed_stage", Policy(exec_, 0, n), [&](std::size_t slot) {
                    if (old_fields.ghost(slot))
                        for (int k = 0; k < 3; ++k)
                            stage(slot, k) = real{0};
                });
            }
            if (ready && moving_groups_) {
                const auto write = [&](LocalIndex slot) { return write_wall(old_fields, start_samples, slot); };
                require(stage_wall_continuity_rate_3d(exec_, old.neighbors(), stage, work.rate_n->values,
                    ready->values, std::cref(write)));
            } else {
                stage_continuity_rate_3d(exec_, old.neighbors(), stage, work.rate_n->values);
            }
            start_samples = {};

            const ContinuityPredictor3D predictor{v, a, dt, parameters_.gravity};
            ensure(work.rate_star, nf, sizeof(real), "sph.continuity.predicted_rate", no_trim);
            if (ready) {
                const auto write = [&](LocalIndex slot) { return write_wall(next.fields(), end_samples, slot); };
                require(stage_predicted_continuity_rate_3d(exec_, next.neighbors(), stage, work.rate_star->values,
                    ready->values, predictor, next.sorted_to_input(), old_row, std::cref(write)));
                ready.reset();
            } else if (use_lazy_prediction) {
                // No wall slots. The correction buffer can track prediction
                // readiness until the Serial rate kernel returns.
                ensure(work.corrected, nf, sizeof(real), "sph.continuity.density", no_trim);
                require(stage_predicted_continuity_rate_3d(exec_, next.neighbors(), stage, work.rate_star->values,
                    work.corrected->values, predictor, next.sorted_to_input(), old_row));
            } else {
                stage_continuity_rate_3d(exec_, next.neighbors(), stage, work.rate_star->values);
            }
            ensure(work.corrected, nf, sizeof(real), "sph.continuity.density", no_trim);
            require(correct_density_3d(exec_, next.owned_slots(), next.sorted_to_input(), old_row,
                               current.density->view(), work.rate_n->values, work.rate_star->values,
                               dt, work.corrected->values));
        }
        require(bind(FieldGroup3D::density, [&](const StepOutputs3D& out) -> Status {
            Kokkos::deep_copy(exec_, out.density, work.corrected->values);
            exec_.fence("sph.continuity.trial_density_ready");
            return {};
        }, [&] {
            const bool released = work.release_rates() || work.stage;
            work.stage.reset();
            return released;
        }));
        evaluate_pressure(candidate, next, bind, &work, end_samples);
        end_samples.values = {};
        end_samples.ticket.reset();
        real largest_speed_squared{};
        require(bind(FieldGroup3D::velocity, [&](const StepOutputs3D& out) -> Status {
            const auto speed = verlet_step::finish_kick(exec_, next, half, candidate.acceleration->view(),
                                                        dt, out.velocity, parameters_.gravity);
            if (speed.ok())
                largest_speed_squared = speed.value();
            return speed.status();
        }, [&] { return work.release_continuity(); }));
        set_next_step(candidate, largest_speed_squared);
        committed_grid_ = std::move(next_grid);
    });
    if (!status.ok())
        workspace_ = {};
    return status;
}

void WcsphStep3D::release_workspace() noexcept {
    workspace_ = {};
    committed_grid_.reset();
}
} // namespace nexora::sph
