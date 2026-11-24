#include "WcsphStep3D.hpp"
#include "GeometryVersions3D.hpp"

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
using Boundary = SphBoundary::FixedWall3D;
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

bool input_is_sorted(const Kokkos::Serial& exec, const CellGrid& grid) {
    const auto slots = grid.sorted_to_input();
    std::size_t mismatches{};
    Kokkos::parallel_reduce("sph.geometry.check_input_order", Policy(exec, 0, slots.extent(0)),
        [&](std::size_t s, std::size_t& count) { count += slots(s) != s; }, mismatches);
    return !mismatches;
}

GeometryLayout3D layout_of(const CellGrid& grid) {
    return {grid.fields(), grid.owned_slots(), grid.sorted_to_input(), grid.input_to_sorted()};
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

InitialInput capture_input(const Kokkos::Serial& exec, const Budget& budget,
                           const ParticleFields3D& fluid, const Boundary* walls) {
    const auto nf = fluid.ids.extent(0), nw = walls ? walls->particles().extent(0) : 0;
    const auto n = nf + nw; // initialize_state verifies the sum before any allocation
    auto ticket = require(budget.reserve(require(detail::checked_array_bytes(
        n, 4 * sizeof(real) + sizeof(std::uint64_t) + sizeof(std::uint8_t)))));
    auto position = allocate<Velocity3D>(exec, "sph.geometry.input_position", n);
    auto mass = allocate<Values>(exec, "sph.geometry.input_mass", n);
    auto ids = allocate<decltype(fluid.ids)::non_const_type>(exec, "sph.geometry.input_ids", n);
    auto ghost = allocate<decltype(fluid.ghost)::non_const_type>(exec, "sph.geometry.input_ghost", n);
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.geometry.capture_input", Policy(exec, 0, n),
                            [&](std::size_t i, std::size_t& errors) {
        ghost(i) = i >= nf;
        if (i < nf) {
            ids(i) = fluid.ids(i);
            mass(i) = fluid.mass(i);
            for (int k = 0; k < 3; ++k)
                position(i, k) = fluid.position(i, k);
        } else {
            const auto& wall = walls->particles()(i - nf);
            const auto sample = walls->sample(wall.id, 0);
            if (!sample.ok()) {
                ++errors;
                return;
            }
            ids(i) = wall.id;
            mass(i) = wall.mass;
            for (int k = 0; k < 3; ++k)
                position(i, k) = sample.value().position[k];
        }
    }, invalid);
    if (invalid)
        throw Failure{{StatusCode::invalid_state, "cannot sample initial WCSPH wall"}};
    exec.fence("sph.geometry.initial_inputs_ready");
    return {std::move(ticket), {position, mass, ids, ghost}};
}

template <class Write>
void sample_walls(const Kokkos::Serial& exec, const Boundary& boundary,
                  const ParticleFields3D& fields, const char* label, const Write& write) {
    std::size_t invalid{};
    Kokkos::parallel_reduce(label, Policy(exec, 0, fields.ids.extent(0)),
        [&](std::size_t s, std::size_t& errors) {
            if (fields.ghost(s)) {
                const auto sample = boundary.sample(fields.ids(s), 0);
                if (sample.ok())
                    write(s, sample.value());
                else
                    ++errors;
            }
        }, invalid);
    if (invalid)
        throw Failure{{StatusCode::invalid_state, "wall ID missing from WCSPH geometry"}};
}

// The operation has no published side effects on budget refusal. Retry only after
// arrays whose access has finished (or has yet to begin this step) are relinquished.
template <class Try, class Trim> auto with_eviction(const Try& attempt, const Trim& trim) {
    auto result = attempt();
    if (!result.ok() && result.status().code() == StatusCode::resource_exhausted && trim()) {
        return attempt();
    }
    return result;
}

template <class Trim>
void bind_group(WcsphState3D& state, FieldGroup3D group, const BindFields3D& bind,
                const FieldWrite3D& write, const Trim& trim) {
    auto status = bind(state, state.stamp, state.geometry.owned_row_to_input, group, write);
    if (status.code() == StatusCode::resource_exhausted && trim()) {
        status = bind(state, state.stamp, state.geometry.owned_row_to_input, group, write);
    }
    require(std::move(status));
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

void WcsphStep3D::evaluate_pressure(WcsphState3D& state, const CellGrid& grid,
                                     const BindFields3D& bind, Workspace* work) {
    const auto& q = grid.neighbors();
    const auto& sources = q.fields();
    const auto n = sources.ids.extent(0), rows = q.owned_slots().extent(0);
    const bool use_wall = walls_ && n != rows && rows;
    const auto trim = [&] { return work && work->release_continuity(); };
    const auto write = [&](const StepOutputs3D& out) -> Status {
        Ticket scratch;
        WallAcceleration3D wall_a;
        if (rows) {
            const auto stride = pressure_workspace_bytes_per_source(n != rows, use_wall) +
                                (use_wall ? 3 * sizeof(double) : 0);
            const auto bytes = require(detail::checked_array_bytes(n, stride));
            auto reserved = with_eviction([&] { return budget_.reserve(bytes); }, trim);
            if (!reserved.ok())
                return reserved.status();
            scratch = std::move(reserved).value();
        }
        if (use_wall) {
            wall_a = allocate<WallAcceleration3D>(exec_, "sph.wall.acceleration", n);
            sample_walls(exec_, *walls_, sources, "sph.wall.sample_acceleration",
                [&](std::size_t s, const auto& sample) {
                    for (int k = 0; k < 3; ++k) wall_a(s, k) = sample.acceleration[k];
                });
        }
        const TaitEos eos(parameters_.eos);
        return use_wall
            ? evaluate_pressure_3d(exec_, q, eos, state.density->view(), out.pressure,
                                   out.acceleration, wall_a, parameters_.gravity)
            : evaluate_pressure_3d(exec_, q, eos, state.density->view(), out.pressure, out.acceleration);
    };
    bind_group(state, FieldGroup3D::pressure, bind, write, trim);
}

Status WcsphStep3D::initialize_state(double, const ParticleFields3D& fluid,
                                     const Velocity3D::const_type& velocity,
                                     const Values::const_type& density,
                                     const std::optional<FixedWallParticles3D>& initial_walls,
                                     WcsphState3D& candidate, const BindFields3D& bind) {
    return guarded([&] {
        require(parameters_.validate());
        const auto nf = fluid.position.extent(0);
        if (!valid_extent(nf, fluid.position, fluid.ids, fluid.mass, fluid.ghost, velocity, density))
            throw Failure{{StatusCode::invalid_argument, "WCSPH fluid field extents disagree"}};
        std::size_t ghosts{};
        Kokkos::parallel_reduce("sph.wall.validate_ownership", Policy(exec_, 0, nf),
            [&](std::size_t i, std::size_t& count) { count += fluid.ghost(i) != 0; }, ghosts);
        if (ghosts)
            throw Failure{{StatusCode::unsupported, "WCSPH requires all-owned fluid input"}};
        const auto nw = initial_walls ? initial_walls->position.extent(0) : 0;
        if (nw > std::numeric_limits<std::size_t>::max() - nf ||
            !detail::checked_entity_count(nf + nw).ok())
            throw Failure{{StatusCode::resource_exhausted, "WCSPH source count exceeds capacity"}};
        if (initial_walls) {
            wall_storage_ = require(budget_.reserve(require(detail::checked_array_bytes(
                nw, sizeof(SphBoundary::WallParticle3D)))));
            walls_.emplace(require(SphBoundary::capture_wall_particles(exec_, *initial_walls)));
        }
        {
            std::optional<InitialInput> captured;
            if (nw)
                captured.emplace(capture_input(exec_, budget_, fluid, &*walls_));
            const auto admit = [&](CellGrid::StorageGroup, std::size_t bytes) { return budget_.reserve(bytes); };
            auto grid = require(CellGrid::build(exec_, captured ? captured->fields : fluid, max_cells_,
                                                admit, parameters_.smoothing_length));
            auto& geometry = candidate.geometry;
            if (input_is_sorted(exec_, grid)) {
                geometry.particle_storage = grid.field_storage();
                geometry.input = grid.fields();
            } else {
                if (!captured)
                    captured.emplace(capture_input(exec_, budget_, fluid, nullptr));
                geometry.particle_storage.fill(captured->ticket);
                geometry.input = captured->fields;
            }
            require(establish_owned_rows(exec_, geometry, grid.owned_slots(), grid.sorted_to_input(), budget_));
            candidate_grid_ = std::make_unique<CellGrid>(std::move(grid));
        }
        stamp_geometry(candidate);
        const auto& map = candidate.geometry.owned_row_to_input;
        const auto rows = Policy(exec_, 0, map.extent(0));
        require(bind(candidate, candidate.stamp, map, FieldGroup3D::density,
            [&](const StepOutputs3D& out) -> Status {
                Kokkos::parallel_for("sph.continuity.initial_density", rows,
                    [&](std::size_t r) { out.density(r) = density(map(r)); });
                return {};
            }));
        require(bind(candidate, candidate.stamp, map, FieldGroup3D::velocity,
            [&](const StepOutputs3D& out) -> Status {
                std::size_t invalid{};
                Kokkos::parallel_reduce("sph.verlet.initial_velocity", rows,
                    [&](std::size_t r, std::size_t& errors) {
                        for (int k = 0; k < 3; ++k) {
                            const auto v = velocity(map(r), k);
                            errors += !Kokkos::isfinite(v);
                            out.velocity(r, k) = v;
                        }
                    }, invalid);
                if (invalid)
                    return {StatusCode::invalid_argument, "WCSPH initial velocity must be finite"};
                return {};
            }));
        if (observe_)
            observe_(observer_context_, StepStage3D::initial, candidate_grid_->neighbors());
        evaluate_pressure(candidate, *candidate_grid_, bind, nullptr);
    });
}

Status WcsphStep3D::try_step(const TimeInterval& interval, const WcsphState3D& current,
                             WcsphState3D& candidate, const BindFields3D& bind) {
    return guarded([&] {
        const auto dt = interval.duration();
        auto& work = workspace_;
        // These hot buffers have no live contents until after the drifted grid is ready.
        const auto trim_drift = [&] {
            const bool released = work.release_rates() || work.corrected;
            work.corrected.reset();
            return released;
        };
        const auto admit_grid = [&](CellGrid::StorageGroup, std::size_t bytes) {
            return with_eviction([&] { return budget_.reserve(bytes); }, trim_drift);
        };
        if (!committed_grid_) {
            const auto& input = current.geometry.fields();
            auto rebuilt = require(CellGrid::build(exec_, input, max_cells_, admit_grid,
                                                   parameters_.smoothing_length));
            const auto& owned = rebuilt.owned_slots();
            const auto& rows = current.geometry.owned_row_to_input;
            if (owned.extent(0) != rows.extent(0))
                throw Failure{{StatusCode::invalid_state, "rebuilt WCSPH grid changed owned rows"}};
            const auto& sorted = rebuilt.fields();
            const auto& to_input = rebuilt.sorted_to_input();
            const auto& to_sorted = rebuilt.input_to_sorted();
            std::size_t mismatches{};
            Kokkos::parallel_reduce(
                "sph.geometry.verify_rebuilt_grid", Policy(exec_, 0, input.ids.extent(0)),
                [&](std::size_t slot, std::size_t& errors) {
                    const auto i = to_input(slot);
                    errors += to_sorted(i) != slot || sorted.ids(slot) != input.ids(i) ||
                              sorted.ghost(slot) != input.ghost(i);
                    if (slot < rows.extent(0))
                        errors += to_input(owned(slot)) != rows(slot);
                }, mismatches);
            if (mismatches)
                throw Failure{{StatusCode::invalid_state, "rebuilt WCSPH grid changed source or row order"}};
            committed_grid_ = std::make_unique<CellGrid>(std::move(rebuilt));
        }
        const auto& old = *committed_grid_;
        const auto& old_fields = old.fields();
        const auto& old_owned = old.owned_slots();
        const auto n = old_fields.ids.extent(0), nf = old_owned.extent(0);
        const auto ensure = [&](auto& buffer, std::size_t count, std::size_t stride,
                                const char* label, const auto& trim) {
            if (!buffer) {
                using V = std::remove_cvref_t<decltype(buffer->values)>;
                const auto bytes = require(detail::checked_array_bytes(count, stride));
                auto ticket = require(with_eviction([&] { return budget_.reserve(bytes); }, trim));
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
        {
            const auto& drift = work.stage->values;
            require(kick_drift_3d(exec_, old, v, a, dt, drift, half, parameters_.gravity));
            if (walls_)
                sample_walls(exec_, *walls_, old_fields, "sph.wall.drift_position",
                    [&](std::size_t s, const auto& sample) {
                        for (int k = 0; k < 3; ++k)
                            drift(s, k) = sample.position[k];
                    });
            const ParticleFields3D input{drift, old_fields.mass, old_fields.ids, old_fields.ghost};
            auto updated = old.update(exec_, input, max_cells_, admit_grid);
            if (updated.status().code() == StatusCode::invalid_argument)
                throw Failure{{StatusCode::numerical_failure, "invalid drifted WCSPH position"}};
            candidate_grid_ = std::make_unique<CellGrid>(require(std::move(updated)));
            const auto& next = *candidate_grid_;
            if (input_is_sorted(exec_, next)) {
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
        const auto& next = *candidate_grid_;
        const auto make_rows = [&] {
            return establish_owned_rows(exec_, geometry, next.owned_slots(), next.sorted_to_input(), budget_);
        };
        auto s = make_rows();
        if (s.code() == StatusCode::resource_exhausted && trim_drift()) {
            s = make_rows();
        }
        require(s);
        geometry.topology = current.geometry.topology;
        geometry.geometry = current.geometry.geometry;
        geometry.layout = current.geometry.layout;
        const auto compare = [&] {
            return update_geometry_versions(exec_, layout_of(old), layout_of(next), budget_,
                                            geometry.topology, geometry.geometry, geometry.layout);
        };
        s = compare();
        if (s.code() == StatusCode::resource_exhausted && trim_drift()) {
            s = compare();
        }
        require(s);
        stamp_geometry(candidate);
        ensure_stage();
        const auto no_trim = [] { return false; };
        ensure(work.old_row, n, sizeof(LocalIndex), "sph.continuity.old_row", no_trim);
        ensure(work.rate_n, nf, sizeof(double), "sph.continuity.accepted_rate", no_trim);
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
            const auto set_wall_velocity = [&](const CellGrid& grid) {
                if (walls_)
                    sample_walls(exec_, *walls_, grid.fields(), "sph.wall.stage_velocity",
                        [&](std::size_t slot, const auto& sample) {
                            for (int k = 0; k < 3; ++k)
                                stage(slot, k) = sample.velocity[k];
                        });
            };
            set_wall_velocity(old);
            const auto& q0 = old.neighbors();
            if (observe_)
                observe_(observer_context_, StepStage3D::start, q0);
            require(continuity_rate_3d(exec_, q0, stage, work.rate_n->values));

            const auto& to_new = next.input_to_sorted();
            std::size_t invalid{};
            Kokkos::parallel_reduce("sph.continuity.predict_velocity", rows,
                [&](std::size_t r, std::size_t& errors) {
                    const auto slot = to_new(old_owned(r));
                    for (int k = 0; k < 3; ++k) {
                        const double predicted = Kokkos::fma(dt, a(r, k) + parameters_.gravity[k], double(v(r, k)));
                        if (Kokkos::abs(predicted) <= std::numeric_limits<real>::max())
                            stage(slot, k) = static_cast<real>(predicted);
                        else
                            ++errors;
                    }
                }, invalid);
            if (invalid)
                throw Failure{{StatusCode::numerical_failure, "invalid predicted WCSPH velocity"}};
            set_wall_velocity(next);
            ensure(work.rate_star, nf, sizeof(double), "sph.continuity.predicted_rate", no_trim);
            const auto& q1 = next.neighbors();
            if (observe_)
                observe_(observer_context_, StepStage3D::end, q1);
            require(continuity_rate_3d(exec_, q1, stage, work.rate_star->values));
            ensure(work.corrected, nf, sizeof(real), "sph.continuity.density", no_trim);
            correct_density_3d(exec_, next.owned_slots(), next.sorted_to_input(), old_row,
                               current.density->view(), work.rate_n->values, work.rate_star->values,
                               dt, work.corrected->values);
            exec_.fence("sph.continuity.correction_ready");
        }
        bind_group(candidate, FieldGroup3D::density, bind, [&](const StepOutputs3D& out) -> Status {
            Kokkos::deep_copy(exec_, out.density, work.corrected->values);
            exec_.fence("sph.continuity.trial_density_ready");
            return {};
        }, [&] {
            const bool released = work.release_rates() || work.stage;
            work.stage.reset();
            return released;
        });
        evaluate_pressure(candidate, next, bind, &work);
        bind_group(candidate, FieldGroup3D::velocity, bind, [&](const StepOutputs3D& out) {
            return finish_kick_3d(exec_, next, half, candidate.acceleration->view(), dt,
                                  out.velocity, parameters_.gravity);
        }, [&] { return work.release_continuity(); });
    });
}

void WcsphStep3D::commit() noexcept {
    committed_grid_.swap(candidate_grid_);
    candidate_grid_.reset();
}
void WcsphStep3D::restore() noexcept {
    candidate_grid_.reset();
    workspace_.release_continuity();
    workspace_.half.reset();
}
void WcsphStep3D::release_workspace() noexcept {
    restore();
    committed_grid_.reset();
}
} // namespace nexora::sph
