#pragma once

#include <methods/sph/Verlet3D.hpp>
#include <nexora/data/FieldStamp.hpp>
#include <nexora/participant/ParticipantIF.hpp>
#include <participants/sph/Geometry3D.hpp>

#include <array>
#include <memory>
#include <optional>
#include <utility>

namespace nexora::sph {

// An issued snapshot pins one immutable geometry and field bank across restore or commit.
// The unchecked indexed operations require an in-range owned row.
class Snapshot3D {
  public:
    FieldStamp stamp;
    std::size_t row_count() const {
        return geometry_ ? geometry_->grid.owned_slots().extent(0) : 0;
    }
    std::uint64_t row_id(std::size_t row) const {
        const auto& grid = geometry_->grid;
        return grid.fields().ids(grid.owned_slots()(row));
    }
    real mass(std::size_t row) const {
        const auto& grid = geometry_->grid;
        return grid.fields().mass(grid.owned_slots()(row));
    }
    real density(std::size_t row) const;
    Result<real> pressure(std::size_t row) const;
    Result<std::array<real, 3>> pressure_acceleration(std::size_t row) const;
    Result<std::array<real, 3>> position(std::size_t row) const;
    Result<std::array<real, 3>> velocity(std::size_t row) const;
    template <class Callback> void for_each_neighbor(std::size_t row, Callback&& visit) const {
        const auto& grid = geometry_->grid;
        grid.for_each_neighbor(row, [&](LocalIndex slot, double squared_distance) {
            visit(grid.fields().ids(slot), squared_distance);
        });
    }

  private:
    friend class Participant3D;
    struct Fields;
    detail::ResourceBudget::Ticket record_;
    std::shared_ptr<const Geometry3D> geometry_;
    std::shared_ptr<const Fields> fields_;
};

struct FixedWallParticles3D {
    decltype(ParticleFields3D::position) position;
    decltype(ParticleFields3D::mass) mass;
    decltype(ParticleFields3D::ids) ids;
};

class Participant3D final : public ParticipantIF {
  public:
    using ResourceLimits = detail::ResourceBudget::Limits;
    // Initial handles are retained until initialize() captures the independent accepted state.
    explicit Participant3D(ParticleFields3D initial, std::size_t max_cells,
                           ParticipantId id = ParticipantId{1}, ResourceLimits limits = {});
    // WCSPH reconstructs participating ghost densities from the supplied sources.
    // Each stage must include the full support of those ghosts.
    Participant3D(ParticleFields3D initial, std::size_t max_cells, WcsphParameters parameters,
                  ParticipantId id = ParticipantId{1}, ResourceLimits limits = {});
    // Initial-only kernel-summation diagnostics. Advancing requires an explicit density.
    Participant3D(ParticleFields3D initial, std::size_t max_cells, WcsphVerletParameters parameters,
                  Velocity3D::const_type initial_velocity, ParticipantId id = ParticipantId{1},
                  ResourceLimits limits = {});
    // Initial-only fixed-wall diagnostic. The fluid input must contain owned particles only.
    Participant3D(ParticleFields3D fluid, std::size_t max_cells, WcsphVerletParameters parameters,
                  Velocity3D::const_type initial_velocity, FixedWallParticles3D walls,
                  ParticipantId id = ParticipantId{1}, ResourceLimits limits = {});
    // Explicit initial densities follow the fluid input order and evolve with continuity.
    Participant3D(ParticleFields3D fluid, std::size_t max_cells, WcsphVerletParameters parameters,
                  Velocity3D::const_type initial_velocity, Values::const_type initial_density,
                  ParticipantId id = ParticipantId{1}, ResourceLimits limits = {});
    Participant3D(ParticleFields3D fluid, std::size_t max_cells, WcsphVerletParameters parameters,
                  Velocity3D::const_type initial_velocity, Values::const_type initial_density,
                  FixedWallParticles3D walls, ParticipantId id = ParticipantId{1},
                  ResourceLimits limits = {});
    Participant3D(const Participant3D&) = delete;
    Participant3D& operator=(const Participant3D&) = delete;

    ParticipantId id() const noexcept override {
        return id_;
    }
    ParticipantCapabilities capabilities() const override {
        return {true};
    }
    std::vector<PortDescriptor> ports() const override {
        return {};
    }
    TimeConstraint time_constraint(double) const override;
    Status initialize(double initial_time) override;
    Status prepare() override;
    Status begin_window(const TimeInterval& interval) override;
    // Externally staged modes copy the end-of-window input for this attempt.
    // Omitting it in those modes keeps geometry stationary.
    Status set_trial_fields(const ParticleFields3D& fields);
    Result<TrialReport> advance_trial(const TimeInterval& interval) override;
    Status prepare_commit(const TimeInterval& interval) override;
    Status commit_window(const TimeInterval& interval) override;
    Status restore_window() override;
    Status finalize() override;

    Result<Snapshot3D> snapshot() const;
    Result<FieldStamp> trial_stamp() const;
    Result<Snapshot3D> trial_snapshot(const FieldStamp& expected) const;
    detail::ResourceBudget::Usage resource_usage() const {
        return budget_.usage();
    }

  private:
    enum class Stage {
        created,
        initialized,
        ready,
        open,
        advanced,
        precommitted,
        failed,
        finalized
    };
    struct IntegrationWorkspace {
        template <class View> struct Buffer {
            detail::ResourceBudget::Ticket reservation;
            View values;
        };
        struct Continuity {
            detail::ResourceBudget::Ticket source_rows, rates;
            CellGrid::Indices::non_const_type old_row;
            DensityRate3D rate_n, rate_star;
        };

        std::optional<Buffer<Velocity3D>> half_velocity, stage;
        std::optional<Buffer<Values>> corrected;
        std::optional<Continuity> continuity;
    };
    Snapshot3D compute(std::shared_ptr<const Geometry3D> geometry, FieldStamp stamp,
                       const Velocity3D::const_type& input_velocity = {}, double dt = 0,
                       const Values::const_type* owned_density = nullptr,
                       IntegrationWorkspace* finished_continuity = nullptr) const;
    Snapshot3D integrate(FieldStamp stamp, double dt);
    Result<Snapshot3D> issue(const Snapshot3D&) const;
    bool matches(const TimeInterval& interval) const;
    void close_window() noexcept;

    ParticleFields3D initial_;
    std::size_t max_cells_;
    ParticipantId id_;
    detail::ResourceBudget budget_;
    std::optional<WcsphParameters> pressure_parameters_;
    std::optional<double> verlet_cfl_;
    double smoothing_length_{1};
    std::array<double, 3> gravity_{};
    std::optional<FixedWallParticles3D> fixed_walls_;
    Velocity3D::const_type initial_velocity_;
    std::optional<Values::const_type> initial_density_;
    std::optional<Kokkos::Serial> exec_;
    IntegrationWorkspace integration_workspace_;
    std::optional<StagedFields3D> staged_;
    std::optional<Snapshot3D> accepted_, trial_, pending_;
    Stage stage_ = Stage::created;
    TimeInterval interval_;
    std::uint64_t attempt_{};
};

} // namespace nexora::sph
