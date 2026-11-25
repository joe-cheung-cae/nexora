#pragma once

#include "WcsphStep3D.hpp"

#include <array>
#include <functional>
#include <memory>
#include <nexora/participant/ParticipantIF.hpp>
#include <optional>

namespace nexora::sph {

struct CommittedRow3D {
    std::uint64_t id;
    real mass;
    std::array<real, 3> position;
    real density;
    real pressure;
    std::array<real, 3> pressure_acceleration;
    std::array<real, 3> velocity;

    friend bool operator==(const CommittedRow3D&, const CommittedRow3D&) = default;
};

struct WallValue3D {
    std::uint64_t id;
    real mass;
    std::array<real, 3> position;
    FieldStamp stamp;
    double phase;

    friend bool operator==(const WallValue3D&, const WallValue3D&) = default;
};

class WcsphParticipant3D final : public ParticipantIF {
  public:
    using ResourceLimits = detail::ResourceBudget::Limits;

    // Initial velocity and density follow the all-owned fluid input order.
    WcsphParticipant3D(ParticleFields3D fluid, std::size_t max_cells,
                       WcsphVerletParameters parameters, Velocity3D::const_type initial_velocity,
                       Values::const_type initial_density, ParticipantId id = ParticipantId{1},
                       ResourceLimits limits = {});
    WcsphParticipant3D(ParticleFields3D fluid, std::size_t max_cells,
                       WcsphVerletParameters parameters, Velocity3D::const_type initial_velocity,
                       Values::const_type initial_density, FixedWallParticles3D walls,
                       ParticipantId id = ParticipantId{1}, ResourceLimits limits = {});
    WcsphParticipant3D(ParticleFields3D fluid, std::size_t max_cells,
                       WcsphVerletParameters parameters, Velocity3D::const_type initial_velocity,
                       Values::const_type initial_density, std::vector<SphBoundary::WallGroupInput3D> walls,
                       ParticipantId id = ParticipantId{1}, ResourceLimits limits = {});
    WcsphParticipant3D(const WcsphParticipant3D&) = delete;
    WcsphParticipant3D& operator=(const WcsphParticipant3D&) = delete;

    ParticipantId id() const noexcept override { return id_; }
    std::vector<PortDescriptor> ports() const override { return {}; }

    TimeConstraint time_constraint() const override;
    Status initialize(double initial_time) override;
    Status finalize() override;

    Status visit_committed_rows(const std::function<void(CommittedRow3D)>& visit) const;
    Result<WallValue3D> read_committed_wall(std::uint64_t id) const;
    Status release_workspace();
    detail::ResourceBudget::Usage resource_usage() const { return budget_.usage(); }

  private:
    friend struct WcsphTestAccess;

    Status initialize_at_finite_time(double time) override;
    Status do_step(const TimeInterval& interval) override;
    BindFields3D bind_fields(WcsphState3D& candidate) const;
    bool idle_with_state() const;

    ParticleFields3D initial_;
    std::size_t max_cells_;
    WcsphVerletParameters parameters_;
    Velocity3D::const_type initial_velocity_;
    Values::const_type initial_density_;
    std::vector<SphBoundary::WallGroupInput3D> initial_walls_;
    ParticipantId id_;
    detail::ResourceBudget budget_;
    std::optional<Kokkos::Serial> exec_;
    std::optional<WcsphStep3D> step_;
    std::unique_ptr<WcsphState3D> accepted_;
    bool finalized_{};
    mutable bool active_{};
    void* observer_context_{};
    void (*observe_)(void*, StepStage3D, const NeighborView3D&){};
    bool observe_kinematics_{};
};

} // namespace nexora::sph
