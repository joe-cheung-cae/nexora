#pragma once

#include "Geometry3D.hpp"

#include <methods/sph/Continuity3D.hpp>
#include <methods/sph/SphBoundary.hpp>
#include <methods/sph/Verlet3D.hpp>
#include <nexora/core/Time.hpp>
#include <nexora/data/FieldBinding.hpp>

#include <functional>
#include <optional>

namespace nexora::sph {

struct WcsphState3D {
    Geometry3D geometry;
    FieldStamp stamp;
    double maximum_step{};
    std::array<detail::ResourceBudget::Ticket, 3> field_storage;
    std::optional<ReadField<Values::const_type>> density, pressure;
    std::optional<ReadField<Acceleration3D::const_type>> acceleration, velocity;
};

enum class FieldGroup3D { density, pressure, velocity };
struct StepOutputs3D {
    Values density, pressure;
    Acceleration3D acceleration;
    Velocity3D velocity;
};
using FieldWrite3D = std::function<Status(const StepOutputs3D&)>;
using BindFields3D = std::function<Status(WcsphState3D&, const FieldStamp&,
                                          const Geometry3D::Indices&, FieldGroup3D,
                                          const FieldWrite3D&)>;
enum class StepStage3D { initial, start, end };

class WcsphStep3D {
  public:
    WcsphStep3D(const Kokkos::Serial&, const detail::ResourceBudget&, std::size_t max_cells,
                WcsphVerletParameters);
    Status initialize_state(double time, const ParticleFields3D&, const Velocity3D::const_type&,
                            const Values::const_type&, const std::optional<FixedWallParticles3D>&,
                            WcsphState3D&, const BindFields3D&);
    Status try_step(const TimeInterval&, const WcsphState3D& current, WcsphState3D& candidate,
                    const BindFields3D&);
    void commit() noexcept;
    void restore() noexcept;
    void release_workspace() noexcept;
    const WcsphVerletParameters& parameters() const { return parameters_; }

  private:
    friend struct WcsphTestAccess;
    friend class WcsphParticipant3D;
    void* observer_context_{};
    void (*observe_)(void*, StepStage3D, const NeighborView3D&){};
    using Ticket = detail::ResourceBudget::Ticket;
    template <class V> struct Buffer {
        Ticket ticket;
        V values;
    };
    struct Workspace {
        std::optional<Buffer<Velocity3D>> half, stage;
        std::optional<Buffer<Geometry3D::Indices::non_const_type>> old_row;
        std::optional<Buffer<DensityRate3D>> rate_n, rate_star;
        std::optional<Buffer<Values>> corrected;

        bool release_rates() noexcept;
        bool release_continuity() noexcept;
    };
    const Kokkos::Serial& exec_;
    const detail::ResourceBudget& budget_;
    std::size_t max_cells_;
    WcsphVerletParameters parameters_;
    Ticket wall_storage_;
    std::optional<SphBoundary::FixedWall3D> walls_;
    std::unique_ptr<CellGrid> committed_grid_, candidate_grid_;
    Workspace workspace_;

    void evaluate_pressure(WcsphState3D&, const CellGrid&, const BindFields3D&, Workspace*);
};
} // namespace nexora::sph
