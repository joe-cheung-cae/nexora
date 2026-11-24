#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <data/CsrAdjacency.hpp>
#include <data/FieldIssuer.hpp>
#include <memory>
#include <nexora/participant/ParticipantIF.hpp>
#include <optional>
#include <participants/ResourceBudget.hpp>
#include <span>
#include <vector>

namespace nexora::sph_fixture {

struct Config {
    std::vector<GlobalEntityId> row_ids, column_ids;
    std::vector<real> positions, masses;
    std::vector<std::uint8_t> ghost, boundary;
    std::vector<Offset> offsets;
    std::vector<LocalIndex> indices;
};

template <class T> using ConstView = Kokkos::View<const T*, Kokkos::HostSpace>;

// A snapshot owns its readable storage. Indexed reads require an in-range
// index, and density reads require a snapshot with a computed density.
struct Snapshot {
    Snapshot() = default;
    Snapshot(const Snapshot&) = default;
    Snapshot(Snapshot&&) noexcept = default;
    Snapshot& operator=(Snapshot other) noexcept;

    std::size_t row_count() const;
    std::size_t column_count() const;
    std::size_t edge_count() const;
    std::uint64_t row_id(std::size_t i) const {
        return geometry_->adjacency.rows().id_at(i).value();
    }
    std::uint64_t column_id(std::size_t i) const {
        return geometry_->adjacency.columns().id_at(i).value();
    }
    Offset offset(std::size_t i) const;
    LocalIndex index(std::size_t i) const;
    std::uint8_t ghost(std::size_t i) const;
    std::uint8_t boundary(std::size_t i) const;
    real density(std::size_t i) const;

    real owned_density_sum{};
    FieldStamp stamp;
    bool density_valid{};

  private:
    friend class Participant;
    struct Geometry {
        Geometry(detail::ResourceBudget::Ticket reservation, detail::CsrAdjacency graph,
                 ConstView<LocalIndex> self_columns, ConstView<real> positions, ConstView<real> masses,
                 ConstView<std::uint8_t> is_ghost, ConstView<std::uint8_t> is_boundary,
                 const FieldStamp& stamp);

        // Declared first so every layout View is released before the reservation.
        detail::ResourceBudget::Ticket budget;
        detail::CsrAdjacency adjacency;
        ConstView<LocalIndex> self;
        ReadField<ConstView<real>> x, mass;
        ConstView<std::uint8_t> ghost, boundary;
    };
    struct Density;
    detail::ResourceBudget::Ticket snapshot_record_;
    std::shared_ptr<const Geometry> geometry_;
    std::shared_ptr<const Density> density_;
    std::shared_ptr<const ReadField<ConstView<real>>> read_grant_;
};

// Destroy the participant and all retained snapshots before Kokkos finalizes.
class Participant final : public ParticipantIF {
  public:
    using ResourceLimits = detail::ResourceBudget::Limits;
    explicit Participant(Config config, ParticipantId id = ParticipantId{1},
                         ResourceLimits limits = {});
    Participant(const Participant&) = delete;
    Participant& operator=(const Participant&) = delete;

    ParticipantId id() const noexcept override { return id_; }
    ParticipantCapabilities capabilities() const override { return {true}; }
    std::vector<PortDescriptor> ports() const override { return {}; }
    TimeConstraint time_constraint(double) const override { return {}; }
    Status initialize(double initial_time) override;
    Status prepare() override;
    Status begin_window(const TimeInterval& interval) override;
    Status advance_trial(const TimeInterval& interval) override;
    Status prepare_commit(const TimeInterval& interval) override;
    Status commit_window(const TimeInterval& interval) override;
    Status restore_window() override;
    Status finalize() override;

    Result<Snapshot> snapshot() const;
    // Only valid at a prepared safe point. Installation of the new layout is atomic.
    Status reorder(std::span<const GlobalEntityId> row_ids,
                   std::span<const GlobalEntityId> column_ids, const FieldStamp& expected);
    Result<FieldStamp> trial_stamp() const;
    // A retained trial snapshot stays readable after restore or publication.
    Result<Snapshot> trial_snapshot(const FieldStamp& expected) const;
    detail::ResourceBudget::Usage resource_usage() const { return budget_.usage(); }

  private:
    using Geometry = Snapshot::Geometry;
    enum class Stage { created, initialized, ready, open, advanced, precommitted, failed, finalized };
    static Snapshot layout_snapshot(const std::shared_ptr<const Geometry>& geometry);
    bool matches(const TimeInterval& interval) const noexcept;
    void close_window() noexcept;

    Config config_;
    ParticipantId id_;
    detail::ResourceBudget budget_;
    detail::ResourceBudget::Ticket trial_budget_;
    std::optional<Kokkos::Serial> exec_;
    std::optional<Snapshot> accepted_, trial_, pending_;
    std::optional<detail::FieldIssuer> trial_issuer_;
    mutable std::weak_ptr<const ReadField<ConstView<real>>> trial_read_;
    Stage stage_ = Stage::created;
    TimeInterval interval_;
    std::uint64_t attempt_{};
};

} // namespace nexora::sph_fixture
