#include <nexora/core/CheckedCast.hpp>
#include <nexora/core/Types.hpp>
#include <nexora/core/Version.hpp>
#include <nexora/participant/TimeConstraint.hpp>
#include <nexora/simulation/Simulation.hpp>

#include <array>
#include <limits>
#include <type_traits>

static_assert(std::is_same_v<nexora::real,
    std::conditional_t<NEXORA_TEST_EXPECT_DOUBLE_PRECISION, double, float>>);

bool check_installed_time_interval();

namespace {

class Counter final : public nexora::ParticipantIF {
  public:
    nexora::ParticipantId id() const noexcept override { return nexora::ParticipantId{1}; }
    std::vector<nexora::PortDescriptor> ports() const override { return {}; }
    nexora::TimeConstraint time_constraint() const override { return {0.5, 0.25, 0.25}; }
    nexora::Status initialize(double time) override {
        total = time;
        return {};
    }
    nexora::Status finalize() override { return {}; }
    double total{};

  private:
    nexora::Status do_step(const nexora::TimeInterval& interval) override {
        total += interval.duration();
        return {};
    }
};

bool check_simulation() {
    Counter participant;
    nexora::Simulation invalid_clock(participant, std::numeric_limits<double>::infinity());
    if (invalid_clock.initialize().code() != nexora::StatusCode::invalid_argument ||
        participant.total != 0.0)
        return false;
    nexora::Simulation simulation(participant, 2.0);
    if (!simulation.initialize().ok())
        return false;
    const auto status = simulation.run(std::array{0.5, 0.25});
    const bool published = simulation.time().accepted_time == 2.75 &&
                           simulation.time().accepted_steps == 2 && participant.total == 2.75;
    return simulation.finalize().ok() && status.ok() && published;
}

} // namespace

int main() {
    const nexora::TimeInterval interval{0.0, 0.5, 0.5};
    const nexora::TimeConstraint constraint{1.0, 0.25, 0.25};
    const auto step_status = nexora::validate_time_step(constraint, interval.duration());
    const auto rejected_step = nexora::validate_time_step(constraint, 0.3);
    auto unavailable = constraint;
    unavailable.available = false;
    const auto unavailable_step = nexora::validate_time_step(unavailable, 0.5);
    const auto index = nexora::checked_integer_cast<nexora::LocalIndex>(42);
    const bool valid = nexora::version[0] != '\0' && step_status.ok() &&
                       rejected_step.code() == nexora::StatusCode::invalid_argument &&
                       unavailable_step.code() == nexora::StatusCode::invalid_state &&
                       index.ok() && index.value() == 42 && interval.duration() == 0.5 &&
                       check_installed_time_interval() && check_simulation();
    return valid ? 0 : 1;
}
