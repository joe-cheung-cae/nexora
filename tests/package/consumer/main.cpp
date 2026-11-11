#include <nexora/core/CheckedCast.hpp>
#include <nexora/core/Types.hpp>
#include <nexora/core/Version.hpp>
#include <nexora/participant/TimeConstraint.hpp>
#include <nexora/simulation/Simulation.hpp>

#include <type_traits>

static_assert(std::is_same_v<nexora::real,
    std::conditional_t<NEXORA_TEST_EXPECT_DOUBLE_PRECISION, double, float>>);

bool check_installed_time_interval();

int main() {
    const nexora::TimeInterval interval{0.0, 0.5, 0.5};
    const nexora::TimeConstraint constraint{1.0, 0.25, 0.25};
    const auto step_status = nexora::validate_time_step(constraint, interval.duration());
    const auto rejected_step = nexora::validate_time_step(constraint, 0.3);
    const auto index = nexora::checked_integer_cast<nexora::LocalIndex>(42);
    const bool valid = nexora::version[0] != '\0' && step_status.ok() &&
                       rejected_step.code() == nexora::StatusCode::invalid_argument &&
                       index.ok() && index.value() == 42 && interval.duration() == 0.5 &&
                       check_installed_time_interval();
    return valid ? 0 : 1;
}
