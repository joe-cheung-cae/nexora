#pragma once

#include <nexora/core/Status.hpp>
#include <nexora/core/Time.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace nexora {

// This first synchronous path accepts a single legal requested step, without
// subcycling. Floating-point interval endpoints are rounded by normal addition;
// the quantum constraint applies to the requested step, not endpoint subtraction.
struct TimeConstraint {
    double maximum_step = 1.0;
    double minimum_step = 0.0;
    std::optional<double> fixed_quantum;
};

inline Status validate_time_constraint(const TimeConstraint& constraint) {
    if (!std::isfinite(constraint.maximum_step) || constraint.maximum_step <= 0.0 ||
        !std::isfinite(constraint.minimum_step) || constraint.minimum_step < 0.0 ||
        constraint.minimum_step > constraint.maximum_step) {
        return {StatusCode::invalid_argument, "invalid participant step bounds"};
    }
    if (constraint.fixed_quantum &&
        (!std::isfinite(*constraint.fixed_quantum) || *constraint.fixed_quantum <= 0.0)) {
        return {StatusCode::invalid_argument, "fixed time quantum must be finite and positive"};
    }
    return {};
}

inline Status validate_time_step(const TimeConstraint& constraint, double step) {
    const auto status = validate_time_constraint(constraint);
    if (!status.ok()) {
        return status;
    }
    // Allow representational roundoff at bounds, without using a unit-sized
    // absolute tolerance that would overwhelm very small physical time steps.
    const double lower_tolerance = 8.0 * std::numeric_limits<double>::epsilon() *
                                   std::max(std::abs(step), constraint.minimum_step);
    const double upper_tolerance = 8.0 * std::numeric_limits<double>::epsilon() *
                                   std::max(std::abs(step), constraint.maximum_step);
    if (!std::isfinite(step) || step <= 0.0 ||
        (step < constraint.minimum_step && constraint.minimum_step - step > lower_tolerance) ||
        (step > constraint.maximum_step && step - constraint.maximum_step > upper_tolerance)) {
        return {StatusCode::invalid_argument, "requested step violates participant bounds"};
    }
    if (constraint.fixed_quantum) {
        const double count = step / *constraint.fixed_quantum;
        const double nearest = std::round(count);
        // Keep a strict fractional-quantum ceiling even at large counts, where
        // an uncapped relative tolerance could accept a half-quantum step.
        const double tolerance = std::min(0.125, 8.0 * std::numeric_limits<double>::epsilon() *
                                                     std::max(1.0, std::abs(count)));
        if (!std::isfinite(count) || nearest < 1.0 || count > 0x1p53 ||
            std::abs(count - nearest) > tolerance) {
            return {StatusCode::invalid_argument, "requested step is not a fixed-quantum multiple"};
        }
    }
    return {};
}

} // namespace nexora
