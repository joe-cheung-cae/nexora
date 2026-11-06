#pragma once

namespace nexora {

struct TimeInterval {
    double begin = 0.0;
    double end = 0.0;
    // Authoritative integration duration. Do not recover it by subtracting the
    // rounded absolute endpoints, which can magnify error at large times.
    double requested_step = 0.0;

    double duration() const noexcept {
        return requested_step;
    }
};

} // namespace nexora
