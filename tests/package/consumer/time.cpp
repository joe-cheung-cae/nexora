#include <nexora/core/Time.hpp>

bool check_installed_time_interval() {
    const nexora::TimeInterval interval{0x1p52, 0x1p52 + 1.0, 0.75};
    return interval.duration() == 0.75 && interval.end - interval.begin == 1.0;
}
