#include "ReferenceCase.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <nexora/participant/TimeConstraint.hpp>
#include <participants/reference/Participant.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace nexora::cli {
namespace {
std::string_view trim(std::string_view text) {
    constexpr std::string_view whitespace = " \t\r\n\f\v";
    const auto begin = text.find_first_not_of(whitespace);
    if (begin == std::string_view::npos)
        return {};
    return text.substr(begin, text.find_last_not_of(whitespace) - begin + 1);
}

struct Entry {
    std::string value;
    std::size_t line;
};
using Entries = std::map<std::string, Entry, std::less<>>;

Status error(const std::filesystem::path& path, std::size_t line, std::string message,
             StatusCode code = StatusCode::invalid_argument) {
    return {code, path.string() + ":" + std::to_string(line) + ": " + std::move(message)};
}

template <class T>
bool parse_number(std::string_view text, T& value) {
    if (!text.empty() && text.front() == '+') {
        text.remove_prefix(1);
        if (!text.empty() && (text.front() == '-' || text.front() == '+'))
            return false;
    }
    if (text.empty())
        return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value,
                                        std::chars_format::general);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
           std::isfinite(value);
}

bool parse_id(std::string_view text, std::uint64_t& value) {
    if (text.empty())
        return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

template <class T, class Parse>
bool parse_list(std::string_view text, std::vector<T>& values, Parse parse) {
    values.clear();
    while (true) {
        const auto comma = text.find(',');
        const auto token = trim(text.substr(0, comma));
        T value{};
        if (!parse(token, value))
            return false;
        values.push_back(value);
        if (comma == std::string_view::npos)
            return true;
        text.remove_prefix(comma + 1);
    }
}

Result<ReferenceCase> load(const std::filesystem::path& path, std::size_t& current_line) {
    std::ifstream input(path);
    if (!input)
        return error(path, 1, "cannot open case file", StatusCode::io_error);
    Entries entries;
    constexpr std::array known{"version", "fixture", "start", "end", "step", "execution",
                                "precision", "lambda", "ids", "x", "q"};
    std::string line;
    while (std::getline(input, line)) {
        ++current_line;
        const auto text = trim(line);
        if (text.empty() || text.front() == '#')
            continue;
        const auto equal = text.find('=');
        if (equal == std::string_view::npos)
            return error(path, current_line, "expected key=value");
        const auto key = trim(text.substr(0, equal));
        const auto value = trim(text.substr(equal + 1));
        if (key == "quantum")
            return error(path, current_line, "quantum is unsupported for the reference fixture",
                         StatusCode::unsupported);
        if (std::find(known.begin(), known.end(), key) == known.end())
            return error(path, current_line, "unknown key '" + std::string(key) + "'");
        if (entries.contains(key))
            return error(path, current_line, "duplicate key '" + std::string(key) + "'");
        if (value.empty())
            return error(path, current_line, "empty value for '" + std::string(key) + "'");
        entries.emplace(std::string(key), Entry{std::string(value), current_line});
    }
    if (input.bad())
        return error(path, std::max(current_line, std::size_t{1}), "cannot read case file",
                     StatusCode::io_error);
    const auto line_of = [&](std::string_view key) {
        const auto it = entries.find(key);
        return it == entries.end() ? std::size_t{1} : it->second.line;
    };
    const auto fail = [&](std::string_view key, std::string message,
                          StatusCode code = StatusCode::invalid_argument) {
        return error(path, line_of(key), std::move(message), code);
    };
    for (const auto* required : {"version", "fixture", "end", "step"}) {
        if (!entries.contains(required))
            return fail(required, "missing required key '" + std::string(required) + "'");
    }
    if (entries.at("version").value != "1")
        return fail("version", "unsupported case version (expected 1)", StatusCode::unsupported);
    if (entries.at("fixture").value != "reference")
        return fail("fixture", "unsupported fixture (expected reference)", StatusCode::unsupported);
    if (entries.contains("execution") && entries.at("execution").value != "serial")
        return fail("execution", "unsupported execution profile (expected serial)",
                    StatusCode::unsupported);
    constexpr auto precision = sizeof(real) == sizeof(double) ? "double" : "float";
    if (entries.contains("precision") && entries.at("precision").value != precision)
        return fail("precision", std::string("precision must match this build: ") + precision,
                    StatusCode::unsupported);

    ReferenceCase result;
    double step = 0;
    for (auto [key, destination] : {std::pair{"start", &result.start},
                                   std::pair{"end", &result.end}, std::pair{"step", &step}}) {
        const auto it = entries.find(key);
        if (it != entries.end() && !parse_number(it->second.value, *destination))
            return fail(key, std::string(key) + " must be a finite decimal number in double range");
    }
    if (result.end <= result.start)
        return fail("end", "end must be greater than start");
    if (entries.contains("lambda") &&
        !parse_number(entries.at("lambda").value, result.config.lambda))
        return fail("lambda", "lambda must be finite and representable in build real precision");
    if (entries.contains("ids") &&
        !parse_list(entries.at("ids").value, result.config.ids, parse_id))
        return fail("ids", "ids must be a nonempty comma-separated list of unsigned 64-bit integers");
    for (auto [key, destination] : {std::pair{"x", &result.config.x},
                                   std::pair{"q", &result.config.q}}) {
        if (entries.contains(key) && !parse_list(entries.at(key).value, *destination, parse_number<real>))
            return fail(key, std::string(key) +
                             " must contain finite comma-separated values representable in build real precision");
    }
    const auto checked_config = reference::Participant::validate_case_config(result.config, result.start);
    if (!checked_config.ok()) {
        const auto key = result.config.lambda <= 0 ? "lambda" :
                         result.config.ids.size() != result.config.x.size() ? "x" :
                         result.config.ids.size() != result.config.q.size() ? "q" : "ids";
        return fail(key, checked_config.message(), checked_config.code());
    }
    const auto constraint = reference::Participant::case_time_constraint(result.config.lambda);
    if (auto status = validate_time_step(constraint, step); !status.ok())
        return fail("step", status.message(), status.code());
    if (auto status = reference::Participant::validate_case_interval({0.0, step, step}, result.config.lambda);
        !status.ok())
        return fail("step", status.message(), status.code());
    // Monotonic rounded addition cannot cross an endpoint predecessor at which
    // the nominal duration already stalls; all tail requests are no larger.
    const double predecessor = std::nextafter(result.end, result.start);
    if (predecessor + step <= predecessor)
        return fail("step", "step does not produce a finite advancing endpoint near end");
    // A positive rounded advancement is at most twice the requested duration.
    // Use this conservative lower bound only to reject impossible capacities;
    // enumeration below remains authoritative for every rounded endpoint.
    constexpr std::size_t maximum_steps = 1000000;
    const auto schedule_limit = [&] {
        return fail("step", "case step count exceeds version 1 limit of 1000000 steps",
                    StatusCode::resource_exhausted);
    };
    // Halve each endpoint before subtracting: long double may have only the
    // range of double, while two finite opposite-sign endpoints can span more.
    const long double minimum_count =
        (static_cast<long double>(result.end) * 0.5L -
         static_cast<long double>(result.start) * 0.5L) / step;
    if (minimum_count > static_cast<long double>(maximum_steps))
        return schedule_limit();
    if (minimum_count > static_cast<long double>(result.steps.max_size()) ||
        minimum_count > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        return fail("step", "case step count exceeds schedule or counter capacity",
                    StatusCode::resource_exhausted);
    double time = result.start;
    while (time < result.end) {
        if (result.steps.size() == maximum_steps)
            return schedule_limit();
        double requested = step;
        double endpoint = time + requested;
        if (endpoint > result.end) {
            requested = result.end - time;
            endpoint = time + requested;
        }
        if (!std::isfinite(endpoint) || endpoint <= time || endpoint > result.end)
            return fail("step", "step does not produce a finite advancing endpoint within end");
        if (auto status = validate_time_step(constraint, requested); !status.ok())
            return fail("step", status.message(), status.code());
        if (auto status = reference::Participant::validate_case_interval({time, endpoint, requested},
                                                        result.config.lambda); !status.ok())
            return fail("step", status.message(), status.code());
        if (result.steps.size() == result.steps.max_size() ||
            result.steps.size() == std::numeric_limits<std::uint64_t>::max())
            return fail("step", "case step count exceeds schedule or counter capacity",
                        StatusCode::resource_exhausted);
        result.steps.push_back(requested);
        time = endpoint;
    }
    return result;
}
} // namespace

Result<ReferenceCase> load_reference_case(const std::filesystem::path& path) {
    std::size_t current_line = 0;
    try {
        return load(path, current_line);
    } catch (const std::bad_alloc&) {
        return error(path, std::max(current_line, std::size_t{1}),
                     "case preflight allocation failed", StatusCode::resource_exhausted);
    } catch (const std::length_error&) {
        return error(path, std::max(current_line, std::size_t{1}),
                     "case preflight storage capacity exceeded", StatusCode::resource_exhausted);
    } catch (const std::filesystem::filesystem_error& e) {
        return error(path, std::max(current_line, std::size_t{1}), e.what(), StatusCode::io_error);
    }
}
} // namespace nexora::cli
