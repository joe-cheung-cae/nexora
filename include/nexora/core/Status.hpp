#pragma once
#include <string>
#include <utility>
namespace nexora {
enum class StatusCode {
    success,
    invalid_argument,
    invalid_state,
    unsupported,
    numerical_failure,
    resource_exhausted,
    communication_failure,
    io_error,
    execution_failure,
    not_found,
    already_exists
};
class Status {
  public:
    Status() = default;
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}
    bool ok() const noexcept {
        return code_ == StatusCode::success;
    }
    StatusCode code() const noexcept {
        return code_;
    }
    const std::string& message() const noexcept {
        return message_;
    }

  private:
    StatusCode code_ = StatusCode::success;
    std::string message_;
};
} // namespace nexora
