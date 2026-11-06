#pragma once
#include <nexora/core/Status.hpp>
#include <optional>
#include <stdexcept>
#include <utility>
namespace nexora {
template <class T> class Result {
  public:
    Result(T value) : value_(std::move(value)) {}
    Result(Status status) : status_(std::move(status)) {
        if (status_.ok())
            throw std::invalid_argument("a failed Result requires an error status");
    }
    bool ok() const noexcept {
        return value_.has_value();
    }
    const Status& status() const noexcept {
        return status_;
    }
    T& value() & {
        return value_.value();
    }
    const T& value() const& {
        return value_.value();
    }
    T&& value() && {
        return std::move(value_).value();
    }

  private:
    std::optional<T> value_;
    Status status_;
};
} // namespace nexora
