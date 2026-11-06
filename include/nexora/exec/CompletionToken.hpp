#pragma once
#include <nexora/core/Status.hpp>
#include <utility>
namespace nexora {
class CompletionToken {
  public:
    CompletionToken() = default;
    explicit CompletionToken(Status status) : status_(std::move(status)) {}
    Status wait() const {
        return status_;
    }

  private:
    Status status_;
};
using Event = CompletionToken;
} // namespace nexora
