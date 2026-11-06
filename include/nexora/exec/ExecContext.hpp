#pragma once
#include <initializer_list>
#include <nexora/exec/CompletionToken.hpp>
#include <utility>
namespace nexora {
// CPU operations finish before submit returns. A failed dependency suppresses the task.
class ExecContext {
  public:
    template <class F>
    CompletionToken submit(F&& task,
                           std::initializer_list<CompletionToken> dependencies = {}) const {
        auto s = check_dependencies(dependencies);
        if (!s.ok())
            return CompletionToken(std::move(s));
        return CompletionToken(std::forward<F>(task)());
    }

  private:
    static Status check_dependencies(std::initializer_list<CompletionToken> dependencies);
};
using CpuContext = ExecContext;
} // namespace nexora
