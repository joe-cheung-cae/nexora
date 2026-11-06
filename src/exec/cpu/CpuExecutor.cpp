#include <nexora/exec/ExecContext.hpp>

namespace nexora {
Status ExecContext::check_dependencies(std::initializer_list<CompletionToken> dependencies) {
    for (const auto& dependency : dependencies) {
        auto status = dependency.wait();
        if (!status.ok())
            return status;
    }
    return {};
}
} // namespace nexora
