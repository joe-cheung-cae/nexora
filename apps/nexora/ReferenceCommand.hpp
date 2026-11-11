#pragma once

namespace nexora::cli {

enum class ReferenceCommand { validate, plan, run, run_fixture };

int reference_command(ReferenceCommand command, const char* path);

} // namespace nexora::cli
