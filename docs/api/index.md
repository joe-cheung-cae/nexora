@mainpage Nexora C++ API

## Use the installed package

Use the exported `nexora::simulation` target to drive a
[Simulation](@ref nexora::Simulation) and read its
[accepted time and step count](@ref nexora::TimeState). Use `nexora::data` for
typed field access. See [ReadField](@ref nexora::ReadField) and
[WriteField](@ref nexora::WriteField) for how long a binding must stay alive.

The [scalar type](@ref nexora::real) reflects the configuration used to build
this reference.

## Implement a participant

Derive from [ParticipantIF](@ref nexora::ParticipantIF) and implement its
initialization, time constraint, step and cleanup contracts. The public
[step](@ref nexora::ParticipantIF::step) checks admission before calling the
protected [do_step](@ref nexora::ParticipantIF::do_step) extension point.
