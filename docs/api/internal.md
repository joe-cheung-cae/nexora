@mainpage Nexora Maintainer Reference

Use this reference to navigate the implementation of Nexora. The separately
generated C++ API reference covers installed headers and extension points.

## Start with the lifecycle

- [Simulation](@ref nexora::Simulation) coordinates a caller-owned participant
  and advances the public accepted clock and step count after successful steps.
- [ParticipantIF](@ref nexora::ParticipantIF) defines the synchronous
  lifecycle. [step](@ref nexora::ParticipantIF::step) checks time-step admission
  before calling the implementation's protected
  [do_step](@ref nexora::ParticipantIF::do_step) extension point.
- [WcsphParticipant3D](@ref nexora::sph::WcsphParticipant3D) owns the
  [accepted state](@ref nexora::sph::WcsphState3D) and publishes its replacement
  after a successful step. [WcsphStep3D](@ref nexora::sph::WcsphStep3D) owns
  reusable workspace, the derived grid cache, and the configured wall groups.
- [SolidWallBoundary3D](@ref nexora::sph::SphBoundary::SolidWallBoundary3D)
  provides access to wall reference particles and samples position, velocity
  and acceleration through the fixed or prescribed moving wall implementation.

## Read a class page

Use navigation or search to find a type and the shortcuts near its title to
jump to member groups. Look near the top for the description, include directive,
and relationship diagrams when available. The member summary lists fields
first and groups declarations by C++ access level and kind. Select a name to
read its full signature and documentation. Use the source links to inspect
the original declaration.

## Read a UML diagram {#maintainer_uml_legend}

For each relationship graph, the overview shows the connected types. The UML
detail directly below it adds member declarations to the same nodes. Follow a
linked type in either view to open its class page. In a narrow window, scroll
the wider diagrams horizontally.

In UML boxes, `+` marks public members, `#` marks protected members, and `-` marks
private members. Each field starts with its access marker, followed by its C++
type and name, as in `- bool operation_active_`. When a field already appears on a
collaboration edge, the box may omit that field. Boxes can also shorten their
member lists. Use the class page's member sections for full declarations and
documentation.

An open triangle points to the base type in an inheritance relationship. An
open diamond marks the class that declares the member in a collaboration
relationship. The edge label identifies that member. Read its declaration and
documentation to determine storage, ownership, and lifetime.
For example,
[Simulation::participant_](@ref nexora::Simulation::participant_) borrows
the caller's participant, [Simulation::time_](@ref nexora::Simulation::time_)
stores the accepted clock by value, and
[WcsphParticipant3D::accepted_](@ref nexora::sph::WcsphParticipant3D::accepted_)
owns the accepted state exclusively.
