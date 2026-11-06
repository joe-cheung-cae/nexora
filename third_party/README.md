# Third-party dependency policy

Nexora currently has no vendored third-party dependencies. The default CPU build must remain offline and dependency-light. Optional dependencies are discovered only when their explicit feature is enabled.

Before adding or vendoring a dependency, record its upstream source, exact version or revision, license, patches, build/runtime role, and whether it is required for the default build. Do not commit binaries or large dependency archives here. The project license and legacy reuse rights are still pending; see [ADR-014](../docs/adr/ADR-014-license-and-provenance.md).
