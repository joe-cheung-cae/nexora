# ADR-014: License and provenance remain pending

- Status: pending maintainer decision
- Recorded at `2026-11-06T11:01:56+08:00`

## Context

The architecture plan explicitly leaves the existing code license and reusable legacy scope undecided. Selecting an open-source license would grant rights and cannot be inferred from the project name or plan.

## Decision

No project license is selected in the initialization work. `LICENSE` is a neutral status and reservation notice, not an OSI license or a grant. No legacy or third-party source is presumed reusable. Before importing it, record origin, authorship/rights holder where known, license terms, and the intended reuse scope.

## Consequences

Distribution and external contribution terms remain unresolved. Maintainers must select the license and replace this notice before describing the project as open source or accepting contributions under assumed terms.
