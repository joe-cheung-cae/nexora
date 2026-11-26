#!/usr/bin/env python3
"""Check direct CMake target links and C/C++ include boundaries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETIRED_TARGETS = {"nexora_exec_api", "nexora_exec_cpu", "nexora_exec_tests",
                   "nexora::exec_api", "nexora::exec_cpu"}
RETIRED_PUBLIC_HEADERS = {"core/ArrayView.hpp", "core/math/Vec.hpp", "core/math/Tensor.hpp",
                          "data/FieldView.hpp"}

FIXED_CSR_TARGET = "nexora_sph_fixed_csr_fixture"
FIXED_CSR_LINKS = {FIXED_CSR_TARGET, "nexora_simulation"}
SPH_LEAF_LINKS = {"nexora_participants_sph", "nexora_simulation"}
ALLOWED_LINKS = {
    "nexora_build_options": set(),
    "nexora_core": {"nexora_build_options"},
    "nexora_data_schema": {"nexora_core"},
    "nexora_data": {"nexora_data_schema", "Kokkos::kokkos"},
    "nexora_participant_api": {"nexora_data_schema"},
    "nexora_simulation": {"nexora_core", "nexora_participant_api"},
    "nexora_methods_reference": {"nexora_data", "Kokkos::kokkos"},
    "nexora_participants_reference": {
        "nexora_participant_api", "nexora_methods_reference", "nexora_data"},
    "nexora_methods_sph": {"nexora_core", "Kokkos::kokkos"},
    "nexora_lepton": set(),
    "nexora_prescribed_motion": {"nexora_methods_sph", "nexora_lepton"},
    "nexora_participants_sph": {
        "nexora_participant_api", "nexora_methods_sph", "nexora_data", "Kokkos::kokkos"},
    FIXED_CSR_TARGET: {"nexora_participant_api", "nexora_data", "Kokkos::kokkos"},
    "nexora_sph_benchmark": FIXED_CSR_LINKS,
    "nexora": {"nexora_core", "nexora_simulation", "nexora_participants_reference",
               "Kokkos::kokkos", "MPI::MPI_C"},
}
TEST_LINKS = set(ALLOWED_LINKS) - {"nexora", "nexora_build_options", "nexora_methods_sph",
                                   "nexora_participants_sph", "nexora_lepton", "nexora_prescribed_motion",
                                   FIXED_CSR_TARGET, "nexora_sph_benchmark"}
REQUIRED_LINKS = {
    "nexora_data_schema": {"nexora_core"},
    "nexora_data": {"nexora_data_schema", "Kokkos::kokkos"},
    "nexora_participant_api": {"nexora_data_schema"},
    "nexora_methods_reference": {"nexora_data", "Kokkos::kokkos"},
    "nexora_participants_reference": {
        "nexora_participant_api", "nexora_methods_reference", "nexora_data"},
    "nexora_methods_sph": ALLOWED_LINKS["nexora_methods_sph"],
    "nexora_prescribed_motion": ALLOWED_LINKS["nexora_prescribed_motion"],
    "nexora_participants_sph": ALLOWED_LINKS["nexora_participants_sph"],
    FIXED_CSR_TARGET: ALLOWED_LINKS[FIXED_CSR_TARGET],
    "nexora_sph_benchmark": FIXED_CSR_LINKS,
    "nexora": {"nexora_core", "nexora_simulation", "nexora_participants_reference", "Kokkos::kokkos"},
}

GTEST_MAIN = "GTest::gtest_main"
KOKKOS_TEST_MAIN = "nexora_kokkos_test_main"
REFERENCE_TEST_LINKS = {"nexora_participants_reference", "nexora_simulation"}
SPECIAL_TEST_LINKS = {
    KOKKOS_TEST_MAIN: {"GTest::gtest", "Kokkos::kokkos"},
    "nexora_core_tests": {GTEST_MAIN, "nexora_core"},
    "nexora_data_tests": {GTEST_MAIN, "nexora_data_schema"},
    "nexora_simulation_tests": {GTEST_MAIN, "nexora_simulation"},
    "nexora_data_uninitialized_tests": {GTEST_MAIN, "nexora_data"},
    "nexora_reference_uninitialized_tests": {GTEST_MAIN, "nexora_participants_reference"},
    "nexora_entity_topology_tests": {KOKKOS_TEST_MAIN, "nexora_data"},
    "nexora_structure_reorder_tests": {KOKKOS_TEST_MAIN, "nexora_data"},
    "nexora_field_group_tests": {KOKKOS_TEST_MAIN, "nexora_data", "Threads::Threads"},
    "nexora_field_binding_tests": {KOKKOS_TEST_MAIN, "nexora_data"},
    "nexora_kokkos_tests": {KOKKOS_TEST_MAIN, "nexora_data"},
    "nexora_reference_tests": {KOKKOS_TEST_MAIN, *REFERENCE_TEST_LINKS},
    "nexora_reference_resource_tests": {KOKKOS_TEST_MAIN, *REFERENCE_TEST_LINKS},
    "nexora_sph_tests": {KOKKOS_TEST_MAIN, *SPH_LEAF_LINKS, "nexora_prescribed_motion"},
    "nexora_sph_fixed_csr_tests": {KOKKOS_TEST_MAIN, *FIXED_CSR_LINKS},
    "nexora_sph_tank_trajectory": SPH_LEAF_LINKS,
    "nexora_transport_tests": {KOKKOS_TEST_MAIN, "nexora_data"},
    "nexora_transport_receive_tests": {KOKKOS_TEST_MAIN, "nexora_data"},
    "nexora_serial_irregular_measurement": {"nexora_data"},
    "nexora_reference_cli_oracle": REFERENCE_TEST_LINKS,
}
REQUIRED_LINKS.update(SPECIAL_TEST_LINKS)

ALLOWED_INCLUDES = {
    "core": {"core"},
    "data_schema": {"core", "data_schema"},
    "data": {"core", "data_schema", "data"},
    "participant": {"core", "data_schema", "participant"},
    "simulation": {"core", "data_schema", "participant", "simulation"},
}
SOURCE_MODULE_INCLUDES = {
    **ALLOWED_INCLUDES,
    "methods": {"core", "data_schema", "data"},
    "participants": {"core", "data_schema", "data", "participant"},
}
SCHEMA_HEADERS = {"FieldDescriptor.hpp", "FieldRegistry.hpp", "FieldStamp.hpp"}
ENTITY_TOPOLOGY_TEST = Path("tests/data/entity_topology_tests.cpp")
STRUCTURE_REORDER_TEST = Path("tests/data/structure_reorder_tests.cpp")
STRUCTURE_FIXTURE = Path("tests/data/structure_fixture.hpp")
SERIAL_IRREGULAR_MEASUREMENT = Path("tests/data/serial_irregular_measurement.cpp")
STRUCTURE_FIXTURE_CONSUMERS = {STRUCTURE_REORDER_TEST, SERIAL_IRREGULAR_MEASUREMENT}
STRUCTURE_STATE = Path("tests/validation/structure/StructureState.hpp")
STRUCTURE_STATE_SOURCES = {STRUCTURE_STATE, STRUCTURE_STATE.with_suffix(".cpp")}
FIELD_GROUP_TEST = Path("tests/data/field_group_tests.cpp")
BINDING_TEST = Path("tests/data/field_binding_tests.cpp")
TRANSPORT_TEST = Path("tests/data/transport_tests.cpp")
RECEIVE_TEST = Path("tests/data/transport_receive_tests.cpp")
REFERENCE_TEST = Path("tests/reference/reference_tests.cpp")
REFERENCE_CASE_HEADER = Path("apps/nexora/ReferenceCase.hpp")
REFERENCE_COMMAND_SOURCE = Path("apps/nexora/ReferenceCommand.cpp")
REFERENCE_ASSEMBLY_SOURCES = {REFERENCE_CASE_HEADER, REFERENCE_CASE_HEADER.with_suffix(".cpp"),
                              REFERENCE_COMMAND_SOURCE}
SPH_METHOD_DIR = ("src", "methods", "sph")
SPH_PARTICIPANT_DIR = ("src", "participants", "sph")
FIXED_CSR_DIR = Path("validation/sph/fixed_csr")
FIXED_CSR_PARTICIPANT = FIXED_CSR_DIR / "Participant.hpp"
SPH_CELL_GRID_TEST = Path("tests/methods/sph/cell_grid_tests.cpp")
RESOURCE_BUDGET_HEADER = "participants/ResourceBudget.hpp"
RESOURCE_BUDGET_CONSUMERS = {
    Path("src/participants/reference/Participant.hpp"),
    FIXED_CSR_PARTICIPANT,
    Path(*SPH_PARTICIPANT_DIR) / "Geometry3D.hpp",
    SPH_CELL_GRID_TEST,
}
FIXED_CSR_CONSUMERS = {
    "nexora_sph_fixed_csr_tests": Path("tests/methods/sph/sph_tests.cpp"),
    "nexora_sph_benchmark": Path("benchmarks/sph/sph_benchmark.cpp"),
}
SPH_ADAMI_TEST = SPH_CELL_GRID_TEST.with_name("adami_wall_tests.cpp")
SPH_ADAMI_HEADER = "methods/sph/AdamiWallPressure3D.hpp"
SPH_ADAMI_CONSUMERS = {
    SPH_ADAMI_TEST,
    Path("src") / SPH_ADAMI_HEADER,
    Path("src") / Path(SPH_ADAMI_HEADER).with_suffix(".cpp"),
    Path(*SPH_METHOD_DIR) / "Pressure3D.cpp",
}
SPH_PRESSURE_TEST = SPH_CELL_GRID_TEST.with_name("pressure_3d_tests.cpp")
SPH_VERLET_TEST = SPH_CELL_GRID_TEST.with_name("verlet_window_tests.cpp")
SPH_MOVING_WALL_HEADER = "methods/sph/PrescribedMovingWall3D.hpp"
PRESCRIBED_MOTION_HEADER = Path("apps/prescribed_motion/PrescribedMotion.hpp")
SPH_MOVING_WALL_CONSUMERS = {
    PRESCRIBED_MOTION_HEADER,
    Path("src") / SPH_MOVING_WALL_HEADER,
    (Path("src") / SPH_MOVING_WALL_HEADER).with_suffix(".cpp"),
    Path(*SPH_PARTICIPANT_DIR) / "WcsphStep3D.hpp",
    SPH_VERLET_TEST,
}
SPH_HYDROSTATIC_INITIAL_TEST = SPH_CELL_GRID_TEST.with_name("hydrostatic_initial_tests.cpp")
SPH_CONTINUITY_TEST = SPH_CELL_GRID_TEST.with_name("continuity_tests.cpp")
SPH_TANK_FIXTURE = SPH_CELL_GRID_TEST.with_name("hydrostatic_tank_fixture.hpp")
SPH_TANK_TRAJECTORY = SPH_CELL_GRID_TEST.with_name("hydrostatic_trajectory.cpp")
SPH_TARGET_DIRS = {
    FIXED_CSR_TARGET: FIXED_CSR_DIR,
    "nexora_sph_tests": SPH_CELL_GRID_TEST.parent,
    "nexora_sph_tank_trajectory": SPH_TANK_TRAJECTORY.parent,
    **{target: source.parent for target, source in FIXED_CSR_CONSUMERS.items()},
}
SPH_PRESSURE_ORACLE = SPH_CELL_GRID_TEST.with_name("wcsph_oracle.hpp")
SPH_WINDOW_TESTS = {SPH_ADAMI_TEST,
                    SPH_CELL_GRID_TEST.with_name("hydrostatic_tank_tests.cpp"),
                    SPH_VERLET_TEST, SPH_HYDROSTATIC_INITIAL_TEST, SPH_CONTINUITY_TEST}
SPH_PARTICIPANT_CONSUMERS = (SPH_WINDOW_TESTS - {SPH_HYDROSTATIC_INITIAL_TEST}) | {SPH_TANK_TRAJECTORY}
SPH_PARTICIPANT_HEADER = "participants/sph/WcsphParticipant3D.hpp"
SPH_STEP_HEADER = "participants/sph/WcsphStep3D.hpp"
SPH_GEOMETRY_SOURCES = {
    Path(*SPH_PARTICIPANT_DIR) / name
    for name in ("Geometry3D.hpp", "Geometry3D.cpp")
}
SPH_GRID_ORACLE = SPH_CELL_GRID_TEST.with_name("cell_grid_oracle.hpp")
DATA_PREINIT_TEST = Path("tests/data/preinitialization_tests.cpp")
KOKKOS_VIEW_PROBE_CONSUMERS = {Path("tests/kokkos/kokkos_tests.cpp"),
                               Path("tests/package/consumer/kokkos.cpp")}
# These test-tree sources are also compiled by standalone programs.
STANDALONE_TEST_SOURCES = {
    SERIAL_IRREGULAR_MEASUREMENT, STRUCTURE_FIXTURE, Path("tests/kokkos/view_probe.hpp"),
    SPH_TANK_TRAJECTORY, SPH_TANK_FIXTURE,
    Path("tests/reference/cli_oracle.cpp"),
}
PRIVATE_DATA_HEADERS = {
    "FieldIssuer.hpp": {FIXED_CSR_PARTICIPANT.with_suffix(".cpp"), DATA_PREINIT_TEST,
                        STRUCTURE_FIXTURE, BINDING_TEST, FIELD_GROUP_TEST,
                        TRANSPORT_TEST, RECEIVE_TEST},
    "ExternalField.hpp": {DATA_PREINIT_TEST, TRANSPORT_TEST},
    "TransportBuilder.hpp": {STRUCTURE_FIXTURE, TRANSPORT_TEST, RECEIVE_TEST},
    "TransportReceiver.hpp": {STRUCTURE_FIXTURE, RECEIVE_TEST},
}
PRIVATE_ENTITY_HEADERS = {"EntityCapacity.hpp", "EntityOrder.hpp", "Permutation.hpp", "CsrAdjacency.hpp"}
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".cuh"}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", *HEADER_SUFFIXES}
INCLUDE = re.compile(r'^\s*#\s*include\s*(?:"([^"\n]+)"|<([^>\n]+)>)', re.MULTILINE)
PLATFORM_HEADER = re.compile(
    r"(?:mpi\.h|cuda[^/]*\.h|device_launch_parameters\.h|vector_types\.h)", re.IGNORECASE
)
KOKKOS_HEADER = re.compile(r"Kokkos(?:_[A-Za-z0-9_]+)?\.(?:h|hpp)", re.IGNORECASE)


def check_link_policy(name: str, links: list[str]) -> list[str]:
    """Check a target's direct dependencies against the narrow target policy."""
    if name in RETIRED_TARGETS:
        return [f"{name}: retired execution target is forbidden"]
    if name in ALLOWED_LINKS:
        allowed = ALLOWED_LINKS[name]
    elif name in SPECIAL_TEST_LINKS:
        allowed = SPECIAL_TEST_LINKS[name]
    else:
        allowed = TEST_LINKS
    errors = [f"{name}: forbidden direct dependency on {link}" for link in links
              if link not in allowed]
    errors.extend(f"{name}: missing required direct dependency on {link}"
                  for link in REQUIRED_LINKS.get(name, set()) if link not in links)
    return errors


def check_kokkos_header(relative: Path, scope: str, module: str, header: str) -> list[str]:
    """Keep Kokkos headers at approved numerical, validation, and CLI boundaries."""
    header_name = header.replace("\\", "/").split("/")[-1]
    if not KOKKOS_HEADER.fullmatch(header_name):
        return []
    if ((scope == "include/nexora" and module == "data")
            or (scope == "src" and module == "data")
            or (scope == "src" and relative.parts[:3] == ("src", "methods", "reference"))
            or relative.parts[:3] in {SPH_METHOD_DIR, SPH_PARTICIPANT_DIR}
            or relative.parent == FIXED_CSR_DIR or relative in FIXED_CSR_CONSUMERS.values()
            or relative in {SPH_CELL_GRID_TEST, SPH_PRESSURE_TEST, SPH_GRID_ORACLE,
                            SPH_TANK_TRAJECTORY, *SPH_WINDOW_TESTS}
            or relative in STRUCTURE_STATE_SOURCES
            or (scope == "tests" and relative.parts[:2] in {
                ("tests", "kokkos"), ("tests", "reference")})
            or relative in {STRUCTURE_FIXTURE, *STRUCTURE_FIXTURE_CONSUMERS,
                            ENTITY_TOPOLOGY_TEST, FIELD_GROUP_TEST,
                            BINDING_TEST, TRANSPORT_TEST, RECEIVE_TEST}
            or (scope == "apps" and module == "nexora")):
        return []
    return [f"{relative}: Kokkos header {header} is outside an approved numerical, validation, or CLI boundary"]


@cache
def implementation_file_paths() -> dict[tuple[int, int], Path]:
    """Retain the repository spelling on filesystems with insensitive lookup."""
    paths = {}
    for path in (ROOT / "src").rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        info = path.stat()
        paths[(info.st_dev, info.st_ino)] = path.relative_to(ROOT)
    return paths


def resolve_project_include(relative: Path, header: str, quoted: bool) -> Path | None:
    """Resolve explicit paths and the policy's canonical implementation names."""
    spelling = Path(header.replace("\\", "/"))
    if spelling.is_absolute():
        candidates = [spelling]
    else:
        candidates = [ROOT / relative.parent / spelling] if quoted else []
        if spelling.parts and spelling.parts[0].casefold() in {"methods", "participants"}:
            candidates.append(ROOT / "src" / spelling)
    for candidate in candidates:
        if candidate.is_file():
            info = candidate.stat()
            target = implementation_file_paths().get((info.st_dev, info.st_ino))
            if target is not None:
                return target
            resolved = candidate.resolve()
            return resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else None
    return None


def check_internal_include(relative: Path, scope: str, header: str,
                           target: Path | None = None) -> list[str]:
    """Enforce downward use of implementation-only headers."""
    if target is not None:
        parts = list(target.relative_to("src").parts if target.is_relative_to("src") else target.parts)
    else:
        parts = [part for part in header.replace("\\", "/").split("/") if part not in {"", "."}]
    if not parts:
        return []
    location = relative.parts
    target_name = "/".join(parts)
    if ("lepton" in (part.lower() for part in parts) or parts[-1].lower() == "lepton.h") and (
            relative != PRESCRIBED_MOTION_HEADER.with_suffix(".cpp")):
        return [f"{relative}: {header} belongs in the prescribed-motion implementation"]
    if parts[-1] == PRESCRIBED_MOTION_HEADER.name and scope != "apps" and relative != SPH_VERLET_TEST:
        return [f"{relative}: {header} belongs in application assembly and its SPH integration test"]
    if ({"gtest", "gmock"}.intersection(parts) or parts[-1].startswith(("gtest", "gmock"))) and (
            scope != "tests" or location[:2] == ("tests", "package") or relative in STANDALONE_TEST_SOURCES):
        return [f"{relative}: {header} is private to GoogleTest unit targets"]
    if parts[-1] == "view_probe.hpp" and relative not in KOKKOS_VIEW_PROBE_CONSUMERS:
        return [f"{relative}: {header} is private to the Kokkos unit test and installed consumer"]
    if parts[-1] == STRUCTURE_STATE.name:
        if relative in {*STRUCTURE_STATE_SOURCES, STRUCTURE_FIXTURE}:
            return []
        return [f"{relative}: {header} is private to the structure validation fixture"]
    if parts[-1] == STRUCTURE_FIXTURE.name:
        if relative in STRUCTURE_FIXTURE_CONSUMERS:
            return []
        return [f"{relative}: {header} is private to the structure test and measurement"]
    if "fixed_csr" in parts or parts[-1] == "sph_fixture.hpp":
        if relative.parent == FIXED_CSR_DIR or relative in FIXED_CSR_CONSUMERS.values():
            return []
        return [f"{relative}: {header} is private to the fixed CSR fixture, test and benchmark"]
    if (target_name == SPH_ADAMI_HEADER
            or (target is None and parts[-1] == Path(SPH_ADAMI_HEADER).name)):
        if target_name == SPH_ADAMI_HEADER and relative in SPH_ADAMI_CONSUMERS:
            return []
        return [f"{relative}: {header} is private to Adami pressure evaluation and its regression"]
    if (target_name == SPH_MOVING_WALL_HEADER
            or (target is None and parts[-1] == Path(SPH_MOVING_WALL_HEADER).name)):
        if target_name == SPH_MOVING_WALL_HEADER and relative in SPH_MOVING_WALL_CONSUMERS:
            return []
        return [f"{relative}: {header} is private to prescribed WCSPH motion and its window tests"]
    if parts[-1] == SPH_PRESSURE_ORACLE.name:
        if relative in {SPH_PRESSURE_TEST, SPH_VERLET_TEST}:
            return []
        return [f"{relative}: {header} is private to WCSPH pressure and integration regressions"]
    if parts[-1] == SPH_GRID_ORACLE.name:
        if relative in {SPH_CELL_GRID_TEST, SPH_PRESSURE_ORACLE,
                        SPH_TANK_TRAJECTORY, SPH_TANK_FIXTURE, *SPH_WINDOW_TESTS}:
            return []
        return [f"{relative}: {header} is private to the 3D SPH regression tests"]
    if relative in SPH_GEOMETRY_SOURCES and parts[-1] == "CellGrid.hpp":
        return [f"{relative}: geometry version comparison uses particle fields and layout indices"]
    if parts[-1] in {"NeighborView3D.hpp", "ParticleFields3D.hpp"}:
        if location[:3] in {SPH_METHOD_DIR, SPH_PARTICIPANT_DIR} or relative == SPH_CELL_GRID_TEST:
            return []
        return [f"{relative}: {header} is private to 3D SPH methods and their assembly"]
    if parts[-1] == "ResourceBudget.hpp":
        if "/".join(parts) == RESOURCE_BUDGET_HEADER and relative in RESOURCE_BUDGET_CONSUMERS:
            return []
        return [f"{relative}: {header} is private to the Reference and SPH participants"]
    if parts[-1] in PRIVATE_ENTITY_HEADERS:
        if (location[:2] == ("src", "data") or relative in STRUCTURE_STATE_SOURCES
                or location[:3] == SPH_PARTICIPANT_DIR
                or relative in {STRUCTURE_FIXTURE, ENTITY_TOPOLOGY_TEST}
                or (relative in {FIXED_CSR_PARTICIPANT, FIXED_CSR_PARTICIPANT.with_suffix(".cpp")}
                    and parts[-1] in {"CsrAdjacency.hpp", "Permutation.hpp"})):
            return []
        return [f"{relative}: {header} is private to data and the approved structure boundaries"]
    if parts[-1] in PRIVATE_DATA_HEADERS:
        if (location[:2] in {("src", "data"), ("src", "participants")}
                or relative in STRUCTURE_STATE_SOURCES
                or relative in PRIVATE_DATA_HEADERS[parts[-1]]):
            return []
        return [f"{relative}: {header} is private to data and concrete participants"]

    if parts[0] not in {"methods", "participants"}:
        return []
    if len(parts) < 3 or parts[1] not in {"reference", "sph"}:
        return [f"{relative}: internal include {header} is not a known module path"]
    if parts[1] == "sph":
        if (location[:3] == SPH_PARTICIPANT_DIR
                or (parts[0] == "methods"
                    and (location[:3] == SPH_METHOD_DIR or relative in {SPH_CELL_GRID_TEST, SPH_GRID_ORACLE}))
                or (relative == SPH_VERLET_TEST and "/".join(parts) == SPH_STEP_HEADER)
                or (relative == SPH_HYDROSTATIC_INITIAL_TEST
                    and parts == ["methods", "sph", "Density3D.hpp"])
                or (relative in {SPH_PRESSURE_TEST, SPH_PRESSURE_ORACLE, SPH_TANK_FIXTURE,
                                    SPH_HYDROSTATIC_INITIAL_TEST}
                    and parts == ["methods", "sph", "Pressure3D.hpp"])
                or (relative == SPH_CONTINUITY_TEST and parts[:2] == ["methods", "sph"]
                    and parts[2:] in (["Continuity3D.hpp"], ["Verlet3D.hpp"]))
                or (relative in SPH_PARTICIPANT_CONSUMERS
                    and "/".join(parts) == SPH_PARTICIPANT_HEADER)):
            return []
        return [f"{relative}: forbidden internal include {header} from this module"]

    in_method = scope == "src" and location[:3] == ("src", "methods", "reference")
    in_participant = scope == "src" and location[:3] == ("src", "participants", "reference")
    in_reference_test = scope == "tests" and location[:2] == ("tests", "reference")
    if (relative in REFERENCE_ASSEMBLY_SOURCES
            and parts == ["participants", "reference", "Participant.hpp"]):
        return []
    if parts[0] == "methods" and (in_method or in_participant or in_reference_test):
        return []
    if parts[0] == "participants" and (in_participant or in_reference_test):
        return []
    return [f"{relative}: forbidden internal include {header} from this module"]


def public_module(parts: tuple[str, ...]) -> str:
    """Map public data headers to their actual target layer."""
    if len(parts) == 2 and parts[0] == "data" and parts[1] in SCHEMA_HEADERS:
        return "data_schema"
    return parts[0]


def check_public_include(relative: Path, scope: str, allowed: set[str], header: str,
                         implementation_header: bool = False) -> list[str]:
    parts = tuple(header.replace("\\", "/").split("/"))
    if ".." in parts:
        if implementation_header and parts[0] != "nexora":
            return []
        return [f"{relative}: relative traversal include is forbidden"]
    if len(parts) > 1 and parts[0] == "nexora":
        public_parts = parts[1:]
    elif scope == "include/nexora" and not Path(header).is_absolute():
        # A same-directory include has the same boundary as its canonical public spelling.
        public_parts = (relative.parent / header).relative_to(scope).parts
        if (not (ROOT / relative.parent / header).is_file()
                and "/".join(public_parts) not in RETIRED_PUBLIC_HEADERS
                and public_parts[0] != "exec"):
            return []
    else:
        return []
    if public_parts[0] == "exec" or "/".join(public_parts) in RETIRED_PUBLIC_HEADERS:
        return [f"{relative}: retired public include {header} is forbidden"]
    target = public_module(public_parts)
    if target not in allowed:
        return [f"{relative}: forbidden include {header}"]
    return []


def check_source_includes(relative: Path, scope: str, module: str, allowed: set[str],
                          source: str) -> list[str]:
    errors: list[str] = []
    for quoted, angled in INCLUDE.findall(source):
        header = quoted or angled
        target = resolve_project_include(relative, header, bool(quoted))
        errors.extend(check_kokkos_header(relative, scope, module, header))
        errors.extend(check_internal_include(relative, scope, header, target))
        implementation_header = (target is not None and target.suffix in HEADER_SUFFIXES
                                 and target.parts[:2] in {("src", "methods"), ("src", "participants")})
        errors.extend(check_public_include(relative, scope, allowed, header, implementation_header))
        header_name = header.replace("\\", "/").split("/")[-1]
        if (scope, module) != ("apps", "nexora") and PLATFORM_HEADER.fullmatch(header_name):
            errors.append(f"{relative}: CUDA/MPI header {header} belongs in the optional CLI")
    return errors


def policy_self_test() -> list[str]:
    """Exercise the Kokkos, schema, and private data boundaries."""
    failures: list[str] = []
    for target in RETIRED_TARGETS:
        if not check_link_policy(target, []):
            failures.append(f"checker self-test failed to reject retired target {target}")
        if not check_link_policy("nexora_core_tests", [target]):
            failures.append(f"checker self-test failed to reject a dependency on {target}")
    for header in sorted(RETIRED_PUBLIC_HEADERS | {"exec/Buffer.hpp", "exec/ExecContext.hpp",
                                                   "exec/CompletionToken.hpp", "exec/MemorySpace.hpp"}):
        for path, scope, allowed in (
            (Path("include/nexora/core/Status.hpp"), "include/nexora", ALLOWED_INCLUDES["core"]),
            (Path("include/nexora/data/FieldBinding.hpp"), "include/nexora", ALLOWED_INCLUDES["data"]),
            (Path("tests/core/core_tests.cpp"), "tests", set(ALLOWED_INCLUDES)),
        ):
            if not check_public_include(path, scope, allowed, f"nexora/{header}"):
                failures.append(f"checker self-test failed to reject retired {header} from {path}")
        path = Path("include/nexora") / Path(header).parent / "test.hpp"
        if not check_public_include(path, "include/nexora", set(ALLOWED_INCLUDES), Path(header).name):
            failures.append(f"checker self-test failed to reject relative retired include {header}")
    for header, module in (("nexora/core/Time.hpp", "core"),
                           ("nexora/core/Real.hpp", "core"),
                           ("nexora/core/Config.hpp", "core"),
                           ("nexora/data/FieldBinding.hpp", "data")):
        path = Path("include/nexora") / module / "test.hpp"
        if check_public_include(path, "include/nexora", ALLOWED_INCLUDES[module], header):
            failures.append(f"checker self-test failed to allow retained include {header}")
    for target, links in (
        ("nexora_data", ["nexora_data_schema", "Kokkos::kokkos"]),
        ("nexora_data_schema", ["nexora_core"]),
        ("nexora_participant_api", ["nexora_data_schema"]),
        ("nexora_participants_reference",
         ["nexora_participant_api", "nexora_methods_reference", "nexora_data"]),
        ("nexora_methods_sph", ["nexora_core", "Kokkos::kokkos"]),
        ("nexora_participants_sph",
         ["nexora_participant_api", "nexora_methods_sph", "nexora_data", "Kokkos::kokkos"]),
        (FIXED_CSR_TARGET, ALLOWED_LINKS[FIXED_CSR_TARGET]),
        ("nexora_sph_benchmark", FIXED_CSR_LINKS),
        ("nexora", ["nexora_core", "nexora_simulation", "nexora_participants_reference", "Kokkos::kokkos"]),
        *SPECIAL_TEST_LINKS.items(),
    ):
        if check_link_policy(target, links) or any(
            not check_link_policy(target, [link for link in links if link != removed])
            for removed in links
        ):
            failures.append(f"checker self-test failed to enforce required edges for {target}")
    if not check_link_policy("nexora_data_schema", ["nexora_core", "Kokkos::kokkos"]):
        failures.append("checker self-test failed to reject Kokkos from data schema")
    if not check_link_policy("nexora_participant_api", ["nexora_data_schema", "nexora_data"]):
        failures.append("checker self-test failed to reject typed data from participant API")
    if not check_link_policy("nexora_core", ["Kokkos::kokkos"]):
        failures.append("checker self-test failed to reject Kokkos from nexora_core")
    for target in ("nexora", "nexora_core", "nexora_data", "nexora_simulation"):
        for docs_target in ("nexora_docs", "nexora_internal_docs", "Doxygen::doxygen"):
            if not check_link_policy(target, [*ALLOWED_LINKS[target], docs_target]):
                failures.append(f"checker self-test allowed {target} to depend on {docs_target}")
    cli_links = ["nexora_core", "nexora_simulation", "nexora_participants_reference", "Kokkos::kokkos"]
    if check_link_policy("nexora", cli_links + ["MPI::MPI_C"]):
        failures.append("checker self-test failed to allow the CLI's optional MPI diagnostic")
    if not check_link_policy("nexora", cli_links + ["nexora_methods_reference"]):
        failures.append("checker self-test failed to reject a direct CLI method target dependency")
    for target in ("nexora_methods_sph", "nexora_participants_sph", FIXED_CSR_TARGET, "nexora_sph_benchmark"):
        if not check_link_policy("nexora", cli_links + [target]):
            failures.append(f"checker self-test failed to reject a CLI dependency on {target}")
        if not check_link_policy("nexora_core_tests", [target]):
            failures.append(f"checker self-test failed to reject an unrelated test dependency on {target}")
    for target in ("nexora_methods_sph", "nexora_participants_sph"):
        if not check_link_policy(target, [*REQUIRED_LINKS[target], FIXED_CSR_TARGET]):
            failures.append(f"checker self-test failed to keep the fixed CSR fixture out of {target}")
    if not check_link_policy("nexora_sph_benchmark", [*FIXED_CSR_LINKS, "nexora_participants_sph"]):
        failures.append("checker self-test failed to isolate the fixed CSR benchmark from the solver")
    if check_link_policy("nexora_methods_reference", ["nexora_data", "Kokkos::kokkos"]):
        failures.append("checker self-test failed to allow reference-method dependencies")
    if not check_link_policy("nexora_methods_reference", ["nexora_data", "Kokkos::kokkos", "nexora_participant_api"]):
        failures.append("checker self-test failed to reject an upward method dependency")
    if not check_link_policy("nexora_participants_reference", [
            "nexora_participant_api", "nexora_methods_reference", "nexora_data", "Kokkos::kokkos"]):
        failures.append("checker self-test failed to reject a direct participant Kokkos dependency")
    for target, deps in SPECIAL_TEST_LINKS.items():
        for forbidden in (set(ALLOWED_LINKS) | {"GTest::gtest", GTEST_MAIN, KOKKOS_TEST_MAIN}) - deps:
            if not check_link_policy(target, [*deps, forbidden]):
                failures.append(f"checker self-test failed to reject {forbidden} from {target}")
    for target in ("nexora_core", "nexora", FIXED_CSR_TARGET, "nexora_sph_benchmark",
                   "nexora_reference_cli_oracle", "nexora_serial_irregular_measurement"):
        for dep in ("GTest::gtest", GTEST_MAIN, KOKKOS_TEST_MAIN):
            if not check_link_policy(target, [*REQUIRED_LINKS.get(target, ()), dep]):
                failures.append(f"checker self-test failed to keep {dep} out of {target}")
    for path in (Path("include/nexora/core/Status.hpp"), Path("src/data/Fields.cpp"),
                 REFERENCE_COMMAND_SOURCE, *STANDALONE_TEST_SOURCES,
                 Path("tests/package/consumer/kokkos.cpp"), FIXED_CSR_CONSUMERS["nexora_sph_benchmark"]):
        if not check_internal_include(path, path.parts[0], "gtest/gtest.h"):
            failures.append(f"checker self-test failed to keep GoogleTest out of {path}")
    for path in (Path("tests/kokkos/gtest_main.cpp"), DATA_PREINIT_TEST, REFERENCE_TEST):
        if check_internal_include(path, "tests", "gtest/gtest.h"):
            failures.append(f"checker self-test failed to allow GoogleTest in {path}")
    for path in KOKKOS_VIEW_PROBE_CONSUMERS:
        if check_internal_include(path, "tests", "view_probe.hpp"):
            failures.append(f"checker self-test failed to allow native View probe in {path}")
    if not check_internal_include(Path("src/data/Fields.cpp"), "src", "view_probe.hpp"):
        failures.append("checker self-test failed to keep native View probe out of production")
    if check_kokkos_header(Path("include/nexora/data/FieldBinding.hpp"),
                           "include/nexora", "data", "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to allow Kokkos in data headers")
    schema = Path("include/nexora/data/FieldRegistry.hpp")
    schema_module = public_module(schema.relative_to("include/nexora").parts)
    if not check_kokkos_header(schema, "include/nexora", schema_module, "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to reject Kokkos in schema headers")
    for header in ("FieldBinding.hpp", "TransportView.hpp"):
        if not check_public_include(schema, "include/nexora", ALLOWED_INCLUDES[schema_module],
                                    f"nexora/data/{header}"):
            failures.append(f"checker self-test failed to reject {header} in schema headers")
    for control in ("participant", "simulation"):
        path = Path("include/nexora") / control / "test.hpp"
        allowed = ALLOWED_INCLUDES[control]
        if check_public_include(path, "include/nexora", allowed, "nexora/data/FieldStamp.hpp"):
            failures.append(f"checker self-test failed to allow schema in {control}")
        for header in ("FieldAccess.hpp", "TransportView.hpp"):
            if not check_public_include(path, "include/nexora", allowed, f"nexora/data/{header}"):
                failures.append(f"checker self-test failed to reject {header} in {control}")
    if check_kokkos_header(RECEIVE_TEST, "tests", "data", "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to allow Kokkos in receive tests")
    for test_path in sorted({STRUCTURE_FIXTURE, *STRUCTURE_FIXTURE_CONSUMERS,
                             ENTITY_TOPOLOGY_TEST, FIELD_GROUP_TEST}):
        if check_kokkos_header(test_path, "tests", "data", "Kokkos_Core.hpp"):
            failures.append(f"checker self-test failed to allow Kokkos in {test_path}")
    if not check_internal_include(TRANSPORT_TEST, "tests", "data/TransportReceiver.hpp"):
        failures.append("checker self-test failed to reject receiver outside dedicated receive tests")
    for header in PRIVATE_DATA_HEADERS:
        if not check_internal_include(REFERENCE_TEST, "tests", f"data/{header}"):
            failures.append(f"checker self-test failed to reject {header} in Reference tests")
    for header, test_paths in PRIVATE_DATA_HEADERS.items():
        for test_path in test_paths:
            if check_internal_include(test_path, test_path.parts[0], f"data/{header}"):
                failures.append(f"checker self-test failed to allow {header} in {test_path}")
        if check_internal_include(Path("src/participants/reference/Participant.cpp"),
                                  "src", f"data/{header}"):
            failures.append(f"checker self-test failed to allow {header} in concrete participants")
        if not check_internal_include(Path("include/nexora/data/TransportView.hpp"),
                                      "include/nexora", header):
            failures.append(f"checker self-test failed to reject {header} in public data headers")
        if not check_internal_include(Path("src/methods/reference/State.hpp"),
                                      "src", f"data/{header}"):
            failures.append(f"checker self-test failed to reject {header} in methods")
    for header in PRIVATE_ENTITY_HEADERS:
        for path, scope in ((Path("src/data/TransportReceiver.hpp"), "src"),
                            (STRUCTURE_STATE, "tests"),
                            (Path("src") / SPH_PARTICIPANT_HEADER, "src"),
                            (ENTITY_TOPOLOGY_TEST, "tests"), (STRUCTURE_FIXTURE, "tests")):
            if check_internal_include(path, scope, f"data/{header}"):
                failures.append(f"checker self-test failed to allow {header} in {path}")
        for path, scope in ((Path("include/nexora/data/FieldBinding.hpp"), "include/nexora"),
                            (Path("src/methods/reference/State.hpp"), "src"),
                            (Path("src/methods/sph/Density3D.hpp"), "src"),
                            (Path("src/participants/reference/Participant.cpp"), "src"),
                            (RECEIVE_TEST, "tests")):
            if not check_internal_include(path, scope, f"data/{header}"):
                failures.append(f"checker self-test failed to reject {header} in {path}")
    if check_internal_include(STRUCTURE_STATE, "tests", "data/FieldIssuer.hpp"):
        failures.append("checker self-test failed to allow the structure state to own field grants")
    if check_internal_include(STRUCTURE_FIXTURE, "tests", "validation/structure/StructureState.hpp"):
        failures.append("checker self-test failed to allow the structure fixture's private state")
    if not check_internal_include(REFERENCE_COMMAND_SOURCE, "apps", "validation/structure/StructureState.hpp"):
        failures.append("checker self-test failed to reject validation structure state in the CLI")
    for path in sorted(STRUCTURE_FIXTURE_CONSUMERS):
        if check_internal_include(path, "tests", STRUCTURE_FIXTURE.name):
            failures.append(f"checker self-test failed to allow the structure fixture in {path}")
    if not check_internal_include(REFERENCE_COMMAND_SOURCE, "apps", STRUCTURE_FIXTURE.name):
        failures.append("checker self-test failed to reject the structure fixture in the CLI")
    for path in sorted({STRUCTURE_FIXTURE, *STRUCTURE_FIXTURE_CONSUMERS}):
        if not check_internal_include(path, "tests", "methods/reference/State.hpp"):
            failures.append(f"checker self-test failed to keep method state out of {path}")
    for path in sorted(RESOURCE_BUDGET_CONSUMERS):
        if check_internal_include(path, path.parts[0], RESOURCE_BUDGET_HEADER):
            failures.append(f"checker self-test failed to allow the participant budget in {path}")
    for path, scope in ((Path("include/nexora/data/TransportView.hpp"), "include/nexora"),
                        (Path("src/methods/sph/Density3D.hpp"), "src"),
                        (REFERENCE_COMMAND_SOURCE, "apps")):
        if not check_internal_include(path, scope, RESOURCE_BUDGET_HEADER):
            failures.append(f"checker self-test failed to reject the participant budget in {path}")
    if not check_kokkos_header(Path("include/nexora/core/Status.hpp"),
                               "include/nexora", "core", "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to reject Kokkos in core headers")
    if check_kokkos_header(Path("src/methods/reference/State.hpp"),
                           "src", "methods", "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to allow Kokkos in reference methods")
    if check_kokkos_header(STRUCTURE_STATE, "tests", "validation", "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to allow Kokkos in structure state")
    if check_internal_include(Path("src/participants/reference/Participant.hpp"),
                              "src", "methods/reference/State.hpp"):
        failures.append("checker self-test failed to allow participant use of method internals")
    if not check_internal_include(Path("src/methods/reference/State.hpp"),
                                  "src", "participants/reference/Participant.hpp"):
        failures.append("checker self-test failed to reject method use of participant internals")
    header = "participants/reference/Participant.hpp"
    for path in sorted(REFERENCE_ASSEMBLY_SOURCES):
        if check_internal_include(path, "apps", header):
            failures.append(f"checker self-test failed to allow the Reference case boundary in {path}")
        if not check_internal_include(path, "apps", "methods/reference/State.hpp"):
            failures.append(f"checker self-test failed to keep method state out of {path}")
    if not check_internal_include(Path("apps/nexora/main.cpp"), "apps", header):
        failures.append("checker self-test failed to restrict the CLI entry point from Participant.hpp")
    if not check_internal_include(Path("src/core/Time.hpp"),
                                  "src", "methods/reference/State.hpp"):
        failures.append("checker self-test failed to reject core use of method internals")
    if check_internal_include(REFERENCE_TEST, "tests", "methods/reference/State.hpp"):
        failures.append("checker self-test failed to allow reference tests to include internals")
    sph_method = Path("src/methods/sph/Density3D.hpp")
    sph_participant = Path("src") / SPH_PARTICIPANT_HEADER
    for path in SPH_PARTICIPANT_CONSUMERS:
        if check_internal_include(path, "tests", SPH_PARTICIPANT_HEADER):
            failures.append(f"checker self-test failed to allow the 3D SPH participant test in {path}")
    for path in (sph_participant, SPH_GRID_ORACLE, REFERENCE_COMMAND_SOURCE):
        if not check_internal_include(path, path.parts[0], SPH_GRID_ORACLE.name):
            failures.append(f"checker self-test failed to keep the 3D test oracle out of {path}")
        if not check_internal_include(path, path.parts[0], SPH_PRESSURE_ORACLE.name):
            failures.append(f"checker self-test failed to keep the WCSPH test oracle out of {path}")
    for path in (sph_method, sph_participant, SPH_CELL_GRID_TEST):
        if check_internal_include(path, path.parts[0], "methods/sph/Density3D.hpp"):
            failures.append(f"checker self-test failed to allow the SPH method in {path}")
    package_consumer = Path("tests/package/consumer/kokkos.cpp")
    for path in (sph_method, SPH_CELL_GRID_TEST, SPH_PRESSURE_TEST,
                 REFERENCE_COMMAND_SOURCE, package_consumer, SPH_HYDROSTATIC_INITIAL_TEST,
                 *FIXED_CSR_CONSUMERS.values()):
        if not check_internal_include(path, path.parts[0], SPH_PARTICIPANT_HEADER):
            failures.append(f"checker self-test failed to reject the SPH participant in {path}")
    for path in SPH_ADAMI_CONSUMERS:
        if check_internal_include(path, path.parts[0], SPH_ADAMI_HEADER):
            failures.append(f"checker self-test failed to allow Adami pressure use in {path}")
    stepper = ROOT / "src" / SPH_STEP_HEADER
    for spelling, forbidden in ((f'<{SPH_ADAMI_HEADER}>', False),
                                (f'"../../../src/{SPH_ADAMI_HEADER}"', False),
                                (f'"{stepper.as_posix()}"', True),
                                (f'<{stepper.as_posix()}>', True),
                                ('"../missing.hpp"', True)):
        errors = check_source_includes(SPH_ADAMI_TEST, "tests", "methods",
                                      set(ALLOWED_INCLUDES), f"#include {spelling}\n")
        if bool(errors) != forbidden:
            failures.append(f"checker self-test misclassified {spelling} in {SPH_ADAMI_TEST}")
    pressure = Path(*SPH_METHOD_DIR) / "Pressure3D.cpp"
    local_adami = Path(SPH_ADAMI_HEADER).name
    for spelling, forbidden in ((f'"{local_adami}"', False), (f'<{local_adami}>', True),
                                (f'"../../{SPH_PARTICIPANT_HEADER}"', True)):
        errors = check_source_includes(pressure, "src", "methods", ALLOWED_INCLUDES["core"],
                                      f"#include {spelling}\n")
        if bool(errors) != forbidden:
            failures.append(f"checker self-test misclassified {spelling} in {pressure}")
    for path in (sph_method, sph_participant, SPH_CELL_GRID_TEST, SPH_GRID_ORACLE,
                 REFERENCE_COMMAND_SOURCE, Path("include/nexora/core/Types.hpp"),
                 package_consumer, *FIXED_CSR_CONSUMERS.values()):
        relative_header = "../" * len(path.parent.parts) + "src/" + SPH_ADAMI_HEADER
        for header in (SPH_ADAMI_HEADER, relative_header):
            if not check_internal_include(path, path.parts[0], header):
                failures.append(f"checker self-test failed to keep Adami pressure out of {path}")
    for header in ("methods/sph/ParticleFields3D.hpp", "methods/sph/neighbors/NeighborView3D.hpp"):
        for path in (sph_method, sph_participant, SPH_CELL_GRID_TEST):
            if check_internal_include(path, path.parts[0], header):
                failures.append(f"checker self-test failed to allow {header} in {path}")
        for path in (REFERENCE_COMMAND_SOURCE, package_consumer, SPH_VERLET_TEST):
            if not check_internal_include(path, path.parts[0], header):
                failures.append(f"checker self-test failed to reject {header} in {path}")
    for header in ("methods/sph/neighbors/CellGrid.hpp", "CellGrid.hpp",
                   "../../methods/sph/neighbors/CellGrid.hpp"):
        for path in SPH_GEOMETRY_SOURCES:
            if not check_internal_include(path, "src", header):
                failures.append(f"checker self-test failed to keep {header} out of {path}")
    for name in ("Density3D.hpp", "Pressure3D.hpp"):
        if check_internal_include(SPH_HYDROSTATIC_INITIAL_TEST, "tests", f"methods/sph/{name}"):
            failures.append(f"checker self-test failed to allow direct {name} static diagnostics")
    if check_internal_include(SPH_VERLET_TEST, "tests", SPH_STEP_HEADER):
        failures.append("checker self-test failed to allow committed step inspection in the Verlet tests")
    for path in (SPH_ADAMI_TEST, SPH_TANK_TRAJECTORY, SPH_HYDROSTATIC_INITIAL_TEST,
                 REFERENCE_COMMAND_SOURCE, package_consumer):
        if not check_internal_include(path, path.parts[0], SPH_STEP_HEADER):
            failures.append(f"checker self-test failed to keep the SPH stepper out of {path}")
        if not check_internal_include(path, path.parts[0], "methods/sph/Continuity3D.hpp"):
            failures.append(f"checker self-test failed to keep continuity internals out of {path}")
    for path in SPH_MOVING_WALL_CONSUMERS:
        if check_internal_include(path, path.parts[0], SPH_MOVING_WALL_HEADER):
            failures.append(f"checker self-test failed to allow prescribed wall motion in {path}")
    for path in (PRESCRIBED_MOTION_HEADER, sph_method, sph_participant, SPH_VERLET_TEST, package_consumer):
        if not check_internal_include(path, path.parts[0], "lepton/Parser.h"):
            failures.append(f"checker self-test failed to keep Lepton out of {path}")
    for path in (SPH_CELL_GRID_TEST, SPH_PRESSURE_TEST, SPH_GRID_ORACLE, SPH_ADAMI_TEST,
                 REFERENCE_COMMAND_SOURCE, package_consumer, *FIXED_CSR_CONSUMERS.values(),
                 Path("include/nexora/core/Types.hpp")):
        for header in (SPH_MOVING_WALL_HEADER,
                       "../" * len(path.parent.parts) + "src/" + SPH_MOVING_WALL_HEADER):
            if not check_internal_include(path, path.parts[0], header):
                failures.append(f"checker self-test failed to keep prescribed wall motion out of {path}")
    cell_grid_header = "methods/sph/neighbors/CellGrid.hpp"
    if check_internal_include(SPH_CELL_GRID_TEST, "tests", cell_grid_header):
        failures.append("checker self-test failed to allow the SPH cell-grid test to include internals")
    if not check_internal_include(package_consumer, "tests", cell_grid_header):
        failures.append("checker self-test failed to reject private SPH grid headers in the installed consumer")
    if not check_internal_include(REFERENCE_COMMAND_SOURCE, "apps", "./methods/sph/Density3D.hpp"):
        failures.append("checker self-test failed to reject an alternate SPH method include in the CLI")
    for header in ("fixed_csr/Density.hpp", "fixed_csr/Participant.hpp", "fixed_csr/sph_fixture.hpp"):
        for path in (FIXED_CSR_PARTICIPANT, *FIXED_CSR_CONSUMERS.values()):
            if check_internal_include(path, path.parts[0], header):
                failures.append(f"checker self-test failed to allow {header} in {path}")
        for path in (sph_method, sph_participant, SPH_CELL_GRID_TEST, REFERENCE_COMMAND_SOURCE, package_consumer):
            if not check_internal_include(path, path.parts[0], header):
                failures.append(f"checker self-test failed to keep {header} out of {path}")
    return failures


def check_graph(path: Path) -> list[str]:
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read CMake target graph {path}: {exc}"]

    if not isinstance(graph, dict) or not isinstance(graph.get("targets"), list):
        return ["target graph must contain a 'targets' array"]
    if not isinstance(graph.get("testing_enabled"), bool):
        return ["target graph must contain a boolean 'testing_enabled' setting"]
    errors: list[str] = []
    seen: set[str] = set()
    for target in graph["targets"]:
        if not isinstance(target, dict) or not isinstance(target.get("name"), str):
            errors.append("invalid target entry; each target needs a string name")
            continue
        name = target["name"]
        source_dir = target.get("source_dir", "")
        is_test = isinstance(source_dir, str) and source_dir.startswith("tests/")
        if name not in ALLOWED_LINKS and not is_test:
            errors.append(f"unclassified in-tree target '{name}'")
            continue
        if name in SPECIAL_TEST_LINKS and not graph["testing_enabled"]:
            errors.append(f"{name}: test target exists with testing disabled")
        if name == KOKKOS_TEST_MAIN and source_dir != "tests/kokkos":
            errors.append(f"{name}: unexpected source directory {source_dir!r}")
        if name in SPH_TARGET_DIRS and source_dir != SPH_TARGET_DIRS[name].as_posix():
            errors.append(f"{name}: unexpected source directory {source_dir!r}")
        seen.add(name)
        links = target.get("links")
        if not isinstance(links, list) or any(not isinstance(link, str) for link in links):
            errors.append(f"{name}: links must be an array of strings")
            continue
        errors.extend(check_link_policy(name, links))

    required_targets = set(ALLOWED_LINKS)
    if graph["testing_enabled"]:
        required_targets.update(SPECIAL_TEST_LINKS)
    errors.extend(f"target graph is missing '{name}'" for name in sorted(required_targets - seen))
    return errors


def check_includes() -> list[str]:
    errors: list[str] = []
    for scope in ("include/nexora", "src", "apps", "tests", "benchmarks", "validation"):
        for path in sorted((ROOT / scope).rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            below_scope = path.relative_to(ROOT / scope)
            module = public_module(below_scope.parts) if scope == "include/nexora" else below_scope.parts[0]
            if relative.parent == FIXED_CSR_DIR:
                allowed = SOURCE_MODULE_INCLUDES["participants"]
            elif scope in {"apps", "tests", "benchmarks", "validation"}:
                allowed = set(ALLOWED_INCLUDES)
            elif relative.parts[:3] == SPH_METHOD_DIR:
                allowed = ALLOWED_INCLUDES["core"]
            elif module in SOURCE_MODULE_INCLUDES:
                allowed = SOURCE_MODULE_INCLUDES[module]
            else:
                errors.append(f"{relative}: unclassified source module '{module}'")
                continue

            cli_source = scope == "apps" and module == "nexora"
            if path.suffix in {".cu", ".cuh"} and not cli_source:
                errors.append(f"{relative}: CUDA sources are limited to the opt-in CLI smoke target")

            errors.extend(check_source_includes(relative, scope, module, allowed,
                                                path.read_text(encoding="utf-8")))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="CMake-emitted target graph JSON")
    args = parser.parse_args()
    graph = ROOT / args.graph
    errors = policy_self_test() + check_graph(graph) + check_includes()
    if errors:
        print("Nexora dependency policy violations:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Dependency policy passed: {graph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
