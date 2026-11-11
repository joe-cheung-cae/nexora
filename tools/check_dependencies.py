#!/usr/bin/env python3
"""Check direct CMake target links and C/C++ include boundaries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETIRED_TARGETS = {"nexora_exec_api", "nexora_exec_cpu", "nexora_exec_tests",
                   "nexora::exec_api", "nexora::exec_cpu"}
RETIRED_PUBLIC_HEADERS = {"core/ArrayView.hpp", "core/math/Vec.hpp", "core/math/Tensor.hpp",
                          "data/FieldView.hpp"}

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
    "nexora": {"nexora_core", "Kokkos::kokkos", "MPI::MPI_C"},
}
TEST_LINKS = set(ALLOWED_LINKS) - {"nexora", "nexora_build_options"}
REQUIRED_LINKS = {
    "nexora_data_schema": {"nexora_core"},
    "nexora_data": {"nexora_data_schema", "Kokkos::kokkos"},
    "nexora_participant_api": {"nexora_data_schema"},
    "nexora_methods_reference": {"nexora_data", "Kokkos::kokkos"},
    "nexora_participants_reference": {
        "nexora_participant_api", "nexora_methods_reference", "nexora_data"},
    "nexora": {"Kokkos::kokkos"},
    "nexora_transport_tests": {"nexora_data"},
    "nexora_transport_receive_tests": {"nexora_data"},
    "nexora_reference_transport_tests": {"nexora_participants_reference", "nexora_simulation"},
}

SPECIAL_TEST_LINKS = {
    "nexora_field_binding_tests": {"nexora_data"},
    "nexora_kokkos_tests": {"nexora_data"},
    "nexora_reference_tests": {"nexora_participants_reference", "nexora_simulation"},
    "nexora_reference_transport_tests": {"nexora_participants_reference", "nexora_simulation"},
    "nexora_transport_tests": {"nexora_data"},
    "nexora_transport_receive_tests": {"nexora_data"},
}

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
BINDING_TEST = Path("tests/data/field_binding_tests.cpp")
TRANSPORT_TEST = Path("tests/data/transport_tests.cpp")
RECEIVE_TEST = Path("tests/data/transport_receive_tests.cpp")
REFERENCE_TRANSPORT_TEST = Path("tests/reference/reference_transport_tests.cpp")
PRIVATE_DATA_HEADERS = {
    "FieldIssuer.hpp": {BINDING_TEST, TRANSPORT_TEST, RECEIVE_TEST, REFERENCE_TRANSPORT_TEST},
    "ExternalField.hpp": {TRANSPORT_TEST},
    "TransportBuilder.hpp": {TRANSPORT_TEST, RECEIVE_TEST, REFERENCE_TRANSPORT_TEST},
    "TransportReceiver.hpp": {RECEIVE_TEST},
}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
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
    """Keep Kokkos headers at approved data, method, test, and CLI boundaries."""
    header_name = header.replace("\\", "/").split("/")[-1]
    if not KOKKOS_HEADER.fullmatch(header_name):
        return []
    if ((scope == "include/nexora" and module == "data")
            or (scope == "src" and module == "data")
            or (scope == "src" and relative.parts[:3] == ("src", "methods", "reference"))
            or (scope == "tests" and relative.parts[:2] in {
                ("tests", "kokkos"), ("tests", "reference")})
            or relative in {BINDING_TEST, TRANSPORT_TEST, RECEIVE_TEST}
            or (scope == "apps" and module == "nexora")):
        return []
    return [f"{relative}: Kokkos header {header} is outside an approved data/method/test/CLI boundary"]


def check_internal_include(relative: Path, scope: str, header: str) -> list[str]:
    """Enforce downward use of implementation-only headers."""
    parts = header.replace("\\", "/").split("/")
    location = relative.parts
    if parts[-1] in PRIVATE_DATA_HEADERS:
        if (location[:2] in {("src", "data"), ("src", "participants")}
                or relative in PRIVATE_DATA_HEADERS[parts[-1]]):
            return []
        return [f"{relative}: {header} is private to data and concrete participants"]

    if parts[0] not in {"methods", "participants"}:
        return []
    if len(parts) < 3 or parts[1] != "reference":
        return [f"{relative}: internal reference include {header} is not a known module path"]

    in_method = scope == "src" and location[:3] == ("src", "methods", "reference")
    in_participant = scope == "src" and location[:3] == ("src", "participants", "reference")
    in_reference_test = scope == "tests" and location[:2] == ("tests", "reference")
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


def check_public_include(relative: Path, scope: str, allowed: set[str], header: str) -> list[str]:
    parts = tuple(header.replace("\\", "/").split("/"))
    if ".." in parts:
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
        ("nexora_transport_tests", ["nexora_data"]),
        ("nexora_transport_receive_tests", ["nexora_data"]),
        ("nexora_reference_transport_tests", ["nexora_participants_reference", "nexora_simulation"]),
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
    if check_link_policy("nexora_methods_reference", ["nexora_data", "Kokkos::kokkos"]):
        failures.append("checker self-test failed to allow reference-method dependencies")
    if not check_link_policy("nexora_methods_reference", ["nexora_data", "Kokkos::kokkos", "nexora_participant_api"]):
        failures.append("checker self-test failed to reject an upward method dependency")
    if not check_link_policy("nexora_participants_reference", [
            "nexora_participant_api", "nexora_methods_reference", "nexora_data", "Kokkos::kokkos"]):
        failures.append("checker self-test failed to reject a direct participant Kokkos dependency")
    if not check_link_policy("nexora_reference_transport_tests", [
            "nexora_participants_reference", "nexora_simulation", "nexora_data"]):
        failures.append("checker self-test failed to reject extra reference transport test dependencies")
    if check_link_policy("nexora_kokkos_tests", ["nexora_data"]):
        failures.append("checker self-test failed to allow the dedicated test's data dependency")
    if not check_link_policy("nexora_kokkos_tests", ["nexora_data", "nexora_core"]):
        failures.append("checker self-test failed to reject extra dependencies from the dedicated test")
    for target in ("nexora_transport_tests", "nexora_transport_receive_tests"):
        if not check_link_policy(target, ["nexora_data", "nexora_core"]):
            failures.append(f"checker self-test failed to reject extra dependencies for {target}")
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
    if not check_internal_include(TRANSPORT_TEST, "tests", "data/TransportReceiver.hpp"):
        failures.append("checker self-test failed to reject receiver outside dedicated receive tests")
    for header in ("ExternalField.hpp", "TransportReceiver.hpp"):
        if not check_internal_include(REFERENCE_TRANSPORT_TEST, "tests", f"data/{header}"):
            failures.append(f"checker self-test failed to reject {header} in reference transport tests")
    for header in ("FieldIssuer.hpp", "TransportBuilder.hpp"):
        if not check_internal_include(Path("tests/reference/reference_tests.cpp"),
                                      "tests", f"data/{header}"):
            failures.append(f"checker self-test failed to reject {header} in other reference tests")
    for header, test_paths in PRIVATE_DATA_HEADERS.items():
        for test_path in test_paths:
            if check_internal_include(test_path, "tests", f"data/{header}"):
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
    if not check_kokkos_header(Path("include/nexora/core/Status.hpp"),
                               "include/nexora", "core", "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to reject Kokkos in core headers")
    if check_kokkos_header(Path("src/methods/reference/State.hpp"),
                           "src", "methods", "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to allow Kokkos in reference methods")
    if check_internal_include(Path("src/participants/reference/Participant.hpp"),
                              "src", "methods/reference/State.hpp"):
        failures.append("checker self-test failed to allow participant use of method internals")
    if not check_internal_include(Path("src/methods/reference/State.hpp"),
                                  "src", "participants/reference/Participant.hpp"):
        failures.append("checker self-test failed to reject method use of participant internals")
    if not check_internal_include(Path("src/core/Time.hpp"),
                                  "src", "methods/reference/State.hpp"):
        failures.append("checker self-test failed to reject core use of method internals")
    if check_internal_include(Path("tests/reference/reference_tests.cpp"),
                              "tests", "methods/reference/State.hpp"):
        failures.append("checker self-test failed to allow reference tests to include internals")
    return failures


def check_graph(path: Path) -> list[str]:
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read CMake target graph {path}: {exc}"]

    if not isinstance(graph, dict) or not isinstance(graph.get("targets"), list):
        return ["target graph must contain a 'targets' array"]
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
        seen.add(name)
        links = target.get("links")
        if not isinstance(links, list) or any(not isinstance(link, str) for link in links):
            errors.append(f"{name}: links must be an array of strings")
            continue
        errors.extend(check_link_policy(name, links))

    errors.extend(f"target graph is missing '{name}'" for name in sorted(ALLOWED_LINKS.keys() - seen))
    return errors


def check_includes() -> list[str]:
    errors: list[str] = []
    for scope in ("include/nexora", "src", "apps", "tests"):
        for path in sorted((ROOT / scope).rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            below_scope = path.relative_to(ROOT / scope)
            module = public_module(below_scope.parts) if scope == "include/nexora" else below_scope.parts[0]
            if scope in {"apps", "tests"}:
                allowed = set(ALLOWED_INCLUDES)
            elif module in SOURCE_MODULE_INCLUDES:
                allowed = SOURCE_MODULE_INCLUDES[module]
            else:
                errors.append(f"{relative}: unclassified source module '{module}'")
                continue

            cli_source = scope == "apps" and module == "nexora"
            if path.suffix in {".cu", ".cuh"} and not cli_source:
                errors.append(f"{relative}: CUDA sources are limited to the opt-in CLI smoke target")

            for header in INCLUDE.findall(path.read_text(encoding="utf-8")):
                errors.extend(check_kokkos_header(relative, scope, module, header))
                errors.extend(check_internal_include(relative, scope, header))
                errors.extend(check_public_include(relative, scope, allowed, header))
                header_parts = header.replace("\\", "/").split("/")
                if not cli_source and PLATFORM_HEADER.fullmatch(header_parts[-1]):
                    errors.append(f"{relative}: CUDA/MPI header {header} belongs in the optional CLI")
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
