#!/usr/bin/env python3
"""Check direct CMake target links and C/C++ include boundaries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LINKS = {
    "nexora_build_options": set(),
    "nexora_core": {"nexora_build_options"},
    "nexora_exec_api": {"nexora_core"},
    "nexora_exec_cpu": {"nexora_exec_api"},
    "nexora_data": {"nexora_core", "nexora_exec_api", "Kokkos::kokkos"},
    "nexora_participant_api": {"nexora_core", "nexora_exec_api", "nexora_data"},
    "nexora_simulation": {"nexora_core", "nexora_participant_api"},
    "nexora_methods_reference": {"nexora_data", "Kokkos::kokkos"},
    "nexora_participants_reference": {"nexora_participant_api", "nexora_methods_reference"},
    "nexora": {"nexora_core", "Kokkos::kokkos", "MPI::MPI_C"},
}
TEST_LINKS = set(ALLOWED_LINKS) - {"nexora", "nexora_build_options"}
REQUIRED_LINKS = {
    "nexora_data": {"Kokkos::kokkos"},
    "nexora_methods_reference": {"nexora_data", "Kokkos::kokkos"},
    "nexora_participants_reference": {"nexora_participant_api", "nexora_methods_reference"},
    "nexora": {"Kokkos::kokkos"},
}

SPECIAL_TEST_LINKS = {
    "nexora_kokkos_tests": {"nexora_data"},
    "nexora_reference_tests": {"nexora_participants_reference", "nexora_simulation"},
}

ALLOWED_INCLUDES = {
    "core": {"core"},
    "exec": {"core", "exec"},
    "data": {"core", "exec", "data"},
    "participant": {"core", "exec", "data", "participant"},
    "simulation": {"core", "exec", "data", "participant", "simulation"},
}
SOURCE_MODULE_INCLUDES = {
    **ALLOWED_INCLUDES,
    "methods": {"core", "data"},
    "participants": {"core", "data", "participant"},
}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
PLATFORM_HEADER = re.compile(
    r"(?:mpi\.h|cuda[^/]*\.h|device_launch_parameters\.h|vector_types\.h)", re.IGNORECASE
)
KOKKOS_HEADER = re.compile(r"Kokkos(?:_[A-Za-z0-9_]+)?\.(?:h|hpp)", re.IGNORECASE)


def check_link_policy(name: str, links: list[str]) -> list[str]:
    """Check a target's direct dependencies against the narrow target policy."""
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
            or (scope == "src" and relative.parts[:3] == ("src", "methods", "reference"))
            or (scope == "tests" and relative.parts[:2] in {
                ("tests", "kokkos"), ("tests", "reference")})
            or (scope == "apps" and module == "nexora")):
        return []
    return [f"{relative}: Kokkos header {header} is outside an approved data/method/test/CLI boundary"]


def check_internal_include(relative: Path, scope: str, header: str) -> list[str]:
    """Enforce downward use of implementation-only reference headers."""
    parts = header.replace("\\", "/").split("/")
    if not parts or parts[0] not in {"methods", "participants"}:
        return []
    if len(parts) < 3 or parts[1] != "reference":
        return [f"{relative}: internal reference include {header} is not a known module path"]

    location = relative.parts
    in_method = scope == "src" and location[:3] == ("src", "methods", "reference")
    in_participant = scope == "src" and location[:3] == ("src", "participants", "reference")
    in_reference_test = scope == "tests" and location[:2] == ("tests", "reference")
    if parts[0] == "methods" and (in_method or in_participant or in_reference_test):
        return []
    if parts[0] == "participants" and (in_participant or in_reference_test):
        return []
    return [f"{relative}: forbidden internal include {header} from this module"]


def policy_self_test() -> list[str]:
    """Exercise the newly introduced Kokkos boundary with allowed and denied cases."""
    failures: list[str] = []
    if check_link_policy("nexora_data", ["Kokkos::kokkos"]):
        failures.append("checker self-test failed to allow nexora_data's Kokkos dependency")
    if not check_link_policy("nexora_data", []):
        failures.append("checker self-test failed to require Kokkos on nexora_data")
    if not check_link_policy("nexora_core", ["Kokkos::kokkos"]):
        failures.append("checker self-test failed to reject Kokkos from nexora_core")
    if check_link_policy("nexora_methods_reference", ["nexora_data", "Kokkos::kokkos"]):
        failures.append("checker self-test failed to allow reference-method dependencies")
    if not check_link_policy("nexora_methods_reference", ["nexora_data", "Kokkos::kokkos", "nexora_participant_api"]):
        failures.append("checker self-test failed to reject an upward method dependency")
    if check_link_policy("nexora_participants_reference", ["nexora_participant_api", "nexora_methods_reference"]):
        failures.append("checker self-test failed to allow reference-participant dependencies")
    if not check_link_policy("nexora_participants_reference", ["nexora_participant_api", "nexora_methods_reference", "Kokkos::kokkos"]):
        failures.append("checker self-test failed to reject a direct participant Kokkos dependency")
    if check_link_policy("nexora_kokkos_tests", ["nexora_data"]):
        failures.append("checker self-test failed to allow the dedicated test's data dependency")
    if not check_link_policy("nexora_kokkos_tests", ["nexora_data", "nexora_core"]):
        failures.append("checker self-test failed to reject extra dependencies from the dedicated test")
    if check_kokkos_header(Path("include/nexora/data/FieldView.hpp"),
                           "include/nexora", "data", "Kokkos_Core.hpp"):
        failures.append("checker self-test failed to allow Kokkos in data headers")
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
            module = below_scope.parts[0]
            if scope in {"apps", "tests"}:
                allowed = ALLOWED_INCLUDES.keys()
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
                header_parts = header.replace("\\", "/").split("/")
                if ".." in header_parts:
                    errors.append(f"{relative}: relative traversal include is forbidden")
                if len(header_parts) > 1 and header_parts[0] == "nexora":
                    if header_parts[1] not in allowed:
                        errors.append(f"{relative}: forbidden include {header}")
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
