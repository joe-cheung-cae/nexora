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
    "nexora_data": {"nexora_core", "nexora_exec_api"},
    "nexora_participant_api": {"nexora_core", "nexora_exec_api", "nexora_data"},
    "nexora_simulation": {"nexora_core", "nexora_participant_api"},
    "nexora": {"nexora_core", "CUDA::cudart", "MPI::MPI_C"},
}
TEST_LINKS = set(ALLOWED_LINKS) - {"nexora", "nexora_build_options"}

ALLOWED_INCLUDES = {
    "core": {"core"},
    "exec": {"core", "exec"},
    "data": {"core", "exec", "data"},
    "participant": {"core", "exec", "data", "participant"},
    "simulation": {"core", "exec", "data", "participant", "simulation"},
}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
PLATFORM_HEADER = re.compile(
    r"(?:mpi\.h|cuda[^/]*\.h|device_launch_parameters\.h|vector_types\.h)", re.IGNORECASE
)


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
        if name in ALLOWED_LINKS:
            allowed = ALLOWED_LINKS[name]
        elif isinstance(source_dir, str) and source_dir.startswith("tests/"):
            allowed = TEST_LINKS
        else:
            errors.append(f"unclassified in-tree target '{name}'")
            continue
        seen.add(name)
        links = target.get("links")
        if not isinstance(links, list) or any(not isinstance(link, str) for link in links):
            errors.append(f"{name}: links must be an array of strings")
            continue
        errors.extend(f"{name}: forbidden direct dependency on {link}" for link in links
                      if link not in allowed)

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
            elif module in ALLOWED_INCLUDES:
                allowed = ALLOWED_INCLUDES[module]
            else:
                errors.append(f"{relative}: unclassified source module '{module}'")
                continue

            cli_source = scope == "apps" and module == "nexora"
            if path.suffix in {".cu", ".cuh"} and not cli_source:
                errors.append(f"{relative}: CUDA sources are limited to the opt-in CLI smoke target")

            for header in INCLUDE.findall(path.read_text(encoding="utf-8")):
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
    errors = check_graph(graph) + check_includes()
    if errors:
        print("Nexora dependency policy violations:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Dependency policy passed: {graph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
