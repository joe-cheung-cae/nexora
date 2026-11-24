#!/usr/bin/env python3
"""Provision and validate the locked Nexora Serial build with local inputs."""

import argparse
import errno
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "5.2.2"
REVISION = "8e15454876298832f013ebcaf08d53825b5bda40"
ARCHIVE = f"kokkos-{VERSION}.tar.gz"
CHECKSUMS = f"kokkos-{VERSION}-SHA-256.txt"
SHA256 = "d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f"
RELEASE = f"https://github.com/kokkos/kokkos/releases/download/{VERSION}/"
PACKAGE_DIR = Path("lib/cmake/Kokkos")
MANIFEST = "nexora-provision.json"

COMMON_OPTIONS = [
    "-G", "Ninja", "-DCMAKE_CXX_STANDARD=20",
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON", "-DCMAKE_CXX_EXTENSIONS=OFF",
    "-DCMAKE_INSTALL_LIBDIR=lib", "-DBUILD_SHARED_LIBS=OFF",
]
KOKKOS_OPTIONS = [
    "-DKokkos_ENABLE_SERIAL=ON",
    *[f"-DKokkos_ENABLE_{option}=OFF" for option in (
        "OPENMP", "THREADS", "HPX", "OPENACC", "CUDA", "HIP", "SYCL", "NEXTSILICON",
        "TESTS", "EXAMPLES", "BENCHMARKS",
    )],
]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as contents:
        for block in iter(lambda: contents.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(args, log):
    args = [str(arg) for arg in args]
    quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
    with log.open("a", encoding="utf-8") as output:
        header = f"[{now()}] {quote(args)}"
        print(header, flush=True)
        print(header, file=output, flush=True)
        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", file=output)
                print(line, end="")
        status = f"[{now()}] exit {process.returncode}"
        print(status, file=output)
        print(status, flush=True)
    if process.returncode:
        raise RuntimeError(f"command failed with exit {process.returncode}; see {log}")


def executable(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required executable not found: {name}")
    # Keep clang++'s driver name when it is a symlink to clang.
    return Path(os.path.abspath(path))


def identity(args):
    toolchain = {}
    for role, name in (("cc", args.cc), ("cxx", args.cxx)):
        path = executable(name)
        # cl.exe does not implement --version. Its exact build ID is also
        # recorded in the generated Kokkos config, using CMake's compiler probe.
        probe = [] if path.name.lower() == "cl.exe" else ["--version"]
        result = subprocess.run([path, *probe], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        toolchain[role] = {"path": path.as_posix(), "sha256": digest(path),
                           "version_output": result.stdout.strip()}
    for name in ("cmake", "ninja"):
        toolchain[name] = subprocess.check_output(
            [name, "--version"], text=True, encoding="utf-8",
        ).strip()
    return {
        "system": platform.system(),
        "architecture": platform.machine(),
        "build_type": args.config,
        "toolchain": toolchain,
        "toolchain_environment": {
            name: os.environ[name] for name in (
                "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "SDKROOT",
                "MACOSX_DEPLOYMENT_TARGET", "VCToolsVersion", "WindowsSDKVersion",
                "CL", "_CL_", "INCLUDE", "LIB", "LIBPATH",
            ) if name in os.environ
        },
        "options": COMMON_OPTIONS + KOKKOS_OPTIONS,
        "source": {"version": VERSION, "commit": REVISION, "archive": RELEASE + ARCHIVE,
                   "sha256": SHA256, "license": "Apache-2.0 WITH LLVM-exception",
                   "patches": []},
    }


def toolchain_options(profile):
    return COMMON_OPTIONS + [
        f"-DCMAKE_BUILD_TYPE={profile['build_type']}",
        f"-DCMAKE_C_COMPILER={profile['toolchain']['cc']['path']}",
        f"-DCMAKE_CXX_COMPILER={profile['toolchain']['cxx']['path']}",
    ]


def installed_files(prefix):
    libs = [path for path in (prefix / "lib").iterdir()
            if path.is_file() and path.suffix in (".a", ".lib")]
    if not any(path.stem in ("libkokkoscore", "kokkoscore") for path in libs):
        raise RuntimeError(f"missing Kokkos core static library in {prefix}")
    config = prefix / PACKAGE_DIR
    required = {"KokkosConfig.cmake", "KokkosConfigVersion.cmake", "KokkosConfigCommon.cmake"}
    if not all((config / name).is_file() for name in required):
        raise RuntimeError(f"incomplete Kokkos package config in {config}")
    exported = (config / "KokkosConfigCommon.cmake").read_text(encoding="utf-8").splitlines()
    if "set(Kokkos_DEVICES SERIAL)" not in exported or "set(Kokkos_CXX_STANDARD 20)" not in exported:
        raise RuntimeError(f"expected exported Kokkos Serial-only C++20 metadata in {config}")
    if "set(Kokkos_COMPILE_LANGUAGE CXX)" not in (
        config / "KokkosConfig.cmake"
    ).read_text(encoding="utf-8").splitlines():
        raise RuntimeError(f"expected exported Kokkos CXX compile language in {config}")
    return {
        path.relative_to(prefix).as_posix(): digest(path)
        for path in sorted(prefix.rglob("*")) if path.is_file() and path != prefix / MANIFEST
    }


def verify_checksums(path):
    entries = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    if [SHA256, ARCHIVE] not in entries:
        raise RuntimeError(f"Kokkos release checksum manifest disagrees with the lock: {path}")
    return digest(path)


def verify_kokkos(prefix, profile, logs):
    manifest = json.loads((prefix / MANIFEST).read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in profile.items()):
        raise RuntimeError(f"Kokkos installation does not match this runner profile: {prefix}")
    if manifest.get("installed_files") != installed_files(prefix):
        raise RuntimeError(f"Kokkos installation digests differ from the manifest: {prefix}")
    write_json(logs / MANIFEST, manifest)


def provision(args, profile):
    archive = args.archive.absolute()
    checksums = archive.with_name(CHECKSUMS)
    prefix = args.prefix.absolute()
    log = args.logs.absolute() / f"provision-{args.config.lower()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if prefix.exists():
        verify_kokkos(prefix, profile, log.parent)
        print(f"[{now()}] Reusing verified Kokkos installation: {prefix}", flush=True)
        return
    if args.download:
        archive.parent.mkdir(parents=True, exist_ok=True)
        for path, artifact in ((archive, ARCHIVE), (checksums, CHECKSUMS)):
            if not path.is_file():
                print(f"[{now()}] Fetching {RELEASE + artifact}", flush=True)
                with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
                    download = Path(scratch) / artifact
                    with urllib.request.urlopen(RELEASE + artifact, timeout=60) as response:
                        with download.open("wb") as target:
                            shutil.copyfileobj(response, target)
                    if artifact == ARCHIVE and digest(download) != SHA256:
                        raise RuntimeError("downloaded Kokkos archive disagrees with the repository lock")
                    if artifact == CHECKSUMS:
                        verify_checksums(download)
                    download.replace(path)
    if digest(archive) != SHA256:
        raise RuntimeError(f"Kokkos archive SHA-256 differs from the repository lock: {archive}")
    if checksums.is_file():
        profile["release_checksums_sha256"] = verify_checksums(checksums)
    profile["archive_verified_at"] = now()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kokkos-build-", dir=prefix.parent) as directory:
        scratch = Path(directory)
        unpacked = scratch / "source"
        unpacked.mkdir()
        shutil.unpack_archive(archive, unpacked)
        source = unpacked / f"kokkos-{VERSION}"
        build = scratch / "build"
        command(["cmake", "-S", source, "-B", build, *toolchain_options(profile),
                 *KOKKOS_OPTIONS, f"-DCMAKE_INSTALL_PREFIX={prefix.as_posix()}"], log)
        command(["cmake", "--build", build, "--parallel", args.jobs], log)
        staged = scratch / "install"
        command(["cmake", "--install", build, "--prefix", staged], log)
        notices = staged / "share/nexora-kokkos-provenance"
        notices.mkdir(parents=True)
        for name in ("LICENSE", "COPYRIGHT.md"):
            shutil.copyfile(source / name, notices / name)
        if checksums.is_file():
            shutil.copyfile(checksums, notices / checksums.name)
        profile["installed_files"] = installed_files(staged)
        profile["provisioned_at"] = now()
        write_json(staged / MANIFEST, profile)
        staged.rename(prefix)


GTEST_VERSION = "1.17.0"
GTEST_REVISION = "52eb8108c5bdec04579160ae17225d66034bd723"
GTEST_URL = f"https://codeload.github.com/google/googletest/tar.gz/refs/tags/v{GTEST_VERSION}"
GTEST_SHA256 = "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c"
GTEST_PACKAGE_DIR = Path("lib/cmake/GTest")
GTEST_MANIFEST = "nexora-gtest-provision.json"
GTEST_OPTIONS = ["-DBUILD_GMOCK=OFF", "-DINSTALL_GTEST=ON", "-Dgtest_build_tests=OFF",
           "-Dgtest_build_samples=OFF", "-Dgtest_force_shared_crt=ON"]


def gtest_identity(profile):
    return {key: profile[key] for key in (
        "system", "architecture", "build_type", "toolchain", "toolchain_environment",
    )} | {"options": toolchain_options(profile) + GTEST_OPTIONS,
         "source": {"version": GTEST_VERSION, "commit": GTEST_REVISION, "archive": GTEST_URL,
                    "sha256": GTEST_SHA256, "license": "BSD-3-Clause", "patches": []}}


def gtest_installed_files(prefix):
    config = prefix / GTEST_PACKAGE_DIR
    for name in ("GTestConfig.cmake", "GTestConfigVersion.cmake", "GTestTargets.cmake"):
        if not (config / name).is_file():
            raise RuntimeError(f"incomplete GoogleTest package: {config / name}")
    if not (prefix / "include/gtest/gtest.h").is_file():
        raise RuntimeError(f"missing GoogleTest headers: {prefix}")
    libraries = {path.stem for path in (prefix / "lib").iterdir()
                 if path.suffix in (".a", ".lib")}
    if not libraries.intersection(("libgtest", "gtest", "libgtestd", "gtestd")):
        raise RuntimeError(f"missing GoogleTest static library: {prefix}")
    return {path.relative_to(prefix).as_posix(): digest(path)
            for path in sorted(prefix.rglob("*")) if path.is_file() and path != prefix / GTEST_MANIFEST}


def provision_gtest(args, profile):
    archive, prefix = args.gtest_archive.absolute(), args.gtest_prefix.absolute()
    log = args.logs.absolute() / f"provision-gtest-{args.config.lower()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if prefix.exists():
        verify_gtest(prefix, profile, log.parent)
        print(f"[{now()}] Reusing verified GoogleTest installation: {prefix}", flush=True)
        return
    if args.download and not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=archive.parent) as directory:
            downloaded = Path(directory) / "googletest.tar.gz"
            with urllib.request.urlopen(GTEST_URL, timeout=60) as response, downloaded.open("wb") as target:
                shutil.copyfileobj(response, target)
            if digest(downloaded) != GTEST_SHA256:
                raise RuntimeError("downloaded GoogleTest archive disagrees with repository lock")
            downloaded.replace(archive)
    if digest(archive) != GTEST_SHA256:
        raise RuntimeError(f"GoogleTest archive SHA-256 differs from repository lock: {archive}")
    record = gtest_identity(profile)
    record["archive_verified_at"] = now()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gtest-build-", dir=prefix.parent) as directory:
        scratch = Path(directory)
        shutil.unpack_archive(archive, scratch)
        source, build = scratch / f"googletest-{GTEST_VERSION}", scratch / "build"
        command(["cmake", "-S", source, "-B", build, *toolchain_options(profile), *GTEST_OPTIONS,
                 f"-DCMAKE_INSTALL_PREFIX={prefix.as_posix()}"], log)
        command(["cmake", "--build", build, "--parallel", args.jobs], log)
        staged = scratch / "install"
        command(["cmake", "--install", build, "--prefix", staged], log)
        notices = staged / "share/nexora-gtest-provenance"
        notices.mkdir(parents=True)
        shutil.copyfile(source / "LICENSE", notices / "LICENSE")
        record["installed_files"] = gtest_installed_files(staged)
        record["provisioned_at"] = now()
        write_json(staged / GTEST_MANIFEST, record)
        staged.rename(prefix)


def verify_gtest(prefix, profile, logs):
    prefix = prefix.absolute()
    record = json.loads((prefix / GTEST_MANIFEST).read_text(encoding="utf-8"))
    if any(record.get(key) != value for key, value in gtest_identity(profile).items()):
        raise RuntimeError(f"GoogleTest installation does not match runner profile: {prefix}")
    if record.get("installed_files") != gtest_installed_files(prefix):
        raise RuntimeError(f"GoogleTest installation digests differ from manifest: {prefix}")
    write_json(logs / GTEST_MANIFEST, record)


def verify_network():
    allowed_errors = {errno.EACCES, errno.EPERM}
    interfaces = [name for _, name in socket.if_nameindex()]
    if sys.platform == "linux" and interfaces == ["lo"]:
        allowed_errors.update((errno.ENETUNREACH, errno.EADDRNOTAVAIL))
    results = {"interfaces": interfaces, "checks": []}
    if sys.platform == "linux":
        results["network_namespace"] = os.readlink("/proc/self/ns/net")
    for family, remote, loopback in (
        (socket.AF_INET, "192.0.2.1", "127.0.0.1"),
        (socket.AF_INET6, "2001:db8::1", "::1"),
    ):
        for protocol, host in (
            (socket.SOCK_STREAM, remote), (socket.SOCK_STREAM, loopback),
            (socket.SOCK_DGRAM, remote), (socket.SOCK_DGRAM, loopback),
        ):
            try:
                with socket.socket(family, protocol) as connection:
                    connection.settimeout(1)
                    if protocol == socket.SOCK_STREAM:
                        connection.connect((host, 9))
                    else:
                        connection.sendto(b"Nexora network isolation probe", (host, 9))
            except OSError as error:
                result = error.errno
            else:
                result = 0
            results["checks"].append(
                {"address": host, "protocol": protocol.name, "errno": result},
            )
        try:
            with socket.socket(family, socket.SOCK_STREAM) as connection:
                connection.bind((loopback, 0))
        except OSError as error:
            result = error.errno
        else:
            result = 0
        results[f"bind_{family.name}_errno"] = result
    return results, all(check["errno"] in allowed_errors for check in results["checks"])


def check_ctest(junit, expected_tests):
    cases = ET.parse(junit).getroot().findall(".//testcase")
    missing = set(expected_tests) - {case.get("name") for case in cases}
    if missing:
        raise RuntimeError(f"required CTest cases absent: {sorted(missing)}; see {junit}")
    skipped = [case.get("name") for case in cases
               if case.find("skipped") is not None or case.get("status") == "notrun"]
    if skipped:
        raise RuntimeError(f"CTest cases skipped: {skipped}; see {junit}")


def measure_irregular(build, logs, precision, jobs):
    """Execute and independently account for the deterministic Serial fixture."""
    target = "nexora_serial_irregular_measurement"
    executable_path = build / "tests" / "data" / target
    if os.name == "nt":
        executable_path = executable_path.with_suffix(".exe")
    source_paths = [
        ROOT / "tests/data/serial_irregular_measurement.cpp",
        ROOT / "tests/data/structure_fixture.hpp",
        ROOT / "src/participants/reference/StructureState.cpp",
        ROOT / "src/participants/reference/StructureState.hpp",
        ROOT / "src/data/EntityOrder.cpp",
        ROOT / "src/data/EntityOrder.hpp",
        ROOT / "src/data/Permutation.hpp",
        ROOT / "src/data/CsrAdjacency.hpp",
    ]
    record_path = logs / "irregular-measurement.json"
    raw_path = logs / "irregular-measurement.stdout.json"
    repeat_path = logs / "irregular-measurement-repeat.stdout.json"
    log_path = logs / "irregular-measurement.log"
    record = {
        "status": "failed",
        "fixture": "structure_reorder_irregular",
        "precision": precision,
        "started": now(),
        "target": target,
        "source_sha256": {},
    }
    try:
        record["source_sha256"] = {
            path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths
        }
        command(["cmake", "--build", build, "--target", target,
                 "--parallel", str(jobs)], logs / "irregular-measurement-build.log")
        if not executable_path.is_file():
            raise RuntimeError(f"missing measurement executable: {executable_path}")
        record["executable"] = executable_path.relative_to(build).as_posix()
        record["executable_sha256"] = digest(executable_path)
        args = [executable_path.as_posix()]
        quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
        runs = []
        execution_started = now()
        for run_number, path in enumerate((raw_path, repeat_path), start=1):
            started = now()
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            output = result.stdout
            path.write_text(output, encoding="utf-8")
            output_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
            runs.append({"run": run_number, "started": started, "finished": now(),
                         "exit_code": result.returncode, "stdout_sha256": output_hash,
                         "stdout": output})
        with log_path.open("w", encoding="utf-8") as log:
            print(f"[{started}] {quote(args)}", file=log)
            for run in runs:
                print(f"run {run['run']} [{run['started']}]", file=log)
                log.write(run["stdout"])
                if run["stdout"] and not run["stdout"].endswith("\n"):
                    log.write("\n")
                print(f"[{run['finished']}] exit {run['exit_code']}", file=log)
        record["command"] = quote(args)
        record["execution_started"] = execution_started
        record["runs"] = [{key: run[key] for key in (
            "run", "started", "finished", "exit_code", "stdout_sha256",
        )} for run in runs]
        if any(run["exit_code"] for run in runs):
            raise RuntimeError(f"measurement executable returned a non-zero status; see {log_path}")
        if runs[0]["stdout"] != runs[1]["stdout"]:
            raise RuntimeError(f"measurement output differs between repeated executions; see {log_path}")
        output = runs[0]["stdout"]
        record["stdout_sha256"] = runs[0]["stdout_sha256"]
        record["repeat_stdout_sha256"] = runs[1]["stdout_sha256"]
        try:
            result_json = json.loads(output)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"measurement output is not JSON; see {raw_path}") from error
        record["result"] = result_json
        identity = {"schema_version": 1, "fixture": "structure_reorder_irregular",
                    "precision": precision, "kokkos_version": 50202,
                    "numerical_validation": "passed"}
        for key, value in identity.items():
            if result_json.get(key) != value:
                raise RuntimeError(f"measurement {key} mismatch: {result_json.get(key)!r}")
        if result_json.get("resource_budget", {}).get("applicable") is not False:
            raise RuntimeError("measurement must report that StructureState has no ResourceBudget")
        events = result_json.get("events")
        if not isinstance(events, list) or not events:
            raise RuntimeError("measurement must include its ordered event ledger")
        counts = result_json.get("counts", {})
        byte_summary = result_json.get("bytes", {})
        if not isinstance(counts, dict) or not isinstance(byte_summary, dict):
            raise RuntimeError("measurement counts and bytes must be JSON objects")
        kinds = {kind: [event for event in events if event.get("kind") == kind]
                 for kind in ("allocate", "deallocate", "deep_copy", "explicit_fence",
                              "kokkos_internal_fence", "parallel_for")}
        count_keys = {"allocate": "allocations", "deallocate": "deallocations",
                      "deep_copy": "deep_copy_calls", "explicit_fence": "explicit_fences",
                      "kokkos_internal_fence": "kokkos_internal_fences",
                      "parallel_for": "parallel_for_calls"}
        for kind, key in count_keys.items():
            if len(kinds[kind]) != counts.get(key):
                raise RuntimeError(f"measurement {key} disagrees with the event ledger")
        allocated = sum(event["bytes"] for event in kinds["allocate"])
        deallocated = sum(event["bytes"] for event in kinds["deallocate"])
        copied = sum(event["bytes"] for event in kinds["deep_copy"])
        if (allocated != byte_summary.get("allocated_payload") or
                deallocated != byte_summary.get("deallocated_payload") or
                copied != counts.get("deep_copy_bytes")):
            raise RuntimeError("measurement payload byte totals disagree with the event ledger")
        live = packed_live = peak = packed_peak = 0
        peak_phase = None
        for event in events:
            if event["kind"] == "allocate":
                live += event["bytes"]
                if event["label"].startswith("transport."):
                    packed_live += event["bytes"]
                if live > peak:
                    peak, peak_phase = live, event["phase"]
                packed_peak = max(packed_peak, packed_live)
            elif event["kind"] == "deallocate":
                live -= event["bytes"]
                if event["label"].startswith("transport."):
                    packed_live -= event["bytes"]
            if live < 0 or packed_live < 0 or event.get("live_bytes") != live:
                raise RuntimeError("measurement live byte ledger is inconsistent")
        if (live != byte_summary.get("final_live_payload") or live != 0 or
                peak != byte_summary.get("peak_live_payload") or
                peak_phase != byte_summary.get("peak_phase") or
                packed_peak != byte_summary.get("peak_live_transport_payload")):
            raise RuntimeError("measurement live payload summary disagrees with its event ledger")
        if sum(event.get("phase") == "finalize" for event in events) != 1:
            raise RuntimeError("measurement event ledger must include exactly one finalize phase")
        record["status"] = "passed"
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        record["error"] = str(error)
        raise
    finally:
        record["finished"] = now()
        write_json(record_path, record)
    return record


def verify_precision(args, profile, precision):
    name = f"{args.config.lower()}-{precision}"
    build = args.work_dir.absolute() / name
    build.mkdir(parents=True)
    logs = args.logs.absolute() / name
    logs.mkdir(parents=True)
    double = "ON" if precision == "double" else "OFF"
    mpi = "ON" if args.mpi else "OFF"
    command(["cmake", "-S", ROOT, "-B", build, *toolchain_options(profile),
             f"-DKokkos_DIR={(args.prefix.absolute() / PACKAGE_DIR).as_posix()}",
             f"-DGTest_DIR={(args.gtest_prefix.absolute() / GTEST_PACKAGE_DIR).as_posix()}",
             "-DBUILD_TESTING=ON", f"-DNEXORA_ENABLE_DOUBLE_PRECISION={double}",
             f"-DNEXORA_ENABLE_MPI={mpi}"], logs / "configure.log")
    command(["cmake", "--build", build, "--parallel", args.jobs], logs / "build.log")
    command([sys.executable, ROOT / "tools/check_dependencies.py", "--graph",
             build / "nexora-target-graph.json"], logs / "dependencies.log")
    ctest = ["ctest", "--test-dir", build, "--no-tests=error", "--output-on-failure"]
    command([*ctest, "--label-exclude", "^package$", "--output-junit", logs / "ctest.xml"],
            logs / "ctest.log")
    mpi_test = "nexora_mpi_smoke" if args.mpi else "nexora_cli_mpi_disabled"
    check_ctest(logs / "ctest.xml", [
        "nexora_reference_tests.Reference.EulerConvergence",
        "nexora_reference_tests.Reference.NonDyadicStepMatchesEulerOracle",
        "nexora_sph_fixed_csr_tests.Sph.FrozenLayoutAndReorder",
        "nexora_sph_tests.SphCellGrid.ThreeDimensionalShuffledStableIdsAndGhosts",
        "nexora_sph_tests.SphCellGrid.NeighborViewPreservesCallbackOrder",
        "nexora_sph_tests.SphCellGrid.IsolatedMassAndSmoothingLengthSetSelfDensity",
        "nexora_sph_tests.SphCellGrid.PairMirrorsAndRelativeVelocityReversalMatchAnalyticRatesAndForces",
        "nexora_sph_tests.SphCellGrid.AdmissionPrecedesRemapAndFollowsSharedQueryLifetime",
        "nexora_sph_tests.SphCellGrid.RebuildFallbackAndComparisonRespectScratchLifetimes",
        "nexora_sph_tests.SphCellGrid.AssignmentKeepsReplacedArraysChargedUntilDeallocation",
        "nexora_sph_tests.SphCellGrid.CrossStepReuseAndRebuildMatchesExplicitCsr",
        "nexora_sph_tests.SphGeometry3D.VersionsDescribeGeometryLayoutAndTopologySeparately",
        "nexora_sph_tests.SphGeometry3D.ThreeAxisCrossingAndInputReorderCompareGeometryById",
        "nexora_sph_tests.SphWcsphOperators.ChangingOwnedSetPublishesPressureRowsByStableId",
        "nexora_sph_tests.SphWcsphOperators.SourceOutsideOwnedSupportChangesGhostPressureForceOnly",
        "nexora_sph_tests.SphWcsphOperators.LinearEquationOfStateResolvesBothSidesOfRestDensity",
        "nexora_sph_tests.SphWcsphOperators.UnequalMassPairAnalyticDirectionAndForceBalanceIncludingTension",
        "nexora_sph_tests.SphVerletWindow.UniformVelocityCrossesCellsAndStrictCfl",
        "nexora_sph_tests.SphVerletWindow.DriftOverflowRestoresAndSmallerStepSucceeds",
        "nexora_sph_tests.SphVerletOperators.CflReductionIncludesNonfiniteVelocityAmongFiniteRows",
        "nexora_sph_tests.SphVerletWindow.UnequalMassTrajectoryMatchesIndependentCsrAndMomentum",
        "nexora_sph_tests.SphVerletWindow.NoninitialRestoreAndSmallerRetryPreserveHeldSnapshots",
        "nexora_sph_tests.SphVerletWindow.SnapshotRecordScopesAndRestoreRetryWithAndWithoutTrialReaders",
        "nexora_sph_tests.SphVerletWindow.RetainedTrialBudgetRefusalRestoresAndRetryWithoutReaderSucceeds",
        "nexora_sph_tests.SphAdamiWall.GravityOnlyMatchesFreeFall",
        "nexora_sph_tests.SphAdamiWall.FixedSourcesMatchIndependentWallForcesAcrossRestore",
        "nexora_sph_tests.SphAdamiWall.WallPressureAndForcesMatchIndependentReference",
        "nexora_sph_tests.SphHydrostaticInitial.ConventionalMassFourCases",
        "nexora_sph_tests.SphHydrostaticInitial.FixedWallShellCoversNonzeroSupport",
        "nexora_sph_tests.SphHydrostaticInitial.AirSideStaysOpen",
        "nexora_sph_tests.SphContinuity.SmoothCoupledPairHasSecondOrderDensityError",
        "nexora_sph_tests.SphContinuity.NonpositiveTrialDensityRestoresAndRetries",
        "nexora_data_uninitialized_tests.DataPreinitialization.FieldIssuerRequiresInitializedRuntime",
        "nexora_data_uninitialized_tests.DataPreinitialization.ExternalFieldRequiresInitializedRuntime",
        "nexora_reference_uninitialized_tests.ReferenceUninitialized.RejectUninitializedRuntime",
        mpi_test,
    ] + (["nexora_sph_tests.SphVerletWindow.DriftJustAboveFloatRangeCannotRoundBackToLargestPosition"]
         if precision == "float" else []))
    test_log = build / "Testing/Temporary/LastTest.log"
    shutil.copyfile(test_log, logs / "reference-ctest.log")
    command([sys.executable, ROOT / "tools/check_reference_precision.py", "--log", test_log,
             "--precision", precision, "--output", logs / "precision.json"], logs / "precision.log")
    command([*ctest, "--tests-regex", "^nexora_install_consumer$", "--verbose",
             "--output-junit", logs / "package.xml"], logs / "package.log")
    check_ctest(logs / "package.xml", ["nexora_install_consumer"])
    if args.measure_irregular:
        return measure_irregular(build, logs, precision, args.jobs)
    return None


def verify(args, profile):
    prefix = args.prefix.absolute()
    logs = args.logs.absolute()
    logs.mkdir(parents=True, exist_ok=True)
    verify_kokkos(prefix, profile, logs)
    verify_gtest(args.gtest_prefix, profile, logs)
    evidence = {"started": now(), "platform": platform.platform(),
                "runner_image": os.environ.get("ImageVersion"), "jobs": args.jobs,
                "mpi": args.mpi, "profiles": {}, "measurements": {}}
    try:
        if args.require_network_blocked:
            evidence["network_before"], blocked = verify_network()
            if not blocked:
                raise RuntimeError("network isolation could not be verified")
        else:
            evidence["network"] = "isolation not requested"
        for precision in args.precisions:
            evidence["measurements"][precision] = {
                "status": "skipped",
                "reason": ("measurement not requested" if not args.measure_irregular else
                           "profile did not reach measurement"),
            }
            try:
                measurement = verify_precision(args, profile, precision)
                if measurement:
                    evidence["measurements"][precision] = {
                        "status": measurement["status"],
                        "record": (args.logs.absolute() / f"{args.config.lower()}-{precision}" /
                                   "irregular-measurement.json").as_posix(),
                    }
                evidence["profiles"][precision] = "passed"
            except (OSError, RuntimeError, ET.ParseError) as error:
                evidence["profiles"][precision] = str(error)
                record_path = (args.logs.absolute() / f"{args.config.lower()}-{precision}" /
                               "irregular-measurement.json")
                if record_path.is_file():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    evidence["measurements"][precision] = {
                        "status": record.get("status", "failed"),
                        "record": record_path.as_posix(),
                    }
                print(error, file=sys.stderr)
        if args.require_network_blocked:
            evidence["network_after"], blocked = verify_network()
            if not blocked:
                raise RuntimeError("network isolation could not be verified after validation")
        if any(status != "passed" for status in evidence["profiles"].values()):
            raise RuntimeError(f"Serial validation failed; see {logs}")
    finally:
        evidence["finished"] = now()
        write_json(logs / f"validation-{args.config.lower()}.json", evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--config", choices=("Debug", "Release"), required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("cache-key")
    for action in ("provision", "verify"):
        phase = subparsers.add_parser(action)
        phase.add_argument("--prefix", type=Path, required=True)
        phase.add_argument("--logs", type=Path, required=True)
        phase.add_argument("--gtest-prefix", type=Path,
                           help="default: sibling gtest-1.17.0-<config> of --prefix")
        phase.add_argument("--jobs", type=int, default=2)
        if action == "provision":
            phase.add_argument("--archive", type=Path, required=True)
            phase.add_argument("--gtest-archive", type=Path,
                               help="default: googletest-1.17.0.tar.gz beside --archive")
            phase.add_argument("--download", action="store_true")
        else:
            phase.add_argument("--work-dir", type=Path, required=True)
            phase.add_argument("--precisions", nargs="+", choices=("float", "double"),
                               default=["float", "double"])
            phase.add_argument("--require-network-blocked", action="store_true")
            phase.add_argument("--mpi", action="store_true")
            phase.add_argument("--measure-irregular", action="store_true",
                               help="build, run twice, and record the irregular Serial measurement")
    args = parser.parse_args()
    if args.action != "cache-key":
        if args.gtest_prefix is None:
            args.gtest_prefix = args.prefix.parent / f"gtest-{GTEST_VERSION}-{args.config.lower()}"
        if args.action == "provision" and args.gtest_archive is None:
            args.gtest_archive = args.archive.with_name(f"googletest-{GTEST_VERSION}.tar.gz")
        if args.gtest_prefix.absolute() == args.prefix.absolute():
            parser.error("Kokkos and GoogleTest require distinct installation prefixes")
    try:
        profile = identity(args)
        if args.action == "cache-key":
            dependencies = {"kokkos": profile, "gtest": gtest_identity(profile)}
            key = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
            print(f"kokkos-{VERSION}-{profile['system']}-{profile['architecture']}-{args.config}-{key}")
        elif args.action == "provision":
            provision(args, profile)
            provision_gtest(args, profile)
        else:
            verify(args, profile)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError, ET.ParseError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
