#!/usr/bin/env bash
# Ubuntu 24.04 amd64, GCC Release Serial CI. Run "prepare" online, "verify" in unshare --net.
set -euo pipefail
export LC_ALL=C

version=1.9.8+ds-2ubuntu0.1
sha256=76a3a24a594d0cb6315eae5bfc78aa4b036b3ba981e3c8fadddcb4dea1792b7c
package="doxygen:amd64=$version"
deb="doxygen_${version}_amd64.deb"
run=${NEXORA_RUN:?set NEXORA_RUN to a fresh absolute validation root}
apt_dir="$run/doxygen-apt"
archives="$apt_dir/archives"
logs="$run/logs/doxygen"
mkdir -p "$logs"

# Private lists and caches limit the authentication chain to the Ubuntu archive keyring.
apt_options=(-o "Dir::Etc::sourcelist=$apt_dir/ubuntu.list" -o "Dir::Etc::sourceparts=-"
    -o "Dir::State::lists=$apt_dir/lists" -o "Dir::Cache::archives=$archives"
    -o Dir::Cache::pkgcache= -o Dir::Cache::srcpkgcache= -o Acquire::Languages=none)

prepare_graphviz() {
    local graphviz_archives="$apt_dir/graphviz-archives" package_status package_version artifact indexed integrity
    apt-cache "${apt_options[@]}" policy graphviz > "$logs/graphviz-policy.txt"
    package_status=$(dpkg-query -W -f='${db:Status-Status}' graphviz 2>/dev/null || true)
    if [[ "$package_status" != installed || ! -x /usr/bin/dot ]]; then
        package_version=$(awk '$1 == "Candidate:" {print $2}' "$logs/graphviz-policy.txt")
        [[ -n "$package_version" && "$package_version" != '(none)' ]]
        apt-cache "${apt_options[@]}" show "graphviz:amd64=$package_version" > "$logs/graphviz-package.txt"
        mkdir -p "$graphviz_archives/partial"
        local install=(sudo -n apt-get "${apt_options[@]}" -o "Dir::Cache::archives=$graphviz_archives"
            --yes --no-remove --no-install-recommends --reinstall)
        "${install[@]}" --download-only install "graphviz:amd64=$package_version"
        artifact="$graphviz_archives/graphviz_${package_version/:/%3a}_amd64.deb"
        indexed=$(awk '/^SHA256: / {print $2}' "$logs/graphviz-package.txt" | sort -u)
        [[ "$indexed" =~ ^[0-9a-f]{64}$ ]]
        printf '%s  %s\n' "$indexed" "$artifact" | sha256sum --check
        (cd "$graphviz_archives" && sha256sum -- *.deb) > "$logs/graphviz-debs.sha256"
        "${install[@]}" --no-download install "graphviz:amd64=$package_version"
        echo "Graphviz supplied from the signed Ubuntu archive"
    else
        package_version=$(dpkg-query -W -f='${Version}' graphviz)
        apt-cache "${apt_options[@]}" show "graphviz:amd64=$package_version" > "$logs/graphviz-package.txt"
        echo "Graphviz supplied by the Ubuntu runner image"
    fi
    dpkg-query -S /usr/bin/dot | grep -E '^graphviz(:amd64)?: /usr/bin/dot$'
    integrity=$(sudo -n dpkg --verify graphviz)
    if [[ -n "$integrity" ]]; then
        printf 'Graphviz package verification failed:\n%s\n' "$integrity" >&2
        return 1
    fi
    dpkg-query -W -f='Installed ${Package} ${Version} ${Architecture} (source ${source:Package} ${source:Version})\n' graphviz
    [[ "$(dpkg-query -W -f='${Version}' graphviz)" == "$package_version" ]]
    sha256sum /usr/bin/dot > "$logs/graphviz-dot.sha256"
    cp /usr/share/doc/graphviz/copyright "$logs/graphviz-copyright"
    cp /usr/share/doc/graphviz/changelog.Debian.gz "$logs/graphviz-changelog.Debian.gz"
    /usr/bin/dot -V
}

prepare() {
    mkdir "$apt_dir"
    mkdir -p "$archives/partial" "$apt_dir/lists/partial"
    # Keep the locked tool available across updates and resolve missing shared libraries
    # against the Noble runner's installed packages and current security updates.
    cat > "$apt_dir/ubuntu.list" <<'LIST'
deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://snapshot.ubuntu.com/ubuntu/20260701T000000Z noble-security universe
deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu noble main universe
deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu noble-updates main universe
deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://security.ubuntu.com/ubuntu noble-security main universe
LIST
    cp "$apt_dir/ubuntu.list" "$logs/"
    sudo -n apt-get "${apt_options[@]}" -o Debug::Acquire::gpgv=true update --error-on=any
    cp "$apt_dir"/lists/*_InRelease "$logs/"
    apt-cache "${apt_options[@]}" policy doxygen | tee "$logs/apt-policy.txt"
    apt-cache "${apt_options[@]}" show "$package" > "$logs/doxygen-package.txt"
    sudo -n apt-get "${apt_options[@]}" --yes --no-remove --no-install-recommends --reinstall \
        --allow-downgrades --download-only install "$package"
    # APT authenticates dependency downloads too. Retain all their names and byte digests.
    (cd "$archives" && sha256sum -- *.deb) > "$logs/debs.sha256"
    local indexed
    indexed=$(awk '/^SHA256: / {print $2}' "$logs/doxygen-package.txt" | sort -u)
    [[ "$indexed" == "$sha256" ]]
    (cd "$archives" && printf '%s  %s\n' "$indexed" "$deb" | sha256sum --check)
    dpkg-deb --field "$archives/$deb" Package Source Version Architecture Depends
    # Debian omits Source when the binary and source have identical names/versions.
    local source
    source=$(dpkg-deb --field "$archives/$deb" Source)
    echo "Ubuntu source: ${source:-doxygen ($version)}"
    prepare_graphviz
}

verify_missing() (
    # Root only normal CMake program searches. The toolchain is inherited by the
    # independent probe, project, try-compiles and relocated installed consumer.
    local root="$run/no-doxygen-tools" probe="$run/doxygen-probe"
    mkdir "$root" "$probe"
    mkdir -p "$root/usr/bin"
    python - "$root/usr/bin" <<'PY'
import os
from pathlib import Path
import sys
dest = Path(sys.argv[1])
for directory in os.get_exec_path():
    if not Path(directory).is_dir():
        continue
    for program in Path(directory).iterdir():
        link = dest / program.name
        if (not program.name.startswith("doxygen") and not link.is_symlink()
                and program.is_file() and os.access(program, os.X_OK)):
            link.symlink_to(program.absolute())
PY
    export CMAKE_TOOLCHAIN_FILE="$root/search.cmake"
    cat > "$CMAKE_TOOLCHAIN_FILE" <<'CMAKE'
list(APPEND CMAKE_FIND_ROOT_PATH "${CMAKE_CURRENT_LIST_DIR}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
CMAKE
    cat > "$probe/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.25.2)
project(probe NONE)
find_program(doxygen_program NAMES doxygen NO_CACHE)
find_package(Doxygen QUIET COMPONENTS doxygen)
if(doxygen_program OR Doxygen_FOUND OR DOXYGEN_EXECUTABLE)
    message(FATAL_ERROR "Doxygen was discoverable: ${doxygen_program} / ${DOXYGEN_EXECUTABLE}")
endif()
message(STATUS "Doxygen absent from normal program and package discovery")
CMAKE
    cmake -S "$probe" -B "$probe/build" -G Ninja
    python tools/validate_serial.py --cc "$NEXORA_CC" --cxx "$NEXORA_CXX" --config Release verify \
        --prefix "$NEXORA_DEPS/kokkos" --gtest-prefix "$NEXORA_DEPS/gtest" \
        --work-dir "$run/build" --logs "$run/logs" --precisions float double --require-network-blocked
    grep -Fx 'NEXORA_BUILD_DOCS:BOOL=OFF' "$run/build/release-float/CMakeCache.txt"
    local result=0
    cmake -S . -B "$run/docs-missing" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        "-DCMAKE_C_COMPILER=$(command -v "$NEXORA_CC")" "-DCMAKE_CXX_COMPILER=$(command -v "$NEXORA_CXX")" \
        "-DKokkos_DIR=$NEXORA_DEPS/kokkos/lib/cmake/Kokkos" \
        -DNEXORA_BUILD_DOCS=ON -DBUILD_TESTING=OFF > "$logs/missing-configure.log" 2>&1 || result=$?
    cat "$logs/missing-configure.log"
    echo "Missing-tool configure exit: $result"
    [[ "$result" != 0 ]]
    grep -F 'Could NOT find Doxygen' "$logs/missing-configure.log"
    grep -F 'DOXYGEN_EXECUTABLE' "$logs/missing-configure.log"
)

verify() {
    # Check isolation before configuring the missing-tool probe.
    python - <<'PYTHON'
import sys
sys.path.insert(0, "tools")
from validate_serial import verify_network
evidence, blocked = verify_network()
print(evidence, flush=True)
if not blocked:
    raise SystemExit("network isolation could not be verified")
PYTHON
    verify_missing
    sha256sum --check "$logs/graphviz-dot.sha256"
    /usr/bin/dot -V
    (cd "$archives" && sha256sum --check "$logs/debs.sha256")
    local stage="$run/doxygen" artifact
    mkdir "$stage"
    for artifact in "$archives"/*.deb; do
        dpkg-deb --extract "$artifact" "$stage"
    done
    # Put unpacked runtime dependencies ahead of libraries already present in the image.
    export LD_LIBRARY_PATH="$stage/usr/lib/x86_64-linux-gnu:$stage/usr/lib/llvm-18/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ldd "$stage/usr/bin/doxygen"
    cp "$stage/usr/share/doc/doxygen/"{copyright,changelog.Debian.gz} /usr/share/common-licenses/GPL-2 "$logs/"
    echo "Ubuntu distribution changes: changelog.Debian.gz; Nexora patches: none"
    cat /etc/os-release
    uname -m
    echo "Runner image: ${ImageOS:-unspecified} ${ImageVersion:-unspecified}"
    "$NEXORA_CC" --version
    "$NEXORA_CXX" --version
    cmake --version
    ninja --version
    "$stage/usr/bin/doxygen" --version
    python tools/validate_docs.py --cc "$NEXORA_CC" --cxx "$NEXORA_CXX" \
        --kokkos-prefix "$NEXORA_DEPS/kokkos" --doxygen "$stage/usr/bin/doxygen" --dot /usr/bin/dot \
        --work-dir "$run/docs" --logs "$logs" --require-network-blocked
}

case "${1:-}" in
    prepare|verify) ;;
    *) echo 'Usage: bash tools/ci_doxygen.sh prepare|verify' >&2; exit 2 ;;
esac
# Retain failure output, commands, exit status and wall time in the uploaded artifact.
phase=$1
exec > >(tee "$logs/$phase.log") 2>&1
trap 'status=$?; echo "[$(date --iso-8601=seconds)] $phase exit $status"' EXIT
echo "[$(date --iso-8601=seconds)] $phase"
set -x
"$phase"
