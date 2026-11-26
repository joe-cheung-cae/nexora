cmake_minimum_required(VERSION 3.25.2)
if(DEFINED CMAKE_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD STREQUAL "20")
    message(FATAL_ERROR "Nexora requires C++20, matching the provisioned Kokkos package.")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(Nexora VERSION 0.2.0 LANGUAGES C CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NEXORA_ENABLE_MPI "Build the MPI C API smoke probe" OFF)
option(NEXORA_BUILD_DOCS "Enable the public and internal documentation targets" OFF)
option(NEXORA_ENABLE_DOUBLE_PRECISION "Use double instead of float for nexora::real" OFF)
set(NEXORA_REAL_IS_DOUBLE 0)
if(NEXORA_ENABLE_DOUBLE_PRECISION)
    set(NEXORA_REAL_IS_DOUBLE 1)
endif()

if(DEFINED NEXORA_GPU_BACKEND)
    message(FATAL_ERROR "NEXORA_GPU_BACKEND was removed; select a verified local Kokkos package with Kokkos_DIR. This slice supports Serial only.")
endif()
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "This CPU slice does not accept CMAKE_CUDA_ARCHITECTURES; GPU profiles require separate implementation and validation.")
endif()

# Search the checkout's pre-provisioned cache after CMake's standard package paths.
set(nexora_kokkos_version 5.2.2)
function(nexora_local_package_paths package output)
    set(paths)
    get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(NOT CMAKE_CROSSCOMPILING AND NOT multi_config)
        set(cache "${PROJECT_SOURCE_DIR}/.cache/dependencies/${package}")
        string(TOLOWER "${CMAKE_BUILD_TYPE}" config)
        if(config)
            list(APPEND paths "${cache}/install-${config}")
        endif()
        # MSVC requires a configuration-matched runtime library.
        if(NOT MSVC AND NOT config STREQUAL "release")
            list(APPEND paths "${cache}/install-release")
        endif()
    endif()
    set("${output}" "${paths}" PARENT_SCOPE)
endfunction()
nexora_local_package_paths("kokkos-${nexora_kokkos_version}" nexora_kokkos_local_paths)

set(Kokkos_LAUNCH_COMPILER OFF)
find_package(Kokkos ${nexora_kokkos_version} EXACT CONFIG REQUIRED
    PATHS ${nexora_kokkos_local_paths})
include(cmake/NexoraKokkos.cmake)
nexora_validate_kokkos()

if(NEXORA_ENABLE_MPI)
    include(cmake/NexoraMPI.cmake)
endif()

set(NEXORA_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/generated/include")
configure_file(cmake/Config.hpp.in "${NEXORA_GENERATED_INCLUDE_DIR}/nexora/core/Config.hpp" @ONLY)
configure_file(cmake/Version.hpp.in "${NEXORA_GENERATED_INCLUDE_DIR}/nexora/core/Version.hpp" @ONLY)

add_library(nexora_build_options INTERFACE)
add_library(nexora::build_options ALIAS nexora_build_options)
set_target_properties(nexora_build_options PROPERTIES EXPORT_NAME build_options)
target_compile_features(nexora_build_options INTERFACE cxx_std_20)

add_subdirectory(src/core)
add_subdirectory(src/data)
add_subdirectory(src/simulation)
add_subdirectory(src/methods/reference)
add_subdirectory(src/methods/sph)
add_subdirectory(src/participants/reference)
add_subdirectory(src/participants/sph)
add_subdirectory(third_party/lepton)
add_subdirectory(apps/prescribed_motion)
add_subdirectory(apps/nexora)
if(NEXORA_BUILD_DOCS)
    include(cmake/NexoraDocs.cmake)
endif()

add_subdirectory(validation/sph/fixed_csr)
add_subdirectory(benchmarks/sph)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(nexora_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Nexora")
configure_package_config_file(cmake/NexoraConfig.cmake.in
    "${PROJECT_BINARY_DIR}/NexoraConfig.cmake"
    INSTALL_DESTINATION "${nexora_package_dir}")
write_basic_package_version_file("${PROJECT_BINARY_DIR}/NexoraConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion)

install(TARGETS
    nexora_build_options
    nexora_core
    nexora_data_schema
    nexora_data
    nexora_participant_api
    nexora_simulation
    EXPORT NexoraTargets
)
install(TARGETS nexora RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION "${CMAKE_INSTALL_DOCDIR}")
install(DIRECTORY include/nexora DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${NEXORA_GENERATED_INCLUDE_DIR}/nexora" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT NexoraTargets NAMESPACE nexora:: DESTINATION "${nexora_package_dir}")
install(FILES
    "${PROJECT_BINARY_DIR}/NexoraConfig.cmake"
    "${PROJECT_BINARY_DIR}/NexoraConfigVersion.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/NexoraKokkos.cmake"
    DESTINATION "${nexora_package_dir}")

include(cmake/NexoraTargetGraph.cmake)
nexora_write_target_graph("${PROJECT_BINARY_DIR}/nexora-target-graph.json")
