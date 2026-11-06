cmake_minimum_required(VERSION 3.24)
project(Nexora VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(NEXORA_GPU_BACKEND "NONE" CACHE STRING "GPU runtime for the build (NONE or CUDA)")
set_property(CACHE NEXORA_GPU_BACKEND PROPERTY STRINGS NONE CUDA)
option(NEXORA_ENABLE_MPI "Build the MPI C API smoke probe" OFF)

if(NEXORA_GPU_BACKEND STREQUAL "CUDA")
    if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)native(;|$)")
        message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to an explicit target, e.g. 80.")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(NOT NEXORA_GPU_BACKEND STREQUAL "NONE")
    message(FATAL_ERROR "Unsupported NEXORA_GPU_BACKEND='${NEXORA_GPU_BACKEND}'. Select NONE or CUDA.")
endif()

if(NEXORA_ENABLE_MPI)
    include(cmake/NexoraMPI.cmake)
endif()

set(NEXORA_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/generated/include")
configure_file(cmake/Version.hpp.in "${NEXORA_GENERATED_INCLUDE_DIR}/nexora/core/Version.hpp" @ONLY)

add_library(nexora_build_options INTERFACE)
add_library(nexora::build_options ALIAS nexora_build_options)
set_target_properties(nexora_build_options PROPERTIES EXPORT_NAME build_options)
target_compile_features(nexora_build_options INTERFACE cxx_std_17)

add_subdirectory(src/core)
add_subdirectory(src/exec)
add_subdirectory(src/data)
add_subdirectory(src/simulation)
add_subdirectory(apps/nexora)

if(BUILD_TESTING)
    function(nexora_add_test name source)
        add_executable("${name}" "${source}")
        target_link_libraries("${name}" PRIVATE ${ARGN})
        add_test(NAME "${name}" COMMAND "${name}")
        set_tests_properties("${name}" PROPERTIES TIMEOUT 30)
    endfunction()

    add_subdirectory(tests/core)
    add_subdirectory(tests/exec)
    add_subdirectory(tests/data)
    add_subdirectory(tests/simulation)
    add_subdirectory(tests/package)
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
    nexora_exec_api
    nexora_exec_cpu
    nexora_data
    nexora_participant_api
    nexora_simulation
    EXPORT NexoraTargets
)
install(TARGETS nexora RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/nexora DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${NEXORA_GENERATED_INCLUDE_DIR}/nexora" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT NexoraTargets NAMESPACE nexora:: DESTINATION "${nexora_package_dir}")
install(FILES
    "${PROJECT_BINARY_DIR}/NexoraConfig.cmake"
    "${PROJECT_BINARY_DIR}/NexoraConfigVersion.cmake"
    DESTINATION "${nexora_package_dir}")

include(cmake/NexoraTargetGraph.cmake)
nexora_write_target_graph("${PROJECT_BINARY_DIR}/nexora-target-graph.json")
