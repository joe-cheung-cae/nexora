cmake_minimum_required(VERSION 3.25.2)

include(CMakePackageConfigHelpers)

if(NOT "${expected_double_precision}" MATCHES "^[01]$")
    message(FATAL_ERROR "expected_double_precision must be independently supplied as 0 or 1")
endif()
math(EXPR conflicting_double_precision "1 - ${expected_double_precision}")

set(config_args)
if(config)
    set(config_args --config "${config}")
endif()

file(REMOVE_RECURSE "${scratch}/installed" "${scratch}/relocated" "${scratch}/consumer"
    "${scratch}/consumer-source" "${scratch}/negative")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${project_build}"
    ${config_args} --prefix "${scratch}/installed" COMMAND_ERROR_IS_FATAL ANY)
file(RENAME "${scratch}/installed" "${scratch}/relocated")
file(GLOB_RECURSE private_headers "${scratch}/relocated/*.hpp")
string(CONCAT private_header_pattern
    "/(FieldIssuer|ExternalField|TransportBuilder|TransportReceiver|"
    "EntityCapacity|EntityOrder|Permutation|CsrAdjacency|StructureState|ResourceBudget|"
    "fixed_csr/.*|sph_fixture|Density|Density3D|Participant|Participant3D|Geometry3D|GeometryVersions3D|Pressure3D|Verlet3D|cell_grid_oracle|wcsph_oracle|"
    "(methods|participants)/(reference|sph)/.*)[.]hpp$")
list(FILTER private_headers INCLUDE REGEX "${private_header_pattern}")
if(private_headers)
    message(FATAL_ERROR "Private implementation headers leaked into the installation: ${private_headers}")
endif()
file(GLOB_RECURSE sph_artifacts "${scratch}/relocated/*")
list(FILTER sph_artifacts INCLUDE REGEX
    "/(lib)?nexora_(methods_sph|participants_sph|sph_fixed_csr_(fixture|tests)|sph_tests|sph_benchmark|sph_tank_trajectory)([.][^/]*)?$")
if(sph_artifacts)
    message(FATAL_ERROR "Internal SPH artifacts leaked into the installation: ${sph_artifacts}")
endif()
file(GLOB_RECURSE retired_headers "${scratch}/relocated/*.hpp")
list(FILTER retired_headers INCLUDE REGEX
    "/nexora/(exec/.*|core/ArrayView|core/math/(Vec|Tensor)|data/FieldView)[.]hpp$")
if(retired_headers)
    message(FATAL_ERROR "Retired public headers are still installed: ${retired_headers}")
endif()

set(consumer_copy "${scratch}/consumer-source")
file(COPY "${consumer_source}/CMakeLists.txt" "${consumer_source}/main.cpp"
    "${consumer_source}/time.cpp" "${consumer_source}/schema.cpp"
    "${consumer_source}/field_binding.cpp" "${consumer_source}/kokkos.cpp" "${view_probe}"
    DESTINATION "${consumer_copy}")

set(generator_args -G "${generator}")
if(generator_platform)
    list(APPEND generator_args -A "${generator_platform}")
endif()
if(generator_toolset)
    list(APPEND generator_args -T "${generator_toolset}")
endif()

set(configure_args ${generator_args}
    "-DCMAKE_MAKE_PROGRAM=${make_program}"
    "-DCMAKE_CXX_COMPILER=${cxx_compiler}"
    "-DCMAKE_CXX_FLAGS=${cxx_flags}"
    "-DCMAKE_BUILD_TYPE=${config}")
if(APPLE)
    string(REPLACE ";" "\\;" osx_architectures "${osx_architectures}")
    list(APPEND configure_args
        "-DCMAKE_OSX_ARCHITECTURES=${osx_architectures}"
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=${osx_deployment_target}"
        "-DCMAKE_OSX_SYSROOT=${osx_sysroot}")
endif()

set(nexora_arg "-DNexora_DIR=${scratch}/relocated/${package_dir}")
set(source_arg "-DCMAKE_C_COMPILER=${c_compiler}")
set(precision_expectation_arg "-DNEXORA_TEST_EXPECT_DOUBLE_PRECISION=${expected_double_precision}")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${consumer_copy}" -B "${scratch}/consumer"
    ${configure_args} "${nexora_arg}" "-DKokkos_DIR=${kokkos_dir}"
    -DCMAKE_DISABLE_FIND_PACKAGE_GTest=TRUE "${precision_expectation_arg}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${scratch}/consumer" ${config_args}
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${scratch}/consumer"
    -C "${config}" --verbose COMMAND_ERROR_IS_FATAL ANY)

# Exercise the relocated executable using only artifacts staged under scratch.
set(cli_oracle_copy "${scratch}/cli oracle${cli_executable_suffix}")
file(COPY_FILE "${cli_oracle}" "${cli_oracle_copy}" ONLY_IF_DIFFERENT)
set(cli_fixture_copy "${scratch}/relocated cli fixtures")
file(REMOVE_RECURSE "${cli_fixture_copy}")
file(MAKE_DIRECTORY "${cli_fixture_copy}")
foreach(fixture IN ITEMS cli-nonbinary-tail.case cli-invalid.case cli-underflow.case
                         cli-invalid-sign.case cli-too-many-steps.case cli-duplicate.case
                         cli-unknown-version.case cli-unsupported-execution.case
                         cli-invalid-interval.case cli-quantum.case cli-nan.case cli-inf.case
                         cli-rounded-stall.case cli-rounded-clock.case
                         cli-max-window-boundary.case cli-over-window-boundary.case)
    file(COPY_FILE "${fixture_dir}/${fixture}" "${cli_fixture_copy}/${fixture}")
endforeach()
file(COPY_FILE "${cli_suite}" "${scratch}/relocated-cli-suite.cmake")
set(relocated_cli "${scratch}/relocated/${cli_install_bindir}/${cli_executable_name}")
if(NOT EXISTS "${relocated_cli}")
    message(FATAL_ERROR "Relocated Nexora CLI was not installed at ${relocated_cli}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}"
    "-Dcli=${relocated_cli}"
    "-Doracle=${cli_oracle_copy}"
    "-Dfixture_dir=${cli_fixture_copy}"
    "-Dscratch=${scratch}/relocated-cli-suite"
    "-Dexpected_precision=${expected_precision}"
    -P "${scratch}/relocated-cli-suite.cmake"
    COMMAND_ERROR_IS_FATAL ANY)

function(expect_configure_rejection case source expected_diagnostic)
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${scratch}/negative/${case}"
        ${configure_args} ${ARGN}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    string(FIND "${output}\n${error}" "${expected_diagnostic}" diagnostic_position)
    if(result STREQUAL "0" OR diagnostic_position EQUAL -1)
        message(FATAL_ERROR "${case} did not reject the configuration as expected.\n${output}\n${error}")
    endif()
    message(STATUS "${case} rejected with '${expected_diagnostic}'")
endfunction()

function(expect_source_and_consumer_rejection case expected_diagnostic)
    expect_configure_rejection("source-${case}" "${project_source}" "${expected_diagnostic}"
        "${source_arg}" ${ARGN})
    expect_configure_rejection("consumer-${case}" "${consumer_copy}" "${expected_diagnostic}"
        "${nexora_arg}" ${ARGN})
endfunction()

expect_configure_rejection(consumer-precision-mismatch "${consumer_copy}" "Nexora precision mismatch"
    "${nexora_arg}" "-DKokkos_DIR=${kokkos_dir}" "${precision_expectation_arg}"
    "-DNEXORA_ENABLE_DOUBLE_PRECISION=${conflicting_double_precision}")

# Restrict package discovery to this scratch root so a system Kokkos cannot mask a failure.
set(fixture_root "${scratch}/negative/packages")
set(isolate_kokkos "-DCMAKE_FIND_ROOT_PATH=${fixture_root}" "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY")
expect_source_and_consumer_rejection(missing-kokkos
    "Could not find a package configuration file provided by \"Kokkos\""
    ${isolate_kokkos} "-DKokkos_DIR=${fixture_root}/missing")

set(wrong_version_dir "${fixture_root}/wrong-version")
file(WRITE "${wrong_version_dir}/KokkosConfig.cmake" "# Mock Kokkos package for version rejection.\n")
write_basic_package_version_file("${wrong_version_dir}/KokkosConfigVersion.cmake"
    VERSION 5.2.1 COMPATIBILITY ExactVersion ARCH_INDEPENDENT)
expect_source_and_consumer_rejection(wrong-kokkos-version "version: 5.2.1"
    ${isolate_kokkos} "-DKokkos_DIR=${wrong_version_dir}")

# A directory-only search keeps a system or checkout-local GTest from masking these failures.
set(gtest_fixture "${fixture_root}/gtest")
file(MAKE_DIRECTORY "${gtest_fixture}")
set(isolate_gtest "-DCMAKE_FIND_ROOT_PATH=${gtest_fixture}" "-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY")
expect_configure_rejection(source-missing-gtest "${project_source}"
    "Could not find a package configuration file provided by \"GTest\""
    "${source_arg}" -DBUILD_TESTING=ON "-DKokkos_DIR=${kokkos_dir}" ${isolate_gtest})
file(WRITE "${gtest_fixture}/GTestConfig.cmake" "# Mock package for version rejection.\n")
write_basic_package_version_file("${gtest_fixture}/GTestConfigVersion.cmake"
    VERSION 1.16.0 COMPATIBILITY ExactVersion ARCH_INDEPENDENT)
expect_configure_rejection(source-wrong-gtest-version "${project_source}" "version: 1.16.0"
    "${source_arg}" -DBUILD_TESTING=ON "-DKokkos_DIR=${kokkos_dir}"
    "-DGTest_DIR=${gtest_fixture}" ${isolate_gtest})

expect_configure_rejection(source-legacy-gpu-backend "${project_source}" "NEXORA_GPU_BACKEND was removed"
    "${source_arg}" "-DKokkos_DIR=${kokkos_dir}" "-DNEXORA_GPU_BACKEND=NONE")
expect_source_and_consumer_rejection(cxx17 "Nexora requires C++20"
    "-DKokkos_DIR=${kokkos_dir}" "-DCMAKE_CXX_STANDARD=17")

function(expect_incompatible_kokkos case expected_diagnostic incompatible_setting)
    set(mock_dir "${fixture_root}/${case}")
    file(WRITE "${mock_dir}/KokkosConfig.cmake" [=[
# Mock same-version Kokkos metadata. The last setting is deliberately incompatible.
set(Kokkos_DEVICES SERIAL)
set(Kokkos_CXX_STANDARD 20)
set(Kokkos_COMPILE_LANGUAGE CXX)
set(Kokkos_CXX_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
set(Kokkos_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
add_library(Kokkos::kokkos INTERFACE IMPORTED)
]=] "${incompatible_setting}\n")
    write_basic_package_version_file("${mock_dir}/KokkosConfigVersion.cmake"
        VERSION 5.2.2 COMPATIBILITY ExactVersion ARCH_INDEPENDENT)
    expect_source_and_consumer_rejection("${case}" "${expected_diagnostic}"
        ${isolate_kokkos} "-DKokkos_DIR=${mock_dir}")
endfunction()

expect_incompatible_kokkos(unsupported-kokkos-devices "requires a Serial-only Kokkos package"
    "set(Kokkos_DEVICES OPENMP SERIAL)")
expect_incompatible_kokkos(wrong-kokkos-standard "requires Kokkos C++20/CXX"
    "set(Kokkos_CXX_STANDARD 23)")
expect_incompatible_kokkos(wrong-kokkos-compiler "requires the Kokkos compiler identity/version"
    "set(Kokkos_CXX_COMPILER_VERSION 0.0.1)")
