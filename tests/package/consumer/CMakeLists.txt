cmake_minimum_required(VERSION 3.25.2)
project(NexoraConsumer LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Nexora 0.1 CONFIG REQUIRED)
foreach(internal_target IN ITEMS methods_reference participants_reference methods_sph
                                 participants_sph sph_tests sph_benchmark sph_tank_trajectory kokkos_test_main)
    if(TARGET "nexora_${internal_target}" OR TARGET "nexora::${internal_target}"
       OR TARGET "nexora::nexora_${internal_target}")
        message(FATAL_ERROR "Internal target is exported: ${internal_target}")
    endif()
endforeach()
foreach(retired_target IN ITEMS nexora_exec_api nexora_exec_cpu nexora::exec_api nexora::exec_cpu)
    if(TARGET "${retired_target}")
        message(FATAL_ERROR "Retired execution target is still exported: ${retired_target}")
    endif()
endforeach()

function(expect_export_links target)
    get_target_property(actual "${target}" INTERFACE_LINK_LIBRARIES)
    set(expected ${ARGN})
    list(SORT actual)
    list(SORT expected)
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "${target} exports dependencies '${actual}', expected '${expected}'")
    endif()
endfunction()

expect_export_links(nexora::core nexora::build_options)
expect_export_links(nexora::data_schema nexora::core)
expect_export_links(nexora::participant_api nexora::data_schema)
expect_export_links(nexora::data nexora::data_schema Kokkos::kokkos)
expect_export_links(nexora::simulation nexora::participant_api)

if(NOT "${NEXORA_TEST_EXPECT_DOUBLE_PRECISION}" MATCHES "^[01]$")
    message(FATAL_ERROR "NEXORA_TEST_EXPECT_DOUBLE_PRECISION must be independently supplied as 0 or 1")
endif()
set(precision_probe "
#include <nexora/core/Real.hpp>
#include <type_traits>
static_assert(NEXORA_ENABLE_DOUBLE_PRECISION == ${NEXORA_TEST_EXPECT_DOUBLE_PRECISION});
static_assert(std::is_same_v<nexora::real,
    std::conditional_t<${NEXORA_TEST_EXPECT_DOUBLE_PRECISION}, double, float>>);
int main() { return 0; }
")
try_compile(matching_precision_compiled
    SOURCE_FROM_CONTENT precision.cpp "${precision_probe}"
    NO_CACHE
    LINK_LIBRARIES nexora::core
    COMPILE_DEFINITIONS "-DNEXORA_ENABLE_DOUBLE_PRECISION=${NEXORA_TEST_EXPECT_DOUBLE_PRECISION}"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    OUTPUT_VARIABLE matching_precision_output)
if(NOT matching_precision_compiled)
    message(FATAL_ERROR "Matching Nexora precision failed to compile:\n${matching_precision_output}")
endif()
try_compile(dynamic_precision_compiled
    SOURCE_FROM_CONTENT precision.cpp
        "#define NEXORA_ENABLE_DOUBLE_PRECISION (${NEXORA_TEST_EXPECT_DOUBLE_PRECISION}+__COUNTER__)\n${precision_probe}"
    NO_CACHE
    LINK_LIBRARIES nexora::core
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    OUTPUT_VARIABLE dynamic_precision_output)
if(NOT dynamic_precision_compiled)
    message(FATAL_ERROR "Nexora precision did not stay fixed after macro validation:\n${dynamic_precision_output}")
endif()
math(EXPR conflicting_double_precision "1 - ${NEXORA_TEST_EXPECT_DOUBLE_PRECISION}")
try_compile(conflicting_precision_compiled
    SOURCE_FROM_CONTENT precision.cpp "${precision_probe}"
    NO_CACHE
    LINK_LIBRARIES nexora::core
    COMPILE_DEFINITIONS "-DNEXORA_ENABLE_DOUBLE_PRECISION=${conflicting_double_precision}"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    OUTPUT_VARIABLE conflicting_precision_output)
string(FIND "${conflicting_precision_output}"
    "NEXORA_ENABLE_DOUBLE_PRECISION conflicts with the Nexora build configuration"
    precision_diagnostic_position)
if(conflicting_precision_compiled OR precision_diagnostic_position EQUAL -1)
    message(FATAL_ERROR "Conflicting Nexora precision did not fail as expected:\n${conflicting_precision_output}")
endif()

add_executable(consumer main.cpp time.cpp)
target_link_libraries(consumer PRIVATE nexora::simulation)
target_compile_definitions(consumer PRIVATE
    "NEXORA_TEST_EXPECT_DOUBLE_PRECISION=${NEXORA_TEST_EXPECT_DOUBLE_PRECISION}")

add_executable(schema_consumer schema.cpp)
target_link_libraries(schema_consumer PRIVATE nexora::data_schema)

add_executable(kokkos_consumer kokkos.cpp field_binding.cpp)
target_link_libraries(kokkos_consumer PRIVATE nexora::data)
target_compile_definitions(kokkos_consumer PRIVATE
    "NEXORA_TEST_EXPECT_DOUBLE_PRECISION=${NEXORA_TEST_EXPECT_DOUBLE_PRECISION}")

enable_testing()
add_test(NAME consumer COMMAND consumer)
add_test(NAME schema_consumer COMMAND schema_consumer)
add_test(NAME kokkos_consumer COMMAND kokkos_consumer)
