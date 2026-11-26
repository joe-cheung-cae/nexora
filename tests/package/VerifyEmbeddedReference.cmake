cmake_minimum_required(VERSION 3.25.2)

file(REMOVE_RECURSE "${scratch}")
file(WRITE "${scratch}/source/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.25.2)\n"
    "project(NexoraEmbeddedReference LANGUAGES C CXX)\n"
    "if(MSVC)\n"
    "    add_compile_options(/fp:fast)\n"
    "else()\n"
    "    add_compile_options(-ffast-math)\n"
    "endif()\n"
    "add_subdirectory(\"${project_source}\" nexora)\n")

set(configure_args -G "${generator}"
    "-DCMAKE_MAKE_PROGRAM=${make_program}"
    "-DCMAKE_C_COMPILER=${c_compiler}"
    "-DCMAKE_CXX_COMPILER=${cxx_compiler}"
    "-DCMAKE_BUILD_TYPE=${config}"
    "-DKokkos_DIR=${kokkos_dir}"
    "-DNEXORA_ENABLE_DOUBLE_PRECISION=${double_precision}"
    -DNEXORA_ENABLE_MPI=OFF -DBUILD_TESTING=OFF)
if(generator_platform)
    list(APPEND configure_args -A "${generator_platform}")
endif()
if(generator_toolset)
    list(APPEND configure_args -T "${generator_toolset}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${scratch}/source" -B "${scratch}/build"
    ${configure_args} COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${scratch}/build" --config "${config}"
    --target nexora --parallel 2 COMMAND_ERROR_IS_FATAL ANY)

set(cli "${scratch}/build/nexora/apps/nexora/nexora${executable_suffix}")
if(EXISTS "${scratch}/build/nexora/apps/nexora/${config}/nexora${executable_suffix}")
    set(cli "${scratch}/build/nexora/apps/nexora/${config}/nexora${executable_suffix}")
endif()
set(case "${project_source}/cases/reference/cli-nonbinary-tail.case")
execute_process(COMMAND "${cli}" plan "${case}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result STREQUAL "0" OR NOT output MATCHES "(^|\n)validated=false(\n|$)"
   OR NOT output MATCHES "(^|\n)validation_evidence=none(\n|$)")
    message(FATAL_ERROR "Embedded compiler options inherited local validation evidence.\nexit=${result}\n${output}\n${error}")
endif()
execute_process(COMMAND "${cli}" run "${case}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result STREQUAL "2" OR NOT output STREQUAL "" OR NOT error MATCHES
   "reference Serial profile requires validation evidence for this build")
    message(FATAL_ERROR "Embedded unvalidated run was not refused.\nexit=${result}\n${output}\n${error}")
endif()
execute_process(COMMAND "${cli}" run --fixture "${case}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result STREQUAL "2" OR NOT error MATCHES
   "controlled reference fixture execution requires a build with BUILD_TESTING enabled")
    message(FATAL_ERROR "Embedded production build enabled controlled fixtures.\nexit=${result}\n${output}\n${error}")
endif()
