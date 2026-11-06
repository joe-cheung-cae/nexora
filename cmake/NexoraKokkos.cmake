# Shared by the build tree and installed package. This is deliberately a Serial-only
# acceptance profile, not a backend selector or the complete Phase 1 build matrix.
function(nexora_validate_kokkos)
    set(devices "${Kokkos_DEVICES}")
    list(TRANSFORM devices TOUPPER)
    list(REMOVE_DUPLICATES devices)
    list(SORT devices)
    if(NOT devices STREQUAL "SERIAL")
        message(FATAL_ERROR "Nexora's CPU slice requires a Serial-only Kokkos package; found '${Kokkos_DEVICES}'. Provision/select Kokkos 5.2.2 with only Kokkos_ENABLE_SERIAL=ON.")
    endif()
    if(NOT Kokkos_CXX_STANDARD STREQUAL "20" OR NOT Kokkos_COMPILE_LANGUAGE STREQUAL "CXX")
        message(FATAL_ERROR "Nexora requires Kokkos C++20/CXX; rebuild the local dependency with that language profile.")
    endif()
    if(DEFINED CMAKE_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD STREQUAL "20")
        message(FATAL_ERROR "Nexora requires C++20 in downstream consumers.")
    endif()
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "Nexora's CPU slice requires a 64-bit target.")
    endif()
    # This first profile conservatively requires the same compiler release. It
    # does not infer ABI compatibility across compiler families or versions.
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL Kokkos_CXX_COMPILER_ID
       OR NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL Kokkos_CXX_COMPILER_VERSION)
        message(FATAL_ERROR "Nexora requires the Kokkos compiler identity/version: ${Kokkos_CXX_COMPILER_ID} ${Kokkos_CXX_COMPILER_VERSION}; consumer is ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
    endif()
    if(NOT TARGET Kokkos::kokkos)
        message(FATAL_ERROR "The selected Kokkos package does not export Kokkos::kokkos.")
    endif()
endfunction()
