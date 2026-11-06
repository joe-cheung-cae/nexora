cmake_minimum_required(VERSION 3.25.2)
project(NexoraConsumer LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Nexora 0.1 CONFIG REQUIRED)
if(TARGET nexora_methods_reference OR TARGET nexora_participants_reference
   OR TARGET nexora::methods_reference OR TARGET nexora::participants_reference
   OR TARGET nexora::nexora_methods_reference OR TARGET nexora::nexora_participants_reference)
    message(FATAL_ERROR "The reference method and participant must remain internal targets")
endif()

add_executable(consumer main.cpp time.cpp)
target_link_libraries(consumer PRIVATE nexora::simulation nexora::exec_cpu)

add_executable(kokkos_consumer kokkos_tests.cpp)
target_link_libraries(kokkos_consumer PRIVATE nexora::data)

enable_testing()
add_test(NAME consumer COMMAND consumer)
add_test(NAME kokkos_consumer COMMAND kokkos_consumer)
