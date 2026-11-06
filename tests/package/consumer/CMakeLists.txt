cmake_minimum_required(VERSION 3.25.2)
project(NexoraConsumer LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Nexora 0.1 CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE nexora::simulation nexora::exec_cpu)

add_executable(kokkos_consumer kokkos_tests.cpp)
target_link_libraries(kokkos_consumer PRIVATE nexora::data)

enable_testing()
add_test(NAME consumer COMMAND consumer)
add_test(NAME kokkos_consumer COMMAND kokkos_consumer)
