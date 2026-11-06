cmake_minimum_required(VERSION 3.24)
project(NexoraConsumer LANGUAGES CXX)

find_package(Nexora 0.1 CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE nexora::simulation nexora::exec_cpu)

enable_testing()
add_test(NAME consumer COMMAND consumer)
