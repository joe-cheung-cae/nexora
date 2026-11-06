add_library(nexora_methods_reference STATIC State.cpp)
target_include_directories(nexora_methods_reference PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_link_libraries(nexora_methods_reference PUBLIC nexora::data Kokkos::kokkos)
