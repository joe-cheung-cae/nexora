# Build this measurement explicitly with --target nexora_sph_benchmark.
add_executable(nexora_sph_benchmark EXCLUDE_FROM_ALL sph_benchmark.cpp)
target_link_libraries(nexora_sph_benchmark PRIVATE nexora_participants_sph nexora::simulation)
target_include_directories(nexora_sph_benchmark PRIVATE "${PROJECT_SOURCE_DIR}/tests/methods/sph")
