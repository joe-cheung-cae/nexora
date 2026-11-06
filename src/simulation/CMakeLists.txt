add_library(nexora_participant_api INTERFACE)
add_library(nexora::participant_api ALIAS nexora_participant_api)
set_target_properties(nexora_participant_api PROPERTIES EXPORT_NAME participant_api)
target_include_directories(nexora_participant_api INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(nexora_participant_api INTERFACE nexora::data)

add_library(nexora_simulation STATIC Simulation.cpp)
add_library(nexora::simulation ALIAS nexora_simulation)
set_target_properties(nexora_simulation PROPERTIES EXPORT_NAME simulation)
target_include_directories(nexora_simulation PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(nexora_simulation PUBLIC nexora::participant_api)
