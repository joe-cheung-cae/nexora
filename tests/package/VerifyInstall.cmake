cmake_minimum_required(VERSION 3.24)

set(config_args)
if(config)
    set(config_args --config "${config}")
endif()

file(REMOVE_RECURSE "${scratch}/installed" "${scratch}/relocated" "${scratch}/consumer")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${project_build}"
    ${config_args} --prefix "${scratch}/installed" COMMAND_ERROR_IS_FATAL ANY)
file(RENAME "${scratch}/installed" "${scratch}/relocated")

set(generator_args -G "${generator}")
if(generator_platform)
    list(APPEND generator_args -A "${generator_platform}")
endif()
if(generator_toolset)
    list(APPEND generator_args -T "${generator_toolset}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -S "${consumer_source}" -B "${scratch}/consumer"
    ${generator_args} "-DCMAKE_CXX_COMPILER=${compiler}" "-DCMAKE_BUILD_TYPE=${config}"
    "-DNexora_DIR=${scratch}/relocated/${package_dir}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${scratch}/consumer" ${config_args}
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${scratch}/consumer"
    -C "${config}" --output-on-failure COMMAND_ERROR_IS_FATAL ANY)
