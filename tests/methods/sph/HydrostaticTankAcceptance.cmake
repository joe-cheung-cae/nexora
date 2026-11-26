string(RANDOM LENGTH 16 suffix)
set(run "${scratch}/tank-n24-${suffix}")
message(STATUS "Full tank acceptance outputs: ${run}")
execute_process(COMMAND "${trajectory}" 24 "${run}/n24"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${python}" "${summarizer}" "${run}"
    --resolutions 24 --precision float --require-targets --output-dir "${run}"
    COMMAND_ERROR_IS_FATAL ANY)
