cmake_minimum_required(VERSION 3.25.2)

foreach(required IN ITEMS cli oracle fixture_dir scratch expected_precision)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "RunCliSuite.cmake requires -D${required}=...")
    endif()
endforeach()
if(NOT expected_precision MATCHES "^(float|double)$")
    message(FATAL_ERROR "expected_precision must be float or double")
endif()
if(expected_precision STREQUAL "float")
    set(expected_dtype float32)
else()
    set(expected_dtype float64)
endif()

set(case_dir "${scratch}/case fixtures with spaces")
file(REMOVE_RECURSE "${case_dir}")
file(MAKE_DIRECTORY "${case_dir}")
foreach(case_name IN ITEMS cli-nonbinary-tail.case cli-invalid.case cli-underflow.case
                           cli-invalid-sign.case cli-too-many-steps.case cli-duplicate.case
                           cli-unknown-version.case cli-unsupported-execution.case
                           cli-invalid-interval.case cli-quantum.case cli-nan.case cli-inf.case
                           cli-rounded-stall.case cli-rounded-clock.case
                           cli-max-window-boundary.case cli-over-window-boundary.case)
    configure_file("${fixture_dir}/${case_name}" "${case_dir}/${case_name}" COPYONLY)
endforeach()
set(valid_case "${case_dir}/cli-nonbinary-tail.case")

function(run_cli output_prefix command case_path)
    execute_process(COMMAND "${cli}" ${command} "${case_path}"
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    set(${output_prefix}_result "${result}" PARENT_SCOPE)
    set(${output_prefix}_stdout "${stdout}" PARENT_SCOPE)
    set(${output_prefix}_stderr "${stderr}" PARENT_SCOPE)
endfunction()

run_cli(validate validate "${valid_case}")
if(NOT validate_result STREQUAL "0" OR NOT validate_stdout MATCHES "(^|\n)valid=true(\n|$)")
    message(FATAL_ERROR "CLI validate failed for a case path containing spaces.\nexit=${validate_result}\n${validate_stdout}\n${validate_stderr}")
endif()

run_cli(plan plan "${valid_case}")
if(NOT plan_result STREQUAL "0")
    message(FATAL_ERROR "CLI plan failed.\nexit=${plan_result}\n${plan_stdout}\n${plan_stderr}")
endif()
foreach(field IN ITEMS "built=true" "available=true" "participant=reference"
                      "entities=2" "steps=3" "field.x.dtype=${expected_dtype}"
                      "field.q.dtype=${expected_dtype}" "field.x.layout=LayoutRight"
                      "field.q.layout=LayoutRight" "layout_conversion=none"
                      "permissions.accepted=read-only leases"
                      "copy.initialize=host input to stable_ids and x and q plus history seed"
                      "copy.step=accepted history to independent next history"
                      "synchronize.step=Serial fence before host diagnostics and publication")
    string(FIND "${plan_stdout}" "${field}" field_position)
    if(field_position EQUAL -1)
        message(FATAL_ERROR "CLI plan omits '${field}'.\n${plan_stdout}")
    endif()
endforeach()
set(run_command run)
if(plan_stdout MATCHES "(^|\n)validated=false(\n|$)")
    run_cli(refused run "${valid_case}")
    if(NOT refused_result STREQUAL "2" OR NOT refused_stderr MATCHES
       "reference Serial profile requires validation evidence for this build")
        message(FATAL_ERROR "Unvalidated profile was not explicitly refused.\nexit=${refused_result}\n${refused_stdout}\n${refused_stderr}")
    endif()
    list(APPEND run_command --fixture)
elseif(NOT plan_stdout MATCHES "(^|\n)validated=true(\n|$)")
    message(FATAL_ERROR "CLI plan does not distinguish validation evidence.\n${plan_stdout}")
endif()
run_cli(run "${run_command}" "${valid_case}")
if(NOT run_result STREQUAL "0" OR
   (NOT run_command STREQUAL "run" AND NOT run_stderr MATCHES "running controlled reference fixture"))
    message(FATAL_ERROR "CLI reference execution failed.\nexit=${run_result}\n${run_stdout}\n${run_stderr}")
endif()
execute_process(COMMAND "${oracle}" RESULT_VARIABLE oracle_result
    OUTPUT_VARIABLE oracle_stdout ERROR_VARIABLE oracle_stderr)
if(NOT oracle_result STREQUAL "0")
    message(FATAL_ERROR "Private direct-library oracle failed.\n${oracle_stdout}\n${oracle_stderr}")
endif()
if(NOT run_stdout STREQUAL oracle_stdout)
    message(FATAL_ERROR "CLI output differs from direct Simulation/Reference calls.\nCLI:\n${run_stdout}\nOracle:\n${oracle_stdout}")
endif()

if(UNIX)
    foreach(command IN ITEMS validate plan "${run_command}")
        # Open stdout read-only so even a buffered write fails on flush.
        execute_process(COMMAND /bin/sh -c "exec \"$@\" 1</dev/null" nexora-output-failure
            "${cli}" ${command} "${valid_case}"
            RESULT_VARIABLE result ERROR_VARIABLE stderr)
        if(NOT result STREQUAL "1")
            message(FATAL_ERROR "${command} did not report stdout failure with exit 1.\nexit=${result}\n${stderr}")
        endif()
    endforeach()
endif()

function(expect_shared_rejection fixture)
    set(case_path "${case_dir}/${fixture}")
    set(reference_diagnostic "")
    foreach(command IN ITEMS validate plan run)
        run_cli(rejected "${command}" "${case_path}")
        if(NOT rejected_result STREQUAL "2" OR rejected_stderr STREQUAL "")
            message(FATAL_ERROR "${command} did not reject ${fixture} with exit 2 and a diagnostic.\nexit=${rejected_result}\n${rejected_stdout}\n${rejected_stderr}")
        endif()
        if(reference_diagnostic STREQUAL "")
            set(reference_diagnostic "${rejected_stderr}")
        elseif(NOT rejected_stderr STREQUAL reference_diagnostic)
            message(FATAL_ERROR "${command} diagnostic differs for ${fixture}.\nExpected: ${reference_diagnostic}\nActual: ${rejected_stderr}")
        endif()
    endforeach()
    if(NOT reference_diagnostic MATCHES "${fixture}:[0-9]+:")
        message(FATAL_ERROR "Diagnostic for ${fixture} does not identify source file and line: ${reference_diagnostic}")
    endif()
    if(ARGC GREATER 1 AND NOT reference_diagnostic MATCHES "${ARGV1}")
        message(FATAL_ERROR "Expected '${ARGV1}' in the diagnostic for ${fixture}: ${reference_diagnostic}")
    endif()
endfunction()
expect_shared_rejection(cli-invalid.case)
expect_shared_rejection(cli-underflow.case)
expect_shared_rejection(cli-invalid-sign.case)
expect_shared_rejection(cli-too-many-steps.case "case step count exceeds version 1 limit of 1000000 steps")
expect_shared_rejection(cli-duplicate.case)
expect_shared_rejection(cli-unknown-version.case)
expect_shared_rejection(cli-unsupported-execution.case)
expect_shared_rejection(cli-invalid-interval.case)
expect_shared_rejection(cli-quantum.case)
expect_shared_rejection(cli-nan.case)
expect_shared_rejection(cli-inf.case)
expect_shared_rejection(cli-rounded-stall.case)
expect_shared_rejection(cli-over-window-boundary.case)
foreach(field IN ITEMS x q)
    set(nonfinite_case "cli-nonfinite-${field}.case")
    file(WRITE "${case_dir}/${nonfinite_case}"
        "version=1\nfixture=reference\nend=1\nstep=0.25\n${field}=NaN\n")
    expect_shared_rejection("${nonfinite_case}" ":5: reference initial values must be finite")
endforeach()

run_cli(boundary plan "${case_dir}/cli-max-window-boundary.case")
if(NOT boundary_result STREQUAL "0" OR NOT boundary_stdout MATCHES "(^|\n)steps=1000000(\n|$)")
    message(FATAL_ERROR "The exact version 1 schedule limit should be accepted.\n${boundary_stdout}\n${boundary_stderr}")
endif()

if(expected_precision STREQUAL "float")
    set(opposite_precision double)
    set(max_real 3.40282347e38)
    set(overflow_seed 1.02e38)
else()
    set(opposite_precision float)
    set(max_real 1.7976931348623157e308)
    set(overflow_seed 5.39e307)
endif()
set(precision_case cli-wrong-precision.case)
file(WRITE "${case_dir}/${precision_case}"
    "version=1\nfixture=reference\nprecision=${opposite_precision}\nend=1\nstep=0.25\n")
expect_shared_rejection("${precision_case}" "precision must match this build")

set(max_case "${case_dir}/cli-max-real.case")
file(WRITE "${max_case}"
    "version=1\nfixture=reference\nexecution=serial\nstart=0\nend=1e-5\nstep=1e-5\nlambda=1\nids=17\nx=${max_real}\nq=0\n")
run_cli(max_validate validate "${max_case}")
if(NOT max_validate_result STREQUAL "0" OR NOT max_validate_stdout MATCHES "valid=true")
    message(FATAL_ERROR "Maximum finite build-real case failed validation.\n${max_validate_stdout}\n${max_validate_stderr}")
endif()
run_cli(max_run "${run_command}" "${max_case}")
if(NOT max_run_result STREQUAL "0")
    message(FATAL_ERROR "Maximum finite build-real case failed execution.\n${max_run_stdout}\n${max_run_stderr}")
endif()
string(TOLOWER "${max_run_stdout}" max_run_lower)
if(max_run_lower MATCHES "nan|inf" OR NOT max_run_stdout MATCHES "accepted_steps=1"
   OR NOT max_run_stdout MATCHES "entity.0.x=" OR NOT max_run_stdout MATCHES "entity.0.q=")
    message(FATAL_ERROR "Maximum finite build-real case emitted invalid output.\n${max_run_stdout}")
endif()

# With dt=0.5, the first increment is half of x. The second is one quarter
# of the initial x, which overflows q once added to its already accepted value.
set(late_failure_case "${case_dir}/cli-late-numerical-failure.case")
file(WRITE "${late_failure_case}"
    "version=1\nfixture=reference\nexecution=serial\nstart=0\nend=1.5\nstep=0.5\nlambda=1\nids=17\nx=${max_real}\nq=${overflow_seed}\n")
run_cli(failure_validate validate "${late_failure_case}")
if(NOT failure_validate_result STREQUAL "0")
    message(FATAL_ERROR "Late numerical failure case did not pass preflight.\n${failure_validate_stderr}")
endif()
run_cli(failure "${run_command}" "${late_failure_case}")
if(NOT failure_result STREQUAL "1" OR NOT failure_stdout STREQUAL "" OR
   NOT failure_stderr MATCHES "step 2:.*nonfinite" OR failure_stderr MATCHES "finalize failed")
    message(FATAL_ERROR "CLI failed to stop at the late numerical error.\nexit=${failure_result}\n${failure_stdout}\n${failure_stderr}")
endif()

set(rounded_case "${case_dir}/cli-rounded-clock.case")
run_cli(rounded "${run_command}" "${rounded_case}")
foreach(field IN ITEMS "accepted_time=9007199254740996" "accepted_steps=1"
                      "entity.0.x=0.625" "entity.0.q=0.375")
    string(FIND "${rounded_stdout}" "${field}" rounded_field_position)
    if(NOT rounded_result STREQUAL "0" OR rounded_field_position EQUAL -1)
        message(FATAL_ERROR "Rounded-clock case differs from the fixed 3-unit authoritative step at '${field}'.\n${rounded_stdout}\n${rounded_stderr}")
    endif()
endforeach()

execute_process(COMMAND "${cli}" --version RESULT_VARIABLE version_result
    OUTPUT_VARIABLE version_stdout ERROR_VARIABLE version_stderr)
string(STRIP "${version_stdout}" version_stdout)
if(NOT version_result STREQUAL "0" OR NOT version_stdout STREQUAL "Nexora 0.2.0")
    message(FATAL_ERROR "Existing --version diagnostic failed.\n${version_stdout}\n${version_stderr}")
endif()
execute_process(COMMAND "${cli}" inspect-build RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_stdout ERROR_VARIABLE build_stderr)
if(NOT build_result STREQUAL "0" OR NOT build_stdout MATCHES "project=Nexora"
   OR NOT build_stdout MATCHES "kokkos_execution_space=Serial")
    message(FATAL_ERROR "Existing inspect-build diagnostic failed.\n${build_stdout}\n${build_stderr}")
endif()
