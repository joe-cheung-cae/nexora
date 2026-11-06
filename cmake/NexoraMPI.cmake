find_package(MPI REQUIRED COMPONENTS C)

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_C)
set(CMAKE_REQUIRED_DEFINITIONS -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1)
unset(NEXORA_MPI_C_API_FROM_CXX_WORKS CACHE)
check_cxx_source_compiles([=[
    #include <mpi.h>
    int main(int argc, char** argv) {
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        return MPI_Finalize();
    }
]=] NEXORA_MPI_C_API_FROM_CXX_WORKS)
cmake_pop_check_state()

if(NOT NEXORA_MPI_C_API_FROM_CXX_WORKS)
    message(FATAL_ERROR "MPI::MPI_C cannot be compiled and linked from the selected C++ toolchain.")
endif()
