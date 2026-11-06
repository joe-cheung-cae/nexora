#include <mpi.h>

#include <iostream>

int mpi_smoke(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed.\n";
        return 1;
    }

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "MPI smoke requires exactly two ranks, got " << size << ".\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int contribution = provided >= MPI_THREAD_FUNNELED ? rank + 1 : 0;
    int sum = 0;
    MPI_Allreduce(&contribution, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int finalized = MPI_Finalize();

    if (sum != 3 || finalized != MPI_SUCCESS) {
        if (rank == 0) {
            std::cerr << "MPI smoke failed: expected MPI_THREAD_FUNNELED on both ranks, "
                         "a rank sum of 3, and successful finalization. Rank sum was "
                      << sum << ".\n";
        }
        return 1;
    }
    if (rank == 0) {
        std::cout << "MPI smoke passed: 2 ranks, MPI_THREAD_FUNNELED, rank sum 3.\n";
    }
    return 0;
}
