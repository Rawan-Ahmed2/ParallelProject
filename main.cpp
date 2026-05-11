#include <mpi.h>
#include <iostream>
#include "heat_diffusion.h"
#include "prefix_sum.h"
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int choice = 0;

    if (rank == 0) {
        cout << "Choose Algorithm:\n";
        cout << "1. Heat Diffusion\n";
        cout << "2. Prefix Sum\n";
        cin >> choice;
    }

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    switch (choice) {
    case 1: runHeatDiffusion(rank, size, MPI_COMM_WORLD); break;
    case 2: runPrefixSum(rank, size, MPI_COMM_WORLD);     break;
    default:
        if (rank == 0) cout << "Invalid choice.\n";
    }

    MPI_Finalize();
    return 0;
}