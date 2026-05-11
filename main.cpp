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

    // MPI_Comm_split — grid algorithms vs data algorithms
    int color;
    if (choice == 1)
        color = 0;   // grid group
    else
        color = 1;   // data group

    MPI_Comm algoComm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &algoComm);

    int subRank, subSize;
    MPI_Comm_rank(algoComm, &subRank);
    MPI_Comm_size(algoComm, &subSize);

    if (rank == 0) {
        cout << "\n[Main] MPI_Comm_split:\n";
        cout << "  color 0 = Grid algorithms (Heat Diffusion)\n";
        cout << "  color 1 = Data algorithms (Prefix Sum)\n";
        cout << "  All " << size << " processes ? color " << color << "\n\n";
    }

    switch (choice) {
    case 1: runHeatDiffusion(subRank, subSize, algoComm); break;
    case 2: runPrefixSum(subRank, subSize, algoComm);     break;
    default:
        if (rank == 0) cout << "Invalid choice.\n";
    }

    MPI_Comm_free(&algoComm);
    MPI_Finalize();
    return 0;
}