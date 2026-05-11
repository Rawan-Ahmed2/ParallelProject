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
        cout << "2. prefix Sum\n";
        cin >> choice;
    }

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ?? MPI_Comm_split ?????????????????????????????????????????????
    // Split processes into 2 groups based on choice:
    //
    // Grid algorithms  (Heat, GameOfLife) ? color = 0
    // Data algorithms  (Matrix)           ? color = 1
    //
    // Even if only one algorithm runs, the split still organizes
    // processes into logical groups showing structured communication
    int color;
    if (choice == 1 || choice == 3)
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
        cout << "  color 0 = Grid  algorithms (Heat Diffusion, Game of Life)\n";
        cout << "  color 1 = Data  algorithms (Matrix Multiplication)\n";
        cout << "  All " << size << " processes ? color " << color << "\n";
        cout << "  subSize = " << subSize << "\n\n";
    }

    switch (choice) {
    case 1:
        runHeatDiffusion(subRank, subSize, algoComm);
        break;
    case 2:
		runPrefixSum(subRank, subSize, algoComm);
        break;

       default:
        if (rank == 0) cout << "Invalid choice.\n";
    }

    MPI_Comm_free(&algoComm);
    MPI_Finalize();
    return 0;
}