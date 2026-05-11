#include "heat_diffusion.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

void runHeatDiffusion(int rank, int size, MPI_Comm comm) {
    const int iterations = 5;
    int rows = 0, cols = 0;
    vector<vector<double>> fullGrid;

    // ?? Step 1: Rank 0 loads grid from file ????????????????????????
    if (rank == 0) {
        ifstream f("grid_input.bin", ios::binary);
        if (!f) {
            // File not found ? use default 8x8
            cout << "[Heat] File not found. Using default 8x8 grid.\n";
            rows = 8;
            cols = 8;
            fullGrid.assign(rows, vector<double>(cols, 25.0));
        }
        else {
            f.read(reinterpret_cast<char*>(&rows), sizeof(int));
            f.read(reinterpret_cast<char*>(&cols), sizeof(int));
            fullGrid.assign(rows, vector<double>(cols));
            for (int i = 0; i < rows; i++)
                f.read(reinterpret_cast<char*>(fullGrid[i].data()),
                    cols * sizeof(double));
            f.close();
            cout << "[Heat] Loaded grid_input.bin ("
                << rows << "x" << cols << ")\n";
        }

        // Set heat source on top row
        for (int j = 0; j < cols; j++)
            fullGrid[0][j] = 100.0;
    }

    // ?? Step 2: Broadcast dimensions to all processes ??????????????
    MPI_Bcast(&rows, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cols, 1, MPI_INT, 0, comm);

    // ?? Step 3: Decide how many rows each process gets ?????????????
    int localRows = rows / size + (rank < rows % size ? 1 : 0);
    int rowStart = rank * (rows / size) + min(rank, rows % size);

    if (rank == 0) {
        cout << "[Heat] Rows: " << rows << "  Cols: " << cols << "\n";
        cout << "[Heat] Processes: " << size << "\n";
        cout << "[Heat] Row distribution:\n";
        for (int p = 0; p < size; p++) {
            int r = rows / size + (p < rows % size ? 1 : 0);
            int s = p * (rows / size) + min(p, rows % size);
            cout << "  Process " << p << ": rows "
                << s << " to " << s + r - 1
                << " (" << r << " rows)\n";
        }
    }

    // ?? Step 4: Scatter rows of fullGrid to each process ???????????
    // Rank 0 sends each process its chunk row by row
    vector<vector<double>> grid(localRows + 2, vector<double>(cols, 25.0));
    vector<vector<double>> newGrid = grid;

    if (rank == 0) {
        // Copy own rows
        for (int i = 0; i < localRows; i++)
            grid[i + 1] = fullGrid[i];

        // Send rows to other processes
        for (int p = 1; p < size; p++) {
            int r_p = rows / size + (p < rows % size ? 1 : 0);
            int s_p = p * (rows / size) + min(p, rows % size);
            for (int i = 0; i < r_p; i++)
                MPI_Send(fullGrid[s_p + i].data(), cols, MPI_DOUBLE,
                    p, 0, comm);
        }
    }
    else {
        // Receive rows from rank 0
        for (int i = 0; i < localRows; i++)
            MPI_Recv(grid[i + 1].data(), cols, MPI_DOUBLE,
                0, 0, comm, MPI_STATUS_IGNORE);
    }

    // ?? Step 5: Run iterations (your exact original code) ??????????
    for (int step = 0; step < iterations; step++) {
        int reqCount = 0;
        MPI_Request requests[4];

        if (rank > 0) {
            MPI_Isend(grid[1].data(), cols, MPI_DOUBLE,
                rank - 1, 0, comm, &requests[reqCount++]);
            MPI_Irecv(grid[0].data(), cols, MPI_DOUBLE,
                rank - 1, 1, comm, &requests[reqCount++]);
        }
        if (rank < size - 1) {
            MPI_Isend(grid[localRows].data(), cols, MPI_DOUBLE,
                rank + 1, 1, comm, &requests[reqCount++]);
            MPI_Irecv(grid[localRows + 1].data(), cols, MPI_DOUBLE,
                rank + 1, 0, comm, &requests[reqCount++]);
        }

        MPI_Waitall(reqCount, requests, MPI_STATUSES_IGNORE);

        for (int i = 1; i <= localRows; i++)
            for (int j = 1; j < cols - 1; j++)
                newGrid[i][j] = (grid[i - 1][j] +
                    grid[i + 1][j] +
                    grid[i][j - 1] +
                    grid[i][j + 1]) / 4.0;

        grid = newGrid;
    }
    // ?? Step 6: Flatten local grid ?????????????????????????????????

    vector<double> flat(localRows* cols);
    for (int i = 1; i <= localRows; i++)
        for (int j = 0; j < cols; j++)
            flat[(i - 1) * cols + j] = grid[i][j];

    // ?? Step 7: Build counts and displacements ?????????????????????
    vector<int> recvCounts(size), displs(size);
    for (int p = 0; p < size; p++) {
        int r = rows / size + (p < rows % size ? 1 : 0);
        int s = p * (rows / size) + min(p, rows % size);
        recvCounts[p] = r * cols;
        displs[p] = s * cols;
    }

    // ?? Step 8: Gather all rows to rank 0 ??????????????????????????
    vector<double> full;
    if (rank == 0)
        full.resize(rows * cols);

    MPI_Gatherv(flat.data(), localRows * cols, MPI_DOUBLE, rank == 0 ? full.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // ?? Step 9: Print full final grid ??????????????????????????????
    if (rank == 0) {
        cout << "\n[Heat] Final Grid:\n";
        for (int i = 0; i < rows; i++) {
            cout << "Row " << i << ": ";
            for (int j = 0; j < cols; j++)
                cout << full[i * cols + j] << " ";
            cout << "\n";
        }
        cout << "\n[Heat] Done!\n";
    }

}