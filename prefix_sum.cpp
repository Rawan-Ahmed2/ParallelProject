#include "prefix_sum.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

static void printPreview(const vector<double>& values, int previewCount) {
    int count = min(previewCount, static_cast<int>(values.size()));
    for (int i = 0; i < count; i++)
        cout << values[i] << " ";
    if (static_cast<int>(values.size()) > count)
        cout << "... ";
}

void runPrefixSum(int rank, int size, MPI_Comm comm) {

    int n = 0;
    vector<double> fullArray;
    double totalStart = MPI_Wtime();

    if (rank == 0) {
        ifstream f("grid_input.bin", ios::binary);
        if (!f) {
            cout << "[Prefix] File not found. Using default array.\n";
            n = 8;
            fullArray = { 3, 1, 4, 1, 5, 9, 2, 6 };
        }
        else {
            int rows = 0, cols = 0;

            f.read(reinterpret_cast<char*>(&rows), sizeof(int));
            f.read(reinterpret_cast<char*>(&cols), sizeof(int));
            
            n = rows * cols;
            fullArray.resize(n);
            f.read(reinterpret_cast<char*>(fullArray.data()), n * sizeof(double));
            f.close();

            cout << "[Prefix] Original matrix size: " << rows << "x" << cols << "\n";
            cout << "[Prefix] Flattened array size: " << n << " elements\n";
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);

    int localN = n / size + (rank < n % size ? 1 : 0);
    int elemStart = rank * (n / size) + min(rank, n % size);

    vector<double> localArr(localN);

    if (rank == 0) {
        for (int i = 0; i < localN; i++)
            localArr[i] = fullArray[i];

        for (int p = 1; p < size; p++) {
            int r_p = n / size + (p < n % size ? 1 : 0);
            int s_p = p * (n / size) + min(p, n % size);
            cout << "Send " << r_p << " elements starting from " << fullArray[s_p] << " to process " << p << endl;
            MPI_Send(&fullArray[s_p], r_p, MPI_DOUBLE, p, 0, comm);
        }
    }
    else {
        MPI_Recv(localArr.data(), localN, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(comm);
    double computeStart = MPI_Wtime();

    vector<double> localPrefix(localN);
    double localTotal = 0.0;
    if (localN > 0) {
        localPrefix[0] = localArr[0];
        for (int i = 1; i < localN; i++)
            localPrefix[i] = localPrefix[i - 1] + localArr[i];
        localTotal = localPrefix[localN - 1];
    }

    for (int p = 0; p < size; p++) {
        if (rank == p) {
            cout << "[Prefix] Process " << rank << " local prefix: ";
            for (int i = 0; i < localN; i++)
                if (i < 10)
                    cout << localPrefix[i] << " ";
                else if (i == 10)
                    cout << "... ";
            cout << "(local total = " << localTotal << ")\n";
            cout.flush();
        }
        MPI_Barrier(comm);
    }

    vector<double> allTotals(size);
    MPI_Gather(&localTotal, 1, MPI_DOUBLE, allTotals.data(), 1, MPI_DOUBLE, 0, comm);

    vector<double> offsets(size, 0.0);
    if (rank == 0) {
        offsets[0] = 0.0;
        for (int p = 1; p < size; p++)
            offsets[p] = offsets[p - 1] + allTotals[p - 1];
    }

    double myOffset = 0.0;
    MPI_Scatter(offsets.data(), 1, MPI_DOUBLE, &myOffset, 1, MPI_DOUBLE, 0, comm);

    for (int i = 0; i < localN; i++)
        localPrefix[i] += myOffset;

    double computeEnd = MPI_Wtime();
    double localComputeTime = computeEnd - computeStart;
    double maxComputeTime = 0.0;
    MPI_Reduce(&localComputeTime, &maxComputeTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    for (int p = 0; p < size; p++) {
        if (rank == p) {
            cout << "[Prefix] Process " << rank << " final prefix: ";
            for (int i = 0; i < localN; i++)
                if (i < 10)
                    cout << localPrefix[i] << " ";
                else if (i == 10)
                    cout << "... ";
            cout << "\n";
            cout.flush();
        }
        MPI_Barrier(comm);
    }

    vector<int> recvCounts(size), displs(size);
    for (int p = 0; p < size; p++) {
        int r = n / size + (p < n % size ? 1 : 0);
        int s = p * (n / size) + min(p, n % size);
        recvCounts[p] = r;
        displs[p] = s;
    }

    vector<double> full;
    if (rank == 0)
        full.resize(n);

    MPI_Gatherv(
        localPrefix.data(),
        localN,
        MPI_DOUBLE,
        rank == 0 ? full.data() : nullptr,
        recvCounts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        comm
    );

    if (rank == 0) {
        cout << "\n[Prefix] Input preview  : ";
        printPreview(fullArray, 20);

        cout << "\n[Prefix] Result preview : ";
        printPreview(full, 20);

        if (!full.empty())
            cout << "\n[Prefix] Final prefix total: " << full.back();

        cout << "\n\n[Prefix] Done!\n";
    }

    double totalEnd = MPI_Wtime();
    double localTotalTime = totalEnd - totalStart;
    double maxTotalTime = 0.0;
    MPI_Reduce(&localTotalTime, &maxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        cout << "[Prefix] Timing:\n";
        cout << "  Max compute time: " << maxComputeTime << " seconds\n";
        cout << "  Max total time  : " << maxTotalTime << " seconds\n";
    }
}
