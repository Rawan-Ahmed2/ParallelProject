# Deadlock Handling and Performance Analysis

## Deadlock Scenario

The heat diffusion algorithm exchanges boundary rows between neighboring MPI processes. A real deadlock can happen if every process tries to send its boundary rows with blocking `MPI_Send` before posting the matching `MPI_Recv`.

Problematic pattern:

```cpp
if (rank > 0) {
    MPI_Send(topRow, cols, MPI_DOUBLE, rank - 1, 0, comm);
}
if (rank < size - 1) {
    MPI_Send(bottomRow, cols, MPI_DOUBLE, rank + 1, 1, comm);
}
if (rank > 0) {
    MPI_Recv(topGhost, cols, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);
}
if (rank < size - 1) {
    MPI_Recv(bottomGhost, cols, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);
}
```

## Why It Happens

With blocking communication, `MPI_Send` may wait until the matching receive is posted. If process 0 sends to process 1 while process 1 is also trying to send to process 0, both processes can wait forever. In a row-based stencil, the same issue can spread across all neighboring ranks because every process is waiting inside `MPI_Send` and nobody reaches `MPI_Recv`.

## When It Happens

This deadlock can happen during the halo exchange step of heat diffusion when:

- The program runs with two or more MPI processes.
- Neighboring processes exchange rows at the same time.
- Blocking sends are posted before receives.
- MPI does not buffer the messages enough to let the sends complete.

It is more likely with larger row messages because MPI implementations are less likely to internally buffer large messages.

## Corrected Solution

The implemented heat diffusion code avoids this deadlock by using non-blocking communication:

```cpp
if (rank > 0) {
    MPI_Isend(grid[1].data(), cols, MPI_DOUBLE, rank - 1, 0, comm, &requests[reqCount++]);
    MPI_Irecv(grid[0].data(), cols, MPI_DOUBLE, rank - 1, 1, comm, &requests[reqCount++]);
}
if (rank < size - 1) {
    MPI_Isend(grid[localRows].data(), cols, MPI_DOUBLE, rank + 1, 1, comm, &requests[reqCount++]);
    MPI_Irecv(grid[localRows + 1].data(), cols, MPI_DOUBLE, rank + 1, 0, comm, &requests[reqCount++]);
}

MPI_Waitall(reqCount, requests, MPI_STATUSES_IGNORE);
```

This works because sends and receives are posted before the program waits. Each process makes its receive buffers available, so neighboring processes can complete their matching sends.

Another valid correction would be to order blocking operations by rank, for example even ranks send first while odd ranks receive first, then reverse the order. The non-blocking solution is better for the stencil algorithm because it naturally supports neighbor exchange and can overlap communication with computation in larger versions.

## Performance Timing

The system now measures performance with `MPI_Wtime` in both algorithms:

- `Max compute time`: the slowest process time for the main parallel algorithm phase.
- `Max total time`: the slowest process time including file loading, data distribution, computation, and gathering output.

The maximum process time is reported because a parallel program finishes only when the slowest process finishes.

## Suggested Test Configurations

Run the project with different process counts:

```text
mpiexec -n 2 x64\Debug\Parallel.exe
mpiexec -n 4 x64\Debug\Parallel.exe
mpiexec -n 8 x64\Debug\Parallel.exe
```

For each run, select:

- `1` for Heat Diffusion
- `2` for Prefix Sum

## Sample Benchmark Results

These results were collected on May 11, 2026 using `grid_input.bin` with a `1000x1000` grid.

| Algorithm | Processes | Max compute time | Max total time |
| --- | ---: | ---: | ---: |
| Heat Diffusion | 2 | 0.152329 s | 6.36735 s |
| Heat Diffusion | 4 | 0.0860792 s | 5.99807 s |
| Prefix Sum | 2 | 0.0059079 s | 0.0320486 s |
| Prefix Sum | 4 | 0.0084791 s | 0.0368758 s |

## Observations

Heat Diffusion:

- Increasing the number of processes reduces the number of rows computed by each process.
- Communication cost increases because processes exchange ghost rows with neighbors every iteration.
- For small iteration counts, communication and gather overhead can dominate the total runtime.
- For larger grids or more iterations, parallelism should become more useful because there is more computation per process.
- In the sample run, compute time improved from `0.152329 s` with 2 processes to `0.0860792 s` with 4 processes.
- Total time improved only slightly because the program still performs file loading, manual distribution, gathering, and printing the final grid.

Prefix Sum:

- Local prefix computation scales with the number of elements assigned to each process.
- The algorithm still needs global coordination to gather local totals, compute offsets, scatter offsets, and gather the final output.
- Prefix Sum now flattens the full `1000x1000` matrix into a 1,000,000-element array.
- With small arrays, communication overhead may be larger than the benefit from adding more processes.
- Prefix Sum still requires global coordination, so communication overhead can affect scaling even with the larger flattened input.

Communication Impact:

- Blocking communication is used for initial manual data distribution.
- Non-blocking communication is used for heat diffusion neighbor exchange to avoid deadlock.
- Collective communication is used for broadcasting choices/dimensions and gathering final results.
- More processes generally means smaller local chunks but more communication and synchronization points.
