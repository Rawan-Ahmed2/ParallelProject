# Advanced Parallel Grid Processing with MPI

## 1. Project Overview

This project is a parallel distributed processing system implemented in C++ using MPI. The system processes grid-based and data-based workloads by dividing the input data across multiple MPI processes, exchanging information between processes, and collecting the final result on rank 0.

The project supports runtime algorithm selection. When the program starts, rank 0 displays a menu and the user chooses which parallel algorithm to run:

```text
1. Heat Diffusion
2. Prefix Sum
```

The selected choice is broadcast to all processes using `MPI_Bcast`, so every process runs the same selected algorithm.

Main source files:

| File | Purpose |
| --- | --- |
| `main.cpp` | Initializes MPI, selects the algorithm, broadcasts the choice, and creates the algorithm communicator. |
| `heat_diffusion.cpp` | Implements the parallel Heat Diffusion stencil algorithm. |
| `prefix_sum.cpp` | Implements the parallel Prefix Sum algorithm. |
| `generate_grid.cpp` | Generates the binary input file `grid_input.bin`. |
| `grid_input.bin` | Large binary input grid used by the algorithms. |

## 2. Implemented Algorithms

### 2.1 Heat Diffusion

Category: **Grid / Spatial**

Heat Diffusion is implemented as a stencil computation over a 2D grid. The input grid is loaded from `grid_input.bin`, which contains a `1000 x 1000` grid of double values. Rank 0 loads the full grid, then distributes rows to the other processes.

Each process receives a horizontal block of rows. Two extra ghost rows are added locally:

- One ghost row above the local block
- One ghost row below the local block

During each iteration, neighboring MPI processes exchange boundary rows. Each cell is updated using the average of its four direct neighbors:

```cpp
newGrid[i][j] = (
    grid[i - 1][j] +
    grid[i + 1][j] +
    grid[i][j - 1] +
    grid[i][j + 1]
) / 4.0;
```

After all iterations finish, the final grid is gathered back to rank 0 using `MPI_Gatherv`.

### 2.2 Prefix Sum

Category: **Data / Computation**

Prefix Sum is implemented as a distributed scan operation. The program loads the full `1000 x 1000` matrix from `grid_input.bin` and flattens it into a single one-dimensional array with `1,000,000` elements.

The algorithm works in stages:

1. Rank 0 loads the full matrix and flattens it into one input array.
2. The input size is broadcast to all processes.
3. Rank 0 distributes chunks of the array using blocking communication.
4. Each process computes a local prefix sum.
5. Each process sends its local total to rank 0 using `MPI_Gather`.
6. Rank 0 computes offsets for each process.
7. Offsets are sent back using `MPI_Scatter`.
8. Each process adds its offset to its local prefix result.
9. The final prefix array is gathered using `MPI_Gatherv`.

The Prefix Sum implementation also handles uneven data sizes and avoids crashes when a process receives zero elements.

## 3. Runtime Algorithm Selection

Runtime selection is implemented in `main.cpp`.

Rank 0 reads the user choice:

```cpp
if (rank == 0) {
    cout << "Choose Algorithm:\n";
    cout << "1. Heat Diffusion\n";
    cout << "2. Prefix Sum\n";
    cin >> choice;
}
```

Then the selected algorithm is broadcast to all processes:

```cpp
MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

This ensures that all MPI ranks execute the same selected algorithm.

## 4. Data Distribution and Scalability

Both algorithms support uneven data sizes using remainder-based distribution.

For Heat Diffusion, rows are divided using:

```cpp
int localRows = rows / size + (rank < rows % size ? 1 : 0);
int rowStart = rank * (rows / size) + min(rank, rows % size);
```

For Prefix Sum, elements are divided using:

```cpp
int localN = n / size + (rank < n % size ? 1 : 0);
int elemStart = rank * (n / size) + min(rank, n % size);
```

This allows the system to run correctly even when the number of rows or elements is not evenly divisible by the number of MPI processes.

The project supports running with different process counts, such as:

```text
mpiexec -n 2 x64\Debug\Parallel.exe
mpiexec -n 4 x64\Debug\Parallel.exe
```

## 5. Process Organization

The project uses `MPI_Comm_split` in `main.cpp` to create an algorithm-specific communicator.

```cpp
int color;
if (choice == 1)
    color = 0;
else
    color = 1;

MPI_Comm algoComm;
MPI_Comm_split(MPI_COMM_WORLD, color, rank, &algoComm);
```

The new communicator `algoComm` is passed into the selected algorithm:

```cpp
runHeatDiffusion(subRank, subSize, algoComm);
runPrefixSum(subRank, subSize, algoComm);
```

This organizes MPI communication by algorithm type:

- Color `0`: Grid / Spatial algorithm group
- Color `1`: Data / Computation algorithm group

All communication inside each algorithm uses the selected algorithm communicator instead of directly using `MPI_COMM_WORLD`.

## 6. Communication Strategies

The system uses multiple MPI communication strategies.

### 6.1 Blocking Communication

Blocking communication is used during manual data distribution from rank 0.

In Heat Diffusion:

```cpp
MPI_Send(fullGrid[s_p + i].data(), cols, MPI_DOUBLE, p, 0, comm);
MPI_Recv(grid[i + 1].data(), cols, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
```

In Prefix Sum:

```cpp
MPI_Send(&fullArray[s_p], r_p, MPI_DOUBLE, p, 0, comm);
MPI_Recv(localArr.data(), localN, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
```

Blocking communication is simple and useful here because rank 0 controls the initial distribution order.

### 6.2 Non-Blocking Communication

Non-blocking communication is used in Heat Diffusion during ghost-row exchange:

```cpp
MPI_Isend(grid[1].data(), cols, MPI_DOUBLE, rank - 1, 0, comm, &requests[reqCount++]);
MPI_Irecv(grid[0].data(), cols, MPI_DOUBLE, rank - 1, 1, comm, &requests[reqCount++]);
MPI_Waitall(reqCount, requests, MPI_STATUSES_IGNORE);
```

This avoids deadlock and allows all processes to post sends and receives before waiting.

### 6.3 Neighbor Exchange Pattern

Heat Diffusion uses a neighbor exchange communication pattern. Each process communicates only with the process directly above it and the process directly below it.

For example:

- Process 1 exchanges boundary rows with process 0 and process 2.
- Process 0 exchanges only with process 1.
- The last process exchanges only with the previous process.

This pattern is appropriate for stencil-based grid processing.

### 6.4 Collective Communication

The system also uses collective MPI communication:

| MPI Function | Usage |
| --- | --- |
| `MPI_Bcast` | Broadcasts selected algorithm and input dimensions. |
| `MPI_Gather` | Gathers local totals in Prefix Sum. |
| `MPI_Scatter` | Sends prefix offsets back to processes. |
| `MPI_Gatherv` | Gathers uneven final results from all processes. |
| `MPI_Reduce` | Computes maximum runtime across processes. |
| `MPI_Barrier` | Synchronizes processes before timing sections. |

## 7. Deadlock Handling

### 7.1 Deadlock Scenario

A real deadlock can occur in the Heat Diffusion halo exchange step if all processes use blocking sends before posting receives.

Problematic version:

```cpp
if (rank > 0)
    MPI_Send(grid[1].data(), cols, MPI_DOUBLE, rank - 1, 0, comm);

if (rank < size - 1)
    MPI_Send(grid[localRows].data(), cols, MPI_DOUBLE, rank + 1, 1, comm);

if (rank > 0)
    MPI_Recv(grid[0].data(), cols, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);

if (rank < size - 1)
    MPI_Recv(grid[localRows + 1].data(), cols, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);
```

### 7.2 Why It Happens

Blocking `MPI_Send` may wait until the matching `MPI_Recv` is posted. If every process sends first, neighboring processes can all become stuck waiting for each other.

For example, with two processes:

1. Process 0 calls `MPI_Send` to process 1.
2. Process 1 calls `MPI_Send` to process 0.
3. Both processes wait for the other side to post `MPI_Recv`.
4. Neither process reaches the receive call.
5. The program deadlocks.

### 7.3 When It Happens

This can happen when:

- The program runs with two or more processes.
- Processes exchange boundary rows with neighbors.
- Blocking sends are posted before receives.
- The messages are large enough that MPI does not internally buffer them.

This scenario is especially relevant in Heat Diffusion because each process must exchange ghost rows during every iteration.

### 7.4 Corrected Solution

The implemented solution uses non-blocking communication:

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

This prevents deadlock because all sends and receives are posted before the program waits for completion.

## 8. Performance Measurement

Performance timing is implemented using `MPI_Wtime`.

Each algorithm reports:

- `Max compute time`
- `Max total time`

The maximum time across processes is calculated using `MPI_Reduce` with `MPI_MAX`:

```cpp
MPI_Reduce(&localTotalTime, &maxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
```

The maximum process time is used because the full parallel program cannot finish until the slowest process finishes.

## 9. Performance Results and Observations

### 9.1 Heat Diffusion Results

The Heat Diffusion algorithm was tested using `grid_input.bin`, which contains a `1000 x 1000` grid.

| Processes | Total Time |
| ---: | ---: |
| 2 | 5.7 seconds |
| 4 | 5.9 seconds |

Observation:

Using 4 processes was slightly slower than using 2 processes. This is normal for this workload because the program only performs a small number of heat diffusion iterations. Adding more processes reduces the number of rows per process, but it also increases communication, synchronization, and gathering overhead.

The final grid is also gathered and printed by rank 0, which adds a large output cost. Because of this, the total runtime is affected by more than just computation.

### 9.2 Prefix Sum Observations

Prefix Sum processes the full input grid as a flattened array, so the input size is currently 1,000,000 elements.

Because Prefix Sum requires global coordination between process chunks, communication overhead still affects runtime. However, flattening the full matrix gives the algorithm a much larger workload than using only one row, so the computation is more meaningful for MPI benchmarking.

Prefix Sum requires several global communication steps:

- Gather local totals
- Compute offsets
- Scatter offsets
- Gather final results

For smaller input sizes, these communication steps can cost more than the computation saved by dividing the work among more processes. With the flattened 1,000,000-element input, Prefix Sum has more computation, but it still depends strongly on communication efficiency.

### 9.3 Communication Impact

The system demonstrates that parallel speedup depends on the balance between computation and communication.

Heat Diffusion has more computation because it works on a full 2D grid, but it also requires neighbor communication every iteration.

Prefix Sum has less computation and more global coordination compared to its workload size. Therefore, Prefix Sum is more sensitive to communication overhead.

Overall:

- More processes reduce local computation.
- More processes increase communication and synchronization.
- Small workloads may run slower with more processes.
- Larger workloads or more iterations are more likely to benefit from parallel execution.

## 10. How to Run the Project

Build the project in Visual Studio using the `Debug|x64` configuration.

Run Heat Diffusion with 2 processes:

```text
mpiexec -n 2 x64\Debug\Parallel.exe
```

Then select:

```text
1
```

Run Heat Diffusion with 4 processes:

```text
mpiexec -n 4 x64\Debug\Parallel.exe
```

Then select:

```text
1
```

Run Prefix Sum with 2 processes:

```text
mpiexec -n 2 x64\Debug\Parallel.exe
```

Then select:

```text
2
```

Run Prefix Sum with 4 processes:

```text
mpiexec -n 4 x64\Debug\Parallel.exe
```

Then select:

```text
2
```

## 11. Demo Screenshots

Add screenshots of the program running here before final submission.

Suggested screenshots:

1. Program menu showing algorithm selection.
2. Heat Diffusion running with 2 processes.
3. Heat Diffusion running with 4 processes.
4. Prefix Sum running with 2 processes.
5. Prefix Sum running with 4 processes.
6. Timing output showing `Max compute time` and `Max total time`.

## 12. Requirement Checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| Two algorithms from different categories | Completed | Heat Diffusion and Prefix Sum |
| Runtime algorithm selection | Completed | Menu in `main.cpp` |
| Real inter-process communication | Completed | `MPI_Send`, `MPI_Recv`, `MPI_Isend`, `MPI_Irecv`, collectives |
| Big data input | Completed | `grid_input.bin`, `1000 x 1000` grid |
| Blocking communication | Completed | Manual distribution with `MPI_Send` / `MPI_Recv` |
| Non-blocking communication | Completed | Heat Diffusion ghost-row exchange |
| Additional communication pattern | Completed | Neighbor exchange and collective communication |
| Deadlock scenario explained | Completed | Heat Diffusion blocking-send scenario |
| Corrected deadlock solution | Completed | Non-blocking send/receive with `MPI_Waitall` |
| Uneven data sizes | Completed | Remainder-based distribution |
| Any number of processes N >= 2 | Supported | Distribution supports different process counts |
| Process organization | Completed | `MPI_Comm_split` and algorithm communicator |
| Performance observations | Completed | 2-process and 4-process observations |

## 13. Conclusion

This project implements a parallel MPI-based processing system with two different algorithm categories: Heat Diffusion for grid/spatial processing and Prefix Sum for data/computation processing. The system demonstrates blocking communication, non-blocking communication, neighbor exchange, and collective communication.

The performance results show that adding more processes does not always reduce runtime. For small workloads or workloads with heavy communication and output overhead, additional processes can make the program slightly slower. This is an important result in parallel computing because scalability depends on both computation size and communication cost.
