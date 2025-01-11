#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // The number of elements is equal to the total number of processes
    N = size;

    int numbers[N];       // Array to store the input numbers
    int received_value;   // Value received by each process
    int result;           // Computed result (square or cube)

    if (rank == 0) {
        // Root process reads N elements from the user
        printf("Enter %d numbers: ", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &numbers[i]);
        }

        // Send each value to the corresponding slave process
        for (int i = 1; i < N; i++) {
            MPI_Ssend(&numbers[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            printf("Root (rank 0) sent number %d to rank %d\n", numbers[i], i);
        }
    } else {
        // Slave processes receive the value
        MPI_Recv(&received_value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received value: %d\n", rank, received_value);

        // Compute square or cube based on rank (even or odd)
        if (rank % 2 == 0) {
            // Even-ranked process computes square
            result = received_value * received_value;
            printf("Process %d computed square: %d\n", rank, result);
        } else {
            // Odd-ranked process computes cube
            result = received_value * received_value * received_value;
            printf("Process %d computed cube: %d\n", rank, result);
        }

    }


    MPI_Finalize();
    return 0;
}
