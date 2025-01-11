#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads an integer
        printf("Enter an integer: ");
        scanf("%d", &value);

        // Send the value to process 1
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Root (rank 0) sent value %d to process 1\n", value);
    } else if (rank == size - 1) {
        // Last process receives the value, increments it and sends it back to root
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += 1;
        printf("Process %d received value %d, incremented to %d, sending back to root\n", rank, value - 1, value);
        MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        // Intermediate process receives value, increments it and sends to the next process
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += 1;
        printf("Process %d received value %d, incremented to %d, sending to process %d\n", rank, value - 1, value, rank + 1);
        MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Root process receives the final value
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Root (rank 0) received final value %d from process %d\n", value, size - 1);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
